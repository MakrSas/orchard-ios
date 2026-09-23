//! Product-path execution of blit fill/copy commands against guest backings.
//!
//! Supported now:
//! - `fillBuffer` (0x132) on buffers
//! - `copyFromBuffer:toBuffer:` (0x12d) on buffers
//! - Rectangular buffer↔texture / texture↔texture copies on linear normal-texture
//! - Same rectangular copies with **mapper-ref-texture IOSurface** texture endpoints
//!   (level 0, slice 0, depth 1) via mapping page tables; multi-plane (biplanar)
//!   sample windows from cached `sIOSurfaceDeviceDescriptor` selected by texture
//!   geometry (width/height/bpe), not a wire plane index
//! - **Texture-view texture views** as copy endpoints: unswizzled views over normal-texture
//!   or mapper-ref-texture bases; multi-level / array / non-2D Metal types when geometry matches
//!   (mapper-ref-texture bases remain single-level / single-slice — see below)
//! - **`MTLBlitOption`**: None; DepthFromDepthStencil / StencilFromDepthStencil;
//!   combined DS plane packing on linear GVA; unknown bits / RowLinearPVRTC fail
//! - **`0x13e` whole-surface** texture→texture: for each level in
//!   `[sourceLevel, sourceLevel+levelCount)`:
//!   - **depth-1 (array/2D):** full `width×height` across `sliceCount` consecutive
//!     slices
//!   - **depth>1 (3D volume):** Metal requires `sliceCount==1` and slices 0;
//!     copies full `width×height×depth` of that mip (depth planes via
//!     `bytes_per_image`); linear normal-texture only
//!   - zero `sliceCount`/`levelCount` are Metal no-ops
//! - **Fences** `0x13c` update / `0x13d` wait: blit-fence domain generation via
//!   [`crate::runtime::plan::event_sync`]; waits that are not yet satisfied are
//!   soft-pending (do not block drain), matching the unified-memory in-order path
//!
//! Not executed (fail visibly / soft miss):
//! - swizzled texture-views (contract: blit rejects remapped swizzle materialization)
//! - multisample view types
//! - RowLinearPVRTC / unknown option bits
//! - overlapping same-buffer B2B windows
//! - mapper-ref-texture multi-mip / non-zero level or slice — **not a missing feature**: Metal
//!   forbids mipmapped IOSurface textures (`newTextureWithDescriptor:iosurface:`
//!   rejects `mipmapLevelCount > 1`). Product path fail-closes; do not invent a
//!   pyramid layout in the mapping.
//! - 3D whole-surface with `sliceCount!=1`, non-zero slices, or mapper-ref-texture endpoint

// The backend the process executes on, reached only through the trait: this
// module names no rail.
use crate::backend::Backend as _;
use crate::model::DeviceState;
use crate::protocol::pixel_format::BlitAspect;
use crate::protocol::pixel_format::{self, MTL_FORMAT_BGRA8_UNORM};
use crate::runtime::decode::resource::{
    decode_buffer_descriptor, decode_iosurface_texture_descriptor, decode_texture_descriptor,
    decode_texture_view_descriptor, texture_view_type_is_3d, texture_view_type_supported,
    texture_view_type_uses_slices, Descriptor as ResourceDescriptor, OBJECT_TYPE_BUFFER,
    OBJECT_TYPE_MAPPER_REF_TEXTURE, OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS,
    OBJECT_TYPE_TEXTURE_VIEW, TEXTURE_VIEW_MTL_TYPE_2D,
};
use crate::runtime::draw::{self, host_alloc_len};
use crate::runtime::fence_exec::{self, FenceStatus};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapper::RectStride;
use crate::runtime::mapping_write;
use crate::runtime::objects;
use crate::runtime::plan::event_sync::{Domain as FenceDomain, FenceAction};
use reims_vgpu_protocol::blit::BlitKind as WireBlitKind;
use reims_vgpu_protocol::decode::blit::{
    BlitRecord, BufferToBuffer, BufferToTexture, FillBuffer, FillPattern, GenerateMipmaps,
    Origin as Point, TextureRegion, TextureSlices, TextureToBuffer,
};

/// Chunk size for fill/copy host staging (bounded guest IO).
const CHUNK: usize = 64 * 1024;

/// The coarse class a blit refusal collapses into.
///
/// Six classes over **177 distinct checks across 182 sites** — this rail is the
/// crate's largest refusal surface, and a 177-arm enum is not a thing anyone
/// writes. The class is what a caller branches on; which of the 177 fired is
/// what a reader needs, and it travels in [`BlitStatus::Failed`] beside the
/// class rather than beside the value in an ambient channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlitFailure {
    /// Missing object, wrong kind, or unreadable descriptor.
    MissingResource,
    /// Opcode / options / view / slice / 3D / mapper-ref-texture not on this path.
    Unsupported,
    /// Offset/length/extent outside allocation or level bounds.
    Bounds,
    /// Guest GVA read/write failed.
    GuestIo,
    /// Pathological size or host staging cap.
    Capacity,
    /// Same buffer, overlapping source/destination windows.
    Overlap,
}

/// Outcome of a product-path blit fill/copy/fence attempt.
///
/// # Why a refusal carries its reason instead of leaving it somewhere
///
/// The reason used to travel in a `BLIT_FAIL_REASON` thread-local, set by [`br`]
/// at the failing site, cleared at every command entry, and read by
/// [`Refusal::refusal`](crate::observe::Refusal::refusal) on the assumption that
/// the dispatch site runs on the same thread and reads it before anything else
/// writes it. Three assumptions, none of them checkable, and the failure mode of
/// each is a line naming the wrong check — which is worse than no line, because
/// it is believed.
///
/// It also could not say anything at all about a site that forgot to call `br`.
/// Those returned a coarse status with the channel empty and rendered as
/// `blit_unattributed`. Naming the gap was better than hiding it, but the gap
/// was still there and nothing stopped a new one appearing.
///
/// Both go away by putting the reason in the value: a refusal is
/// `Failed { failure, reason }` and there is no way to spell one without a
/// reason. `blit_unattributed` has no site left to come from, the channel has
/// no thread to be read on the wrong one of, and the clear at command entry has
/// nothing to clear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlitStatus {
    Ok,
    /// Zero-size fill or zero-extent rectangular copy (Metal no-op → soft ok).
    ZeroExtent,
    /// Fence wait not yet satisfied (soft; does not block drain).
    FencePending,
    /// The blit was refused: which class, and which of the checks in that class.
    Failed {
        failure: BlitFailure,
        /// The registered slug of the check that fired — greppable, and specific
        /// to the check rather than to the class. A slug may appear at more
        /// than one site when one check is written in more than one arm; it
        /// never covers two different checks.
        reason: &'static str,
    },
}

impl BlitStatus {
    /// The class this refusal collapsed into, or `None` if it is not a refusal.
    pub fn failure(self) -> Option<BlitFailure> {
        match self {
            Self::Failed { failure, .. } => Some(failure),
            _ => None,
        }
    }

    /// The slug of the check that fired, or `None` if this is not a refusal.
    pub fn reason(self) -> Option<&'static str> {
        match self {
            Self::Failed { reason, .. } => Some(reason),
            _ => None,
        }
    }
}

/// Compare a status against the class it collapsed into, ignoring which of that
/// class's checks named it.
///
/// For callers and tests that branch on the class — the class is what decides
/// whether the guest is told the resource was missing or the extent was out of
/// bounds, and pinning every such site to one of 177 slugs would make the slug
/// vocabulary unrefinable without editing them. A test that is *about* a
/// specific check asserts on [`BlitStatus::reason`] instead.
impl PartialEq<BlitFailure> for BlitStatus {
    fn eq(&self, other: &BlitFailure) -> bool {
        self.failure() == Some(*other)
    }
}

impl crate::observe::Refusal for BlitStatus {
    /// The reason comes out of the value, which is the only place it can be
    /// read from without an assumption about which thread is asking.
    ///
    /// `Ok`, `ZeroExtent` and `FencePending` are control flow — the first two are
    /// the dispatch site's success arm, the third is a soft wait the guest
    /// re-polls — so they name nothing, which is the same condition the two
    /// dispatch sites logged under before the reason moved into the value.
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok | Self::ZeroExtent | Self::FencePending => None,
            Self::Failed { reason, .. } => Some(reason),
        }
    }
}

/// Build a refusal: the class it collapses into, and the check that fired.
///
/// Short because it is at 182 sites, and the only constructor because a refusal
/// with no reason is what this rail spent a release unable to explain.
#[inline]
fn br(failure: BlitFailure, reason: &'static str) -> BlitStatus {
    BlitStatus::Failed { failure, reason }
}

/// Dedup set for the `tex_wrong_type` enrichment line, keyed by
/// `(task_id, texture_ref, object_type)`. A blit that binds a non-texture ref
/// fails once per draw (observed ~67/six-app-launch), so the enrichment must
/// dedup — but the bare `reason=tex_wrong_type` dispatch slug hides *what* the
/// object actually is (buffer bound as texture = a decode/tracking bug vs. a
/// legit guest race), which is the load-bearing field for diagnosis.
static TEX_WRONG_TYPE_SEEN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<(u32, u32, u8)>>,
> = std::sync::OnceLock::new();

/// Emit ONE always-on `blit tex_wrong_type` line per distinct
/// `(task, ref, object_type)` naming the actual object type a blit tried to use
/// as a texture. Deduped so a per-draw repeat cannot flood. Returns whether it
/// emitted (tests use it). Diagnostic only.
fn note_tex_wrong_type(
    task_id: u32,
    texture_ref: u32,
    object_type: u8,
    level: u16,
    slice: u16,
) -> bool {
    let set = TEX_WRONG_TYPE_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    if let Ok(mut g) = set.lock() {
        if !g.insert((task_id, texture_ref, object_type)) {
            return false;
        }
    }
    crate::observe::fail(format!(
        "blit tex_wrong_type task={task_id} ref={texture_ref} object_type={object_type} level={level} slice={slice}"
    ));
    true
}

#[cfg(test)]
fn reset_tex_wrong_type_dedup_for_test() {
    if let Some(set) = TEX_WRONG_TYPE_SEEN.get() {
        if let Ok(mut g) = set.lock() {
            g.clear();
        }
    }
}

/// Dedup set for the `t5_view_decode` diagnostic, keyed by surface id.
static T5_DECODE_FAIL_SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<u32>>> =
    std::sync::OnceLock::new();

/// One always-on diagnostic per surface id when a ref-texture RefTexture's view
/// record fails to decode: dumps `desc_len` + head hex so the exact blit-path
/// ref-texture layout can be read offline (the decoder wants tag 0x42 at +0x14, 2D
/// nonzero geom, depth==1). Deduped so a per-draw repeat cannot flood.
fn note_t5_decode_fail(sid: u32, bytes: &[u8]) {
    let set = T5_DECODE_FAIL_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    if let Ok(mut g) = set.lock() {
        if !g.insert(sid) {
            return;
        }
    }
    let n = bytes.len().min(40);
    let mut hex = String::with_capacity(n * 2);
    for b in &bytes[..n] {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    crate::observe::fail(format!(
        "blit t5_view_decode sid={sid} desc_len={} head_hex={hex}",
        bytes.len()
    ));
}

/// `(task, side, format)` — what `repack_storage_assumed` reports once per.
type RepackAssumedKey = (u32, &'static str, u16);

/// Dedup set for `blit repack_storage_assumed`.
static REPACK_STORAGE_ASSUMED_SEEN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<RepackAssumedKey>>,
> = std::sync::OnceLock::new();

/// Name an aspect repack whose storage stride could not be derived.
///
/// `bytes_per_pixel` refused this texture's format — in practice a zero, which
/// the texture-to-texture format check deliberately admits — so the repack runs
/// with the copied aspect's own width instead. That is right whenever the two
/// widths agree and reads the wrong bytes when they do not, and nothing
/// downstream can tell which happened. This is a **healthy zero**: a firing says
/// a real guest workload reached a combined depth/stencil repack against a
/// texture whose format this device never learned, which is the evidence that
/// would justify deriving the storage width from the backing instead of
/// assuming it. Deduped per task, side and format so a per-blit repeat cannot
/// flood.
fn note_repack_storage_assumed(
    task_id: u32,
    side: &'static str,
    format: u16,
    assumed_bpp: u32,
) -> bool {
    let set = REPACK_STORAGE_ASSUMED_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    if let Ok(mut g) = set.lock() {
        if !g.insert((task_id, side, format)) {
            return false;
        }
    }
    crate::observe::fail(format!(
        "blit repack_storage_assumed task={task_id} side={side} format={format} \
         assumed_bpp={assumed_bpp} (storage width underivable; aspect width used)"
    ));
    true
}

/// Dedup set for the `copy_region_*_io` enrichment, keyed by
/// `(task, gva_page, is_write)`.
static COPY_REGION_IO_SEEN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<(u32, u64, bool)>>,
> = std::sync::OnceLock::new();

/// Emit ONE always-on `blit copy_region_io` line per distinct
/// `(task, failing-gva-page, is_write)` naming the exact guest address a
/// rectangular copy row could not read/write. A guest that tears down the
/// destination surface mid-copy (teardown race) shows a plausible-but-unmapped
/// gva; a decode/geometry bug shows a wild gva. Deduped per page so a strided
/// multi-row failure cannot flood. Diagnostic only.
fn note_copy_region_io(
    task_id: u32,
    is_write: bool,
    gva: u64,
    row: u64,
    image: u64,
    row_bytes: u64,
    page_shift: u32,
) -> bool {
    let set = COPY_REGION_IO_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    let key = (task_id, gva >> page_shift, is_write);
    if let Ok(mut g) = set.lock() {
        if !g.insert(key) {
            return false;
        }
    }
    let dir = if is_write { "write" } else { "read" };
    crate::observe::fail(format!(
        "blit copy_region_io dir={dir} task={task_id} gva={gva:#x} row={row} image={image} row_bytes={row_bytes}"
    ));
    true
}

struct LinearBuffer {
    gva: u64,
    size: u64,
}

pub(crate) struct LinearTextureLevel {
    /// Allocation base GVA (`handle << page_shift` for the device).
    base_gva: u64,
    alloc_size: u64,
    level_offset: u64,
    row_stride: u64,
    /// Byte stride between array slices / cube faces at this level.
    /// 0 means single-slice (no slice offset applied).
    slice_stride: u64,
    /// Absolute array slice / cube face selected for this resolve.
    slice_index: u32,
    width: u32,
    height: u32,
    depth: u32,
    bpp: u32,
    /// The storage grid one `bpp` unit covers.
    ///
    /// 1x1 for every uncompressed format, so `bpp` and this agree for all of
    /// them and nothing downstream changes. 4x4 for the BC families, where `bpp`
    /// is bytes per **block** — which is why the one rail that admits a
    /// compressed texture, `exec_copy_texture_to_texture`, converts its
    /// coordinates into block space before using any of the per-unit helpers
    /// below. A compressed copy is an uncompressed copy of the block image, and
    /// converting once at the top is what lets that be true rather than
    /// threading a grid through every helper.
    block: reims_vgpu_protocol::extent::BlockGeometry,
    pixel_format: u16,
}

/// Mapper-ref-texture IOSurface texture (single level, 2D).
///
/// Metal rejects mipmapped IOSurface textures (`mipmapLevelCount > 1` fails
/// descriptor validation on `newTextureWithDescriptor:iosurface:plane:`). The
/// product path therefore never materializes non-zero mip levels or invents a
/// multi-mip packing inside the mapping — non-zero `level`/`slice` fails closed.
///
/// Multi-plane (biplanar 420): sample window comes from the cached guest device
/// descriptor via geometry match (texture width/height/bpe); `surface_offset` is
/// the plane base in the shared mapping.
pub(crate) struct MapperRefTexture {
    mapping_id: u32,
    width: u32,
    height: u32,
    /// Byte offset of this texture/plane in the mapping allocation.
    surface_offset: u64,
    /// IOSurface-aligned surface row stride (bytes).
    ///
    /// `u32` to match both ends it sits between: `mapper_ref_texture_sample_window` and
    /// `ref_texture_sample_window` each return it as one, and its only readers hand
    /// it to [`mapping_write::SurfaceWindow::bpr`], which is one. It was `u64`,
    /// so both construction sites widened and both readers narrowed straight
    /// back — a round trip that reads exactly like an unchecked truncation of a
    /// 64-bit guest field and has been mistaken for one.
    row_stride: u32,
    /// Exclusive end of the sample window (for page-span planning).
    span_end: u64,
    bpp: u32,
    pixel_format: u16,
}

enum TextureBacking {
    Linear(LinearTextureLevel),
    MapperRefTexture(MapperRefTexture),
}

impl TextureBacking {
    fn width(&self) -> u32 {
        match self {
            TextureBacking::Linear(t) => t.width,
            TextureBacking::MapperRefTexture(t) => t.width,
        }
    }
    fn height(&self) -> u32 {
        match self {
            TextureBacking::Linear(t) => t.height,
            TextureBacking::MapperRefTexture(t) => t.height,
        }
    }
    fn depth(&self) -> u32 {
        match self {
            TextureBacking::Linear(t) => t.depth,
            TextureBacking::MapperRefTexture(_) => 1,
        }
    }
    fn bpp(&self) -> u32 {
        match self {
            TextureBacking::Linear(t) => t.bpp,
            TextureBacking::MapperRefTexture(t) => t.bpp,
        }
    }
    /// The storage grid one [`Self::bpp`] unit covers.
    fn block(&self) -> reims_vgpu_protocol::extent::BlockGeometry {
        match self {
            TextureBacking::Linear(t) => t.block,
            // A mapper-ref-texture IOSurface is never block-compressed: its resolve takes
            // `bytes_per_pixel`, which has no answer for a compressed format, so
            // such a surface is refused as `t11_fmt_bpp` long before here.
            TextureBacking::MapperRefTexture(t) => reims_vgpu_protocol::extent::BlockGeometry {
                width: 1,
                height: 1,
                bytes: t.bpp,
            },
        }
    }
    fn pixel_format(&self) -> u16 {
        match self {
            TextureBacking::Linear(t) => t.pixel_format,
            TextureBacking::MapperRefTexture(t) => t.pixel_format,
        }
    }
    fn is_mapper_ref_texture(&self) -> bool {
        matches!(self, TextureBacking::MapperRefTexture(_))
    }
}

impl LinearTextureLevel {
    /// Bytes one depth plane / array slice of this level occupies.
    ///
    /// Counted in rows of **storage**: a block-compressed level is a quarter as
    /// tall in rows as it is in texels, so the texel form overstated one by four
    /// and would have strided a `z` plane or an array slice past its own image.
    /// `block_rows` answers `height` for every uncompressed format, so this is
    /// the same product it always was for them.
    fn bytes_per_image(&self) -> Option<u64> {
        self.row_stride
            .checked_mul(u64::from(self.block.block_rows(self.height)))
    }

    /// Byte offset of texel origin (x,y,z) within the allocation (includes slice).
    fn texel_offset(&self, x: u64, y: u64, z: u64) -> Option<u64> {
        let bpi = self.bytes_per_image()?;
        let row = y.checked_mul(self.row_stride)?;
        let col = x.checked_mul(self.bpp as u64)?;
        let plane = z.checked_mul(bpi)?;
        let slice = if self.slice_index == 0 || self.slice_stride == 0 {
            0u64
        } else {
            (self.slice_index as u64).checked_mul(self.slice_stride)?
        };
        self.level_offset
            .checked_add(slice)?
            .checked_add(plane)?
            .checked_add(row)?
            .checked_add(col)
    }
}

fn resolve_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
) -> Result<LinearBuffer, BlitStatus> {
    if buffer_ref == 0 {
        return Err(br(BlitFailure::MissingResource, "buf_ref_zero"));
    }
    let (_entry, bytes) =
        objects::resolve_descriptor(state, host, task_id, buffer_ref, &[OBJECT_TYPE_BUFFER])
            .map_err(|rung| {
                br(
                    BlitFailure::MissingResource,
                    crate::observe::ladder_slugs!("buf")(rung),
                )
            })?;
    let Ok(buf) = decode_buffer_descriptor(&bytes) else {
        return Err(br(
            BlitFailure::MissingResource,
            crate::observe::ladder_slug!("buf", desc_decode),
        ));
    };
    let Some((gva, size)) = buf.backing_gva_size(state.page_shift) else {
        return Err(br(BlitFailure::MissingResource, "buf_no_backing"));
    };
    Ok(LinearBuffer { gva, size })
}

fn resolve_texture_backing<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u16,
    slice: u16,
) -> Result<TextureBacking, BlitStatus> {
    resolve_texture_backing_depth(state, host, task_id, texture_ref, level, slice, 0)
}

fn resolve_texture_backing_depth<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u16,
    slice: u16,
    // How many texture-view hops deep this recursion is — **not** a texture's
    // depth. It was spelled `depth`, and a local named `depth` further down held
    // the level's plane count and shadowed it, so one word meant two unrelated
    // things in one body and `LinearTextureLevel { depth }` took whichever was
    // in scope at that line. Removing the local silently rebound that field to
    // the view hop count, and only
    // `whole_surface_0x13e_volume_rejects_multi_slice` noticed — by the refusal
    // order changing, not by the field.
    view_depth: u32,
) -> Result<TextureBacking, BlitStatus> {
    if texture_ref == 0 {
        return Err(br(BlitFailure::MissingResource, "tex_ref_zero"));
    }
    // Shared with the draw/sample walk on purpose: the two arms read one wire
    // form, and a chain that resolver would follow must not be a copy this one
    // drops. See `MAX_TEXTURE_VIEW_CHAIN` for why the contract's own depth is
    // the right number for both.
    //
    // `>` and not `>=` because this recursion counts *calls* and the other walk
    // counts *hops*: the last call here is the non-view base, one level below
    // the deepest view. A chain of MAX views arrives at its base at `depth ==
    // MAX`, so admitting that depth is what makes the two arms accept and
    // refuse the same chains.
    if view_depth as usize > crate::runtime::draw::MAX_TEXTURE_VIEW_CHAIN {
        return Err(br(BlitFailure::Unsupported, "tex_view_depth_cap"));
    }
    // **This function is the whole blit rail's definition of "the guest bytes of
    // a texture", and guest bytes are only a resource's content once everything
    // this device rendered into it has landed.** Every endpoint of every blit —
    // source and destination, texture-to-texture, texture-to-buffer and back —
    // arrives here, so this is the one place the rail has to state that.
    //
    // A render pass whose colour attachment is an ordinary private `MTLTexture`
    // resolves on `render_target`'s fourth rung to a linear guest VA, and
    // `writeback_debt::arm_gva` then keeps the result in the engine's resident
    // and arms a debt against `(task_id, texture_ref)` instead of copying it
    // out. Nothing lands until a **guest-byte reader** names the resource. The
    // two readers that did name it are `draw::texture_view` and `compute_exec`;
    // a blit is exactly as much a guest-byte reader as either, and it was
    // silent. Its copy read whatever the pages held before the pass — which for
    // a freshly allocated private target is zeros, at scale, with no error, the
    // "Never Fail Silently" class in its worst form.
    //
    // `pay_for_texture` is the call that covers both spellings a debt can have
    // (the task-local GVA resource and the surface mapping), and it early-returns
    // on an empty ledger, so the cost on a rail with nothing owed is one
    // `is_empty`. Paying here rather than at the copy also covers the view chain:
    // the recursion below re-enters with the base ref, and the debt may be armed
    // against either spelling of the pair.
    //
    // Paying a *destination*'s debt before overwriting it is deliberate, not
    // waste. The blit writes a rect, not the plane, so the pixels outside that
    // rect are the resource's content and must be real; and leaving the debt
    // armed would let it land the pre-blit resident over the blit's own bytes
    // later.
    note_blit_endpoint_debt(state, task_id, texture_ref);
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, texture_ref) else {
        return Err(br(
            BlitFailure::MissingResource,
            crate::observe::ladder_slug!("tex", no_list_entry),
        ));
    };

    // Texture-view view → base texture (unswizzled; multi-level / array / non-2D allowed).
    if entry.object_type == OBJECT_TYPE_TEXTURE_VIEW {
        let Some(bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
            return Err(br(
                BlitFailure::MissingResource,
                crate::observe::ladder_slug!("view", desc_read),
            ));
        };
        let Ok(view) = decode_texture_view_descriptor(&bytes) else {
            return Err(br(
                BlitFailure::MissingResource,
                crate::observe::ladder_slug!("view", desc_decode),
            ));
        };
        if view.base_texture_ref == 0 {
            return Err(br(BlitFailure::MissingResource, "view_base_ref_zero"));
        }
        // Blit rejects swizzled materialization (contract).
        if view.carries_swizzle() {
            let plan = pixel_format::swizzle_plan(&view.swizzle)
                .ok_or_else(|| br(BlitFailure::Unsupported, "view_swizzle_plan"))?;
            if !pixel_format::swizzle_is_identity(&plan) {
                return Err(br(BlitFailure::Unsupported, "view_swizzle_nonident"));
            }
        }
        let view_type = if view.carries_range() {
            if !texture_view_type_supported(view.texture_type) {
                return Err(br(BlitFailure::Unsupported, "view_type_unsupported"));
            }
            view.texture_type
        } else {
            TEXTURE_VIEW_MTL_TYPE_2D
        };
        // Relative command level → absolute on the base (multi-level ranges ok).
        let rel_level = level as u64;
        let level_count = if view.carries_range() {
            if view.level_count == 0 {
                1
            } else {
                view.level_count
            }
        } else {
            // Simple form: no level range; command level is absolute on base.
            u64::MAX
        };
        if view.carries_range() && rel_level >= level_count {
            return Err(br(BlitFailure::Bounds, "view_level_oob"));
        }
        let abs_level = if view.carries_range() {
            view.level_base
                .checked_add(rel_level)
                .ok_or_else(|| br(BlitFailure::Bounds, "view_level_overflow"))?
        } else {
            rel_level
        };
        if abs_level > u16::MAX as u64 {
            return Err(br(BlitFailure::Bounds, "view_level_u16"));
        }
        // Relative command slice → absolute (array / cube faces).
        let rel_slice = slice as u64;
        let slice_count = if view.carries_range() {
            if view.slice_count == 0 {
                1
            } else {
                view.slice_count
            }
        } else {
            u64::MAX
        };
        if view.carries_range() && rel_slice >= slice_count {
            return Err(br(BlitFailure::Bounds, "view_slice_oob"));
        }
        let abs_slice = if view.carries_range() {
            view.slice_base
                .checked_add(rel_slice)
                .ok_or_else(|| br(BlitFailure::Bounds, "view_slice_overflow"))?
        } else {
            rel_slice
        };
        if abs_slice > u16::MAX as u64 {
            return Err(br(BlitFailure::Bounds, "view_slice_u16"));
        }
        // 3D views use depth planes, not array slices.
        if texture_view_type_is_3d(view_type) && abs_slice != 0 {
            return Err(br(BlitFailure::Unsupported, "view_3d_slice"));
        }
        // Non-array 2D/1D: only slice 0.
        if !texture_view_type_uses_slices(view_type)
            && !texture_view_type_is_3d(view_type)
            && abs_slice != 0
        {
            return Err(br(BlitFailure::Unsupported, "view_2d_slice"));
        }
        let mut backing = resolve_texture_backing_depth(
            state,
            host,
            task_id,
            view.base_texture_ref,
            abs_level as u16,
            abs_slice as u16,
            view_depth + 1,
        )?;
        // Geometry constraints for non-2D types.
        match &backing {
            TextureBacking::Linear(t) => {
                if matches!(
                    view_type,
                    crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D
                        | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
                ) && t.height != 1
                {
                    return Err(br(BlitFailure::Unsupported, "view_1d_height"));
                }
            }
            TextureBacking::MapperRefTexture(_) => {
                // Metal forbids mipmapped / multi-slice IOSurface textures; see
                // MapperRefTexture. Fail closed rather than inventing layout.
                if abs_level != 0 || abs_slice != 0 {
                    return Err(br(BlitFailure::Unsupported, "view_t11_level_slice"));
                }
                if texture_view_type_uses_slices(view_type) || texture_view_type_is_3d(view_type) {
                    return Err(br(BlitFailure::Unsupported, "view_t11_type"));
                }
            }
        }
        // View pixel_format overrides base when bpp-compatible.
        if let Some(declared) = view.declared_pixel_format() {
            let base_fmt = backing.pixel_format();
            let eff = draw::effective_view_sample_format(base_fmt, Some(declared))
                .ok_or_else(|| br(BlitFailure::Unsupported, "view_fmt_incompat"))?;
            match &mut backing {
                TextureBacking::Linear(t) => {
                    t.pixel_format = eff;
                    t.bpp = pixel_format::bytes_per_pixel(eff)
                        .ok_or_else(|| br(BlitFailure::Unsupported, "view_fmt_bpp"))?;
                }
                TextureBacking::MapperRefTexture(t) => {
                    t.pixel_format = eff;
                    t.bpp = pixel_format::bytes_per_pixel(eff)
                        .ok_or_else(|| br(BlitFailure::Unsupported, "view_fmt_bpp"))?;
                }
            }
        }
        return Ok(backing);
    }

    // Mapper-ref-texture IOSurface: single level, 2D, mapping page table.
    // Non-zero level/slice is fail-closed (Metal disallows mipmapped IOSurfaces).
    // Texture object dims/format select the plane when the mapping is multi-plane.
    if entry.object_type == OBJECT_TYPE_MAPPER_REF_TEXTURE {
        if level != 0 || slice != 0 {
            return Err(br(BlitFailure::Unsupported, "t11_level_slice"));
        }
        let Some(bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
            return Err(br(
                BlitFailure::MissingResource,
                crate::observe::ladder_slug!("t11", desc_read),
            ));
        };
        let Ok(ResourceDescriptor::IOSurfaceTexture {
            mapping_id,
            pixel_format: tex_fmt,
            width: tex_w,
            height: tex_h,
            ..
        }) = decode_iosurface_texture_descriptor(&bytes)
        else {
            return Err(br(
                BlitFailure::MissingResource,
                crate::observe::ladder_slug!("t11", desc_decode),
            ));
        };
        if mapping_id == 0 || tex_w == 0 || tex_h == 0 {
            return Err(br(BlitFailure::MissingResource, "t11_zero_geom"));
        }
        // Latch texture→mapping and refresh pages / device desc.
        let _ = objects::resolve_mapper_ref_texture(state, host, task_id, texture_ref);
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        let Some(m) = state.mappings.get(&mapping_id) else {
            return Err(br(BlitFailure::MissingResource, "t11_no_mapping"));
        };
        if !m.mapped || m.page_entries.is_empty() {
            return Err(br(BlitFailure::MissingResource, "t11_unmapped"));
        }
        let format = if tex_fmt != 0 {
            tex_fmt
        } else if m.format != 0 {
            m.format
        } else {
            MTL_FORMAT_BGRA8_UNORM
        };
        let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
            return Err(br(BlitFailure::Unsupported, "t11_fmt_bpp"));
        };
        let Some((surface_offset, surface_bpr, span_end)) =
            mapping_write::mapper_ref_texture_sample_window(m, tex_w, tex_h, format)
        else {
            return Err(br(BlitFailure::Bounds, "t11_sample_window"));
        };
        crate::backend::selected().note_blit_t11_resident(state, mapping_id);
        return Ok(TextureBacking::MapperRefTexture(MapperRefTexture {
            mapping_id,
            width: tex_w,
            height: tex_h,
            surface_offset,
            row_stride: surface_bpr,
            span_end,
            bpp,
            pixel_format: format,
        }));
    }

    // Ref-texture RefTexture: a serialized Metal texture VIEW over an IOSurface
    // (surfaceID at +0). The compute stage path already resolves these; the
    // blit path previously dropped every one as `tex_wrong_type` (~99/six-app
    // launch, all object_type=5), so a blit COPY from a video/biplanar plane
    // or a row-byte-equivalent reinterpretation view (e.g. RGBA32Uint over
    // BGRA8) never landed.
    //
    // Resolve it with the view's own geometry, format **and plane index**. The
    // plane is on the wire here (record `+0x20`) and must be used: it is the
    // whole difference between this and mapper-ref-texture, whose window resolves the plane
    // by matching geometry and bytes-per-element and so cannot tell two planes
    // that share both apart. A biplanar COPY names exactly such a pair, so this
    // is the path where dropping the index lands. `ref_texture_sample_window` states
    // the case; a plane it cannot resolve declines here rather than binding
    // whichever plane shares the geometry.
    if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
        if level != 0 || slice != 0 {
            return Err(br(BlitFailure::Unsupported, "t5_level_slice"));
        }
        let Some(bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
            return Err(br(
                BlitFailure::MissingResource,
                crate::observe::ladder_slug!("t5", desc_read),
            ));
        };
        let Ok(t5) = reims_vgpu_wire::device_desc::ref_texture_header(&bytes) else {
            return Err(br(BlitFailure::MissingResource, "t5_desc_short"));
        };
        let sid = t5.surface_id.get();
        if sid == 0 {
            return Err(br(BlitFailure::MissingResource, "t5_no_sid"));
        }
        let Some(view) = objects::decode_ref_texture_view(&bytes) else {
            // A short/zero-geom record fails closed — no fallback to base geom.
            // Capture why (len/tag/geom) deduped per sid so the exact blit-path
            // ref-texture layout can be decoded without flooding.
            note_t5_decode_fail(sid, &bytes);
            return Err(br(BlitFailure::Unsupported, "t5_view_decode"));
        };
        // Surface id IS the backing mapping mid (never the task object-list ref —
        // those id spaces collide). Resolve the backing, then the mapping.
        let _ = objects::ensure_surface_for_present(state, host, sid);
        let _ = mapper::ensure_resolved_for_scanout(state, host, sid);
        let Some(m) = state.mappings.get(&sid) else {
            return Err(br(BlitFailure::MissingResource, "t5_no_mapping"));
        };
        if !m.mapped || m.page_entries.is_empty() {
            return Err(br(BlitFailure::MissingResource, "t5_unmapped"));
        }
        let format = view.pixel_format;
        let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
            return Err(br(BlitFailure::Unsupported, "t5_fmt_bpp"));
        };
        let Some((surface_offset, surface_bpr, span_end)) =
            mapping_write::ref_texture_sample_window(
                m,
                view.plane_index,
                view.width,
                view.height,
                format,
            )
        else {
            return Err(br(BlitFailure::Bounds, "t5_sample_window"));
        };
        // Whether this arm runs at all. Without it a change to the window this
        // arm resolves cannot be attributed: an unchanged screen and an arm that
        // never executed look identical, and so do a repaired blit and a blit
        // that never happened.
        //
        // Read on a driven x86/Vulkan boot (Safari window drag + two
        // web-content-probe runs): **0** — this arm does not execute on that
        // workload at all, while `blit_dest_bound` reads 26, so the blit path
        // itself does run and it is the ref-texture source that is absent. The
        // plane-index resolution above is therefore contract fidelity, not a
        // repair of anything this workload does, and a screen that looks the
        // same after changing it says nothing either way.
        crate::runtime::drain::note_store_route("blit_t5_plane_device");
        crate::backend::selected().note_blit_t11_resident(state, sid);
        return Ok(TextureBacking::MapperRefTexture(MapperRefTexture {
            mapping_id: sid,
            width: view.width,
            height: view.height,
            surface_offset,
            row_stride: surface_bpr,
            span_end,
            bpp,
            pixel_format: format,
        }));
    }

    if entry.object_type != OBJECT_TYPE_TEXTURE
        && entry.object_type != OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS
    {
        let _ = note_tex_wrong_type(task_id, texture_ref, entry.object_type, level, slice);
        return Err(br(
            BlitFailure::MissingResource,
            crate::observe::ladder_slug!("tex", wrong_type),
        ));
    }
    let Some(bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
        return Err(br(
            BlitFailure::MissingResource,
            crate::observe::ladder_slug!("tex", desc_read),
        ));
    };
    let Ok(tex) = decode_texture_descriptor(&bytes) else {
        return Err(br(
            BlitFailure::MissingResource,
            crate::observe::ladder_slug!("tex", desc_decode),
        ));
    };
    if tex.declared_pixel_format().is_none() {
        crate::observe::fail(format!(
            "blit tex no_pixel_format ref={texture_ref} w={} h={} fmt={}",
            tex.width, tex.height, tex.pixel_format
        ));
        return Err(br(BlitFailure::Unsupported, "tex_no_pixel_format"));
    }
    // The storage grid rather than a bytes-per-texel, so a block-compressed
    // level resolves instead of being refused here. `tex_bad_bpp` fired 448
    // times on one driven Asphalt 8 leg, every one of them a `kind=Copy` between
    // two BC3 textures — a copy that moves whole blocks and converts nothing.
    //
    // Resolving is not admitting: only the texture-to-texture copy handles a
    // compressed grid, and every other rail that takes this backing refuses one
    // by name.
    let Some(block) = pixel_format::block_geometry(tex.pixel_format) else {
        crate::observe::fail(format!(
            "blit tex bad_bpp ref={texture_ref} fmt={}",
            tex.pixel_format
        ));
        return Err(br(BlitFailure::Unsupported, "tex_bad_bpp"));
    };
    let bpp = block.bytes;
    let Some((layout_gva, layout)) = tex.level_gva(level as u32, state.page_shift) else {
        crate::observe::fail(format!(
            "blit tex level_gva_shift fail ref={texture_ref} lvl={level} handle={} alloc={} mips={} page_shift={} w={} h={} fmt={:#x}",
            tex.handle,
            tex.allocation_size,
            tex.mipmap_level_count,
            state.page_shift,
            tex.width,
            tex.height,
            tex.pixel_format
        ));
        return Err(br(BlitFailure::Bounds, "tex_level_gva"));
    };
    let Some(base_gva) = tex.allocation_base_gva(state.page_shift) else {
        return Err(br(BlitFailure::MissingResource, "tex_no_base_gva"));
    };
    // level_gva already applied offset; keep offset relative to base for plane math.
    let level_offset = match layout_gva.checked_sub(base_gva) {
        Some(v) => v,
        None => {
            crate::observe::fail(format!(
                "blit tex level_offset underflow layout_gva={layout_gva:#x} base={base_gva:#x} page_shift={}",
                state.page_shift
            ));
            return Err(br(BlitFailure::Bounds, "tex_level_offset_underflow"));
        }
    };
    if layout.width == 0 || layout.height == 0 {
        crate::observe::fail(format!(
            "blit tex zero_geom ref={texture_ref} lvl={level} layout={}x{}x{}",
            layout.width, layout.height, layout.depth
        ));
        return Err(br(BlitFailure::Bounds, "tex_zero_geom"));
    }
    // Array-slice packing: contiguous images at this mip
    // (row_stride × height × planes). `TextureLevelLayout` owns both the packing
    // and the "depth 0 means one plane" encoding; this used to normalize the
    // depth here and hand it to a helper that normalized it again.
    // Prefer level.size when it is an exact multiple of one-slice bytes (multi-slice alloc).
    let one_slice = layout
        .slice_stride()
        .ok_or_else(|| br(BlitFailure::Capacity, "tex_slice_stride"))?;
    if slice != 0 {
        // Bounds: selected slice must fit in allocation when known.
        // Live x86 buffer→texture (opcode 0x12c) uses slice=1,2 with
        // size=16384x1x1 at off=64K/128K — array packing into one allocation
        // even when the L0 level record's `size` equals one_slice. Prefer
        // allocation_size over the level-size single-slice reject.
        //
        // The selected slice is charged the bytes it is *read* through, not a
        // whole `one_slice` stride: `texel_offset` walks rows and planes and
        // the trailing padding after the final row is never touched. Charging
        // it refuses allocations sized exactly for the array — see
        // `TextureLevelLayout::slice_read_span`.
        let tight_row = pixel_format::tight_row_bytes(layout.width, tex.pixel_format)
            .ok_or_else(|| br(BlitFailure::Unsupported, "tex_slice_tight_row"))?;
        let slice_read = layout
            .slice_read_span(tight_row)
            .ok_or_else(|| br(BlitFailure::Bounds, "tex_slice_read_span"))?;
        let slice_end = (slice as u64)
            .checked_mul(one_slice)
            .and_then(|o| o.checked_add(level_offset))
            .and_then(|o| o.checked_add(slice_read))
            .ok_or_else(|| br(BlitFailure::Bounds, "tex_slice_overflow"))?;
        if tex.allocation_size != 0 && slice_end > tex.allocation_size {
            crate::observe::fail(format!(
                "blit tex slice Bounds slice={slice} end={slice_end} alloc={} one_slice={one_slice} lvl_off={level_offset}",
                tex.allocation_size
            ));
            return Err(br(BlitFailure::Bounds, "tex_slice_bounds"));
        }
        if tex.allocation_size == 0 && layout.size != 0 && layout.size == one_slice && slice != 0 {
            // Unknown alloc and level size covers a single slice only.
            return Err(br(BlitFailure::Bounds, "tex_slice_single"));
        }
    }
    Ok(TextureBacking::Linear(LinearTextureLevel {
        base_gva,
        alloc_size: tex.allocation_size,
        level_offset,
        row_stride: layout.row_stride,
        block,
        slice_stride: one_slice,
        slice_index: slice as u32,
        width: layout.width,
        height: layout.height,
        depth: layout.planes(),
        bpp,
        pixel_format: tex.pixel_format,
    }))
}

/// Read one texture row (tight `row_bytes`) at texel (ox, oy+row_i) plane z into `buf`.
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper still names the plane geometry a row walk needs"
)]
fn read_texture_row<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_i: u64,
    row_bytes: u64,
    buf: &mut [u8],
) -> Result<(), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    if row_bytes as usize > buf.len() {
        return Err(br(BlitFailure::Capacity, "rd_row_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let off = t
                .texel_offset(
                    ox,
                    oy.checked_add(row_i)
                        .ok_or_else(|| br(BlitFailure::Bounds, "rd_row_y_overflow"))?,
                    oz,
                )
                .ok_or_else(|| br(BlitFailure::Bounds, "rd_row_texel_oob"))?;
            let gva = t
                .base_gva
                .checked_add(off)
                .ok_or_else(|| br(BlitFailure::Bounds, "rd_row_gva_overflow"))?;
            if gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                gva,
                &mut buf[..row_bytes as usize],
                state.page_shift,
            )
            .is_err()
            {
                return Err(br(BlitFailure::GuestIo, "rd_row_linear_io"));
            }
            Ok(())
        }
        TextureBacking::MapperRefTexture(t) => {
            if oz != 0 {
                return Err(br(BlitFailure::Unsupported, "rd_row_t11_z"));
            }
            let y = oy
                .checked_add(row_i)
                .ok_or_else(|| br(BlitFailure::Bounds, "rd_row_t11_y_overflow"))?;
            if y > u32::MAX as u64 || ox > u32::MAX as u64 {
                return Err(br(BlitFailure::Bounds, "rd_row_t11_coord_range"));
            }
            let pixels = (row_bytes / t.bpp as u64) as u32;
            if !mapping_write::read_rect_raw_at(
                state,
                host,
                t.mapping_id,
                mapping_write::SurfaceWindow {
                    base_off: t.surface_offset,
                    bpr: t.row_stride,
                    span_end: t.span_end,
                    bpp: t.bpp,
                },
                mapping_write::Rect {
                    origin_x: ox as u32,
                    origin_y: y as u32,
                    width: pixels,
                    height: 1,
                },
                &mut buf[..row_bytes as usize],
                row_bytes as u32,
            ) {
                return Err(br(BlitFailure::GuestIo, "rd_row_t11_io"));
            }
            Ok(())
        }
    }
}

/// Write one texture row from `buf`, bounded to the pages the copy's whole
/// destination region resolved to before its row loop started
/// ([`texture_region_window`]).
///
/// `allowed` is not consulted on the mapper-ref-texture arm: that write goes through the
/// mapping rail, whose authorisation is the page list the guest declared for
/// the mapping itself. That is a different and equally explicit model, not an
/// unbounded one.
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper still names the plane geometry a row walk needs"
)]
fn write_texture_row<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_i: u64,
    row_bytes: u64,
    buf: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    if row_bytes as usize > buf.len() {
        return Err(br(BlitFailure::Capacity, "wr_row_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let off = t
                .texel_offset(
                    ox,
                    oy.checked_add(row_i)
                        .ok_or_else(|| br(BlitFailure::Bounds, "wr_row_y_overflow"))?,
                    oz,
                )
                .ok_or_else(|| br(BlitFailure::Bounds, "wr_row_texel_oob"))?;
            let gva = t
                .base_gva
                .checked_add(off)
                .ok_or_else(|| br(BlitFailure::Bounds, "wr_row_gva_overflow"))?;
            if gva_mem::write_task_gva_product_within(
                state,
                host,
                task_id,
                gva,
                &buf[..row_bytes as usize],
                allowed,
            )
            .is_err()
            {
                return Err(br(BlitFailure::GuestIo, "wr_row_linear_io"));
            }
            Ok(())
        }
        TextureBacking::MapperRefTexture(t) => {
            if oz != 0 {
                return Err(br(BlitFailure::Unsupported, "wr_row_t11_z"));
            }
            let y = oy
                .checked_add(row_i)
                .ok_or_else(|| br(BlitFailure::Bounds, "wr_row_t11_y_overflow"))?;
            if y > u32::MAX as u64 || ox > u32::MAX as u64 {
                return Err(br(BlitFailure::Bounds, "wr_row_t11_coord_range"));
            }
            let pixels = (row_bytes / t.bpp as u64) as u32;
            if !mapping_write::write_rect_raw_at(
                state,
                host,
                t.mapping_id,
                mapping_write::SurfaceWindow {
                    base_off: t.surface_offset,
                    bpr: t.row_stride,
                    span_end: t.span_end,
                    bpp: t.bpp,
                },
                mapping_write::Rect {
                    origin_x: ox as u32,
                    origin_y: y as u32,
                    width: pixels,
                    height: 1,
                },
                &buf[..row_bytes as usize],
                row_bytes as u32,
            ) {
                return Err(br(BlitFailure::GuestIo, "wr_row_t11_io"));
            }
            Ok(())
        }
    }
}

/// Read a whole `row_bytes`-wide, `row_count`-tall rectangle at `origin` into
/// `buf`, rows packed `row_bytes` apart.
///
/// # A rect is the unit the mapping rail is built for, and a row is not
///
/// [`mapping_write::read_rect_raw_at`] takes a height because every per-call
/// cost it carries is per *rect*, not per row: a writeback settle, a mapping
/// lookup, a window revalidation, and — on a fragmented mapping, which is the
/// arm a driven x86 boot takes — a fresh QEMU memory-region import per guest
/// page run. Handing it `height: 1` in a loop pays all of that `row_count`
/// times to move one row of texels.
///
/// That is not a small constant. A driven macos-13 Maps leg measured the
/// slice/level copy's row loop at **30.15 s of a 30.28 s blit rail** while
/// moving 14.6 MB — 0.48 MB/s, against 0.22 s for every strided guest-RAM copy
/// in the device put together. The bytes were never the cost; the per-row
/// re-entry into the mapping rail was.
///
/// The linear arm keeps its row loop, because there the per-row call is a bare
/// guest-RAM read with none of that preamble, and rows of a strided level are
/// genuinely discontiguous.
#[allow(
    clippy::too_many_arguments,
    reason = "the rect helper names the same geometry its row counterpart does"
)]
fn read_texture_rect<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_bytes: u64,
    row_count: u64,
    buf: &mut [u8],
) -> Result<(), BlitStatus> {
    let need = row_bytes
        .checked_mul(row_count)
        .ok_or_else(|| br(BlitFailure::Capacity, "rd_rect_span_overflow"))?;
    if need as usize > buf.len() {
        return Err(br(BlitFailure::Capacity, "rd_rect_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let (gva, rect) = linear_rect(t, origin, row_bytes, row_count, "rd_rect_linear_shape")?;
            crate::runtime::gva_view::read_rect(state, host, task_id, gva, rect, buf)
                .map_err(|_| br(BlitFailure::GuestIo, "rd_rect_linear_io"))?;
            crate::runtime::drain::note_store_route("blit_rect_linear_read_walk");
            crate::runtime::drain::note_store_route_n(
                "blit_rect_linear_read_rows_hoisted",
                row_count.saturating_sub(1),
            );
            Ok(())
        }
        TextureBacking::MapperRefTexture(t) => {
            let (pixels, height, origin_x, origin_y) =
                t11_rect_extent(t, origin, row_bytes, row_count)?;
            if !mapping_write::read_rect_raw_at(
                state,
                host,
                t.mapping_id,
                t11_window(t),
                mapping_write::Rect {
                    origin_x,
                    origin_y,
                    width: pixels,
                    height,
                },
                &mut buf[..need as usize],
                row_bytes as u32,
            ) {
                return Err(br(BlitFailure::GuestIo, "rd_rect_t11_io"));
            }
            Ok(())
        }
    }
}

/// A linear level's rectangle as the GVA rail's own shape: where it starts and
/// how its rows are laid out.
///
/// This is the linear endpoint's missing rect description. The mapper-ref-texture endpoint
/// has had one since it landed — [`mapping_write::write_rect_raw_at`] and
/// friends — while the linear endpoint had only [`write_texture_row`] and
/// [`read_texture_row`], so [`write_texture_rect`] and [`read_texture_rect`]
/// each served a rectangle by re-entering the GVA rail `row_count` times. Every
/// one of those re-entries walks the task page table afresh for a row of the
/// same allocation, so all but the first re-derive an answer already in hand. A
/// driven macos-13 boot charged that loop 906.7 ms of a 916.6 ms
/// texture-to-texture rail across 118 464 rows — about 7.6 us for a 4 KiB row,
/// which is the walk and not the bytes.
///
/// **The stride is the contract's, not an observation.**
/// [`LinearTextureLevel::texel_offset`]'s `y` term is exactly `y * row_stride`,
/// so consecutive rows of one rectangle are `row_stride` apart by construction.
/// That makes the whole rectangle one [`RectStride`] over one span, which is
/// what lets a single walk place every row.
///
/// Only the last row's offset is resolved alongside the first. That is not a
/// two-endpoint sample of a range: `texel_offset` is affine and increasing in
/// `y`, and its only `y` bound is `y < height`, so the largest `y` is the one
/// that can fail and checking it checks them all.
fn linear_rect(
    t: &LinearTextureLevel,
    origin: Point,
    row_bytes: u64,
    row_count: u64,
    site: &'static str,
) -> Result<(u64, RectStride), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    let last_y = oy
        .checked_add(row_count.saturating_sub(1))
        .ok_or_else(|| br(BlitFailure::Bounds, site))?;
    let first = t
        .texel_offset(ox, oy, oz)
        .ok_or_else(|| br(BlitFailure::Bounds, site))?;
    t.texel_offset(ox, last_y, oz)
        .ok_or_else(|| br(BlitFailure::Bounds, site))?;
    let gva = t
        .base_gva
        .checked_add(first)
        .ok_or_else(|| br(BlitFailure::Bounds, site))?;
    let rect = RectStride::new(t.row_stride, row_bytes, row_count)
        .ok_or_else(|| br(BlitFailure::Bounds, site))?;
    Ok((gva, rect))
}

/// Write a whole `row_bytes`-wide, `row_count`-tall rectangle at `origin` from
/// `buf`, rows packed `row_bytes` apart. The rect counterpart of
/// [`write_texture_row`]; see [`read_texture_rect`] for why the rect is the
/// unit.
///
/// A rect that covers a mapper-ref-texture plane entirely goes through
/// [`mapping_write::write_full_rect_raw_at`], whose fragmented arm imports each
/// maximal packed GPA run once instead of once per row. The two calls address
/// identical guest bytes; only the fragmented staging differs.
#[allow(
    clippy::too_many_arguments,
    reason = "the rect helper names the same geometry its row counterpart does"
)]
fn write_texture_rect<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_bytes: u64,
    row_count: u64,
    buf: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), BlitStatus> {
    let need = row_bytes
        .checked_mul(row_count)
        .ok_or_else(|| br(BlitFailure::Capacity, "wr_rect_span_overflow"))?;
    if need as usize > buf.len() {
        return Err(br(BlitFailure::Capacity, "wr_rect_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let (gva, rect) = linear_rect(t, origin, row_bytes, row_count, "wr_rect_linear_shape")?;
            crate::runtime::gva_view::write_rect_within(
                state, host, task_id, gva, rect, buf, allowed,
            )
            .map_err(|_| br(BlitFailure::GuestIo, "wr_rect_linear_io"))?;
            crate::runtime::drain::note_store_route("blit_rect_linear_walk");
            crate::runtime::drain::note_store_route_n(
                "blit_rect_linear_rows_hoisted",
                row_count.saturating_sub(1),
            );
            Ok(())
        }
        TextureBacking::MapperRefTexture(t) => {
            let (pixels, height, origin_x, origin_y) =
                t11_rect_extent(t, origin, row_bytes, row_count)?;
            let src = &buf[..need as usize];
            let ok = if origin_x == 0 && origin_y == 0 && pixels == t.width && height == t.height {
                mapping_write::write_full_rect_raw_at(
                    state,
                    host,
                    t.mapping_id,
                    t.surface_offset,
                    t.row_stride,
                    t.span_end,
                    pixels,
                    height,
                    t.bpp,
                    src,
                    row_bytes as u32,
                )
            } else {
                mapping_write::write_rect_raw_at(
                    state,
                    host,
                    t.mapping_id,
                    t11_window(t),
                    mapping_write::Rect {
                        origin_x,
                        origin_y,
                        width: pixels,
                        height,
                    },
                    src,
                    row_bytes as u32,
                )
            };
            if !ok {
                return Err(br(BlitFailure::GuestIo, "wr_rect_t11_io"));
            }
            Ok(())
        }
    }
}

/// The mapping-rail sample window a mapper-ref-texture backing names.
///
/// Spelled once so the four rect/row call sites cannot drift on which of the
/// four fields a copy presents.
fn t11_window(t: &MapperRefTexture) -> mapping_write::SurfaceWindow {
    mapping_write::SurfaceWindow {
        base_off: t.surface_offset,
        bpr: t.row_stride,
        span_end: t.span_end,
        bpp: t.bpp,
    }
}

/// Narrow a rect's texel geometry to the `u32` the mapping rail's [`mapping_write::Rect`]
/// is expressed in, refusing by name rather than truncating.
fn t11_rect_extent(
    t: &MapperRefTexture,
    origin: Point,
    row_bytes: u64,
    row_count: u64,
) -> Result<(u32, u32, u32, u32), BlitStatus> {
    if origin.z != 0 {
        return Err(br(BlitFailure::Unsupported, "rect_t11_z"));
    }
    if t.bpp == 0 {
        return Err(br(BlitFailure::Bounds, "rect_t11_bpp_zero"));
    }
    let origin_x =
        u32::try_from(origin.x).map_err(|_| br(BlitFailure::Bounds, "rect_t11_x_range"))?;
    let origin_y =
        u32::try_from(origin.y).map_err(|_| br(BlitFailure::Bounds, "rect_t11_y_range"))?;
    let height =
        u32::try_from(row_count).map_err(|_| br(BlitFailure::Bounds, "rect_t11_height_range"))?;
    let pixels = u32::try_from(row_bytes / t.bpp as u64)
        .map_err(|_| br(BlitFailure::Bounds, "rect_t11_width_range"))?;
    Ok((pixels, height, origin_x, origin_y))
}

/// Census: does the surface this blit is about to copy through its **guest
/// pages** have live GPU-resident content instead?
///
/// The sampled rail and the blit rail consume the same wire form — a mapper-ref-texture
/// IOSurface, named directly or through a ref-texture view — and they resolve it
/// completely differently. `draw::vulkan`'s sampled resolver runs a four-rung
/// ladder whose top rung is `t11rung_resident`, the engine image, and a driven
/// session puts 64-93 % of its binds there. This resolver has no ladder at all:
/// it returns a [`MapperRefTexture`] over the mapping's guest pages every time, and
/// the copy then reads and writes those pages on the CPU.
///
/// That is only sound while the guest pages hold the surface's newest content.
/// The writeback debt is what is supposed to make that true, and
/// `mapping_write`'s settle pays it before every read — but a resident carrying
/// `gpu_only_content` with no debt armed owes nothing, so nothing lands, and the
/// copy reads whatever the pages held before. A blit is not a decode failure and
/// not a refusal: it succeeds, and the pixels are simply not the guest's.
///
/// So this counts rather than branches. `blit_t11_resident_ready` above zero
/// says this device is copying a surface whose authoritative content is on the
/// GPU, which is the reading that decides whether the blit rail needs the
/// sampled rail's ladder. `_not_ready` beside it is the denominator, so a zero
/// can be told from an arm that never ran.
/// What a texture-to-texture copy is actually made of: which pair of backings it
/// joins, and how many bytes it moves through the host.
///
/// `walk_blit_us` says this rail costs 33.6 s of a 45 s driven Maps window
/// against 0.45 s for 1.49 M render records, and a per-record average cannot say
/// whether that is a few enormous copies or many small ones, nor which pair of
/// endpoint kinds is paying it. A GPU-side rail can only serve pairs whose
/// **both** ends this device can hold as images, so the pair split is what
/// decides whether such a rail would serve the workload or a corner of it.
///
/// `blit_t2t_bytes` is the denominator for every later claim about this rail: a
/// copy that is genuinely moving a gigabyte a second through `memcpy` is a
/// bandwidth problem, and one that is not is a per-row overhead problem, and the
/// two have opposite repairs. The two reverted attempts in `09a45414` and
/// `81f99f4f` were both aimed at granularity without this number in hand.
fn note_t2t_shape(
    src: &TextureBacking,
    dst: &TextureBacking,
    copy_w: u64,
    copy_h: u64,
    copy_d: u64,
    copy_bpp: u32,
) {
    use crate::runtime::drain::{note_store_route, note_store_route_n};
    note_store_route(
        match (src.is_mapper_ref_texture(), dst.is_mapper_ref_texture()) {
            (false, false) => "blit_t2t_linear_linear",
            (false, true) => "blit_t2t_linear_t11",
            (true, false) => "blit_t2t_t11_linear",
            (true, true) => "blit_t2t_t11_t11",
        },
    );
    let bytes = copy_w
        .saturating_mul(copy_h)
        .saturating_mul(copy_d)
        .saturating_mul(u64::from(copy_bpp));
    note_store_route_n("blit_t2t_bytes", bytes);
    // Banded rather than averaged, because the mean of a full-window copy and a
    // 16x16 icon says nothing about either and this rail issues both.
    note_store_route(match bytes {
        0..=4_095 => "blit_t2t_band_tiny",
        4_096..=262_143 => "blit_t2t_band_small",
        262_144..=4_194_303 => "blit_t2t_band_medium",
        _ => "blit_t2t_band_large",
    });
}

/// Whether a blit endpoint arrived owing this device a writeback, split by which
/// spelling of the debt named it.
///
/// The payment itself is unconditional and silent — `pay_for_texture` early-exits
/// on an empty ledger — so without this the repair is unattributable: a rail that
/// never had a debt to pay and a rail that pays thousands look identical from
/// outside, and the screen cannot tell them apart either.
///
/// `gva` is the interesting one. It counts blit endpoints whose real content was sitting in an
/// engine resident behind an armed [`crate::runtime::writeback_debt::GvaWritebackDebt`], i.e.
/// copies that read guest pages the render never reached. `surface` is the
/// mapper-ref-texture/backing spelling, which `mapping_write`'s own settle already covered from the
/// other side, so a large `surface` next to a zero `gva` says this change bought nothing new.
fn note_blit_endpoint_debt(state: &DeviceState, task_id: u32, texture_ref: u32) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    if state
        .pending_writebacks
        .has_gva(crate::runtime::writeback_debt::GvaResourceKey {
            task_id,
            texture_ref,
        })
    {
        crate::runtime::drain::note_store_route("blit_endpoint_owed_gva");
    }
    let mapped = state
        .texture_to_mapping
        .get(&(task_id, texture_ref))
        .copied();
    if state.pending_writebacks.get(texture_ref).is_some()
        || mapped.is_some_and(|id| state.pending_writebacks.get(id).is_some())
    {
        crate::runtime::drain::note_store_route("blit_endpoint_owed_surface");
    }
}

fn range_fits(offset: u64, length: u64, size: u64) -> bool {
    offset <= size && length <= size - offset
}

fn ranges_overlap(a0: u64, a_len: u64, b0: u64, b_len: u64) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let a1 = a0.saturating_add(a_len);
    let b1 = b0.saturating_add(b_len);
    a0 < b1 && b0 < a1
}

use gva_mem::dest_window;

// The Vulkan rail's two blit fast paths. Named rather than re-exported flat, so
// `execute_blit` reads the same on both rails and reaches either only through
// `Backend`.
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

/// [`dest_window`] over the region of a texture a row loop is about to write.
///
/// The span is measured with the texture's own [`LinearTextureLevel::texel_offset`],
/// first row to last, so a level, slice or plane stride this bound does not
/// model cannot place a row outside the set it authorises.
///
/// `Ok(None)` for a mapper-ref-texture: that write goes through the mapping rail,
/// authorised by the page list the guest declared for the mapping. Walking a
/// GVA span for it would bound the wrong address space.
///
/// Every other way this can fail to produce a bound is an `Err`, never a
/// `None`. `None` reaches [`write_texture_row`] as "authorised by the command",
/// so an arithmetic failure answered with `None` would *widen* the write from
/// the region the guest named to the whole address space — the opposite of what
/// failing to measure that region should do. The geometry here is the copy the
/// command decoded; if it does not resolve, the command does not execute.
#[allow(
    clippy::too_many_arguments,
    reason = "the window mirrors the copy extent the row loop walks"
)]
fn texture_region_window<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    copy_w: u32,
    copy_h: u64,
    copy_d: u64,
    bpp: u32,
) -> Result<Option<std::collections::HashSet<u64>>, BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    let TextureBacking::Linear(t) = tex else {
        return Ok(None);
    };
    // An empty extent authorises no page, which is exact: every caller's row
    // loop is `for z in 0..copy_d { for y in 0..copy_h`, so it writes nothing.
    let (Some(last_row), Some(last_plane)) = (copy_h.checked_sub(1), copy_d.checked_sub(1)) else {
        return Ok(Some(std::collections::HashSet::new()));
    };
    let oob = |slug| br(BlitFailure::Bounds, slug);
    let first = t
        .texel_offset(ox, oy, oz)
        .ok_or_else(|| oob("tex_window_first_texel_oob"))?;
    let last = oy
        .checked_add(last_row)
        .zip(oz.checked_add(last_plane))
        .and_then(|(y, z)| t.texel_offset(ox, y, z))
        .ok_or_else(|| oob("tex_window_last_texel_oob"))?;
    let row_bytes = (copy_w as u64)
        .checked_mul(bpp as u64)
        .ok_or_else(|| oob("tex_window_row_bytes_overflow"))?;
    let span = last
        .checked_add(row_bytes)
        .and_then(|end| end.checked_sub(first))
        .ok_or_else(|| oob("tex_window_span_overflow"))?;
    let base = t
        .base_gva
        .checked_add(first)
        .ok_or_else(|| oob("tex_window_base_overflow"))?;
    Ok(dest_window(state, host, task_id, base, span))
}

/// Highest byte offset past `base` a strided plane/row walk reaches.
///
/// The bound wants the span the command names, not the resource's whole
/// allocation: a copy into the top-left corner of a 64 MiB texture must not be
/// authorised for the other 63. Derived from the walk's own geometry — last
/// plane, last row, one row of bytes — so it cannot drift from the loop it
/// bounds.
fn strided_span(
    row_bytes: u64,
    row_stride: u64,
    row_count: u64,
    image_stride: u64,
    image_count: u64,
) -> Option<u64> {
    let last_image = image_count.checked_sub(1)?.checked_mul(image_stride)?;
    let last_row = row_count.checked_sub(1)?.checked_mul(row_stride)?;
    last_image.checked_add(last_row)?.checked_add(row_bytes)
}

fn write_fill_range<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut DeviceState,
    task_id: u32,
    gva: u64,
    length: u64,
    value: u8,
) -> Result<(), BlitStatus> {
    write_fill_pattern(host, state, task_id, gva, length, &[value])
}

/// Write `length` bytes at `gva` by repeating `pattern` from its first byte.
///
/// One body for both fill records rather than a copy per record. The byte fill
/// is this with a one-byte pattern, which is not a generalisation for its own
/// sake: the bounds check, the `dest_window` authorisation, the chunked write
/// and the GVA advance are the four things a second copy would have to keep in
/// step, and this rail has already produced three bugs from two arms of one
/// guest-memory write drifting apart.
///
/// Phase is preserved across chunks because the tile is a whole number of
/// patterns, so every chunk begins on pattern byte 0 exactly as the first one
/// does. That is asserted rather than assumed — a `CHUNK` that stopped being a
/// multiple of the pattern width would shift every byte after the first tile.
fn write_fill_pattern<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut DeviceState,
    task_id: u32,
    gva: u64,
    length: u64,
    pattern: &[u8],
) -> Result<(), BlitStatus> {
    if length == 0 {
        return Ok(());
    }
    debug_assert!(!pattern.is_empty(), "a fill with no pattern writes nothing");
    debug_assert_eq!(
        CHUNK % pattern.len(),
        0,
        "the staging tile must hold a whole number of patterns, or the phase \
         shifts at every chunk boundary"
    );
    let allowed = dest_window(state, host, task_id, gva, length);
    let mut remaining = length;
    let mut cur = gva;
    let mut chunk = vec![0u8; CHUNK];
    for (i, b) in chunk.iter_mut().enumerate() {
        *b = pattern[i % pattern.len()];
    }
    while remaining > 0 {
        let n = remaining.min(CHUNK as u64) as usize;
        if gva_mem::write_task_gva_product_within(
            state,
            host,
            task_id,
            cur,
            &chunk[..n],
            allowed.as_ref(),
        )
        .is_err()
        {
            return Err(br(BlitFailure::GuestIo, "fill_write_io"));
        }
        cur = cur
            .checked_add(n as u64)
            .ok_or_else(|| br(BlitFailure::Capacity, "fill_gva_advance_overflow"))?;
        remaining -= n as u64;
    }
    Ok(())
}

fn copy_bytes<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut DeviceState,
    task_id: u32,
    src_gva: u64,
    dst_gva: u64,
    length: u64,
) -> Result<(), BlitStatus> {
    if length == 0 {
        return Ok(());
    }
    let allowed = dest_window(state, host, task_id, dst_gva, length);
    copy_bytes_within(
        host,
        state,
        task_id,
        src_gva,
        dst_gva,
        length,
        allowed.as_ref(),
    )
}

/// [`copy_bytes`] with the destination window supplied rather than captured.
///
/// The split exists so a test can run the identical loop unbounded and show
/// where the bytes land without it. Product code calls [`copy_bytes`]; there is
/// no product caller that should be choosing its own window here, because the
/// window this copy is entitled to is the destination span it was given.
#[allow(
    clippy::too_many_arguments,
    reason = "the bounded form adds the window to the copy's own geometry"
)]
fn copy_bytes_within<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut DeviceState,
    task_id: u32,
    src_gva: u64,
    dst_gva: u64,
    length: u64,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), BlitStatus> {
    if length == 0 {
        return Ok(());
    }
    let mut remaining = length;
    let mut s = src_gva;
    let mut d = dst_gva;
    let mut buf = vec![0u8; CHUNK.min(length as usize).max(1)];
    while remaining > 0 {
        let n = remaining.min(buf.len() as u64) as usize;
        if gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            s,
            &mut buf[..n],
            state.page_shift,
        )
        .is_err()
        {
            return Err(br(BlitFailure::GuestIo, "copy_bytes_read_io"));
        }
        if gva_mem::write_task_gva_product_within(state, host, task_id, d, &buf[..n], allowed)
            .is_err()
        {
            return Err(br(BlitFailure::GuestIo, "copy_bytes_write_io"));
        }
        s = s
            .checked_add(n as u64)
            .ok_or_else(|| br(BlitFailure::Capacity, "copy_bytes_src_overflow"))?;
        d = d
            .checked_add(n as u64)
            .ok_or_else(|| br(BlitFailure::Capacity, "copy_bytes_dst_overflow"))?;
        remaining -= n as u64;
    }
    Ok(())
}

/// Copy a rectangular multi-plane region with independent source/dest strides.
#[allow(
    clippy::too_many_arguments,
    reason = "the copy helper mirrors independent source and destination row geometry"
)]
fn copy_row_region<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut DeviceState,
    task_id: u32,
    src_base: u64,
    src_row_stride: u64,
    src_image_stride: u64,
    dst_base: u64,
    dst_row_stride: u64,
    dst_image_stride: u64,
    row_bytes: u64,
    row_count: u64,
    image_count: u64,
) -> Result<(), BlitStatus> {
    if row_bytes == 0 || row_count == 0 || image_count == 0 {
        return Ok(());
    }
    // Stride/row contract only — no host MiB byte budget (chunked row I/O).
    if row_bytes > src_row_stride || row_bytes > dst_row_stride {
        return Err(br(BlitFailure::Bounds, "copy_region_row_gt_stride"));
    }
    let _total = row_bytes
        .checked_mul(row_count)
        .and_then(|v| v.checked_mul(image_count))
        .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_total_overflow"))?;
    let row_len = host_alloc_len(row_bytes)
        .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_row_alloc"))?;
    let dst_span = strided_span(
        row_bytes,
        dst_row_stride,
        row_count,
        dst_image_stride,
        image_count,
    )
    .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_dst_span_overflow"))?;
    // The window is built once and the rows run against it, so a single total
    // over the pair cannot say which is the cost. `dest_window` walks the whole
    // destination span's guest page table into a `HashSet`, which is per-record
    // work that does not shrink with the copy; the row loop is per-row work that
    // does. Timed apart because the repair differs.
    let window_started = std::time::Instant::now();
    let allowed = dest_window(state, host, task_id, dst_base, dst_span);
    crate::runtime::drain::note_store_route_us(
        "blit_window_us",
        window_started.elapsed().as_micros() as u64,
    );
    let rows_started = std::time::Instant::now();
    crate::runtime::drain::note_store_route_n("blit_rows_n", row_count.saturating_mul(image_count));
    let mut row_buf = vec![0u8; row_len];
    for z in 0..image_count {
        let src_plane = src_base
            .checked_add(
                z.checked_mul(src_image_stride)
                    .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_src_plane_overflow"))?,
            )
            .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_src_plane_overflow"))?;
        let dst_plane = dst_base
            .checked_add(
                z.checked_mul(dst_image_stride)
                    .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_dst_plane_overflow"))?,
            )
            .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_dst_plane_overflow"))?;
        for y in 0..row_count {
            let s = src_plane
                .checked_add(
                    y.checked_mul(src_row_stride)
                        .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_src_row_overflow"))?,
                )
                .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_src_row_overflow"))?;
            let d = dst_plane
                .checked_add(
                    y.checked_mul(dst_row_stride)
                        .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_dst_row_overflow"))?,
                )
                .ok_or_else(|| br(BlitFailure::Capacity, "copy_region_dst_row_overflow"))?;
            if gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                s,
                &mut row_buf,
                state.page_shift,
            )
            .is_err()
            {
                note_copy_region_io(task_id, false, s, y, z, row_bytes, state.page_shift);
                return Err(br(BlitFailure::GuestIo, "copy_region_read_io"));
            }
            if gva_mem::write_task_gva_product_within(
                state,
                host,
                task_id,
                d,
                &row_buf,
                allowed.as_ref(),
            )
            .is_err()
            {
                note_copy_region_io(task_id, true, d, y, z, row_bytes, state.page_shift);
                return Err(br(BlitFailure::GuestIo, "copy_region_write_io"));
            }
        }
    }
    crate::runtime::drain::note_store_route_us(
        "blit_rows_us",
        rows_started.elapsed().as_micros() as u64,
    );
    Ok(())
}

fn exec_fill_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &FillBuffer,
    value: u8,
) -> BlitStatus {
    if cmd.length == 0 {
        return BlitStatus::ZeroExtent;
    }
    let buf = match resolve_buffer(state, host, task_id, cmd.buffer_ref) {
        Ok(b) => b,
        Err(st) => return st,
    };
    if !range_fits(cmd.location, cmd.length, buf.size) {
        return br(BlitFailure::Bounds, "fill_range_oob");
    }
    let gva = match buf.gva.checked_add(cmd.location) {
        Some(v) => v,
        None => return br(BlitFailure::Bounds, "fill_gva_overflow"),
    };
    match write_fill_range(host, state, task_id, gva, cmd.length, value) {
        Ok(()) => BlitStatus::Ok,
        Err(st) => st,
    }
}

/// `fillBuffer:range:pattern4:` — the byte fill with a repeating 32-bit unit.
///
/// Same resolve, same bounds check and same write path as
/// [`exec_fill_buffer`]; only the chunk this repeats differs. That is the whole
/// difference between the two records on the wire too — one length, one layout,
/// and a last field that is one byte wide in `0x132` and four in `0x13f`.
///
/// # Why an unaligned range is refused rather than filled
///
/// The record settles *what* repeats and says nothing about the **phase** it
/// repeats on, and there are two readings that differ:
///
/// - the pattern restarts at `range.location`, so byte `i` of the range takes
///   `pattern[i % 4]`;
/// - the pattern is anchored to the buffer, so the byte at buffer offset `o`
///   takes `pattern[o % 4]`.
///
/// The two agree for every byte exactly when `range.location % 4 == 0`, and
/// nothing this project can reach decides between them otherwise: the
/// serializer forwards the arguments untouched, so the choice belongs to
/// Apple's host implementation and no capture of the command stream can see it.
///
/// So the aligned case is executed — no guess is being made there — and the
/// unaligned case is refused by name. Filling it under either reading would put
/// bytes in guest memory that are plausible and possibly wrong, which is worse
/// than a refusal the fail log explains. `fill_pattern4_unaligned_range` is a
/// healthy zero, and a non-zero reading is the measured argument for going and
/// deriving the phase rule.
fn exec_fill_buffer_pattern4<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &FillBuffer,
    pattern_word: u32,
) -> BlitStatus {
    if cmd.length == 0 {
        return BlitStatus::ZeroExtent;
    }
    let pattern = pattern_word.to_le_bytes();
    if !cmd.location.is_multiple_of(pattern.len() as u64) {
        return br(BlitFailure::Unsupported, "fill_pattern4_unaligned_range");
    }
    let buf = match resolve_buffer(state, host, task_id, cmd.buffer_ref) {
        Ok(b) => b,
        Err(st) => return st,
    };
    if !range_fits(cmd.location, cmd.length, buf.size) {
        return br(BlitFailure::Bounds, "fill_pattern4_range_oob");
    }
    let gva = match buf.gva.checked_add(cmd.location) {
        Some(v) => v,
        None => return br(BlitFailure::Bounds, "fill_pattern4_gva_overflow"),
    };
    match write_fill_pattern(host, state, task_id, gva, cmd.length, &pattern) {
        Ok(()) => BlitStatus::Ok,
        Err(st) => st,
    }
}

fn exec_copy_buffer_to_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &BufferToBuffer,
) -> BlitStatus {
    if cmd.size == 0 {
        return BlitStatus::ZeroExtent;
    }
    let src = match resolve_buffer(state, host, task_id, cmd.source_ref) {
        Ok(b) => b,
        Err(st) => return st,
    };
    let dst = match resolve_buffer(state, host, task_id, cmd.dest_ref) {
        Ok(b) => b,
        Err(st) => return st,
    };
    if !range_fits(cmd.source_offset, cmd.size, src.size)
        || !range_fits(cmd.dest_offset, cmd.size, dst.size)
    {
        return br(BlitFailure::Bounds, "b2b_range_oob");
    }
    // Same allocation (same GVA base + size): reject overlapping windows.
    if src.gva == dst.gva
        && src.size == dst.size
        && ranges_overlap(cmd.source_offset, cmd.size, cmd.dest_offset, cmd.size)
    {
        return br(BlitFailure::Overlap, "b2b_overlap");
    }
    let s = match src.gva.checked_add(cmd.source_offset) {
        Some(v) => v,
        None => return br(BlitFailure::Bounds, "b2b_src_gva_overflow"),
    };
    let d = match dst.gva.checked_add(cmd.dest_offset) {
        Some(v) => v,
        None => return br(BlitFailure::Bounds, "b2b_dst_gva_overflow"),
    };
    match copy_bytes(host, state, task_id, s, d, cmd.size) {
        Ok(()) => BlitStatus::Ok,
        Err(st) => st,
    }
}

/// A copy extent the guest asked for, checked against what the texture holds.
///
/// `None` means the region reaches past the edge and the caller must refuse.
///
/// **This used to clamp**, at nine sites here plus three destination-side cuts
/// in the texture-to-texture path that did not even come through it, and it
/// reported nothing. A truncation there was a smaller copy returned as `Ok`:
/// the guest asked Metal to move a W x H region, got fewer texels, and the
/// texels outside the cut kept whatever the destination held before, with no
/// status saying so.
///
/// Three things say refusing is the faithful answer, and no one of them would
/// have been enough alone:
///
/// * **Metal refuses.** A region reaching past a texture is a validation
///   failure there, not a clipped copy. Emulating the clip emulates a device
///   Apple does not ship.
/// * **The origin check beside every call site already refuses**
///   (`t2t_origin_oob` and its siblings). Origin and extent are two halves of
///   one region out of one wire record and were handled opposite ways — which
///   is exactly the divergence `AGENTS.md` says to look for by diffing two arms
///   that consume one wire form, and which nobody had diffed.
/// * **The cut never fires.** Measured before it was changed, on a driven x86
///   boot with Safari composited on a Ventura desktop: `blit_extent_fits` 66,
///   `blit_extent_cut` **0**. The path is reached and the clamp is not.
///
/// The counters stay, and `blit_extent_cut` keeps its name so a boot series
/// spanning this change stays comparable — it is now the refusal's volume
/// rather than a silent truncation's. A firing is a workload this rig has not
/// seen, and it will be loud instead of a wrong texture.
fn copy_extent(kind: &'static str, axis: &'static str, requested: u64, max: u64) -> Option<u64> {
    if requested == 0 {
        // Metal size 0 is a no-op extent; keep 0. The callers below turn an
        // all-zero extent into `ZeroExtent`, which is not a refusal.
        Some(0)
    } else if requested > max {
        note_extent_over(kind, axis, requested, max);
        None
    } else {
        crate::runtime::drain::note_store_route("blit_extent_fits");
        Some(requested)
    }
}

/// The always-on half of [`copy_extent`], shared with the destination-side
/// checks in the texture-to-texture path that do not go through it.
///
/// Latched per `(kind, axis)` rather than per size: what a reader needs first is
/// which copy was refused and on which dimension, and a per-size latch on a
/// window drag would emit once per distinct window width.
fn note_extent_over(kind: &'static str, axis: &'static str, requested: u64, max: u64) {
    crate::runtime::drain::note_store_route("blit_extent_cut");
    let key = kind
        .bytes()
        .chain(axis.bytes())
        .fold(0u64, |acc, b| acc.rotate_left(7) ^ u64::from(b));
    if !crate::observe::first_sight("blit_extent_cut", key) {
        return;
    }
    crate::observe::fail(format!(
        "blit_extent reason=blit_extent_over kind={kind} axis={axis}          requested={requested} available={max} (the guest asked to copy past          the edge of a texture; Metal refuses that region and so does this,          where it used to copy less and report Ok)"
    ));
}

/// Resolve `MTLBlitOption` → aspect flags + buffer-side plane bpp.
/// The plane a copy addresses, and what one of its texels costs.
///
/// Takes the `MTLBlitOption` word rather than the record, because the three
/// copies that carry one carry it at three different widths and only one of
/// them can be right for any given call. The absent word is a zero: a record
/// with no `options:` field selects the whole texel, which is what
/// `select_aspect(0)` answers, so "this record has no options word" and "this
/// record's options word is zero" no longer need to be two states.
fn copy_aspect_for_options(
    texture_format: u16,
    options: u32,
) -> Result<(BlitAspect, u32), BlitStatus> {
    // The three option checks used to collapse into a bare `Unsupported` with
    // the reason discarded by `map_err(|_| ..)`. The blit reason channel carries
    // the specific slug to the dispatch-site line, so an unknown option bit and
    // a depth+stencil conflict no longer read identically.
    let aspect = reims_vgpu_protocol::blit::select_aspect(options)
        .map_err(|e| br(BlitFailure::Unsupported, e.slug()))?;
    let bpp = pixel_format::blit_aspect_bytes_per_pixel(texture_format, aspect)
        .ok_or(br(BlitFailure::Unsupported, "aspect_bpp_unknown"))?;
    Ok((aspect, bpp))
}

/// Texture-side full texel bpp (storage). Plane copies use this for GVA strides.
fn texture_storage_bpp(format: u16) -> Result<u32, BlitStatus> {
    pixel_format::bytes_per_pixel(format).ok_or(br(BlitFailure::Unsupported, "storage_bpp_unknown"))
}

/// Read one packed texture row (tight `width * storage_bpp`) at (ox, oy+row_i, oz).
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper keeps packed texture coordinates and format explicit"
)]
fn read_texture_storage_row<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_i: u64,
    width: u32,
    storage_bpp: u32,
    buf: &mut [u8],
) -> Result<(), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    let row_bytes = (width as u64).checked_mul(storage_bpp as u64).ok_or(br(
        BlitFailure::Capacity,
        "bt_read_storage_row_bytes_overflow",
    ))?;
    if row_bytes as usize > buf.len() {
        return Err(br(BlitFailure::Capacity, "bt_read_storage_row_over_buf"));
    }
    // Reuse read_texture_row but with storage row size (not plane size).
    // Temporarily: call the same GVA path with storage row_bytes.
    read_texture_row(
        state,
        host,
        task_id,
        tex,
        Point {
            x: ox,
            y: oy,
            z: oz,
        },
        row_i,
        row_bytes,
        buf,
    )
}

/// Write one packed texture row.
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper keeps packed texture coordinates and format explicit"
)]
fn write_texture_storage_row<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_i: u64,
    width: u32,
    storage_bpp: u32,
    buf: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    let row_bytes = (width as u64).checked_mul(storage_bpp as u64).ok_or(br(
        BlitFailure::Capacity,
        "bt_write_storage_row_bytes_overflow",
    ))?;
    write_texture_row(
        state,
        host,
        task_id,
        tex,
        Point {
            x: ox,
            y: oy,
            z: oz,
        },
        row_i,
        row_bytes,
        buf,
        allowed,
    )
}

/// Copy buffer plane rows ↔ texture with optional combined-DS plane repack.
#[allow(
    clippy::too_many_arguments,
    reason = "the blit executor mirrors the decoded buffer, texture, and aspect fields"
)]
fn copy_buffer_texture_rows_aspect<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buf_base_gva: u64,
    buf_row_stride: u64,
    buf_image_stride: u64,
    tex: &TextureBacking,
    tex_origin: Point,
    copy_w: u32,
    copy_h: u64,
    copy_d: u64,
    plane_bpp: u32,
    aspect: BlitAspect,
    to_texture: bool,
) -> Result<(), BlitStatus> {
    let Point {
        x: tex_ox,
        y: tex_oy,
        z: tex_oz,
    } = tex_origin;
    let fmt = tex.pixel_format();
    let repack = pixel_format::blit_aspect_needs_repack(fmt, aspect);
    let storage_bpp = if repack {
        texture_storage_bpp(fmt)?
    } else {
        plane_bpp
    };
    let plane_row = (copy_w as u64)
        .checked_mul(plane_bpp as u64)
        .ok_or(br(BlitFailure::Capacity, "bt_plane_row_bytes_overflow"))?
        as usize;
    let storage_row = (copy_w as u64)
        .checked_mul(storage_bpp as u64)
        .ok_or(br(BlitFailure::Capacity, "bt_storage_row_bytes_overflow"))?
        as usize;
    // The other half of the rail. `note_t2t_shape` covers texture-to-texture
    // only, and that left 4 243 of a driven Maps leg's 26 234 blit records
    // uncounted — a population big enough to hold the whole per-record cost if
    // its copies are large, and invisible in a census that stops at one copy
    // kind. Same three readings, so the two halves are comparable line for line.
    {
        use crate::runtime::drain::{note_store_route, note_store_route_n};
        note_store_route(match (to_texture, tex.is_mapper_ref_texture()) {
            (true, false) => "blit_b2t_linear",
            (true, true) => "blit_b2t_t11",
            (false, false) => "blit_t2b_linear",
            (false, true) => "blit_t2b_t11",
        });
        let bytes = (plane_row as u64)
            .saturating_mul(copy_h)
            .saturating_mul(copy_d);
        note_store_route_n("blit_bt_bytes", bytes);
        note_store_route_n("blit_bt_rows_n", copy_h.saturating_mul(copy_d));
        note_store_route(match bytes {
            0..=4_095 => "blit_bt_band_tiny",
            4_096..=262_143 => "blit_bt_band_small",
            262_144..=4_194_303 => "blit_bt_band_medium",
            _ => "blit_bt_band_large",
        });
    }
    let mut plane = vec![0u8; plane_row];
    let mut packed = vec![0u8; storage_row.max(plane_row)];
    // Destination pages, taken before the loop below rather than per row. The
    // loop *is* the copy -- up to `copy_d * copy_h` guest reads and writes -- and
    // it re-derives its destination from `tex`/`buf_base_gva` on every one of
    // them, which is the drift `dest_window` exists to close.
    let allowed = if to_texture {
        texture_region_window(
            state,
            host,
            task_id,
            tex,
            Point {
                x: tex_ox,
                y: tex_oy,
                z: tex_oz,
            },
            copy_w,
            copy_h,
            copy_d,
            storage_bpp,
        )?
    } else {
        let span = strided_span(
            plane_row as u64,
            buf_row_stride,
            copy_h,
            buf_image_stride,
            copy_d,
        )
        .ok_or(br(BlitFailure::Capacity, "bt_dest_span_overflow"))?;
        dest_window(state, host, task_id, buf_base_gva, span)
    };
    // The half `blit_rows_us` cannot see. That counter sits in `copy_row_region`,
    // which only the linear-to-linear fast path reaches; every mapper-ref-texture and
    // ref-texture endpoint stages through this loop instead, and each of its rows
    // re-vouches the mapping's guest page table. `mapw_pages_vouched` reads over
    // a million on a driven Maps leg and nothing timed the loop that spends them.
    let bt_rows_started = std::time::Instant::now();
    for z in 0..copy_d {
        for y in 0..copy_h {
            let buf_gva = buf_base_gva
                .checked_add(
                    z.checked_mul(buf_image_stride)
                        .ok_or(br(BlitFailure::Capacity, "bt_slice_offset_overflow"))?,
                )
                .ok_or(br(BlitFailure::Capacity, "bt_slice_base_overflow"))?
                .checked_add(
                    y.checked_mul(buf_row_stride)
                        .ok_or(br(BlitFailure::Capacity, "bt_row_offset_overflow"))?,
                )
                .ok_or(br(BlitFailure::Capacity, "bt_row_base_overflow"))?;
            if to_texture {
                if gva_mem::read_task_gva_by_id(
                    host,
                    &state.tasks,
                    task_id,
                    buf_gva,
                    &mut plane,
                    state.page_shift,
                )
                .is_err()
                {
                    return Err(br(BlitFailure::GuestIo, "bt_read_guest_row"));
                }
                if repack {
                    // RMW: load existing packed row, insert plane, store.
                    read_texture_storage_row(
                        state,
                        host,
                        task_id,
                        tex,
                        Point {
                            x: tex_ox,
                            y: tex_oy,
                            z: tex_oz + z,
                        },
                        y,
                        copy_w,
                        storage_bpp,
                        &mut packed,
                    )?;
                    if !pixel_format::insert_plane_row(
                        fmt,
                        aspect,
                        &plane,
                        copy_w,
                        &mut packed[..storage_row],
                    ) {
                        return Err(br(BlitFailure::Unsupported, "bt_insert_plane_row"));
                    }
                    write_texture_storage_row(
                        state,
                        host,
                        task_id,
                        tex,
                        Point {
                            x: tex_ox,
                            y: tex_oy,
                            z: tex_oz + z,
                        },
                        y,
                        copy_w,
                        storage_bpp,
                        &packed[..storage_row],
                        allowed.as_ref(),
                    )?;
                } else {
                    write_texture_row(
                        state,
                        host,
                        task_id,
                        tex,
                        Point {
                            x: tex_ox,
                            y: tex_oy,
                            z: tex_oz + z,
                        },
                        y,
                        plane_row as u64,
                        &plane,
                        allowed.as_ref(),
                    )?;
                }
            } else if repack {
                read_texture_storage_row(
                    state,
                    host,
                    task_id,
                    tex,
                    Point {
                        x: tex_ox,
                        y: tex_oy,
                        z: tex_oz + z,
                    },
                    y,
                    copy_w,
                    storage_bpp,
                    &mut packed,
                )?;
                if !pixel_format::extract_plane_row(
                    fmt,
                    aspect,
                    &packed[..storage_row],
                    copy_w,
                    &mut plane,
                ) {
                    return Err(br(BlitFailure::Unsupported, "bt_extract_plane_row"));
                }
                if gva_mem::write_task_gva_product_within(
                    state,
                    host,
                    task_id,
                    buf_gva,
                    &plane,
                    allowed.as_ref(),
                )
                .is_err()
                {
                    return Err(br(BlitFailure::GuestIo, "bt_write_guest_row_repack"));
                }
            } else {
                read_texture_row(
                    state,
                    host,
                    task_id,
                    tex,
                    Point {
                        x: tex_ox,
                        y: tex_oy,
                        z: tex_oz + z,
                    },
                    y,
                    plane_row as u64,
                    &mut plane,
                )?;
                if gva_mem::write_task_gva_product_within(
                    state,
                    host,
                    task_id,
                    buf_gva,
                    &plane,
                    allowed.as_ref(),
                )
                .is_err()
                {
                    return Err(br(BlitFailure::GuestIo, "bt_write_guest_row"));
                }
            }
        }
    }
    crate::runtime::drain::note_store_route_us(
        "blit_bt_rows_us",
        bt_rows_started.elapsed().as_micros() as u64,
    );
    Ok(())
}

fn exec_copy_buffer_to_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &BufferToTexture,
) -> BlitStatus {
    let src = match resolve_buffer(state, host, task_id, cmd.source_ref) {
        Ok(b) => b,
        Err(st) => return st,
    };
    let dst = match resolve_texture_backing(
        state,
        host,
        task_id,
        cmd.dest.texture_ref,
        cmd.dest.level,
        cmd.dest.slice,
    ) {
        Ok(t) => t,
        Err(st) => return st,
    };
    // A compressed texture reaches this rail only because
    // `resolve_texture_backing` now admits one for the texture-to-texture copy.
    // Everything below this line strides in texels, and a block is four of them
    // in each axis — so the honest answer is a named refusal rather than a
    // buffer sized a sixteenth of what it needs. See
    // `exec_copy_texture_to_texture`, which converts to block space instead.
    if dst.block().is_compressed() {
        return br(BlitFailure::Unsupported, "b2t_compressed");
    }
    let (aspect, copy_bpp) = match copy_aspect_for_options(dst.pixel_format(), cmd.options) {
        Ok(v) => v,
        Err(st) => return st,
    };
    let repack = pixel_format::blit_aspect_needs_repack(dst.pixel_format(), aspect);
    // Mapper-ref-texture is 2D only.
    if dst.is_mapper_ref_texture() && (cmd.dest.origin.z != 0 || cmd.size.depth > 1) {
        if cmd.size.depth == 0 {
            return BlitStatus::ZeroExtent;
        }
        if cmd.dest.origin.z != 0 || cmd.size.depth != 1 {
            return br(BlitFailure::Unsupported, "b2t_t11_z_or_depth");
        }
    }
    let ox = cmd.dest.origin.x;
    let oy = cmd.dest.origin.y;
    let oz = cmd.dest.origin.z;
    if ox > dst.width() as u64 || oy > dst.height() as u64 || oz > dst.depth() as u64 {
        return br(BlitFailure::Bounds, "b2t_origin_oob");
    }
    // Refused rather than clipped, and the origin check directly above is why
    // the two now agree: one wire record names a region, and both halves of it
    // are checked the same way.
    let (Some(copy_w), Some(copy_h)) = (
        copy_extent("b2t", "w", cmd.size.width, dst.width() as u64 - ox),
        copy_extent("b2t", "h", cmd.size.height, dst.height() as u64 - oy),
    ) else {
        return br(BlitFailure::Bounds, "b2t_extent_oob");
    };
    let copy_d = if cmd.size.depth == 0 {
        0
    } else {
        match copy_extent("b2t", "d", cmd.size.depth, dst.depth() as u64 - oz) {
            Some(d) => d,
            None => return br(BlitFailure::Bounds, "b2t_extent_oob"),
        }
    };
    if copy_w == 0 || copy_h == 0 || copy_d == 0 {
        return BlitStatus::ZeroExtent;
    }
    // Buffer-side plane bpp (aspect-aware).
    let row_bytes = match copy_w.checked_mul(copy_bpp as u64) {
        Some(v) => v,
        None => return br(BlitFailure::Capacity, "b2t_row_bytes_overflow"),
    };
    let src_bpr = if cmd.bytes_per_row != 0 {
        cmd.bytes_per_row
    } else {
        row_bytes
    };
    if src_bpr < row_bytes {
        return br(BlitFailure::Bounds, "b2t_src_bpr_lt_row");
    }
    let src_bpi = if cmd.bytes_per_image != 0 {
        cmd.bytes_per_image
    } else {
        match src_bpr.checked_mul(copy_h) {
            Some(v) => v,
            None => return br(BlitFailure::Capacity, "b2t_src_bpi_overflow"),
        }
    };
    // Combined DS + aspect: plane repack path (not raw GVA span).
    if repack {
        let src_gva = match src.gva.checked_add(cmd.source_offset) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "b2t_repack_gva_overflow"),
        };
        return match copy_buffer_texture_rows_aspect(
            state,
            host,
            task_id,
            src_gva,
            src_bpr,
            src_bpi,
            &dst,
            Point {
                x: ox,
                y: oy,
                z: oz,
            },
            copy_w as u32,
            copy_h,
            copy_d,
            copy_bpp,
            aspect,
            true,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    // Prefer direct GVA row-span when both sides linear (dst only texture here).
    if let TextureBacking::Linear(ref lt) = dst {
        let dst_off = match lt.texel_offset(ox, oy, oz) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "b2t_dst_texel_oob"),
        };
        let dst_bpi = match lt.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitFailure::Capacity, "b2t_dst_bpi_overflow"),
        };
        let last = match dst_off
            .checked_add((copy_d - 1).saturating_mul(dst_bpi))
            .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(lt.row_stride)))
            .and_then(|v| v.checked_add(row_bytes))
        {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "b2t_dst_span_overflow"),
        };
        if lt.alloc_size != 0 && last > lt.alloc_size {
            return br(BlitFailure::Bounds, "b2t_dst_alloc_oob");
        }
        let src_span = match cmd
            .source_offset
            .checked_add((copy_d - 1).saturating_mul(src_bpi))
            .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(src_bpr)))
            .and_then(|v| v.checked_add(row_bytes))
        {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "b2t_src_span_overflow"),
        };
        if src_span > src.size {
            return br(BlitFailure::Bounds, "b2t_src_span_oob");
        }
        let src_gva = match src.gva.checked_add(cmd.source_offset) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "b2t_src_gva_overflow"),
        };
        let dst_gva = match lt.base_gva.checked_add(dst_off) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "b2t_dst_gva_overflow"),
        };
        return match copy_row_region(
            host,
            state,
            task_id,
            src_gva,
            src_bpr,
            src_bpi,
            dst_gva,
            lt.row_stride,
            dst_bpi,
            row_bytes,
            copy_h,
            copy_d,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    // Mapper-ref-texture destination: row-stage from buffer GVA.
    let src_span = match cmd
        .source_offset
        .checked_add((copy_d - 1).saturating_mul(src_bpi))
        .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(src_bpr)))
        .and_then(|v| v.checked_add(row_bytes))
    {
        Some(v) => v,
        None => return br(BlitFailure::Bounds, "b2t_t11_src_span_overflow"),
    };
    if src_span > src.size {
        return br(BlitFailure::Bounds, "b2t_t11_src_span_oob");
    }
    // `None` for the mapper-ref-texture destination this arm is for, which the mapping rail
    // authorises instead; a linear destination reaching here is still bounded.
    let allowed = match texture_region_window(
        state,
        host,
        task_id,
        &dst,
        Point {
            x: ox,
            y: oy,
            z: oz,
        },
        copy_w as u32,
        copy_h,
        copy_d,
        copy_bpp,
    ) {
        Ok(v) => v,
        Err(st) => return st,
    };
    let mut row = vec![0u8; row_bytes as usize];
    for z in 0..copy_d {
        for y in 0..copy_h {
            let s = match src
                .gva
                .checked_add(cmd.source_offset)
                .and_then(|b| b.checked_add(z.saturating_mul(src_bpi)))
                .and_then(|b| b.checked_add(y.saturating_mul(src_bpr)))
            {
                Some(v) => v,
                None => return br(BlitFailure::Bounds, "b2t_t11_src_gva_overflow"),
            };
            if gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                s,
                &mut row,
                state.page_shift,
            )
            .is_err()
            {
                return br(BlitFailure::GuestIo, "b2t_t11_read_io");
            }
            if let Err(st) = write_texture_row(
                state,
                host,
                task_id,
                &dst,
                Point {
                    x: ox,
                    y: oy,
                    z: oz + z,
                },
                y,
                row_bytes,
                &row,
                allowed.as_ref(),
            ) {
                return st;
            }
        }
    }
    BlitStatus::Ok
}

fn exec_copy_texture_to_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &TextureToBuffer,
) -> BlitStatus {
    let src = match resolve_texture_backing(
        state,
        host,
        task_id,
        cmd.source.texture_ref,
        cmd.source.level,
        cmd.source.slice,
    ) {
        Ok(t) => t,
        Err(st) => return st,
    };
    // A compressed texture reaches this rail only because
    // `resolve_texture_backing` now admits one for the texture-to-texture copy.
    // Everything below this line strides in texels, and a block is four of them
    // in each axis — so the honest answer is a named refusal rather than a
    // buffer sized a sixteenth of what it needs. See
    // `exec_copy_texture_to_texture`, which converts to block space instead.
    if src.block().is_compressed() {
        return br(BlitFailure::Unsupported, "t2b_compressed");
    }
    let (aspect, copy_bpp) =
        match copy_aspect_for_options(src.pixel_format(), u32::from(cmd.options)) {
            Ok(v) => v,
            Err(st) => return st,
        };
    let repack = pixel_format::blit_aspect_needs_repack(src.pixel_format(), aspect);
    let dst = match resolve_buffer(state, host, task_id, cmd.dest_ref) {
        Ok(b) => b,
        Err(st) => return st,
    };
    if src.is_mapper_ref_texture() && (cmd.source.origin.z != 0 || cmd.size.depth > 1) {
        if cmd.size.depth == 0 {
            return BlitStatus::ZeroExtent;
        }
        if cmd.source.origin.z != 0 || cmd.size.depth != 1 {
            return br(BlitFailure::Unsupported, "t2b_t11_z_or_depth");
        }
    }
    let ox = cmd.source.origin.x;
    let oy = cmd.source.origin.y;
    let oz = cmd.source.origin.z;
    if ox > src.width() as u64 || oy > src.height() as u64 || oz > src.depth() as u64 {
        return br(BlitFailure::Bounds, "t2b_origin_oob");
    }
    // Refused rather than clipped, and the origin check directly above is why
    // the two now agree: one wire record names a region, and both halves of it
    // are checked the same way.
    let (Some(copy_w), Some(copy_h)) = (
        copy_extent("t2b", "w", cmd.size.width, src.width() as u64 - ox),
        copy_extent("t2b", "h", cmd.size.height, src.height() as u64 - oy),
    ) else {
        return br(BlitFailure::Bounds, "t2b_extent_oob");
    };
    let copy_d = if cmd.size.depth == 0 {
        0
    } else {
        match copy_extent("t2b", "d", cmd.size.depth, src.depth() as u64 - oz) {
            Some(d) => d,
            None => return br(BlitFailure::Bounds, "t2b_extent_oob"),
        }
    };
    if copy_w == 0 || copy_h == 0 || copy_d == 0 {
        return BlitStatus::ZeroExtent;
    }
    let row_bytes = match copy_w.checked_mul(copy_bpp as u64) {
        Some(v) => v,
        None => return br(BlitFailure::Capacity, "t2b_row_bytes_overflow"),
    };
    let dst_bpr = if cmd.bytes_per_row != 0 {
        cmd.bytes_per_row
    } else {
        row_bytes
    };
    if dst_bpr < row_bytes {
        return br(BlitFailure::Bounds, "t2b_dst_bpr_lt_row");
    }
    let dst_bpi = if cmd.bytes_per_image != 0 {
        cmd.bytes_per_image
    } else {
        match dst_bpr.checked_mul(copy_h) {
            Some(v) => v,
            None => return br(BlitFailure::Capacity, "t2b_dst_bpi_overflow"),
        }
    };
    if repack {
        let dst_gva = match dst.gva.checked_add(cmd.dest_offset) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "t2b_repack_gva_overflow"),
        };
        return match copy_buffer_texture_rows_aspect(
            state,
            host,
            task_id,
            dst_gva,
            dst_bpr,
            dst_bpi,
            &src,
            Point {
                x: ox,
                y: oy,
                z: oz,
            },
            copy_w as u32,
            copy_h,
            copy_d,
            copy_bpp,
            aspect,
            false,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    if let TextureBacking::Linear(ref lt) = src {
        let src_off = match lt.texel_offset(ox, oy, oz) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "t2b_src_texel_oob"),
        };
        let src_bpi = match lt.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitFailure::Capacity, "t2b_src_bpi_overflow"),
        };
        let dst_span = match cmd
            .dest_offset
            .checked_add((copy_d - 1).saturating_mul(dst_bpi))
            .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(dst_bpr)))
            .and_then(|v| v.checked_add(row_bytes))
        {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "t2b_dst_span_overflow"),
        };
        if dst_span > dst.size {
            return br(BlitFailure::Bounds, "t2b_dst_span_oob");
        }
        let src_gva = match lt.base_gva.checked_add(src_off) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "t2b_src_gva_overflow"),
        };
        let dst_gva = match dst.gva.checked_add(cmd.dest_offset) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "t2b_dst_gva_overflow"),
        };
        return match copy_row_region(
            host,
            state,
            task_id,
            src_gva,
            lt.row_stride,
            src_bpi,
            dst_gva,
            dst_bpr,
            dst_bpi,
            row_bytes,
            copy_h,
            copy_d,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    let dst_span = match cmd
        .dest_offset
        .checked_add((copy_d - 1).saturating_mul(dst_bpi))
        .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(dst_bpr)))
        .and_then(|v| v.checked_add(row_bytes))
    {
        Some(v) => v,
        None => return br(BlitFailure::Bounds, "t2b_stage_dst_span_overflow"),
    };
    if dst_span > dst.size {
        return br(BlitFailure::Bounds, "t2b_stage_dst_span_oob");
    }
    // `dst_span` is measured from `dst.gva`, so the span this loop writes starts
    // one `destination_offset` in.
    let dst_base = match dst.gva.checked_add(cmd.dest_offset) {
        Some(v) => v,
        None => return br(BlitFailure::Bounds, "t2b_stage_dst_gva_overflow"),
    };
    let allowed = dest_window(
        state,
        host,
        task_id,
        dst_base,
        dst_span.saturating_sub(cmd.dest_offset),
    );
    // `blit_rows_us` lives in `copy_row_region`, which only the linear-to-linear
    // fast path reaches. A texture-to-buffer copy stages every row through
    // `read_texture_row` instead, and for a mapper-ref-texture or ref-texture source that
    // re-vouches the mapping's guest page table per row.
    let stage_rows_started = std::time::Instant::now();
    let mut row = vec![0u8; row_bytes as usize];
    for z in 0..copy_d {
        for y in 0..copy_h {
            if let Err(st) = read_texture_row(
                state,
                host,
                task_id,
                &src,
                Point {
                    x: ox,
                    y: oy,
                    z: oz + z,
                },
                y,
                row_bytes,
                &mut row,
            ) {
                return st;
            }
            let d = match dst_base
                .checked_add(z.saturating_mul(dst_bpi))
                .and_then(|b| b.checked_add(y.saturating_mul(dst_bpr)))
            {
                Some(v) => v,
                None => return br(BlitFailure::Bounds, "t2b_stage_dst_gva_overflow"),
            };
            if gva_mem::write_task_gva_product_within(
                state,
                host,
                task_id,
                d,
                &row,
                allowed.as_ref(),
            )
            .is_err()
            {
                return br(BlitFailure::GuestIo, "t2b_stage_write_io");
            }
        }
    }
    crate::runtime::drain::note_store_route_us(
        "blit_t2b_stage_us",
        stage_rows_started.elapsed().as_micros() as u64,
    );
    crate::runtime::drain::note_store_route_n("blit_t2b_stage_rows", copy_h.saturating_mul(copy_d));
    BlitStatus::Ok
}

fn exec_copy_texture_to_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &TextureRegion,
) -> BlitStatus {
    // `walk_blit_us` says this call costs ~1.1 ms and `blit_t2t_bytes` says it
    // moves ~800 of them, so the cost is not in the copy and a single total
    // cannot say where it is instead. The three phases below are the whole body:
    // resolving both endpoints, arming the destination's page window, and the
    // row loop. Whichever of them holds the millisecond is the one to repair.
    let phase_started = std::time::Instant::now();
    let src = match resolve_texture_backing(
        state,
        host,
        task_id,
        cmd.source.texture_ref,
        cmd.source.level,
        cmd.source.slice,
    ) {
        Ok(t) => t,
        Err(st) => return st,
    };
    let dst = match resolve_texture_backing(
        state,
        host,
        task_id,
        cmd.dest.texture_ref,
        cmd.dest.level,
        cmd.dest.slice,
    ) {
        Ok(t) => t,
        Err(st) => return st,
    };
    // Options apply to both ends; plane bpp must agree under the selected aspect.
    let (aspect, src_bpp) = match copy_aspect_for_options(src.pixel_format(), cmd.options) {
        Ok(v) => v,
        Err(st) => return st,
    };
    let (_, dst_bpp) = match copy_aspect_for_options(dst.pixel_format(), cmd.options) {
        Ok(v) => v,
        Err(st) => return st,
    };
    if src_bpp != dst_bpp {
        return br(BlitFailure::Unsupported, "t2t_bpp_mismatch");
    }
    if src.pixel_format() != 0
        && dst.pixel_format() != 0
        && src.pixel_format() != dst.pixel_format()
    {
        return br(BlitFailure::Unsupported, "t2t_format_mismatch");
    }
    let copy_bpp = src_bpp;
    let repack_src = pixel_format::blit_aspect_needs_repack(src.pixel_format(), aspect);
    let repack_dst = pixel_format::blit_aspect_needs_repack(dst.pixel_format(), aspect);
    let any_t11 = src.is_mapper_ref_texture() || dst.is_mapper_ref_texture();
    if any_t11 && (cmd.source.origin.z != 0 || cmd.dest.origin.z != 0) {
        return br(BlitFailure::Unsupported, "t2t_t11_z");
    }
    let sox = cmd.source.origin.x;
    let soy = cmd.source.origin.y;
    let soz = cmd.source.origin.z;
    let dox = cmd.dest.origin.x;
    let doy = cmd.dest.origin.y;
    let doz = cmd.dest.origin.z;
    if sox > src.width() as u64
        || soy > src.height() as u64
        || soz > src.depth() as u64
        || dox > dst.width() as u64
        || doy > dst.height() as u64
        || doz > dst.depth() as u64
    {
        return br(BlitFailure::Bounds, "t2t_origin_oob");
    }
    // One region, checked against both textures. The source and destination
    // halves are separate `kind`s so a refusal says which end was too small,
    // and the destination half used to have no instrument at all — it did not
    // even come through the extent helper, which made it the quieter of two
    // quiet paths.
    let (Some(copy_w), Some(_)) = (
        copy_extent("t2t_src", "w", cmd.size.width, src.width() as u64 - sox),
        copy_extent("t2t_dst", "w", cmd.size.width, dst.width() as u64 - dox),
    ) else {
        return br(BlitFailure::Bounds, "t2t_extent_oob");
    };
    let (Some(copy_h), Some(_)) = (
        copy_extent("t2t_src", "h", cmd.size.height, src.height() as u64 - soy),
        copy_extent("t2t_dst", "h", cmd.size.height, dst.height() as u64 - doy),
    ) else {
        return br(BlitFailure::Bounds, "t2t_extent_oob");
    };
    let copy_d = if cmd.size.depth == 0 {
        0
    } else {
        let (Some(d), Some(_)) = (
            copy_extent("t2t_src", "d", cmd.size.depth, src.depth() as u64 - soz),
            copy_extent("t2t_dst", "d", cmd.size.depth, dst.depth() as u64 - doz),
        ) else {
            return br(BlitFailure::Bounds, "t2t_extent_oob");
        };
        d
    };
    if any_t11 && copy_d > 1 {
        return br(BlitFailure::Unsupported, "t2t_t11_volume");
    }
    if copy_w == 0 || copy_h == 0 || copy_d == 0 {
        return BlitStatus::ZeroExtent;
    }
    note_t2t_shape(&src, &dst, copy_w, copy_h, copy_d, copy_bpp);
    crate::runtime::drain::note_store_route_us(
        "blit_t2t_resolve_us",
        phase_started.elapsed().as_micros() as u64,
    );
    // From here down the coordinates are in units of `copy_bpp`, and for a
    // block-compressed format that unit is a 4x4 **block**.
    //
    // A compressed copy is an uncompressed copy of the block image — same row
    // stride, same staging, same page window — so the conversion happens once,
    // here, and every per-unit helper below runs unchanged: `texel_offset`
    // multiplies x by `bpp` and y by `row_stride`, which in block space is
    // exactly a block column and a block row. Threading a grid through each
    // helper instead would put the same division in six places.
    //
    // Above this line everything is texels, which is what the bounds checks and
    // `note_t2t_shape` want; below it nothing is. That is the whole reason the
    // conversion is a single shadowing binding rather than a flag.
    let block = src.block();
    let (sox, soy, dox, doy, copy_w, copy_h) = if !block.is_compressed() {
        (sox, soy, dox, doy, copy_w, copy_h)
    } else {
        // Each refusal below is a copy this rail could describe wrongly rather
        // than one Metal forbids, so each is named and none is a clamp.
        if aspect != BlitAspect::Full || repack_src || repack_dst {
            // There is no depth or stencil plane inside a colour block, and a
            // repack pass rewrites texels.
            return br(BlitFailure::Unsupported, "t2t_compressed_aspect");
        }
        if any_t11 {
            return br(BlitFailure::Unsupported, "t2t_compressed_t11");
        }
        if dst.block() != block {
            // One allocation reinterpreted at two grids is not a copy; the
            // format-mismatch check above lets a format-0 side through, and this
            // is where that pairing stops.
            return br(BlitFailure::Unsupported, "t2t_compressed_grid_mismatch");
        }
        if copy_d > 1 || soz != 0 || doz != 0 {
            // `bytes_per_image` strides a depth plane by the *texel* height, and
            // this conversion does not reach it. A 2D copy never asks.
            return br(BlitFailure::Unsupported, "t2t_compressed_volume");
        }
        let (bw, bh) = (u64::from(block.width), u64::from(block.height));
        if !sox.is_multiple_of(bw)
            || !dox.is_multiple_of(bw)
            || !soy.is_multiple_of(bh)
            || !doy.is_multiple_of(bh)
        {
            // A block is the smallest unit this copy can move, so an origin
            // inside one names bytes it cannot address.
            return br(BlitFailure::Unsupported, "t2t_compressed_origin_unaligned");
        }
        // An extent may end mid-block only where it reaches the level edge on
        // *both* ends — the one case a partial trailing block is the whole
        // remainder of the image rather than a slice of a block.
        let w_edge = sox + copy_w == src.width() as u64 && dox + copy_w == dst.width() as u64;
        let h_edge = soy + copy_h == src.height() as u64 && doy + copy_h == dst.height() as u64;
        if (!copy_w.is_multiple_of(bw) && !w_edge) || (!copy_h.is_multiple_of(bh) && !h_edge) {
            return br(BlitFailure::Unsupported, "t2t_compressed_extent_unaligned");
        }
        crate::runtime::drain::note_store_route("blit_t2t_compressed");
        (
            sox / bw,
            soy / bh,
            dox / bw,
            doy / bh,
            copy_w.div_ceil(bw),
            copy_h.div_ceil(bh),
        )
    };
    let row_bytes = match copy_w.checked_mul(copy_bpp as u64) {
        Some(v) => v,
        None => return br(BlitFailure::Capacity, "t2t_row_bytes_overflow"),
    };
    // Combined DS + aspect: extract plane from src, insert into dst (RMW).
    if repack_src || repack_dst {
        // The repack strides are the two textures' *storage* widths, which for a
        // combined depth/stencil format is wider than the aspect being copied.
        // Either side may still be format 0 here: the mismatch check above lets a
        // zero through, so a format-less texture can pair with a combined one and
        // only the combined side sets its `repack_*`. `bytes_per_pixel` cannot
        // answer for a zero, and the aspect's own `copy_bpp` is then a guess at a
        // stride this device is about to read guest bytes with. It stays the
        // guess — refusing would drop a copy that is correct whenever the two
        // widths agree — but it is a guess about guest data, so it says so.
        let src_storage = texture_storage_bpp(src.pixel_format()).unwrap_or_else(|_| {
            note_repack_storage_assumed(task_id, "src", src.pixel_format(), copy_bpp);
            copy_bpp
        });
        let dst_storage = texture_storage_bpp(dst.pixel_format()).unwrap_or_else(|_| {
            note_repack_storage_assumed(task_id, "dst", dst.pixel_format(), copy_bpp);
            copy_bpp
        });
        let plane_row = row_bytes as usize;
        let mut plane = vec![0u8; plane_row];
        let mut src_packed = vec![0u8; (copy_w as usize).saturating_mul(src_storage as usize)];
        let mut dst_packed = vec![0u8; (copy_w as usize).saturating_mul(dst_storage as usize)];
        let allowed = match texture_region_window(
            state,
            host,
            task_id,
            &dst,
            Point {
                x: dox,
                y: doy,
                z: doz,
            },
            copy_w as u32,
            copy_h,
            copy_d,
            // This loop has two write arms: the repack one stores a packed
            // `dst_storage` row, the other a `copy_bpp` plane row. The window
            // has to cover whichever runs, so it is measured on the wider.
            dst_storage.max(copy_bpp),
        ) {
            Ok(v) => v,
            Err(st) => return st,
        };
        for z in 0..copy_d {
            for y in 0..copy_h {
                if repack_src {
                    if let Err(st) = read_texture_storage_row(
                        state,
                        host,
                        task_id,
                        &src,
                        Point {
                            x: sox,
                            y: soy,
                            z: soz + z,
                        },
                        y,
                        copy_w as u32,
                        src_storage,
                        &mut src_packed,
                    ) {
                        return st;
                    }
                    if !pixel_format::extract_plane_row(
                        src.pixel_format(),
                        aspect,
                        &src_packed,
                        copy_w as u32,
                        &mut plane,
                    ) {
                        return br(BlitFailure::Unsupported, "t2t_extract_plane");
                    }
                } else if let Err(st) = read_texture_row(
                    state,
                    host,
                    task_id,
                    &src,
                    Point {
                        x: sox,
                        y: soy,
                        z: soz + z,
                    },
                    y,
                    row_bytes,
                    &mut plane,
                ) {
                    return st;
                }
                if repack_dst {
                    if let Err(st) = read_texture_storage_row(
                        state,
                        host,
                        task_id,
                        &dst,
                        Point {
                            x: dox,
                            y: doy,
                            z: doz + z,
                        },
                        y,
                        copy_w as u32,
                        dst_storage,
                        &mut dst_packed,
                    ) {
                        return st;
                    }
                    if !pixel_format::insert_plane_row(
                        dst.pixel_format(),
                        aspect,
                        &plane,
                        copy_w as u32,
                        &mut dst_packed,
                    ) {
                        return br(BlitFailure::Unsupported, "t2t_insert_plane");
                    }
                    if let Err(st) = write_texture_storage_row(
                        state,
                        host,
                        task_id,
                        &dst,
                        Point {
                            x: dox,
                            y: doy,
                            z: doz + z,
                        },
                        y,
                        copy_w as u32,
                        dst_storage,
                        &dst_packed,
                        allowed.as_ref(),
                    ) {
                        return st;
                    }
                } else if let Err(st) = write_texture_row(
                    state,
                    host,
                    task_id,
                    &dst,
                    Point {
                        x: dox,
                        y: doy,
                        z: doz + z,
                    },
                    y,
                    row_bytes,
                    &plane,
                    allowed.as_ref(),
                ) {
                    return st;
                }
            }
        }
        return BlitStatus::Ok;
    }
    // Fast path: both linear → existing GVA span copy.
    if let (TextureBacking::Linear(ref sl), TextureBacking::Linear(ref dl)) = (&src, &dst) {
        let src_off = match sl.texel_offset(sox, soy, soz) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "t2t_src_texel_oob"),
        };
        let dst_off = match dl.texel_offset(dox, doy, doz) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "t2t_dst_texel_oob"),
        };
        let src_bpi = match sl.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitFailure::Capacity, "t2t_src_bpi_overflow"),
        };
        let dst_bpi = match dl.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitFailure::Capacity, "t2t_dst_bpi_overflow"),
        };
        let src_gva = match sl.base_gva.checked_add(src_off) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "t2t_src_gva_overflow"),
        };
        let dst_gva = match dl.base_gva.checked_add(dst_off) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "t2t_dst_gva_overflow"),
        };
        // Exact identity self-copy is a no-op: source and destination name the
        // same guest bytes with the same layout, so every row reads and writes
        // the same address — the destination already holds the source content.
        // Observed live (Ventura x86, media apps): the guest issues
        // copyFromTexture:X toTexture:X sourceOrigin==destinationOrigin on small
        // window textures (src_ref==dst_ref, src_off==dst_off). Copying bytes
        // onto themselves changes nothing, so succeed without work rather than
        // rejecting it as Overlap (which returned a spurious error to the guest
        // blit encoder and dropped a copy the guest treats as complete). A
        // genuinely-shifted overlap falls through to source-before-destination
        // staging below.
        if src_gva == dst_gva && sl.row_stride == dl.row_stride && src_bpi == dst_bpi {
            return BlitStatus::Ok;
        }
        // Same allocation (self-copy or aliased view) but a different region:
        // direct row-by-row execution is safe only when the source and
        // destination texel rectangles do not intersect. Two axis-aligned
        // rectangles overlap iff they overlap on every axis. If the layouts
        // differ, their texel grids are incomparable, so the byte spans give
        // the conservative answer.
        if sl.base_gva == dl.base_gva {
            let same_layout = sl.row_stride == dl.row_stride && src_bpi == dst_bpi;
            let overlaps = if same_layout {
                let x = sox < dox + copy_w && dox < sox + copy_w;
                let y = soy < doy + copy_h && doy < soy + copy_h;
                let z = soz < doz + copy_d && doz < soz + copy_d;
                x && y && z
            } else {
                let s_end =
                    src_off.saturating_add(row_bytes.saturating_mul(copy_h).saturating_mul(copy_d));
                let d_end =
                    dst_off.saturating_add(row_bytes.saturating_mul(copy_h).saturating_mul(copy_d));
                ranges_overlap(
                    src_off,
                    s_end.saturating_sub(src_off),
                    dst_off,
                    d_end.saturating_sub(dst_off),
                )
            };
            if overlaps {
                // The serialized copy record carries the complete source and
                // destination regions and does not define an overlap refusal.
                // Snapshot every source plane before writing any destination
                // plane: a row-at-a-time loop can overwrite bytes a later row
                // or depth plane still has to read.
                let plane_bytes = match row_bytes.checked_mul(copy_h) {
                    Some(v) => v,
                    None => return br(BlitFailure::Capacity, "t2t_overlap_plane_overflow"),
                };
                let total_bytes = match plane_bytes.checked_mul(copy_d) {
                    Some(v) => v,
                    None => return br(BlitFailure::Capacity, "t2t_overlap_total_overflow"),
                };
                let Some(total_len) = host_alloc_len(total_bytes) else {
                    return br(BlitFailure::Capacity, "t2t_overlap_alloc");
                };
                let allowed = match texture_region_window(
                    state,
                    host,
                    task_id,
                    &dst,
                    Point {
                        x: dox,
                        y: doy,
                        z: doz,
                    },
                    copy_w as u32,
                    copy_h,
                    copy_d,
                    copy_bpp,
                ) {
                    Ok(v) => v,
                    Err(st) => return st,
                };
                let mut staged = vec![0u8; total_len];
                for z in 0..copy_d {
                    let start = (z * plane_bytes) as usize;
                    let end = start + plane_bytes as usize;
                    if let Err(st) = read_texture_rect(
                        state,
                        host,
                        task_id,
                        &src,
                        Point {
                            x: sox,
                            y: soy,
                            z: soz + z,
                        },
                        row_bytes,
                        copy_h,
                        &mut staged[start..end],
                    ) {
                        return st;
                    }
                }
                for z in 0..copy_d {
                    let start = (z * plane_bytes) as usize;
                    let end = start + plane_bytes as usize;
                    if let Err(st) = write_texture_rect(
                        state,
                        host,
                        task_id,
                        &dst,
                        Point {
                            x: dox,
                            y: doy,
                            z: doz + z,
                        },
                        row_bytes,
                        copy_h,
                        &staged[start..end],
                        allowed.as_ref(),
                    ) {
                        return st;
                    }
                }
                crate::runtime::drain::note_store_route("blit_t2t_overlap_staged");
                crate::runtime::drain::note_store_route_n(
                    "blit_t2t_overlap_staged_bytes",
                    total_bytes,
                );
                return BlitStatus::Ok;
            }
        }
        return match copy_row_region(
            host,
            state,
            task_id,
            src_gva,
            sl.row_stride,
            src_bpi,
            dst_gva,
            dl.row_stride,
            dst_bpi,
            row_bytes,
            copy_h,
            copy_d,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    // Mixed or mapper-ref-texture↔mapper-ref-texture: stage rows.
    let allowed = match texture_region_window(
        state,
        host,
        task_id,
        &dst,
        Point {
            x: dox,
            y: doy,
            z: doz,
        },
        copy_w as u32,
        copy_h,
        copy_d,
        copy_bpp,
    ) {
        Ok(v) => v,
        Err(st) => return st,
    };
    // The last untimed loop on the rail: a texture-to-texture copy with a
    // mapper-ref-texture or ref-texture end on either side stages through here rather than
    // through `copy_row_region`, so `blit_rows_us` reports nothing for it.
    //
    // A plane at a time, not a row at a time: this is the same staging shape the
    // slice/level form carried, and there a driven Maps leg charged the row loop
    // 30.15 s of a 30.28 s rail. See [`read_texture_rect`] for what a per-row
    // call into the mapping rail re-pays.
    // Whether a GPU-side copy serves this pair, and if not, which term stops it.
    // `engine::copy_target_to_guest_pages` takes no source rectangle: it copies
    // level 0 of the resident whole, at origin zero, into a destination whose
    // geometry is the resident's own. So a mapper-ref-texture source going to a linear
    // destination is reachable only when both ends are the whole plane at the
    // origin, and the three counters partition the population so a reading says
    // how much of it that is. See [`try_copy_t11_plane_to_linear_on_gpu`] for
    // what the arm below is instead of, which is the settle the staging loop
    // pays to make the source's guest bytes readable.
    if src.is_mapper_ref_texture() && !dst.is_mapper_ref_texture() {
        let whole_src =
            sox == 0 && soy == 0 && copy_w == src.width() as u64 && copy_h == src.height() as u64;
        let whole_dst =
            dox == 0 && doy == 0 && copy_w == dst.width() as u64 && copy_h == dst.height() as u64;
        crate::runtime::drain::note_store_route(match (whole_src, whole_dst) {
            (true, true) => "blit_t2t_t11_whole_plane",
            (true, false) => "blit_t2t_t11_dst_partial",
            (false, _) => "blit_t2t_t11_src_partial",
        });
        if whole_src && whole_dst {
            if let (TextureBacking::MapperRefTexture(s), TextureBacking::Linear(d)) = (&src, &dst) {
                if let Some(status) = crate::backend::selected()
                    .try_copy_t11_plane_to_linear_on_gpu(
                        state,
                        host,
                        task_id,
                        cmd.dest.texture_ref,
                        s,
                        d,
                    )
                {
                    return status;
                }
            }
        }
    }
    let t2t_stage_started = std::time::Instant::now();
    let mut staged = vec![0u8; row_bytes.saturating_mul(copy_h) as usize];
    for z in 0..copy_d {
        if let Err(st) = read_texture_rect(
            state,
            host,
            task_id,
            &src,
            Point {
                x: sox,
                y: soy,
                z: soz + z,
            },
            row_bytes,
            copy_h,
            &mut staged,
        ) {
            return st;
        }
        if let Err(st) = write_texture_rect(
            state,
            host,
            task_id,
            &dst,
            Point {
                x: dox,
                y: doy,
                z: doz + z,
            },
            row_bytes,
            copy_h,
            &staged,
            allowed.as_ref(),
        ) {
            return st;
        }
    }
    crate::runtime::drain::note_store_route_us(
        "blit_t2t_stage_us",
        t2t_stage_started.elapsed().as_micros() as u64,
    );
    crate::runtime::drain::note_store_route_n("blit_t2t_stage_rows", copy_h.saturating_mul(copy_d));
    BlitStatus::Ok
}

/// `0x13e copyFromTexture:…sliceCount:levelCount:` — whole-surface multi-slice/level.
///
/// For each level offset in `0..level_count`:
/// - **depth == 1:** copies full `width×height` across `slice_count` consecutive
///   array slices (`origin (0,0,0)`, size `w×h×1`).
/// - **depth > 1 (3D volume):** Metal requires `sliceCount == 1` and source/
///   destination slices 0; copies full `width×height×depth` of that mip with
///   depth planes strided by `bytes_per_image`. Linear normal-texture only.
///
/// One resolved blit endpoint, reduced to what the GPU whole-plane arm reads.
///
/// A [`TextureBacking`] says where a texture's *guest bytes* are, which is the
/// question every other consumer in this module asks. This is the other one:
/// which plane of which surface a GPU-side copy would land in.
#[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuPlane {
    width: u32,
    height: u32,
    /// Byte offset of this texture's plane within its mapping's allocation.
    surface_offset: u64,
    row_stride: u32,
    pixel_format: u16,
}

/// The plane `mapping_write::vulkan::write_bgra8_from_resident_gpu` will address, from
/// the destination mapping's own declaration rather than from the texture
/// descriptor the blit named.
///
/// The two are independent derivations of one plane and the whole safety of the
/// GPU arm is that they agree. See [`mapping_write::resident_gpu_plane`].
#[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuMappingWindow {
    surface_offset: u64,
    row_stride: u32,
    pixel_format: u16,
}

/// The source's real content, as the engine holds it behind an armed
/// [`crate::runtime::writeback_debt::GvaWritebackDebt`].
#[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuResidentSource {
    width: u32,
    height: u32,
    pixel_format: u16,
}

/// Why one whole-surface texture-to-texture copy is not the GPU arm's.
///
/// Every variant is a **fall-through and not a loss**: the host path below runs
/// unchanged and lands the same pixels. They are counters rather than fail-log
/// records for that reason, and they partition the whole-surface population with
/// `sl_gpu_landed`, so a census that does not add up is the bug.
///
/// There is deliberately no partial-rect variant. `0x13e` is the *whole-surface*
/// form: its origins are (0,0,0) and its extent is the endpoints' full
/// width/height by construction of the opcode, so a rect check here would be an
/// arm no guest command can take.
#[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuPlaneRefusal {
    /// More than one level or slice: the GPU arm copies one plane.
    MultiLevel,
    /// Source and destination are the same reference, so resolving the
    /// destination would pay away the very debt holding the source's content.
    SelfCopy,
    /// The source's bytes are its guest pages' bytes already — nothing to copy
    /// from a resident, and the host path is the cheap one.
    SrcNotResident,
    /// The destination is a linear guest allocation, which has no mapping for a
    /// GPU-side copy to name.
    DstNotMapperRefTexture,
    /// The destination mapping declines a resident-to-guest-pages copy at this
    /// extent, so there is no window to write.
    DstWindowUnresolved,
    /// The two derivations of the destination plane disagree. A ref-texture view's
    /// wire plane index lands here: it can name a plane the mapping's own
    /// geometry scan does not resolve to, and the GPU rail takes no index.
    PlaneOffset,
    /// The resident is not the destination's size, so a full-plane copy would
    /// be a resize.
    GeometryDiffers,
    /// A copy converts nothing, so all three of source, destination and mapping
    /// must already agree on the texel.
    FormatDiffers,
}

impl GpuPlaneRefusal {
    #[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
    fn route(self) -> &'static str {
        match self {
            Self::MultiLevel => "sl_gpu_multi_level",
            Self::SelfCopy => "sl_gpu_self_copy",
            Self::SrcNotResident => "sl_gpu_src_not_resident",
            Self::DstNotMapperRefTexture => "sl_gpu_dst_not_t11",
            Self::DstWindowUnresolved => "sl_gpu_dst_window",
            Self::PlaneOffset => "sl_gpu_plane_offset",
            Self::GeometryDiffers => "sl_gpu_geometry_differs",
            Self::FormatDiffers => "sl_gpu_format_differs",
        }
    }
}

/// Everything the GPU arm can decide before resolving anything.
///
/// Split from [`gpu_whole_plane_destination`] because resolving the destination
/// is the expensive half and it is also the half with the side effect: it pays
/// the destination's own debt. This decides whether that is worth doing.
#[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
fn gpu_whole_plane_admissible(
    level_count: u16,
    slice_count: u16,
    source_ref: u32,
    destination_ref: u32,
    source_is_resident: bool,
) -> Result<(), GpuPlaneRefusal> {
    if level_count != 1 || slice_count != 1 {
        return Err(GpuPlaneRefusal::MultiLevel);
    }
    if source_ref == destination_ref {
        return Err(GpuPlaneRefusal::SelfCopy);
    }
    if !source_is_resident {
        return Err(GpuPlaneRefusal::SrcNotResident);
    }
    Ok(())
}

/// Whether the resolved destination is the plane the GPU rail will actually
/// write, and whether the resident is a whole-plane copy into it.
///
/// `dst` is `None` for a linear endpoint and `window` is `None` when the mapping
/// declines the extent, so the two `Option`s are the caller's two resolution
/// steps and not defensive wrapping.
#[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
fn gpu_whole_plane_destination(
    dst: Option<GpuPlane>,
    window: Option<GpuMappingWindow>,
    src: GpuResidentSource,
) -> Result<(), GpuPlaneRefusal> {
    let Some(dst) = dst else {
        return Err(GpuPlaneRefusal::DstNotMapperRefTexture);
    };
    let Some(window) = window else {
        return Err(GpuPlaneRefusal::DstWindowUnresolved);
    };
    // The plane the guest's descriptor named against the plane the rail will
    // write. `mapping_write/mod.rs`'s own record of a bound error landing in the
    // next plane's pixels of a multi-plane IOSurface is what this is here for:
    // that failure is silent at every layer, so it has to be refused before the
    // copy rather than detected after it.
    if window.surface_offset != dst.surface_offset || window.row_stride != dst.row_stride {
        return Err(GpuPlaneRefusal::PlaneOffset);
    }
    if src.width != dst.width || src.height != dst.height {
        return Err(GpuPlaneRefusal::GeometryDiffers);
    }
    // All three, not two. The engine refuses a resident whose texel is not the
    // destination's (`copy_target_to_guest_pages`), and the mapping's declared
    // format is what the guest will read these bytes back as — a chain that
    // agrees pairwise but not as a whole would land a converted-looking frame
    // with nothing having converted anything.
    //
    // **Stored texels, not declared formats.** A copy converts nothing, so what
    // has to agree is what the bytes *are*, and the transfer function is not part
    // of that: it says how a sampler reads a texel, not how one is stored.
    // `store_texel_order` is this crate's single answer to that question and its
    // own doc states the fold ("only the storage matters to a copy"), which is
    // also what `translate::pixel::stored_bytes_agree` encodes one layer down for
    // the Vulkan spelling of the same comparison.
    //
    // Equality here is not a stricter version of that rule, it is a different and
    // wrong one, and it cost this arm every record it was written for: a guest
    // render target declared `BGRA8Unorm_sRGB` meets an IOSurface mapping that
    // declares plain `BGRA8Unorm` for the same four stored bytes, so the triple
    // reads 81/81/80 and equality calls that a disagreement forever. A `None`
    // still refuses — a format with no byte-copy layout is one where the copy
    // would have to convert, which this arm does not do.
    let texel = crate::protocol::pixel_format::store_texel_order(dst.pixel_format);
    if texel.is_none()
        || crate::protocol::pixel_format::store_texel_order(src.pixel_format) != texel
        || crate::protocol::pixel_format::store_texel_order(window.pixel_format) != texel
    {
        return Err(GpuPlaneRefusal::FormatDiffers);
    }
    Ok(())
}

/// Zero `slice_count` or `level_count` is a Metal no-op ([`BlitStatus::ZeroExtent`]).
fn exec_copy_texture_to_texture_slice_level<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &TextureSlices,
) -> BlitStatus {
    if cmd.slice_count == 0 || cmd.level_count == 0 {
        return BlitStatus::ZeroExtent;
    }
    if cmd.source_ref == 0 || cmd.dest_ref == 0 {
        return br(BlitFailure::MissingResource, "sl_missing_ref");
    }
    // Before the loop, because the loop's first act is to resolve the source and
    // resolving is what pays its debt. See
    // [`crate::backend::Backend::try_copy_whole_plane_on_gpu`].
    if let Some(status) =
        crate::backend::selected().try_copy_whole_plane_on_gpu(state, host, task_id, cmd)
    {
        return status;
    }
    for level_i in 0..cmd.level_count {
        let src_level = match cmd.source_level.checked_add(level_i) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "sl_src_level_overflow"),
        };
        let dst_level = match cmd.dest_level.checked_add(level_i) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "sl_dst_level_overflow"),
        };
        let last_slice_delta = match cmd.slice_count.checked_sub(1) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "sl_slice_count_underflow"),
        };
        let src_last_slice = match cmd.source_slice.checked_add(last_slice_delta) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "sl_src_slice_overflow"),
        };
        let dst_last_slice = match cmd.dest_slice.checked_add(last_slice_delta) {
            Some(v) => v,
            None => return br(BlitFailure::Bounds, "sl_dst_slice_overflow"),
        };

        // `blit_kind_t2t_sl_us` charges this function 28.8 s of a 29.1 s rail
        // while `blit_rows_us` — its linear arm's whole copy — reads 0.275 s.
        // Between those two numbers sit the resolves below, and this form runs
        // them once per level and again per slice, so their count is the
        // multiplier nobody has measured. `sl_levels_n` is the denominator that
        // says whether a level loop of two or of twelve is being paid for.
        crate::runtime::drain::note_store_route("sl_levels_n");
        let sl_resolve_started = std::time::Instant::now();
        // Resolve the starting slice at this level for geometry / format.
        // Volume (depth>1) forms use slice 0 only; non-zero source_slice on a
        // depth-1 packing fails at resolve (Bounds). For volumes we require
        // sliceCount==1 and slices 0 before any multi-slice last-index walk.
        let src0 = match resolve_texture_backing(
            state,
            host,
            task_id,
            cmd.source_ref,
            src_level,
            cmd.source_slice,
        ) {
            Ok(t) => t,
            Err(st) => return st,
        };
        let dst0 = match resolve_texture_backing(
            state,
            host,
            task_id,
            cmd.dest_ref,
            dst_level,
            cmd.dest_slice,
        ) {
            Ok(t) => t,
            Err(st) => return st,
        };
        if src0.bpp() != dst0.bpp() {
            return br(BlitFailure::Unsupported, "sl_bpp_mismatch");
        }
        if src0.pixel_format() != 0
            && dst0.pixel_format() != 0
            && src0.pixel_format() != dst0.pixel_format()
        {
            return br(BlitFailure::Unsupported, "sl_format_mismatch");
        }
        if src0.width() != dst0.width() || src0.height() != dst0.height() {
            return br(BlitFailure::Bounds, "sl_dim_mismatch");
        }
        if src0.depth() != dst0.depth() {
            return br(BlitFailure::Bounds, "sl_depth_mismatch");
        }
        crate::runtime::drain::note_store_route_us(
            "sl_resolve_us",
            sl_resolve_started.elapsed().as_micros() as u64,
        );
        let w = src0.width();
        let h = src0.height();
        let d = src0.depth();
        if w == 0 || h == 0 || d == 0 {
            return br(BlitFailure::Bounds, "sl_zero_geom");
        }
        let is_volume = d > 1;
        // Metal 3D whole-surface: sliceCount must be 1, slices 0; full depth of mip.
        if is_volume {
            if cmd.slice_count != 1 || cmd.source_slice != 0 || cmd.dest_slice != 0 {
                return br(BlitFailure::Unsupported, "sl_volume_slice_constraint");
            }
            // Mapper-ref-texture is 2D (depth 1); volume endpoints are linear only.
            if src0.is_mapper_ref_texture() || dst0.is_mapper_ref_texture() {
                return br(BlitFailure::Unsupported, "sl_volume_t11");
            }
        } else if cmd.slice_count > 1 {
            // Array form: last slice must resolve (view / packing bounds).
            if let Err(st) = resolve_texture_backing(
                state,
                host,
                task_id,
                cmd.source_ref,
                src_level,
                src_last_slice,
            ) {
                return st;
            }
            if let Err(st) = resolve_texture_backing(
                state,
                host,
                task_id,
                cmd.dest_ref,
                dst_level,
                dst_last_slice,
            ) {
                return st;
            }
        }
        let bpp = src0.bpp();
        // Whole levels, so a compressed one needs no origin or partial-block
        // reasoning — only its row width and row *count* in blocks. The grids
        // must agree for the same reason the `bpp`s must.
        let block = src0.block();
        if dst0.block() != block {
            return br(BlitFailure::Unsupported, "sl_compressed_grid_mismatch");
        }
        let rows = u64::from(block.block_rows(h));
        let row_bytes = match u64::from(block.blocks_across(w)).checked_mul(bpp as u64) {
            Some(v) => v,
            None => return br(BlitFailure::Capacity, "sl_row_bytes_overflow"),
        };

        // Linear: multi-slice (depth-1) or full volume (depth>1).
        if let (TextureBacking::Linear(ref sl), TextureBacking::Linear(ref dl)) = (&src0, &dst0) {
            if !is_volume && cmd.slice_count > 1 && (sl.slice_stride == 0 || dl.slice_stride == 0) {
                return br(BlitFailure::Unsupported, "sl_slice_stride_zero");
            }
            let src_off = match sl.texel_offset(0, 0, 0) {
                Some(v) => v,
                None => return br(BlitFailure::Bounds, "sl_src_texel_oob"),
            };
            let dst_off = match dl.texel_offset(0, 0, 0) {
                Some(v) => v,
                None => return br(BlitFailure::Bounds, "sl_dst_texel_oob"),
            };
            let src_gva = match sl.base_gva.checked_add(src_off) {
                Some(v) => v,
                None => return br(BlitFailure::Bounds, "sl_src_gva_overflow"),
            };
            let dst_gva = match dl.base_gva.checked_add(dst_off) {
                Some(v) => v,
                None => return br(BlitFailure::Bounds, "sl_dst_gva_overflow"),
            };
            // Volume: image_count = depth, stride = bytes_per_image (z planes).
            // Array: image_count = slice_count, stride = slice_stride when multi.
            let (src_img_stride, dst_img_stride, image_count) = if is_volume {
                let src_bpi = match sl.bytes_per_image() {
                    Some(v) if v > 0 => v,
                    _ => return br(BlitFailure::Bounds, "sl_src_bpi_zero"),
                };
                let dst_bpi = match dl.bytes_per_image() {
                    Some(v) if v > 0 => v,
                    _ => return br(BlitFailure::Bounds, "sl_dst_bpi_zero"),
                };
                (src_bpi, dst_bpi, d as u64)
            } else if cmd.slice_count <= 1 {
                // One image, so neither stride is ever stepped: both consumers
                // scale it by `image_count - 1`, which is zero here —
                // `strided_span`'s `last_image` term and `copy_region`'s `z`
                // loop. That is why this arm may fall back on `row_bytes` where
                // the volume arm above refuses a missing `bytes_per_image` by
                // name: there the stride selects z planes and a wrong one reads
                // the wrong plane, here it is multiplied away. The fallback is
                // inert rather than a second opinion about the layout.
                (
                    sl.bytes_per_image().unwrap_or(row_bytes),
                    dl.bytes_per_image().unwrap_or(row_bytes),
                    1u64,
                )
            } else {
                (sl.slice_stride, dl.slice_stride, cmd.slice_count as u64)
            };
            // Same allocation overlap check (conservative).
            if sl.base_gva == dl.base_gva {
                let span = row_bytes.saturating_mul(rows).saturating_mul(image_count);
                if ranges_overlap(src_off, span, dst_off, span) {
                    return br(BlitFailure::Overlap, "sl_overlap");
                }
            }
            if let Err(st) = copy_row_region(
                host,
                state,
                task_id,
                src_gva,
                sl.row_stride,
                src_img_stride,
                dst_gva,
                dl.row_stride,
                dst_img_stride,
                row_bytes,
                rows,
                image_count,
            ) {
                return st;
            }
            continue;
        }

        // Mapper-ref-texture / mixed: depth-1 only (mapper-ref-texture is 2D); per-slice
        // whole-surface.
        if is_volume {
            return br(BlitFailure::Unsupported, "sl_volume_mixed");
        }
        // The slice/level form's mapper-ref-texture arm. It used to stage one row at a
        // time, and a driven Maps leg charged that loop 30.15 s of a 30.28 s
        // blit rail to move 14.6 MB, against 0.12 s for the resolves beside it
        // and 0.22 s for every strided guest-RAM copy in the device. The bytes
        // were never the cost — re-entering the mapping rail per row was. It
        // stages the slice whole now; see [`read_texture_rect`].
        let sl_mixed_started = std::time::Instant::now();
        let mut staged = vec![0u8; (row_bytes.saturating_mul(rows)) as usize];
        for si in 0..cmd.slice_count {
            let ss = match cmd.source_slice.checked_add(si) {
                Some(v) => v,
                None => return br(BlitFailure::Bounds, "sl_inner_src_slice_overflow"),
            };
            let ds = match cmd.dest_slice.checked_add(si) {
                Some(v) => v,
                None => return br(BlitFailure::Bounds, "sl_inner_dst_slice_overflow"),
            };
            let src = match resolve_texture_backing(
                state,
                host,
                task_id,
                cmd.source_ref,
                src_level,
                ss,
            ) {
                Ok(t) => t,
                Err(st) => return st,
            };
            let dst =
                match resolve_texture_backing(state, host, task_id, cmd.dest_ref, dst_level, ds) {
                    Ok(t) => t,
                    Err(st) => return st,
                };
            if src.width() != w || src.height() != h || dst.width() != w || dst.height() != h {
                return br(BlitFailure::Bounds, "sl_inner_dim_mismatch");
            }
            let allowed = match texture_region_window(
                state,
                host,
                task_id,
                &dst,
                Point { x: 0, y: 0, z: 0 },
                w,
                rows,
                1,
                bpp,
            ) {
                Ok(v) => v,
                Err(st) => return st,
            };
            if let Err(st) = read_texture_rect(
                state,
                host,
                task_id,
                &src,
                Point { x: 0, y: 0, z: 0 },
                row_bytes,
                rows,
                &mut staged,
            ) {
                return st;
            }
            if let Err(st) = write_texture_rect(
                state,
                host,
                task_id,
                &dst,
                Point { x: 0, y: 0, z: 0 },
                row_bytes,
                h as u64,
                &staged,
                allowed.as_ref(),
            ) {
                return st;
            }
        }
        crate::runtime::drain::note_store_route_us(
            "sl_mixed_us",
            sl_mixed_started.elapsed().as_micros() as u64,
        );
    }
    BlitStatus::Ok
}

/// Execute blit fence update (`0x13c`) or wait (`0x13d`) on the blit-fence domain.
///
/// See [`fence_exec::execute_fence`].
pub fn execute_blit_fence(
    state: &mut DeviceState,
    task_id: u32,
    record: &reims_vgpu_protocol::decode::sync::FenceRecord,
) -> BlitStatus {
    use reims_vgpu_protocol::sync::FenceKind;
    // Which side of the fence this is comes off the record, and the record only
    // exists for the two opcodes that are fences. Both refusals this function
    // used to have — "the caller handed me a non-fence" and "the opcode is not
    // one of the two" — were about a `Command` that could carry any kind with
    // any opcode, and neither is expressible now.
    let action = match record.kind {
        FenceKind::Update => FenceAction::Update,
        FenceKind::Wait => FenceAction::Wait,
    };
    blit_status_from_fence(fence_exec::execute_fence(
        state,
        task_id,
        FenceDomain::BlitFence,
        record.fence_ref,
        action,
    ))
}

/// Re-express a fence outcome as a blit outcome, carrying the reason across.
///
/// Named rather than inlined so the reason-forwarding is directly testable: the
/// `Unsupported` arm used to write a flat `fence_unsupported`, so all seven
/// fence/event refusals — bad domain, event-on-fence-path, either timeout form,
/// either invalid plan, unknown event kind — reached the blit dispatch line as
/// one indistinguishable reason. The forwarded slug is registered by
/// `FenceStatus`, not by this file.
pub(crate) fn blit_status_from_fence(status: FenceStatus) -> BlitStatus {
    match status {
        FenceStatus::Ok => BlitStatus::Ok,
        FenceStatus::Pending => BlitStatus::FencePending,
        FenceStatus::Missing => br(BlitFailure::MissingResource, "fence_missing"),
        FenceStatus::Unsupported(why) => br(BlitFailure::Unsupported, why),
    }
}

/// Execute a decoded blit command on the product path.
///
/// Returns [`BlitFailure::Unsupported`] for resource/image/mipmap opcodes
/// that other modules own or that are protocol no-ops (caller should not count
/// those as copy/fill failures). Fences use [`execute_blit_fence`].
///
/// Nothing follows a successful copy. A blit into a mapper-ref-texture destination writes
/// the guest pages directly and no GPU object caches those bytes, so the
/// content is coherent by construction and there is nothing to invalidate.
/// Each arm resolved its own destination in order to write it, so a second
/// resolve afterwards only repeats the page walk.
pub fn execute_blit<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    record: &BlitRecord,
) -> BlitStatus {
    // One clock over the whole dispatch, attributed by the arm that ran.
    //
    // The per-loop clocks added alongside this are the ones that say *why* an
    // arm is slow, but each of them had to be placed by hand and between them
    // they accounted for 0.7 % of `walk_blit_us`. Being exhaustive by
    // construction is what this one buys: every record entering `execute_blit`
    // leaves through exactly one arm and is charged to it, so the sum of
    // `blit_kind_*_us` cannot be less than the rail's cost the way a hand-placed
    // set can. A family that turns out to hold the wall clock and has no inner
    // clock yet is then a known gap rather than an invisible one.
    //
    // The route and the arm both come from the record's own shape, so a record
    // charged to one family and executed by another is no longer expressible —
    // it used to be two independent reads of a `kind` tag and a `copy_kind` tag
    // that a decoder set separately.
    let kind_started = std::time::Instant::now();
    let kind_route = match record.kind() {
        WireBlitKind::FillBuffer => "blit_kind_fill_us",
        WireBlitKind::FillBufferPattern4 => "blit_kind_fill4_us",
        WireBlitKind::BufferToBuffer => "blit_kind_b2b_us",
        WireBlitKind::BufferToTexture => "blit_kind_b2t_us",
        WireBlitKind::TextureToBuffer => "blit_kind_t2b_us",
        WireBlitKind::TextureRegion | WireBlitKind::TextureRegionOptions => "blit_kind_t2t_us",
        WireBlitKind::TextureSlices => "blit_kind_t2t_sl_us",
        WireBlitKind::GenerateMipmaps => "blit_kind_mipmap_us",
    };
    let status = match record {
        BlitRecord::FillBuffer(fill) => match fill.pattern {
            // Which fill this is decides which write path runs *and* which of
            // the two refusal families the arm reports under, so the pattern is
            // matched once here and each arm receives only the width it writes.
            FillPattern::Byte(value) => exec_fill_buffer(state, host, task_id, fill, value),
            FillPattern::Pattern4(word) => {
                exec_fill_buffer_pattern4(state, host, task_id, fill, word)
            }
        },
        BlitRecord::BufferToBuffer(r) => exec_copy_buffer_to_buffer(state, host, task_id, r),
        BlitRecord::BufferToTexture(r) => exec_copy_buffer_to_texture(state, host, task_id, r),
        BlitRecord::TextureToBuffer(r) => exec_copy_texture_to_buffer(state, host, task_id, r),
        BlitRecord::TextureRegion(r) => exec_copy_texture_to_texture(state, host, task_id, r),
        BlitRecord::TextureSlices(r) => {
            exec_copy_texture_to_texture_slice_level(state, host, task_id, r)
        }
        // `generateMipmapsForTexture:` is a transfer record by opcode and a
        // filter chain by execution, and it has an owner of its own —
        // `runtime::mipmap`, which the caller reaches directly. A record
        // arriving here means that dispatch stopped routing it.
        BlitRecord::GenerateMipmaps(GenerateMipmaps { .. }) => {
            br(BlitFailure::Unsupported, "blit_kind_mipmap_misrouted")
        }
    };
    crate::runtime::drain::note_store_route_us(
        kind_route,
        kind_started.elapsed().as_micros() as u64,
    );
    status
}

#[cfg(test)]
mod tests;
