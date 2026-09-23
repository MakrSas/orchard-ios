//! Opcode 0x1a — the render pass descriptor, and its six capability siblings.
//!
//! `-[PGSerializerRenderCommandEncoder writeDescriptor]`.
//!
//! # Why this record was believed not to exist
//!
//! `reims-vgpu` has decoded `0x1a` since long before this crate, and recorded it
//! as a number "absent from Apple's render manifest" — an opcode this device's
//! own framing carries rather than one `PGSerializerRenderCommandEncoder`
//! writes. That claim held only because no case had ever driven the selector
//! that emits it. Constructing an encoder emits the pass record, every case
//! resets the capture arena immediately afterwards, and so the one record every
//! case produced was the one no case could reach.
//!
//! `writeDescriptor` re-emits it on demand, which is what makes the perturbation
//! family below possible: the record *is* the pass descriptor, so moving a field
//! means building a different encoder rather than calling a different selector.
//!
//! # Layout
//!
//! Total 592 bytes: the 8-byte [`crate::op::OpHeader`] then a 584-byte payload.
//!
//! ```text
//! payload +000  40 bytes  depth attachment
//! payload +028  36 bytes  stencil attachment
//! payload +04c  8 x 60    colour attachments 0..7
//! payload +22c  u32       visibility_result_buffer_ref
//! payload +230  u64       render_target_array_length
//! payload +238  u64       render_target_width
//! payload +240  u64       render_target_height
//! ```
//!
//! All three attachment shapes open with the same 28-byte prefix
//! ([`AttachmentPrefix`]) and differ only in what follows their `store_action`
//! options word:
//!
//! ```text
//! prefix +000  u32  texture_ref
//! prefix +004  u32  resolve_texture_ref
//! prefix +008  u16  level
//! prefix +00a  u16  slice
//! prefix +00c  u16  depth_plane
//! prefix +00e  u16  resolve_level
//! prefix +010  u16  resolve_slice
//! prefix +012  u16  resolve_depth_plane
//! prefix +014  u16  load_action
//! prefix +016  u16  store_action
//! prefix +018  u16  store_action_options
//! prefix +01a  2 bytes  unwritten
//!
//! colour  +01c  4 x f64  clear_color                      -> 60
//! depth   +01c  f64 clear_depth, +024 u16 resolve filter   -> 40
//! stencil +01c  u32 clear_stencil, +020 u16 resolve filter -> 36
//! ```
//!
//! Nothing in the 592 bytes is unidentified.
//!
//! # `level` is sixteen bits, and reading it as thirty-two takes the slice
//!
//! `slice` sits immediately above `level`, so a thirty-two bit load at
//! `prefix +008` returns `level | (slice << 16)`. `reims-vgpu` did exactly that
//! on the *colour* arm while its depth and stencil arms read sixteen — with a
//! comment on the depth arm asserting that "colour uses u32", which nothing had
//! ever checked. Any pass rendering to a non-zero array slice or cube face
//! therefore produced an enormous mip level.
//!
//! That is the fifth time a field's **width** rather than its offset was the
//! thing this crate moved; see `useOffset`, `copyFromTexture:toBuffer:`'s
//! `options`, the IOSurface `plane`, and `usage` on the wide descriptor.
//!
//! `store_action_options` and both resolve filters are the same shape one step
//! further: each occupies a four-byte slot of which the serializer writes only
//! the low **sixteen bits**, so a `u32` read of any of the three picks up the
//! guest's stale ring in its top half. Those are the twelve two-byte spans the
//! written-bit mask reports for this record and the only unwritten bytes in it.
//!
//! # How the layout was derived
//!
//! Perturbation across nineteen cases, each moving one property of the pass
//! descriptor off a baseline of a single cleared colour attachment: the target
//! size, the array length, the visibility buffer, the colour attachment's
//! level/slice/plane, its resolve half, a second attachment at slot three (which
//! is what pins the 60-byte stride rather than assuming eight equal slots), the
//! depth and stencil textures with their clear values, both load actions moved
//! *off* the value `MTLRenderPassDescriptor` hands back, all three
//! `storeActionOptions` and both resolve filters.
//!
//! Every remaining `MTLRenderPassDescriptor` property has been driven too, and
//! none of them is a field of this record. Six emit records of their own, each
//! behind a capability that defaults off — see [`is_render_pass_opcode`]. The
//! seventh, `sampleBufferAttachments`, moves no byte and adds no record, driven
//! both with a forwarding stub and with a real `MTLCounterSampleBuffer`.
//!
//! The four tile properties are the ones worth not re-deriving.
//! `imageblockSampleLength`, `threadgroupMemoryLength`, `tileWidth` and
//! `tileHeight` reach no byte of these 592 with Metal confirmed to have kept the
//! values the cases set — so "measured absent from a record" meant "in a
//! different record, behind a flag" rather than "not on the wire".

use crate::le::{U16le, U32le, U64le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

/// Opcode for the render pass descriptor.
pub const OPCODE_RENDER_PASS: u32 = 0x1a;

/// Total wire length of a render-pass record, header included.
pub const RENDER_PASS_TOTAL_LEN: u32 = 592;

/// Colour attachment slots the record always carries, written or not.
pub const RENDER_PASS_COLOR_ATTACHMENTS: usize = 8;

/// The 28 bytes every attachment shape opens with.
///
/// Shared by the colour, depth and stencil slots, so a layout error fails on all
/// three rather than leaving two arms right and one wrong — which is the state
/// `reims-vgpu` was in.
#[repr(C)]
#[derive(PartialEq, Eq)]
pub struct AttachmentPrefix {
    /// Serializer ref of the attached texture; 0 when the slot is unattached.
    ///
    /// Observed: the stub texture's `4242` in colour slot 0, `4343` in colour
    /// slot 3 and in the stencil slot, `4444` in the depth slot.
    pub texture_ref: U32le,
    /// Serializer ref of the multisample resolve target, 0 when there is none.
    /// Observed: `4343` with `resolveTexture` set and nothing else moved.
    pub resolve_texture_ref: U32le,
    /// Mip level. Sixteen bits — see the module doc. Observed: 2.
    pub level: U16le,
    /// Array slice. Observed: 3.
    pub slice: U16le,
    /// Depth plane of a 3D attachment. Observed: 4.
    pub depth_plane: U16le,
    /// Mip level of the resolve target. Observed: 5.
    pub resolve_level: U16le,
    /// Array slice of the resolve target. Observed: 6.
    pub resolve_slice: U16le,
    /// Depth plane of the resolve target. Observed: 7.
    pub resolve_depth_plane: U16le,
    /// `MTLLoadAction`, carried verbatim. Observed: 2 (`Clear`) on a colour slot
    /// set to clear, 1 (`Load`) on one set to load, 0 (`DontCare`) on an
    /// unattached slot.
    pub load_action: U16le,
    /// `MTLStoreAction`, carried verbatim. Observed: 1 (`Store`), 0
    /// (`DontCare`), 2 (`MultisampleResolve`).
    pub store_action: U16le,
    /// `MTLStoreActionOptions`, sixteen bits of a four-byte slot. Observed: 1
    /// (`CustomSamplePositions`) on each of the colour, depth and stencil
    /// slots, driven separately rather than carried across.
    pub store_action_options: U16le,
    /// Never written by the serializer; the guest's stale ring on a real wire.
    pub unwritten_above_store_action_options: [u8; 2],
}

// SAFETY: `le` scalars and a `[u8; 2]`, all align-1 and valid for every byte
// pattern.
unsafe impl Wire for AttachmentPrefix {}

/// One colour attachment slot. 60 bytes.
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub struct ColorAttachmentBody {
    pub prefix: AttachmentPrefix,
    /// `MTLClearColor`'s four components, each an IEEE-754 double. Observed:
    /// 0.25, 0.5, 0.75, 1.0 on slot 0 and 0.125, 0.375, 0.625, 0.875 on slot 3.
    /// Prefer [`ColorAttachmentBody::clear_color`].
    pub clear_color_bits: [U64le; 4],
}

// SAFETY: a `Wire` struct and an array of `le` scalars, align-1 and valid for
// every byte pattern.
unsafe impl Wire for ColorAttachmentBody {}

impl ColorAttachmentBody {
    /// The clear colour, component by component.
    #[inline]
    pub fn clear_color(&self) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        let mut i = 0;
        while i < 4 {
            out[i] = f64::from_bits(self.clear_color_bits[i].get());
            i += 1;
        }
        out
    }
}

/// The depth attachment slot. 40 bytes.
#[repr(C)]
#[derive(PartialEq, Eq)]
pub struct DepthAttachmentBody {
    pub prefix: AttachmentPrefix,
    /// `clearDepth` as an IEEE-754 double. Observed: 1.0 (the value
    /// `MTLRenderPassDescriptor` hands back), 0.375 and 0.625. Prefer
    /// [`DepthAttachmentBody::clear_depth`].
    pub clear_depth_bits: U64le,
    /// `MTLMultisampleDepthResolveFilter`, sixteen bits of a four-byte slot.
    /// Observed: 2 (`Max`).
    pub resolve_filter: U16le,
    /// Never written by the serializer.
    pub unwritten_above_resolve_filter: [u8; 2],
}

// SAFETY: a `Wire` struct, `le` scalars and a `[u8; 2]`, align-1 and valid for
// every byte pattern.
unsafe impl Wire for DepthAttachmentBody {}

impl DepthAttachmentBody {
    #[inline]
    pub fn clear_depth(&self) -> f64 {
        f64::from_bits(self.clear_depth_bits.get())
    }
}

/// The stencil attachment slot. 36 bytes.
#[repr(C)]
#[derive(PartialEq, Eq)]
pub struct StencilAttachmentBody {
    pub prefix: AttachmentPrefix,
    /// `clearStencil`, a full 32 bits. Observed: `0x5a` and `0xa5`.
    pub clear_stencil: U32le,
    /// `MTLMultisampleStencilResolveFilter`, sixteen bits of a four-byte slot.
    /// Observed: 1 (`DepthResolvedSample`).
    pub resolve_filter: U16le,
    /// Never written by the serializer.
    pub unwritten_above_resolve_filter: [u8; 2],
}

// SAFETY: a `Wire` struct, `le` scalars and a `[u8; 2]`, align-1 and valid for
// every byte pattern.
unsafe impl Wire for StencilAttachmentBody {}

// --- `Debug` without the bytes this module declares meaningless -------------
//
// The three attachment bodies each carry a field the serializer never writes,
// which on a real wire is the guest's stale ring. A derived `Debug` renders
// those bytes, which is this crate advertising noise it has already documented
// as noise — and a reader comparing two renderings of the same record would see
// them differ on bytes no field means anything by. `PartialEq` is untouched:
// byte equality over a descriptor is a legitimate question and a different one.

impl core::fmt::Debug for AttachmentPrefix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AttachmentPrefix")
            .field("texture_ref", &self.texture_ref)
            .field("resolve_texture_ref", &self.resolve_texture_ref)
            .field("level", &self.level)
            .field("slice", &self.slice)
            .field("depth_plane", &self.depth_plane)
            .field("resolve_level", &self.resolve_level)
            .field("resolve_slice", &self.resolve_slice)
            .field("resolve_depth_plane", &self.resolve_depth_plane)
            .field("load_action", &self.load_action)
            .field("store_action", &self.store_action)
            .field("store_action_options", &self.store_action_options)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for DepthAttachmentBody {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DepthAttachmentBody")
            .field("prefix", &self.prefix)
            .field("clear_depth_bits", &self.clear_depth_bits)
            .field("resolve_filter", &self.resolve_filter)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for StencilAttachmentBody {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StencilAttachmentBody")
            .field("prefix", &self.prefix)
            .field("clear_stencil", &self.clear_stencil)
            .field("resolve_filter", &self.resolve_filter)
            .finish_non_exhaustive()
    }
}

/// Payload of a render-pass record.
///
/// Comparable, along with the attachment bodies it contains. Equality here is
/// byte equality over a descriptor a caller holds by reference — it asserts
/// nothing about what any field *means*, which is what the rule against
/// comparing bodies with `unidentified_` fields is about. What it buys is that
/// a reader can ask whether two passes were described the same way without
/// copying 592 bytes to find out.
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub struct RenderPassBody {
    pub depth: DepthAttachmentBody,
    pub stencil: StencilAttachmentBody,
    pub color: [ColorAttachmentBody; RENDER_PASS_COLOR_ATTACHMENTS],
    /// Serializer ref of `visibilityResultBuffer`, 0 when there is none.
    ///
    /// Observed: the stub buffer's `5151`. This is the other half of
    /// `setVisibilityResultMode:offset:` — that record carries the mode and the
    /// offset, and the buffer they index lives only here.
    pub visibility_result_buffer_ref: U32le,
    /// `renderTargetArrayLength`. Observed: `0x11`.
    ///
    /// Declared `Q` by the property and given an eight-byte slot by the record's
    /// own arithmetic — `render_target_width` starts eight bytes later. Only the
    /// low byte has been driven; a case setting a length above `u32::MAX` would
    /// pin the written extent, and Metal is expected to clamp one.
    pub render_target_array_length: U64le,
    /// `renderTargetWidth`. Observed: `0x1234`, all eight bytes written.
    ///
    /// The pass's explicit target extent, which is not the attached texture's:
    /// a guest may bind a large texture and render into a corner of it.
    pub render_target_width: U64le,
    /// `renderTargetHeight`. Observed: `0x5678`.
    pub render_target_height: U64le,
}

// SAFETY: `Wire` structs, an array of them, and `le` scalars — all align-1 and
// valid for every byte pattern.
unsafe impl Wire for RenderPassBody {}

/// View the payload of a render-pass record.
pub fn render_pass<'a>(op: &Op<'a>) -> Result<&'a RenderPassBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_RENDER_PASS);
    view::<RenderPassBody>(op.payload)
}

/// Opcode for the pass's `defaultRasterSampleCount`, a record of its own.
///
/// Emitted by `writeDescriptor` **in addition to** the pass record, and only
/// under `-setSupportsDefaultRasterSampleCount:`. At the default capability
/// state the encoder's designated initializer returns nil for a descriptor
/// carrying a non-default count, so this is not a record that can be reached by
/// asking the serializer more politely.
pub const OPCODE_DEFAULT_RASTER_SAMPLE_COUNT: u32 = 0x1e;

/// Total wire length of a default-raster-sample-count record.
pub const DEFAULT_RASTER_SAMPLE_COUNT_TOTAL_LEN: u32 = 12;

/// Payload of a default-raster-sample-count record.
#[repr(C)]
#[derive(Debug)]
pub struct DefaultRasterSampleCountBody {
    /// Observed: 4, the value the case set, all four bytes written.
    pub count: U32le,
}

// SAFETY: one `le` scalar, align-1 and valid for every byte pattern.
unsafe impl Wire for DefaultRasterSampleCountBody {}

pub fn default_raster_sample_count<'a>(
    op: &Op<'a>,
) -> Result<&'a DefaultRasterSampleCountBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DEFAULT_RASTER_SAMPLE_COUNT);
    view::<DefaultRasterSampleCountBody>(op.payload)
}

/// Opcode for the pass's `rasterizationRateMap`, a record of its own.
///
/// Emitted by `writeDescriptor` alongside the pass record under
/// `-setSupportsRasterizationRateMap:`. This is *not* [`crate::ops::rate_map`],
/// which is the map's creation record; this one names an already-created map by
/// ref and binds it to a pass.
///
/// It reaches this device through the command stream rather than through an
/// object list, so unlike the creation record it is not blocked on knowing which
/// `object_type` a rate map arrives under.
pub const OPCODE_RASTERIZATION_RATE_MAP: u32 = 0x21;

/// Total wire length of a pass rasterization-rate-map record.
pub const RASTERIZATION_RATE_MAP_TOTAL_LEN: u32 = 12;

/// Payload of a pass rasterization-rate-map record.
#[repr(C)]
#[derive(Debug)]
pub struct PassRateMapBody {
    /// Serializer ref of the rate map. Observed: the stub's `6767`.
    pub rate_map_ref: U32le,
}

// SAFETY: one `le` scalar, align-1 and valid for every byte pattern.
unsafe impl Wire for PassRateMapBody {}

pub fn pass_rate_map<'a>(op: &Op<'a>) -> Result<&'a PassRateMapBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_RASTERIZATION_RATE_MAP);
    view::<PassRateMapBody>(op.payload)
}

/// Opcode for the pass's programmable sample positions.
///
/// Emitted alongside the pass record under `-setSupportsProgrammableSamplePositions:`.
/// Variable length: a count then that many `(x, y)` pairs of `f32`.
pub const OPCODE_SAMPLE_POSITIONS: u32 = 0x20;

/// Fixed head of a sample-positions record, header included.
pub const SAMPLE_POSITIONS_HEAD_LEN: u32 = 12;

/// Bytes each sample position occupies.
pub const SAMPLE_POSITION_LEN: u32 = 8;

/// One programmable sample position, in pixel-relative coordinates.
///
/// Observed: `(0.25, 0.75)` and `(0.125, 0.375)`, both verbatim as `f32` where
/// `MTLSamplePosition` declares `float`.
#[repr(C)]
#[derive(Debug)]
pub struct SamplePosition {
    pub x_bits: U32le,
    pub y_bits: U32le,
}

// SAFETY: two `le` scalars, align-1 and valid for every byte pattern.
unsafe impl Wire for SamplePosition {}

impl SamplePosition {
    #[inline]
    pub fn x(&self) -> f32 {
        f32::from_bits(self.x_bits.get())
    }
    #[inline]
    pub fn y(&self) -> f32 {
        f32::from_bits(self.y_bits.get())
    }
}

/// Fixed head of a sample-positions record.
#[repr(C)]
#[derive(Debug)]
pub struct SamplePositionsHead {
    /// How many positions follow. Observed: 2, with the record 28 bytes long,
    /// which is `12 + 2 * 8` exactly.
    pub count: U32le,
}

// SAFETY: one `le` scalar, align-1 and valid for every byte pattern.
unsafe impl Wire for SamplePositionsHead {}

/// View a sample-positions record: its head and its positions.
///
/// The count is guest-controlled, so the positions are bounded by the record's
/// own length rather than trusted; a count claiming more than the record holds
/// is refused.
pub fn sample_positions<'a>(
    op: &Op<'a>,
) -> Result<(&'a SamplePositionsHead, &'a [SamplePosition]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SAMPLE_POSITIONS);
    let (head, rest) = crate::view::split::<SamplePositionsHead>(op.payload)?;
    let positions = crate::view::view_slice::<SamplePosition>(rest, head.count.get() as usize)?;
    Ok((head, positions))
}

/// Opcode for the pass's `imageblockSampleLength`.
///
/// One of three records `TileShaders` moves *out* of the pass descriptor; see
/// [`OPCODE_TILE_SIZE`].
pub const OPCODE_IMAGEBLOCK_SAMPLE_LENGTH: u32 = 0x22;

/// Opcode for the pass's `threadgroupMemoryLength`.
pub const OPCODE_THREADGROUP_MEMORY_LENGTH: u32 = 0x23;

/// Total wire length of either tile-memory record.
pub const TILE_MEMORY_TOTAL_LEN: u32 = 12;

/// Payload of an imageblock-sample-length or threadgroup-memory-length record.
///
/// Both carry one `NSUInteger` property narrowed to 32 bits. Observed: `0x40`
/// and `0x80`, the values the cases set, all four bytes written.
#[repr(C)]
#[derive(Debug)]
pub struct TileMemoryBody {
    pub length: U32le,
}

// SAFETY: one `le` scalar, align-1 and valid for every byte pattern.
unsafe impl Wire for TileMemoryBody {}

pub fn tile_memory<'a>(op: &Op<'a>) -> Result<&'a TileMemoryBody, WireError> {
    debug_assert!(
        op.opcode() == OPCODE_IMAGEBLOCK_SAMPLE_LENGTH
            || op.opcode() == OPCODE_THREADGROUP_MEMORY_LENGTH
    );
    view::<TileMemoryBody>(op.payload)
}

/// Opcode for the pass's `tileWidth` and `tileHeight`, one record for the pair.
///
/// `tileWidth`, `tileHeight`, `imageblockSampleLength` and
/// `threadgroupMemoryLength` are measured **not** to appear anywhere in the pass
/// record at the default capability state, with Metal confirmed to have kept the
/// values the cases set. Under `-setSupportsTileShaders:` they emit these three
/// records instead. So the four are not fields this crate failed to find: they
/// leave the descriptor entirely, and which of the two happens depends on a flag
/// nothing in `reims-vgpu` observes the guest negotiating.
pub const OPCODE_TILE_SIZE: u32 = 0x24;

/// Total wire length of a tile-size record.
pub const TILE_SIZE_TOTAL_LEN: u32 = 12;

/// Payload of a tile-size record.
#[repr(C)]
#[derive(Debug)]
pub struct TileSizeBody {
    /// Observed: `0x21`. Sixteen bits, from a property declared `Q`.
    pub width: U16le,
    /// Observed: `0x22`.
    pub height: U16le,
}

// SAFETY: two `le` scalars, align-1 and valid for every byte pattern.
unsafe impl Wire for TileSizeBody {}

pub fn tile_size<'a>(op: &Op<'a>) -> Result<&'a TileSizeBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_TILE_SIZE);
    view::<TileSizeBody>(op.payload)
}

/// Whether this opcode is one `writeDescriptor` emits.
///
/// The run is `0x1a` and `0x1e`..`0x24` with one hole at `0x1f`, and that hole
/// is **not** any `MTLRenderPassDescriptor` property: every one of them has been
/// driven. The last to go was `sampleBufferAttachments`, and it was driven
/// twice, because the first result could not be trusted — a forwarding stub
/// answering zero to something the serializer asks for produces the same silence
/// as a property that does not reach the wire, which is the distinction
/// `gCrashed` exists to keep.
///
/// So `render_pass_sample_buffer` sets it with a stub and
/// `render_pass_sample_buffer_real` sets it with a genuine
/// `MTLCounterSampleBuffer` built from this host's device and its first counter
/// set, both with four distinctive sample indices. Both emit exactly one record
/// and neither moves a byte of it. The property does not reach the wire.
///
/// What is left for `0x1f` is therefore not a property of this descriptor. It is
/// a hole in a run, which this crate's `AGENTS.md` records as "a selector nobody
/// has driven yet" — so the next place to look is another selector, not another
/// field.
pub fn is_render_pass_opcode(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_RENDER_PASS
            | OPCODE_DEFAULT_RASTER_SAMPLE_COUNT
            | OPCODE_SAMPLE_POSITIONS
            | OPCODE_RASTERIZATION_RATE_MAP
            | OPCODE_IMAGEBLOCK_SAMPLE_LENGTH
            | OPCODE_THREADGROUP_MEMORY_LENGTH
            | OPCODE_TILE_SIZE
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op, OP_HEADER_LEN};
    use core::mem::{align_of, size_of};

    fn synth() -> [u8; RENDER_PASS_TOTAL_LEN as usize] {
        let mut b = [0xAAu8; RENDER_PASS_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_RENDER_PASS.to_le_bytes());
        b[4..8].copy_from_slice(&RENDER_PASS_TOTAL_LEN.to_le_bytes());
        b
    }

    /// Payload offset of colour attachment `i`, computed the way a hex dump is
    /// read rather than the way the struct is laid out.
    const fn color_at(i: usize) -> usize {
        0x4c + i * 0x3c
    }

    #[test]
    fn the_payload_is_exactly_the_record_minus_its_header() {
        assert_eq!(
            size_of::<RenderPassBody>() + OP_HEADER_LEN,
            RENDER_PASS_TOTAL_LEN as usize
        );
        assert_eq!(align_of::<RenderPassBody>(), 1);
        assert_eq!(
            size_of::<DefaultRasterSampleCountBody>() + OP_HEADER_LEN,
            DEFAULT_RASTER_SAMPLE_COUNT_TOTAL_LEN as usize
        );
        assert_eq!(
            size_of::<PassRateMapBody>() + OP_HEADER_LEN,
            RASTERIZATION_RATE_MAP_TOTAL_LEN as usize
        );
    }

    /// The three attachment shapes share a prefix and differ only after it.
    #[test]
    fn every_attachment_shape_is_its_prefix_plus_its_own_clear_value() {
        assert_eq!(size_of::<AttachmentPrefix>(), 0x1c);
        assert_eq!(size_of::<ColorAttachmentBody>(), 0x3c);
        assert_eq!(size_of::<DepthAttachmentBody>(), 0x28);
        assert_eq!(size_of::<StencilAttachmentBody>(), 0x24);
        // The three slots tile the head of the payload with no gap, which is
        // what lets a reader take the colour array's base as a constant.
        assert_eq!(
            size_of::<DepthAttachmentBody>() + size_of::<StencilAttachmentBody>(),
            color_at(0)
        );
        assert_eq!(
            color_at(RENDER_PASS_COLOR_ATTACHMENTS),
            size_of::<RenderPassBody>() - 28,
            "the tail is the last 28 bytes and the colour array ends where it starts"
        );
    }

    #[test]
    fn a_render_pass_reads_back_every_field_of_every_shape() {
        let mut b = synth();
        let p = OP_HEADER_LEN;
        // Depth: ref, level, load/store, clear, resolve filter.
        b[p..p + 4].copy_from_slice(&4444u32.to_le_bytes());
        b[p + 0x08..p + 0x0a].copy_from_slice(&1u16.to_le_bytes());
        b[p + 0x14..p + 0x16].copy_from_slice(&2u16.to_le_bytes());
        b[p + 0x16..p + 0x18].copy_from_slice(&1u16.to_le_bytes());
        b[p + 0x18..p + 0x1a].copy_from_slice(&1u16.to_le_bytes());
        b[p + 0x1c..p + 0x24].copy_from_slice(&0.375f64.to_bits().to_le_bytes());
        b[p + 0x24..p + 0x26].copy_from_slice(&2u16.to_le_bytes());
        // Stencil.
        let s = p + 0x28;
        b[s..s + 4].copy_from_slice(&4343u32.to_le_bytes());
        b[s + 0x1c..s + 0x20].copy_from_slice(&0x5au32.to_le_bytes());
        b[s + 0x20..s + 0x22].copy_from_slice(&1u16.to_le_bytes());
        // Colour slot 3, to exercise the stride.
        let c = p + color_at(3);
        b[c..c + 4].copy_from_slice(&4343u32.to_le_bytes());
        b[c + 4..c + 8].copy_from_slice(&4242u32.to_le_bytes());
        b[c + 0x08..c + 0x0a].copy_from_slice(&2u16.to_le_bytes());
        b[c + 0x0a..c + 0x0c].copy_from_slice(&3u16.to_le_bytes());
        b[c + 0x0c..c + 0x0e].copy_from_slice(&4u16.to_le_bytes());
        b[c + 0x0e..c + 0x10].copy_from_slice(&5u16.to_le_bytes());
        b[c + 0x10..c + 0x12].copy_from_slice(&6u16.to_le_bytes());
        b[c + 0x12..c + 0x14].copy_from_slice(&7u16.to_le_bytes());
        b[c + 0x14..c + 0x16].copy_from_slice(&1u16.to_le_bytes());
        b[c + 0x16..c + 0x18].copy_from_slice(&0u16.to_le_bytes());
        for (i, v) in [0.125f64, 0.375, 0.625, 0.875].iter().enumerate() {
            let o = c + 0x1c + i * 8;
            b[o..o + 8].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        // The tail.
        let t = p + 0x22c;
        b[t..t + 4].copy_from_slice(&5151u32.to_le_bytes());
        b[t + 4..t + 12].copy_from_slice(&0x11u64.to_le_bytes());
        b[t + 12..t + 20].copy_from_slice(&0x1234u64.to_le_bytes());
        b[t + 20..t + 28].copy_from_slice(&0x5678u64.to_le_bytes());

        let o = op(&b, 0).expect("well formed");
        let rp = render_pass(&o).expect("fits");

        assert_eq!(rp.depth.prefix.texture_ref.get(), 4444);
        assert_eq!(rp.depth.prefix.level.get(), 1);
        assert_eq!(rp.depth.prefix.load_action.get(), 2);
        assert_eq!(rp.depth.prefix.store_action.get(), 1);
        assert_eq!(rp.depth.prefix.store_action_options.get(), 1);
        assert_eq!(rp.depth.clear_depth(), 0.375);
        assert_eq!(rp.depth.resolve_filter.get(), 2);

        assert_eq!(rp.stencil.prefix.texture_ref.get(), 4343);
        assert_eq!(rp.stencil.clear_stencil.get(), 0x5a);
        assert_eq!(rp.stencil.resolve_filter.get(), 1);

        let c3 = &rp.color[3];
        assert_eq!(c3.prefix.texture_ref.get(), 4343);
        assert_eq!(c3.prefix.resolve_texture_ref.get(), 4242);
        assert_eq!(c3.prefix.level.get(), 2);
        assert_eq!(c3.prefix.slice.get(), 3);
        assert_eq!(c3.prefix.depth_plane.get(), 4);
        assert_eq!(c3.prefix.resolve_level.get(), 5);
        assert_eq!(c3.prefix.resolve_slice.get(), 6);
        assert_eq!(c3.prefix.resolve_depth_plane.get(), 7);
        assert_eq!(c3.prefix.load_action.get(), 1);
        assert_eq!(c3.prefix.store_action.get(), 0);
        assert_eq!(c3.clear_color(), [0.125, 0.375, 0.625, 0.875]);

        assert_eq!(rp.visibility_result_buffer_ref.get(), 5151);
        assert_eq!(rp.render_target_array_length.get(), 0x11);
        assert_eq!(rp.render_target_width.get(), 0x1234);
        assert_eq!(rp.render_target_height.get(), 0x5678);
    }

    /// A sixteen-bit `level` beside a sixteen-bit `slice`.
    ///
    /// The bug this record taught: a 32-bit load at the level's offset returns
    /// `level | (slice << 16)`, so a pass rendering to array slice 1 reported
    /// mip level 65536. The view cannot express that read, and this pins why.
    #[test]
    fn level_and_slice_are_two_fields_and_not_one_dword() {
        let mut b = synth();
        let c = OP_HEADER_LEN + color_at(0);
        b[c + 0x08..c + 0x0a].copy_from_slice(&1u16.to_le_bytes());
        b[c + 0x0a..c + 0x0c].copy_from_slice(&0xffffu16.to_le_bytes());
        let o = op(&b, 0).expect("well formed");
        let rp = render_pass(&o).expect("fits");
        assert_eq!(rp.color[0].prefix.level.get(), 1);
        assert_eq!(rp.color[0].prefix.slice.get(), 0xffff);
    }

    /// The two bytes above each sixteen-bit option word are the guest's ring.
    #[test]
    fn the_option_words_ignore_the_two_unwritten_bytes_above_them() {
        let mut b = synth();
        let p = OP_HEADER_LEN;
        b[p + 0x18..p + 0x1a].copy_from_slice(&1u16.to_le_bytes());
        b[p + 0x1a] = 0x5a;
        b[p + 0x1b] = 0xa5;
        b[p + 0x24..p + 0x26].copy_from_slice(&2u16.to_le_bytes());
        b[p + 0x26] = 0xff;
        b[p + 0x27] = 0xff;
        let o = op(&b, 0).expect("well formed");
        let rp = render_pass(&o).expect("fits");
        assert_eq!(rp.depth.prefix.store_action_options.get(), 1);
        assert_eq!(rp.depth.resolve_filter.get(), 2);
    }

    #[test]
    fn a_truncated_render_pass_is_refused_rather_than_read_short() {
        let b = synth();
        let o = op(&b, 0).expect("well formed");
        let short = Op {
            header: o.header,
            payload: &o.payload[..0x22c],
            offset: 0,
        };
        assert!(matches!(
            render_pass(&short),
            Err(WireError::Short {
                need: 584,
                have: 556
            })
        ));
    }

    #[test]
    fn the_tile_records_carry_the_four_properties_the_pass_record_does_not() {
        let mut b = [0xAAu8; TILE_SIZE_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_TILE_SIZE.to_le_bytes());
        b[4..8].copy_from_slice(&TILE_SIZE_TOTAL_LEN.to_le_bytes());
        b[8..10].copy_from_slice(&0x21u16.to_le_bytes());
        b[10..12].copy_from_slice(&0x22u16.to_le_bytes());
        let o = op(&b, 0).expect("well formed");
        let t = tile_size(&o).expect("fits");
        assert_eq!(t.width.get(), 0x21);
        assert_eq!(t.height.get(), 0x22);

        for opcode in [
            OPCODE_IMAGEBLOCK_SAMPLE_LENGTH,
            OPCODE_THREADGROUP_MEMORY_LENGTH,
        ] {
            let mut b = [0xAAu8; TILE_MEMORY_TOTAL_LEN as usize];
            b[0..4].copy_from_slice(&opcode.to_le_bytes());
            b[4..8].copy_from_slice(&TILE_MEMORY_TOTAL_LEN.to_le_bytes());
            b[8..12].copy_from_slice(&0x40u32.to_le_bytes());
            let o = op(&b, 0).expect("well formed");
            assert_eq!(tile_memory(&o).expect("fits").length.get(), 0x40);
        }

        assert!(is_render_pass_opcode(OPCODE_RENDER_PASS));
        assert!(is_render_pass_opcode(OPCODE_TILE_SIZE));
        // The hole in the run. Claiming it would put a record under this
        // module's name that no capture has ever produced.
        assert!(!is_render_pass_opcode(0x1f));
    }

    #[test]
    fn sample_positions_are_bounded_by_the_record_rather_than_by_their_count() {
        const N: usize = 2;
        const TOTAL: usize = SAMPLE_POSITIONS_HEAD_LEN as usize + N * SAMPLE_POSITION_LEN as usize;
        let n = N;
        let total = TOTAL;
        let mut b = [0xAAu8; TOTAL];
        b[0..4].copy_from_slice(&OPCODE_SAMPLE_POSITIONS.to_le_bytes());
        b[4..8].copy_from_slice(&(total as u32).to_le_bytes());
        b[8..12].copy_from_slice(&(n as u32).to_le_bytes());
        for (i, (x, y)) in [(0.25f32, 0.75f32), (0.125, 0.375)].iter().enumerate() {
            let o = 12 + i * 8;
            b[o..o + 4].copy_from_slice(&x.to_bits().to_le_bytes());
            b[o + 4..o + 8].copy_from_slice(&y.to_bits().to_le_bytes());
        }
        let o = op(&b, 0).expect("well formed");
        let (head, positions) = sample_positions(&o).expect("fits");
        assert_eq!(head.count.get(), 2);
        assert_eq!(positions.len(), 2);
        assert_eq!((positions[0].x(), positions[0].y()), (0.25, 0.75));
        assert_eq!((positions[1].x(), positions[1].y()), (0.125, 0.375));

        // A count the record cannot hold is guest data, not a panic.
        b[8..12].copy_from_slice(&9u32.to_le_bytes());
        let o = op(&b, 0).expect("well formed");
        assert!(sample_positions(&o).is_err());
    }

    #[test]
    fn the_two_capability_siblings_carry_one_scalar_each() {
        let mut b = [0xAAu8; DEFAULT_RASTER_SAMPLE_COUNT_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_DEFAULT_RASTER_SAMPLE_COUNT.to_le_bytes());
        b[4..8].copy_from_slice(&DEFAULT_RASTER_SAMPLE_COUNT_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&4u32.to_le_bytes());
        let o = op(&b, 0).expect("well formed");
        assert_eq!(
            default_raster_sample_count(&o).expect("fits").count.get(),
            4
        );

        let mut b = [0xAAu8; RASTERIZATION_RATE_MAP_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_RASTERIZATION_RATE_MAP.to_le_bytes());
        b[4..8].copy_from_slice(&RASTERIZATION_RATE_MAP_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&6767u32.to_le_bytes());
        let o = op(&b, 0).expect("well formed");
        assert_eq!(pass_rate_map(&o).expect("fits").rate_map_ref.get(), 6767);
    }
}
