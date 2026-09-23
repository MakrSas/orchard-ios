//! The Vulkan rail's blit fast paths.
//!
//! A blit is a host copy on both rails: [`super::execute_blit`] resolves both
//! endpoints and moves guest bytes, and every function it calls is
//! backend-neutral. What this module adds is the two cases where the content is
//! already on the GPU and copying it through guest memory would cross a frame
//! twice — a whole-plane mapper-ref-texture to mapper-ref-texture copy out of a resident, and a
//! whole-plane mapper-ref-texture to guest-linear one.
//!
//! Both are **fall-throughs, never losses**: each returns `Option<BlitStatus>`,
//! and `None` means the host loop runs unchanged and lands the same pixels. The
//! Metal rail answers `None` to both, which is why
//! [`crate::backend::Backend`]'s two methods have a shape that reads the same
//! on either rail rather than a `cfg` at the call site.

use super::*;

/// Land the source's engine resident straight into the destination's guest pages
/// with the GPU, for the one shape where that is exactly the copy the guest asked
/// for.
///
/// # Why a blit is not a guest-byte reader here
///
/// `resolve_texture_backing` pays every endpoint's writeback debt, because its
/// answer is "where are this texture's guest bytes" and guest bytes are only a
/// resource's content once everything rendered into it has landed. That is right
/// for every endpoint the host row loops read or write. It is wrong for *this*
/// shape, and expensively so: a whole-plane copy out of a resident makes the
/// device read the resident back into the source's guest pages, then memcpy those
/// pages into the destination's — two crossings of a frame to move content the
/// GPU already holds.
///
/// So this arm never resolves the source, and **never pays the source's debt**.
/// The source's own guest pages stay stale and stay owed; the debt stays armed,
/// and the next genuine guest-byte reader — a sample, a compute bind, a
/// `CmdSynchronizeResources` — is what lands them. That is what the Metal
/// contract says: `copyFromTexture:toTexture:` is a blit-encoder command with no
/// host visibility, and `synchronizeResource:` is the separate call that means
/// "make this CPU-visible".
///
/// The *destination*'s debt is still paid, by the resolve below, and must be:
/// leaving it armed would let a pre-blit resident land over this copy's bytes
/// later.
///
/// Returns `None` for every fall-through, having named it on a counter. The
/// caller then runs the host path unchanged, so nothing here can lose a frame —
/// only spend one.
pub(crate) fn try_copy_whole_plane_on_gpu<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &TextureSlices,
) -> Option<BlitStatus> {
    use crate::runtime::drain::note_store_route;
    let key = crate::runtime::writeback_debt::GvaResourceKey {
        task_id,
        texture_ref: cmd.source_ref,
    };
    let debt = state.pending_writebacks.get_gva(key);
    if let Err(refusal) = gpu_whole_plane_admissible(
        cmd.level_count,
        cmd.slice_count,
        cmd.source_ref,
        cmd.dest_ref,
        debt.is_some(),
    ) {
        note_store_route(refusal.route());
        return None;
    }
    let debt = debt?;
    // Resolving the destination — and only the destination — is what pays its
    // debt, and it is the reason this call sits here rather than after the loop
    // below has resolved both endpoints.
    let dst = match resolve_texture_backing(
        state,
        host,
        task_id,
        cmd.dest_ref,
        cmd.dest_level,
        cmd.dest_slice,
    ) {
        Ok(t) => t,
        Err(_) => {
            // The host path re-resolves and returns this same refusal with its
            // own reason, so saying anything more here would double-count one
            // failure under two names.
            note_store_route("sl_gpu_dst_unresolved");
            return None;
        }
    };
    let TextureBacking::MapperRefTexture(t) = &dst else {
        note_store_route(GpuPlaneRefusal::DstNotMapperRefTexture.route());
        return None;
    };
    let plane = GpuPlane {
        width: t.width,
        height: t.height,
        surface_offset: t.surface_offset,
        row_stride: t.row_stride,
        pixel_format: t.pixel_format,
    };
    let mapping_id = t.mapping_id;
    let window = state
        .mappings
        .get(&mapping_id)
        .and_then(|m| mapping_write::resident_gpu_plane(m, plane.width, plane.height))
        .map(
            |(surface_offset, row_stride, pixel_format)| GpuMappingWindow {
                surface_offset,
                row_stride,
                pixel_format,
            },
        );
    let src = GpuResidentSource {
        width: debt.width,
        height: debt.height,
        pixel_format: debt.format,
    };
    if let Err(refusal) = gpu_whole_plane_destination(Some(plane), window, src) {
        note_store_route(refusal.route());
        // "The formats differ" does not say which of the three does, and the
        // three have different answers: a source that disagrees is a converting
        // copy the contract does not describe, while a *mapping* that disagrees
        // with a destination the source already matches is this device's own
        // declaration being narrower than the texture it describes. Name all
        // three once per distinct triple so the reading is the diagnosis.
        if refusal == GpuPlaneRefusal::FormatDiffers {
            let discriminant = (u64::from(src.pixel_format) << 32)
                | (u64::from(plane.pixel_format) << 16)
                | u64::from(window.map_or(u16::MAX, |w| w.pixel_format));
            if crate::observe::first_sight("sl_gpu_format_differs", discriminant) {
                crate::observe::fail(format!(
                    "blit_gpu_plane reason=sl_gpu_format_differs src_format={} \
                     dst_format={} mapping_format={} width={} height={}",
                    src.pixel_format,
                    plane.pixel_format,
                    window.map_or(u16::MAX, |w| w.pixel_format),
                    plane.width,
                    plane.height
                ));
            }
        }
        return None;
    }
    let identity = crate::backend::vulkan::gva_identity(&debt);
    match mapping_write::vulkan::write_bgra8_from_resident_gpu(
        state,
        host,
        mapping_id,
        &identity,
        plane.width,
        plane.height,
    ) {
        Ok(_) => {
            note_store_route("sl_gpu_landed");
            Some(BlitStatus::Ok)
        }
        Err(decline) => {
            // A decline is a routing answer, so the counter is the record and the
            // off channel carries which check answered — the engine's format
            // comparison in particular, which is the one that can refuse every
            // payment on one texture and leave a canvas black.
            note_store_route("sl_gpu_engine_declined");
            crate::observe::off(format!(
                "blit_gpu_plane mid={mapping_id} {}x{} decline={decline:?}",
                plane.width, plane.height
            ));
            None
        }
    }
}

/// Why one whole-plane mapper-ref-texture to guest-linear copy is not the GPU arm's, for
/// the terms that can be decided from the two endpoints alone.
///
/// Every variant is a **fall-through and not a loss**: the staging loop runs
/// unchanged and lands the same pixels. They are counters for that reason, and
/// with `t2t_gpu_src_not_resident`, `t2t_gpu_dst_unbounded`,
/// `t2t_gpu_engine_declined` and `t2t_gpu_landed` they partition
/// `blit_t2t_t11_whole_plane`, so a census that does not add up is the bug.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum T2tGvaRefusal {
    /// The source names no mapping this device holds, or one that has not
    /// declared its geometry, so there is no surface identity to ask about.
    NoSurface,
    /// The source is a plane of a larger surface, or disagrees with the mapping
    /// about the surface's size. The resident is keyed by the mapping's own
    /// geometry and this copy lands it whole, so anything but the whole surface
    /// would land it into a window that is not the whole of it.
    SrcNotWholeSurface,
    /// The destination level's own base does not resolve.
    DstOffsetOverflow,
    /// The destination's pitch does not fit the guest's own 32-bit declaration
    /// of one.
    DstStrideWide,
    /// The destination has no byte-copy geometry — the plane's own typed
    /// reason, carried whole so a reading names the same check the copy would
    /// have named.
    DstPlane(crate::runtime::render_writeback::vulkan::GvaWritebackDecline),
    /// The plane runs past the allocation the level lives in.
    DstExtentOob,
}

impl T2tGvaRefusal {
    fn route(&self) -> &'static str {
        match self {
            Self::NoSurface => "t2t_gpu_no_surface",
            Self::SrcNotWholeSurface => "t2t_gpu_src_not_whole_surface",
            Self::DstOffsetOverflow => "t2t_gpu_dst_offset_overflow",
            Self::DstStrideWide => "t2t_gpu_dst_stride_wide",
            Self::DstPlane(_) => "t2t_gpu_dst_plane",
            Self::DstExtentOob => "t2t_gpu_dst_extent_oob",
        }
    }
}

/// The destination plane a whole-plane mapper-ref-texture to guest-linear copy would write,
/// and its span, or the typed reason there is none.
///
/// Everything [`try_copy_t11_plane_to_linear_on_gpu`] can decide before it asks
/// the engine anything or walks the guest's page table, which is also everything
/// about it that a test can reach without a GPU. `surface` is the mapping's own
/// declared geometry and `None` when it has none.
fn gpu_t2t_gva_plane(
    surface: Option<(u32, u32)>,
    src: &MapperRefTexture,
    dst: &LinearTextureLevel,
    destination_ref: u32,
) -> Result<
    (
        crate::runtime::render_writeback::vulkan::GvaPlaneDestination,
        crate::runtime::render_writeback::vulkan::GvaPlaneGeometry,
    ),
    T2tGvaRefusal,
> {
    let Some((sw, sh)) = surface else {
        return Err(T2tGvaRefusal::NoSurface);
    };
    if sw != src.width || sh != src.height || src.surface_offset != 0 || sw == 0 || sh == 0 {
        return Err(T2tGvaRefusal::SrcNotWholeSurface);
    }
    // The destination plane, from the level the blit resolved. Origin zero on
    // both ends is the caller's admission, so this is the level's own base.
    let Some(level_base) = dst.texel_offset(0, 0, 0) else {
        return Err(T2tGvaRefusal::DstOffsetOverflow);
    };
    let Some(target_gva) = dst.base_gva.checked_add(level_base) else {
        return Err(T2tGvaRefusal::DstOffsetOverflow);
    };
    let Ok(row_stride) = u32::try_from(dst.row_stride) else {
        return Err(T2tGvaRefusal::DstStrideWide);
    };
    let plane = crate::runtime::render_writeback::vulkan::GvaPlaneDestination {
        target_gva,
        width: dst.width,
        height: dst.height,
        row_stride,
        format: dst.pixel_format,
        texture_ref: destination_ref,
    };
    // The span to walk, from the destination's own terms, so the licence covers
    // exactly the bytes the copy writes and not one page more.
    let geometry = plane.geometry().map_err(T2tGvaRefusal::DstPlane)?;
    // Against the allocation and not against the span: a copy that runs off the
    // level's own bytes is the class `texture_region_window` bounds the host
    // path with, and this arm owes the same check before it walks anything.
    if !range_fits(level_base, geometry.extent, dst.alloc_size) {
        return Err(T2tGvaRefusal::DstExtentOob);
    }
    Ok((plane, geometry))
}

/// The whole-plane copy out of an IOSurface the GPU already holds, into a
/// guest-linear destination, issued by the GPU.
///
/// # What this is instead of
///
/// The host path below reads the source through the mapping rail and writes the
/// destination through the GVA rail. Reading the source's *guest bytes* is what
/// makes it expensive, and the cost is not the copy: a mapping read must first
/// settle, which pays the surface's writeback debt and then waits for this
/// device's own submitted writes to land in those pages. On a driven macos-13
/// x86 boot that settle was 91 % of the staging window and the memcpy behind it
/// was 4.5 %.
///
/// None of it is owed here. The source's authoritative content is the engine's
/// resident, the destination is a plane of guest pages, and
/// `engine::copy_target_to_guest_pages` moves exactly that — so this arm never
/// touches the source's guest bytes and has nothing to wait for. What the guest
/// asked for is `copyFromTexture:toTexture:`, a blit-encoder command with no
/// host visibility; making the source CPU-readable is `synchronizeResource:`,
/// which is a different call the guest did not make.
///
/// # Why it is only the whole plane
///
/// `copy_target_to_guest_pages` takes no source rectangle: it copies level 0 of
/// the resident whole, at origin zero. So a partial rect on either end is not
/// this arm's, and the caller's census — which partitions the population — is
/// what says how much of it that leaves. On the boot above it left all of it:
/// 511 of 511.
///
/// Returns `None` for every fall-through, having named it on a counter. The
/// caller then runs the host path unchanged, so nothing here can lose a frame —
/// only spend one.
pub(crate) fn try_copy_t11_plane_to_linear_on_gpu<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    destination_ref: u32,
    src: &MapperRefTexture,
    dst: &LinearTextureLevel,
) -> Option<BlitStatus> {
    use crate::runtime::drain::note_store_route;
    let surface = state
        .mappings
        .get(&src.mapping_id)
        .filter(|m| m.has_geom)
        .map(|m| (m.width, m.height));
    let (plane, geometry) = match gpu_t2t_gva_plane(surface, src, dst, destination_ref) {
        Ok(v) => v,
        Err(refusal) => {
            note_store_route(refusal.route());
            return None;
        }
    };
    let identity = crate::backend::vulkan::present_identity::surface_identity(
        state,
        src.mapping_id,
        src.width,
        src.height,
    );
    if matches!(
        crate::backend::vulkan::engine::resident_content_backing(&identity),
        crate::backend::vulkan::engine::ResidentContentBacking::NotReady
    ) {
        // The source's bytes are its guest pages' bytes already, so the host
        // path is reading what it should and is the cheap arm rather than the
        // wasteful one.
        note_store_route("t2t_gpu_src_not_resident");
        return None;
    }
    // The destination's pages, captured once. The host path's `dest_window`
    // takes the same walk for the same reason — the guest's vCPUs run
    // throughout, so the licence must be the walk the command itself was
    // authorised by rather than whatever the address names later.
    let gpas = gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        plane.target_gva,
        geometry.extent,
        state.page_shift,
    );
    if gpas.is_empty() {
        note_store_route("t2t_gpu_dst_unbounded");
        return None;
    }
    let pages = crate::runtime::draw::StoreTargetPages::from_ordered(&gpas, geometry.extent);
    match crate::runtime::render_writeback::vulkan::copy_resident_into_gva_plane(
        state,
        host,
        task_id,
        &identity,
        &plane,
        Some(&pages),
    ) {
        Ok(_) => {
            note_store_route("t2t_gpu_landed");
            Some(BlitStatus::Ok)
        }
        Err(decline) => {
            note_store_route("t2t_gpu_engine_declined");
            crate::observe::off(format!(
                "blit_gpu_gva mid={} {}x{} decline={decline:?}",
                src.mapping_id, dst.width, dst.height
            ));
            None
        }
    }
}

/// Whether this rail already holds a mapper-ref-texture surface's content, as a counter.
///
/// A census and nothing else: it changes no decision, and every caller runs the
/// same host copy afterwards either way. The Metal rail has no resident registry
/// to ask, so [`crate::backend::Backend::note_blit_t11_resident`] does nothing
/// there — an absent reading, not a `blit_t11_resident_not_ready` this rail
/// would have to read as a real one.
pub(crate) fn note_blit_t11_resident(state: &DeviceState, mapping_id: u32) {
    // This census asks the engine a question, and asking takes the engine
    // lock — the same lock the draw rail holds while it encodes and submits.
    // A probe that blocks is not a probe, so time it: if this reads anywhere
    // near `walk_blit_us`, the blit rail's cost is this instrument waiting
    // for the renderer rather than anything the blit itself does.
    let probe_started = std::time::Instant::now();
    let _probe = ProbeClock(probe_started);
    struct ProbeClock(std::time::Instant);
    impl Drop for ProbeClock {
        fn drop(&mut self) {
            crate::runtime::drain::note_store_route_us(
                "blit_resident_probe_us",
                self.0.elapsed().as_micros() as u64,
            );
        }
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return;
    };
    if !m.has_geom || m.width == 0 || m.height == 0 {
        crate::runtime::drain::note_store_route("blit_t11_resident_no_geom");
        return;
    }
    let (w, h) = (m.width, m.height);
    let id = crate::backend::vulkan::present_identity::surface_identity(state, mapping_id, w, h);
    crate::runtime::drain::note_store_route(
        match crate::backend::vulkan::engine::resident_content_backing(&id) {
            crate::backend::vulkan::engine::ResidentContentBacking::NotReady => {
                "blit_t11_resident_not_ready"
            }
            _ => "blit_t11_resident_ready",
        },
    );
}

/// This rail's own checks.
///
/// Here rather than in `super::tests` because they reach `gpu_t2t_gva_plane`,
/// which is this module's working part — reaching it from a sibling would mean
/// widening it so a test could see it.
#[cfg(test)]
mod tests {
    use super::*;

    /// The GPU arm for a whole-plane mapper-ref-texture source going to a guest-linear
    /// destination, decided from the two endpoints alone.
    ///
    /// The staging loop this stands in front of reads the source's *guest bytes*,
    /// and a mapping read must first settle: pay the surface's writeback debt and
    /// then wait for this device's own submitted writes to land. That wait, not the
    /// memcpy behind it, is what the rail costs. The GPU arm owes the source no
    /// settle at all, because the resident is the content — so every term below is
    /// about whether the resident and the destination plane are the two ends of one
    /// byte copy, and never about whether the guest's pages are readable.
    #[test]
    fn a_whole_surface_mapper_ref_texture_source_reaches_the_destinations_own_guest_plane() {
        use super::T2tGvaRefusal::*;

        const W: u32 = 64;
        const H: u32 = 32;
        const BPR: u64 = 256;
        let src = MapperRefTexture {
            mapping_id: 7,
            width: W,
            height: H,
            surface_offset: 0,
            row_stride: BPR as u32,
            span_end: BPR * H as u64,
            bpp: 4,
            pixel_format: MTL_FORMAT_BGRA8_UNORM,
        };
        let dst = LinearTextureLevel {
            base_gva: 0x4000,
            alloc_size: BPR * H as u64,
            level_offset: 0,
            row_stride: BPR,
            slice_stride: 0,
            slice_index: 0,
            width: W,
            height: H,
            depth: 1,
            bpp: 4,
            // This tree carries the block grid the v6 branch predates. Bgra8 is a
            // 1x1 grid, so it agrees with `bpp` and the whole-plane licence under
            // test is unaffected by it.
            block: crate::protocol::pixel_format::block_geometry(MTL_FORMAT_BGRA8_UNORM)
                .expect("bgra8 has a grid"),
            pixel_format: MTL_FORMAT_BGRA8_UNORM,
        };

        let (plane, geometry) =
            gpu_t2t_gva_plane(Some((W, H)), &src, &dst, 9).expect("one whole plane into another");
        assert_eq!(
            plane.target_gva, 0x4000,
            "the level's own base is the plane"
        );
        assert_eq!(
            plane.texture_ref, 9,
            "the destination is the resource whose host-side pixel caches this invalidates"
        );
        assert_eq!(
            geometry.extent,
            (H as u64 - 1) * BPR + W as u64 * 4,
            "the span is the copy's own bytes and not the row padding after the last of them"
        );

        assert_eq!(
            gpu_t2t_gva_plane(None, &src, &dst, 9).unwrap_err(),
            NoSurface,
            "a mapping with no declared geometry has no surface identity to ask about"
        );
        // The resident is keyed by the mapping's own geometry and this copy lands it
        // whole, so a source that is one plane of a larger surface would land the
        // whole resident into a window that is not the whole of it.
        assert_eq!(
            gpu_t2t_gva_plane(
                Some((W, H)),
                &MapperRefTexture {
                    surface_offset: 0x8000,
                    ..src
                },
                &dst,
                9
            )
            .unwrap_err(),
            SrcNotWholeSurface,
            "a second plane of a biplanar surface is not the resident"
        );
        assert_eq!(
            gpu_t2t_gva_plane(Some((W, H * 2)), &src, &dst, 9).unwrap_err(),
            SrcNotWholeSurface,
            "a texture that disagrees with its mapping about the surface's size is not the resident"
        );
        // The class `texture_region_window` bounds the staging loop with: a copy that
        // runs off its resource paints whatever the guest handed those pages to next.
        assert_eq!(
            gpu_t2t_gva_plane(
                Some((W, H)),
                &src,
                &LinearTextureLevel {
                    alloc_size: BPR * H as u64 - 1,
                    ..dst
                },
                9
            )
            .unwrap_err(),
            DstExtentOob,
            "a plane running past its allocation is refused before anything is walked"
        );
        // Carried whole rather than collapsed, so the reading names the same check
        // the copy itself would have named.
        assert!(
            matches!(
                gpu_t2t_gva_plane(
                    Some((W, H)),
                    &src,
                    &LinearTextureLevel {
                        row_stride: 3,
                        ..dst
                    },
                    9
                ),
                Err(DstPlane(
                    crate::runtime::render_writeback::vulkan::GvaWritebackDecline::PitchNotTexels { .. }
                ))
            ),
            "a pitch that is not a whole number of texels is the plane's own refusal"
        );
    }
}
