//! The `Backend` trait implementation for device lifecycle.
//!
//! Probing the device is [`super::runtime`]'s job, not this module's. Two
//! wrappers here used to say otherwise: a `MetalRuntime` unit struct whose one
//! associated function forwarded `system_device`, and a `system_device_name`
//! that forwarded the identically-named function it imported. Neither was
//! constructed or called anywhere outside this file's own test, and the second
//! put one name on two functions in two modules — so a `grep` for it reported
//! two producers and the arm a reader landed on was arbitrary.

use crate::backend::compute_session::ComputeSession;
use crate::backend::metal::runtime::system_device;
use crate::backend::{Backend, CensusSite, MipmapGeneration, Rail};
use crate::model::{DeviceInfoLimits, DeviceState};
use crate::protocol::mipmap::MetalMipmapError;
use crate::runtime::compute_exec::{self, ComputeAccum, ComputeStatus};
use crate::runtime::compute_session;
use crate::runtime::draw::{self, DrawEncodeRequest, EncodeStatus};
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mipmap::MipmapStatus;
use reims_vgpu_protocol::compute::DispatchType;
use reims_vgpu_protocol::decode::compute::DispatchRecord;

/// The Metal rail's [`Backend`] handle.
///
/// Fieldless, because there is no per-device Metal state to hold: the
/// `MTLDevice` is [`system_device`]'s process-global `OnceCell` and the command
/// queues are thread-locals beside it. This carried a `ready: bool` that nothing
/// read, kept only because constructing it was what first created that
/// `MTLDevice` — a side effect hidden in a constructor, which is why the probe
/// is now [`Self::probe`] and says so in its name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalBackend;

impl MetalBackend {
    /// Bring up the process's `MTLDevice` and report whether the host has one.
    ///
    /// The probe is the structural capability a build carrying both rails
    /// selects on — "this host can execute Metal" — and it is measured, never
    /// inferred from a device name. On a Metal-only build the answer cannot
    /// change what runs, so it is recorded and the handle is returned either
    /// way; refusing here would replace "the draw found no Metal device" with a
    /// failure at device create, which names the wrong thing.
    pub fn probe() -> Self {
        if system_device().is_none() {
            crate::observe::fail(
                "backend_probe reason=metal_no_system_device \
                 (this host exposes no MTLDevice)",
            );
        }
        Self
    }

    /// Whether this host exposes an `MTLDevice` at all.
    pub fn available() -> bool {
        system_device().is_some()
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        Rail::Metal.name()
    }

    fn present_resident_carries(
        &self,
        state: &DeviceState,
        mapping: u32,
        width: u32,
        height: u32,
    ) -> Option<bool> {
        crate::runtime::scanout::metal::present_resident_carries(state, mapping, width, height)
    }

    fn try_capture_from_resident(
        &self,
        state: &mut DeviceState,
        buf: &mut Vec<u8>,
        mapping_id: u32,
        width: u32,
        height: u32,
    ) -> bool {
        crate::runtime::scanout::metal::try_capture_from_resident(
            state, buf, mapping_id, width, height,
        )
    }

    fn published_frame_rgba8(
        &self,
        _state: &DeviceState,
        mapping_id: u32,
        width: u32,
        height: u32,
        generation: u64,
    ) -> Option<Vec<u8>> {
        crate::backend::metal::resident::read_published_rgba8(
            &crate::backend::metal::resident::ResidentColorKey::for_surface(
                mapping_id, width, height,
            ),
            generation,
        )
    }

    fn reset(&self) {
        crate::runtime::icb::clear_icb_cache();
    }

    fn forget_host_icbs(&self) {
        crate::runtime::icb::metal::clear_host_icb_cache();
    }

    fn encode_draw_chain<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        req: &mut DrawEncodeRequest,
        writeback_guest: bool,
        force_full_store: bool,
    ) -> (EncodeStatus, Option<Vec<u8>>) {
        draw::metal::encode_draw_chain(state, host, req, writeback_guest, force_full_store)
    }

    fn execute_dispatch<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        acc: &ComputeAccum,
        dispatch: &DispatchRecord,
    ) -> ComputeStatus {
        compute_exec::metal::execute_dispatch_metal(state, host, task_id, acc, dispatch, None)
    }

    #[allow(clippy::result_large_err, reason = "see the `Backend` declaration")]
    fn open_compute_session(
        &self,
        dispatch_type: DispatchType,
    ) -> Result<ComputeSession, ComputeStatus> {
        compute_session::metal::MetalSession::open(dispatch_type).map(ComputeSession::from_metal)
    }

    fn execute_dispatch_nested<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        acc: &ComputeAccum,
        dispatch: &DispatchRecord,
        session: &mut ComputeSession,
    ) -> ComputeStatus {
        // `None` cannot happen: `backend::selected()` is latched, so every
        // session in this process was opened by this rail. Named rather than
        // unwrapped, because a panic must never cross the QEMU FFI boundary.
        let Some(rail) = session.metal_mut() else {
            return ComputeStatus::NoMetal("compute_nested_session_not_metal");
        };
        compute_exec::metal::execute_dispatch_metal(state, host, task_id, acc, dispatch, Some(rail))
    }

    fn encode_icb_execute_and_writeback<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        req: &DrawEncodeRequest,
        icb_ref: u32,
        range_location: u64,
        range_length: u64,
    ) -> EncodeStatus {
        draw::metal::encode_icb_execute_and_writeback(
            state,
            host,
            req,
            icb_ref,
            range_location,
            range_length,
        )
    }

    /// This rail serves an Apple GPU to an Apple guest, so the table's own
    /// values already describe the executing device and there is nothing to
    /// reduce. Saturating rather than reflecting is deliberate: every one of
    /// these keys bounds what the guest's *own* Metal will then ask this device
    /// to run, and the guest's Metal is the same framework version running on
    /// the same silicon, so a smaller number here would refuse work the host
    /// can execute. The rail that has to reduce is the one whose host GPU is
    /// not the guest's — see [`crate::backend::vulkan::VulkanBackend`].
    fn device_info_limits(&self) -> DeviceInfoLimits {
        DeviceInfoLimits {
            max_sample_count: u32::MAX,
            d24_stencil8: true,
            max_threads_per_threadgroup: [u32::MAX; 3],
            max_threadgroup_memory_bytes: u32::MAX,
            native_fp16: true,
        }
    }

    /// Apple GPUs report 1024 and 32 across every family the arm64 pathway
    /// targets, and by the argument in [`Self::device_info_limits`] the GPU
    /// behind this rail is one of them.
    fn compute_threadgroup_limits(&self) -> (u32, u32) {
        (1024, 32)
    }

    /// This rail's caches are its own and are not held in the device's rail
    /// slot, so the device is not read here. It is in the signature because the
    /// Vulkan rail's caches *are*, and the trait carries the more demanding of
    /// the two.
    fn emit_census(&self, _state: &crate::model::DeviceState, site: CensusSite) {
        // One line, at one site. The other three are engine counters, phase
        // windows and a mutex census that this rail has no counterpart for —
        // absent rather than zeroed, so a reader cannot mistake "no such engine"
        // for "an idle one".
        if site == CensusSite::Levels {
            super::census::emit_object_cache_levels();
            super::census::emit_resident_color_levels();
        }
    }

    /// This rail keys its retained colour render targets by mapping id, so a
    /// mapping that stops naming its surface takes them with it. Correctness
    /// does not depend on it — a retained target can only be loaded from under
    /// the surface cache generation it was published at, and an entry that is
    /// gone issues no generation — so this is about the bytes.
    fn forget_mapping(&self, mapping_id: u32) {
        super::resident::forget(mapping_id);
    }

    fn generate_mipmap_chain(
        &self,
        texture_ref: u32,
        fmt: u16,
        width: u32,
        height: u32,
        levels: u32,
        level0: &[u8],
    ) -> MipmapGeneration {
        match super::mipmap::generate_mipmaps_filtered(fmt, width, height, levels, level0) {
            Ok(chain) => MipmapGeneration::Chain(
                chain
                    .into_iter()
                    .map(|level| (level.width, level.height, level.tight_bytes))
                    .collect(),
            ),
            // Correct but slower: let the caller run the shared box filter, and
            // make the missing device visible as a typed degradation. This is
            // the *only* error that declines rather than refuses — every other
            // one means the filtered path was available and rejected the work.
            Err(error @ MetalMipmapError::NoDevice) => {
                crate::observe::Emit::decline("mipmap_metal_fallback", &error)
                    .field("texture", texture_ref)
                    .field("format", format!("{fmt:#x}"))
                    .field("width", width)
                    .field("height", height)
                    .off();
                MipmapGeneration::Unfiltered
            }
            Err(error) => MipmapGeneration::Refused(MipmapStatus::Metal(error)),
        }
    }

    #[cfg(feature = "host-window")]
    fn presents_host_window(&self) -> bool {
        // The drawable half of that window is `super::window`, so this rail can
        // fill one. Whether this *build* compiled a window at all is a
        // different question with a different owner — `device::window_publish`,
        // where the `host-window` feature is asked.
        true
    }

    #[cfg(feature = "host-window")]
    fn window_attach(
        &self,
        surface: &crate::backend::window::WindowSurface,
    ) -> Result<(), crate::backend::window::WindowDecline> {
        super::window::attach(surface).map_err(window_decline)
    }

    #[cfg(feature = "host-window")]
    fn window_attached(&self) -> bool {
        super::window::attached()
    }

    #[cfg(feature = "host-window")]
    fn window_present(
        &self,
        resident: Option<&crate::backend::window::WindowResident>,
        cpu: Option<crate::backend::window::WindowCpuFrame<'_>>,
    ) -> Result<crate::backend::window::WindowPresentOutcome, crate::backend::window::WindowDecline>
    {
        if resident.is_some() {
            // Unreachable through the publish path — this rail's
            // `window_resident` refuses, so nothing ever parks one for it — and
            // typed rather than ignored because reaching it means the publisher
            // and the presenter disagree about which rail is running, which is
            // exactly the class of defect a `--backend both` binary exists to
            // make visible.
            return Err(crate::backend::window::WindowDecline::Refused(
                crate::backend::window::WindowDeclineReason::ResidentFromOtherRail,
            ));
        }
        super::window::present(cpu).map_err(window_decline)
    }

    #[cfg(feature = "host-window")]
    fn window_resize(&self, width: u32, height: u32) {
        super::window::resize(width, height);
    }

    #[cfg(feature = "host-window")]
    fn window_detach(&self) {
        super::window::detach();
    }

    // The rest of `Backend` takes the trait's defaults, and each default is the
    // accurate statement for this rail rather than a stub:
    //
    // * The two blit fast paths, the resident census, and `window_resident` —
    //   no resident registry to copy out of, to count, or to name a present
    //   from, so the host window takes this rail's CPU frames.
    // * The guest-memory group — this rail's Store is a host copy that has
    //   already executed when it returns, so nothing is ever outstanding, it
    //   holds no alias of guest RAM past the call, and it pins no linear
    //   resident to release.
    // * The cadence pair — nothing is batched or deferred, so there is nothing
    //   for the heartbeat or the drain tail to flush.
}

/// Which disposition one of this rail's window refusals carries.
///
/// The window acts on exactly one distinction — a presenter that is *gone* gets
/// rebuilt — and only this rail knows which of its refusals means that. Losing
/// the presenter is losing the `CAMetalLayer`, and the layer is dropped in
/// exactly one place: `window::detach`, from the window's own `exiting`. Every
/// other refusal leaves a presenter standing and is named and dropped.
#[cfg(feature = "host-window")]
fn window_decline(
    error: super::window::MetalWindowDecline,
) -> crate::backend::window::WindowDecline {
    let lost = matches!(error, super::window::MetalWindowDecline::NotAttached);
    let reason = crate::backend::window::WindowDeclineReason::Metal(error);
    if lost {
        crate::backend::window::WindowDecline::PresenterLost(reason)
    } else {
        crate::backend::window::WindowDecline::Refused(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::runtime::system_device_name;

    /// Named for what it asserts. It was called `system_device`, which shadowed
    /// the imported function of that name inside the test module.
    #[test]
    fn the_probe_finds_a_device_and_the_backend_reports_it_ready() {
        assert!(system_device().is_some());
        assert!(system_device_name().is_some());
        assert!(MetalBackend::available());
        assert_eq!(MetalBackend::probe().name(), "metal");
    }
}
