//! The encoder one rail holds open across a compute segment.
//!
//! Here and not in `runtime::compute_session` for the reason [`super::Backend`]
//! itself is here: this is the one type in the device whose *shape* is neutral
//! and whose *contents* are one rail's, so naming both rails is this layer's
//! job and nobody else's. The runtime module beside it owns the segment's
//! sequencing — when a session opens, what latches a block, when it commits —
//! and never learns which rail is behind the handle it is sequencing.

use crate::model::DeviceState;
use crate::runtime::compute_exec::{ComputeAccum, ComputeStatus};
use crate::runtime::decode::compute_spi::Command as ComputeCommand;
use crate::runtime::host::{HostMemory, HostOps};

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::compute_session::metal;

/// An encoder a rail holds open across one compute segment.
///
/// **One shape on every arm.** This was two structs sharing a name — eleven
/// fields on the Metal arm, one on the Vulkan arm — which is how a type starts
/// disagreeing with itself across a feature boundary, and why a build could
/// carry only one of them. What the segment needs is neutral: something it can
/// hold, hand to a record, and finish. What is behind it is the rail's.
pub struct ComputeSession {
    rail: SessionRail,
}

/// The rail half of an open session.
///
/// One variant per rail that can hold something open, and Vulkan is not one:
/// `engine::execute_compute_request` is one-shot per dispatch, so that rail
/// refuses at [`crate::backend::Backend::open_compute_session`] and no session
/// is ever built on it.
///
/// [`SessionRail::Unbacked`] therefore carries [`std::convert::Infallible`] —
/// a value nothing can produce, so the variant is not a state a session can be
/// in and the arms below that name it cannot run. It exists for one mechanical
/// reason: without it a build carrying no session-capable rail would make this
/// enum empty, and `&mut` to an uninhabited type is still something every
/// method has to match on (`references are always considered inhabited`).
enum SessionRail {
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    Metal(metal::MetalSession),
    /// No rail. Unconstructible; see the enum's doc.
    #[allow(
        dead_code,
        reason = "unconstructible by design — it carries an Infallible, and its \
                  job is to keep this enum non-empty on a build with no \
                  session-capable rail"
    )]
    Unbacked(std::convert::Infallible),
}

impl ComputeSession {
    /// Wrap a Metal encoder as this segment's session.
    ///
    /// Crate-private and rail-specific on purpose: the only caller is
    /// [`crate::backend::Backend::open_compute_session`], which is what makes
    /// "every session in this process belongs to the process's rail" true by
    /// construction rather than by convention.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    pub(crate) fn from_metal(rail: metal::MetalSession) -> Self {
        Self {
            rail: SessionRail::Metal(rail),
        }
    }

    /// This session's Metal encoder, if that is the rail behind it.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    pub(crate) fn metal_mut(&mut self) -> Option<&mut metal::MetalSession> {
        match &mut self.rail {
            SessionRail::Metal(rail) => Some(rail),
            SessionRail::Unbacked(_) => None,
        }
    }

    /// How deep the guest's control-flow records currently nest.
    ///
    /// Kept by the rail rather than beside it: the depth is only ever moved by
    /// the encode that opens or closes a block, and a copy up here would be a
    /// second spelling of the same counter.
    // Read only by this module's tests, and only on a rail that can open a
    // session — so a build with no such rail compiles them and runs neither.
    #[cfg(test)]
    #[cfg_attr(
        not(all(feature = "backend-metal", target_os = "macos")),
        allow(dead_code)
    )]
    pub(crate) fn control_depth(&self) -> i32 {
        match &self.rail {
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            SessionRail::Metal(rail) => rail.control_depth,
            SessionRail::Unbacked(_) => 0,
        }
    }

    /// Dispatches encoded on this session whose writeback is still deferred.
    #[cfg(test)]
    #[cfg_attr(
        not(all(feature = "backend-metal", target_os = "macos")),
        allow(dead_code)
    )]
    pub(crate) fn deferred_writeback_count(&self) -> usize {
        match &self.rail {
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            SessionRail::Metal(rail) => rail.nested_jobs.len(),
            SessionRail::Unbacked(_) => 0,
        }
    }

    /// Encode one guest control-flow record onto the open encoder.
    pub fn encode_control<M: HostMemory + HostOps>(
        &mut self,
        state: &DeviceState,
        host: &M,
        task_id: u32,
        cmd: &ComputeCommand,
    ) -> ComputeStatus {
        match &mut self.rail {
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            SessionRail::Metal(rail) => rail.encode_control(state, host, task_id, cmd),
            SessionRail::Unbacked(_) => {
                let _ = (state, host, task_id, cmd);
                ComputeStatus::NoMetal("compute_control_no_session_rail")
            }
        }
    }

    /// Materialize and execute an indirect command buffer on the open encoder.
    pub fn encode_icb<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        cmd: &ComputeCommand,
        acc: &ComputeAccum,
    ) -> ComputeStatus {
        if cmd.indirect_command_buffer_ref == 0 {
            return ComputeStatus::MissingBuffer("compute_icb_ref_zero");
        }
        match &mut self.rail {
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            SessionRail::Metal(rail) => rail.encode_icb(state, host, task_id, cmd, acc),
            SessionRail::Unbacked(_) => {
                let _ = (state, host, task_id, cmd, acc);
                ComputeStatus::NoMetal("compute_icb_no_session_rail")
            }
        }
    }

    /// Commit the segment's work and land everything it deferred.
    ///
    /// By value: a session is finished exactly once, and the nested dispatches
    /// it holds cannot read their own output back until it is.
    pub fn finish<M: HostMemory + HostOps>(
        self,
        host: &mut M,
        state: &mut DeviceState,
        task_id: u32,
    ) -> ComputeStatus {
        match self.rail {
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            SessionRail::Metal(rail) => rail.finish(host, state, task_id),
            SessionRail::Unbacked(_) => {
                let _ = (host, state, task_id);
                ComputeStatus::NoMetal("compute_finish_no_session_rail")
            }
        }
    }
}
