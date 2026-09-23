//! The rail-neutral vocabulary of the host-owned presentation window
//! ([[host-window]]).
//!
//! The window is one native surface, one event loop and one aspect-fit
//! viewport, and none of those depend on which rail draws into it. What *is*
//! rail-specific is exactly three things: how a surface is created for the
//! native handle, what a GPU-resident frame is called, and why a present was
//! refused. This module names those three so that
//! [`crate::host_window::present`] can drive a presenter without ever
//! mentioning a rail, and [`super::Backend`] can carry them as ordinary trait
//! methods.
//!
//! Everything here is `host-window`-gated, which is the lawful question for a
//! `cfg`: whether this build compiled a window at all is a fact about the build.
//! Whether the *running* rail can fill one is [`super::Backend::presents_host_window`],
//! and it is a run-time answer — a `--backend both` binary compiles this module
//! for its Metal boot and its Vulkan boot alike.

/// Where a guest frame lands inside a window, and where a window position lands
/// inside a guest frame.
///
/// Here rather than beside the window, because both rails' presenters need it
/// and `backend` may not reach up into [`crate::host_window`] — a lower layer
/// importing from a higher one is how the two halves of "presentation and
/// pointer move as one unit" once ended up compiled for different platforms.
/// The window reaches down for the same functions, which is the direction that
/// works.
pub mod viewport;

use crate::observe::Decline;

/// The native surface a rail attaches its presenter to.
///
/// The handles come from `winit` and are only valid while the window that
/// vended them is alive, which is why attach and detach are both driven from
/// the window's own lifecycle callbacks rather than from the device.
#[derive(Clone, Copy)]
pub struct WindowSurface {
    pub display: raw_window_handle::RawDisplayHandle,
    pub window: raw_window_handle::RawWindowHandle,
    pub width: u32,
    pub height: u32,
}

/// A published frame's CPU bytes, offered to a rail's presenter.
///
/// The presenter prefers a [`WindowResident`] and reads these only when no
/// resident carries the display — the firmware framebuffer, and any mapping the
/// compositor has not rendered into. `bgra` is empty on presents the device
/// elided the readback for, and a presenter must reject a short buffer rather
/// than blit a torn frame.
#[derive(Clone, Copy, Debug)]
pub struct WindowCpuFrame<'a> {
    pub bgra: &'a [u8],
    pub width: u32,
    pub height: u32,
    /// Publish sequence of the frame these bytes came from. A presenter that
    /// stages the bytes keeps the last sequence it uploaded, so a forced redraw
    /// (resize, self-heal) re-blits without re-copying a framebuffer that has
    /// not changed.
    pub seq: u64,
}

impl WindowCpuFrame<'_> {
    /// Whether these bytes hold every pixel of the geometry they claim.
    ///
    /// A short buffer is not a degraded frame, it is a torn one: the blit would
    /// show whatever the staging surface held below the copied rows, which is
    /// the previous frame at whatever geometry it had. A value test, so both
    /// rails reject the same frames for the same reason instead of each
    /// spelling a length check inside its own upload.
    pub fn complete(&self) -> bool {
        let need = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        need != 0 && self.bgra.len() >= need
    }
}

/// A rail-owned handle to the GPU-resident frame the window may present without
/// the pixels crossing host memory.
///
/// Opaque to the window: it is produced by [`super::Backend::window_resident`]
/// on the drain worker, parked in the frame slot, and handed back to
/// [`super::Backend::window_present`] on the window thread. Nothing between
/// those two points looks inside it.
///
/// An enum over the compiled rails rather than a `dyn` object, for the reason
/// [`super::SelectedBackend`] is one: a `--backend both` binary has a closed set
/// of rails, and naming them keeps the resident's type — and therefore its
/// lifetime — visible to the rail that owns it.
///
/// The Metal rail contributes no variant, and that is a statement rather than
/// an omission: it retains no texture past the encode that produced it — every
/// colour target in `backend::metal::render` is created, CPU-seeded, read back
/// and dropped inside one `render_core_mrt` — so there is nothing for a present
/// to name. On a Metal-only build this enum is therefore uninhabited and
/// `Option<WindowResident>` is provably `None`, which is the same fact the type
/// system can check. Give Metal a resident registry and it gets a variant here.
#[derive(Clone)]
pub enum WindowResident {
    #[cfg(feature = "backend-vulkan")]
    Vulkan(crate::backend::vulkan::engine::WindowPresentSource),
}

/// What a completed present did, in terms the window acts on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowPresentOutcome {
    /// The rail had no drawable free. Nothing was consumed; the window retries
    /// within its redraw backstop.
    Busy,
    Presented {
        /// The pixels came from a [`WindowResident`] rather than from host
        /// memory. Reported so a boot can be read for whether the zero-copy
        /// path is actually carrying the desktop.
        direct: bool,
        width: u32,
        height: u32,
        /// How many drawables the rail is cycling through. A count, not a
        /// swapchain: Metal's layer and Vulkan's swapchain both have one.
        buffers: usize,
        /// The surface reported that its drawables no longer match the window,
        /// so a rebuild is armed. The window must schedule another redraw
        /// promptly instead of waiting for the next guest frame — boot-era
        /// presents can be seconds apart, which would leave a mismatched
        /// drawable on screen for that long.
        suboptimal: bool,
    },
}

/// Why a rail could not put a frame on the screen.
///
/// Two variants, because the window acts on exactly one distinction: a
/// presenter that is *gone* gets rebuilt, and every other refusal is named and
/// dropped. Which side a rail's own refusal falls on is the rail's answer —
/// only Vulkan knows that a `VK_ERROR_DEVICE_LOST` destroyed the swapchain, and
/// only Metal knows that its layer outlived the view it was attached to — so
/// the rails construct this, and the window never matches on a rail's type.
pub enum WindowDecline {
    /// This rail no longer has a presenter for this window. On a running window
    /// that means the rail's device was lost and took the presenter with it;
    /// the window rebuilds it, bounded.
    PresenterLost(WindowDeclineReason),
    /// The rail refused this present, and has named itself.
    Refused(WindowDeclineReason),
}

/// The rail's own refusal, in the rail's own vocabulary.
///
/// Every arm delegates [`Decline::owner`] as well as [`Decline::slug`], because
/// an outer enum that claims its inner type's slug reads as a second claimant
/// for one check — see the `Decline` trait's own note.
pub enum WindowDeclineReason {
    /// This rail compiled no presenter for the host window.
    ///
    /// Unreachable while [`super::Backend::presents_host_window`] is the
    /// question that decides whether a window opens at all — which is exactly
    /// why it is a named refusal rather than a panic or a silent `Ok`: if the
    /// two answers ever disagree, the log says which rail was asked.
    RailHasNoPresenter,
    /// The frame slot held a resident produced by the other rail. One process
    /// runs one rail, so this is a wiring defect rather than guest work, and it
    /// is typed instead of a panic because it reaches the window thread.
    ResidentFromOtherRail,
    #[cfg(feature = "backend-vulkan")]
    Vulkan(crate::backend::vulkan::engine::DrawError),
    #[cfg(feature = "backend-metal")]
    Metal(crate::backend::metal::window::MetalWindowDecline),
}

impl WindowDecline {
    /// The rail's reason, whichever disposition it carries.
    pub fn reason(&self) -> &WindowDeclineReason {
        match self {
            Self::PresenterLost(reason) | Self::Refused(reason) => reason,
        }
    }

    /// Whether the window must rebuild this rail's presenter before it can
    /// present again.
    pub fn presenter_lost(&self) -> bool {
        matches!(self, Self::PresenterLost(_))
    }
}

impl Decline for WindowDecline {
    fn slug(&self) -> &'static str {
        self.reason().slug()
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        self.reason().fields()
    }

    fn owner(&self) -> &'static str {
        self.reason().owner()
    }
}

impl Decline for WindowDeclineReason {
    fn slug(&self) -> &'static str {
        match self {
            Self::RailHasNoPresenter => "window_rail_has_no_presenter",
            Self::ResidentFromOtherRail => "window_resident_other_rail",
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(error) => error.slug(),
            #[cfg(feature = "backend-metal")]
            Self::Metal(error) => error.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::RailHasNoPresenter | Self::ResidentFromOtherRail => Vec::new(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(error) => error.fields(),
            #[cfg(feature = "backend-metal")]
            Self::Metal(error) => error.fields(),
        }
    }

    fn owner(&self) -> &'static str {
        match self {
            Self::RailHasNoPresenter | Self::ResidentFromOtherRail => std::any::type_name::<Self>(),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan(error) => error.owner(),
            #[cfg(feature = "backend-metal")]
            Self::Metal(error) => error.owner(),
        }
    }
}

crate::observe::decline_display!(WindowDecline);
crate::observe::decline_display!(WindowDeclineReason);

impl std::fmt::Debug for WindowDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for WindowDecline {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A presenter's upload copies `height` rows of `width * 4` bytes out of the
    /// published buffer. A buffer short of that would read whatever the staging
    /// surface held below the copied rows — the previous frame, at whatever
    /// geometry it had — and blit the result as though it were current.
    ///
    /// The short case is not hypothetical: every present the device elides the
    /// readback for publishes an EMPTY buffer, because the resident is carrying
    /// that frame. Those arrive here whenever the resident then turns out not to
    /// be presentable, which is exactly when the fallback runs.
    #[test]
    fn a_cpu_frame_shorter_than_its_own_geometry_is_refused() {
        let full = vec![0u8; 8 * 4 * 4];
        assert!(WindowCpuFrame::complete(&WindowCpuFrame {
            bgra: &full,
            width: 8,
            height: 4,
            seq: 1,
        }));
        // Slop is fine — the copy reads exactly what the geometry names.
        assert!(WindowCpuFrame::complete(&WindowCpuFrame {
            bgra: &full,
            width: 8,
            height: 3,
            seq: 1,
        }));
        assert!(
            !WindowCpuFrame::complete(&WindowCpuFrame {
                bgra: &full[..full.len() - 1],
                width: 8,
                height: 4,
                seq: 1,
            }),
            "one byte short is still a torn last row"
        );
        assert!(
            !WindowCpuFrame::complete(&WindowCpuFrame {
                bgra: &[],
                width: 8,
                height: 4,
                seq: 1,
            }),
            "the elided-readback publish carries no bytes at all"
        );
        assert!(
            !WindowCpuFrame::complete(&WindowCpuFrame {
                bgra: &full,
                width: 0,
                height: 4,
                seq: 1,
            }),
            "a zero dimension names no pixels and blits nothing"
        );
    }

    /// The disposition is what the window reads, and it survives the reason.
    #[test]
    fn only_a_lost_presenter_asks_the_window_to_rebuild() {
        let lost = WindowDecline::PresenterLost(WindowDeclineReason::ResidentFromOtherRail);
        let refused = WindowDecline::Refused(WindowDeclineReason::ResidentFromOtherRail);
        assert!(lost.presenter_lost());
        assert!(!refused.presenter_lost());
        // Both render the rail's reason, so the disposition never becomes a
        // second `reason=` the reader has to reconcile.
        assert_eq!(lost.slug(), "window_resident_other_rail");
        assert_eq!(lost.slug(), refused.slug());
        assert_eq!(lost.to_string(), "reason=window_resident_other_rail");
    }

    /// The two neutral reasons are two checks, so they are two slugs.
    ///
    /// Neither is reachable while both compiled rails implement the window
    /// methods and refuse a resident that is not theirs — which is why they are
    /// asserted here rather than trusted: a slug shared with the arm beside it
    /// is invisible until the day one of them fires, and the whole point of the
    /// pair is telling "this rail has no presenter" from "the publisher and the
    /// presenter disagree about which rail is running".
    #[test]
    fn the_neutral_window_reasons_do_not_share_a_slug() {
        let no_presenter = WindowDeclineReason::RailHasNoPresenter;
        let other_rail = WindowDeclineReason::ResidentFromOtherRail;
        assert_ne!(no_presenter.slug(), other_rail.slug());
        assert_eq!(no_presenter.owner(), other_rail.owner());
        for reason in [no_presenter, other_rail] {
            assert!(reason.fields().is_empty(), "{reason} names no value");
            assert!(reason.slug().starts_with("window_"));
        }
    }
}
