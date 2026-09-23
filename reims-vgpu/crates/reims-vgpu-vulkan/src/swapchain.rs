//! Choosing a swapchain from what a surface reported, and refusing what it did
//! not offer.
//!
//! # What this owns that the other two presentation modules do not
//!
//! [`reims_vgpu_core::present`] owns the semantic stream: which frames are in
//! flight, what parks when no image is free, and what the old swapchain is
//! still owed. [`crate::frames`] owns the native slots and the binary
//! semaphores that cannot be reused after a failure. Neither decides what the
//! swapchain *is*, and that is one decision made once per surface generation
//! out of five independent capability answers — so it lives here, where it can
//! be made with no device and asserted with no window.
//!
//! # An offered format is not a substitutable one
//!
//! The composition writes the guest's scanout bytes. A surface that does not
//! offer that exact format is refused by name rather than substituted, because
//! every substitution available is wrong in a way nothing downstream can
//! detect:
//!
//! - `B8G8R8A8_SRGB` for `B8G8R8A8_UNORM` applies the sRGB transfer function
//!   when the image is written. The guest composited its frame already; writing
//!   it through an sRGB view encodes it a second time, and the frame comes out
//!   washed out with no error anywhere.
//! - `R8G8B8A8_UNORM` for `B8G8R8A8_UNORM` is the same bits in a different
//!   channel order, so red and blue swap.
//!
//! So [`Refusal::FormatNotOffered`] carries what the surface *did* offer, and
//! the caller either composes into something the surface has or reports the
//! mismatch. That is [`crate::vertex`]'s rule for a declined vertex format
//! reached from the other side: there, a wider sibling is a real substitute and
//! the plan records that it was used; here there is no sibling that preserves
//! the picture, so there is nothing to record and the answer is no.
//!
//! # One more image than the minimum, on purpose — and three under `MAILBOX`
//!
//! With exactly `minImageCount` images, `vkAcquireNextImageKHR` may have to
//! wait for the presentation engine to release one — and waiting on a host
//! inside the path a frame takes is the thing this architecture does not do.
//! One more gives the ring a free slot to acquire into while the engine still
//! holds the last. `maxImageCount` of zero means unbounded, which is a real
//! answer and not a missing one, so the clamp only applies where the surface
//! named a maximum.
//!
//! `MAILBOX` needs a third: one image queued for display, one being drawn, and
//! one for the queued image to be *replaced* with. That replacement is the
//! whole reason to be on the rung — with two images the mode still works and
//! degrades to waiting for the queued frame, which is the behaviour `FIFO`
//! already had. So [`MAILBOX_MIN_IMAGES`] is a floor on that arm only, and the
//! surface's own maximum still wins over it: a surface that caps at two cannot
//! be argued with.
//!
//! The floor is also the depth a presenter can usefully run at, which is why it
//! is public. A presenter with more presents in flight than the swapchain has
//! images has slots that can never be filled.
//!
//! The count requested is still only a request: `vkGetSwapchainImagesKHR`
//! reports what the driver actually made, and that number is the one
//! [`reims_vgpu_core::present::PresentStream::configure`] is told. Nothing here
//! is a bound on anything.
//!
//! # A surface with no area is not ready, and is not refused
//!
//! A minimized window reports a current extent of zero, and a swapchain of zero
//! extent cannot be created. That is not a capability this device lacks and not
//! a guest error: it resolves by itself when the window comes back. So it is
//! [`NotReady`] and not a [`Refusal`] — a refusal is terminal and would leave
//! the surface dead for the session.
//!
//! # Present mode is a two-rung ladder and the bottom rung is guaranteed
//!
//! `FIFO` is the one mode the specification requires every surface to support,
//! so the ladder is total and there is no "no present mode" refusal.
//! `MAILBOX` is the rung above it: same absence of tearing, lower latency,
//! because a frame that arrives before the next vertical blank replaces the
//! pending one instead of queueing behind it.
//!
//! `IMMEDIATE` is deliberately not on the ladder. It tears, and nothing in the
//! guest's presentation contract asks for a torn frame — a mode that trades
//! correctness for latency is not one an operator switch may reach, because
//! [`reims_vgpu_config`] switches may only narrow what the device does. The switch
//! here narrows to `FIFO`, which is the direction that is allowed.

use ash::vk;

/// What the surface answered, in the four questions a swapchain is chosen from.
///
/// Taken as a value rather than queried, for the reason
/// [`crate::census::Reported`] is: this module asks no driver anything, so the
/// whole choice is testable against surfaces that do not exist.
#[derive(Clone, Debug)]
pub struct Surface<'a> {
    /// `vkGetPhysicalDeviceSurfaceSupportKHR` for the queue family this rail
    /// presents on.
    ///
    /// Separate from every other answer here because it is the only one that
    /// is about the *pairing* rather than about the surface: the other three
    /// describe what the surface offers, and this says whether this device's
    /// present queue can address it at all. `VK_KHR_swapchain` being
    /// enumerated does not imply it --- a host whose display is driven by a
    /// different adapter enumerates the extension and supports no family for
    /// the window's surface --- and it cannot be answered by
    /// [`crate::census::Census`], which is taken before any surface exists.
    pub supported_by_present_family: bool,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: &'a [vk::SurfaceFormatKHR],
    pub present_modes: &'a [vk::PresentModeKHR],
}

/// What the composition needs from the swapchain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wanted {
    /// The format the composition writes. Not negotiable; see the module doc.
    pub format: vk::Format,
    /// The size to ask for where the surface lets the swapchain choose. Ignored
    /// where the surface names a current extent, which it usually does.
    pub extent: vk::Extent2D,
    /// Whether the composition copies into the image rather than rendering to
    /// it. A surface whose `supportedUsageFlags` lacks `TRANSFER_DST` refuses.
    pub transfer_destination: bool,
}

/// How far up the present-mode ladder an operator switch allows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Narrowing {
    /// `REIMS_VGPU_SWAPCHAIN_FIFO`: take the bottom rung whatever the surface
    /// offers.
    pub fifo_only: bool,
}

impl Narrowing {
    /// Read the switch. See [`reims_vgpu_config`] for the narrow-never-widen rule.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            fifo_only: reims_vgpu_config::switch(reims_vgpu_config::SWAPCHAIN_FIFO)
                == reims_vgpu_config::Switch::On,
        }
    }
}

/// Why the surface has nothing to build on yet.
///
/// Distinct from [`Refusal`] because it resolves without anything changing
/// about the device or the guest: the window comes back, and the next attempt
/// succeeds. A refusal here would leave the surface dead for the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotReady {
    /// The surface's current extent has a zero dimension. A minimized window.
    NoArea { width: u32, height: u32 },
}

impl NotReady {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoArea { .. } => "vk_swapchain_surface_has_no_area",
        }
    }
}

impl std::fmt::Display for NotReady {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::NoArea { width, height } => {
                write!(f, "{} width={width} height={height}", self.slug())
            }
        }
    }
}

/// Why this surface cannot carry this composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The surface offers no format the composition can write. Carries what it
    /// does offer, because the refusal is only actionable with that list.
    FormatNotOffered {
        wanted: vk::Format,
        offered: Vec<vk::Format>,
    },
    /// The surface offers the format and not with a colour space this device
    /// can write through. A separate arm from the one above: the same format
    /// under a different colour space is a different picture, and folding them
    /// together would report a format that is present as absent.
    ColorSpaceNotOffered {
        format: vk::Format,
        offered: Vec<vk::ColorSpaceKHR>,
    },
    /// The composition copies into the image and the surface will not allow it.
    NoTransferDestination { supported: vk::ImageUsageFlags },
    /// Neither opaque nor inherited composite alpha. Every other mode blends
    /// the frame against the desktop using an alpha channel the guest did not
    /// author.
    NoOpaqueComposite {
        supported: vk::CompositeAlphaFlagsKHR,
    },
    /// The surface reported no formats at all.
    NoFormats,
    /// No queue family this rail presents on supports this surface.
    ///
    /// `VkSwapchainCreateInfoKHR`'s surface must be one the device supports as
    /// determined by `vkGetPhysicalDeviceSurfaceSupportKHR`
    /// (VUID-VkSwapchainCreateInfoKHR-surface-01270). A refusal rather than
    /// [`NotReady`]: nothing about this pairing changes when the window comes
    /// back.
    NotSupportedByPresentFamily,
}

impl Refusal {
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::FormatNotOffered { .. } => "vk_swapchain_format_not_offered",
            Self::ColorSpaceNotOffered { .. } => "vk_swapchain_color_space_not_offered",
            Self::NoTransferDestination { .. } => "vk_swapchain_no_transfer_destination",
            Self::NoOpaqueComposite { .. } => "vk_swapchain_no_opaque_composite",
            Self::NoFormats => "vk_swapchain_no_formats",
            Self::NotSupportedByPresentFamily => "vk_swapchain_family_cannot_present",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FormatNotOffered { wanted, offered } => write!(
                f,
                "{} wanted={wanted:?} offered={}",
                self.slug(),
                offered.len()
            ),
            Self::ColorSpaceNotOffered { format, offered } => write!(
                f,
                "{} format={format:?} offered={}",
                self.slug(),
                offered.len()
            ),
            Self::NoTransferDestination { supported } => {
                write!(f, "{} supported={supported:?}", self.slug())
            }
            Self::NoOpaqueComposite { supported } => {
                write!(f, "{} supported={supported:?}", self.slug())
            }
            Self::NoFormats | Self::NotSupportedByPresentFamily => f.write_str(self.slug()),
        }
    }
}

/// What a surface answered, turned into a swapchain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ready(Plan),
    /// Try again later; nothing is wrong. See [`NotReady`].
    NotReady(NotReady),
}

/// Everything a `VkSwapchainKHR` is created from.
///
/// Spelled out rather than held as the ash structure, which is not `Eq` — and
/// a choice that cannot be compared is one whose ladder cannot be asserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
    /// What was asked for. The count that governs anything is the one
    /// `vkGetSwapchainImagesKHR` reports; see the module doc.
    pub requested_images: u32,
    pub usage: vk::ImageUsageFlags,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub transform: vk::SurfaceTransformFlagsKHR,
    /// Whether the extent came from the surface or from [`Wanted`]. Carried
    /// because a caller that sized its attachments from the request needs to
    /// know the surface overrode it.
    pub extent_from_surface: bool,
}

impl Plan {
    /// The create info, for `surface`, replacing `old`.
    ///
    /// `old` is `vk::SwapchainKHR::null()` for the first swapchain of a
    /// surface. Passing the previous one lets the driver reuse its resources —
    /// and retires it: it may no longer acquire, and it may not be destroyed
    /// until every present against it has completed, which is
    /// [`reims_vgpu_core::present::PresentStream`]'s accounting and not this
    /// module's.
    pub fn create_info(
        &self,
        surface: vk::SurfaceKHR,
        old: vk::SwapchainKHR,
    ) -> vk::SwapchainCreateInfoKHR<'static> {
        vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(self.requested_images)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(self.extent)
            // One. A swapchain image is a scanout, and the stereoscopic form
            // this field reaches has no counterpart in the guest's contract.
            .image_array_layers(1)
            .image_usage(self.usage)
            // The composition records on one queue, so there is no ownership
            // to transfer and concurrent access would cost a mode switch for
            // nothing.
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(self.transform)
            .composite_alpha(self.composite_alpha)
            .present_mode(self.present_mode)
            // Pixels the compositor obscures may be discarded. The guest never
            // reads a presented image back — a Store goes to its own texture,
            // not to the drawable — so there is nothing that could observe one.
            .clipped(true)
            .old_swapchain(old)
    }
}

/// The extent the surface will accept.
///
/// A current extent whose width is `u32::MAX` is the surface saying the
/// swapchain chooses; anything else must be matched exactly, and clamping a
/// named current extent would produce a create info the driver refuses.
fn extent(capabilities: &vk::SurfaceCapabilitiesKHR, wanted: vk::Extent2D) -> (vk::Extent2D, bool) {
    let current = capabilities.current_extent;
    if current.width != u32::MAX {
        return (current, true);
    }
    let min = capabilities.min_image_extent;
    let max = capabilities.max_image_extent;
    (
        vk::Extent2D {
            width: wanted.width.clamp(min.width, max.width),
            height: wanted.height.clamp(min.height, max.height),
        },
        false,
    )
}

/// One more than the minimum — three under `MAILBOX` — bounded by a maximum
/// where the surface named one. See the module doc for both floors.
fn image_count(capabilities: &vk::SurfaceCapabilitiesKHR, mode: vk::PresentModeKHR) -> u32 {
    let mut wanted = capabilities.min_image_count.saturating_add(1);
    if mode == vk::PresentModeKHR::MAILBOX {
        wanted = wanted.max(MAILBOX_MIN_IMAGES);
    }
    // Zero means unbounded, which is an answer and not an absent one.
    if capabilities.max_image_count == 0 {
        wanted
    } else {
        wanted.min(capabilities.max_image_count)
    }
}

/// The images `MAILBOX` needs to be `MAILBOX`: one queued, one being drawn, one
/// to replace the queued one with. See the module doc.
pub const MAILBOX_MIN_IMAGES: u32 = 3;

/// The best present mode this surface offers, under `narrowing`.
fn present_mode(offered: &[vk::PresentModeKHR], narrowing: Narrowing) -> vk::PresentModeKHR {
    if !narrowing.fifo_only && offered.contains(&vk::PresentModeKHR::MAILBOX) {
        return vk::PresentModeKHR::MAILBOX;
    }
    // Required of every surface by the specification, so the ladder is total
    // and there is no arm below this one.
    vk::PresentModeKHR::FIFO
}

/// Choose a swapchain for `wanted` on `surface`.
///
/// # Errors
///
/// [`Refusal`] for a surface that cannot carry this composition at all.
/// A surface that merely has no area yet is [`Outcome::NotReady`] and not an
/// error; see the module doc.
pub fn plan(
    surface: &Surface<'_>,
    wanted: Wanted,
    narrowing: Narrowing,
) -> Result<Outcome, Refusal> {
    // Before anything the surface offers is read, because none of it is a
    // property of a pairing this device cannot make: a surface the present
    // family does not support may still answer formats and present modes, and
    // choosing among them would be planning a swapchain that
    // `vkCreateSwapchainKHR` is not permitted to build.
    if !surface.supported_by_present_family {
        return Err(Refusal::NotSupportedByPresentFamily);
    }
    if surface.formats.is_empty() {
        return Err(Refusal::NoFormats);
    }
    // The format first, because it is the one refusal a caller can act on by
    // composing differently, and deciding it before the extent keeps a
    // minimized window from hiding it until the window comes back.
    let matching: Vec<vk::SurfaceFormatKHR> = surface
        .formats
        .iter()
        .copied()
        .filter(|offered| offered.format == wanted.format)
        .collect();
    if matching.is_empty() {
        return Err(Refusal::FormatNotOffered {
            wanted: wanted.format,
            offered: surface.formats.iter().map(|f| f.format).collect(),
        });
    }
    // The colour space the composition writes through. Two arms rather than
    // one so a surface that offers the format under some other space reports
    // the space and not the format — the two are fixed by different changes.
    let Some(format) = matching
        .iter()
        .copied()
        .find(|offered| offered.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
    else {
        return Err(Refusal::ColorSpaceNotOffered {
            format: wanted.format,
            offered: matching.iter().map(|f| f.color_space).collect(),
        });
    };

    let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    if wanted.transfer_destination {
        if !surface
            .capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(Refusal::NoTransferDestination {
                supported: surface.capabilities.supported_usage_flags,
            });
        }
        usage |= vk::ImageUsageFlags::TRANSFER_DST;
    }

    let supported = surface.capabilities.supported_composite_alpha;
    let composite_alpha = if supported.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
        vk::CompositeAlphaFlagsKHR::OPAQUE
    } else if supported.contains(vk::CompositeAlphaFlagsKHR::INHERIT) {
        // The window system decides, which for an opaque window is the same
        // answer. Not preferred, because it is the window system's answer and
        // not this device's.
        vk::CompositeAlphaFlagsKHR::INHERIT
    } else {
        return Err(Refusal::NoOpaqueComposite { supported });
    };

    let (extent, extent_from_surface) = extent(&surface.capabilities, wanted.extent);
    if extent.width == 0 || extent.height == 0 {
        return Ok(Outcome::NotReady(NotReady::NoArea {
            width: extent.width,
            height: extent.height,
        }));
    }

    // The mode is chosen before the count because the count depends on it, and
    // the two used to be computed apart: the mode was picked, the count derived
    // from it, and then a literal `FIFO` handed to the create info — so a log
    // read `present_mode=mailbox` beside a swapchain that was not, and the
    // change that introduced the choice measured no effect because it never
    // reached the driver. One value, decided once, in one structure.
    let present_mode = present_mode(surface.present_modes, narrowing);
    Ok(Outcome::Ready(Plan {
        format,
        present_mode,
        extent,
        requested_images: image_count(&surface.capabilities, present_mode),
        usage,
        composite_alpha,
        // The surface's own transform, so the image is never rotated. Asking
        // for identity on a surface that is rotated makes the compositor rotate
        // the frame a second time or refuse the swapchain outright.
        transform: surface.capabilities.current_transform,
        extent_from_surface,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    const BGRA: vk::Format = vk::Format::B8G8R8A8_UNORM;

    fn srgb_space(format: vk::Format) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }
    }

    /// A surface that answers everything the ordinary way.
    fn capabilities() -> vk::SurfaceCapabilitiesKHR {
        vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: 4,
            current_extent: vk::Extent2D {
                width: 1280,
                height: 720,
            },
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            max_image_array_layers: 1,
            supported_transforms: vk::SurfaceTransformFlagsKHR::IDENTITY,
            current_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            supported_composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            supported_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_DST,
        }
    }

    fn wanted() -> Wanted {
        Wanted {
            format: BGRA,
            extent: vk::Extent2D {
                width: 800,
                height: 600,
            },
            transfer_destination: false,
        }
    }

    fn ready(
        capabilities: vk::SurfaceCapabilitiesKHR,
        formats: &[vk::SurfaceFormatKHR],
        modes: &[vk::PresentModeKHR],
        wanted: Wanted,
        narrowing: Narrowing,
    ) -> Plan {
        let surface = Surface {
            supported_by_present_family: true,
            capabilities,
            formats,
            present_modes: modes,
        };
        match plan(&surface, wanted, narrowing).unwrap_or_else(|refusal| panic!("{refusal}")) {
            Outcome::Ready(plan) => plan,
            Outcome::NotReady(not_ready) => panic!("{not_ready}"),
        }
    }

    fn refusal(
        capabilities: vk::SurfaceCapabilitiesKHR,
        formats: &[vk::SurfaceFormatKHR],
        wanted: Wanted,
    ) -> Refusal {
        let surface = Surface {
            supported_by_present_family: true,
            capabilities,
            formats,
            present_modes: &[vk::PresentModeKHR::FIFO],
        };
        plan(&surface, wanted, Narrowing::default()).expect_err("refused")
    }

    /// A surface no present family supports is refused before anything it
    /// offers is read.
    ///
    /// The regression: the pairing was never asked about. `VK_KHR_swapchain`
    /// being enumerated says the entry points resolve, not that this device
    /// can address this window --- on a host whose display is driven by a
    /// different adapter it is enumerated and no family supports the surface
    /// --- and the census cannot answer it, being taken before any surface
    /// exists. So a perfectly ordinary-looking surface produced a plan and
    /// `vkCreateSwapchainKHR` was handed a surface
    /// VUID-VkSwapchainCreateInfoKHR-surface-01270 forbids.
    #[test]
    fn a_surface_the_present_family_cannot_address_is_refused_before_it_is_read() {
        // Everything else about this surface is as ordinary as it gets: the
        // format, the colour space and the composite alpha all match, so the
        // only thing left to refuse on is the pairing.
        let surface = Surface {
            supported_by_present_family: false,
            capabilities: capabilities(),
            formats: &[vk::SurfaceFormatKHR {
                format: BGRA,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            }],
            present_modes: &[vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX],
        };
        let refused = plan(&surface, wanted(), Narrowing::default()).expect_err("refused");
        assert_eq!(refused, Refusal::NotSupportedByPresentFamily);
        assert_eq!(refused.slug(), "vk_swapchain_family_cannot_present");

        // And it is not a not-ready: a minimized window comes back, an
        // unsupported pairing does not. Checked with a zero extent, which is
        // the one input that would otherwise produce `NotReady`.
        let minimized = Surface {
            capabilities: vk::SurfaceCapabilitiesKHR {
                current_extent: vk::Extent2D {
                    width: 0,
                    height: 0,
                },
                ..capabilities()
            },
            ..surface
        };
        assert_eq!(
            plan(&minimized, wanted(), Narrowing::default()).expect_err("refused"),
            Refusal::NotSupportedByPresentFamily
        );
    }

    #[test]
    fn the_ordinary_surface_takes_the_top_rung_and_one_spare_image() {
        let plan = ready(
            capabilities(),
            &[srgb_space(BGRA)],
            &[vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX],
            wanted(),
            Narrowing::default(),
        );
        assert_eq!(plan.format, srgb_space(BGRA));
        assert_eq!(plan.present_mode, vk::PresentModeKHR::MAILBOX);
        // One more than the minimum, so an acquire has somewhere to go while
        // the presentation engine still holds the last image. Also the
        // `MAILBOX` floor, which this surface reaches without it — see
        // `only_the_mailbox_rung_asks_for_a_third_image` for the case that
        // separates them.
        assert_eq!(plan.requested_images, 3);
        assert_eq!(plan.extent, capabilities().current_extent);
        assert!(plan.extent_from_surface);
        assert_eq!(plan.usage, vk::ImageUsageFlags::COLOR_ATTACHMENT);
        assert_eq!(plan.composite_alpha, vk::CompositeAlphaFlagsKHR::OPAQUE);
    }

    /// The ladder is total: FIFO is required of every surface, so there is no
    /// bottom to fall off. And the switch moves down it and never up.
    #[test]
    fn the_present_mode_ladder_has_no_bottom_and_the_switch_only_descends() {
        let both = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX];
        let fifo = [vk::PresentModeKHR::FIFO];
        // Offered and wanted.
        assert_eq!(
            ready(
                capabilities(),
                &[srgb_space(BGRA)],
                &both,
                wanted(),
                Narrowing::default()
            )
            .present_mode,
            vk::PresentModeKHR::MAILBOX
        );
        // Offered and narrowed away.
        assert_eq!(
            ready(
                capabilities(),
                &[srgb_space(BGRA)],
                &both,
                wanted(),
                Narrowing { fifo_only: true }
            )
            .present_mode,
            vk::PresentModeKHR::FIFO
        );
        // Not offered, and the surface still gets a swapchain.
        assert_eq!(
            ready(
                capabilities(),
                &[srgb_space(BGRA)],
                &fifo,
                wanted(),
                Narrowing::default()
            )
            .present_mode,
            vk::PresentModeKHR::FIFO
        );
        // A surface that reported no modes at all is still served, because the
        // one this falls back to is the one the specification guarantees.
        assert_eq!(
            ready(
                capabilities(),
                &[srgb_space(BGRA)],
                &[],
                wanted(),
                Narrowing::default()
            )
            .present_mode,
            vk::PresentModeKHR::FIFO
        );
        // Nothing reaches the tearing mode, whatever the surface offers.
        for narrowing in [Narrowing::default(), Narrowing { fifo_only: true }] {
            assert_ne!(
                ready(
                    capabilities(),
                    &[srgb_space(BGRA)],
                    &[
                        vk::PresentModeKHR::IMMEDIATE,
                        vk::PresentModeKHR::FIFO,
                        vk::PresentModeKHR::MAILBOX,
                    ],
                    wanted(),
                    narrowing
                )
                .present_mode,
                vk::PresentModeKHR::IMMEDIATE
            );
        }
    }

    /// The two near-misses are refused rather than taken. Both are offered
    /// here, so the refusal is a decision and not an empty list.
    #[test]
    fn neither_the_srgb_sibling_nor_the_swapped_channels_is_a_substitute() {
        let offered = [
            srgb_space(vk::Format::B8G8R8A8_SRGB),
            srgb_space(vk::Format::R8G8B8A8_UNORM),
        ];
        let Refusal::FormatNotOffered {
            wanted: w,
            offered: list,
        } = refusal(capabilities(), &offered, wanted())
        else {
            panic!("the format refusal");
        };
        assert_eq!(w, BGRA);
        assert_eq!(
            list,
            vec![vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_UNORM],
            "the refusal has to say what is available or nobody can act on it"
        );
        // And adding the exact format to the same list makes it succeed, so
        // the refusal was about the format and not about the list's length.
        let with_exact = [offered[0], srgb_space(BGRA), offered[1]];
        assert_eq!(
            ready(
                capabilities(),
                &with_exact,
                &[vk::PresentModeKHR::FIFO],
                wanted(),
                Narrowing::default()
            )
            .format,
            srgb_space(BGRA)
        );
    }

    /// A format present under some other colour space reports the colour
    /// space. The two are fixed by different changes, so folding them together
    /// would send a caller looking for the wrong thing.
    #[test]
    fn a_format_under_the_wrong_colour_space_is_its_own_refusal() {
        let offered = [vk::SurfaceFormatKHR {
            format: BGRA,
            color_space: vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
        }];
        let Refusal::ColorSpaceNotOffered {
            format,
            offered: spaces,
        } = refusal(capabilities(), &offered, wanted())
        else {
            panic!("the colour-space refusal");
        };
        assert_eq!(format, BGRA);
        assert_eq!(spaces, vec![vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT]);
    }

    #[test]
    fn a_surface_with_no_formats_is_refused_by_name() {
        assert_eq!(refusal(capabilities(), &[], wanted()), Refusal::NoFormats);
    }

    /// A minimized window is not ready and is not refused. A refusal is
    /// terminal and would leave the surface dead for the session.
    #[test]
    fn a_minimized_window_is_not_ready_rather_than_refused() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
            ..capabilities()
        };
        let surface = Surface {
            supported_by_present_family: true,
            capabilities,
            formats: &[srgb_space(BGRA)],
            present_modes: &[vk::PresentModeKHR::FIFO],
        };
        assert_eq!(
            plan(&surface, wanted(), Narrowing::default()),
            Ok(Outcome::NotReady(NotReady::NoArea {
                width: 0,
                height: 0
            }))
        );
        // One zero dimension is enough.
        let one_zero = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 1280,
                height: 0,
            },
            ..capabilities
        };
        assert!(matches!(
            plan(
                &Surface {
                    supported_by_present_family: true,
                    capabilities: one_zero,
                    ..surface.clone()
                },
                wanted(),
                Narrowing::default()
            ),
            Ok(Outcome::NotReady(_))
        ));
    }

    /// The format is decided before the extent, so a minimized window does not
    /// hide a refusal that will still be there when it comes back.
    #[test]
    fn a_minimized_window_does_not_hide_a_format_refusal() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
            ..capabilities()
        };
        assert!(matches!(
            refusal(
                capabilities,
                &[srgb_space(vk::Format::R8G8B8A8_UNORM)],
                wanted()
            ),
            Refusal::FormatNotOffered { .. }
        ));
    }

    /// A named current extent is used exactly. Substituting the requested one
    /// produces a create info the driver refuses.
    #[test]
    fn a_named_current_extent_wins_over_the_request() {
        let plan = ready(
            capabilities(),
            &[srgb_space(BGRA)],
            &[vk::PresentModeKHR::FIFO],
            wanted(),
            Narrowing::default(),
        );
        assert_eq!(plan.extent.width, 1280);
        assert_ne!(plan.extent.width, wanted().extent.width);
        assert!(plan.extent_from_surface);
    }

    /// The deferred extent is the request, clamped into what the surface will
    /// take — and the caller is told the size is its own.
    #[test]
    fn a_deferred_extent_takes_the_request_and_clamps_it() {
        let deferred = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 640,
                height: 480,
            },
            max_image_extent: vk::Extent2D {
                width: 1024,
                height: 1024,
            },
            ..capabilities()
        };
        let go = |width, height| {
            ready(
                deferred,
                &[srgb_space(BGRA)],
                &[vk::PresentModeKHR::FIFO],
                Wanted {
                    extent: vk::Extent2D { width, height },
                    ..wanted()
                },
                Narrowing::default(),
            )
        };
        // Within bounds, unchanged.
        let inside = go(800, 600);
        assert_eq!(
            inside.extent,
            vk::Extent2D {
                width: 800,
                height: 600
            }
        );
        assert!(!inside.extent_from_surface);
        // Below and above, on each axis independently.
        assert_eq!(
            go(1, 4096).extent,
            vk::Extent2D {
                width: 640,
                height: 1024
            }
        );
    }

    /// One more than the minimum, and the maximum is a bound only where the
    /// surface named one. Zero means unbounded, which is an answer.
    #[test]
    fn the_image_count_is_one_spare_bounded_only_by_a_named_maximum() {
        let count = |min, max| {
            ready(
                vk::SurfaceCapabilitiesKHR {
                    min_image_count: min,
                    max_image_count: max,
                    ..capabilities()
                },
                &[srgb_space(BGRA)],
                &[vk::PresentModeKHR::FIFO],
                wanted(),
                Narrowing::default(),
            )
            .requested_images
        };
        assert_eq!(count(2, 4), 3);
        // The spare would exceed the maximum, so it is given up rather than
        // requested and refused.
        assert_eq!(count(2, 2), 2);
        assert_eq!(count(3, 3), 3);
        // Unbounded.
        assert_eq!(count(2, 0), 3);
        assert_eq!(count(8, 0), 9);
        // A driver reporting the largest possible minimum does not wrap.
        assert_eq!(count(u32::MAX, 0), u32::MAX);
    }

    /// `MAILBOX` asks for a third image, and only `MAILBOX` does.
    ///
    /// The mode's whole advantage is replacing the queued frame rather than
    /// queueing behind it, and with two images there is nothing to replace it
    /// with — the rung is taken and its behaviour is `FIFO`'s. So the floor is
    /// on that arm alone, which is the half a single fixture cannot show: a
    /// floor applied to both modes and a floor applied to neither both pass a
    /// test that only ever asks one of them.
    #[test]
    fn only_the_mailbox_rung_asks_for_a_third_image() {
        let count = |min, max, offered: &[vk::PresentModeKHR]| {
            ready(
                vk::SurfaceCapabilitiesKHR {
                    min_image_count: min,
                    max_image_count: max,
                    ..capabilities()
                },
                &[srgb_space(BGRA)],
                offered,
                wanted(),
                Narrowing::default(),
            )
            .requested_images
        };
        let fifo: &[vk::PresentModeKHR] = &[vk::PresentModeKHR::FIFO];
        let mailbox: &[vk::PresentModeKHR] =
            &[vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX];

        // One spare would be two; the rung asks for the third.
        assert_eq!(count(1, 0, fifo), 2);
        assert_eq!(count(1, 0, mailbox), MAILBOX_MIN_IMAGES);
        // Above the floor the ordinary rule governs both.
        assert_eq!(count(4, 0, fifo), 5);
        assert_eq!(count(4, 0, mailbox), 5);
        // A surface that caps at two cannot be argued with, on either rung.
        assert_eq!(count(1, 2, mailbox), 2);
        // And the switch down to FIFO takes the floor away with the rung.
        assert_eq!(
            ready(
                vk::SurfaceCapabilitiesKHR {
                    min_image_count: 1,
                    max_image_count: 0,
                    ..capabilities()
                },
                &[srgb_space(BGRA)],
                mailbox,
                wanted(),
                Narrowing { fifo_only: true },
            )
            .requested_images,
            2
        );
    }

    /// Transfer destination is asked for only when the composition copies, and
    /// refused rather than dropped when the surface will not allow it.
    #[test]
    fn the_transfer_usage_is_requested_only_when_it_is_needed_and_refused_when_absent() {
        let copying = Wanted {
            transfer_destination: true,
            ..wanted()
        };
        assert_eq!(
            ready(
                capabilities(),
                &[srgb_space(BGRA)],
                &[vk::PresentModeKHR::FIFO],
                copying,
                Narrowing::default()
            )
            .usage,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST
        );
        // Available and not asked for is not requested: a usage flag narrows
        // what the driver may choose for the image.
        assert_eq!(
            ready(
                capabilities(),
                &[srgb_space(BGRA)],
                &[vk::PresentModeKHR::FIFO],
                wanted(),
                Narrowing::default()
            )
            .usage,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        );
        // Asked for and unavailable refuses by name rather than silently
        // dropping the flag and failing at the copy.
        let no_transfer = vk::SurfaceCapabilitiesKHR {
            supported_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..capabilities()
        };
        assert_eq!(
            refusal(no_transfer, &[srgb_space(BGRA)], copying),
            Refusal::NoTransferDestination {
                supported: vk::ImageUsageFlags::COLOR_ATTACHMENT
            }
        );
        // And a composition that does not copy runs on the same surface.
        assert!(matches!(
            plan(
                &Surface {
                    supported_by_present_family: true,
                    capabilities: no_transfer,
                    formats: &[srgb_space(BGRA)],
                    present_modes: &[vk::PresentModeKHR::FIFO],
                },
                wanted(),
                Narrowing::default()
            ),
            Ok(Outcome::Ready(_))
        ));
    }

    /// Opaque, then inherited, then nothing. Every other mode blends the frame
    /// against the desktop using an alpha channel the guest did not author.
    #[test]
    fn the_composite_alpha_ladder_prefers_opaque_and_refuses_the_blending_modes() {
        let with = |supported| {
            let capabilities = vk::SurfaceCapabilitiesKHR {
                supported_composite_alpha: supported,
                ..capabilities()
            };
            let surface = Surface {
                supported_by_present_family: true,
                capabilities,
                formats: &[srgb_space(BGRA)],
                present_modes: &[vk::PresentModeKHR::FIFO],
            };
            plan(&surface, wanted(), Narrowing::default())
        };
        use vk::CompositeAlphaFlagsKHR as A;
        for supported in [
            A::OPAQUE,
            A::OPAQUE | A::INHERIT,
            A::OPAQUE | A::PRE_MULTIPLIED,
        ] {
            let Ok(Outcome::Ready(plan)) = with(supported) else {
                panic!("{supported:?}");
            };
            assert_eq!(plan.composite_alpha, A::OPAQUE);
        }
        let Ok(Outcome::Ready(plan)) = with(A::INHERIT | A::POST_MULTIPLIED) else {
            panic!("inherit is the fallback");
        };
        assert_eq!(plan.composite_alpha, A::INHERIT);
        for supported in [A::PRE_MULTIPLIED, A::POST_MULTIPLIED, A::empty()] {
            assert_eq!(
                with(supported),
                Err(Refusal::NoOpaqueComposite { supported }),
                "{supported:?}"
            );
        }
    }

    /// The surface's own transform, not identity. Asking for identity on a
    /// rotated surface makes the compositor rotate the frame a second time or
    /// refuse the swapchain.
    #[test]
    fn the_transform_is_the_surfaces_current_one() {
        for current in [
            vk::SurfaceTransformFlagsKHR::IDENTITY,
            vk::SurfaceTransformFlagsKHR::ROTATE_90,
            vk::SurfaceTransformFlagsKHR::ROTATE_180,
        ] {
            let plan = ready(
                vk::SurfaceCapabilitiesKHR {
                    current_transform: current,
                    supported_transforms: vk::SurfaceTransformFlagsKHR::IDENTITY
                        | vk::SurfaceTransformFlagsKHR::ROTATE_90
                        | vk::SurfaceTransformFlagsKHR::ROTATE_180,
                    ..capabilities()
                },
                &[srgb_space(BGRA)],
                &[vk::PresentModeKHR::FIFO],
                wanted(),
                Narrowing::default(),
            );
            assert_eq!(plan.transform, current);
        }
    }

    /// The create info is the plan, and the old swapchain passes through it.
    #[test]
    fn the_create_info_is_the_plan_and_carries_the_swapchain_it_replaces() {
        let plan = ready(
            capabilities(),
            &[srgb_space(BGRA)],
            &[vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX],
            Wanted {
                transfer_destination: true,
                ..wanted()
            },
            Narrowing::default(),
        );
        let surface = vk::SurfaceKHR::from_raw(7);
        let old = vk::SwapchainKHR::from_raw(9);
        let info = plan.create_info(surface, old);
        assert_eq!(info.surface, surface);
        assert_eq!(info.old_swapchain, old);
        assert_eq!(info.min_image_count, plan.requested_images);
        assert_eq!(info.image_format, BGRA);
        assert_eq!(info.image_color_space, vk::ColorSpaceKHR::SRGB_NONLINEAR);
        assert_eq!(info.image_extent, plan.extent);
        assert_eq!(info.image_usage, plan.usage);
        assert_eq!(info.present_mode, vk::PresentModeKHR::MAILBOX);
        assert_eq!(info.pre_transform, plan.transform);
        assert_eq!(info.composite_alpha, plan.composite_alpha);
        // A scanout is one layer, recorded on one queue, and never read back.
        assert_eq!(info.image_array_layers, 1);
        assert_eq!(info.image_sharing_mode, vk::SharingMode::EXCLUSIVE);
        assert_eq!(info.clipped, vk::TRUE);

        // The first swapchain of a surface replaces nothing.
        assert_eq!(
            plan.create_info(surface, vk::SwapchainKHR::null())
                .old_swapchain,
            vk::SwapchainKHR::null()
        );
    }

    /// Every refusal and every not-ready names itself, and no two share a
    /// slug — the failure channel is read by grep.
    #[test]
    fn every_reason_names_itself_distinctly() {
        let refusals = [
            Refusal::FormatNotOffered {
                wanted: BGRA,
                offered: Vec::new(),
            },
            Refusal::ColorSpaceNotOffered {
                format: BGRA,
                offered: Vec::new(),
            },
            Refusal::NoTransferDestination {
                supported: vk::ImageUsageFlags::empty(),
            },
            Refusal::NoOpaqueComposite {
                supported: vk::CompositeAlphaFlagsKHR::empty(),
            },
            Refusal::NoFormats,
        ];
        let mut slugs = std::collections::BTreeSet::new();
        for refusal in &refusals {
            assert!(slugs.insert(refusal.slug()), "{}", refusal.slug());
            assert!(refusal.to_string().starts_with(refusal.slug()));
        }
        let not_ready = NotReady::NoArea {
            width: 0,
            height: 0,
        };
        assert!(slugs.insert(not_ready.slug()));
        assert!(not_ready.to_string().starts_with(not_ready.slug()));
        assert_eq!(slugs.len(), 6);
    }
}
