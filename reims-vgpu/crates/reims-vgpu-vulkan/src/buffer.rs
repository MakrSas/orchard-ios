//! What a guest buffer becomes here: one `VkBuffer` that can be bound as
//! anything, because the guest's contract never said it could not.
//!
//! # Why the usage is the whole set
//!
//! A guest buffer is created with a length and a storage mode and nothing
//! else. There is no field on it saying "this is a vertex buffer", and no
//! later command is refused for binding it somewhere its creation did not
//! anticipate — a buffer written by a compute dispatch and then read as index
//! data is an ordinary thing for a guest to do, and it does not re-create the
//! buffer in between.
//!
//! Vulkan does require the usage up front. So the only translation that
//! preserves the guest's contract is the complete set of operation classes
//! this rail can ever bind a buffer as. Deriving a narrower set from the first
//! use observed is the alternative, and it fails in the way that is hardest to
//! find: the buffer is created, works for every command that matches the
//! guess, and produces a validation error or a driver fault at the first one
//! that does not — arbitrarily far from the creation that decided it.
//!
//! The cost of the wide set is that a driver may place the allocation more
//! conservatively. That is a placement question, and [`crate::placement`] and
//! [`crate::memory`] are where it is answered; it is never a reason to make a
//! guest command fail.
//!
//! # What a device can still refuse
//!
//! Size. A zero-length buffer is not a buffer, and `maxBufferSize` is a real
//! bound on hosts that report one. Hosts that do not report one are not
//! assumed to be unbounded and are not assumed to be bounded either: the
//! refusal simply does not apply, and the allocation-size bound in
//! [`crate::memory`] is what remains.

use ash::vk;

use crate::placement::Route;

/// Why a guest buffer cannot become a `VkBuffer` on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A buffer of no bytes. Not an empty buffer to be created and never
    /// read: `vkCreateBuffer` rejects a zero size, so admitting one here would
    /// move the failure to the driver with none of the guest's context left.
    ZeroLength,
    /// Longer than this device creates buffers.
    LengthBeyondDevice { declared: u64, max: u64 },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ZeroLength => "vk_buffer_zero_length",
            Self::LengthBeyondDevice { .. } => "vk_buffer_length_beyond_device",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroLength => f.write_str(self.slug()),
            Self::LengthBeyondDevice { declared, max } => {
                write!(f, "{} declared={declared} max={max}", self.slug())
            }
        }
    }
}

/// The largest buffer this device creates, when it says.
///
/// `VkPhysicalDeviceMaintenance4Properties::maxBufferSize` arrived in Vulkan
/// 1.3, and the baseline here is 1.2. `None` is "this device did not report
/// one" and is deliberately not a number: substituting `u64::MAX` would make
/// the refusal below claim a device answered a question it was never asked,
/// and substituting a guessed ceiling would refuse buffers a device creates
/// perfectly well.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferLimits {
    pub max_buffer_size: Option<u64>,
}

/// Every operation class this rail can bind a buffer as.
///
/// One constant rather than a set assembled per call site, because two call
/// sites that assembled it separately would differ by exactly the bit that
/// only some guest command needs — which is the failure this module exists to
/// prevent.
///
/// `TRANSFER_SRC`/`TRANSFER_DST` are here for the same reason as on an image:
/// guest bytes reach and leave every buffer through a copy on some route.
pub const EVERY_CLASS: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
);

/// A native buffer, as it would be created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a buffer plan that is not created allocates nothing"]
pub struct BufferPlan {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    /// From the route, like an image's tiling: a direct alias is the guest's
    /// own pages and every other route allocates a representation this device
    /// owns. Carried so a caller can tell the two apart without re-deriving
    /// the placement.
    pub aliased: bool,
}

/// Plan the buffer for a guest allocation of `size` bytes.
///
/// # Errors
///
/// [`Refusal`] for a length no buffer can have, or one past what this device
/// reported.
pub fn plan(size: u64, route: Route, limits: BufferLimits) -> Result<BufferPlan, Refusal> {
    if size == 0 {
        return Err(Refusal::ZeroLength);
    }
    if let Some(max) = limits.max_buffer_size {
        if size > max {
            return Err(Refusal::LengthBeyondDevice {
                declared: size,
                max,
            });
        }
    }
    Ok(BufferPlan {
        size,
        usage: EVERY_CLASS,
        aliased: matches!(route, Route::DirectAlias),
    })
}

impl BufferPlan {
    /// The create info for this plan.
    ///
    /// `EXCLUSIVE` for the same reason an image is: one queue family owns
    /// everything on this rail — see [`crate::queues`].
    pub fn create_info(&self) -> vk::BufferCreateInfo<'static> {
        vk::BufferCreateInfo::default()
            .size(self.size)
            .usage(self.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryClass;
    use std::collections::BTreeSet;

    fn staged() -> Route {
        Route::HostStaging {
            working: MemoryClass::DeviceLocal,
        }
    }

    fn unbounded() -> BufferLimits {
        BufferLimits {
            max_buffer_size: None,
        }
    }

    #[test]
    fn every_buffer_can_be_bound_as_every_class() {
        let plan = plan(1024, staged(), unbounded()).expect("plannable");
        for class in [
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
            vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::BufferUsageFlags::INDIRECT_BUFFER,
        ] {
            assert!(plan.usage.contains(class), "{class:?}");
        }
    }

    #[test]
    fn the_usage_does_not_depend_on_anything_about_the_allocation() {
        // The claim is that nothing narrows it — not the size, not the route.
        // A future arm that made one of these matter would be re-introducing
        // the first-use guess this module refuses to make.
        let first = plan(1, Route::DirectAlias, unbounded()).expect("plannable");
        let second = plan(
            1 << 30,
            Route::ImportedTransfer {
                working: MemoryClass::DeviceLocal,
            },
            unbounded(),
        )
        .expect("plannable");
        assert_eq!(first.usage, second.usage);
        assert_eq!(first.usage, EVERY_CLASS);
    }

    #[test]
    fn the_route_is_the_only_thing_that_says_whether_the_pages_are_the_guests() {
        assert!(
            plan(16, Route::DirectAlias, unbounded())
                .expect("plannable")
                .aliased
        );
        for route in [
            staged(),
            Route::ImportedTransfer {
                working: MemoryClass::Upload,
            },
        ] {
            assert!(
                !plan(16, route, unbounded()).expect("plannable").aliased,
                "{}",
                route.name()
            );
        }
    }

    #[test]
    fn a_zero_length_buffer_is_refused_here_rather_than_by_the_driver() {
        assert_eq!(plan(0, staged(), unbounded()), Err(Refusal::ZeroLength));
    }

    #[test]
    fn a_device_that_reported_no_maximum_refuses_nothing_for_length() {
        // Not treated as unbounded and not treated as bounded: the check does
        // not apply, which is a third answer and the only honest one.
        assert!(plan(u64::MAX, staged(), unbounded()).is_ok());
    }

    #[test]
    fn a_length_past_a_reported_maximum_refuses_with_both_numbers() {
        let limits = BufferLimits {
            max_buffer_size: Some(1 << 20),
        };
        assert_eq!(
            plan((1 << 20) + 1, staged(), limits),
            Err(Refusal::LengthBeyondDevice {
                declared: (1 << 20) + 1,
                max: 1 << 20,
            })
        );
        // The bound itself is admitted, so the comparison is not off by one.
        assert!(plan(1 << 20, staged(), limits).is_ok());
    }

    #[test]
    fn a_create_info_carries_the_plan() {
        let plan = plan(4096, staged(), unbounded()).expect("plannable");
        let info = plan.create_info();
        assert_eq!(info.size, 4096);
        assert_eq!(info.usage, EVERY_CLASS);
        assert_eq!(info.sharing_mode, vk::SharingMode::EXCLUSIVE);
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::ZeroLength,
            Refusal::LengthBeyondDevice {
                declared: 2,
                max: 1,
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_buffer_"));
        }
    }
}
