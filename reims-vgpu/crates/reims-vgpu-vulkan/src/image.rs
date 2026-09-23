//! What a checked texture declaration becomes on this host: one image plan,
//! and the device query that has to admit it before anything is allocated.
//!
//! # Two halves, and the seam between them is the device query
//!
//! [`plan`] turns a [`Texture`] into the exact tuple `vkCreateImage` would
//! take. It decides nothing about capacity — every limit that matters is a
//! property of one `(format, type, tiling, usage, flags)` combination on one
//! physical device, and no static table has that. [`ImagePlan::admitted`]
//! takes what `vkGetPhysicalDeviceImageFormatProperties` reported for exactly
//! that tuple and checks the plan against it.
//!
//! Asking the general limits instead — `maxImageDimension2D` and friends — is
//! the shape that looks equivalent and is not: those are the ceiling over all
//! usages, and a format that supports 16384 sampled will often support far
//! less as a storage image or with a multisample count. Validating against
//! them and allocating anyway turns a refusal this device would have given
//! into a `vkCreateImage` the validation layers reject, or worse, into
//! undefined behaviour on a driver that does not check.
//!
//! Both halves are pure. `VkImageFormatProperties` is plain data, so the whole
//! admission is tested with no GPU, and a real device is needed only to obtain
//! the properties — not to decide anything from them.
//!
//! # Where the tiling comes from
//!
//! The route, and nowhere else. [`Route::DirectAlias`] means the guest's own
//! pages *are* the image, and guest pages have a row-major layout the device
//! did not choose — so the image is `LINEAR` or it is not an alias. Every
//! other route allocates a working representation the device does lay out, so
//! it is `OPTIMAL`. Reading the tiling from anywhere else would let a placement
//! decision and an image plan disagree about what the memory underneath the
//! image looks like.
//!
//! # `MTLTextureUsageUnknown` means every usage
//!
//! Metal reads a zero usage as "any", and this translates it as such: sampled,
//! storage and attachment together. Narrowing it to sampled would produce an
//! image that works until the first pass attaches it, and that failure names
//! the pass rather than this descriptor. Widening it here can make a device
//! refuse a texture it would have taken under a narrower declaration — that
//! refusal is visible and typed, which is the trade this makes deliberately.

use ash::vk;
use reims_vgpu_core::pixel_format::{format_has_depth_aspect, format_has_stencil_aspect};
use reims_vgpu_core::texture_shape::{Dimensions, Texture, TextureUsage};

use crate::placement::Route;

/// Why a texture cannot become an image on this host.
///
/// Nothing here restates a rule [`Texture`] already enforced; these are the
/// facts only a physical device can answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// More samples than a `VkSampleCountFlags` bit exists for. The guest API
    /// admits any power of two; Vulkan's mask stops at 64.
    SampleCountUnrepresentable {
        declared: u32,
    },
    /// The guest declared shader writes to a depth or stencil format. Vulkan
    /// has no storage image for one, so the declaration cannot be honoured
    /// rather than quietly dropped.
    DepthShaderWrite {
        format: u16,
    },
    /// A dimension larger than this device allows for this exact tuple.
    ExtentBeyondDevice {
        axis: &'static str,
        declared: u32,
        max: u32,
    },
    MipLevelsBeyondDevice {
        declared: u32,
        max: u32,
    },
    LayersBeyondDevice {
        declared: u32,
        max: u32,
    },
    /// The device supports this tuple, but not at this sample count.
    SampleCountUnsupported {
        declared: u32,
        supported: u32,
    },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::SampleCountUnrepresentable { .. } => "vk_image_sample_count_unrepresentable",
            Self::DepthShaderWrite { .. } => "vk_image_depth_shader_write",
            Self::ExtentBeyondDevice { .. } => "vk_image_extent_beyond_device",
            Self::MipLevelsBeyondDevice { .. } => "vk_image_mip_levels_beyond_device",
            Self::LayersBeyondDevice { .. } => "vk_image_layers_beyond_device",
            Self::SampleCountUnsupported { .. } => "vk_image_sample_count_unsupported",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SampleCountUnrepresentable { declared } => {
                write!(f, "{} declared={declared}", self.slug())
            }
            Self::DepthShaderWrite { format } => write!(f, "{} format={format}", self.slug()),
            Self::ExtentBeyondDevice {
                axis,
                declared,
                max,
            } => write!(
                f,
                "{} axis={axis} declared={declared} max={max}",
                self.slug()
            ),
            Self::MipLevelsBeyondDevice { declared, max }
            | Self::LayersBeyondDevice { declared, max } => {
                write!(f, "{} declared={declared} max={max}", self.slug())
            }
            Self::SampleCountUnsupported {
                declared,
                supported,
            } => write!(
                f,
                "{} declared={declared} supported=0x{supported:x}",
                self.slug()
            ),
        }
    }
}

/// The tuple `vkGetPhysicalDeviceImageFormatProperties` takes.
///
/// A value rather than five arguments, because it and the create info have to
/// name the same combination: a query for one usage followed by a create for
/// another is a validated-then-unvalidated allocation, and nothing downstream
/// can see that it happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatQuery {
    pub format: vk::Format,
    pub image_type: vk::ImageType,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
}

/// A native image, as it would be created, before any device has agreed to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "an image plan that is not admitted has not been checked against any device"]
pub struct ImagePlan {
    pub image_type: vk::ImageType,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
}

/// A plan one physical device has reported it can satisfy.
///
/// The only thing that carries a `VkImageCreateInfo`, so an allocation cannot
/// be made from a plan that skipped the query.
///
/// # The compression `MUTABLE_FORMAT` costs, and what it would take to keep it
///
/// A driver may disable lossless framebuffer compression for a mutable image,
/// because it cannot know which reinterpretations are coming.
/// `VkImageFormatListCreateInfo` — core in Vulkan 1.2, so always available on
/// this rail's baseline — is how that is recovered: naming the exact formats up
/// front lets the driver keep compression that is valid for all of them.
///
/// No list is promised here, and the reason is that a list is a *promise*. A
/// view naming a format outside it is invalid usage, so an incomplete list is
/// worse than none at all. The set is bounded only if allocations really are
/// keyed on [`crate::pixel::storage_format`], because then every bind reaching
/// one image shares its stored texel and differs only in the transfer function
/// — making the set exactly `{linear, srgb}`. That keying is a property of the
/// caller that owns residency, and no such caller exists yet. When one does,
/// this is where the list goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admitted {
    plan: ImagePlan,
}

/// Turn a checked texture into the image this host would create for it.
///
/// # Errors
///
/// [`Refusal`] for the two facts a declaration can carry that no Vulkan image
/// can express. Every geometric rule was already settled by [`Texture`].
pub fn plan(texture: Texture, format: vk::Format, route: Route) -> Result<ImagePlan, Refusal> {
    let kind = texture.kind();
    let guest_format = texture.pixel_format();
    let depth_stencil =
        format_has_depth_aspect(guest_format) || format_has_stencil_aspect(guest_format);

    let declared = texture.usage();
    // Zero means "any" and is the widest declaration, not the narrowest.
    let any = declared.is_unknown();
    let wants = |bit: TextureUsage| any || declared.contains(bit);
    // Both spellings of a shader store. `SHADER_ATOMIC` is the other one --- an
    // atomic *is* a store, which is why the usage below asks for one storage
    // image for either bit --- so a depth declaration carrying it is the same
    // declaration this refuses, and answering only the first spelling dropped
    // the second quietly, which is the thing the refusal exists to not do.
    let stores = declared.contains(TextureUsage::SHADER_WRITE)
        || declared.contains(TextureUsage::SHADER_ATOMIC);
    if !any && stores && depth_stencil {
        return Err(Refusal::DepthShaderWrite {
            format: guest_format,
        });
    }

    // Transfers unconditionally: the guest's bytes reach every texture through
    // one, whichever route places it, and a texture the device cannot copy
    // into is a texture that can never hold guest content.
    let mut usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    if wants(TextureUsage::SHADER_READ) {
        usage |= vk::ImageUsageFlags::SAMPLED;
    }
    // A depth or stencil format has no storage image on any host, so "any
    // usage" does not include one for it.
    if (wants(TextureUsage::SHADER_WRITE) || wants(TextureUsage::SHADER_ATOMIC)) && !depth_stencil {
        usage |= vk::ImageUsageFlags::STORAGE;
    }
    if wants(TextureUsage::RENDER_TARGET) {
        usage |= if depth_stencil {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
    }

    let mut flags = vk::ImageCreateFlags::empty();
    if kind.is_cube() {
        // Six 2D layers the device will let a cube view be made over. Without
        // this the layers exist and no cube sampler can address them.
        flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
    }
    // Mutable whenever a `VkImageView` of a *different* format may be made over
    // this image, which is two questions and not one.
    //
    // The guest's own declaration is the first, and it is read through `wants`
    // rather than `declared.contains`: `MTLTextureUsageUnknown` enables every
    // usage, so reading the bit literally here while reading zero as "any"
    // twenty lines up made one declaration mean two different things.
    //
    // The second is this rail's own sampling. `crate::pixel::storage_format`
    // keys an allocation on the spelling with no transfer function on it, and
    // `crate::pixel::sample_view_format` then answers a sampled bind with the
    // bind's spelling of that same stored texel — one surface serving both
    // `BGRA8Unorm` and `BGRA8Unorm_sRGB` through one image and two views, which
    // `storage_format`'s own documentation states is why the allocation folds.
    // That view is invalid usage without this flag, whatever the guest
    // declared, so a format with a second spelling gets it unasked.
    let reinterpretable = crate::pixel::texel_layout_of(format)
        .and_then(crate::pixel::srgb_texel_layout)
        .is_some_and(|srgb| srgb != crate::pixel::storage_format(format));
    if wants(TextureUsage::PIXEL_FORMAT_VIEW) || reinterpretable {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
    }
    if kind.dimensions() == Dimensions::Three && wants(TextureUsage::RENDER_TARGET) {
        // A render pass attaches one depth slice of a volume through a 2D view
        // over it, and that view is illegal without this flag. The flag is a
        // property of the image, so it cannot be added when the pass arrives —
        // and a volume the guest declared as a target that could not be
        // attached would fail at pass setup, naming the pass. See
        // [`crate::view::attachments`].
        flags |= vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE;
    }

    let samples = sample_flags(texture.sample_count())?;
    let extent = texture.extent();

    Ok(ImagePlan {
        image_type: match kind.dimensions() {
            Dimensions::One => vk::ImageType::TYPE_1D,
            Dimensions::Two => vk::ImageType::TYPE_2D,
            Dimensions::Three => vk::ImageType::TYPE_3D,
        },
        format,
        extent: vk::Extent3D {
            width: extent.x,
            height: extent.y,
            depth: extent.z,
        },
        mip_levels: texture.mip_levels(),
        // Slices, not elements: a cube array of two is twelve. See
        // [`reims_vgpu_core::texture_shape::Texture::layers`].
        array_layers: texture.layers(),
        samples,
        tiling: match route {
            Route::DirectAlias => vk::ImageTiling::LINEAR,
            Route::ImportedTransfer { .. } | Route::HostStaging { .. } => vk::ImageTiling::OPTIMAL,
        },
        usage,
        flags,
    })
}

/// The `VkSampleCountFlags` bit for a sample count.
///
/// The bit values *are* the counts, so this is an identity with a bound rather
/// than a table — and the bound is the whole reason it is a function: a guest
/// power of two above 64 has no bit, and `from_raw` on it would produce a mask
/// with a reserved bit set.
fn sample_flags(count: u32) -> Result<vk::SampleCountFlags, Refusal> {
    if count == 0 || !count.is_power_of_two() || count > 64 {
        return Err(Refusal::SampleCountUnrepresentable { declared: count });
    }
    Ok(vk::SampleCountFlags::from_raw(count))
}

impl ImagePlan {
    /// The tuple to query this device with.
    #[must_use]
    pub const fn query(&self) -> FormatQuery {
        FormatQuery {
            format: self.format,
            image_type: self.image_type,
            tiling: self.tiling,
            usage: self.usage,
            flags: self.flags,
        }
    }

    /// Check the plan against what the device reported for [`Self::query`].
    ///
    /// # Errors
    ///
    /// [`Refusal`] naming the one limit the plan exceeded, with both numbers.
    /// Checks run extent, then mips, then layers, then samples — outermost
    /// first, because a mip count is only meaningful for an extent the device
    /// admits.
    pub fn admitted(self, reported: vk::ImageFormatProperties) -> Result<Admitted, Refusal> {
        for (axis, declared, max) in [
            ("width", self.extent.width, reported.max_extent.width),
            ("height", self.extent.height, reported.max_extent.height),
            ("depth", self.extent.depth, reported.max_extent.depth),
        ] {
            if declared > max {
                return Err(Refusal::ExtentBeyondDevice {
                    axis,
                    declared,
                    max,
                });
            }
        }
        if self.mip_levels > reported.max_mip_levels {
            return Err(Refusal::MipLevelsBeyondDevice {
                declared: self.mip_levels,
                max: reported.max_mip_levels,
            });
        }
        if self.array_layers > reported.max_array_layers {
            return Err(Refusal::LayersBeyondDevice {
                declared: self.array_layers,
                max: reported.max_array_layers,
            });
        }
        if !reported.sample_counts.contains(self.samples) {
            return Err(Refusal::SampleCountUnsupported {
                declared: self.samples.as_raw(),
                supported: reported.sample_counts.as_raw(),
            });
        }
        Ok(Admitted { plan: self })
    }
}

impl Admitted {
    pub const fn plan(&self) -> &ImagePlan {
        &self.plan
    }

    /// The create info for this plan.
    ///
    /// `SharingMode::EXCLUSIVE` and `ImageLayout::UNDEFINED` are not choices
    /// this makes: one queue family owns every image on this rail — see
    /// [`crate::queues`] — and an image whose initial layout was `PREINITIALIZED`
    /// would be claiming its memory already holds texels, which no freshly
    /// allocated image's does. Both are tracked from the first transition by
    /// [`crate::layout`].
    pub fn create_info(&self) -> vk::ImageCreateInfo<'static> {
        vk::ImageCreateInfo::default()
            .flags(self.plan.flags)
            .image_type(self.plan.image_type)
            .format(self.plan.format)
            .extent(self.plan.extent)
            .mip_levels(self.plan.mip_levels)
            .array_layers(self.plan.array_layers)
            .samples(self.plan.samples)
            .tiling(self.plan.tiling)
            .usage(self.plan.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryClass;
    use reims_vgpu_core::pixel_format::{MTL_FORMAT_DEPTH32_FLOAT, MTL_FORMAT_RGBA8_UNORM};
    use reims_vgpu_core::texture_shape::{TextureKind, TextureShape};
    use std::collections::BTreeSet;

    const COLOR: vk::Format = vk::Format::R8G8B8A8_UNORM;
    const DEPTH: vk::Format = vk::Format::D32_SFLOAT;

    fn optimal() -> Route {
        Route::HostStaging {
            working: MemoryClass::DeviceLocal,
        }
    }

    fn texture(kind: TextureKind, usage: TextureUsage) -> Texture {
        TextureShape {
            kind: kind.ordinal(),
            width: 64,
            height: if kind.dimensions() == Dimensions::One {
                1
            } else {
                64
            },
            depth: if kind.dimensions() == Dimensions::Three {
                4
            } else {
                1
            },
            mipmap_level_count: 1,
            sample_count: if kind.is_multisample() { 4 } else { 1 },
            array_length: 1,
            pixel_format: MTL_FORMAT_RGBA8_UNORM,
            usage,
        }
        .checked()
        .expect("a valid declaration")
    }

    /// Properties generous enough that nothing is refused for capacity, so a
    /// test about one limit sets only that limit.
    fn generous() -> vk::ImageFormatProperties {
        vk::ImageFormatProperties {
            max_extent: vk::Extent3D {
                width: 16384,
                height: 16384,
                depth: 2048,
            },
            max_mip_levels: 15,
            max_array_layers: 2048,
            sample_counts: vk::SampleCountFlags::TYPE_1
                | vk::SampleCountFlags::TYPE_4
                | vk::SampleCountFlags::TYPE_8,
            max_resource_size: 1 << 32,
        }
    }

    #[test]
    fn each_dimensionality_becomes_the_image_type_that_has_it() {
        for kind in TextureKind::ALL {
            let plan = plan(texture(kind, TextureUsage::SHADER_READ), COLOR, optimal())
                .expect("a plannable texture");
            let expected = match kind.dimensions() {
                Dimensions::One => vk::ImageType::TYPE_1D,
                Dimensions::Two => vk::ImageType::TYPE_2D,
                Dimensions::Three => vk::ImageType::TYPE_3D,
            };
            assert_eq!(plan.image_type, expected, "{}", kind.name());
            // A cube is a 2D image with the flag, never an image type of its
            // own: Vulkan has no cube image, only a cube view over one.
            assert_eq!(
                plan.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE),
                kind.is_cube(),
                "{}",
                kind.name()
            );
        }
    }

    #[test]
    fn a_cube_array_plans_six_layers_per_element() {
        let cube = TextureShape {
            array_length: 3,
            ..TextureShape {
                kind: TextureKind::CubeArray.ordinal(),
                width: 32,
                height: 32,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 3,
                pixel_format: MTL_FORMAT_RGBA8_UNORM,
                usage: TextureUsage::SHADER_READ,
            }
        }
        .checked()
        .expect("a cube array");
        let plan = plan(cube, COLOR, optimal()).expect("plannable");
        assert_eq!(plan.array_layers, 18);
        assert_eq!(plan.extent.depth, 1);
    }

    #[test]
    fn a_volume_plans_a_depth_and_one_layer() {
        let plan = plan(
            texture(TextureKind::D3, TextureUsage::SHADER_READ),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert_eq!(plan.extent.depth, 4);
        assert_eq!(plan.array_layers, 1);
    }

    #[test]
    fn every_texture_can_be_copied_into_whatever_it_was_declared_for() {
        for usage in [
            TextureUsage::UNKNOWN,
            TextureUsage::SHADER_READ,
            TextureUsage::RENDER_TARGET,
            TextureUsage::SHADER_WRITE,
        ] {
            let plan = plan(texture(TextureKind::D2, usage), COLOR, optimal()).expect("plannable");
            assert!(plan.usage.contains(vk::ImageUsageFlags::TRANSFER_DST));
            assert!(plan.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC));
        }
    }

    #[test]
    fn each_declared_usage_bit_becomes_exactly_its_own_capability() {
        let read = plan(
            texture(TextureKind::D2, TextureUsage::SHADER_READ),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert!(read.usage.contains(vk::ImageUsageFlags::SAMPLED));
        assert!(!read.usage.contains(vk::ImageUsageFlags::STORAGE));
        assert!(!read.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));

        let write = plan(
            texture(TextureKind::D2, TextureUsage::SHADER_WRITE),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert!(write.usage.contains(vk::ImageUsageFlags::STORAGE));
        assert!(!write.usage.contains(vk::ImageUsageFlags::SAMPLED));

        let target = plan(
            texture(TextureKind::D2, TextureUsage::RENDER_TARGET),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert!(target.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));
        assert!(!target
            .usage
            .contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT));

        // An atomic is a store, and Vulkan spells both with the same bit.
        let atomic = plan(
            texture(TextureKind::D2, TextureUsage::SHADER_ATOMIC),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert!(atomic.usage.contains(vk::ImageUsageFlags::STORAGE));
    }

    #[test]
    fn an_unknown_usage_asks_for_every_capability_rather_than_the_narrowest() {
        let plan = plan(
            texture(TextureKind::D2, TextureUsage::UNKNOWN),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        for capability in [
            vk::ImageUsageFlags::SAMPLED,
            vk::ImageUsageFlags::STORAGE,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ] {
            assert!(plan.usage.contains(capability), "{capability:?}");
        }
        // Including the pixel-format view. Reading zero as "any" for the usage
        // bits and as "not asked" for this one made one declaration mean two
        // different things.
        assert!(plan.flags.contains(vk::ImageCreateFlags::MUTABLE_FORMAT));
    }

    /// The failure this exists to prevent: `pixel::sample_view_format` answers
    /// a sampled bind with the bind's spelling of a texel the allocation holds
    /// under the other one, which is how one surface serves `BGRA8Unorm` and
    /// `BGRA8Unorm_sRGB` through one image. That view is invalid usage on an
    /// image without `MUTABLE_FORMAT`, and no guest usage bit says so.
    #[test]
    fn a_format_with_a_second_spelling_is_mutable_whatever_the_guest_declared() {
        let colour = plan(
            texture(TextureKind::D2, TextureUsage::SHADER_READ),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert!(!colour.usage.contains(vk::ImageUsageFlags::STORAGE));
        assert!(
            colour.flags.contains(vk::ImageCreateFlags::MUTABLE_FORMAT),
            "the sRGB view this rail samples through would be invalid usage"
        );
        // And the two spellings really are the pair the flag is for.
        assert_eq!(crate::pixel::storage_format(COLOR), COLOR);
        assert_eq!(
            crate::pixel::texel_layout_of(COLOR).and_then(crate::pixel::srgb_texel_layout),
            Some(vk::Format::R8G8B8A8_SRGB)
        );

        // A depth format has no second spelling, so nothing may reinterpret it
        // and the flag stays off --- which is where the compression is kept.
        let depth = TextureShape {
            pixel_format: MTL_FORMAT_DEPTH32_FLOAT,
            kind: TextureKind::D2.ordinal(),
            width: 64,
            height: 64,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            usage: TextureUsage::RENDER_TARGET,
        }
        .checked()
        .expect("a depth target");
        assert!(!plan(depth, DEPTH, optimal())
            .expect("plannable")
            .flags
            .contains(vk::ImageCreateFlags::MUTABLE_FORMAT));
    }

    #[test]
    fn a_render_target_on_a_depth_format_takes_the_depth_attachment_slot() {
        let depth = TextureShape {
            pixel_format: MTL_FORMAT_DEPTH32_FLOAT,
            kind: TextureKind::D2.ordinal(),
            width: 64,
            height: 64,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        }
        .checked()
        .expect("a depth target");
        let plan = plan(depth, DEPTH, optimal()).expect("plannable");
        assert!(plan
            .usage
            .contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT));
        assert!(!plan.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));
        assert!(plan.usage.contains(vk::ImageUsageFlags::SAMPLED));
        assert!(!plan.usage.contains(vk::ImageUsageFlags::STORAGE));
    }

    #[test]
    fn shader_writes_to_a_depth_format_refuse_rather_than_being_dropped() {
        // Both spellings, because the usage below asks for one storage image
        // for either bit --- an atomic is a store --- so a depth declaration
        // carrying `SHADER_ATOMIC` is the same declaration, and answering it
        // with a plan that has no storage image is the quiet drop this refusal
        // was written against.
        for usage in [TextureUsage::SHADER_WRITE, TextureUsage::SHADER_ATOMIC] {
            let depth = TextureShape {
                pixel_format: MTL_FORMAT_DEPTH32_FLOAT,
                kind: TextureKind::D2.ordinal(),
                width: 64,
                height: 64,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 1,
                usage,
            }
            .checked()
            .expect("a declaration");
            assert_eq!(
                plan(depth, DEPTH, optimal()),
                Err(Refusal::DepthShaderWrite {
                    format: MTL_FORMAT_DEPTH32_FLOAT
                }),
                "{usage:?}"
            );
        }
    }

    #[test]
    fn an_unknown_usage_on_a_depth_format_asks_for_no_storage_image() {
        let depth = TextureShape {
            pixel_format: MTL_FORMAT_DEPTH32_FLOAT,
            kind: TextureKind::D2.ordinal(),
            width: 64,
            height: 64,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            usage: TextureUsage::UNKNOWN,
        }
        .checked()
        .expect("a declaration");
        // "Any usage" cannot include one no host has for this format, and this
        // is why it is a filter rather than a constant mask.
        let plan = plan(depth, DEPTH, optimal()).expect("plannable");
        assert!(!plan.usage.contains(vk::ImageUsageFlags::STORAGE));
        assert!(plan
            .usage
            .contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT));
    }

    #[test]
    fn a_pixel_format_view_declaration_is_the_mutable_format_flag() {
        let plan = plan(
            texture(
                TextureKind::D2,
                TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
            ),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert!(plan.flags.contains(vk::ImageCreateFlags::MUTABLE_FORMAT));
        // And a usage rather than a flag: it does not become an image usage
        // bit, which is the confusion the two names invite.
        assert_eq!(
            plan.usage,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
        );
    }

    #[test]
    fn only_a_direct_alias_is_linear() {
        for route in [
            Route::ImportedTransfer {
                working: MemoryClass::DeviceLocal,
            },
            Route::HostStaging {
                working: MemoryClass::DeviceLocal,
            },
        ] {
            let plan = plan(
                texture(TextureKind::D2, TextureUsage::SHADER_READ),
                COLOR,
                route,
            )
            .expect("plannable");
            assert_eq!(plan.tiling, vk::ImageTiling::OPTIMAL, "{}", route.name());
        }
        let aliased = plan(
            texture(TextureKind::D2, TextureUsage::SHADER_READ),
            COLOR,
            Route::DirectAlias,
        )
        .expect("plannable");
        assert_eq!(aliased.tiling, vk::ImageTiling::LINEAR);
    }

    #[test]
    fn a_sample_count_is_its_own_flag_bit_up_to_the_top_of_the_mask() {
        for count in [1u32, 2, 4, 8, 16, 32, 64] {
            assert_eq!(
                sample_flags(count),
                Ok(vk::SampleCountFlags::from_raw(count))
            );
        }
        for count in [0u32, 3, 128, 1 << 31] {
            assert_eq!(
                sample_flags(count),
                Err(Refusal::SampleCountUnrepresentable { declared: count })
            );
        }
    }

    #[test]
    fn a_query_names_the_same_combination_the_create_info_will() {
        let plan = plan(
            texture(TextureKind::Cube, TextureUsage::RENDER_TARGET),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        let query = plan.query();
        let info = plan.admitted(generous()).expect("admitted").create_info();
        assert_eq!(query.format, info.format);
        assert_eq!(query.image_type, info.image_type);
        assert_eq!(query.tiling, info.tiling);
        assert_eq!(query.usage, info.usage);
        assert_eq!(query.flags, info.flags);
    }

    #[test]
    fn a_create_info_carries_the_whole_plan_and_starts_undefined() {
        let plan = plan(
            texture(TextureKind::D2Multisample, TextureUsage::RENDER_TARGET),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        let info = plan.admitted(generous()).expect("admitted").create_info();
        assert_eq!(info.extent, plan.extent);
        assert_eq!(info.mip_levels, 1);
        assert_eq!(info.array_layers, 1);
        assert_eq!(info.samples, vk::SampleCountFlags::TYPE_4);
        assert_eq!(info.initial_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(info.sharing_mode, vk::SharingMode::EXCLUSIVE);
    }

    #[test]
    fn each_reported_limit_refuses_with_both_numbers() {
        let wide = TextureShape {
            kind: TextureKind::D2.ordinal(),
            width: 4096,
            height: 2048,
            depth: 1,
            mipmap_level_count: 12,
            sample_count: 1,
            array_length: 1,
            pixel_format: MTL_FORMAT_RGBA8_UNORM,
            usage: TextureUsage::SHADER_READ,
        }
        .checked()
        .expect("a declaration");
        let base = plan(wide, COLOR, optimal()).expect("plannable");

        let narrow_width = vk::ImageFormatProperties {
            max_extent: vk::Extent3D {
                width: 1024,
                ..generous().max_extent
            },
            ..generous()
        };
        assert_eq!(
            base.admitted(narrow_width),
            Err(Refusal::ExtentBeyondDevice {
                axis: "width",
                declared: 4096,
                max: 1024,
            })
        );

        // Height is checked only once the width is admitted, so the refusal
        // names the outermost failure rather than whichever the loop reached.
        let narrow_height = vk::ImageFormatProperties {
            max_extent: vk::Extent3D {
                width: 1024,
                height: 512,
                depth: 1,
            },
            ..generous()
        };
        assert!(matches!(
            base.admitted(narrow_height),
            Err(Refusal::ExtentBeyondDevice { axis: "width", .. })
        ));

        assert_eq!(
            base.admitted(vk::ImageFormatProperties {
                max_mip_levels: 4,
                ..generous()
            }),
            Err(Refusal::MipLevelsBeyondDevice {
                declared: 12,
                max: 4,
            })
        );
    }

    #[test]
    fn a_layer_count_past_the_device_refuses_with_slices_and_not_elements() {
        let cube = TextureShape {
            kind: TextureKind::CubeArray.ordinal(),
            width: 32,
            height: 32,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 4,
            pixel_format: MTL_FORMAT_RGBA8_UNORM,
            usage: TextureUsage::SHADER_READ,
        }
        .checked()
        .expect("a cube array");
        let plan = plan(cube, COLOR, optimal()).expect("plannable");
        assert_eq!(
            plan.admitted(vk::ImageFormatProperties {
                max_array_layers: 12,
                ..generous()
            }),
            // Twenty-four slices, not four elements: a device with room for
            // twelve layers cannot hold this cube array, and comparing the
            // element count would have said it could.
            Err(Refusal::LayersBeyondDevice {
                declared: 24,
                max: 12,
            })
        );
    }

    #[test]
    fn a_sample_count_the_device_does_not_offer_for_this_tuple_refuses() {
        let plan = plan(
            texture(TextureKind::D2Multisample, TextureUsage::RENDER_TARGET),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        let no_four = vk::ImageFormatProperties {
            sample_counts: vk::SampleCountFlags::TYPE_1 | vk::SampleCountFlags::TYPE_2,
            ..generous()
        };
        assert_eq!(
            plan.admitted(no_four),
            Err(Refusal::SampleCountUnsupported {
                declared: 4,
                supported: 3,
            })
        );
        assert!(plan.admitted(generous()).is_ok());
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::SampleCountUnrepresentable { declared: 3 },
            Refusal::DepthShaderWrite { format: 252 },
            Refusal::ExtentBeyondDevice {
                axis: "width",
                declared: 2,
                max: 1,
            },
            Refusal::MipLevelsBeyondDevice {
                declared: 2,
                max: 1,
            },
            Refusal::LayersBeyondDevice {
                declared: 2,
                max: 1,
            },
            Refusal::SampleCountUnsupported {
                declared: 4,
                supported: 3,
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_image_"));
        }
    }

    #[test]
    fn a_volume_declared_as_a_target_can_have_its_slices_attached() {
        let target = plan(
            texture(TextureKind::D3, TextureUsage::RENDER_TARGET),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert!(target
            .flags
            .contains(vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE));

        // Not on a volume nobody said would be rendered to: the flag can
        // restrict what a driver does with the image, and it buys nothing for
        // one that will only ever be sampled.
        let sampled = plan(
            texture(TextureKind::D3, TextureUsage::SHADER_READ),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert!(!sampled
            .flags
            .contains(vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE));

        // And on no flat texture, whose attachments are ordinary layer views.
        for kind in TextureKind::ALL {
            if kind.dimensions() == Dimensions::Three {
                continue;
            }
            let flat = plan(texture(kind, TextureUsage::RENDER_TARGET), COLOR, optimal())
                .expect("plannable");
            assert!(
                !flat
                    .flags
                    .contains(vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE),
                "{}",
                kind.name()
            );
        }
    }

    #[test]
    fn an_unknown_usage_volume_is_attachable_because_unknown_includes_a_target() {
        let any = plan(
            texture(TextureKind::D3, TextureUsage::UNKNOWN),
            COLOR,
            optimal(),
        )
        .expect("plannable");
        assert!(any
            .flags
            .contains(vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE));
    }
}
