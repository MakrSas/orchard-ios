//! The views one image is addressed through: the whole texture a shader
//! samples, and the one exact slice a render pass attaches.
//!
//! # Why the attachment views are expanded eagerly
//!
//! A render pass attaches *one* mip level of *one* layer. Vulkan has no way to
//! say that in the attachment itself — the selection is the view, so a texture
//! with seven levels and twelve layers needs eighty-four distinct
//! `VkImageView`s to be fully attachable, and which one a pass wants is not
//! known until the pass arrives.
//!
//! Creating them on demand is the shape that looks cheaper. It is not: the
//! demand arrives inside pass setup, so a creation failure there loses a frame
//! that was already recorded against, and a cache keyed by (level, layer) is
//! one more thing whose entries have to outlive submissions naming them. The
//! set is finite, known at allocation, and small — [`Texture::subresources`]
//! is levels times layers — so it is built once with the image and its
//! lifetime is the image's.
//!
//! [`attachments`] is what produces that set, in a fixed order, so a pass can
//! index it rather than search it.
//!
//! # A volume's slices are not its layers
//!
//! Rendering to one depth slice of a 3D texture is an attachment of a 2D view
//! over that slice — the layer count is one and the depth is what varies. So
//! the expansion iterates *slices*, which are the layers for every other type
//! and the level's depth for a volume. Iterating `layers()` for a volume would
//! produce exactly one attachment for a texture that has one per slice per
//! level, and the missing ones would be discovered as a pass that cannot be
//! set up.
//!
//! That view also needs a create flag the image does not otherwise carry, so
//! [`crate::image::plan`] sets it for a volume declared as a render target.
//! It is a property of the image and cannot be added later.
//!
//! # A sampled view names one plane; an attachment names what the format is
//!
//! A descriptor's image view may name at most one of `DEPTH` and `STENCIL`
//! (VUID-VkDescriptorImageInfo-imageView-01976), while an attachment over a
//! combined format names both. So the two views this module produces do not
//! share an aspect rule, and [`aspect`] — which answers what a format is made
//! of — is the attachment's answer and the transfer paths', not the sampled
//! one's.
//!
//! Which plane a sampled binding wants is not a guess: the guest says it in
//! the *view's* pixel format rather than the texture's, and
//! `MTLPixelFormatX32_Stencil8` / `X24_Stencil8` exist for no other purpose.
//! Both translate to the same `VkFormat` as the combined spelling they view,
//! so the plane survives only in the guest ordinal, which is why [`whole`]
//! takes it.
//!
//! # No Vulkan call
//!
//! Every function here produces a create info and none of them creates
//! anything, so the whole expansion — the counts, the order, the aspect masks
//! and the ranges — is tested with no GPU.

use ash::vk;
use reims_vgpu_core::pixel_format::{format_has_depth_aspect, format_has_stencil_aspect};
use reims_vgpu_core::texture_shape::{Dimensions, Texture, TextureKind};

/// The aspects a guest format's texels are made of.
///
/// Depth and stencil together for a combined format: a view used as an
/// attachment names both, and the transfer paths that want one plane pick it
/// from here rather than re-deriving it from the format code.
///
/// Not the answer for a sampled view — see [`sampled_aspect`].
#[must_use]
pub fn aspect(guest_format: u16) -> vk::ImageAspectFlags {
    let mut aspect = vk::ImageAspectFlags::empty();
    if format_has_depth_aspect(guest_format) {
        aspect |= vk::ImageAspectFlags::DEPTH;
    }
    if format_has_stencil_aspect(guest_format) {
        aspect |= vk::ImageAspectFlags::STENCIL;
    }
    if aspect.is_empty() {
        vk::ImageAspectFlags::COLOR
    } else {
        aspect
    }
}

/// The single aspect a *sampled* binding reads, from the guest format of the
/// view rather than of the texture.
///
/// [`aspect`]'s answer for a combined format is not a legal sampled view:
/// VUID-VkDescriptorImageInfo-imageView-01976 allows a descriptor's view at
/// most one of `DEPTH` and `STENCIL`. The rejection lands when the view is
/// written into a set, not when it is created, so the mistake would surface as
/// a validation failure about a texture the shader merely reads.
///
/// The narrowing is not a choice made here. `MTLPixelFormatX32_Stencil8` and
/// `X24_Stencil8` are the guest's only spelling for "the stencil plane of this
/// combined texture", and [`aspect`] already reads them as stencil alone. So a
/// bind that named a combined format named the depth plane, by having declined
/// the spelling that names the other one.
#[must_use]
pub fn sampled_aspect(guest_view_format: u16) -> vk::ImageAspectFlags {
    let aspect = aspect(guest_view_format);
    if aspect.contains(vk::ImageAspectFlags::DEPTH) {
        vk::ImageAspectFlags::DEPTH
    } else {
        aspect
    }
}

/// The view type a texture is sampled through.
///
/// A multisample texture has no view type of its own: it is a 2D or 2D-array
/// view over an image whose sample count says the rest. A cube is a cube view
/// over a 2D image, which is why the image needed `CUBE_COMPATIBLE`.
#[must_use]
pub const fn view_type(kind: TextureKind) -> vk::ImageViewType {
    match kind {
        TextureKind::D1 => vk::ImageViewType::TYPE_1D,
        TextureKind::D1Array => vk::ImageViewType::TYPE_1D_ARRAY,
        TextureKind::D2 | TextureKind::D2Multisample => vk::ImageViewType::TYPE_2D,
        TextureKind::D2Array | TextureKind::D2MultisampleArray => vk::ImageViewType::TYPE_2D_ARRAY,
        TextureKind::Cube => vk::ImageViewType::CUBE,
        TextureKind::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        TextureKind::D3 => vk::ImageViewType::TYPE_3D,
    }
}

/// Which subresources of an image a view covers.
///
/// Spelled out rather than held as a `VkImageSubresourceRange`, which is
/// neither comparable nor `Eq` — and an expansion whose ranges cannot be
/// compared is one whose eighty-four views cannot be asserted about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubresourceRange {
    pub aspect_mask: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

impl SubresourceRange {
    pub const fn native(self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect_mask,
            base_mip_level: self.base_mip_level,
            level_count: self.level_count,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count,
        }
    }
}

/// One view over an image, as it would be created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewPlan {
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub range: SubresourceRange,
}

impl ViewPlan {
    /// The create info for this view over `image`.
    ///
    /// Identity component mapping: a swizzle is a property of how a *binding*
    /// reads the texture and not of the image, so it belongs to the view the
    /// binder makes and not to the two this module produces.
    pub fn create_info(&self, image: vk::Image) -> vk::ImageViewCreateInfo<'static> {
        vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(self.view_type)
            .format(self.format)
            .components(vk::ComponentMapping::default())
            .subresource_range(self.range.native())
    }
}

/// The view covering the whole texture, which is what a sampled binding uses.
///
/// `guest_view_format` is the format the *bind* named, not the texture's: it
/// is the only thing that says which plane of a combined depth-stencil texture
/// is being read, because both spellings translate to one `VkFormat`. See
/// [`sampled_aspect`].
#[must_use]
pub fn whole(texture: Texture, guest_view_format: u16, format: vk::Format) -> ViewPlan {
    ViewPlan {
        view_type: view_type(texture.kind()),
        format,
        range: SubresourceRange {
            aspect_mask: sampled_aspect(guest_view_format),
            base_mip_level: 0,
            level_count: texture.mip_levels(),
            base_array_layer: 0,
            // A volume is one layer whatever its depth; see
            // [`reims_vgpu_core::texture_shape::Texture::layers`].
            layer_count: texture.layers(),
        },
    }
}

/// One attachable slice of a texture, and where it sits in the expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttachmentView {
    /// The mip level this view selects.
    pub level: u32,
    /// The layer for an arrayed or cube texture, and the depth slice for a
    /// volume. One number because a pass attaches one of them and never both.
    pub slice: u32,
    pub plan: ViewPlan,
}

/// How many attachable slices a level has.
///
/// Layers for everything but a volume, whose slices are the level's depth —
/// and the level's, not the base's, because a mip chain halves the depth too.
#[must_use]
pub fn slices_at(texture: Texture, level: u32) -> u32 {
    if texture.kind().dimensions() == Dimensions::Three {
        texture.level_extent(level).map_or(0, |extent| extent.z)
    } else {
        texture.layers()
    }
}

/// Every attachment view this texture needs, level-major.
///
/// Level-major so that the views of one level are contiguous, which is the
/// order a pass over a cube's six faces or an array's layers walks.
#[must_use]
pub fn attachments(texture: Texture, format: vk::Format) -> Vec<AttachmentView> {
    let aspect_mask = aspect(texture.pixel_format());
    let volume = texture.kind().dimensions() == Dimensions::Three;
    let mut views = Vec::new();
    for level in 0..texture.mip_levels() {
        for slice in 0..slices_at(texture, level) {
            views.push(AttachmentView {
                level,
                slice,
                plan: ViewPlan {
                    // Always a single 2D-ish view: a pass attaches one slice,
                    // never an array of them, and a 1D target's attachment is
                    // a 1D view rather than a 2D one over the same memory.
                    view_type: if volume {
                        vk::ImageViewType::TYPE_2D
                    } else if texture.kind().dimensions() == Dimensions::One {
                        vk::ImageViewType::TYPE_1D
                    } else {
                        vk::ImageViewType::TYPE_2D
                    },
                    format,
                    range: SubresourceRange {
                        aspect_mask,
                        base_mip_level: level,
                        level_count: 1,
                        base_array_layer: slice,
                        layer_count: 1,
                    },
                },
            });
        }
    }
    views
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::pixel_format::{
        MTL_FORMAT_DEPTH24_UNORM_STENCIL8, MTL_FORMAT_DEPTH32_FLOAT,
        MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, MTL_FORMAT_RGBA8_UNORM, MTL_FORMAT_STENCIL8,
        MTL_FORMAT_X24_STENCIL8, MTL_FORMAT_X32_STENCIL8,
    };
    use reims_vgpu_core::texture_shape::{TextureShape, TextureUsage, CUBE_FACES};

    const COLOR: vk::Format = vk::Format::R8G8B8A8_UNORM;

    fn shape(kind: TextureKind) -> TextureShape {
        TextureShape {
            kind: kind.ordinal(),
            width: 64,
            height: if kind.dimensions() == Dimensions::One {
                1
            } else {
                64
            },
            depth: if kind.dimensions() == Dimensions::Three {
                8
            } else {
                1
            },
            mipmap_level_count: 1,
            sample_count: if kind.is_multisample() { 4 } else { 1 },
            array_length: 1,
            pixel_format: MTL_FORMAT_RGBA8_UNORM,
            usage: TextureUsage::RENDER_TARGET,
        }
    }

    fn texture(kind: TextureKind) -> Texture {
        shape(kind).checked().expect("a valid declaration")
    }

    #[test]
    fn a_multisample_texture_views_as_its_single_sample_shape() {
        assert_eq!(
            view_type(TextureKind::D2Multisample),
            view_type(TextureKind::D2)
        );
        assert_eq!(
            view_type(TextureKind::D2MultisampleArray),
            view_type(TextureKind::D2Array)
        );
    }

    #[test]
    fn every_kind_has_a_view_type_and_the_cubes_are_cube_views() {
        for kind in TextureKind::ALL {
            let view = view_type(kind);
            assert_eq!(
                matches!(
                    view,
                    vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY
                ),
                kind.is_cube(),
                "{}",
                kind.name()
            );
        }
    }

    #[test]
    fn a_colour_format_is_the_colour_aspect_and_a_combined_one_is_both() {
        assert_eq!(aspect(MTL_FORMAT_RGBA8_UNORM), vk::ImageAspectFlags::COLOR);
        assert_eq!(
            aspect(MTL_FORMAT_DEPTH32_FLOAT),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(aspect(MTL_FORMAT_STENCIL8), vk::ImageAspectFlags::STENCIL);
        assert_eq!(
            aspect(MTL_FORMAT_DEPTH24_UNORM_STENCIL8),
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
    }

    #[test]
    fn the_whole_view_covers_every_level_and_every_layer() {
        let cube = TextureShape {
            mipmap_level_count: 7,
            array_length: 2,
            ..shape(TextureKind::CubeArray)
        }
        .checked()
        .expect("a cube array");
        let plan = whole(cube, MTL_FORMAT_RGBA8_UNORM, COLOR);
        assert_eq!(plan.view_type, vk::ImageViewType::CUBE_ARRAY);
        assert_eq!(plan.range.level_count, 7);
        assert_eq!(plan.range.layer_count, 2 * CUBE_FACES);
        assert_eq!(plan.range.base_mip_level, 0);
        assert_eq!(plan.range.base_array_layer, 0);
    }

    #[test]
    fn a_volume_is_one_layer_in_the_whole_view_however_deep_it_is() {
        let plan = whole(texture(TextureKind::D3), MTL_FORMAT_RGBA8_UNORM, COLOR);
        assert_eq!(plan.view_type, vk::ImageViewType::TYPE_3D);
        assert_eq!(plan.range.layer_count, 1);
    }

    #[test]
    fn a_render_target_expands_into_one_view_per_level_and_layer() {
        let cube = TextureShape {
            width: 16,
            height: 16,
            mipmap_level_count: 5,
            array_length: 2,
            ..shape(TextureKind::CubeArray)
        }
        .checked()
        .expect("a cube array");
        let views = attachments(cube, COLOR);
        assert_eq!(views.len() as u32, cube.subresources());
        assert_eq!(views.len(), 5 * 12);

        // Level-major, so one level's twelve faces are contiguous.
        for (index, view) in views.iter().enumerate() {
            assert_eq!(view.level, index as u32 / 12);
            assert_eq!(view.slice, index as u32 % 12);
            assert_eq!(view.plan.range.base_mip_level, view.level);
            assert_eq!(view.plan.range.level_count, 1);
            assert_eq!(view.plan.range.base_array_layer, view.slice);
            assert_eq!(view.plan.range.layer_count, 1);
            // A cube's attachment is a 2D view over one face, never a cube
            // view: a pass renders to a face and not to a cube.
            assert_eq!(view.plan.view_type, vk::ImageViewType::TYPE_2D);
        }
    }

    #[test]
    fn a_volume_expands_per_slice_and_the_slice_count_halves_with_the_level() {
        let volume = TextureShape {
            width: 8,
            height: 8,
            depth: 8,
            mipmap_level_count: 4,
            ..shape(TextureKind::D3)
        }
        .checked()
        .expect("a volume");
        assert_eq!(slices_at(volume, 0), 8);
        assert_eq!(slices_at(volume, 1), 4);
        assert_eq!(slices_at(volume, 2), 2);
        assert_eq!(slices_at(volume, 3), 1);

        let views = attachments(volume, COLOR);
        // Fifteen, not four: iterating the layer count would have produced one
        // per level for a texture that has one per slice per level.
        assert_eq!(views.len(), 15);
        assert_eq!(views[0].slice, 0);
        assert_eq!(views[7].level, 0);
        assert_eq!(views[8].level, 1);
        assert!(views
            .iter()
            .all(|v| v.plan.view_type == vk::ImageViewType::TYPE_2D));
        // A level past the top has no slices at all rather than a clamped one.
        assert_eq!(slices_at(volume, 4), 0);
    }

    #[test]
    fn a_one_dimensional_target_attaches_as_a_one_dimensional_view() {
        let views = attachments(texture(TextureKind::D1Array), COLOR);
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].plan.view_type, vk::ImageViewType::TYPE_1D);
    }

    #[test]
    fn a_depth_target_attaches_with_the_depth_aspect() {
        let depth = TextureShape {
            pixel_format: MTL_FORMAT_DEPTH32_FLOAT,
            ..shape(TextureKind::D2)
        }
        .checked()
        .expect("a depth target");
        let views = attachments(depth, vk::Format::D32_SFLOAT);
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].plan.range.aspect_mask, vk::ImageAspectFlags::DEPTH);
        assert_eq!(
            whole(depth, MTL_FORMAT_DEPTH32_FLOAT, vk::Format::D32_SFLOAT)
                .range
                .aspect_mask,
            vk::ImageAspectFlags::DEPTH
        );
    }

    #[test]
    fn a_sampled_view_of_a_combined_texture_names_one_plane_and_the_bind_says_which() {
        // Both binds are over the same texture and translate to the same
        // `VkFormat`; the guest ordinal is the only thing that separates them.
        let combined = TextureShape {
            pixel_format: MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            ..shape(TextureKind::D2)
        }
        .checked()
        .expect("a combined depth-stencil texture");
        let native = vk::Format::D32_SFLOAT_S8_UINT;

        // What the texture is made of, which is the attachment's answer and
        // an illegal sampled view.
        assert_eq!(
            aspect(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8),
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            attachments(combined, native)[0].plan.range.aspect_mask,
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );

        // A bind that named the combined format declined the spelling that
        // names stencil, so it reads depth.
        assert_eq!(
            whole(combined, MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, native)
                .range
                .aspect_mask,
            vk::ImageAspectFlags::DEPTH
        );
        // And the spelling that exists for no other purpose reads stencil,
        // over the very same texture.
        assert_eq!(
            whole(combined, MTL_FORMAT_X32_STENCIL8, native)
                .range
                .aspect_mask,
            vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            whole(combined, MTL_FORMAT_X24_STENCIL8, native)
                .range
                .aspect_mask,
            vk::ImageAspectFlags::STENCIL
        );
    }

    #[test]
    fn no_sampled_aspect_names_two_planes() {
        // The rule a descriptor write enforces, asserted over every format the
        // guest can name rather than the four this module happens to import.
        for format in 0..=u16::MAX {
            let sampled = sampled_aspect(format);
            assert_eq!(
                sampled.as_raw().count_ones(),
                1,
                "format {format} sampled as {sampled:?}"
            );
            // And it is one of the aspects the format actually has: narrowing,
            // never inventing.
            assert!(aspect(format).contains(sampled), "format {format}");
        }
    }

    #[test]
    fn a_create_info_carries_the_plan_and_the_identity_swizzle() {
        let plan = whole(texture(TextureKind::D2), MTL_FORMAT_RGBA8_UNORM, COLOR);
        let image = vk::Image::null();
        let info = plan.create_info(image);
        assert_eq!(info.view_type, plan.view_type);
        assert_eq!(info.format, COLOR);
        assert_eq!(info.subresource_range.level_count, plan.range.level_count);
        assert_eq!(info.subresource_range.layer_count, plan.range.layer_count);
        assert_eq!(
            info.subresource_range.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
        // The identity swizzle, which is what `ComponentMapping::default()`
        // spells: every channel `IDENTITY` rather than a channel-for-channel
        // list. A view that reordered here would reorder for every binding.
        assert_eq!(info.components.r, vk::ComponentSwizzle::IDENTITY);
        assert_eq!(info.components.a, vk::ComponentSwizzle::IDENTITY);
    }
}
