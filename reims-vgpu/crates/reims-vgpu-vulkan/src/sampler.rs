//! What a checked sampler declaration becomes here, and the three things about
//! it that are host capabilities.
//!
//! # A sampler has one border and a guest may ask for two
//!
//! `MTLSamplerAddressModeClampToZero` clamps to transparent black, and
//! `MTLSamplerAddressModeClampToBorderColor` clamps to the declared colour.
//! Both are border modes and both are per-axis, while the border *colour* is
//! per-sampler in both APIs. A descriptor using both modes with a
//! non-transparent declaration is asking for two borders on one sampler, and
//! there is no arrangement of a `VkSamplerCreateInfo` that has two. Picking
//! one silently gives the other axis the wrong border — a soft, wrong edge
//! that looks like a filtering artefact — so it refuses.
//!
//! # Three capabilities
//!
//! `MirrorClampToEdge` is a Vulkan feature (`samplerMirrorClampToEdge`,
//! promoted into 1.2's feature block), anisotropy above one is a feature with
//! a limit, and a border colour outside the three fixed ones would be another.
//! Only the first two arise from this wire. Anisotropy is *clamped* to the
//! device's limit rather than refused, because that is what Metal itself does
//! and because the difference is a quality one; the effective value is carried
//! so a census can see it. The mirror mode refuses, because clamping it to
//! `ClampToEdge` mirrors nothing and the seam is visible.
//!
//! # `NotMipmapped` is a level clamp, not a filter
//!
//! Vulkan has no "no mipmapping" mip mode. The way to say it is a maximum
//! level of detail of zero, which makes every sample come from the top level
//! whatever the chain holds. Setting `NEAREST` and leaving the guest's clamp
//! would sample lower levels wherever the guest's clamp allowed it.
//!
//! # Planned, not created
//!
//! The plan is a value and nothing here calls `vkCreateSampler`, so every
//! mapping and every refusal is tested with no GPU.

use ash::vk;
use reims_vgpu_core::sampler::{
    AddressMode, BorderColor, CompareFunction, Filter, MipFilter, SamplerState,
};

/// What this host offers for the parts of a sampler that are not mappings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplerCell {
    /// `VkPhysicalDeviceVulkan12Features::samplerMirrorClampToEdge`, or the
    /// extension below 1.2. The census joins them.
    pub mirror_clamp_to_edge: bool,
    /// `VkPhysicalDeviceFeatures::samplerAnisotropy`.
    pub anisotropy: bool,
    /// `VkPhysicalDeviceLimits::maxSamplerAnisotropy`. Meaningless without the
    /// feature, and never read without it.
    pub max_anisotropy: f32,
}

/// Why a sampler cannot be built here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Two border modes and a non-transparent border colour. See the module
    /// doc.
    TwoBorders { declared: BorderColor },
    /// The guest asked for `MirrorClampToEdge` and this device has no such
    /// address mode.
    NoMirrorClampToEdge,
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::TwoBorders { .. } => "vk_sampler_two_borders",
            Self::NoMirrorClampToEdge => "vk_sampler_no_mirror_clamp_to_edge",
        }
    }
}

/// The always-on failure channel's view of a refusal.
///
/// Implemented here rather than at the call site, so the slug and the values
/// behind it travel with the refusal: a consumer that had to spell them would
/// be a second account of one event, and a second consumer a third.
impl reims_vgpu_observe::Decline for Refusal {
    fn slug(&self) -> &'static str {
        (*self).slug()
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::TwoBorders { declared } => vec![("declared", declared.name().to_string())],
            Self::NoMirrorClampToEdge => Vec::new(),
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TwoBorders { declared } => {
                write!(f, "{} declared={}", self.slug(), declared.name())
            }
            Self::NoMirrorClampToEdge => f.write_str(self.slug()),
        }
    }
}

/// A native sampler, as it would be created.
///
/// Spelled out rather than held as a `VkSamplerCreateInfo`, which is neither
/// comparable nor `Eq` — and a translation whose result cannot be compared is
/// one whose mappings cannot be asserted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerPlan {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    /// `u`, `v`, `w` — the guest's `s`, `t`, `r`.
    pub address: [vk::SamplerAddressMode; 3],
    pub anisotropy_enable: bool,
    /// The value actually used, after the device's limit. Equal to the guest's
    /// request when the device allows it, and lower when it does not.
    pub max_anisotropy: f32,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
}

impl SamplerPlan {
    pub fn create_info(&self) -> vk::SamplerCreateInfo<'static> {
        vk::SamplerCreateInfo::default()
            .mag_filter(self.mag_filter)
            .min_filter(self.min_filter)
            .mipmap_mode(self.mipmap_mode)
            .address_mode_u(self.address[0])
            .address_mode_v(self.address[1])
            .address_mode_w(self.address[2])
            .mip_lod_bias(0.0)
            .anisotropy_enable(self.anisotropy_enable)
            .max_anisotropy(self.max_anisotropy)
            .compare_enable(self.compare_enable)
            .compare_op(self.compare_op)
            .min_lod(self.min_lod)
            .max_lod(self.max_lod)
            .border_color(self.border_color)
            .unnormalized_coordinates(self.unnormalized_coordinates)
    }
}

/// The filter a guest filter becomes. Total, and the two sets match one to
/// one.
#[must_use]
pub const fn filter(filter: Filter) -> vk::Filter {
    match filter {
        Filter::Nearest => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    }
}

/// The comparison a guest comparison becomes. Total.
#[must_use]
pub const fn compare_op(function: CompareFunction) -> vk::CompareOp {
    match function {
        CompareFunction::Never => vk::CompareOp::NEVER,
        CompareFunction::Less => vk::CompareOp::LESS,
        CompareFunction::Equal => vk::CompareOp::EQUAL,
        CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFunction::Greater => vk::CompareOp::GREATER,
        CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

/// The fixed border a guest border colour becomes.
///
/// The float variants, always. Vulkan's integer variants exist for integer
/// image formats, and a sampler is independent of the image it is used with in
/// both models — so choosing between them here would be choosing on behalf of
/// an image this sampler has not been paired with.
#[must_use]
pub const fn border_color(color: BorderColor) -> vk::BorderColor {
    match color {
        BorderColor::TransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        BorderColor::OpaqueBlack => vk::BorderColor::FLOAT_OPAQUE_BLACK,
        BorderColor::OpaqueWhite => vk::BorderColor::FLOAT_OPAQUE_WHITE,
    }
}

/// The address mode a guest mode becomes.
///
/// # Errors
///
/// [`Refusal::NoMirrorClampToEdge`] for the one mode that is a feature.
pub const fn address_mode(
    mode: AddressMode,
    cell: SamplerCell,
) -> Result<vk::SamplerAddressMode, Refusal> {
    Ok(match mode {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        // Both border modes are one Vulkan mode; which border it reads is the
        // per-sampler colour, and `plan` is what checks the two agree.
        AddressMode::ClampToZero | AddressMode::ClampToBorderColor => {
            vk::SamplerAddressMode::CLAMP_TO_BORDER
        }
        AddressMode::MirrorClampToEdge => {
            if cell.mirror_clamp_to_edge {
                vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE
            } else {
                return Err(Refusal::NoMirrorClampToEdge);
            }
        }
    })
}

/// Translate a checked sampler declaration.
///
/// # Errors
///
/// [`Refusal`], with nothing partially translated.
pub fn plan(state: SamplerState, cell: SamplerCell) -> Result<SamplerPlan, Refusal> {
    if state.border_modes_disagree() {
        return Err(Refusal::TwoBorders {
            declared: state.border_color(),
        });
    }
    let guest = state.address();
    let address = [
        address_mode(guest[0], cell)?,
        address_mode(guest[1], cell)?,
        address_mode(guest[2], cell)?,
    ];

    // `ClampToZero` is transparent black whatever the descriptor's colour
    // field says, and `border_modes_disagree` above is what makes that safe to
    // apply: the two cannot both be present with a different declaration.
    let border = if guest.contains(&AddressMode::ClampToZero) {
        BorderColor::TransparentBlack
    } else {
        state.border_color()
    };

    let (min_lod, max_lod) = state.lod_clamp();
    let (mipmap_mode, min_lod, max_lod) = match state.mip_filter() {
        // Not a filter but a level clamp: a maximum level of detail of zero
        // makes every sample come from the top level whatever the chain holds.
        // Keeping the guest's clamp would sample lower levels wherever it
        // allowed them.
        MipFilter::NotMipmapped => (vk::SamplerMipmapMode::NEAREST, 0.0, 0.0),
        MipFilter::Nearest => (vk::SamplerMipmapMode::NEAREST, min_lod, max_lod),
        MipFilter::Linear => (vk::SamplerMipmapMode::LINEAR, min_lod, max_lod),
    };

    // Clamped rather than refused: Metal clamps to the device maximum too, and
    // the difference is a quality one rather than a wrong image. The value
    // used is carried so a census can price it.
    let requested = state.max_anisotropy() as f32;
    let anisotropy_enable = requested > 1.0 && cell.anisotropy;
    let max_anisotropy = if anisotropy_enable {
        requested.min(cell.max_anisotropy.max(1.0))
    } else {
        1.0
    };

    Ok(SamplerPlan {
        mag_filter: filter(state.mag_filter()),
        min_filter: filter(state.min_filter()),
        mipmap_mode,
        address,
        anisotropy_enable,
        max_anisotropy,
        compare_enable: state.compare().is_some(),
        compare_op: state.compare().map_or(vk::CompareOp::NEVER, compare_op),
        min_lod,
        max_lod,
        border_color: border_color(border),
        unnormalized_coordinates: !state.normalized_coordinates(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::sampler::{SamplerShape, MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR};
    use reims_vgpu_core::sampler::{
        MTL_COMPARE_FUNCTION_LESS_EQUAL, MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO, MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
        MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT, MTL_SAMPLER_ADDRESS_MODE_REPEAT,
        MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE, MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
        MTL_SAMPLER_MIN_MAG_FILTER_LINEAR, MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
        MTL_SAMPLER_MIP_FILTER_LINEAR, MTL_SAMPLER_MIP_FILTER_NEAREST,
        MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
    };
    use std::collections::BTreeSet;

    fn all() -> SamplerCell {
        SamplerCell {
            mirror_clamp_to_edge: true,
            anisotropy: true,
            max_anisotropy: 16.0,
        }
    }

    fn shape() -> SamplerShape {
        SamplerShape {
            min_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mag_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mip_filter: MTL_SAMPLER_MIP_FILTER_LINEAR,
            s_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            t_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            r_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            max_anisotropy: 1,
            lod_min_clamp: 0.0,
            lod_max_clamp: 8.0,
            compare_function: 0,
            compare_enabled: false,
            border_color: MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
            normalized_coordinates: true,
        }
    }

    fn planned(shape: SamplerShape, cell: SamplerCell) -> Result<SamplerPlan, Refusal> {
        plan(
            shape.checked().expect("a declaration the guest API admits"),
            cell,
        )
    }

    #[test]
    fn every_filter_and_comparison_maps_one_to_one() {
        for guest in Filter::ALL {
            let mapped = filter(guest);
            assert!(mapped == vk::Filter::NEAREST || mapped == vk::Filter::LINEAR);
        }
        assert_eq!(filter(Filter::Nearest), vk::Filter::NEAREST);
        assert_eq!(filter(Filter::Linear), vk::Filter::LINEAR);

        let ops: BTreeSet<i32> = CompareFunction::ALL
            .iter()
            .map(|f| compare_op(*f).as_raw())
            .collect();
        // Eight distinct comparisons, so no two guest functions collapse onto
        // one Vulkan operation.
        assert_eq!(ops.len(), CompareFunction::ALL.len());
        assert_eq!(
            compare_op(CompareFunction::LessEqual),
            vk::CompareOp::LESS_OR_EQUAL
        );
    }

    /// Every comparison lands on the Vulkan operation of the same name.
    ///
    /// Injectivity above says no two guest comparisons collapse onto one
    /// Vulkan operation. It says nothing about a *swap*: `Less` bound to
    /// `GREATER` and `Greater` to `LESS` is still eight distinct operations,
    /// and one pinned arm cannot catch it. A swapped depth comparison is a
    /// frame that draws inside out, on every guest, silently.
    ///
    /// So the answer is checked against its own name. The derivation shares
    /// nothing with the table: it reads the guest variant's spelling, applies
    /// the one vocabulary rule Metal and Vulkan disagree about, and compares
    /// with what the `VkCompareOp` calls itself.
    #[test]
    fn every_comparison_lands_on_the_operation_of_the_same_name() {
        for guest in CompareFunction::ALL {
            assert_eq!(
                format!("{:?}", compare_op(guest)),
                crate::naming::vulkan_spelling(&format!("{guest:?}")),
                "{guest:?}"
            );
        }
    }

    #[test]
    fn every_border_colour_maps_to_a_float_variant() {
        for guest in BorderColor::ALL {
            let mapped = border_color(guest);
            assert!(
                matches!(
                    mapped,
                    vk::BorderColor::FLOAT_TRANSPARENT_BLACK
                        | vk::BorderColor::FLOAT_OPAQUE_BLACK
                        | vk::BorderColor::FLOAT_OPAQUE_WHITE
                ),
                "{}",
                guest.name()
            );
        }
    }

    #[test]
    fn both_border_modes_become_one_vulkan_mode() {
        for guest in [AddressMode::ClampToZero, AddressMode::ClampToBorderColor] {
            assert_eq!(
                address_mode(guest, all()),
                Ok(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            );
        }
    }

    #[test]
    fn the_mirror_clamp_mode_refuses_where_the_feature_is_absent() {
        assert_eq!(
            address_mode(AddressMode::MirrorClampToEdge, SamplerCell::default()),
            Err(Refusal::NoMirrorClampToEdge)
        );
        // Not clamped to `CLAMP_TO_EDGE`, which mirrors nothing and leaves a
        // visible seam.
        assert_eq!(
            address_mode(AddressMode::MirrorClampToEdge, all()),
            Ok(vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE)
        );
        // And the whole sampler refuses, naming that mode.
        assert_eq!(
            planned(
                SamplerShape {
                    t_address: MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
                    ..shape()
                },
                SamplerCell::default(),
            ),
            Err(Refusal::NoMirrorClampToEdge)
        );
    }

    #[test]
    fn the_modes_that_are_not_capabilities_need_none() {
        for (guest, expected) in [
            (
                AddressMode::ClampToEdge,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
            ),
            (AddressMode::Repeat, vk::SamplerAddressMode::REPEAT),
            (
                AddressMode::MirrorRepeat,
                vk::SamplerAddressMode::MIRRORED_REPEAT,
            ),
        ] {
            assert_eq!(address_mode(guest, SamplerCell::default()), Ok(expected));
        }
    }

    #[test]
    fn two_border_modes_with_a_non_transparent_declaration_refuse() {
        let clash = SamplerShape {
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
            t_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
            border_color: MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE,
            ..shape()
        };
        assert_eq!(
            planned(clash, all()),
            Err(Refusal::TwoBorders {
                declared: BorderColor::OpaqueWhite,
            })
        );

        // The same pair with a transparent declaration is one border twice,
        // and it plans.
        let agreed = SamplerShape {
            border_color: MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
            ..clash
        };
        let plan = planned(agreed, all()).expect("one border");
        assert_eq!(plan.border_color, vk::BorderColor::FLOAT_TRANSPARENT_BLACK);
    }

    #[test]
    fn clamp_to_zero_reads_transparent_black_whatever_the_colour_field_says() {
        // The descriptor still carries a colour and this mode ignores it,
        // which is safe precisely because the refusal above rules out the case
        // where another axis wanted that colour.
        let plan = planned(
            SamplerShape {
                s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
                border_color: MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE,
                ..shape()
            },
            all(),
        )
        .expect("one border mode");
        assert_eq!(plan.border_color, vk::BorderColor::FLOAT_TRANSPARENT_BLACK);
    }

    #[test]
    fn not_mipmapped_is_a_level_clamp_and_not_a_mip_mode() {
        let plan = planned(
            SamplerShape {
                mip_filter: MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
                lod_min_clamp: 2.0,
                lod_max_clamp: 9.0,
                ..shape()
            },
            all(),
        )
        .expect("plannable");
        // The guest's clamp is replaced rather than carried: keeping it would
        // sample lower levels wherever it allowed them.
        assert_eq!(plan.min_lod, 0.0);
        assert_eq!(plan.max_lod, 0.0);
        assert_eq!(plan.mipmap_mode, vk::SamplerMipmapMode::NEAREST);
    }

    #[test]
    fn a_mipmapped_sampler_keeps_the_guests_clamp() {
        for (guest, expected) in [
            (
                MTL_SAMPLER_MIP_FILTER_NEAREST,
                vk::SamplerMipmapMode::NEAREST,
            ),
            (MTL_SAMPLER_MIP_FILTER_LINEAR, vk::SamplerMipmapMode::LINEAR),
        ] {
            let plan = planned(
                SamplerShape {
                    mip_filter: guest,
                    lod_min_clamp: 2.0,
                    lod_max_clamp: 9.0,
                    ..shape()
                },
                all(),
            )
            .expect("plannable");
            assert_eq!(plan.mipmap_mode, expected);
            assert_eq!(plan.min_lod, 2.0);
            assert_eq!(plan.max_lod, 9.0);
        }
    }

    #[test]
    fn anisotropy_is_clamped_to_the_device_and_off_without_the_feature() {
        let asked = SamplerShape {
            max_anisotropy: 16,
            ..shape()
        };
        // Clamped down rather than refused: Metal clamps too, and the
        // difference is quality rather than a wrong image.
        let clamped = planned(
            asked,
            SamplerCell {
                anisotropy: true,
                max_anisotropy: 4.0,
                ..all()
            },
        )
        .expect("plannable");
        assert!(clamped.anisotropy_enable);
        assert_eq!(clamped.max_anisotropy, 4.0);

        let honoured = planned(asked, all()).expect("plannable");
        assert_eq!(honoured.max_anisotropy, 16.0);

        // Without the feature it is off, and the value is the one Vulkan
        // requires when it is off.
        let absent = planned(
            asked,
            SamplerCell {
                anisotropy: false,
                ..all()
            },
        )
        .expect("plannable");
        assert!(!absent.anisotropy_enable);
        assert_eq!(absent.max_anisotropy, 1.0);

        // A sampler that never asked keeps it off on a device that has it.
        let unasked = planned(shape(), all()).expect("plannable");
        assert!(!unasked.anisotropy_enable);
    }

    #[test]
    fn a_comparison_reaches_the_plan_only_where_one_was_enabled() {
        let none = planned(shape(), all()).expect("plannable");
        assert!(!none.compare_enable);
        assert_eq!(none.compare_op, vk::CompareOp::NEVER);

        let some = planned(
            SamplerShape {
                compare_enabled: true,
                compare_function: MTL_COMPARE_FUNCTION_LESS_EQUAL,
                ..shape()
            },
            all(),
        )
        .expect("plannable");
        assert!(some.compare_enable);
        assert_eq!(some.compare_op, vk::CompareOp::LESS_OR_EQUAL);
    }

    #[test]
    fn the_axes_reach_the_plan_in_the_guests_order() {
        let plan = planned(
            SamplerShape {
                s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                t_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
                r_address: MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT,
                ..shape()
            },
            all(),
        )
        .expect("plannable");
        assert_eq!(
            plan.address,
            [
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                vk::SamplerAddressMode::REPEAT,
                vk::SamplerAddressMode::MIRRORED_REPEAT,
            ]
        );
        let info = plan.create_info();
        assert_eq!(info.address_mode_u, plan.address[0]);
        assert_eq!(info.address_mode_v, plan.address[1]);
        assert_eq!(info.address_mode_w, plan.address[2]);
    }

    #[test]
    fn an_unnormalized_sampler_reaches_the_plan_as_one() {
        let plan = planned(
            SamplerShape {
                min_filter: MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
                mag_filter: MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
                mip_filter: MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
                s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                t_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                r_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                normalized_coordinates: false,
                ..shape()
            },
            all(),
        )
        .expect("plannable");
        assert!(plan.unnormalized_coordinates);
        // Vulkan's own restrictions on the mode, which the protocol layer
        // already enforced: one level, no anisotropy, no comparison.
        assert_eq!(plan.min_lod, 0.0);
        assert_eq!(plan.max_lod, 0.0);
        assert!(!plan.anisotropy_enable);
        assert!(!plan.compare_enable);
        assert_eq!(plan.min_filter, plan.mag_filter);
    }

    /// Every unnormalized declaration the protocol layer admits plans to a
    /// sampler Vulkan will create.
    ///
    /// A sweep rather than a case, because the six restrictions
    /// `VkSamplerCreateInfo` places on `unnormalizedCoordinates` are checked
    /// by the driver and not by this type: one declaration that reaches
    /// `vkCreateSampler` breaking any of them is an invalid call. The protocol
    /// layer conforms the declaration and this asserts the conformance is the
    /// one Vulkan asks for — the two are stated in different vocabularies, so
    /// agreement between them is not something either side can assume.
    ///
    /// Every axis mode, both filters and every mip filter, against a device
    /// that offers everything — the permissive cell, because a capability that
    /// refuses is a refusal and this is about the declarations that plan.
    #[test]
    fn every_unnormalized_declaration_that_plans_satisfies_the_vulkan_rules() {
        let modes = [
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
            MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT,
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
        ];
        let mut planned_count = 0usize;
        for s_address in modes {
            for t_address in modes {
                for min_filter in [
                    MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
                    MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
                ] {
                    for mip_filter in [
                        MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
                        MTL_SAMPLER_MIP_FILTER_NEAREST,
                        MTL_SAMPLER_MIP_FILTER_LINEAR,
                    ] {
                        let plan = planned(
                            SamplerShape {
                                min_filter,
                                mag_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
                                mip_filter,
                                s_address,
                                t_address,
                                r_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
                                max_anisotropy: 16,
                                lod_min_clamp: 1.0,
                                lod_max_clamp: 9.0,
                                normalized_coordinates: false,
                                ..shape()
                            },
                            all(),
                        )
                        .expect("an unnormalized declaration the protocol admits");
                        planned_count += 1;
                        // `-01072`, `-01073`, `-01074`, `-01076`, `-01077`.
                        assert_eq!(plan.min_filter, plan.mag_filter);
                        assert_eq!(plan.mipmap_mode, vk::SamplerMipmapMode::NEAREST);
                        assert_eq!((plan.min_lod, plan.max_lod), (0.0, 0.0));
                        assert!(!plan.anisotropy_enable);
                        assert!(!plan.compare_enable);
                        // `-01075`, which names U and V and not W: an
                        // unnormalized sampler reads a 2D view, so the third
                        // axis is never consulted and is not restricted.
                        for axis in &plan.address[..2] {
                            assert!(
                                *axis == vk::SamplerAddressMode::CLAMP_TO_EDGE
                                    || *axis == vk::SamplerAddressMode::CLAMP_TO_BORDER,
                                "address {axis:?} is not one -01075 allows"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(planned_count, modes.len() * modes.len() * 2 * 3);
    }

    #[test]
    fn a_create_info_carries_the_plan() {
        let plan = planned(shape(), all()).expect("plannable");
        let info = plan.create_info();
        assert_eq!(info.mag_filter, plan.mag_filter);
        assert_eq!(info.min_filter, plan.min_filter);
        assert_eq!(info.mipmap_mode, plan.mipmap_mode);
        assert_eq!(info.min_lod, plan.min_lod);
        assert_eq!(info.max_lod, plan.max_lod);
        assert_eq!(info.border_color, plan.border_color);
        assert_eq!(info.mip_lod_bias, 0.0);
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::TwoBorders {
                declared: BorderColor::OpaqueWhite,
            },
            Refusal::NoMirrorClampToEdge,
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_sampler_"));
        }
    }
}
