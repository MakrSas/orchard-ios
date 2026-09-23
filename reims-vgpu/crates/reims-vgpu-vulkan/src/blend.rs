//! A checked colour attachment as `VkPipelineColorBlendAttachmentState` would
//! hold it, and the one device feature four of its factors need.
//!
//! # The blend colour is dynamic state
//!
//! `MTLBlendFactorBlendColor` and its three siblings read a value the guest
//! sets on the encoder with `setBlendColorRed:…`, so it changes without the
//! pipeline changing. It is `VK_DYNAMIC_STATE_BLEND_CONSTANTS` here and is
//! supplied per draw. Baking it into the plan would key a pipeline cache on a
//! value that is not part of the pipeline, and a guest animating a fade would
//! compile a pipeline per frame.
//!
//! # Dual-source blending is a capability, and asking is not translating
//!
//! The four `Source1` factors need `VkPhysicalDeviceFeatures::dualSrcBlend`,
//! and a pipeline naming one without it is invalid. That is a fact about the
//! host, so it is a [`BlendCell`] read from the census and a typed refusal —
//! not a parse failure. Whether the guest asked for a dual-source blend and
//! whether this host can run one are two different questions, and answering
//! the first with the second reports a capability gap as a malformed stream.
//!
//! The device's own limit on how many attachments may blend *differently*
//! (`independentBlend`) is the second half of the same cell. Metal declares
//! blending per attachment with no such restriction, so a host without it can
//! only serve a pipeline whose attachments all agree — and that is a decision
//! about a whole pipeline rather than about one attachment, which is why
//! [`independent`] takes the list.
//!
//! # Planned, not created
//!
//! Nothing here builds a pipeline. Every mapping is tested with no GPU.

use ash::vk;
use reims_vgpu_core::blend::{
    Blend, BlendFactor, BlendOperation, ColorAttachmentState, ColorWriteMask,
};

/// What this host offers for the parts of a blend that are not mappings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlendCell {
    /// `VkPhysicalDeviceFeatures::dualSrcBlend`.
    pub dual_source: bool,
    /// `VkPhysicalDeviceFeatures::independentBlend`.
    pub independent: bool,
}

/// Why a colour attachment cannot be built here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The guest named a factor reading the second fragment output and this
    /// device has no `dualSrcBlend`.
    NoDualSource { factor: BlendFactor },
    /// Attachments declare different blend states and this device has no
    /// `independentBlend`.
    NoIndependentBlend { slots: u32 },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoDualSource { .. } => "vk_blend_no_dual_source",
            Self::NoIndependentBlend { .. } => "vk_blend_no_independent_blend",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDualSource { factor } => {
                write!(f, "{} ordinal={}", self.slug(), factor.ordinal())
            }
            Self::NoIndependentBlend { slots } => write!(f, "{} slots={slots}", self.slug()),
        }
    }
}

/// `MTLBlendFactor` → `VkBlendFactor`.
///
/// Total, and written out rather than cast. Metal's constant-colour factors
/// are Vulkan's `CONSTANT_*`, which is the same encoder value under a
/// different name; the orders happen to agree for the first eleven and do not
/// past them, which is exactly the kind of coincidence a cast would enshrine.
#[must_use]
pub const fn factor(guest: BlendFactor) -> vk::BlendFactor {
    match guest {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SourceColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSourceColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::SourceAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSourceAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DestinationColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDestinationColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::DestinationAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDestinationAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        BlendFactor::SourceAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        BlendFactor::BlendColor => vk::BlendFactor::CONSTANT_COLOR,
        BlendFactor::OneMinusBlendColor => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        BlendFactor::BlendAlpha => vk::BlendFactor::CONSTANT_ALPHA,
        BlendFactor::OneMinusBlendAlpha => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
        BlendFactor::Source1Color => vk::BlendFactor::SRC1_COLOR,
        BlendFactor::OneMinusSource1Color => vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
        BlendFactor::Source1Alpha => vk::BlendFactor::SRC1_ALPHA,
        BlendFactor::OneMinusSource1Alpha => vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
    }
}

/// `MTLBlendOperation` → `VkBlendOp`. Total.
#[must_use]
pub const fn operation(guest: BlendOperation) -> vk::BlendOp {
    match guest {
        BlendOperation::Add => vk::BlendOp::ADD,
        BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOperation::Min => vk::BlendOp::MIN,
        BlendOperation::Max => vk::BlendOp::MAX,
    }
}

/// `MTLColorWriteMask` → `VkColorComponentFlags`.
///
/// Metal's bits run alpha-first from the low end; Vulkan's run red-first. The
/// two are bit-reversed over four bits, not equal, so a cast would swap red
/// and alpha and exchange green and blue — an alpha-only mask would write red
/// only. Total by construction: the input's only constructor range-checks.
#[must_use]
pub fn write_mask(mask: ColorWriteMask) -> vk::ColorComponentFlags {
    let mut out = vk::ColorComponentFlags::empty();
    for (present, flag) in [
        (mask.red(), vk::ColorComponentFlags::R),
        (mask.green(), vk::ColorComponentFlags::G),
        (mask.blue(), vk::ColorComponentFlags::B),
        (mask.alpha(), vk::ColorComponentFlags::A),
    ] {
        if present {
            out |= flag;
        }
    }
    out
}

/// One attachment, as `VkPipelineColorBlendAttachmentState` would hold it.
///
/// Spelled out rather than held as the ash structure, which is not `Eq` — and
/// a translation whose result cannot be compared is one whose mappings cannot
/// be asserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttachmentPlan {
    pub blend_enable: bool,
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
    pub color_write_mask: vk::ColorComponentFlags,
}

impl AttachmentPlan {
    pub const fn native(self) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState {
            blend_enable: if self.blend_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            src_color_blend_factor: self.src_color_blend_factor,
            dst_color_blend_factor: self.dst_color_blend_factor,
            color_blend_op: self.color_blend_op,
            src_alpha_blend_factor: self.src_alpha_blend_factor,
            dst_alpha_blend_factor: self.dst_alpha_blend_factor,
            alpha_blend_op: self.alpha_blend_op,
            color_write_mask: self.color_write_mask,
        }
    }
}

/// The factors a disabled attachment carries.
///
/// Vulkan ignores every one of them with `blendEnable` clear, so they are
/// fixed rather than taken from the declaration: two attachments that behave
/// identically then plan identically, and a cache keyed on the plan compiles
/// one pipeline rather than one per unused equation.
const REPLACE: (vk::BlendFactor, vk::BlendFactor, vk::BlendOp) = (
    vk::BlendFactor::ONE,
    vk::BlendFactor::ZERO,
    vk::BlendOp::ADD,
);

/// Translate one checked colour attachment.
///
/// # Errors
///
/// [`Refusal::NoDualSource`] when a factor needs a feature this device lacks,
/// with nothing partially translated.
pub fn plan(state: &ColorAttachmentState, cell: BlendCell) -> Result<AttachmentPlan, Refusal> {
    let mask = write_mask(state.write_mask());
    let Some(blend) = state.blend() else {
        return Ok(AttachmentPlan {
            blend_enable: false,
            src_color_blend_factor: REPLACE.0,
            dst_color_blend_factor: REPLACE.1,
            color_blend_op: REPLACE.2,
            src_alpha_blend_factor: REPLACE.0,
            dst_alpha_blend_factor: REPLACE.1,
            alpha_blend_op: REPLACE.2,
            color_write_mask: mask,
        });
    };
    if !cell.dual_source {
        // Named individually so the refusal says which of the four the guest
        // asked for, and checked before anything is built.
        for candidate in [
            blend.src_color,
            blend.dst_color,
            blend.src_alpha,
            blend.dst_alpha,
        ] {
            if candidate.is_dual_source() {
                return Err(Refusal::NoDualSource { factor: candidate });
            }
        }
    }
    Ok(AttachmentPlan {
        blend_enable: true,
        src_color_blend_factor: factor(blend.src_color),
        dst_color_blend_factor: factor(blend.dst_color),
        color_blend_op: operation(blend.color_operation),
        src_alpha_blend_factor: factor(blend.src_alpha),
        dst_alpha_blend_factor: factor(blend.dst_alpha),
        alpha_blend_op: operation(blend.alpha_operation),
        color_write_mask: mask,
    })
}

/// Check a whole pipeline's attachments against `independentBlend`.
///
/// Vulkan requires every attachment's state to be identical without the
/// feature, and Metal places no such restriction on what a guest may declare,
/// so this is a refusal rather than a translation. Separate from [`plan`]
/// because it is a property of the list: no single attachment is the one that
/// broke it.
///
/// # Errors
///
/// [`Refusal::NoIndependentBlend`] naming how many attachments disagreed.
pub fn independent(plans: &[AttachmentPlan], cell: BlendCell) -> Result<(), Refusal> {
    if cell.independent || plans.len() < 2 {
        return Ok(());
    }
    let first = plans[0];
    let disagreeing = plans.iter().filter(|p| **p != first).count();
    if disagreeing == 0 {
        return Ok(());
    }
    Err(Refusal::NoIndependentBlend {
        slots: u32::try_from(plans.len()).unwrap_or(u32::MAX),
    })
}

/// Whether any attachment in a pipeline reads the encoder's blend colour.
///
/// The rail declares `VK_DYNAMIC_STATE_BLEND_CONSTANTS` regardless — it costs
/// nothing and removes the value from the cache key — so this answers only
/// whether the value the guest set has to be pushed before a draw.
#[must_use]
pub fn reads_blend_color(states: &[ColorAttachmentState]) -> bool {
    states
        .iter()
        .filter_map(ColorAttachmentState::blend)
        .any(|b: Blend| b.reads_blend_color())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::blend::{
        ColorAttachmentShape, MTL_BLEND_FACTOR_ONE, MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_1_ALPHA,
        MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA, MTL_BLEND_FACTOR_SOURCE_1_COLOR,
        MTL_BLEND_FACTOR_SOURCE_ALPHA, MTL_BLEND_FACTOR_ZERO, MTL_BLEND_OPERATION_ADD,
        MTL_BLEND_OPERATION_REVERSE_SUBTRACT, MTL_COLOR_WRITE_MASK_ALPHA, MTL_COLOR_WRITE_MASK_RED,
    };
    use std::collections::BTreeSet;

    const ALL: BlendCell = BlendCell {
        dual_source: true,
        independent: true,
    };
    const NEITHER: BlendCell = BlendCell {
        dual_source: false,
        independent: false,
    };

    fn shape() -> ColorAttachmentShape {
        ColorAttachmentShape {
            blending_enabled: true,
            src_rgb: MTL_BLEND_FACTOR_SOURCE_ALPHA,
            dst_rgb: MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
            op_rgb: MTL_BLEND_OPERATION_ADD,
            src_alpha: MTL_BLEND_FACTOR_ONE,
            dst_alpha: MTL_BLEND_FACTOR_ZERO,
            op_alpha: MTL_BLEND_OPERATION_REVERSE_SUBTRACT,
            write_mask: ColorWriteMask::ALL,
        }
    }

    fn state(shape: ColorAttachmentShape) -> ColorAttachmentState {
        shape.checked().expect("a declaration the guest API admits")
    }

    #[test]
    fn every_factor_and_operation_maps_to_a_distinct_one() {
        let factors: BTreeSet<i32> = BlendFactor::ALL
            .iter()
            .map(|f| factor(*f).as_raw())
            .collect();
        assert_eq!(factors.len(), BlendFactor::ALL.len());
        let ops: BTreeSet<i32> = BlendOperation::ALL
            .iter()
            .map(|o| operation(*o).as_raw())
            .collect();
        assert_eq!(ops.len(), BlendOperation::ALL.len());

        // Metal's blend colour is Vulkan's blend constant, and the four
        // dual-source factors are the `SRC1_*` set. Both are the pairs a
        // numeric cast would get wrong.
        assert_eq!(
            factor(BlendFactor::BlendColor),
            vk::BlendFactor::CONSTANT_COLOR
        );
        assert_eq!(
            factor(BlendFactor::Source1Color),
            vk::BlendFactor::SRC1_COLOR
        );
        assert_eq!(
            operation(BlendOperation::ReverseSubtract),
            vk::BlendOp::REVERSE_SUBTRACT
        );
    }

    /// The bit orders are reversed over four bits, so a cast would turn an
    /// alpha-only mask into a red-only one.
    #[test]
    fn the_write_mask_is_reordered_rather_than_passed_through() {
        let red = write_mask(ColorWriteMask::new(MTL_COLOR_WRITE_MASK_RED).expect("in range"));
        assert_eq!(red, vk::ColorComponentFlags::R);
        let alpha = write_mask(ColorWriteMask::new(MTL_COLOR_WRITE_MASK_ALPHA).expect("in range"));
        assert_eq!(alpha, vk::ColorComponentFlags::A);
        assert_ne!(red, alpha);
        assert_eq!(
            write_mask(ColorWriteMask::ALL),
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A
        );
        assert_eq!(
            write_mask(ColorWriteMask::NONE),
            vk::ColorComponentFlags::empty()
        );
    }

    #[test]
    fn a_blending_attachment_carries_both_equations() {
        let plan = plan(&state(shape()), ALL).expect("no dual-source factor");
        assert!(plan.blend_enable);
        assert_eq!(plan.src_color_blend_factor, vk::BlendFactor::SRC_ALPHA);
        assert_eq!(
            plan.dst_color_blend_factor,
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA
        );
        assert_eq!(plan.color_blend_op, vk::BlendOp::ADD);
        assert_eq!(plan.src_alpha_blend_factor, vk::BlendFactor::ONE);
        assert_eq!(plan.dst_alpha_blend_factor, vk::BlendFactor::ZERO);
        assert_eq!(plan.alpha_blend_op, vk::BlendOp::REVERSE_SUBTRACT);
    }

    /// Vulkan ignores every factor with `blendEnable` clear, so two
    /// attachments that behave identically must plan identically.
    #[test]
    fn a_disabled_attachment_plans_the_same_whatever_it_declared() {
        let bare = plan(
            &state(ColorAttachmentShape {
                blending_enabled: false,
                ..ColorAttachmentShape::default()
            }),
            NEITHER,
        )
        .expect("nothing to refuse");
        let with_bytes = plan(
            &state(ColorAttachmentShape {
                blending_enabled: false,
                ..shape()
            }),
            NEITHER,
        )
        .expect("nothing to refuse");
        assert_eq!(bare, with_bytes);
        assert!(!bare.blend_enable);
        assert_eq!(bare.src_color_blend_factor, vk::BlendFactor::ONE);
        assert_eq!(bare.dst_color_blend_factor, vk::BlendFactor::ZERO);

        // The mask still differs, because it is not part of the blend.
        let masked = plan(
            &state(ColorAttachmentShape {
                blending_enabled: false,
                write_mask: ColorWriteMask::NONE,
                ..shape()
            }),
            NEITHER,
        )
        .expect("nothing to refuse");
        assert_ne!(masked, bare);
        assert_eq!(masked.color_write_mask, vk::ColorComponentFlags::empty());
    }

    /// A capability gap is a refusal naming the factor, not a parse failure —
    /// and the same declaration succeeds where the device reports the feature.
    #[test]
    fn each_dual_source_factor_refuses_by_name_without_the_feature() {
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        for (index, ordinal) in [
            MTL_BLEND_FACTOR_SOURCE_1_COLOR,
            MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_1_ALPHA,
        ]
        .into_iter()
        .enumerate()
        {
            for slot in 0..4 {
                let mut bad = shape();
                match slot {
                    0 => bad.src_rgb = ordinal,
                    1 => bad.dst_rgb = ordinal,
                    2 => bad.src_alpha = ordinal,
                    _ => bad.dst_alpha = ordinal,
                }
                let refused = plan(&state(bad), NEITHER).expect_err("no dualSrcBlend here");
                let Refusal::NoDualSource { factor } = refused else {
                    panic!("wrong refusal: {refused}");
                };
                assert_eq!(factor.ordinal(), ordinal);
                seen.insert(ordinal);
                assert_eq!(refused.slug(), "vk_blend_no_dual_source");
                // The same declaration is fine where the device reports it.
                assert!(plan(&state(bad), ALL).is_ok());
                let _ = index;
            }
        }
        assert_eq!(seen.len(), 2);

        // And a declaration with no such factor is unaffected by the cell.
        assert_eq!(
            plan(&state(shape()), NEITHER).expect("no dual-source factor"),
            plan(&state(shape()), ALL).expect("no dual-source factor")
        );
    }

    /// Nothing is refused for a factor that cannot be evaluated: with
    /// blending off, a `SRC1_*` ordinal never reaches a pipeline.
    #[test]
    fn a_dual_source_factor_behind_a_clear_flag_refuses_nothing() {
        let off = state(ColorAttachmentShape {
            blending_enabled: false,
            src_rgb: MTL_BLEND_FACTOR_SOURCE_1_COLOR,
            ..shape()
        });
        assert!(plan(&off, NEITHER).is_ok());
    }

    #[test]
    fn independent_blend_is_a_property_of_the_list_and_not_of_one_attachment() {
        let a = plan(&state(shape()), ALL).expect("legal");
        let b = plan(
            &state(ColorAttachmentShape {
                blending_enabled: false,
                ..shape()
            }),
            ALL,
        )
        .expect("legal");

        // One attachment, or attachments that agree, need no feature.
        assert!(independent(&[a], NEITHER).is_ok());
        assert!(independent(&[a, a, a], NEITHER).is_ok());
        assert!(independent(&[], NEITHER).is_ok());
        // Disagreeing attachments need it.
        let refused = independent(&[a, b], NEITHER).expect_err("no independentBlend here");
        assert_eq!(refused, Refusal::NoIndependentBlend { slots: 2 });
        assert_eq!(refused.slug(), "vk_blend_no_independent_blend");
        // And with the feature they do not.
        assert!(independent(&[a, b], ALL).is_ok());
    }

    /// The blend colour is dynamic state, so nothing a plan holds carries one
    /// — but a caller still has to know whether to push it.
    #[test]
    fn the_blend_colour_is_asked_about_rather_than_planned() {
        let constant = state(ColorAttachmentShape {
            src_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_BLEND_ALPHA,
            ..shape()
        });
        assert!(reads_blend_color(&[constant]));
        assert!(reads_blend_color(&[state(shape()), constant]));
        assert!(!reads_blend_color(&[state(shape())]));
        assert!(!reads_blend_color(&[]));
        // A constant factor behind a clear flag is not read either.
        assert!(!reads_blend_color(&[state(ColorAttachmentShape {
            blending_enabled: false,
            src_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_BLEND_ALPHA,
            ..shape()
        })]));
    }

    #[test]
    fn the_native_state_carries_the_plan() {
        let native = plan(&state(shape()), ALL).expect("legal").native();
        assert_eq!(native.blend_enable, vk::TRUE);
        assert_eq!(native.src_color_blend_factor, vk::BlendFactor::SRC_ALPHA);
        assert_eq!(native.alpha_blend_op, vk::BlendOp::REVERSE_SUBTRACT);
        assert_eq!(native.color_write_mask, vk::ColorComponentFlags::RGBA);

        let off = plan(
            &state(ColorAttachmentShape {
                blending_enabled: false,
                ..shape()
            }),
            ALL,
        )
        .expect("legal")
        .native();
        assert_eq!(off.blend_enable, vk::FALSE);
    }

    /// Every blend factor and operation lands on the Vulkan value of the same
    /// name.
    ///
    /// Nineteen factors written out by hand, most of them one word apart from a
    /// neighbour: `SourceAlpha` beside `OneMinusSourceAlpha`, `SourceColor`
    /// beside `SourceAlpha`, `BlendAlpha` beside `DestinationAlpha`. A swap or
    /// a slip between any pair compiles, plans, and produces a frame blended
    /// with the wrong term --- which looks like a shading bug and not like a
    /// table.
    ///
    /// [`crate::naming`] derives the Vulkan spelling from the guest one through
    /// word-for-word vocabulary --- `SOURCE` is `SRC`, `DESTINATION` is `DST`,
    /// Metal's `BLEND` constant is Vulkan's `CONSTANT` --- so a rule serves
    /// every arm that contains its word and none of them is written down twice.
    #[test]
    fn every_factor_and_operation_lands_on_the_value_of_the_same_name() {
        for guest in BlendFactor::ALL {
            assert_eq!(
                format!("{:?}", factor(guest)),
                crate::naming::vulkan_spelling(&format!("{guest:?}")),
                "{guest:?}"
            );
        }
        for guest in BlendOperation::ALL {
            assert_eq!(
                format!("{:?}", operation(guest)),
                crate::naming::vulkan_spelling(&format!("{guest:?}")),
                "{guest:?}"
            );
        }
    }

    /// And neither table collapses two guest values onto one Vulkan value.
    #[test]
    fn the_factors_and_operations_stay_distinct() {
        let factors: std::collections::BTreeSet<i32> = BlendFactor::ALL
            .iter()
            .map(|f| factor(*f).as_raw())
            .collect();
        assert_eq!(factors.len(), BlendFactor::ALL.len());
        let ops: std::collections::BTreeSet<i32> = BlendOperation::ALL
            .iter()
            .map(|o| operation(*o).as_raw())
            .collect();
        assert_eq!(ops.len(), BlendOperation::ALL.len());
    }
}
