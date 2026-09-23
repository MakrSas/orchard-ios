//! `MTLVertexStepFunction` ordinals, and the step rate that pairs with each.
//!
//! A serializer-object pipeline descriptor's buffer layout carries a step function and a
//! step rate, decoded by `crate::runtime::decode::resource` into
//! `VertexAttribute::step_function_ordinal` and `VertexAttribute::step_rate`.
//! The two are one rule and not two fields: `MTLVertexBufferLayoutDescriptor`
//! requires `stepRate == 0` for `MTLVertexStepFunctionConstant` and rejects it
//! for every other step function, because a constant-rate attribute is fetched
//! once for the whole draw and a rate of zero is how that is spelled.
//!
//! # Why the pair lives here
//!
//! Both backends narrow it and they disagreed. The Metal arm asked
//! `rate == 0 && step != Constant`; the Vulkan arm's draw validation asked
//! `rate == 0` alone, so it declined the canonical Constant spelling — with
//! `vk_draw_validate_zero_vertex_step_rate`, a decline that loses the whole
//! draw — while ignoring the rate for exactly that step function everywhere
//! downstream (its divisor is 0 whatever the rate says). The decoder's own doc
//! settles which arm was right: "a layout that declared **zero** means zero —
//! that is what `MTLVertexStepFunctionConstant` pairs with — so nothing here
//! clamps it up".
//!
//! The ordinals are here for a second reason. `backend::metal` reached them
//! through `metal` 0.33's `MTLVertexStepFunction` discriminants, and that crate
//! is measured to number the *sibling* `MTLStepFunction` wrongly in six of nine
//! places — see `backend::metal::mtl_enum`, which carries the table and now also
//! carries `const` assertions pinning this enum's five discriminants to the
//! ordinals below. (Named in prose, not linked: that module is
//! `backend-metal`-gated, so a link from here is unresolved on every Vulkan-arm
//! doc build.)

/// `MTLVertexStepFunctionConstant` — one fetch for the whole draw.
pub const MTL_VERTEX_STEP_FUNCTION_CONSTANT: u32 = 0;
/// `MTLVertexStepFunctionPerVertex`.
pub const MTL_VERTEX_STEP_FUNCTION_PER_VERTEX: u32 = 1;
/// `MTLVertexStepFunctionPerInstance` — the rate is the instance divisor.
pub const MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE: u32 = 2;
/// `MTLVertexStepFunctionPerPatch` — tessellation only.
pub const MTL_VERTEX_STEP_FUNCTION_PER_PATCH: u32 = 3;
/// `MTLVertexStepFunctionPerPatchControlPoint` — tessellation only.
pub const MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT: u32 = 4;

/// Whether the `(step function, step rate)` pair is one Metal accepts.
///
/// Zero is legal for exactly one step function and required by it. Under any
/// other, a zero rate advances nothing and `MTLVertexDescriptor` validation
/// rejects the descriptor — so refusing it by name is a report, not a policy.
///
/// The step function is taken as its raw ordinal rather than a narrowed type on
/// purpose: an undeclared ordinal has its own refusal at each backend, and this
/// predicate must not double as that one. It answers only "is the rate right for
/// this step", and for an ordinal neither backend accepts the answer is the same
/// as for `PerVertex` — which is what a caller checking the pair before the
/// ordinal would want anyway.
#[must_use]
pub fn step_rate_in_contract(step_function_ordinal: u32, step_rate: u32) -> bool {
    step_rate != 0 || step_function_ordinal == MTL_VERTEX_STEP_FUNCTION_CONSTANT
}

/// `MTLVertexStepFunction`, parsed.
///
/// A total type rather than a raw ordinal, so a rail cannot fold an
/// unrecognised value onto `PerVertex` — which fetches per vertex where the
/// guest asked per instance, drawing one object's worth of geometry for a
/// whole instanced draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StepFunction {
    /// One fetch for the whole draw. Pairs with a rate of zero and only that.
    Constant,
    PerVertex,
    PerInstance,
    PerPatch,
    PerPatchControlPoint,
}

impl StepFunction {
    pub const ALL: [StepFunction; 5] = [
        Self::Constant,
        Self::PerVertex,
        Self::PerInstance,
        Self::PerPatch,
        Self::PerPatchControlPoint,
    ];

    /// Metal's own default, and what an absent field means.
    ///
    /// The serializer omits the field for `PerVertex`, so absence is part of
    /// the contract rather than a caller-side fallback — with one exception a
    /// caller does own: a post-tessellation vertex descriptor defaults to
    /// indexing control points instead, which is why
    /// [`crate::decode`]'s reader takes the default it applies.
    pub const DEFAULT: Self = Self::PerVertex;

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_VERTEX_STEP_FUNCTION_CONSTANT => Self::Constant,
            MTL_VERTEX_STEP_FUNCTION_PER_VERTEX => Self::PerVertex,
            MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE => Self::PerInstance,
            MTL_VERTEX_STEP_FUNCTION_PER_PATCH => Self::PerPatch,
            MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT => Self::PerPatchControlPoint,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::Constant => MTL_VERTEX_STEP_FUNCTION_CONSTANT,
            Self::PerVertex => MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
            Self::PerInstance => MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
            Self::PerPatch => MTL_VERTEX_STEP_FUNCTION_PER_PATCH,
            Self::PerPatchControlPoint => MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT,
        }
    }

    /// Whether this step function only means anything inside a tessellation
    /// pipeline.
    ///
    /// A recognised value that a rail may still decline — and the distinction
    /// matters: declining it as *unknown* would say the stream was malformed
    /// when what happened is that this device builds no tessellation pipeline
    /// for the attribute to belong to.
    #[must_use]
    pub const fn is_tessellation(self) -> bool {
        matches!(self, Self::PerPatch | Self::PerPatchControlPoint)
    }

    /// Whether a rate of zero is the rate this step function takes.
    ///
    /// The typed half of [`step_rate_in_contract`], which stays as it is for
    /// the callers that hold a raw ordinal.
    #[must_use]
    pub const fn takes_zero_rate(self) -> bool {
        matches!(self, Self::Constant)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::PerVertex => "per_vertex",
            Self::PerInstance => "per_instance",
            Self::PerPatch => "per_patch",
            Self::PerPatchControlPoint => "per_patch_control_point",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero is in contract for Constant and for nothing else.
    ///
    /// Swept over every declared ordinal and past the top of the enum, because
    /// the rule is an equality against one ordinal and the cheap wrong version
    /// of it is a `<=` band.
    #[test]
    fn a_zero_step_rate_pairs_with_constant_and_nothing_else() {
        assert!(step_rate_in_contract(MTL_VERTEX_STEP_FUNCTION_CONSTANT, 0));
        for step in [
            MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
            MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
            MTL_VERTEX_STEP_FUNCTION_PER_PATCH,
            MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT,
        ] {
            assert!(
                !step_rate_in_contract(step, 0),
                "step function {step} does not pair with a zero rate"
            );
        }
        for step in 5..=64u32 {
            assert!(!step_rate_in_contract(step, 0));
        }
        assert!(!step_rate_in_contract(u32::MAX, 0));
    }

    /// A nonzero rate is in contract under every step function, including the
    /// one that ignores it.
    ///
    /// Constant with a rate of 1 is what the tree's own Vulkan validation test
    /// was built from, and it is not what a guest sends — but nothing rejects
    /// it: `MTLVertexBufferLayoutDescriptor` only constrains the zero, and this
    /// predicate is not the place to invent a second constraint.
    #[test]
    fn a_nonzero_rate_is_in_contract_under_every_step_function() {
        for step in 0..=8u32 {
            for rate in [1u32, 2, 7, u32::MAX] {
                assert!(step_rate_in_contract(step, rate));
            }
        }
    }

    /// The typed step function and the raw predicate answer the same question,
    /// so neither can drift from the other.
    #[test]
    fn the_typed_step_function_agrees_with_the_raw_predicate() {
        for step in StepFunction::ALL {
            assert_eq!(StepFunction::parse(step.ordinal()), Some(step));
            assert_eq!(
                step.takes_zero_rate(),
                step_rate_in_contract(step.ordinal(), 0)
            );
        }
        assert_eq!(StepFunction::parse(5), None);
        assert_eq!(StepFunction::DEFAULT, StepFunction::PerVertex);
    }

    /// Recognised and declinable are different things. The two tessellation
    /// step functions are in the enum, and a rail that builds no tessellation
    /// pipeline says so rather than calling them unknown.
    #[test]
    fn exactly_the_two_patch_step_functions_are_tessellation_only() {
        let tess: alloc::vec::Vec<u32> = StepFunction::ALL
            .iter()
            .filter(|s| s.is_tessellation())
            .map(|s| s.ordinal())
            .collect();
        assert_eq!(tess, alloc::vec![3, 4]);
        for step in [
            StepFunction::Constant,
            StepFunction::PerVertex,
            StepFunction::PerInstance,
        ] {
            assert!(!step.is_tessellation());
        }
    }
}
