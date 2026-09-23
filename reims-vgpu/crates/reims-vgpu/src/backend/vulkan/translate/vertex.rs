//! What a layout entry's `MTLVertexStepFunction` field means when it is absent.
//!
//! The serializer omits the field for Metal's own default, so absence is part
//! of the translation rather than a caller-side fallback — that, and the two
//! refusal names, are all that is left here.
//!
//! The format table that used to live above it — `MTLVertexFormat` → an engine
//! enum → `VkFormat`, with the byte size beside it — is
//! `reims_vgpu_core::vertex_format::VertexFormat` and
//! `reims_vgpu_vulkan::vertex::format`. The reason the two halves were together
//! still holds: a `Short3` occupies six bytes *because* it is three 16-bit
//! components, which is also why it is `R16G16B16_UINT`. The owning type states
//! the size and the component count, the rail states the spelling, and neither
//! can drift from a copy here because there is no copy here. The signedness
//! coupling moved with them — twelve signed Metal formats bind **unsigned**
//! Vulkan formats, correct only while the shader translator emits unsigned
//! integer stage inputs, and `reims_vgpu_vulkan::vertex::SIGNED_AS_UNSIGNED` is
//! where that blast radius is enumerated.

use super::reason::TranslateReason;
use crate::backend::vulkan::engine::VertexStepFunction;

/// A layout entry's declared `MTLVertexStepFunction` → engine step mode.
///
/// The serializer omits the field for Metal's default `PerVertex` behavior, so
/// absence is part of this translation rather than a caller-side fallback.
///
/// The SDK enum runs 0-4. Only 0-2 have a `VkVertexInputRate` — Vulkan has
/// `VERTEX` and `INSTANCE` and nothing else — so 3 (`PerPatch`) and 4
/// (`PerPatchControlPoint`) decline, but under their own reason rather than as
/// unrecognised values. They are recognised; this backend builds no
/// tessellation pipeline for them to belong to.
pub fn step_function(declared: Option<u32>) -> Result<VertexStepFunction, TranslateReason> {
    let Some(mtl) = declared else {
        return Ok(VertexStepFunction::DEFAULT);
    };
    let step =
        VertexStepFunction::parse(mtl).ok_or(TranslateReason::UnknownVertexStepFunction(mtl))?;
    // Which step functions have a `VkVertexInputRate` is a fact about Vulkan,
    // not about `MTLVertexStepFunction`, so it is asked of the rail that owns
    // it rather than restated as two more arms of the parse above. The refusal
    // is translated into this layer's vocabulary because this is the early
    // gate: a draw refused here never reaches a pipeline key.
    reims_vgpu_vulkan::vertex::input_rate(step)
        .map(|_| step)
        .map_err(|_| TranslateReason::VertexStepFunctionPerPatch(mtl))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_vertex_step_function_maps_and_absence_is_per_vertex() {
        assert_eq!(step_function(None).unwrap(), VertexStepFunction::PerVertex);
        assert_eq!(
            step_function(Some(0)).unwrap(),
            VertexStepFunction::Constant
        );
        assert_eq!(
            step_function(Some(1)).unwrap(),
            VertexStepFunction::PerVertex
        );
        assert_eq!(
            step_function(Some(2)).unwrap(),
            VertexStepFunction::PerInstance
        );
        assert_eq!(
            step_function(Some(5)).unwrap_err(),
            TranslateReason::UnknownVertexStepFunction(5)
        );
    }

    /// Every ordinal this backend accepts survives the round trip back to the
    /// wire value it came from.
    ///
    /// [`VertexStepFunction::mtl_ordinal`] exists so a rule stated over the
    /// guest's ordinal can be asked on this side — the step/rate pair in
    /// `protocol::vertex_step` is the one that does — and a rule asked through
    /// an inverse that is not an inverse is a rule asked about a different
    /// attribute. The three accepted ordinals are named from the contract here
    /// rather than spelled again, so this also pins that the `match` above
    /// agrees with the declaration.
    #[test]
    fn an_accepted_step_function_round_trips_to_its_own_ordinal() {
        use crate::protocol::vertex_step as step;
        for ordinal in [
            step::MTL_VERTEX_STEP_FUNCTION_CONSTANT,
            step::MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
            step::MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
        ] {
            let translated = step_function(Some(ordinal)).expect("an accepted ordinal");
            assert_eq!(
                translated.ordinal(),
                ordinal,
                "{translated:?} came from {ordinal}"
            );
        }
    }

    /// The two tessellation step rates decline under their own reason, and the
    /// first value genuinely off the end of the SDK enum declines under the
    /// other.
    ///
    /// This test asserted `UnknownVertexStepFunction(3)` before, which encoded
    /// the wrong end of `MTLVertexStepFunction` as the intended behaviour: the
    /// enum runs to `PerPatchControlPoint = 4`, so 3 and 4 are declared values
    /// this backend recognises and cannot spell, not values it fails to
    /// recognise. Reading a boot's log, the two want different answers — one is
    /// "the guest ran a tessellation pipeline", the other is "something is
    /// wrong upstream of here".
    #[test]
    fn the_two_tessellation_step_rates_decline_by_their_own_name() {
        for mtl in [3u32, 4] {
            assert_eq!(
                step_function(Some(mtl)).unwrap_err(),
                TranslateReason::VertexStepFunctionPerPatch(mtl),
                "MTLVertexStepFunction {mtl}"
            );
        }
        use crate::observe::Decline as _;
        assert_eq!(
            TranslateReason::VertexStepFunctionPerPatch(3).slug(),
            "vertex_step_function_per_patch"
        );
        assert_ne!(
            TranslateReason::VertexStepFunctionPerPatch(3).slug(),
            TranslateReason::UnknownVertexStepFunction(5).slug()
        );
        // Absence is its own state now rather than a flag beside a word, so a
        // record that never carried the field cannot reach either refusal.
        assert_eq!(step_function(None).unwrap(), VertexStepFunction::PerVertex);
    }
}
