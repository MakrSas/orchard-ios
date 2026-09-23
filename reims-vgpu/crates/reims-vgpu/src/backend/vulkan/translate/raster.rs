//! Cull mode, winding, fill mode, depth clip mode, compare function, stencil
//! operation and index type → their Vulkan spellings.
//!
//! Metal and Vulkan agree on the *ordering* of several of these enums, which
//! makes a numeric cast tempting and wrong: the agreement is a coincidence of
//! two independent specs, not a contract, and a cast turns a future divergence
//! into silently wrong rasterization. Every arm is spelled out so the compiler
//! catches a drift instead of the screen.

use ash::vk;

use super::reason::TranslateReason;
use crate::backend::vulkan::engine::{
    IndexType, SamplerCompareFunction, StencilOp, VisibilityResultMode,
};

/// `MTLCompareFunction`. Depth test, stencil test and sampler compare all carry
/// this same Metal enum.
///
/// **Which ordinal means what is not decided here.** The contract owns that —
/// [`CompareFunction::parse`](crate::protocol::sampler::CompareFunction::parse)
/// is the single place a `MTLCompareFunction` word becomes a comparison — and
/// this is the total correspondence from that answer onto the engine's own
/// spelling. It used to be a second table of bare ordinals, which the compiler
/// could not compare with the first: a contract corrected there would have left
/// this one meaning what it meant before, and the test below would still have
/// passed because it pinned this table against the same literals.
pub fn compare_function(mtl: u32) -> Result<SamplerCompareFunction, TranslateReason> {
    use crate::protocol::sampler::CompareFunction as C;
    let parsed = C::parse(mtl).ok_or(TranslateReason::UnknownCompareFunction(mtl))?;
    Ok(match parsed {
        C::Never => SamplerCompareFunction::Never,
        C::Less => SamplerCompareFunction::Less,
        C::Equal => SamplerCompareFunction::Equal,
        C::LessEqual => SamplerCompareFunction::LessEqual,
        C::Greater => SamplerCompareFunction::Greater,
        C::NotEqual => SamplerCompareFunction::NotEqual,
        C::GreaterEqual => SamplerCompareFunction::GreaterEqual,
        C::Always => SamplerCompareFunction::Always,
    })
}

/// `MTLStencilOperation` (SDK numeric values).
/// `MTLStencilOperation`, from the contract's parse for
/// [`compare_function`]'s reason.
pub fn stencil_operation(mtl: u32) -> Result<StencilOp, TranslateReason> {
    use crate::protocol::depth_stencil::StencilOperation as O;
    let parsed = O::parse(mtl).ok_or(TranslateReason::UnknownStencilOperation(mtl))?;
    Ok(match parsed {
        O::Keep => StencilOp::Keep,
        O::Zero => StencilOp::Zero,
        O::Replace => StencilOp::Replace,
        O::IncrementClamp => StencilOp::IncrementClamp,
        O::DecrementClamp => StencilOp::DecrementClamp,
        O::Invert => StencilOp::Invert,
        O::IncrementWrap => StencilOp::IncrementWrap,
        O::DecrementWrap => StencilOp::DecrementWrap,
    })
}

/// `MTLIndexType` (SDK numeric values).
///
/// The shared runtime loader owns the typed refusal because both Metal and
/// Vulkan consume it; `None` therefore remains a classification here, and the
/// caller turns it into `IndexLoadReason::TypeUnsupported`.
pub fn index_type(mtl: u32) -> Option<IndexType> {
    use crate::protocol::render::IndexType as Contract;
    // The contract carries the ordinal at the width the wire does; a value that
    // does not fit is not an index type by that fact alone, and narrowing it
    // silently would let `0x1_0000` read as `Uint16`.
    Some(match Contract::parse(u16::try_from(mtl).ok()?)? {
        Contract::Uint16 => IndexType::U16,
        Contract::Uint32 => IndexType::U32,
    })
}

pub fn vk_index_type(index: IndexType) -> vk::IndexType {
    match index {
        IndexType::U16 => vk::IndexType::UINT16,
        IndexType::U32 => vk::IndexType::UINT32,
    }
}

/// `MTLVisibilityResultMode` (SDK numeric values) → whether a draw arms an
/// occlusion query, and what it counts.
///
/// `Ok(None)` is `MTLVisibilityResultModeDisabled`, the Metal default: the guest
/// disarmed the query, so the draw runs without one. That is why this returns
/// an `Option` inside the `Result` rather than folding `0` into the error arm —
/// disarming is a thing the guest is entitled to ask for, and an unknown
/// ordinal is not.
pub fn visibility_result_mode(mtl: u32) -> Result<Option<VisibilityResultMode>, TranslateReason> {
    use crate::protocol::visibility::VISIBILITY_RESULT_MODE_DISABLED;
    Ok(match mtl {
        // The one arm that is a *meaning* rather than a mode, so it is the one
        // arm spelled from the contract rather than as a literal beside its
        // neighbours.
        VISIBILITY_RESULT_MODE_DISABLED => None,
        1 => Some(VisibilityResultMode::Boolean),
        2 => Some(VisibilityResultMode::Counting),
        other => return Err(TranslateReason::UnknownVisibilityResultMode(other)),
    })
}

/// The query-control flags an armed mode records with.
///
/// `PRECISE` is what makes the result an exact sample count rather than a
/// non-zero-if-any. It requires `VkPhysicalDeviceFeatures::occlusionQueryPrecise`;
/// the caller gates on it, because the alternative — quietly recording without
/// the bit — hands a counting guest a number that is neither the count nor
/// recognisably wrong, which is the same failure `vk_polygon_mode` above
/// refuses to make for wireframe.
pub fn vk_query_control_flags(mode: VisibilityResultMode) -> vk::QueryControlFlags {
    match mode {
        VisibilityResultMode::Boolean => vk::QueryControlFlags::empty(),
        VisibilityResultMode::Counting => vk::QueryControlFlags::PRECISE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every primitive type this device *advertises* has an arm here.
    ///
    /// [`crate::protocol::draw::EXECUTABLE_PRIMITIVE_TYPES`] is what the guest
    /// reads as permission, so a bit set there without an arm here is a draw the
    /// guest was invited to make and this rail refuses. Both directions are
    /// asserted: an arm without the bit would be a type this device can execute
    /// and has told the guest not to use.
    #[test]
    fn the_advertised_primitive_types_are_the_executable_ones() {
        for mtl in 0..=8u32 {
            assert_eq!(
                reims_vgpu_core::topology::PrimitiveType::parse(mtl).is_some(),
                crate::protocol::draw::primitive_type_executable(mtl),
                "primitive type {mtl}: advertisement and translation disagree"
            );
        }
    }

    /// Every SDK value in range maps, and the first one past the end declines
    /// by its own slug rather than a shared one.
    #[test]
    fn each_raster_enum_is_total_over_its_sdk_range() {
        for mtl in 0..=7u32 {
            assert!(compare_function(mtl).is_ok(), "compare {mtl}");
            assert!(stencil_operation(mtl).is_ok(), "stencil {mtl}");
        }
        assert_eq!(
            compare_function(8).unwrap_err(),
            TranslateReason::UnknownCompareFunction(8)
        );
        assert_eq!(
            stencil_operation(8).unwrap_err(),
            TranslateReason::UnknownStencilOperation(8)
        );
    }

    /// **The two ordinal witnesses agree, over the whole domain.**
    ///
    /// The engine's `mtl_ordinal` is its declaration order and the contract's
    /// `ordinal` is spelled from named `MTL_*` constants — two independent
    /// statements of the same fact, and until these tables were one they could
    /// drift with nothing failing. Sweeping the domain rather than the variant
    /// list is what makes "admits exactly what the contract admits" a property
    /// and not a list to keep in step.
    #[test]
    fn the_engine_spelling_carries_the_ordinal_the_contract_parsed() {
        use crate::protocol::depth_stencil::StencilOperation;
        use crate::protocol::sampler::CompareFunction;

        for c in CompareFunction::ALL {
            assert_eq!(
                compare_function(c.ordinal()).unwrap().mtl_ordinal(),
                c.ordinal(),
                "{c:?} round-trips through a different ordinal"
            );
        }
        for o in StencilOperation::ALL {
            assert_eq!(
                stencil_operation(o.ordinal()).unwrap().mtl_ordinal(),
                o.ordinal(),
                "{o:?} round-trips through a different ordinal"
            );
        }
        for mtl in (0..=1024u32).chain([u32::MAX - 1, u32::MAX]) {
            assert_eq!(
                compare_function(mtl).is_ok(),
                CompareFunction::parse(mtl).is_some(),
                "compare {mtl}"
            );
            assert_eq!(
                stencil_operation(mtl).is_ok(),
                StencilOperation::parse(mtl).is_some(),
                "stencil {mtl}"
            );
            assert_eq!(
                index_type(mtl).is_some(),
                u16::try_from(mtl)
                    .ok()
                    .and_then(crate::protocol::render::IndexType::parse)
                    .is_some(),
                "index {mtl}"
            );
        }
    }

    /// The exact wire order of `MTLCompareFunction`, value by value.
    ///
    /// Injectivity below proves no two values collide; it does not prove the
    /// table is not *rotated*. A rotation still round-trips and still renders —
    /// it just inverts occlusion for every 3D draw — so the mapping is pinned
    /// arm by arm. (Moved here from `runtime/draw/mod.rs`, which held a second
    /// copy of this table; the assertion outlived the duplicate.)
    #[test]
    fn compare_function_matches_the_metal_abi_order() {
        use SamplerCompareFunction as C;
        assert_eq!(compare_function(0), Ok(C::Never));
        assert_eq!(compare_function(1), Ok(C::Less));
        assert_eq!(compare_function(2), Ok(C::Equal));
        assert_eq!(compare_function(3), Ok(C::LessEqual));
        assert_eq!(compare_function(4), Ok(C::Greater));
        assert_eq!(compare_function(5), Ok(C::NotEqual));
        assert_eq!(compare_function(6), Ok(C::GreaterEqual));
        assert_eq!(compare_function(7), Ok(C::Always));
        assert_eq!(
            compare_function(99).unwrap_err(),
            TranslateReason::UnknownCompareFunction(99)
        );
    }

    /// Same, for `MTLStencilOperation`. The increment/decrement pairs are the
    /// transcription hazard: clamp and wrap differ only in overflow behaviour.
    #[test]
    fn stencil_operation_matches_the_metal_abi_order() {
        use StencilOp as O;
        assert_eq!(stencil_operation(0), Ok(O::Keep));
        assert_eq!(stencil_operation(1), Ok(O::Zero));
        assert_eq!(stencil_operation(2), Ok(O::Replace));
        assert_eq!(stencil_operation(3), Ok(O::IncrementClamp));
        assert_eq!(stencil_operation(4), Ok(O::DecrementClamp));
        assert_eq!(stencil_operation(5), Ok(O::Invert));
        assert_eq!(stencil_operation(6), Ok(O::IncrementWrap));
        assert_eq!(stencil_operation(7), Ok(O::DecrementWrap));
        assert_eq!(
            stencil_operation(99).unwrap_err(),
            TranslateReason::UnknownStencilOperation(99)
        );
    }

    /// One Apple enum, spelled once here and once in
    /// `backend::metal::mtl_enum::visibility_result_mode`, with nothing in the
    /// toolchain comparing them. A mode this arm records and that one refuses
    /// is a guest that culls correctly on one host and reads a stale word on
    /// the other, so both are held to [`crate::protocol::visibility`] — this
    /// one as a test, the Metal one as a `const` block, because its tests run
    /// on no machine anybody edits from.
    #[test]
    fn the_recorded_visibility_modes_are_the_ones_the_contract_names() {
        use crate::protocol::visibility::{
            visibility_result_mode_recordable, VISIBILITY_RESULT_MODE_DISABLED,
            VISIBILITY_RESULT_MODE_SWEEP_END,
        };
        for mtl in 0..VISIBILITY_RESULT_MODE_SWEEP_END {
            let recorded = matches!(visibility_result_mode(mtl), Ok(Some(_)));
            assert_eq!(
                recorded,
                visibility_result_mode_recordable(mtl),
                "ordinal {mtl}: this arm and the device contract disagree about \
                 whether an occlusion query armed with it is recorded"
            );
        }
        // The disarming ordinal is `Ok(None)` rather than a refusal: it is the
        // absence of a query, which is a thing a stream legitimately says.
        assert_eq!(
            visibility_result_mode(VISIBILITY_RESULT_MODE_DISABLED),
            Ok(None)
        );
    }

    #[test]
    fn index_types_map_by_width() {
        assert_eq!(index_type(0), Some(IndexType::U16));
        assert_eq!(index_type(1), Some(IndexType::U32));
        assert_eq!(index_type(2), None);
        assert_eq!(vk_index_type(IndexType::U16), vk::IndexType::UINT16);
        assert_eq!(vk_index_type(IndexType::U32), vk::IndexType::UINT32);
    }
}
