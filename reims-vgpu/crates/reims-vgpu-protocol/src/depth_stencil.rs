//! `MTLDepthStencilDescriptor`'s ordinals, and which of its two faces are
//! there at all.
//!
//! # A face that is off has no bytes worth reading
//!
//! The record carries two 24-byte face blocks and two enable bits. A block
//! behind a clear bit is *not* a face set to defaults — it is storage the
//! serializer never wrote, whose contents are whatever the guest's ring last
//! held. So [`DepthStencilState`] holds `Option<StencilFace>` per face rather
//! than a face plus a flag: a consumer cannot read a face that is not there,
//! and a fixture cannot accidentally assert on noise.
//!
//! That also decides where the ordinal check goes. [`DepthStencilShape::checked`]
//! parses a face's four enumerations only behind that face's bit, because
//! refusing a descriptor for an unrecognised ordinal in a block nobody reads
//! would drop a declaration the guest API admits.
//!
//! # Metal has no depth-test enable
//!
//! `MTLDepthStencilDescriptor` has a compare function and a write flag and
//! nothing else; depth is always tested, and "no depth test" is spelled
//! `Always` with writes off. That is a fact about the guest API and it is why
//! there is no `depth_test_enabled` field here to be tempted into
//! mistranslating — what an executor does with it belongs to the executor.
//!
//! # What is deliberately not here
//!
//! Whether the pass being drawn into has a depth attachment at all. That is a
//! property of the render target, not of the state the guest declared, and a
//! state is legal to declare with no pass bound.

use crate::sampler::CompareFunction;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// `MTLStencilOperation`.
pub const MTL_STENCIL_OPERATION_KEEP: u32 = 0;
pub const MTL_STENCIL_OPERATION_ZERO: u32 = 1;
pub const MTL_STENCIL_OPERATION_REPLACE: u32 = 2;
pub const MTL_STENCIL_OPERATION_INCREMENT_CLAMP: u32 = 3;
pub const MTL_STENCIL_OPERATION_DECREMENT_CLAMP: u32 = 4;
pub const MTL_STENCIL_OPERATION_INVERT: u32 = 5;
pub const MTL_STENCIL_OPERATION_INCREMENT_WRAP: u32 = 6;
pub const MTL_STENCIL_OPERATION_DECREMENT_WRAP: u32 = 7;

/// What a stencil test does to the buffer at one of its three outcomes.
///
/// The clamping and wrapping pairs are distinct operations and not a mode on
/// one: folding `IncrementWrap` onto `IncrementClamp` differs only at the
/// value 255, which is exactly the case a wrapping counter exists for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

impl StencilOperation {
    pub const ALL: [StencilOperation; 8] = [
        Self::Keep,
        Self::Zero,
        Self::Replace,
        Self::IncrementClamp,
        Self::DecrementClamp,
        Self::Invert,
        Self::IncrementWrap,
        Self::DecrementWrap,
    ];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_STENCIL_OPERATION_KEEP => Self::Keep,
            MTL_STENCIL_OPERATION_ZERO => Self::Zero,
            MTL_STENCIL_OPERATION_REPLACE => Self::Replace,
            MTL_STENCIL_OPERATION_INCREMENT_CLAMP => Self::IncrementClamp,
            MTL_STENCIL_OPERATION_DECREMENT_CLAMP => Self::DecrementClamp,
            MTL_STENCIL_OPERATION_INVERT => Self::Invert,
            MTL_STENCIL_OPERATION_INCREMENT_WRAP => Self::IncrementWrap,
            MTL_STENCIL_OPERATION_DECREMENT_WRAP => Self::DecrementWrap,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::Keep => MTL_STENCIL_OPERATION_KEEP,
            Self::Zero => MTL_STENCIL_OPERATION_ZERO,
            Self::Replace => MTL_STENCIL_OPERATION_REPLACE,
            Self::IncrementClamp => MTL_STENCIL_OPERATION_INCREMENT_CLAMP,
            Self::DecrementClamp => MTL_STENCIL_OPERATION_DECREMENT_CLAMP,
            Self::Invert => MTL_STENCIL_OPERATION_INVERT,
            Self::IncrementWrap => MTL_STENCIL_OPERATION_INCREMENT_WRAP,
            Self::DecrementWrap => MTL_STENCIL_OPERATION_DECREMENT_WRAP,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Zero => "zero",
            Self::Replace => "replace",
            Self::IncrementClamp => "increment_clamp",
            Self::DecrementClamp => "decrement_clamp",
            Self::Invert => "invert",
            Self::IncrementWrap => "increment_wrap",
            Self::DecrementWrap => "decrement_wrap",
        }
    }
}

/// One face's declaration as the fields arrived.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StencilFaceShape {
    /// `MTLStencilDescriptor.stencilCompareFunction`.
    pub compare_function: u32,
    pub stencil_failure_operation: u32,
    pub depth_failure_operation: u32,
    pub depth_stencil_pass_operation: u32,
    pub read_mask: u32,
    pub write_mask: u32,
}

/// One face, parsed.
///
/// The masks are carried unparsed because they are masks: every one of the
/// thirty-two bits is a legal value, and there is no ordinal set to be outside
/// of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFace {
    pub compare: CompareFunction,
    pub stencil_failure: StencilOperation,
    pub depth_failure: StencilOperation,
    pub depth_stencil_pass: StencilOperation,
    pub read_mask: u32,
    pub write_mask: u32,
}

impl StencilFace {
    /// The face `MTLStencilDescriptor` describes when nothing is set: test
    /// always passes, nothing is written, both masks whole.
    ///
    /// This is the value a *bound* face takes when the guest left it alone. It
    /// is deliberately not what [`DepthStencilState`] substitutes for a face
    /// whose enable bit is clear — an absent face is `None`, and what an
    /// executor puts in its place is that executor's translation decision.
    pub const DEFAULT: Self = Self {
        compare: CompareFunction::Always,
        stencil_failure: StencilOperation::Keep,
        depth_failure: StencilOperation::Keep,
        depth_stencil_pass: StencilOperation::Keep,
        read_mask: u32::MAX,
        write_mask: u32::MAX,
    };
}

impl Default for StencilFace {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A depth-stencil declaration as the fields arrived.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilShape {
    /// `MTLCompareFunction` ordinal for the depth test.
    pub depth_compare_function: u32,
    pub depth_write_enabled: bool,
    pub front_stencil_enabled: bool,
    pub back_stencil_enabled: bool,
    /// Meaningful only behind `front_stencil_enabled`. See the module doc.
    pub front: StencilFaceShape,
    /// Meaningful only behind `back_stencil_enabled`.
    pub back: StencilFaceShape,
}

/// Why a depth-stencil declaration is not one the guest API admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthStencilRefusal {
    UnknownOrdinal { field: &'static str, ordinal: u32 },
}

impl reims_vgpu_observe::Decline for DepthStencilRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownOrdinal { .. } => "depth_stencil_unknown_ordinal",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnknownOrdinal { field, ordinal } => vec![
                ("field", (*field).to_string()),
                ("ordinal", ordinal.to_string()),
            ],
        }
    }
}

/// A depth-stencil declaration whose fields have been parsed.
///
/// A face is present or it is not; there is no flag to consult and no block to
/// read behind a clear one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthStencilState {
    depth_compare: CompareFunction,
    depth_write: bool,
    front: Option<StencilFace>,
    back: Option<StencilFace>,
}

fn parsed<T>(
    field: &'static str,
    ordinal: u32,
    value: Option<T>,
) -> Result<T, DepthStencilRefusal> {
    value.ok_or(DepthStencilRefusal::UnknownOrdinal { field, ordinal })
}

/// The four field names one face contributes, so a refusal says which face as
/// well as which enumeration.
struct FaceNames {
    compare: &'static str,
    stencil_failure: &'static str,
    depth_failure: &'static str,
    depth_stencil_pass: &'static str,
}

const FRONT: FaceNames = FaceNames {
    compare: "front_compare_function",
    stencil_failure: "front_stencil_failure_operation",
    depth_failure: "front_depth_failure_operation",
    depth_stencil_pass: "front_depth_stencil_pass_operation",
};

const BACK: FaceNames = FaceNames {
    compare: "back_compare_function",
    stencil_failure: "back_stencil_failure_operation",
    depth_failure: "back_depth_failure_operation",
    depth_stencil_pass: "back_depth_stencil_pass_operation",
};

impl StencilFaceShape {
    fn checked(self, names: &FaceNames) -> Result<StencilFace, DepthStencilRefusal> {
        Ok(StencilFace {
            compare: parsed(
                names.compare,
                self.compare_function,
                CompareFunction::parse(self.compare_function),
            )?,
            stencil_failure: parsed(
                names.stencil_failure,
                self.stencil_failure_operation,
                StencilOperation::parse(self.stencil_failure_operation),
            )?,
            depth_failure: parsed(
                names.depth_failure,
                self.depth_failure_operation,
                StencilOperation::parse(self.depth_failure_operation),
            )?,
            depth_stencil_pass: parsed(
                names.depth_stencil_pass,
                self.depth_stencil_pass_operation,
                StencilOperation::parse(self.depth_stencil_pass_operation),
            )?,
            read_mask: self.read_mask,
            write_mask: self.write_mask,
        })
    }
}

impl DepthStencilShape {
    /// Parse the declaration.
    ///
    /// A face is parsed only behind its own enable bit. See the module doc for
    /// why refusing on a disabled face's bytes would be wrong.
    ///
    /// # Errors
    ///
    /// [`DepthStencilRefusal`] naming the field and the face that failed.
    pub fn checked(self) -> Result<DepthStencilState, DepthStencilRefusal> {
        Ok(DepthStencilState {
            depth_compare: parsed(
                "depth_compare_function",
                self.depth_compare_function,
                CompareFunction::parse(self.depth_compare_function),
            )?,
            depth_write: self.depth_write_enabled,
            front: self
                .front_stencil_enabled
                .then(|| self.front.checked(&FRONT))
                .transpose()?,
            back: self
                .back_stencil_enabled
                .then(|| self.back.checked(&BACK))
                .transpose()?,
        })
    }
}

impl DepthStencilState {
    #[must_use]
    pub const fn depth_compare(&self) -> CompareFunction {
        self.depth_compare
    }

    #[must_use]
    pub const fn depth_write(&self) -> bool {
        self.depth_write
    }

    #[must_use]
    pub const fn front(&self) -> Option<StencilFace> {
        self.front
    }

    #[must_use]
    pub const fn back(&self) -> Option<StencilFace> {
        self.back
    }

    /// Whether any stencil testing happens at all.
    ///
    /// Either face being present is enough: a guest that enables only the back
    /// face still gets a stencil test, and the front face's behaviour is the
    /// executor's to supply.
    #[must_use]
    pub const fn stencil_engaged(&self) -> bool {
        self.front.is_some() || self.back.is_some()
    }

    /// Whether this state can change any attachment.
    ///
    /// `Always` with writes off and no stencil face is the identity: every
    /// fragment passes and nothing is written. Worth naming because it is what
    /// a guest that never asked for depth at all declares, and an executor may
    /// legitimately treat it as "no depth-stencil state".
    #[must_use]
    pub const fn is_inert(&self) -> bool {
        matches!(self.depth_compare, CompareFunction::Always)
            && !self.depth_write
            && !self.stencil_engaged()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{MTL_COMPARE_FUNCTION_ALWAYS, MTL_COMPARE_FUNCTION_LESS_EQUAL};
    use alloc::collections::BTreeSet;
    use reims_vgpu_observe::Decline;

    fn face() -> StencilFaceShape {
        StencilFaceShape {
            compare_function: MTL_COMPARE_FUNCTION_LESS_EQUAL,
            stencil_failure_operation: MTL_STENCIL_OPERATION_KEEP,
            depth_failure_operation: MTL_STENCIL_OPERATION_ZERO,
            depth_stencil_pass_operation: MTL_STENCIL_OPERATION_REPLACE,
            read_mask: 0xff,
            write_mask: 0x0f,
        }
    }

    fn shape() -> DepthStencilShape {
        DepthStencilShape {
            depth_compare_function: MTL_COMPARE_FUNCTION_LESS_EQUAL,
            depth_write_enabled: true,
            front_stencil_enabled: false,
            back_stencil_enabled: false,
            front: StencilFaceShape::default(),
            back: StencilFaceShape::default(),
        }
    }

    #[test]
    fn every_stencil_operation_round_trips_and_none_collapses() {
        let ordinals: BTreeSet<u32> = StencilOperation::ALL.iter().map(|o| o.ordinal()).collect();
        assert_eq!(ordinals.len(), StencilOperation::ALL.len());
        for op in StencilOperation::ALL {
            assert_eq!(StencilOperation::parse(op.ordinal()), Some(op));
        }
        // The set is closed: the ordinal past the last one is not an
        // operation, and the wrapping pair is distinct from the clamping one.
        assert_eq!(StencilOperation::parse(8), None);
        assert_ne!(
            StencilOperation::IncrementWrap,
            StencilOperation::IncrementClamp
        );
        let names: BTreeSet<&str> = StencilOperation::ALL.iter().map(|o| o.name()).collect();
        assert_eq!(names.len(), StencilOperation::ALL.len());
    }

    #[test]
    fn a_disabled_face_is_absent_rather_than_defaulted() {
        let state = shape().checked().expect("a legal declaration");
        assert_eq!(state.front(), None);
        assert_eq!(state.back(), None);
        assert!(!state.stencil_engaged());
        assert_eq!(state.depth_compare(), CompareFunction::LessEqual);
        assert!(state.depth_write());
    }

    /// The bytes under a clear enable bit are the guest's stale ring, so an
    /// unrecognised ordinal there must not refuse the declaration. This is the
    /// property that makes a decoded record a function of what was written.
    #[test]
    fn noise_behind_a_clear_enable_bit_refuses_nothing() {
        let noise = StencilFaceShape {
            compare_function: 0xdead_beef,
            stencil_failure_operation: 0xdead_beef,
            depth_failure_operation: 0xdead_beef,
            depth_stencil_pass_operation: 0xdead_beef,
            read_mask: 0xdead_beef,
            write_mask: 0xdead_beef,
        };
        let state = DepthStencilShape {
            front: noise,
            back: noise,
            ..shape()
        }
        .checked()
        .expect("a face nobody reads cannot refuse a declaration");
        assert_eq!(state.front(), None);
        assert_eq!(state.back(), None);

        // And the same noise behind a *set* bit does refuse, so the test above
        // is about the bit and not about the parser being lax.
        let refused = DepthStencilShape {
            front: noise,
            back: noise,
            front_stencil_enabled: true,
            ..shape()
        }
        .checked()
        .expect_err("an enabled face is parsed");
        assert_eq!(
            refused,
            DepthStencilRefusal::UnknownOrdinal {
                field: "front_compare_function",
                ordinal: 0xdead_beef,
            }
        );
    }

    /// A refusal that said only "a stencil ordinal was unknown" would leave
    /// eight fields to guess between.
    #[test]
    fn a_refusal_names_the_face_and_the_field() {
        let mut seen: BTreeSet<&'static str> = BTreeSet::new();
        for (front, back) in [(true, false), (false, true)] {
            for field in 0..4 {
                let mut bad = face();
                match field {
                    0 => bad.compare_function = 99,
                    1 => bad.stencil_failure_operation = 99,
                    2 => bad.depth_failure_operation = 99,
                    _ => bad.depth_stencil_pass_operation = 99,
                }
                let refused = DepthStencilShape {
                    front_stencil_enabled: front,
                    back_stencil_enabled: back,
                    front: if front { bad } else { face() },
                    back: if back { bad } else { face() },
                    ..shape()
                }
                .checked()
                .expect_err("an unknown ordinal on a live face");
                let DepthStencilRefusal::UnknownOrdinal { field, ordinal } = refused;
                assert_eq!(ordinal, 99);
                assert!(seen.insert(field), "{field} named twice");
                assert!(field.starts_with(if front { "front_" } else { "back_" }));
            }
        }
        assert_eq!(seen.len(), 8);
        assert_eq!(
            DepthStencilRefusal::UnknownOrdinal {
                field: "front_compare_function",
                ordinal: 99,
            }
            .slug(),
            "depth_stencil_unknown_ordinal"
        );
    }

    #[test]
    fn one_enabled_face_engages_the_stencil_test_and_leaves_the_other_absent() {
        let state = DepthStencilShape {
            back_stencil_enabled: true,
            back: face(),
            ..shape()
        }
        .checked()
        .expect("a legal declaration");
        assert!(state.stencil_engaged());
        assert_eq!(state.front(), None);
        assert_eq!(
            state.back(),
            Some(StencilFace {
                compare: CompareFunction::LessEqual,
                stencil_failure: StencilOperation::Keep,
                depth_failure: StencilOperation::Zero,
                depth_stencil_pass: StencilOperation::Replace,
                read_mask: 0xff,
                write_mask: 0x0f,
            })
        );
    }

    /// Masks have no closed set, so every bit pattern survives.
    #[test]
    fn a_mask_is_carried_whole() {
        for mask in [0, 1, 0x8000_0000, u32::MAX] {
            let state = DepthStencilShape {
                front_stencil_enabled: true,
                front: StencilFaceShape {
                    read_mask: mask,
                    write_mask: !mask,
                    ..face()
                },
                ..shape()
            }
            .checked()
            .expect("a mask is not an ordinal");
            let front = state.front().expect("enabled");
            assert_eq!(front.read_mask, mask);
            assert_eq!(front.write_mask, !mask);
        }
    }

    #[test]
    fn the_identity_state_is_the_one_that_can_change_nothing() {
        let inert = DepthStencilShape {
            depth_compare_function: MTL_COMPARE_FUNCTION_ALWAYS,
            depth_write_enabled: false,
            ..shape()
        }
        .checked()
        .expect("a legal declaration");
        assert!(inert.is_inert());

        // Each of the three ways out of inertness, one at a time.
        for changed in [
            DepthStencilShape {
                depth_compare_function: MTL_COMPARE_FUNCTION_LESS_EQUAL,
                depth_write_enabled: false,
                ..shape()
            },
            DepthStencilShape {
                depth_compare_function: MTL_COMPARE_FUNCTION_ALWAYS,
                depth_write_enabled: true,
                ..shape()
            },
            DepthStencilShape {
                depth_compare_function: MTL_COMPARE_FUNCTION_ALWAYS,
                depth_write_enabled: false,
                front_stencil_enabled: true,
                front: face(),
                ..shape()
            },
        ] {
            assert!(!changed.checked().expect("legal").is_inert());
        }
    }

    /// The default a *bound* face takes is Metal's documented one, and it is
    /// not what an absent face becomes.
    #[test]
    fn the_documented_face_default_passes_and_writes_nothing() {
        assert_eq!(StencilFace::default(), StencilFace::DEFAULT);
        assert_eq!(StencilFace::DEFAULT.compare, CompareFunction::Always);
        for op in [
            StencilFace::DEFAULT.stencil_failure,
            StencilFace::DEFAULT.depth_failure,
            StencilFace::DEFAULT.depth_stencil_pass,
        ] {
            assert_eq!(op, StencilOperation::Keep);
        }
        assert_eq!(StencilFace::DEFAULT.read_mask, u32::MAX);
        assert_eq!(StencilFace::DEFAULT.write_mask, u32::MAX);
    }
}
