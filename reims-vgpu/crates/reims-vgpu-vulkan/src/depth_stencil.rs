//! A checked depth-stencil state as `VkPipelineDepthStencilStateCreateInfo`
//! would hold it.
//!
//! # Metal has no depth-test enable, and Vulkan's gates the write
//!
//! `MTLDepthStencilDescriptor` carries a compare function and a write flag.
//! Depth is always tested; "no depth test" is `Always` with writes off.
//! Vulkan's `depthTestEnable` is not the same switch — with it clear, Vulkan
//! performs no test *and no write*, whatever `depthWriteEnable` says.
//!
//! So this rail sets `depthTestEnable` unconditionally true and lets the
//! compare operation carry the guest's meaning. That is exact in both
//! directions: a guest asking for `Always` with writes off gets a pipeline
//! that passes every fragment and writes nothing, which is what a cleared
//! `depthTestEnable` would also have produced, and a guest asking for `Always`
//! *with* writes gets the write — which a cleared `depthTestEnable` would have
//! silently dropped.
//!
//! # One stencil enable for two faces
//!
//! Metal binds a stencil descriptor per face and either may be off. Vulkan has
//! a single `stencilTestEnable` covering both. The test is therefore engaged
//! when *either* face is, and the face that is not gets Metal's documented
//! `MTLStencilDescriptor` default — compare `Always`, every operation `Keep`,
//! full masks — which is a face that reads the buffer and changes nothing.
//!
//! That substitution is this layer's, not the protocol layer's: there, an
//! absent face is `None` and stays `None`, because the bytes behind a clear
//! enable bit are the guest's stale ring rather than a declaration.
//!
//! # The reference value is not here
//!
//! Metal sets the stencil reference on the encoder
//! (`setStencilReferenceValue:`), so it changes without the state changing. It
//! is `VK_DYNAMIC_STATE_STENCIL_REFERENCE` on this rail and is supplied per
//! draw; a plan that baked one in would key a pipeline cache on a value that
//! is not part of the state.
//!
//! # All of it is encoder state, and Vulkan bakes it
//!
//! `setDepthStencilState:` binds an `MTLDepthStencilState` on the render
//! command encoder, so a guest changes the whole of this between two draws of
//! one pipeline — the depth-write flag around an opaque and a transparent pass
//! is the textbook case. `VkPipelineDepthStencilStateCreateInfo` is pipeline
//! creation state, so on a bare 1.2 baseline each of those draws needs its own
//! pipeline and the whole state is part of the cache key.
//!
//! `VK_EXT_extended_dynamic_state` moves every member the guest can reach onto
//! the command buffer: `vkCmdSetDepthWriteEnable`, `vkCmdSetDepthCompareOp`,
//! `vkCmdSetStencilTestEnable` and `vkCmdSetStencilOp`. The two stencil masks
//! were already dynamic in Vulkan 1.0 (`vkCmdSetStencilCompareMask`,
//! `vkCmdSetStencilWriteMask`) and so is the reference, so with that one
//! extension *nothing* the guest declares is left in the pipeline and every
//! depth-stencil state this device can build collapses onto one.
//!
//! [`DepthStencilCell`] is that capability and [`Plan`] is the split: the
//! members this host supplies per draw carry a fixed placeholder in
//! [`Plan::state`] and the guest's values in [`Plan::dynamic`], so a member is
//! in exactly one half.
//!
//! # The pass decides whether any of it exists
//!
//! Vulkan attaches no depth-stencil state to a pipeline whose subpass has no
//! depth attachment, and a dynamic state declared for a structure that is not
//! there is one nothing will ever read. So [`plan`] takes whether the pass
//! carries a depth attachment alongside the host's capability, and answers
//! [`Plan::dynamic`] `None` when it does not — the state is inert either way,
//! and this is the layer that should say so rather than each caller.
//!
//! # Planned, not created
//!
//! Nothing here creates a pipeline. The plan is a value, so every mapping is
//! tested with no GPU.

use ash::vk;
use reims_vgpu_core::depth_stencil::{DepthStencilState, StencilFace, StencilOperation as GuestOp};

/// `MTLCompareFunction` → `VkCompareOp`, spelled once.
///
/// The guest type is the sampler's — one comparison enumeration serves both a
/// sampler's compare-sample and this state's depth and stencil tests — so the
/// mapping is the sampler module's too, and this is a name for it rather than
/// a second copy. Two copies of a total mapping are two things that can come
/// to disagree, and nothing would fail if they did.
pub use crate::sampler::compare_op;

/// `MTLStencilOperation` → `VkStencilOp`.
///
/// The two enumerations name the same eight operations. They are written out
/// rather than cast, because a numeric coincidence between two vendors' enums
/// is not a contract and a reordering on either side would be silent.
#[must_use]
pub const fn stencil_op(guest: GuestOp) -> vk::StencilOp {
    match guest {
        GuestOp::Keep => vk::StencilOp::KEEP,
        GuestOp::Zero => vk::StencilOp::ZERO,
        GuestOp::Replace => vk::StencilOp::REPLACE,
        GuestOp::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        GuestOp::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        GuestOp::Invert => vk::StencilOp::INVERT,
        GuestOp::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        GuestOp::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

/// One face, as `VkStencilOpState` would hold it.
///
/// Spelled out rather than held as a `vk::StencilOpState`, which is not `Eq` —
/// and a translation whose result cannot be compared is one whose mappings
/// cannot be asserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FacePlan {
    pub fail_op: vk::StencilOp,
    pub pass_op: vk::StencilOp,
    pub depth_fail_op: vk::StencilOp,
    pub compare_op: vk::CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
}

impl FacePlan {
    #[must_use]
    pub fn of(face: StencilFace) -> Self {
        Self {
            fail_op: stencil_op(face.stencil_failure),
            pass_op: stencil_op(face.depth_stencil_pass),
            depth_fail_op: stencil_op(face.depth_failure),
            compare_op: compare_op(face.compare),
            compare_mask: face.read_mask,
            write_mask: face.write_mask,
        }
    }

    /// The face substituted for one whose enable bit was clear. See the module
    /// doc.
    #[must_use]
    pub fn pass_through() -> Self {
        Self::PASS_THROUGH
    }

    /// The same face as a constant, so a placeholder pipeline state can be one.
    /// Derived from [`StencilFace::DEFAULT`] by the same rule
    /// [`Self::of`] applies, and the test below is that the two agree.
    pub const PASS_THROUGH: Self = Self {
        fail_op: stencil_op(StencilFace::DEFAULT.stencil_failure),
        pass_op: stencil_op(StencilFace::DEFAULT.depth_stencil_pass),
        depth_fail_op: stencil_op(StencilFace::DEFAULT.depth_failure),
        compare_op: compare_op(StencilFace::DEFAULT.compare),
        compare_mask: StencilFace::DEFAULT.read_mask,
        write_mask: StencilFace::DEFAULT.write_mask,
    };

    /// The reference is deliberately zero here and set dynamically. See the
    /// module doc.
    pub const fn native(self) -> vk::StencilOpState {
        vk::StencilOpState {
            fail_op: self.fail_op,
            pass_op: self.pass_op,
            depth_fail_op: self.depth_fail_op,
            compare_op: self.compare_op,
            compare_mask: self.compare_mask,
            write_mask: self.write_mask,
            reference: 0,
        }
    }
}

/// A depth-stencil state as this rail would build it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthStencilPlan {
    /// Always true. See the module doc.
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    /// Always false: Metal has no depth-bounds test, so nothing the guest can
    /// declare turns this on.
    pub depth_bounds_test_enable: bool,
    pub stencil_test_enable: bool,
    pub front: FacePlan,
    pub back: FacePlan,
    /// Whether every member above is a placeholder because the encoder
    /// supplies the guest's state instead.
    ///
    /// In the pipeline state rather than beside it, for the reason
    /// `raster::RasterizationState::dynamic` is: this value *is* a pipeline
    /// cache key, and the dynamic states a pipeline declares have to be
    /// readable from the key alone or the cache would need a second derivation
    /// that could disagree with the first.
    pub dynamic: bool,
}

impl DepthStencilPlan {
    /// The dynamic states a pipeline built with this state must declare.
    ///
    /// Read off the state that holds the placeholder, never re-derived from a
    /// capability: a pipeline that declares a placeholder without declaring the
    /// states to replace it *runs the placeholder*, and on a host with no
    /// validation layers nothing says so.
    ///
    /// The two stencil masks and the reference are in the dynamic list too.
    /// They cost no capability — all three were dynamic in Vulkan 1.0 — but
    /// they are only listed where the rest of the state is: a baked pipeline
    /// still carries the guest's masks and would gain nothing from taking them
    /// per draw, while a dynamic one has no masks left in it to use.
    ///
    /// The reference alone is what a baking pipeline declares, and only where
    /// its stencil test is on: `setStencilReferenceValue:` is a Metal encoder
    /// command separate from `setDepthStencilState:`, so it changes without
    /// this state changing, but a pipeline whose test is off has nothing to
    /// apply it to. On the dynamic rung the test enable is itself supplied per
    /// draw, so the reference is always declared and always supplied.
    #[must_use]
    pub const fn states(self) -> &'static [vk::DynamicState] {
        if self.dynamic {
            &[
                vk::DynamicState::DEPTH_TEST_ENABLE,
                vk::DynamicState::DEPTH_WRITE_ENABLE,
                vk::DynamicState::DEPTH_COMPARE_OP,
                vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
                vk::DynamicState::STENCIL_TEST_ENABLE,
                vk::DynamicState::STENCIL_OP,
                vk::DynamicState::STENCIL_COMPARE_MASK,
                vk::DynamicState::STENCIL_WRITE_MASK,
                vk::DynamicState::STENCIL_REFERENCE,
            ]
        } else if self.stencil_test_enable {
            &[vk::DynamicState::STENCIL_REFERENCE]
        } else {
            &[]
        }
    }

    pub const fn native(self) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
        vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: as_bool32(self.depth_test_enable),
            depth_write_enable: as_bool32(self.depth_write_enable),
            depth_compare_op: self.depth_compare_op,
            depth_bounds_test_enable: as_bool32(self.depth_bounds_test_enable),
            stencil_test_enable: as_bool32(self.stencil_test_enable),
            front: self.front.native(),
            back: self.back.native(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
            _marker: core::marker::PhantomData,
        }
    }
}

const fn as_bool32(value: bool) -> vk::Bool32 {
    if value {
        vk::TRUE
    } else {
        vk::FALSE
    }
}

/// What this host offers for changing depth-stencil state without a rebuild.
///
/// One bit, because it is one feature bit: `extendedDynamicState` reaches
/// `vkCmdSetDepthWriteEnable`, `vkCmdSetDepthCompareOp`,
/// `vkCmdSetStencilTestEnable` and `vkCmdSetStencilOp` together, and the two
/// stencil masks and the reference were dynamic in Vulkan 1.0 and need no
/// capability at all. So this host either supplies the whole state per draw or
/// bakes the whole state, and there is no partial rung to name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilCell {
    /// `VkPhysicalDeviceExtendedDynamicStateFeaturesEXT::extendedDynamicState`,
    /// or 1.3 core.
    pub extended_dynamic_state: bool,
}

/// The guest's values for a state this host supplies per draw.
///
/// The whole state rather than a field per member: a caller that has this has
/// reproduced the guest's `MTLDepthStencilState` and a caller that has `None`
/// is one whose pipeline baked it, which is a distinction a struct of options
/// could only restate five times.
///
/// The masks are here and the reference is not, for the reason the module doc
/// gives: `setStencilReferenceValue:` is a *separate* encoder command from
/// `setDepthStencilState:`, so it changes without this state changing and is
/// supplied by whoever tracks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DynamicDepthStencil {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub depth_bounds_test_enable: bool,
    pub stencil_test_enable: bool,
    pub front: FacePlan,
    pub back: FacePlan,
}

/// A translated depth-stencil state: what the pipeline is built with, and what
/// the encoder sets before each draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub state: DepthStencilPlan,
    /// `Some` exactly when [`Plan::state`] is the placeholder rather than the
    /// guest's state — so a caller that records every `Some` has reproduced the
    /// guest and a caller that records none of them is one whose pipeline
    /// carries it.
    pub dynamic: Option<DynamicDepthStencil>,
}

/// What a pipeline bakes when this host supplies the state per draw.
///
/// Every member at its Vulkan default, which is also the state that draws
/// nothing surprising if it were ever reached: no test, no write, no stencil.
/// One constant rather than a value derived from the guest, because the point
/// of the dynamic rung is that the pipeline stops depending on the guest's
/// state at all — a placeholder that varied would still be a cache dimension.
const PLACEHOLDER: DepthStencilPlan = DepthStencilPlan {
    depth_test_enable: false,
    depth_write_enable: false,
    depth_compare_op: vk::CompareOp::NEVER,
    depth_bounds_test_enable: false,
    stencil_test_enable: false,
    front: FacePlan::PASS_THROUGH,
    back: FacePlan::PASS_THROUGH,
    dynamic: true,
};

/// Translate a checked depth-stencil state against this host and this pass.
///
/// `attached` is whether the pass this pipeline belongs to carries a depth
/// attachment. Where it does not, Vulkan attaches no depth-stencil state at
/// all and nothing here is dynamic — see the module doc.
///
/// Total: every state the guest API admits has a plan here, so there is no
/// refusal to return. The ordinals that could have failed were closed one
/// layer down.
#[must_use]
pub fn plan(state: &DepthStencilState, cell: DepthStencilCell, attached: bool) -> Plan {
    let stencil_test_enable = state.stencil_engaged();
    // A face is planned only where the test runs at all. With the test off,
    // both faces are the pass-through — which is what Vulkan ignores anyway,
    // and keeping the guest's ops there would make two states that behave
    // identically compare unequal in a pipeline cache key.
    let face = |present: Option<StencilFace>| {
        if stencil_test_enable {
            present.map_or_else(FacePlan::pass_through, FacePlan::of)
        } else {
            FacePlan::pass_through()
        }
    };
    let guest = DepthStencilPlan {
        depth_test_enable: true,
        depth_write_enable: state.depth_write(),
        depth_compare_op: compare_op(state.depth_compare()),
        depth_bounds_test_enable: false,
        stencil_test_enable,
        front: face(state.front()),
        back: face(state.back()),
        dynamic: false,
    };
    if !(cell.extended_dynamic_state && attached) {
        return Plan {
            state: guest,
            dynamic: None,
        };
    }
    Plan {
        state: PLACEHOLDER,
        dynamic: Some(DynamicDepthStencil {
            depth_test_enable: guest.depth_test_enable,
            depth_write_enable: guest.depth_write_enable,
            depth_compare_op: guest.depth_compare_op,
            depth_bounds_test_enable: guest.depth_bounds_test_enable,
            stencil_test_enable: guest.stencil_test_enable,
            front: guest.front,
            back: guest.back,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::depth_stencil::{
        DepthStencilShape, StencilFaceShape, MTL_STENCIL_OPERATION_INCREMENT_WRAP,
        MTL_STENCIL_OPERATION_INVERT, MTL_STENCIL_OPERATION_KEEP, MTL_STENCIL_OPERATION_REPLACE,
        MTL_STENCIL_OPERATION_ZERO,
    };
    use reims_vgpu_core::sampler::{
        CompareFunction, MTL_COMPARE_FUNCTION_ALWAYS, MTL_COMPARE_FUNCTION_EQUAL,
        MTL_COMPARE_FUNCTION_GREATER, MTL_COMPARE_FUNCTION_LESS, MTL_COMPARE_FUNCTION_LESS_EQUAL,
        MTL_COMPARE_FUNCTION_NEVER,
    };
    use std::collections::BTreeSet;

    fn face_shape() -> StencilFaceShape {
        StencilFaceShape {
            compare_function: MTL_COMPARE_FUNCTION_GREATER,
            stencil_failure_operation: MTL_STENCIL_OPERATION_ZERO,
            depth_failure_operation: MTL_STENCIL_OPERATION_INCREMENT_WRAP,
            depth_stencil_pass_operation: MTL_STENCIL_OPERATION_REPLACE,
            read_mask: 0xf0,
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

    /// A host that bakes the whole state, which is the rung every one of the
    /// mapping tests below is about: the mapping is the same either way, and
    /// this is where it lands in the pipeline.
    const BAKED: DepthStencilCell = DepthStencilCell {
        extended_dynamic_state: false,
    };
    /// A host that supplies it per draw, with a pass that has somewhere to put
    /// it.
    const DYNAMIC: DepthStencilCell = DepthStencilCell {
        extended_dynamic_state: true,
    };

    fn planned(shape: DepthStencilShape) -> DepthStencilPlan {
        plan(
            &shape.checked().expect("a declaration the guest API admits"),
            BAKED,
            true,
        )
        .state
    }

    fn planned_dynamic(shape: DepthStencilShape) -> Plan {
        plan(
            &shape.checked().expect("a declaration the guest API admits"),
            DYNAMIC,
            true,
        )
    }

    #[test]
    fn every_stencil_operation_maps_to_a_distinct_one() {
        let mapped: BTreeSet<i32> = GuestOp::ALL
            .iter()
            .map(|o| stencil_op(*o).as_raw())
            .collect();
        assert_eq!(mapped.len(), GuestOp::ALL.len());
        // The pair that differs only at the ends of the range, which is the
        // pair a cast would have got right by accident and a reordering wrong.
        assert_eq!(
            stencil_op(GuestOp::IncrementWrap),
            vk::StencilOp::INCREMENT_AND_WRAP
        );
        assert_eq!(
            stencil_op(GuestOp::IncrementClamp),
            vk::StencilOp::INCREMENT_AND_CLAMP
        );
        assert_eq!(stencil_op(GuestOp::Keep), vk::StencilOp::KEEP);
    }

    /// The claim the module doc makes about `depthTestEnable`: it is on for
    /// every state, so `Always` with writes on keeps the write that a cleared
    /// enable would have dropped.
    #[test]
    fn the_depth_test_is_always_enabled_so_a_write_is_never_dropped() {
        for compare in [MTL_COMPARE_FUNCTION_ALWAYS, MTL_COMPARE_FUNCTION_LESS_EQUAL] {
            for write in [false, true] {
                let plan = planned(DepthStencilShape {
                    depth_compare_function: compare,
                    depth_write_enabled: write,
                    ..shape()
                });
                assert!(plan.depth_test_enable);
                assert_eq!(plan.depth_write_enable, write);
                assert!(!plan.depth_bounds_test_enable);
            }
        }
        // The case that motivates it, named on its own.
        let always_writing = planned(DepthStencilShape {
            depth_compare_function: MTL_COMPARE_FUNCTION_ALWAYS,
            depth_write_enabled: true,
            ..shape()
        });
        assert!(always_writing.depth_test_enable);
        assert!(always_writing.depth_write_enable);
        assert_eq!(always_writing.depth_compare_op, vk::CompareOp::ALWAYS);
    }

    #[test]
    fn a_state_with_no_face_leaves_the_stencil_test_off() {
        let plan = planned(shape());
        assert!(!plan.stencil_test_enable);
        assert_eq!(plan.front, FacePlan::pass_through());
        assert_eq!(plan.back, FacePlan::pass_through());
    }

    /// One Vulkan enable for two Metal faces: the test runs when either face
    /// is bound, and the unbound one changes nothing.
    #[test]
    fn one_bound_face_engages_the_test_and_the_other_passes_through() {
        for (front_on, back_on) in [(true, false), (false, true), (true, true)] {
            let plan = planned(DepthStencilShape {
                front_stencil_enabled: front_on,
                back_stencil_enabled: back_on,
                front: face_shape(),
                back: face_shape(),
                ..shape()
            });
            assert!(plan.stencil_test_enable);

            let bound = FacePlan {
                fail_op: vk::StencilOp::ZERO,
                pass_op: vk::StencilOp::REPLACE,
                depth_fail_op: vk::StencilOp::INCREMENT_AND_WRAP,
                compare_op: vk::CompareOp::GREATER,
                compare_mask: 0xf0,
                write_mask: 0x0f,
            };
            assert_eq!(
                plan.front,
                if front_on {
                    bound
                } else {
                    FacePlan::pass_through()
                }
            );
            assert_eq!(
                plan.back,
                if back_on {
                    bound
                } else {
                    FacePlan::pass_through()
                }
            );
        }
    }

    /// The pass-through is a face that reads and changes nothing, which is
    /// what makes it a safe substitute for one the guest never declared.
    #[test]
    fn the_pass_through_face_writes_nothing_whatever_the_test_does() {
        let face = FacePlan::pass_through();
        assert_eq!(face.compare_op, vk::CompareOp::ALWAYS);
        for op in [face.fail_op, face.pass_op, face.depth_fail_op] {
            assert_eq!(op, vk::StencilOp::KEEP);
        }
    }

    /// Two states that behave identically must plan identically, or a pipeline
    /// cache keyed on the plan compiles the same pipeline twice.
    #[test]
    fn a_disengaged_test_plans_the_same_whatever_the_faces_held() {
        let bare = planned(shape());
        let with_bytes = planned(DepthStencilShape {
            front: face_shape(),
            back: face_shape(),
            ..shape()
        });
        assert_eq!(bare, with_bytes);
        assert!(!bare.stencil_test_enable);
    }

    #[test]
    fn every_comparison_reaches_the_depth_test_and_the_faces_alike() {
        let mut seen: BTreeSet<i32> = BTreeSet::new();
        for guest in CompareFunction::ALL {
            let plan = planned(DepthStencilShape {
                depth_compare_function: guest.ordinal(),
                front_stencil_enabled: true,
                front: StencilFaceShape {
                    compare_function: guest.ordinal(),
                    ..face_shape()
                },
                ..shape()
            });
            assert_eq!(plan.depth_compare_op, compare_op(guest));
            assert_eq!(plan.front.compare_op, compare_op(guest));
            assert!(seen.insert(plan.depth_compare_op.as_raw()));
        }
        assert_eq!(seen.len(), CompareFunction::ALL.len());
    }

    /// The reference is dynamic state, so nothing a plan holds carries one.
    #[test]
    fn the_native_face_leaves_the_reference_for_the_draw() {
        let plan = planned(DepthStencilShape {
            front_stencil_enabled: true,
            front: face_shape(),
            ..shape()
        });
        let native = plan.front.native();
        assert_eq!(native.reference, 0);
        assert_eq!(native.compare_mask, 0xf0);
        assert_eq!(native.write_mask, 0x0f);
    }

    #[test]
    fn the_native_state_carries_the_plan() {
        let plan = planned(DepthStencilShape {
            front_stencil_enabled: true,
            front: face_shape(),
            ..shape()
        });
        let native = plan.native();
        assert_eq!(native.depth_test_enable, vk::TRUE);
        assert_eq!(native.depth_write_enable, vk::TRUE);
        assert_eq!(native.depth_compare_op, vk::CompareOp::LESS_OR_EQUAL);
        assert_eq!(native.depth_bounds_test_enable, vk::FALSE);
        assert_eq!(native.stencil_test_enable, vk::TRUE);
        assert_eq!(native.front.compare_op, vk::CompareOp::GREATER);
        assert_eq!(native.back.compare_op, vk::CompareOp::ALWAYS);
        assert_eq!(
            native.s_type,
            vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO
        );
    }

    /// The claim the whole rung rests on: on a host that supplies this state
    /// per draw, every depth-stencil state the guest can declare builds the
    /// *same* pipeline — so the whole space compiles once instead of once per
    /// bound `MTLDepthStencilState`.
    #[test]
    fn every_guest_state_collapses_to_one_pipeline_where_the_host_is_dynamic() {
        let mut dynamic_states = BTreeSet::new();
        let mut baked_states = BTreeSet::new();
        let mut count = 0;
        for depth_write_enabled in [false, true] {
            for depth_compare in [
                MTL_COMPARE_FUNCTION_NEVER,
                MTL_COMPARE_FUNCTION_LESS,
                MTL_COMPARE_FUNCTION_ALWAYS,
            ] {
                for stencil in [false, true] {
                    for op in [
                        MTL_STENCIL_OPERATION_KEEP,
                        MTL_STENCIL_OPERATION_REPLACE,
                        MTL_STENCIL_OPERATION_INVERT,
                    ] {
                        let face = StencilFaceShape {
                            compare_function: MTL_COMPARE_FUNCTION_EQUAL,
                            stencil_failure_operation: op,
                            depth_failure_operation: op,
                            depth_stencil_pass_operation: op,
                            read_mask: 0x0f,
                            write_mask: op + 1,
                        };
                        let shape = DepthStencilShape {
                            depth_compare_function: depth_compare,
                            depth_write_enabled,
                            front_stencil_enabled: stencil,
                            back_stencil_enabled: stencil,
                            front: face,
                            back: face,
                        };
                        dynamic_states.insert(format!("{:?}", planned_dynamic(shape).state));
                        baked_states.insert(format!("{:?}", planned(shape)));
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 36, "the whole product was walked");
        // Non-vacuous the other way: a baking host needs a pipeline per
        // *distinguishable* state, and these are distinguishable. The stencil
        // ops collapse where the test is off, which is the one deliberate
        // merge this plan makes, so the baked count is below the product and
        // still far above one.
        assert!(
            baked_states.len() > 6,
            "baking must not collapse these: {}",
            baked_states.len()
        );
        assert_eq!(
            dynamic_states.len(),
            1,
            "one pipeline serves every depth-stencil state"
        );
    }

    /// A member is in exactly one half. The pipeline carries the guest's state
    /// or the encoder does, and the placeholder is never the guest's value by
    /// accident.
    #[test]
    fn the_guests_state_is_in_exactly_one_half() {
        let shape = DepthStencilShape {
            depth_compare_function: MTL_COMPARE_FUNCTION_GREATER,
            depth_write_enabled: true,
            front_stencil_enabled: true,
            back_stencil_enabled: true,
            front: StencilFaceShape {
                compare_function: MTL_COMPARE_FUNCTION_EQUAL,
                stencil_failure_operation: MTL_STENCIL_OPERATION_ZERO,
                depth_failure_operation: MTL_STENCIL_OPERATION_INVERT,
                depth_stencil_pass_operation: MTL_STENCIL_OPERATION_REPLACE,
                read_mask: 0x0f,
                write_mask: 0xf0,
            },
            back: StencilFaceShape::default(),
        };
        let baked = planned(shape);
        let split = planned_dynamic(shape);

        // The dynamic half is exactly what the baking host would have baked.
        let recorded = split.dynamic.expect("a dynamic host supplies the state");
        assert_eq!(recorded.depth_write_enable, baked.depth_write_enable);
        assert_eq!(recorded.depth_compare_op, baked.depth_compare_op);
        assert_eq!(recorded.stencil_test_enable, baked.stencil_test_enable);
        assert_eq!(recorded.front, baked.front);
        assert_eq!(recorded.back, baked.back);

        // And the pipeline half holds none of it.
        assert_eq!(split.state, PLACEHOLDER);
        assert_ne!(split.state, baked);
        // The marker in the state and the presence of the encoder half are one
        // fact, which is what lets a cache key answer "which states does this
        // pipeline declare" without re-asking the host.
        assert_eq!(split.state.dynamic, split.dynamic.is_some());
        assert!(!baked.dynamic);
    }

    /// A pass with no depth attachment has nowhere to put a depth-stencil
    /// state, so nothing is dynamic there however capable the host is — and a
    /// dynamic state declared for a structure Vulkan will not read is one
    /// nothing would ever set.
    #[test]
    fn an_unattached_pass_makes_nothing_dynamic() {
        let shape = shape();
        let state = shape.checked().expect("the default is admissible");
        for cell in [BAKED, DYNAMIC] {
            let unattached = plan(&state, cell, false);
            assert!(unattached.dynamic.is_none());
            assert_eq!(unattached.state, plan(&state, BAKED, true).state);
            assert!(
                unattached.state.states().is_empty(),
                "the default shape has no stencil test, so not even the reference"
            );
        }
    }

    /// The state list and the placeholder are two readings of one decision.
    /// Every member the placeholder stands in for must appear in the list, or
    /// the pipeline rasterizes the placeholder.
    #[test]
    fn every_member_the_placeholder_stands_in_for_is_declared_dynamic() {
        let split = planned_dynamic(shape());
        let declared: BTreeSet<i32> = split.state.states().iter().map(|s| s.as_raw()).collect();
        for required in [
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
        ] {
            assert!(
                declared.contains(&required.as_raw()),
                "{required:?} is a placeholder nobody replaces"
            );
        }
        assert_eq!(
            declared.len(),
            split.state.states().len(),
            "no state listed twice"
        );

        // The baked rung declares the reference and nothing else — and only
        // where the stencil test is on, which is exactly what the pipeline
        // cache declared before any of this.
        let mut with_stencil = shape();
        with_stencil.front_stencil_enabled = true;
        with_stencil.front = face_shape();
        assert_eq!(
            plan(&with_stencil.checked().expect("admissible"), BAKED, true)
                .state
                .states(),
            &[vk::DynamicState::STENCIL_REFERENCE]
        );
        assert!(plan(&shape().checked().expect("admissible"), BAKED, true)
            .state
            .states()
            .is_empty());
    }

    /// `FacePlan::PASS_THROUGH` is written out as a constant so a placeholder
    /// pipeline state can be one. It must be the same face `of` produces from
    /// the same default, or the two spellings drift.
    #[test]
    fn the_constant_pass_through_face_is_the_one_the_mapping_produces() {
        assert_eq!(FacePlan::PASS_THROUGH, FacePlan::of(StencilFace::DEFAULT));
        assert_eq!(FacePlan::PASS_THROUGH, FacePlan::pass_through());
    }

    /// Every stencil operation lands on the Vulkan operation of the same name.
    ///
    /// The eight arms are written out because a numeric coincidence between
    /// two vendors' enums is not a contract, and writing them out is where a
    /// swapped pair gets in: `IncrementClamp` bound to `DECREMENT_AND_CLAMP`
    /// and `DecrementClamp` to `INCREMENT_AND_CLAMP` is still eight distinct
    /// operations and still compiles. What it produces is a stencil buffer
    /// that counts the wrong way, which no test of the plan's *shape* can see.
    ///
    /// So the answer is checked against its own name, through the same
    /// derivation [`crate::sampler`] uses for the comparison table --- the
    /// guest spelling, plus the one conjunction Vulkan writes and Metal does
    /// not.
    #[test]
    fn every_stencil_operation_lands_on_the_operation_of_the_same_name() {
        for guest in GuestOp::ALL {
            assert_eq!(
                format!("{:?}", stencil_op(guest)),
                crate::naming::vulkan_spelling(&format!("{guest:?}")),
                "{guest:?}"
            );
        }
    }

    /// And no two of them collapse onto one, which a name check alone would
    /// miss if two guest spellings ever normalised alike.
    #[test]
    fn the_eight_stencil_operations_stay_eight() {
        let mapped: std::collections::BTreeSet<i32> = GuestOp::ALL
            .iter()
            .map(|op| stencil_op(*op).as_raw())
            .collect();
        assert_eq!(mapped.len(), GuestOp::ALL.len());
    }
}
