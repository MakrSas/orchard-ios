//! Lifting the fence, event and barrier records.
//!
//! # One class of meaning, four record shapes, and no shared layout
//!
//! Every record here is an ordering constraint, and that is where the
//! similarity stops. A render fence is a ref beside a 32-bit stage mask; a blit
//! fence is a ref alone, in a shorter record; a render scope barrier puts its
//! two stage masks in **one byte each**, while its resource sibling gives them
//! sixteen bits each; the compute barriers carry no stage masks at all, because
//! their selectors take none.
//!
//! So the widths a lifted record carries are per record and not per field name.
//! They are widened into [`RenderStages`] and [`BarrierScope`] here — a
//! widening loses nothing and the semantic question is the same at every width
//! — but the *absence* of a field is kept as `None` rather than as a zero mask.
//! A selector that has no stages and a guest that passed zero stages are
//! different facts, and only one of them is a guest decision.
//!
//! # A refused row still decodes, and still must not become an operation
//!
//! `waitForEvent:value:timeoutMS:` has an established contract and a settled
//! refusal: this device runs no clock against the guest's, so executing it as
//! the unbounded wait it resembles would turn a guest's timeout into a hang.
//! Its record is well formed and this module still declines it, through
//! [`DecodeRefusal::RefusedByContract`] rather than through the shape refusals
//! — a refusal on a record nothing is wrong with.

use super::{no_record, short, DecodeRefusal};
use crate::closure::Rail;
use crate::residency::RenderStages;
use crate::sync::{
    barrier_kind, event_kind, fence_kind, BarrierKind, BarrierScope, EventKind, FenceKind,
    OPCODE_SIGNAL_EVENT, OPCODE_WAIT_EVENT,
};
use reims_vgpu_wire::op::Op;
use reims_vgpu_wire::ops::render::RefBind;
use reims_vgpu_wire::ops::{
    blit as wire_blit, compute as wire_compute, event as wire_event, render as wire_render,
};

/// A fence update or wait, with the guest's ref.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FenceRecord {
    pub kind: FenceKind,
    pub fence_ref: u32,
    /// The stages the record carried, or `None` when its selector has none.
    pub stages: Option<RenderStages>,
}

/// An event signal or unbounded wait, with the guest's ref.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventRecord {
    pub kind: EventKind,
    pub event_ref: u32,
    pub value: u64,
}

/// A barrier, in whichever of the three shapes its rail and opcode name.
///
/// The resource form borrows the refs out of the record rather than collecting
/// them. A barrier's list is guest-sized and arrives once per packet; copying
/// it here would allocate on a path that has an arena waiting for it downstream
/// — the model appends the resolved ids into the transaction's own resource
/// arena and names a window of it — so a copy here would be a copy nobody
/// wanted, made before the ids even exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierRecord<'a> {
    Resources {
        refs: &'a [RefBind],
        after_stages: Option<RenderStages>,
        before_stages: Option<RenderStages>,
    },
    Scope {
        scope: BarrierScope,
        after_stages: Option<RenderStages>,
        before_stages: Option<RenderStages>,
        /// The byte the render rail's scope record writes beside its scope and
        /// that nothing has been made to move.
        ///
        /// Carried rather than dropped because dropping it would be a claim: a
        /// build that started using it would be absorbed silently, and the
        /// whole reason the wire crate keeps the field separate from `scope` is
        /// to keep that alarm reachable. `None` on the rails whose record has
        /// no such byte.
        unidentified_u8: Option<u8>,
    },
    /// `textureBarrier`, whose record is its header and nothing else.
    Texture,
}

impl BarrierRecord<'_> {
    /// Which barrier this is.
    #[must_use]
    pub const fn kind(&self) -> BarrierKind {
        match self {
            Self::Resources { .. } => BarrierKind::Resources,
            Self::Scope { .. } => BarrierKind::Scope,
            Self::Texture => BarrierKind::Texture,
        }
    }
}

/// One lifted ordering record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncRecord<'a> {
    Fence(FenceRecord),
    Event(EventRecord),
    Barrier(BarrierRecord<'a>),
}

/// Lift an ordering record out of its bytes.
///
/// The rail is a parameter because the three encoders number their fences and
/// barriers independently and the numbers are nowhere near each other: the same
/// opcode is a different record on a different rail, and there is no rail-free
/// form of this to reach for.
pub fn decode<'a>(rail: Rail, op: &Op<'a>) -> Result<SyncRecord<'a>, DecodeRefusal> {
    let opcode = op.opcode();
    if let Some(kind) = fence_kind(rail, opcode) {
        return Ok(SyncRecord::Fence(fence(rail, kind, op)?));
    }
    if let Some(kind) = barrier_kind(rail, opcode) {
        return Ok(SyncRecord::Barrier(barrier(rail, kind, op)?));
    }
    if let Some(kind) = event_kind(rail, opcode) {
        return Ok(SyncRecord::Event(event(kind, op)?));
    }
    Err(no_record(rail, opcode))
}

fn fence(rail: Rail, kind: FenceKind, op: &Op<'_>) -> Result<FenceRecord, DecodeRefusal> {
    let opcode = op.opcode();
    let have = op.payload.len();
    match rail {
        Rail::Render => {
            let r = wire_render::fence(op).map_err(|_| {
                short(
                    rail,
                    opcode,
                    have,
                    core::mem::size_of::<wire_render::Fence>(),
                )
            })?;
            Ok(FenceRecord {
                kind,
                fence_ref: r.fence_ref.get(),
                stages: Some(RenderStages(r.stages.get())),
            })
        }
        // The blit encoder's fence record is a ref and nothing else, in a
        // record four bytes shorter than the render one. Sharing a layout
        // across encoders is precisely the mistake this arm exists to avoid.
        _ => {
            let r = wire_blit::object_ref(op)
                .map_err(|_| short(rail, opcode, have, core::mem::size_of::<wire_blit::Ref>()))?;
            Ok(FenceRecord {
                kind,
                fence_ref: r.object_ref.get(),
                stages: None,
            })
        }
    }
}

fn event(kind: EventKind, op: &Op<'_>) -> Result<EventRecord, DecodeRefusal> {
    debug_assert!(matches!(
        op.opcode(),
        OPCODE_SIGNAL_EVENT | OPCODE_WAIT_EVENT
    ));
    let r = wire_event::signal_wait(op).map_err(|_| {
        short(
            Rail::Event,
            op.opcode(),
            op.payload.len(),
            core::mem::size_of::<wire_event::SignalWait>(),
        )
    })?;
    Ok(EventRecord {
        kind,
        event_ref: r.event_ref.get(),
        value: r.value.get(),
    })
}

fn barrier<'a>(
    rail: Rail,
    kind: BarrierKind,
    op: &Op<'a>,
) -> Result<BarrierRecord<'a>, DecodeRefusal> {
    let opcode = op.opcode();
    let have = op.payload.len();
    Ok(match (kind, rail) {
        (BarrierKind::Texture, _) => BarrierRecord::Texture,
        (BarrierKind::Resources, Rail::Render) => {
            let (head, refs) = wire_render::memory_barrier_resources(op).map_err(|_| {
                super::counted_head(
                    rail,
                    op,
                    core::mem::size_of::<wire_render::MemoryBarrierResources>(),
                )
            })?;
            BarrierRecord::Resources {
                refs,
                after_stages: Some(RenderStages(u32::from(head.after_stages.get()))),
                before_stages: Some(RenderStages(u32::from(head.before_stages.get()))),
            }
        }
        (BarrierKind::Resources, _) => {
            let (_, refs) = wire_compute::memory_barrier_resources(op).map_err(|_| {
                super::counted_head(
                    rail,
                    op,
                    core::mem::size_of::<wire_compute::MemoryBarrierResources>(),
                )
            })?;
            BarrierRecord::Resources {
                refs,
                after_stages: None,
                before_stages: None,
            }
        }
        (BarrierKind::Scope, Rail::Render) => {
            let r = wire_render::memory_barrier_scope(op).map_err(|_| {
                short(
                    rail,
                    opcode,
                    have,
                    core::mem::size_of::<wire_render::MemoryBarrierScope>(),
                )
            })?;
            BarrierRecord::Scope {
                scope: BarrierScope(u32::from(r.scope)),
                after_stages: Some(RenderStages(u32::from(r.after_stages))),
                before_stages: Some(RenderStages(u32::from(r.before_stages))),
                unidentified_u8: Some(r.unidentified_u8),
            }
        }
        (BarrierKind::Scope, _) => {
            let r = wire_compute::memory_barrier_scope(op).map_err(|_| {
                short(
                    rail,
                    opcode,
                    have,
                    core::mem::size_of::<wire_compute::MemoryBarrierScope>(),
                )
            })?;
            BarrierRecord::Scope {
                scope: BarrierScope(u32::from(r.scope.get())),
                after_stages: None,
                before_stages: None,
                unidentified_u8: None,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use reims_vgpu_wire::op::{op, OP_HEADER_LEN};

    fn record(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let total = (OP_HEADER_LEN + payload.len()) as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&total.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn lift<'a>(rail: Rail, bytes: &'a [u8]) -> Result<SyncRecord<'a>, DecodeRefusal> {
        let view = op(bytes, 0).expect("framed");
        decode(rail, &view)
    }

    /// The render rail's fence carries a stage mask and the blit rail's does
    /// not, and the same ref reads back through both. A record that shared one
    /// layout would read the blit fence's four bytes of nothing as stages.
    #[test]
    fn the_two_fence_shapes_agree_on_the_ref_and_disagree_on_the_stages() {
        let mut render_payload = 6464u32.to_le_bytes().to_vec();
        render_payload.extend_from_slice(&RenderStages::FRAGMENT.to_le_bytes());
        let render = record(wire_render::OPCODE_UPDATE_FENCE, &render_payload);
        assert_eq!(
            lift(Rail::Render, &render),
            Ok(SyncRecord::Fence(FenceRecord {
                kind: FenceKind::Update,
                fence_ref: 6464,
                stages: Some(RenderStages(RenderStages::FRAGMENT)),
            }))
        );

        let blit = record(wire_blit::OPCODE_WAIT_FOR_FENCE, &6464u32.to_le_bytes());
        assert_eq!(
            lift(Rail::Blit, &blit),
            Ok(SyncRecord::Fence(FenceRecord {
                kind: FenceKind::Wait,
                fence_ref: 6464,
                stages: None,
            }))
        );
    }

    /// A signal and an unbounded wait lift the same two fields, and the value
    /// is a full 64 bits: an event generation is not a counter this device gets
    /// to narrow.
    #[test]
    fn an_event_lifts_its_ref_and_its_whole_value() {
        let mut payload = 77u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        for (opcode, kind) in [
            (OPCODE_SIGNAL_EVENT, EventKind::Signal),
            (OPCODE_WAIT_EVENT, EventKind::Wait),
        ] {
            assert_eq!(
                lift(Rail::Event, &record(opcode, &payload)),
                Ok(SyncRecord::Event(EventRecord {
                    kind,
                    event_ref: 77,
                    value: 0x1234_5678_9abc_def0,
                }))
            );
        }
    }

    /// The bounded wait is well formed and still refused, and the refusal says
    /// the contract settled it rather than that nothing is known about it.
    #[test]
    fn the_bounded_wait_is_refused_by_contract_and_not_as_an_open_question() {
        let mut payload = 77u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&5u64.to_le_bytes());
        payload.extend_from_slice(&42u32.to_le_bytes());
        let bytes = record(crate::sync::OPCODE_WAIT_EVENT_TIMEOUT, &payload);
        assert_eq!(
            lift(Rail::Event, &bytes),
            Err(DecodeRefusal::RefusedByContract {
                rail: Rail::Event,
                opcode: crate::sync::OPCODE_WAIT_EVENT_TIMEOUT,
            })
        );
    }

    /// A resource barrier's refs are borrowed out of the guest's own bytes:
    /// the lifted record points into the buffer it was given rather than into a
    /// copy of it.
    #[test]
    fn a_resource_barrier_borrows_its_list_rather_than_copying_it() {
        let mut payload = 2u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&1u16.to_le_bytes());
        payload.extend_from_slice(&2u16.to_le_bytes());
        payload.extend_from_slice(&5151u32.to_le_bytes());
        payload.extend_from_slice(&4343u32.to_le_bytes());
        let bytes = record(wire_render::OPCODE_MEMORY_BARRIER_RESOURCES, &payload);
        let SyncRecord::Barrier(BarrierRecord::Resources {
            refs,
            after_stages,
            before_stages,
        }) = lift(Rail::Render, &bytes).expect("lifted")
        else {
            panic!("not a resource barrier");
        };
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].object_ref.get(), 5151);
        assert_eq!(refs[1].object_ref.get(), 4343);
        assert_eq!(after_stages, Some(RenderStages(1)));
        assert_eq!(before_stages, Some(RenderStages(2)));
        let base = bytes.as_ptr_range();
        assert!(base.contains(&refs.as_ptr().cast::<u8>()));
    }

    /// The compute rail's resource barrier is the same list with no stage
    /// masks, and `None` is how that is said. A zero mask here would be
    /// indistinguishable from a render guest that passed zero.
    #[test]
    fn the_compute_resource_barrier_has_no_stages_rather_than_zero_ones() {
        let mut payload = 1u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&5151u32.to_le_bytes());
        let bytes = record(wire_compute::OPCODE_MEMORY_BARRIER_RESOURCES, &payload);
        assert_eq!(
            lift(Rail::Compute, &bytes),
            Ok(SyncRecord::Barrier(BarrierRecord::Resources {
                refs: reims_vgpu_wire::view_slice::<RefBind>(&bytes[OP_HEADER_LEN + 4..], 1)
                    .expect("refs"),
                after_stages: None,
                before_stages: None,
            }))
        );
    }

    /// A guest count larger than the record it sits in is reported with both
    /// numbers, because "the guest asked for 200" and "the record held 12" is
    /// the pair that says which of the two is wrong.
    #[test]
    fn a_list_longer_than_its_record_reports_the_count_and_the_bytes() {
        let mut payload = 200u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&5151u32.to_le_bytes());
        let bytes = record(wire_render::OPCODE_MEMORY_BARRIER_RESOURCES, &payload);
        assert_eq!(
            lift(Rail::Render, &bytes),
            Err(DecodeRefusal::CountOverruns {
                rail: Rail::Render,
                opcode: wire_render::OPCODE_MEMORY_BARRIER_RESOURCES,
                count: 200,
                have: payload.len(),
            })
        );
    }

    /// A record too short to hold even the count is short, not overrunning:
    /// there is no count to report.
    #[test]
    fn a_barrier_with_no_room_for_its_count_is_short() {
        let bytes = record(wire_render::OPCODE_MEMORY_BARRIER_RESOURCES, &[0u8, 0, 0]);
        assert!(matches!(
            lift(Rail::Render, &bytes),
            Err(DecodeRefusal::Short { .. })
        ));
    }

    /// The two scope barriers put their fields at different widths, and the
    /// render one's unidentified byte survives the lift. Both reach the same
    /// scope value, which is the whole reason the widths are widened here.
    #[test]
    fn the_two_scope_barriers_reach_one_scope_and_keep_their_own_widths() {
        let render = record(
            wire_render::OPCODE_MEMORY_BARRIER_SCOPE,
            &[BarrierScope::RENDER_TARGETS as u8, 0, 1, 2],
        );
        assert_eq!(
            lift(Rail::Render, &render),
            Ok(SyncRecord::Barrier(BarrierRecord::Scope {
                scope: BarrierScope(BarrierScope::RENDER_TARGETS),
                after_stages: Some(RenderStages(1)),
                before_stages: Some(RenderStages(2)),
                unidentified_u8: Some(0),
            }))
        );

        // The compute record writes two bytes of a four-byte payload; the rest
        // is ring residue and must not reach the lifted record.
        let compute = record(
            wire_compute::OPCODE_MEMORY_BARRIER_SCOPE,
            &[BarrierScope::RENDER_TARGETS as u8, 0, 0xaa, 0xaa],
        );
        assert_eq!(
            lift(Rail::Compute, &compute),
            Ok(SyncRecord::Barrier(BarrierRecord::Scope {
                scope: BarrierScope(BarrierScope::RENDER_TARGETS),
                after_stages: None,
                before_stages: None,
                unidentified_u8: None,
            }))
        );
    }

    /// `textureBarrier` is a header and nothing else, and it lifts to the one
    /// target that names nothing.
    #[test]
    fn the_texture_barrier_lifts_from_an_empty_payload() {
        let bytes = record(wire_render::OPCODE_TEXTURE_BARRIER, &[]);
        let lifted = lift(Rail::Render, &bytes).expect("lifted");
        assert_eq!(lifted, SyncRecord::Barrier(BarrierRecord::Texture));
        let SyncRecord::Barrier(barrier) = lifted else {
            panic!("not a barrier");
        };
        assert_eq!(barrier.kind(), BarrierKind::Texture);
    }

    /// The compute rail's fence pair is unresolved in the ledger, so it lifts
    /// nothing — and the refusal says the contract is open rather than that the
    /// opcode is unknown. Reading it as the render rail's fence is exactly what
    /// the rail parameter prevents.
    #[test]
    fn the_compute_fence_pair_is_unjudged_rather_than_lifted_as_another_rails() {
        for opcode in [
            wire_compute::OPCODE_UPDATE_FENCE,
            wire_compute::OPCODE_WAIT_FOR_FENCE,
        ] {
            let bytes = record(opcode, &6464u32.to_le_bytes());
            assert_eq!(
                lift(Rail::Compute, &bytes),
                Err(DecodeRefusal::Unjudged {
                    rail: Rail::Compute,
                    opcode,
                })
            );
        }
    }

    /// Every ordering opcode the vocabulary names lifts a record of the class
    /// it names — no arm of the vocabulary is reachable and undecodable.
    #[test]
    fn every_named_ordering_opcode_lifts_a_record_of_its_own_class() {
        let payload = [0u8; 32];
        let mut seen = 0usize;
        for rail in [Rail::Render, Rail::Blit, Rail::Compute, Rail::Event] {
            for opcode in 0u32..0x200 {
                let expected_fence = fence_kind(rail, opcode).is_some();
                let expected_barrier = barrier_kind(rail, opcode).is_some();
                let expected_event = event_kind(rail, opcode).is_some();
                if !(expected_fence || expected_barrier || expected_event) {
                    continue;
                }
                seen += 1;
                let bytes = record(opcode, &payload);
                match lift(rail, &bytes).expect("a named opcode lifts") {
                    SyncRecord::Fence(_) => assert!(expected_fence),
                    SyncRecord::Barrier(_) => assert!(expected_barrier),
                    SyncRecord::Event(_) => assert!(expected_event),
                }
            }
        }
        assert_eq!(seen, 11);
    }
}
