//! Fences, events, and barriers, as the model carries them.
//!
//! # A barrier is an operation here, and the ledger's no-op is not this layer's
//!
//! Every barrier opcode on this wire is `ProvenNoOp` in
//! [`reims_vgpu_protocol::closure`], and each one's capability cell says why:
//! "one host submission per pass boundary", "one host submission per dispatch".
//! Those are true statements about the *current executor's* submission
//! granularity. They are not statements about the guest's request, and this
//! crate cannot see a submission at all.
//!
//! The replacement exists in order to batch differently. A dependency compiler
//! that merges two dispatches into one submission has, at that moment, removed
//! the thing that made the barrier free — and a model that had folded the
//! barrier away at decode would have no record left to honour. So the barrier
//! is a resolved operation with its declared scope and stages intact, and
//! whether an executor's own ordering already satisfies it is that executor's
//! judgement to make against its own batching, one submission at a time.
//!
//! This is the plan's rule about where an invariant lives, applied to the one
//! place it is easy to get backwards: the no-op is a *host* fact, so it belongs
//! to the host.
//!
//! # A fence belongs to a rail; an event does not
//!
//! `updateFence:` has three different opcodes on three encoders, and the model
//! keeps the rail that carried it — a blit fence and a render fence are the
//! same object waited on from different encoders, and the stages they name are
//! different vocabularies. The render pair carries `MTLRenderStages`; the blit
//! pair carries none, because its selector has none. That asymmetry is in the
//! type: [`FenceOp::stages`] is an `Option`, and `None` means the record had no
//! stage argument rather than that the guest asked for no stages.
//!
//! Events are not encoder-scoped. They carry a value, they compare against a
//! monotonic generation, and a wait for a value nothing has signalled yet is
//! ordinary rather than an error — which is why an event wait is a
//! prerequisite, not a refusal. [`crate::ready`] already draws exactly that
//! distinction for stamp waits, and an event wait joins it.

use crate::access::Participations;
use crate::identity::ResourceId;
/// Re-exported because a barrier's stage mask is part of this vocabulary, and
/// the executor that has to translate it sees this crate and not the protocol
/// one. Reaching past core for it would be the executor taking a dependency its
/// own layering rule forbids.
pub use reims_vgpu_protocol::residency::RenderStages;
pub use reims_vgpu_protocol::sync::{BarrierKind, BarrierScope, EventKind, FenceKind};

/// A fence update or wait, on the encoder that issued it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FenceOp {
    pub kind: FenceKind,
    pub fence: ResourceId,
    /// The stages the record named, or `None` when its selector carries none.
    ///
    /// The blit encoder's fence pair has no stage argument at all. Storing a
    /// zero mask for it would be indistinguishable from a render record whose
    /// guest passed zero, and the two mean different things: one is a selector
    /// without the field, the other is a guest declining to use it.
    pub stages: Option<RenderStages>,
}

impl FenceOp {
    /// The memory this record names: none.
    ///
    /// A fence is an ordering object, not memory. The `ResourceId` it carries
    /// names the fence itself, and declaring a participation on it would put
    /// the ordering primitive into the hazard graph as though it were bytes a
    /// draw could race against — an edge built for the wrong reason, and one
    /// that would make two waits on one fence look like a write conflict.
    /// What a fence orders is [`crate::depend`]'s to model, from the fence's
    /// own semantics.
    #[must_use]
    pub const fn participations(&self) -> Participations {
        Participations::NONE
    }
}

/// An event signal or wait.
///
/// The value is `u64` because the wire's is. An event's ordering rule is the
/// API's own: a signal that does not advance the generation changes nothing,
/// and a wait is satisfied when the generation reaches the value or has already
/// passed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventOp {
    pub kind: EventKind,
    pub event: ResourceId,
    pub value: u64,
}

impl EventOp {
    /// Whether this signal advances an event currently at `generation`.
    ///
    /// A signal that does not advance is a no-op *by the API's monotonic rule*,
    /// which is a different claim from this device choosing to drop it — and
    /// the difference is why it is answered here rather than by an executor
    /// silently skipping the work.
    #[must_use]
    pub const fn advances(&self, generation: u64) -> bool {
        matches!(self.kind, EventKind::Signal) && self.value > generation
    }

    /// Whether this wait is already satisfied by an event at `generation`.
    #[must_use]
    pub const fn satisfied_by(&self, generation: u64) -> bool {
        matches!(self.kind, EventKind::Wait) && generation >= self.value
    }
}

impl EventOp {
    /// The memory this record names: none, for the reason
    /// [`FenceOp::participations`] gives.
    #[must_use]
    pub const fn participations(&self) -> Participations {
        Participations::NONE
    }
}

/// A contiguous window of an EXEC's resolved resource list.
///
/// A resource barrier names a counted list, and the list can be any length. It
/// lives in the transaction's own resource arena and the operation names a
/// window of it, so a barrier costs two `u32` rather than an allocation — and
/// so the same arena serves every operation class that names a list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceSpan {
    pub start: u32,
    pub len: u32,
}

impl ResourceSpan {
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// The window as a range into the arena.
    #[must_use]
    pub const fn range(self) -> core::ops::Range<usize> {
        self.start as usize..(self.start as usize + self.len as usize)
    }
}

/// What a barrier orders against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierTarget {
    /// The resources the record listed.
    Resources(ResourceSpan),
    /// A whole class of memory.
    Scope(BarrierScope),
    /// The render rail's `textureBarrier`, which names nothing.
    ///
    /// Its target is the pass's own colour attachments — the API defines it as
    /// making previously written fragments readable by later ones in the same
    /// pass — and those are the pass's attachments rather than anything the
    /// record carries. Naming it as its own target keeps that "the record
    /// carried no list" fact visible instead of turning into an empty list,
    /// which orders nothing.
    Texture,
}

/// A barrier, with everything the record declared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BarrierOp {
    pub target: BarrierTarget,
    /// Stages the barrier waits on, or `None` when the selector carries none.
    ///
    /// The compute rail's barriers have no stage masks: its selectors do not
    /// take them, and the record is four bytes shorter rather than four bytes
    /// of zero. The render rail's carry both.
    pub after_stages: Option<RenderStages>,
    /// Stages the barrier blocks.
    pub before_stages: Option<RenderStages>,
}

impl BarrierOp {
    #[must_use]
    pub const fn kind(&self) -> BarrierKind {
        match self.target {
            BarrierTarget::Resources(_) => BarrierKind::Resources,
            BarrierTarget::Scope(_) => BarrierKind::Scope,
            BarrierTarget::Texture => BarrierKind::Texture,
        }
    }

    /// The memory this record names: none, and the list it carries is why
    /// that has to be said rather than assumed.
    ///
    /// A `BarrierTarget::Resources` span names resources, and it would be easy
    /// to read that list as participation. It is not: a barrier neither reads
    /// nor writes the resources it names, it declares an ordering *about*
    /// them. Turning the list into read-write accesses would make every
    /// barrier conflict with every access to everything it named, which is a
    /// serialisation the guest did not ask for and a publication claim the
    /// barrier never made. The ordering it does ask for is
    /// [`Self::orders_anything`] and the dependency layer's to act on.
    #[must_use]
    pub const fn participations(&self) -> Participations {
        Participations::NONE
    }

    /// Whether this barrier asks for any ordering at all.
    ///
    /// An empty resource list and an empty scope both order nothing, and a
    /// guest may legally send either. Saying so here means an executor can tell
    /// "the guest asked for nothing" from "the device decided it owed nothing",
    /// which are the two answers a barrier census has to keep apart.
    #[must_use]
    pub const fn orders_anything(&self) -> bool {
        match self.target {
            BarrierTarget::Resources(span) => !span.is_empty(),
            BarrierTarget::Scope(scope) => !scope.is_empty(),
            BarrierTarget::Texture => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};
    use crate::operation::{classify, OperationClass, OperationHome};
    use reims_vgpu_protocol::closure::{Rail, LEDGER};
    use reims_vgpu_protocol::sync::{barrier_kind, event_kind, fence_kind};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    /// Whether the ledger refuses this operation by name.
    ///
    /// A refused row is *judged*: the device knows exactly what the guest asked
    /// for and knows it cannot honour it. It keeps its operation class, because
    /// refusing by name needs the name — but it gets no executable payload,
    /// because there is nothing to execute.
    fn is_refused(op: &reims_vgpu_protocol::closure::Op) -> bool {
        matches!(
            op.closure,
            reims_vgpu_protocol::closure::Closure::Refused { .. }
        )
    }

    /// The vocabulary and the classification agree in both directions: every
    /// operation the model classifies as a fence, an event or a barrier has a
    /// payload kind here — unless it is refused, in which case it has a route
    /// instead and deliberately has no way to be executed.
    #[test]
    fn the_synchronisation_classes_are_exactly_the_operations_with_a_kind() {
        for op in LEDGER {
            let Some(opcode) = op.opcode else { continue };
            let class = match classify(op) {
                Some(OperationHome::Stream(class)) => Some(class),
                _ => None,
            };
            let executable = |c: OperationClass| class == Some(c) && !is_refused(op);
            assert_eq!(
                fence_kind(op.rail, opcode).is_some(),
                executable(OperationClass::Fence),
                "{:?} {opcode:#x} disagrees about being an executable fence",
                op.rail
            );
            assert_eq!(
                event_kind(op.rail, opcode).is_some(),
                executable(OperationClass::Event),
                "{:?} {opcode:#x} disagrees about being an executable event",
                op.rail
            );
            assert_eq!(
                barrier_kind(op.rail, opcode).is_some(),
                executable(OperationClass::Barrier),
                "{:?} {opcode:#x} disagrees about being an executable barrier",
                op.rail
            );
        }
    }

    /// The refused operations in these classes keep their class and their
    /// route, and get no payload.
    ///
    /// One row today — the bounded event wait — and the count is asserted so a
    /// second one cannot appear without this claim being read again.
    #[test]
    fn a_refused_operation_keeps_its_name_and_gains_no_payload() {
        let refused: Vec<_> = LEDGER
            .iter()
            .filter(|o| {
                is_refused(o)
                    && matches!(
                        classify(o),
                        Some(OperationHome::Stream(
                            OperationClass::Fence | OperationClass::Event | OperationClass::Barrier
                        ))
                    )
            })
            .collect();
        assert_eq!(refused.len(), 1);
        let only = refused[0];
        assert_eq!(only.rail, Rail::Event);
        assert_eq!(
            only.opcode,
            Some(reims_vgpu_protocol::sync::OPCODE_WAIT_EVENT_TIMEOUT)
        );
        assert_eq!(event_kind(only.rail, only.opcode.expect("opcode")), None);
        let reims_vgpu_protocol::closure::Closure::Refused { route, .. } = only.closure else {
            unreachable!("filtered")
        };
        assert_eq!(route, "event_wait_timeout_unsupported");
    }

    /// Every barrier opcode is a proven no-op in the ledger, and every one of
    /// them still becomes a barrier operation here.
    ///
    /// This is the module's central claim, so it is a test rather than a
    /// comment: the ledger's no-op rests on a submission-granularity cell that
    /// belongs to an executor, and the replacement is going to change that
    /// granularity. A barrier folded away at decode cannot be honoured later.
    #[test]
    fn a_proven_host_no_op_is_still_a_modelled_operation() {
        let mut barriers = 0;
        for op in LEDGER {
            let Some(opcode) = op.opcode else { continue };
            if barrier_kind(op.rail, opcode).is_none() {
                continue;
            }
            barriers += 1;
            assert!(
                matches!(
                    op.closure,
                    reims_vgpu_protocol::closure::Closure::ProvenNoOp { .. }
                ),
                "{:?} {opcode:#x} is no longer a proven no-op; the reasoning in \
                 this module's documentation needs revisiting",
                op.rail
            );
            assert_eq!(
                classify(op),
                Some(OperationHome::Stream(OperationClass::Barrier))
            );
        }
        assert_eq!(barriers, 5, "three render barriers and two compute ones");
    }

    /// The blit fence pair has no stage argument, and that is not the same as
    /// zero stages.
    #[test]
    fn a_selector_without_stages_is_none_and_not_zero() {
        let blit = FenceOp {
            kind: FenceKind::Update,
            fence: res(1),
            stages: None,
        };
        let render_zero = FenceOp {
            kind: FenceKind::Update,
            fence: res(1),
            stages: Some(RenderStages(0)),
        };
        assert_ne!(blit, render_zero);
    }

    /// The API's own monotonic rule, answered by the model rather than by an
    /// executor quietly skipping work.
    #[test]
    fn a_signal_that_does_not_advance_changes_nothing() {
        let signal = EventOp {
            kind: EventKind::Signal,
            event: res(2),
            value: 5,
        };
        assert!(signal.advances(4));
        assert!(!signal.advances(5));
        assert!(!signal.advances(6));
    }

    /// A wait for a value already passed is satisfied; a wait for one not yet
    /// reached is a prerequisite, not a refusal.
    #[test]
    fn a_wait_is_satisfied_at_or_past_its_value() {
        let wait = EventOp {
            kind: EventKind::Wait,
            event: res(2),
            value: 5,
        };
        assert!(!wait.satisfied_by(4));
        assert!(wait.satisfied_by(5));
        assert!(wait.satisfied_by(6));
        // A wait is not a signal and a signal is not a wait, whatever the
        // values say.
        assert!(!wait.advances(0));
    }

    /// A barrier over nothing is distinguishable from a barrier over
    /// something, so a census can tell a guest asking for nothing from a device
    /// deciding it owes nothing.
    #[test]
    fn an_empty_barrier_orders_nothing_and_says_so() {
        let empty_list = BarrierOp {
            target: BarrierTarget::Resources(ResourceSpan::default()),
            after_stages: None,
            before_stages: None,
        };
        let empty_scope = BarrierOp {
            target: BarrierTarget::Scope(BarrierScope(0)),
            after_stages: None,
            before_stages: None,
        };
        let real = BarrierOp {
            target: BarrierTarget::Scope(BarrierScope(BarrierScope::BUFFERS)),
            after_stages: Some(RenderStages(RenderStages::VERTEX)),
            before_stages: Some(RenderStages(RenderStages::FRAGMENT)),
        };
        assert!(!empty_list.orders_anything());
        assert!(!empty_scope.orders_anything());
        assert!(real.orders_anything());
        assert!(BarrierOp {
            target: BarrierTarget::Texture,
            after_stages: None,
            before_stages: None,
        }
        .orders_anything());
    }

    /// A span names a window of the arena and nothing outside it.
    #[test]
    fn a_resource_span_is_a_window_of_the_transactions_own_list() {
        let span = ResourceSpan { start: 3, len: 2 };
        assert_eq!(span.range(), 3..5);
        assert!(!span.is_empty());
        assert!(ResourceSpan { start: 3, len: 0 }.is_empty());
    }
}
