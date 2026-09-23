//! What a stream operation becomes in the model, and where it is reached from.
//!
//! # Why this is a classification and not an enum of payloads yet
//!
//! The replacement's resolved vocabulary has to be exhaustive over the wire
//! contract — that is the whole reason the four-arm `ResolvedCommand` the old
//! seam started from cannot be wrapped, because wrapping it preserves the
//! silent side paths it does not name. "Exhaustive" is checkable only against
//! an enumeration of what exists, and [`reims_vgpu_protocol::closure`] is that
//! enumeration.
//!
//! So this comes first: a total map from every judged operation to the class of
//! resolved operation it becomes. The payload types follow, one class at a
//! time, and each one lands with the guarantee that the operations it has to
//! cover are already named rather than discovered.
//!
//! # Two homes, and the difference is structural
//!
//! Not every serializer record is a stream operation. The root rail's records —
//! `newTextureWithDescriptor:`, `deleteSamplerStateRef:` — never appear inside
//! an EXEC's command stream. They arrive as the payload of an object-list or
//! object-destroy packet, which is a `ResourceLifecycle` transaction, and they
//! have no encoder, no segment and no position in a command buffer. Folding
//! them into the stream vocabulary would give them ordering they do not have.
//! [`OperationHome`] says which of the two a record is, and the root rail is
//! the whole of one side.

use reims_vgpu_protocol::closure::{Op, Rail, LEDGER};

/// The class of resolved operation a stream record becomes.
///
/// One variant per family the wire contract distinguishes, not one per opcode:
/// the payload types below each class carry the opcode's own fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperationClass {
    /// A segment begins or ends, and an encoder's lifetime crosses — or does
    /// not cross — that boundary with it.
    EncoderBoundary,
    /// A render-encoder draw, bind, or pipeline/pass state change.
    Render,
    /// A compute-encoder dispatch, bind, or state change.
    Compute,
    /// A blit copy, fill, or mipmap generation.
    Blit,
    /// A cross-encoder event signal or wait.
    Event,
    /// An encoder-scoped fence update or wait.
    Fence,
    /// A memory or texture barrier.
    Barrier,
    /// Residency participation, and the content-representation operations —
    /// optimize, synchronize, invalidate — that say something about where a
    /// resource's content is rather than what it is.
    ResourceState,
    /// A question with a reply destination.
    InfoQuery,
    /// An indirect-command-buffer mutation or execution.
    IndirectCommand,
    /// An effect the model publishes when work completes rather than when the
    /// record is recorded.
    ///
    /// **Empty today, and that is a reading rather than an omission.** Every
    /// completion-shaped thing on the wire so far is a record that *arms* an
    /// effect — `setVisibilityResultMode:` arms an occlusion count — and the
    /// effect itself is a fact an executor returns. The variant exists because
    /// the plan's vocabulary names it and because a completion effect that
    /// arrives as its own record would otherwise be classified as whatever
    /// encoder it happened to sit in; the census prints its zero so the
    /// emptiness stays visible.
    CompletionEffect,
}

impl OperationClass {
    pub const ALL: &'static [OperationClass] = &[
        OperationClass::EncoderBoundary,
        OperationClass::Render,
        OperationClass::Compute,
        OperationClass::Blit,
        OperationClass::Event,
        OperationClass::Fence,
        OperationClass::Barrier,
        OperationClass::ResourceState,
        OperationClass::InfoQuery,
        OperationClass::IndirectCommand,
        OperationClass::CompletionEffect,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::EncoderBoundary => "encoder_boundary",
            Self::Render => "render",
            Self::Compute => "compute",
            Self::Blit => "blit",
            Self::Event => "event",
            Self::Fence => "fence",
            Self::Barrier => "barrier",
            Self::ResourceState => "resource_state",
            Self::InfoQuery => "info_query",
            Self::IndirectCommand => "indirect_command",
            Self::CompletionEffect => "completion_effect",
        }
    }
}

/// Where in the model an operation is reached from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperationHome {
    /// A resolved operation inside an `Exec` transaction's command stream,
    /// ordered by its encoder and its segment.
    Stream(OperationClass),
    /// An object create or destroy record carried by a `ResourceLifecycle`
    /// transaction. It has no encoder and no position in a command buffer.
    ObjectLifecycle,
}

/// The home of one judged operation, or `None` when the ledger has not
/// established what it is.
///
/// `None` is the same answer [`crate::transaction::classify`] gives for an
/// unjudged packet and for the same reason: an operation the model cannot
/// describe must not be admitted into a vocabulary that promises ordering and
/// completion for everything in it.
#[must_use]
pub fn classify(op: &Op) -> Option<OperationHome> {
    use OperationClass::*;
    if op.closure.blocks_cutover() {
        return None;
    }
    if op.rail == Rail::Root {
        return Some(OperationHome::ObjectLifecycle);
    }
    let Some(opcode) = op.opcode else {
        // The records with no fixed opcode. `beginSegment:protectionOptions:`
        // writes the segment header, which is the boundary itself; the blit
        // `withCommand:` selectors write their command argument into the
        // record's opcode field, so they are whichever opcode they emitted and
        // are classified by the family they belong to.
        return Some(OperationHome::Stream(
            if op.selector.starts_with("beginSegment") {
                EncoderBoundary
            } else {
                ResourceState
            },
        ));
    };
    Some(OperationHome::Stream(match (op.rail, opcode) {
        // Encoder-scoped fences. Each encoder numbers its own, and the three
        // pairs are nowhere near each other — matching one rail's pair on
        // another's records would read a blit fence as a render one.
        (Rail::Render, 0x18 | 0x19) => Fence,
        (Rail::Compute, 0xd4 | 0xd5) => Fence,
        (Rail::Blit, 0x13c | 0x13d) => Fence,

        (Rail::Render, 0x16 | 0x17 | 0x85) => Barrier,
        (Rail::Compute, 0xd6 | 0xd7) => Barrier,

        // Indirect command buffers: executing one, and mutating one.
        (Rail::Render, 0x14 | 0x15) => IndirectCommand,
        (Rail::Compute, 0xe4 | 0xe5) => IndirectCommand,
        (Rail::Blit, 0x131 | 0x138 | 0x139) => IndirectCommand,

        // Residency, and the operations that speak about where content is
        // rather than what it is.
        (Rail::Render | Rail::Compute, 0x86 | 0x87) => ResourceState,
        (Rail::Render, 0x1b | 0x89) => ResourceState,
        (Rail::Compute, 0xe3) => ResourceState,
        (Rail::Blit, 0x134..=0x137 | 0x13a | 0x13b | 0x142 | 0x143) => ResourceState,

        (Rail::Render, _) => Render,
        (Rail::Compute, _) => Compute,
        (Rail::Blit, _) => Blit,
        (Rail::Info, _) => InfoQuery,
        (Rail::Event, _) => Event,
        // The root rail returned above; it has no stream operations.
        (Rail::Root, _) => return None,
    }))
}

/// How many judged operations sit in each class, plus the two homes.
///
/// Returned rather than printed so a test can assert on it and a report can
/// render it, and ordered by [`OperationClass::ALL`] so a reader can diff two
/// runs line by line.
#[must_use]
pub fn census() -> Census {
    let mut c = Census::default();
    for op in LEDGER {
        match classify(op) {
            None => c.unclassified += 1,
            Some(OperationHome::ObjectLifecycle) => c.object_lifecycle += 1,
            Some(OperationHome::Stream(class)) => {
                c.stream += 1;
                c.per_class[OperationClass::ALL
                    .iter()
                    .position(|k| *k == class)
                    .expect("known")] += 1;
            }
        }
    }
    c
}

/// The reading [`census`] produces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Judged operations that become stream operations.
    pub stream: usize,
    /// Judged operations carried by a resource-lifecycle transaction instead.
    pub object_lifecycle: usize,
    /// Operations the ledger has not judged, which the model refuses.
    pub unclassified: usize,
    /// Stream operations per [`OperationClass::ALL`] position.
    pub per_class: [usize; 11],
}

impl Census {
    pub const fn total(&self) -> usize {
        self.stream + self.object_lifecycle + self.unclassified
    }

    /// The count for one class.
    #[must_use]
    pub fn of(&self, class: OperationClass) -> usize {
        self.per_class[OperationClass::ALL
            .iter()
            .position(|k| *k == class)
            .expect("known")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim the module exists to make: every operation the ledger has
    /// judged has a home, and every one it has not has none.
    #[test]
    fn every_judged_operation_has_exactly_one_home() {
        for op in LEDGER {
            let home = classify(op);
            if op.closure.blocks_cutover() {
                assert_eq!(
                    home, None,
                    "{:?} {:#x?} has no established contract and must not enter the vocabulary",
                    op.rail, op.opcode
                );
            } else {
                assert!(
                    home.is_some(),
                    "{:?} {:#x?} is judged {} and reaches no operation class",
                    op.rail,
                    op.opcode,
                    op.closure.name()
                );
            }
        }
        let c = census();
        assert_eq!(c.total(), LEDGER.len());
    }

    /// The root rail is the whole of the non-stream side, and none of the
    /// stream side.
    #[test]
    fn object_lifecycle_records_are_exactly_the_root_rails() {
        for op in LEDGER {
            match classify(op) {
                Some(OperationHome::ObjectLifecycle) => assert_eq!(op.rail, Rail::Root),
                Some(OperationHome::Stream(_)) => assert_ne!(op.rail, Rail::Root),
                None => {}
            }
        }
    }

    /// Each encoder numbers its own fences, and the three pairs are far apart.
    /// A rail matching another's pair would read one encoder's synchronisation
    /// as a different encoder's — which is the exact shape of mistake the wire
    /// crate's per-module opcode constants exist to prevent.
    #[test]
    fn each_rails_fence_pair_is_classified_on_its_own_rail_alone() {
        let fences: Vec<_> = LEDGER
            .iter()
            .filter(|o| classify(o) == Some(OperationHome::Stream(OperationClass::Fence)))
            .map(|o| (o.rail, o.opcode.expect("fences carry opcodes")))
            .collect();
        assert_eq!(
            fences,
            vec![
                (Rail::Render, 0x18),
                (Rail::Render, 0x19),
                (Rail::Blit, 0x13c),
                (Rail::Blit, 0x13d),
            ],
            "the compute pair is unresolved, so it is absent rather than \
             classified — and the render and blit pairs must not have adopted \
             each other's numbers"
        );
    }

    /// The event rail's operations are events, and nothing else is.
    #[test]
    fn the_event_class_is_the_event_rail() {
        for op in LEDGER {
            let is_event = classify(op) == Some(OperationHome::Stream(OperationClass::Event));
            assert_eq!(
                is_event,
                op.rail == Rail::Event && !op.closure.blocks_cutover()
            );
        }
    }

    /// Every class either has a payload module or is empty, and which of the
    /// two is a checked fact rather than a reading of the source tree.
    ///
    /// This is the vocabulary's completeness claim, and the emptiness half is
    /// the load-bearing one. `InfoQuery` is empty because every info opcode is
    /// unresolved — eighteen questions with no established contract — and
    /// `CompletionEffect` is empty because every completion-shaped record on
    /// this wire *arms* an effect rather than being one. Neither is an omission,
    /// and neither may become one silently: if a row closes, the count moves off
    /// zero and this test asks for the payload.
    ///
    /// The populated classes are each checked against their own module's
    /// vocabulary test — this only says they are populated, so that a class
    /// quietly emptying out is also a failure.
    #[test]
    fn every_class_has_a_payload_or_a_reason_to_be_empty() {
        let c = census();
        let empty_by_contract = [OperationClass::InfoQuery, OperationClass::CompletionEffect];
        for &class in OperationClass::ALL {
            let count = c.of(class);
            if empty_by_contract.contains(&class) {
                assert_eq!(
                    count,
                    0,
                    "{} has judged operations and now needs a payload module",
                    class.name()
                );
            } else {
                assert!(
                    count > 0,
                    "{} has emptied out; its payload module is describing nothing",
                    class.name()
                );
            }
        }
        assert_eq!(
            c.stream,
            OperationClass::ALL.iter().map(|&k| c.of(k)).sum::<usize>()
        );
    }

    /// Printed rather than pinned: these numbers move every time a row is
    /// closed, and a test that had to be edited for that would be edited
    /// without being read.
    #[test]
    fn report_the_vocabulary() {
        let c = census();
        println!(
            "operation vocabulary: {} stream, {} object-lifecycle, {} refused at ingress",
            c.stream, c.object_lifecycle, c.unclassified
        );
        for &class in OperationClass::ALL {
            println!("  {:18} {}", class.name(), c.of(class));
        }
    }
}
