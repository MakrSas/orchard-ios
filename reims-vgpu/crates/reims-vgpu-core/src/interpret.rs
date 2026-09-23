//! The serial reference interpreter: what the guest is owed, executed one
//! transaction at a time.
//!
//! # Why a second executor, and why a slow one
//!
//! The replacement's whole risk is that a parallel schedule and a serial one
//! stop meaning the same thing. That is only checkable against something that
//! *defines* the serial meaning, so this is it: a backend-independent
//! interpreter that runs transactions in ingress order, one at a time, with no
//! concurrency to be wrong about.
//!
//! It is not a fallback path and it will never execute a guest's frame. What it
//! produces is a **trace** — an ordered list of the things a guest could
//! observe — and the seam that introduces parallel scheduling has to produce the
//! same trace or explain why not.
//!
//! # What counts as observable
//!
//! [`Observation`] is deliberately short. A guest cannot see how many host
//! submissions happened, which queue ran the work, or whether a transfer was a
//! copy or an import. It can see a completion stamp reach a value, a content
//! version become current, an event advance, and a refusal on the failure
//! channel. Anything not on that list is an implementation detail, and putting
//! it on the list would make the equivalence test fail for changes that are not
//! guest-visible.
//!
//! # Publication order is the one rule the trace enforces
//!
//! A transaction's content versions become visible when its work completes.
//! Its completion stamp becomes visible later, when ordered guest publication
//! releases it — see [`crate::publish`] — and never earlier. So the two halves
//! are [`Interpreter::complete`], which applies the work and hands back the
//! stamp the transaction now *owes*, and [`Interpreter::publish`], which pays
//! it. [`Interpreter::run`] is the two back to back, which is what a schedule
//! of one transaction at a time makes them.
//!
//! Keeping them apart is the point: a guest that polled the stamp and then read
//! the content must not be able to see the flag without the bytes. An
//! interpreter that wrote the stamp inside `complete` would agree with an
//! implementation that has exactly that bug.

use crate::access::{AccessKey, ContentVersion};
use crate::content::{ContentLedger, Replica};
use crate::exec::{published_versions, Prerequisite, ResolvedOperation, VersionPublication};
use crate::identity::{
    CompletionStamp, IngressOrdinal, ResourceId, SessionGeneration, StampSlot, StampValue,
};
use crate::sync::{EventKind, FenceKind};
use crate::transaction::DeviceTransaction;
use std::collections::HashMap;

/// Something a guest could observe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    /// A region of a backing's content reached a version.
    ///
    /// The region is part of the observation, not decoration. A guest that
    /// wrote two disjoint ranges of one buffer observes two independent
    /// histories, and folding them into one per-backing history would make two
    /// legal orders read as a disagreement.
    VersionPublished {
        backing: crate::access::BackingId,
        region: AccessKey,
        version: ContentVersion,
    },
    /// A completion stamp reached a value.
    StampPublished { slot: StampSlot, value: StampValue },
    /// An event's monotonic generation advanced.
    EventAdvanced { event: ResourceId, to: u64 },
    /// A fence was updated by the encoder that owns it.
    FenceUpdated { fence: ResourceId },
    /// A published version reached memory something newer already held, so
    /// none of it — or only part of it — became current.
    ///
    /// A completion, not a refusal: the transaction ran and its work happened.
    /// What did not happen is the bytes becoming readable, because a newer
    /// write owns them. That is a lawful outcome of two writers racing and it
    /// is exactly the outcome that must not be silent — see
    /// [`crate::coverage`]. `landed` is what did become current, in bytes.
    VersionBeaten {
        backing: crate::access::BackingId,
        region: AccessKey,
        version: ContentVersion,
        landed: u64,
    },
    /// A transaction could not run, with the reason.
    Refused {
        ingress: IngressOrdinal,
        reason: Refusal,
    },
    /// A frame was handed to the display.
    ///
    /// **The one observation a guest does not read back**, and it is here
    /// because leaving it out makes two schedules that show *different frames*
    /// compare equal. A present's accesses are reads, reads publish no version,
    /// and a present writes no completion word beyond the envelope's — so
    /// without this the trace of a run that showed mapping A is byte-identical
    /// to one that showed mapping B, and showing the wrong frame is the one
    /// failure this device exists to avoid.
    ///
    /// Ordered per domain rather than globally: two channels' presents have no
    /// ordering obligation to each other, and comparing one interleaved
    /// sequence would report a divergence where the contract allows both.
    ///
    /// Recorded even for [`crate::identity::MappingId`] zero, which the guest
    /// sends to mean *nothing to show*: a run that showed nothing and a run
    /// that showed a frame are not the same run.
    FramePresented {
        domain: crate::identity::ChannelId,
        mapping: crate::identity::MappingId,
    },
    /// A transaction ran and its lifetime operation did not happen.
    ///
    /// **Not [`Self::Refused`], and the difference is the completion word.** A
    /// refused transaction never runs and owes nothing; a declined operation
    /// belongs to a transaction that ran and whose stamp is still owed in full
    /// — the guest is waiting on it, and a lifetime event this device chose not
    /// to act on is still an event the guest performed. Folding the two would
    /// make the reference publish no stamp where the device publishes one,
    /// which is a hang rather than a disagreement.
    ///
    /// Observable because it lands on the always-on failure channel, which is
    /// the same reason [`Self::Refused`] is.
    OperationDeclined {
        ingress: IngressOrdinal,
        reason: crate::lifecycle::Refusal,
    },
    /// A query ran and no answer was written to its destination.
    ///
    /// **The guest reads that destination either way**, which is what makes
    /// this observable and what makes it different from every other
    /// not-happening in this list. An unanswered query is not lost work the
    /// guest can retry: it takes whatever the window already held and proceeds
    /// on it, so a run that answered and a run that did not are two different
    /// runs even though both published the same completion word.
    ///
    /// The stamp is still owed in full — see [`crate::query::PendingQuery`] —
    /// because a query that neither answers nor completes is a hang, which is
    /// strictly worse than a wrong value the guest can be told about.
    QueryUnanswered {
        ingress: IngressOrdinal,
        kind: crate::query::QueryKind,
        reason: crate::query::Stall,
    },
    /// A discard the guest asked for freed nothing, because the copy it named
    /// was the only one holding those bytes.
    ///
    /// The safe branch, and the reason it is nonetheless observed: it is guest
    /// work the device declined to carry out, and this crate's rule is that
    /// such work gets a typed reason rather than a silent return. Before this
    /// [`crate::lifecycle::Lifecycle::complete`]'s answer was dropped at the
    /// call site with no comment, so a device that never freed anything the
    /// guest discarded read exactly like one that freed all of it.
    ///
    /// **Deliberately not part of the equivalence outcome.** Whether a spare
    /// copy existed at the moment a discard completed is a function of which
    /// transfers had finished, so two schedules the dependency graph permits
    /// may legitimately decline different discards — and neither is wrong,
    /// because [`crate::lifecycle::Lifecycle::complete`] drops bytes only
    /// where another replica already holds them. The content a later read
    /// returns is identical either way. Requiring the two to match would be
    /// requiring the device to serialise work the guest left independent,
    /// which is the strict-direction error [`crate::schedule`] warns about.
    DiscardDeclined {
        ingress: IngressOrdinal,
        resource: ResourceId,
        backing: crate::access::BackingId,
        /// Bytes that exist in no other replica, so dropping them would have
        /// destroyed content rather than freed a copy.
        sole_authority_bytes: u64,
    },
}

/// Why the interpreter would not run a transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A prerequisite is not satisfied and, in a serial schedule, never will
    /// be: everything that could satisfy it has already run.
    ///
    /// This is the serial interpreter's version of the diagnosis
    /// [`crate::ready::Scheduler::stalled`] makes for a concurrent one, and it
    /// is a *stronger* statement here — running one at a time means nothing is
    /// outstanding, so an unmet wait is unmeetable.
    UnmeetableWait,
    /// The transaction belongs to a generation that has closed.
    StaleGeneration,
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnmeetableWait => "interpret_unmeetable_wait",
            Self::StaleGeneration => "interpret_stale_generation",
        }
    }
}

/// What running one transaction did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ran,
    Refused(Refusal),
}

/// The serial reference interpreter.
///
/// Holds the semantic state a guest can observe and nothing else. No queue, no
/// submission, no host object — the point is that anything it cannot represent
/// is, by construction, not guest-visible.
#[derive(Debug)]
pub struct Interpreter {
    /// The semantic lifetime work must belong to.
    ///
    /// A reset opens a new one. Work from a closed generation is refused rather
    /// than run: it names objects that no longer exist, and running it would be
    /// executing against whatever now occupies their slots.
    generation: SessionGeneration,
    /// The semantic model the transactions act on.
    ///
    /// It owns the session's content authority, so the versions a completed
    /// transaction publishes and the transfers a lifetime operation owes land
    /// in one ledger. A second ledger beside it would be two answers to where
    /// the current bytes are.
    model: crate::lifecycle::Lifecycle,
    /// How long this device's answer to each question is.
    ///
    /// `None` is a device that answers nothing, and that is the default rather
    /// than an oversight: this crate cannot see a host, so an interpreter
    /// nobody handed an answer source to must stall every query rather than
    /// invent a reply length and publish a version for bytes that were never
    /// written.
    answers: Option<Box<dyn crate::query::AnswerLength>>,
    events: HashMap<ResourceId, u64>,
    fences: HashMap<ResourceId, u64>,
    stamps: HashMap<StampSlot, StampValue>,
    trace: Vec<Observation>,
    ran: usize,
    refused: usize,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self {
            generation: SessionGeneration::FIRST,
            model: crate::lifecycle::Lifecycle::new(),
            answers: None,
            events: HashMap::new(),
            fences: HashMap::new(),
            stamps: HashMap::new(),
            trace: Vec::new(),
            ran: 0,
            refused: 0,
        }
    }
}

impl Interpreter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The generation work must belong to.
    #[must_use]
    pub const fn generation(&self) -> SessionGeneration {
        self.generation
    }

    /// Open the next generation, as a reset does.
    ///
    /// Everything the previous generation published stays published — a stamp a
    /// guest already read does not un-read — and everything it had outstanding
    /// is simply never run, because a serial interpreter has nothing
    /// outstanding. What changes is which work is admissible from here on.
    pub fn reset(&mut self) {
        self.generation = self.generation.next();
    }

    /// The observation trace, in order.
    #[must_use]
    pub fn trace(&self) -> &[Observation] {
        &self.trace
    }

    /// The content ledger, for a caller declaring backings before a run.
    pub const fn content_mut(&mut self) -> &mut ContentLedger {
        self.model.content_mut()
    }

    /// The model the transactions act on, for a caller setting up the tasks and
    /// heaps a run needs.
    pub const fn model_mut(&mut self) -> &mut crate::lifecycle::Lifecycle {
        &mut self.model
    }

    /// Install the source of answer lengths this run's queries are judged
    /// against.
    ///
    /// Without one every query stalls, which is the honest default for a crate
    /// that cannot see a host. A harness comparing this reference against a
    /// real schedule installs the same source both sides answer from — the
    /// point of the comparison is the ordering, and an interpreter with a
    /// different idea of how long an answer is would report a divergence that
    /// is its own.
    pub fn answers_from(&mut self, answers: Box<dyn crate::query::AnswerLength>) {
        self.answers = Some(answers);
    }

    #[must_use]
    pub fn event_generation(&self, event: ResourceId) -> u64 {
        self.events.get(&event).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn stamp(&self, slot: StampSlot) -> Option<StampValue> {
        self.stamps.get(&slot).copied()
    }

    /// How many transactions ran and how many were refused.
    #[must_use]
    pub const fn census(&self) -> (usize, usize) {
        (self.ran, self.refused)
    }

    /// Run one transaction to completion and publish its stamp at once.
    ///
    /// Serial in the strongest sense: when this returns, the transaction's work
    /// is done and everything it publishes is visible. There is no pending
    /// state for a later call to discover.
    ///
    /// That is only the serial case. A device publishes in channel order, so
    /// the two halves are [`Self::complete`] and [`Self::publish`]; this is
    /// them back to back, which is what a schedule of one transaction at a
    /// time makes them.
    pub fn run(&mut self, tx: &DeviceTransaction) -> Outcome {
        match self.complete(tx) {
            Ok(owed) => {
                if let Some(stamp) = owed {
                    self.publish(stamp);
                }
                Outcome::Ran
            }
            Err(refusal) => Outcome::Refused(refusal),
        }
    }

    /// Apply a transaction's work and publish everything that becomes visible
    /// when it completes: its records' effects and its content versions.
    ///
    /// Returns the completion stamp it now **owes**. A stamp is not visible at
    /// completion — see [`crate::publish`] — so handing it back rather than
    /// writing it is what keeps the two events apart in the reference as well
    /// as in the device.
    ///
    /// # Errors
    ///
    /// The refusal, which is also appended to the trace: a guest that is
    /// refused observes the refusal.
    pub fn complete(&mut self, tx: &DeviceTransaction) -> Result<Option<CompletionStamp>, Refusal> {
        if tx.identity.session != self.generation {
            self.refuse(tx.identity.ingress, Refusal::StaleGeneration);
            return Err(Refusal::StaleGeneration);
        }
        if let Some(refusal) = self.unmet_prerequisite(tx) {
            self.refuse(tx.identity.ingress, refusal);
            return Err(refusal);
        }

        // The records run in order, and the ones with observable state are the
        // synchronisation records. Everything else contributes through the
        // transaction's accesses, which is where its memory effect lives.
        //
        // Only an EXEC has records. The four other classes reach the trace
        // through their accesses and their completion word alone, which is
        // exactly what a guest can observe of them.
        if let Some(exec) = tx.exec() {
            for record in exec.records() {
                self.apply_record(&record.op);
            }
        }

        // What the guest asked to show becomes visible when the transaction's
        // work completes, like the content versions below and unlike the
        // completion word — a guest that polls the word and then reads the
        // surface must not see the flag without the frame.
        if let crate::transaction::Payload::Present(present) = &tx.payload {
            self.trace.push(Observation::FramePresented {
                domain: tx.identity.domain,
                mapping: present.packet().mapping,
            });
        }

        // A lifetime operation acts on the model, and a model that declines it
        // does not stop the transaction: the guest is waiting on the completion
        // word and a lifetime event this device chose not to act on is still an
        // event the guest performed. So the decline is observed and the run
        // continues to its publication.
        if let crate::transaction::Payload::ResourceLifecycle(lifecycle) = &tx.payload {
            match self.model.apply(lifecycle.op()) {
                Ok(effects) => {
                    // Serial in the strongest sense: the transfers this
                    // operation owes have executed by the time its completion
                    // word could be read, so the content-authority changes it
                    // deferred to that point are taken here.
                    //
                    // The discards it offered and could not take are observed
                    // rather than dropped. See
                    // [`Observation::DiscardDeclined`]: the guest asked for
                    // something the device did not do, which is a typed reason
                    // whether or not it changes a later read.
                    for declined in self.model.complete(&effects) {
                        self.trace.push(Observation::DiscardDeclined {
                            ingress: tx.identity.ingress,
                            resource: declined.resource,
                            backing: declined.backing,
                            sole_authority_bytes: declined.sole_authority_bytes,
                        });
                    }
                }
                Err(reason) => {
                    self.trace.push(Observation::OperationDeclined {
                        ingress: tx.identity.ingress,
                        reason,
                    });
                    self.ran += 1;
                    // The word, and nothing else --- the same shape the
                    // stalled query below has, and for the same reason.
                    //
                    // A lifecycle payload's accesses are exactly its
                    // operation's own resource list, which
                    // [`crate::transaction::LifecyclePayload::new`] enforces in
                    // both directions, and `Lifecycle::apply` is all-or-nothing.
                    // So a declined operation moved *no* bytes for *any* of
                    // them. Falling through to the publication below would
                    // certify, in the trace an implementation is checked
                    // against, that content nothing wrote had become current
                    // --- and a later read planned against that version is one
                    // that skips the transfer which would have made it true.
                    //
                    // The stamp is still owed in full, for the reason
                    // [`Observation::OperationDeclined`] gives: the guest is
                    // blocked on the word whether or not the device acted.
                    return Ok(tx.completion);
                }
            }
        }

        // A query's only write is its answer, so whether the answer happened
        // decides whether anything was written at all. Running it through
        // `PendingQuery` is what makes that structural: there is no path from
        // an admitted query to a published version that does not go through
        // either `answer` or `unanswerable`, and the second one publishes no
        // version while still handing back the stamp the guest is blocked on.
        //
        // Without this the reference published the destination's version for
        // every query unconditionally — certifying, in the trace an
        // implementation is checked against, that a reply nothing produced had
        // landed.
        if let crate::transaction::Payload::Query(query) = &tx.payload {
            let request = *query.request();
            let pending = crate::query::PendingQuery::new(request, tx.completion);
            let completed = match self.answers.as_ref().and_then(|a| a.bytes(&request)) {
                Some(len) => match pending.answer(len) {
                    Ok(done) => done,
                    // A reply that does not fit is not clamped: a partial one is
                    // a reply, and the guest cannot tell it from a whole one. It
                    // completes with the stall, like an answer that never was.
                    Err((pending, stall)) => pending.unanswerable(stall),
                },
                None => pending.unanswerable(crate::query::Stall::NoAnswer),
            };
            match completed.answer() {
                crate::query::Answer::None(reason) => {
                    self.trace.push(Observation::QueryUnanswered {
                        ingress: tx.identity.ingress,
                        kind: completed.kind(),
                        reason,
                    });
                    self.ran += 1;
                    // The stamp, and nothing else. This is the whole of the
                    // difference between a stalled query and an answered one.
                    return Ok(completed.publication());
                }
                // **The answer's own window, not the destination's.** The guest
                // hands over whatever buffer it had — a page is ordinary — and
                // the reply is sixteen bytes or a short run of pairs.
                // Publishing the access's whole range would tell the content
                // authority that everything past the reply is current device
                // content, so a later read of those bytes would skip the
                // transfer that would have made that true. `ReplyWrite` names
                // the window for exactly this reason.
                crate::query::Answer::Written(reply) => {
                    if let Some(to) = query.access().output_content_version {
                        self.publish_version(&VersionPublication {
                            backing: reply.backing,
                            region: AccessKey::Range(
                                crate::access::ResourceKey {
                                    backing: reply.backing,
                                    // A reply destination is a guest buffer the
                                    // request named, never a heap window.
                                    heap: None,
                                },
                                reply.bytes,
                            ),
                            to,
                        });
                    }
                    self.ran += 1;
                    return Ok(completed.publication());
                }
            }
        }

        // The bytes land, and then the version that says they are current
        // becomes visible. Both come from the same list — the accesses — so a
        // version cannot be published for memory nothing wrote.
        for published in published_versions(tx.accesses()) {
            self.publish_version(&published);
        }
        self.ran += 1;
        Ok(tx.completion)
    }

    /// Make one version current, and observe what that did.
    ///
    /// One body for the access-derived publications and for a query's reply
    /// window, because the rule about *what happened* is the same for both: a
    /// version may be beaten by a newer write, and a beaten one is a completion
    /// rather than a refusal.
    fn publish_version(&mut self, published: &VersionPublication) {
        // The version the access reserved, not one this ledger mints: the
        // reservation happened when the transaction was planned, and a
        // completion that took a fresh number here would beat every writer
        // that reserved after it and lose to none.
        let beaten = written_bytes(self.model.content(), published.region).map(|bytes| {
            self.model.content_mut().materialize(
                published.backing,
                bytes,
                published.to,
                Replica::DeviceOwned,
            )
        });
        match beaten {
            Some(applied) if applied.was_partly_stale() => {
                self.trace.push(Observation::VersionBeaten {
                    backing: published.backing,
                    region: published.region,
                    version: published.to,
                    landed: applied.taken.len(),
                });
                if applied.is_empty() {
                    // Nothing became current, so nothing was published.
                    return;
                }
            }
            // A subresource or whole-backing write names no bytes this crate
            // can place — see `written_bytes` — so its effect is the version
            // alone and there is nothing for a newer write to have beaten it
            // over.
            _ => {}
        }
        self.trace.push(Observation::VersionPublished {
            backing: published.backing,
            region: published.region,
            version: published.to,
        });
    }

    /// Make a completion word readable, as ordered guest publication releases
    /// it.
    ///
    /// Later in the wrapping order, and only observed when it advances. A
    /// completion word is a monotone point the guest polls: a value that does
    /// not advance it writes nothing the guest can read, and a plain overwrite
    /// would let the slot go backwards. This is the same rule
    /// [`crate::ready::Scheduler::publish`] applies and the same rule a signal
    /// that does not advance an event gets above — stating it in one of the
    /// three places and not the others is how the reference and the scheduler
    /// come to mean different things.
    pub fn publish(&mut self, stamp: CompletionStamp) {
        let standing = self.stamps.get(&stamp.slot).copied();
        if standing.is_none_or(|at| stamp.value.follows(at)) {
            self.stamps.insert(stamp.slot, stamp.value);
            self.trace.push(Observation::StampPublished {
                slot: stamp.slot,
                value: stamp.value,
            });
        }
    }

    fn refuse(&mut self, ingress: IngressOrdinal, reason: Refusal) {
        self.refused += 1;
        self.trace.push(Observation::Refused { ingress, reason });
    }

    /// The first prerequisite this transaction cannot meet, if any.
    fn unmet_prerequisite(&self, tx: &DeviceTransaction) -> Option<Refusal> {
        // The envelope's waits, which every class of packet carries.
        for wait in &tx.stamp_waits {
            let met = self
                .stamps
                .get(&wait.slot)
                .is_some_and(|published| !wait.value.follows(*published));
            if !met {
                return Some(Refusal::UnmeetableWait);
            }
        }
        let exec = tx.exec()?;
        for prerequisite in exec.prerequisites() {
            let met = match *prerequisite {
                Prerequisite::Event { event, value } => self.event_generation(event) >= value,
                // A fence is encoder-scoped and its producer is inside this
                // packet or an earlier one; a serial run has already finished
                // every earlier one, so an outstanding fence is one this packet
                // updates itself.
                Prerequisite::Fence { fence } => self.fences.contains_key(&fence),
            };
            if !met {
                return Some(Refusal::UnmeetableWait);
            }
        }
        None
    }

    fn apply_record(&mut self, op: &ResolvedOperation) {
        match op {
            ResolvedOperation::Event(event) => match event.kind {
                EventKind::Signal => {
                    let current = self.event_generation(event.event);
                    if event.advances(current) {
                        self.events.insert(event.event, event.value);
                        self.trace.push(Observation::EventAdvanced {
                            event: event.event,
                            to: event.value,
                        });
                    }
                }
                // A wait inside a serial run is satisfied or the transaction
                // would not have started; the prerequisite check is where an
                // unmeetable one is caught.
                EventKind::Wait => {}
            },
            ResolvedOperation::Fence(fence) => match fence.kind {
                FenceKind::Update => {
                    *self.fences.entry(fence.fence).or_insert(0) += 1;
                    self.trace
                        .push(Observation::FenceUpdated { fence: fence.fence });
                }
                FenceKind::Wait => {}
            },
            // Everything else contributes through the transaction's accesses.
            // A barrier orders nothing in a schedule that is already serial, a
            // content directive is answered by whatever placement an executor
            // chose, and a draw's memory effect is its declared participation.
            ResolvedOperation::Render(_)
            | ResolvedOperation::Compute(_)
            | ResolvedOperation::Blit(_)
            | ResolvedOperation::Barrier(_)
            | ResolvedOperation::ResourceState(_)
            | ResolvedOperation::IndirectCommand(_) => {}
        }
    }
}

/// The byte range a write access covers, when it names one.
///
/// Three answers, and the two `None`s are not the same fact.
///
/// A `Range` names its bytes. `Whole` names the backing, and the backing's
/// bytes are the extent its declaration gave it — so this asks the ledger
/// rather than returning nothing. It used to return nothing, and a
/// whole-backing write therefore published a version over no bytes: nothing
/// was covered, so a later *older* write was not beaten by it, and the replica
/// that produced the content did not become fresh for it — which makes the
/// next read from that replica owe a transfer that copies stale bytes over
/// what the device just wrote.
///
/// A subresource write is the genuine `None`: it names image coordinates
/// rather than bytes, and relating the two needs the image's layout, which is
/// an executor's and not this crate's. Its effect is the version alone. So is
/// a heap declaration's and an unparticipating access's, neither of which
/// names memory at all.
fn written_bytes(
    content: &crate::content::ContentLedger,
    key: AccessKey,
) -> Option<crate::access::ByteRange> {
    match key {
        AccessKey::Range(_, range) => Some(range),
        AccessKey::Whole(resource) => content.extent(resource.backing),
        AccessKey::Subresource(..) | AccessKey::Heap(_) | AccessKey::DomainOnly => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{
        AccessIntent, AccessMode, BackingId, ByteRange, ResourceKey, StubRegistry,
    };
    use crate::identity::{ChannelId, CompletionStamp, ObjectListRef, SlotGeneration, StampWait};
    use crate::stream::{SegmentKind, SegmentLifetime};
    use crate::sync::EventOp;

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    /// Define task 1 and put `res(1)` in it on `backing`, so a lifetime
    /// operation naming that resource actually applies.
    ///
    /// The interpreter runs lifetime operations against the model now, so a
    /// synchronise of a resource nobody declared is a *declined* operation —
    /// correctly, and not what the tests below are about.
    fn declare_resource(interp: &mut Interpreter, backing: BackingId, length: u64) {
        let model = interp.model_mut();
        // A definition and a declaration owe nothing: no transfer, no teardown.
        let _ = model
            .apply(&crate::lifecycle::LifecycleOp::DefineTask {
                task: crate::identity::TaskId(1),
                kernel: false,
                directory: crate::identity::DirectoryFrame(0x1000),
            })
            .expect("a fresh task");
        let _ = model
            .apply(&crate::lifecycle::LifecycleOp::CreateResource {
                task: crate::identity::TaskId(1),
                slot: ObjectListRef(1),
                storage: crate::lifecycle::Storage::Dedicated {
                    backing,
                    extent: ByteRange { offset: 0, length },
                },
            })
            .expect("a free slot");
    }

    fn builder(ingress: u64) -> crate::testing::At<'static> {
        crate::testing::At::new(1, ingress)
    }

    /// A packet that is not GPU work: a control command, with only the
    /// envelope every transaction has.
    fn control(ingress: u64) -> DeviceTransaction {
        DeviceTransaction {
            identity: crate::testing::identity(1, ingress),
            stamp_waits: Vec::new(),
            completion: None,
            payload: crate::transaction::Payload::Control(crate::control::ControlOp::Inert {
                kind: crate::control::ControlKind::Nop,
            }),
        }
    }

    /// A present of `mapping`: one read of the surface it shows.
    fn present(ingress: u64, mapping: u32) -> DeviceTransaction {
        let packet = crate::present::PresentPacket {
            form: crate::present::PresentForm::SwapMapping,
            mapping: crate::identity::MappingId(mapping),
            task: None,
        };
        let read = AccessIntent {
            domain: ChannelId(1),
            key: AccessKey::Whole(ResourceKey {
                backing: BackingId(u64::from(mapping)),
                heap: None,
            }),
            mode: AccessMode::Read,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        };
        DeviceTransaction {
            identity: crate::testing::identity(1, ingress),
            stamp_waits: Vec::new(),
            completion: None,
            payload: crate::transaction::Payload::Present(
                crate::transaction::PresentPayload::new(packet, vec![(packet.mapping, read)])
                    .expect("one read of the packet's own target"),
            ),
        }
    }

    /// Two runs that show different frames must not compare equal.
    ///
    /// Everything else about them is identical: a present's accesses are reads,
    /// reads publish no version, and neither carries a completion word. Without
    /// [`Observation::FramePresented`] the two traces are byte-identical, and
    /// showing the wrong frame is the one failure this device exists to avoid.
    #[test]
    fn a_run_that_showed_a_different_frame_is_a_different_run() {
        let trace_of = |mapping: u32| {
            let mut interp = Interpreter::new();
            assert_eq!(interp.run(&present(1, mapping)), Outcome::Ran);
            interp.trace().to_vec()
        };
        let seven = trace_of(7);
        assert_eq!(
            seven,
            vec![Observation::FramePresented {
                domain: crate::testing::identity(1, 1).domain,
                mapping: crate::identity::MappingId(7),
            }]
        );
        assert_ne!(seven, trace_of(8));
        assert_ne!(
            seven,
            trace_of(0),
            "nothing to show is not the same as showing something"
        );
    }

    /// A device that answers every question with the same number of bytes.
    ///
    /// A length and nothing else, because that is the whole of what this crate
    /// can be told: the values are the host's and the trace records that a
    /// window reached a version, not what is in it.
    #[derive(Debug)]
    struct AnswersWith(u64);

    impl crate::query::AnswerLength for AnswersWith {
        fn bytes(&self, _request: &crate::query::QueryRequest) -> Option<u64> {
            Some(self.0)
        }
    }

    /// A compute-info query writing its reply into `backing`, versioned so the
    /// answer is something the trace can show landing.
    ///
    /// Key-value rather than the heap-texture question, because these tests are
    /// about the *length* of an answer against the window it goes into, and a
    /// fixed record has exactly one lawful length — see
    /// [`crate::query::Stall::FixedReplyWrongSize`]. The bounds are generous so
    /// the request's own limit is never the one being tested.
    fn query(ingress: u64, backing: BackingId, window: u64) -> DeviceTransaction {
        let request = crate::query::QueryRequest {
            kind: crate::query::QueryKind::ComputeInfo,
            destination: crate::query::ReplyDestination {
                backing,
                bytes: ByteRange {
                    offset: 0,
                    length: window,
                },
            },
            reply: crate::query::ReplyShape::KeyValue(
                reims_vgpu_protocol::info_reply::ReplyBounds {
                    key_table_len: 64,
                    count: 4096,
                },
            ),
        };
        DeviceTransaction {
            identity: crate::testing::identity(1, ingress),
            stamp_waits: Vec::new(),
            completion: Some(CompletionStamp {
                slot: StampSlot(1),
                value: StampValue(ingress as u32),
            }),
            payload: crate::transaction::Payload::Query(crate::transaction::QueryPayload::new(
                request,
                ChannelId(1),
                Some(ContentVersion(1)),
            )),
        }
    }

    fn ran_query(answers: Option<u64>, window: u64) -> Vec<Observation> {
        let mut interp = Interpreter::new();
        interp.content_mut().declare(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 4096,
            },
            Replica::GuestPages,
        );
        if let Some(len) = answers {
            interp.answers_from(Box::new(AnswersWith(len)));
        }
        assert_eq!(interp.run(&query(1, BackingId(9), window)), Outcome::Ran);
        interp.trace().to_vec()
    }

    /// An interpreter with no answer source publishes the stamp and no version.
    ///
    /// Both halves matter and they are the query contract's two obligations.
    /// The stamp, because a guest blocked on a completion word it never
    /// receives is a hang. No version, because the reference would otherwise
    /// certify that a reply nothing produced had landed — and an implementation
    /// checked against that trace would pass while writing nothing.
    #[test]
    fn a_query_with_no_answer_publishes_its_stamp_and_no_content() {
        assert_eq!(
            ran_query(None, 4096),
            vec![
                Observation::QueryUnanswered {
                    ingress: IngressOrdinal(1),
                    kind: crate::query::QueryKind::ComputeInfo,
                    reason: crate::query::Stall::NoAnswer,
                },
                Observation::StampPublished {
                    slot: StampSlot(1),
                    value: StampValue(1),
                },
            ]
        );
    }

    /// An answer publishes **the window it occupies**, then the stamp.
    ///
    /// The order first: a guest that polls the word and then reads the reply
    /// must not see the flag without the bytes.
    ///
    /// The window second, and it is the destination's *prefix*, not the
    /// destination. The guest hands over whatever buffer it had — a page here —
    /// and the reply is sixteen bytes. Publishing the access's whole range
    /// would tell the content authority that the 4080 bytes after the reply are
    /// current device content, so a later read of them would skip the transfer
    /// that would have made that true.
    #[test]
    fn an_answered_query_publishes_the_window_it_wrote_and_not_the_one_it_was_given() {
        let reply = |length| {
            vec![
                Observation::VersionPublished {
                    backing: BackingId(9),
                    region: AccessKey::Range(
                        ResourceKey {
                            backing: BackingId(9),
                            heap: None,
                        },
                        ByteRange { offset: 0, length },
                    ),
                    version: ContentVersion(1),
                },
                Observation::StampPublished {
                    slot: StampSlot(1),
                    value: StampValue(1),
                },
            ]
        };
        assert_eq!(ran_query(Some(16), 4096), reply(16));
        // And an answer that fills its window publishes all of it: the rule is
        // the answer's length, not "less than the destination".
        assert_eq!(ran_query(Some(4096), 4096), reply(4096));
    }

    /// The bytes past an answer keep the version they had.
    ///
    /// The failure this prevents is silent: a query into a page marks the page
    /// device-fresh, a later read of its tail finds nothing owed, and the guest
    /// is served whatever the device replica happened to hold.
    #[test]
    fn the_bytes_past_a_reply_are_not_made_current_by_it() {
        let mut interp = Interpreter::new();
        interp.content_mut().declare(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 4096,
            },
            Replica::GuestPages,
        );
        interp.answers_from(Box::new(AnswersWith(16)));
        assert_eq!(interp.run(&query(1, BackingId(9), 4096)), Outcome::Ran);

        let content = interp.content_mut();
        assert!(
            content.is_fresh(
                BackingId(9),
                ByteRange {
                    offset: 0,
                    length: 16,
                },
                Replica::DeviceOwned,
            ),
            "the reply is device content"
        );
        assert!(
            !content.is_fresh(
                BackingId(9),
                ByteRange {
                    offset: 16,
                    length: 4080,
                },
                Replica::DeviceOwned,
            ),
            "and nothing wrote the rest of the page"
        );
    }

    /// An answer larger than the window it was given is a stall, not a
    /// truncation — and the run that stalled is not the run that answered.
    ///
    /// A partial reply is a reply and the guest cannot tell it from a whole
    /// one, so the only lawful outcome is to publish the word and name the
    /// reason.
    #[test]
    fn a_reply_too_large_for_its_window_stalls_rather_than_being_clamped() {
        let stalled = ran_query(Some(64), 16);
        assert_eq!(
            stalled[0],
            Observation::QueryUnanswered {
                ingress: IngressOrdinal(1),
                kind: crate::query::QueryKind::ComputeInfo,
                reason: crate::query::Stall::ReplyTooLarge {
                    needed: 64,
                    available: 16,
                },
            }
        );
        assert_ne!(stalled, ran_query(Some(16), 16));
    }

    /// A lifecycle packet that synchronises a resource: no records, and an
    /// access that produces content the guest reads back.
    fn synchronize(ingress: u64, accesses: Vec<AccessIntent>) -> DeviceTransaction {
        DeviceTransaction {
            identity: crate::testing::identity(1, ingress),
            stamp_waits: Vec::new(),
            completion: None,
            payload: crate::transaction::Payload::ResourceLifecycle(
                crate::transaction::LifecyclePayload::new(
                    crate::lifecycle::LifecycleOp::Synchronize {
                        task: crate::identity::TaskId(1),
                        resources: vec![res(1)],
                    },
                    accesses.into_iter().map(|a| (res(1), a)).collect(),
                )
                .expect("every access is for the one resource the op names"),
            ),
        }
    }

    fn signal(event: ResourceId, value: u64) -> ResolvedOperation {
        ResolvedOperation::Event(EventOp {
            kind: EventKind::Signal,
            event,
            value,
        })
    }

    fn write_access(backing: u64, offset: u64, length: u64) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(1),
            key: AccessKey::Range(
                ResourceKey {
                    backing: BackingId(backing),
                    heap: None,
                },
                ByteRange { offset, length },
            ),
            mode: AccessMode::Write,
            api_stages: 0,
            input_content_version: None,
            output_content_version: Some(ContentVersion(1)),
        }
    }

    /// A completion word is a monotone point, so a stamp that does not advance
    /// it is not something a guest polling that word can observe — and a plain
    /// overwrite would let the slot go backwards, which is the failure a guest
    /// waiting on the higher value never wakes from.
    #[test]
    fn a_stamp_that_does_not_advance_its_slot_publishes_nothing() {
        let mut interp = Interpreter::new();
        for value in [9u32, 4, 9, 10] {
            let mut b = builder(u64::from(value));
            b.publish_stamp(CompletionStamp {
                slot: StampSlot(3),
                value: StampValue(value),
            });
            assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        }
        assert_eq!(
            interp.trace(),
            &[
                Observation::StampPublished {
                    slot: StampSlot(3),
                    value: StampValue(9)
                },
                Observation::StampPublished {
                    slot: StampSlot(3),
                    value: StampValue(10)
                },
            ],
            "4 is behind 9 and the second 9 is 9; neither is a new reading"
        );
        assert_eq!(interp.stamp(StampSlot(3)), Some(StampValue(10)));
    }

    /// And the wrapping order is the one that decides, so a timeline that
    /// wraps keeps advancing rather than freezing at `u32::MAX`.
    #[test]
    fn a_wrapped_stamp_still_advances_its_slot() {
        let mut interp = Interpreter::new();
        for value in [u32::MAX - 1, u32::MAX, 0, 1] {
            let mut b = builder(u64::from(value) + 1);
            b.publish_stamp(CompletionStamp {
                slot: StampSlot(3),
                value: StampValue(value),
            });
            assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        }
        assert_eq!(interp.trace().len(), 4, "every step advanced");
        assert_eq!(interp.stamp(StampSlot(3)), Some(StampValue(1)));
    }

    /// The publication order is versions then stamp, and it is the rule the
    /// trace exists to be able to fail on.
    #[test]
    fn a_transaction_publishes_its_versions_before_its_stamp() {
        let mut b = builder(1);
        let access = AccessIntent {
            output_content_version: Some(ContentVersion(2)),
            ..write_access(7, 0, 64)
        };
        b.declare_access(access);
        b.publish_stamp(CompletionStamp {
            slot: StampSlot(3),
            value: StampValue(9),
        });
        let tx = b.finish().expect("frozen");

        let mut interp = Interpreter::new();
        assert_eq!(interp.run(&tx), Outcome::Ran);
        assert_eq!(
            interp.trace(),
            &[
                Observation::VersionPublished {
                    backing: BackingId(7),
                    region: access.key,
                    version: ContentVersion(2)
                },
                Observation::StampPublished {
                    slot: StampSlot(3),
                    value: StampValue(9)
                },
            ]
        );
        assert_eq!(interp.stamp(StampSlot(3)), Some(StampValue(9)));
    }

    /// A wait for a stamp an earlier transaction published is met; one for a
    /// stamp nothing published is unmeetable, because in a serial run there is
    /// nothing outstanding to meet it later.
    #[test]
    fn an_unmet_wait_in_a_serial_run_is_unmeetable_rather_than_pending() {
        let mut producer = builder(1);
        producer.publish_stamp(CompletionStamp {
            slot: StampSlot(1),
            value: StampValue(5),
        });
        let producer = producer.finish().expect("frozen");

        let mut waiter = builder(2);
        waiter.wait_for(StampWait {
            slot: StampSlot(1),
            value: StampValue(5),
        });
        let waiter = waiter.finish().expect("frozen");

        let mut early = Interpreter::new();
        assert_eq!(
            early.run(&waiter),
            Outcome::Refused(Refusal::UnmeetableWait),
            "nothing has published slot 1"
        );

        let mut ordered = Interpreter::new();
        assert_eq!(ordered.run(&producer), Outcome::Ran);
        assert_eq!(ordered.run(&waiter), Outcome::Ran);
        assert_eq!(ordered.census(), (2, 0));
    }

    /// A signal that advances the generation is observable; one that does not
    /// is silent, by the API's own monotonic rule.
    #[test]
    fn only_an_advancing_signal_reaches_the_trace() {
        let mut b = builder(1);
        b.begin_segment(
            SegmentKind::Event.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        b.record(signal(res(4), 5), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        b.record(signal(res(4), 3), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        b.record(signal(res(4), 7), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        b.end_segment().expect("end");
        let tx = b.finish().expect("frozen");

        let mut interp = Interpreter::new();
        interp.run(&tx);
        assert_eq!(
            interp.trace(),
            &[
                Observation::EventAdvanced {
                    event: res(4),
                    to: 5
                },
                Observation::EventAdvanced {
                    event: res(4),
                    to: 7
                },
            ]
        );
        assert_eq!(interp.event_generation(res(4)), 7);
    }

    /// An event wait is met by a value an earlier transaction signalled, and
    /// waiting for a value at or below the generation is met immediately.
    #[test]
    fn an_event_wait_is_met_at_or_past_its_value() {
        let mut producer = builder(1);
        producer
            .begin_segment(
                SegmentKind::Event.wire_type(),
                SegmentLifetime::SELF_CONTAINED,
            )
            .expect("open");
        producer
            .record(signal(res(4), 5), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        producer.end_segment().expect("end");
        let producer = producer.finish().expect("frozen");

        let mut interp = Interpreter::new();
        interp.run(&producer);

        for (value, expected) in [(4u64, Outcome::Ran), (5, Outcome::Ran)] {
            let mut b = builder(2);
            b.require(Prerequisite::Event {
                event: res(4),
                value,
            });
            assert_eq!(interp.run(&b.finish().expect("frozen")), expected);
        }
        let mut b = builder(3);
        b.require(Prerequisite::Event {
            event: res(4),
            value: 6,
        });
        assert_eq!(
            interp.run(&b.finish().expect("frozen")),
            Outcome::Refused(Refusal::UnmeetableWait)
        );
    }

    /// A byte-ranged write reaches the content ledger, and each other access
    /// shape names its own bytes or says why it cannot.
    ///
    /// The whole backing is the extent its declaration gave it; a backing no
    /// declaration reached names nothing, because the model does not know how
    /// big it is; and a subresource names nothing because relating image
    /// coordinates to bytes needs a layout this crate cannot see. Three
    /// answers, and the two silences are different facts.
    #[test]
    fn a_ranged_write_advances_content_and_each_shape_names_its_own_bytes() {
        let mut b = builder(1);
        b.declare_access(write_access(9, 0, 0x40));
        let tx = b.finish().expect("frozen");

        let mut interp = Interpreter::new();
        interp.content_mut().declare(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 0x100,
            },
            Replica::GuestPages,
        );
        interp.run(&tx);
        assert!(interp.model.content().is_fresh(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 0x40
            },
            Replica::DeviceOwned
        ));
        assert!(!interp.model.content().is_fresh(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 0x40
            },
            Replica::GuestPages
        ));

        // A whole-backing write names the extent the declaration gave it.
        // Above it named nothing, so a whole-backing write covered no bytes and
        // published a version over memory it never claimed.
        let whole = AccessKey::Whole(ResourceKey {
            backing: BackingId(9),
            heap: None,
        });
        assert_eq!(
            written_bytes(interp.model.content(), whole),
            Some(ByteRange {
                offset: 0,
                length: 0x100
            })
        );
        // And a backing no declaration reached still names nothing: the model
        // does not know how big it is, and a guessed size would claim memory
        // the guest never gave it.
        assert_eq!(
            written_bytes(
                interp.model.content(),
                AccessKey::Whole(ResourceKey {
                    backing: BackingId(404),
                    heap: None
                })
            ),
            None
        );
        // A subresource is the genuine unknown: its bytes need a layout.
        assert_eq!(
            written_bytes(
                interp.model.content(),
                AccessKey::Subresource(
                    ResourceKey {
                        backing: BackingId(9),
                        heap: None
                    },
                    crate::access::SubresourceRange {
                        base_level: 0,
                        level_count: 1,
                        base_slice: 0,
                        slice_count: 1,
                        plane: 0,
                    }
                )
            ),
            None
        );
    }

    /// A refusal is observable, so a trace comparison catches a schedule that
    /// silently accepted what the serial one refused.
    #[test]
    fn a_refusal_is_part_of_the_trace() {
        let mut b = builder(1);
        b.wait_for(StampWait {
            slot: StampSlot(1),
            value: StampValue(1),
        });
        let tx = b.finish().expect("frozen");
        let mut interp = Interpreter::new();
        interp.run(&tx);
        assert_eq!(
            interp.trace(),
            &[Observation::Refused {
                ingress: IngressOrdinal(1),
                reason: Refusal::UnmeetableWait,
            }]
        );
        assert_eq!(interp.census(), (0, 1));
    }

    /// Work from a closed generation is refused, and refused by name rather
    /// than by silently doing nothing.
    #[test]
    fn work_from_a_closed_generation_is_refused() {
        let tx = builder(1).finish().expect("frozen");
        let mut interp = Interpreter::new();
        interp.reset();
        assert_eq!(
            interp.run(&tx),
            Outcome::Refused(Refusal::StaleGeneration),
            "the transaction was built in the first generation"
        );
        assert_eq!(
            interp.trace(),
            &[Observation::Refused {
                ingress: IngressOrdinal(1),
                reason: Refusal::StaleGeneration,
            }]
        );
    }

    /// A reset does not un-publish what the previous generation published.
    #[test]
    fn a_reset_keeps_what_was_already_visible() {
        let mut b = builder(1);
        b.publish_stamp(CompletionStamp {
            slot: StampSlot(2),
            value: StampValue(4),
        });
        let tx = b.finish().expect("frozen");
        let mut interp = Interpreter::new();
        interp.run(&tx);
        interp.reset();
        assert_eq!(interp.stamp(StampSlot(2)), Some(StampValue(4)));
        assert_ne!(interp.generation(), SessionGeneration::FIRST);
    }

    /// Running the same transactions twice produces the same trace. The
    /// interpreter is the definition of the serial meaning, so it must not
    /// depend on anything but its inputs.
    #[test]
    fn the_trace_is_a_function_of_the_transactions_alone() {
        let make = || {
            let mut b = builder(1);
            b.begin_segment(
                SegmentKind::Event.wire_type(),
                SegmentLifetime::SELF_CONTAINED,
            )
            .expect("open");
            b.record(signal(res(4), 2), &mut StubRegistry(ChannelId(1)))
                .expect("record");
            b.end_segment().expect("end");
            b.publish_stamp(CompletionStamp {
                slot: StampSlot(1),
                value: StampValue(1),
            });
            b.finish().expect("frozen")
        };
        let tx = make();
        let mut a = Interpreter::new();
        let mut b = Interpreter::new();
        a.run(&tx);
        b.run(&make());
        assert_eq!(a.trace(), b.trace());
    }

    /// Two transactions writing disjoint ranges of one backing are both
    /// current, whichever order they complete in.
    ///
    /// The failure a per-backing version cannot avoid: the later reservation is
    /// the higher number, so under one version per backing the earlier writer's
    /// completion arrives holding a version the backing has already passed.
    /// Here they never meet, because they cover different bytes — and the
    /// interpreter is the reference every parallel schedule is checked against,
    /// so getting this wrong here would make the whole equivalence proof agree
    /// about the wrong answer.
    #[test]
    fn two_writers_of_disjoint_ranges_are_both_current_in_either_order() {
        for late_first in [false, true] {
            let mut interp = Interpreter::new();
            interp.content_mut().declare(
                BackingId(7),
                ByteRange {
                    offset: 0,
                    length: 128,
                },
                Replica::GuestPages,
            );
            let front = AccessIntent {
                output_content_version: Some(ContentVersion(5)),
                ..write_access(7, 0, 64)
            };
            let back = AccessIntent {
                output_content_version: Some(ContentVersion(6)),
                ..write_access(7, 64, 64)
            };
            let order = if late_first {
                [back, front]
            } else {
                [front, back]
            };
            for (n, access) in order.into_iter().enumerate() {
                let mut b = builder(n as u64 + 1);
                b.declare_access(access);
                assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
            }
            assert!(
                !interp
                    .trace()
                    .iter()
                    .any(|o| matches!(o, Observation::VersionBeaten { .. })),
                "disjoint writers must not beat each other ({late_first})"
            );
            let content = interp.content_mut();
            assert_eq!(
                content.version_of(
                    BackingId(7),
                    ByteRange {
                        offset: 0,
                        length: 64
                    }
                ),
                Some(ContentVersion(5))
            );
            assert_eq!(
                content.version_of(
                    BackingId(7),
                    ByteRange {
                        offset: 64,
                        length: 64
                    }
                ),
                Some(ContentVersion(6))
            );
        }
    }

    /// A completion that lost the race publishes nothing and says so.
    ///
    /// The stale-completion rule, at the seam where a guest would see it: the
    /// newer write owns the bytes, so the older one's are never readable. It is
    /// an observation and not a refusal — the transaction ran, and what did not
    /// happen is its bytes becoming visible.
    #[test]
    fn a_completion_beaten_by_newer_content_publishes_nothing_and_names_it() {
        let mut interp = Interpreter::new();
        let newer = AccessIntent {
            output_content_version: Some(ContentVersion(9)),
            ..write_access(7, 0, 128)
        };
        let older = AccessIntent {
            output_content_version: Some(ContentVersion(4)),
            ..write_access(7, 32, 64)
        };
        for (n, access) in [newer, older].into_iter().enumerate() {
            let mut b = builder(n as u64 + 1);
            b.declare_access(access);
            assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        }
        assert_eq!(
            interp.trace(),
            &[
                Observation::VersionPublished {
                    backing: BackingId(7),
                    region: newer.key,
                    version: ContentVersion(9),
                },
                Observation::VersionBeaten {
                    backing: BackingId(7),
                    region: older.key,
                    version: ContentVersion(4),
                    landed: 0,
                },
            ],
            "the beaten write must not also read as published"
        );
        assert_eq!(
            interp.content_mut().version_of(
                BackingId(7),
                ByteRange {
                    offset: 32,
                    length: 64
                }
            ),
            Some(ContentVersion(9)),
            "the newer content still owns those bytes"
        );
    }

    /// A partly beaten completion publishes: some of its bytes did become
    /// current, and a guest reading those sees them.
    #[test]
    fn a_partly_beaten_completion_publishes_the_part_that_landed() {
        let mut interp = Interpreter::new();
        let newer = AccessIntent {
            output_content_version: Some(ContentVersion(9)),
            ..write_access(7, 64, 64)
        };
        let straddling = AccessIntent {
            output_content_version: Some(ContentVersion(4)),
            ..write_access(7, 0, 128)
        };
        for (n, access) in [newer, straddling].into_iter().enumerate() {
            let mut b = builder(n as u64 + 1);
            b.declare_access(access);
            assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        }
        assert_eq!(
            interp.trace()[1..],
            [
                Observation::VersionBeaten {
                    backing: BackingId(7),
                    region: straddling.key,
                    version: ContentVersion(4),
                    landed: 64,
                },
                Observation::VersionPublished {
                    backing: BackingId(7),
                    region: straddling.key,
                    version: ContentVersion(4),
                },
            ],
            "half landed, so it is both beaten and published"
        );
    }

    /// A whole-backing write makes the writing replica fresh for the whole
    /// backing, so a later read from it owes no transfer.
    ///
    /// The failure this replaces: a `Whole` access named no bytes, so the
    /// device's write covered nothing and the guest stayed fresh for
    /// everything. The next read from device storage then owed a copy *from*
    /// the guest — stale bytes, over content the device had just produced,
    /// with a version published saying the device's content was current.
    #[test]
    fn a_whole_backing_write_makes_its_replica_fresh_for_the_whole_backing() {
        let extent = ByteRange {
            offset: 0,
            length: 0x100,
        };
        let mut interp = Interpreter::new();
        interp
            .content_mut()
            .declare(BackingId(9), extent, Replica::GuestPages);

        let mut b = builder(1);
        b.declare_access(AccessIntent {
            key: AccessKey::Whole(ResourceKey {
                backing: BackingId(9),
                heap: None,
            }),
            output_content_version: Some(ContentVersion(5)),
            ..write_access(9, 0, 0x100)
        });
        assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);

        let content = interp.content_mut();
        assert!(content.is_fresh(BackingId(9), extent, Replica::DeviceOwned));
        assert!(!content.is_fresh(BackingId(9), extent, Replica::GuestPages));
        assert!(
            content
                .transfer_for_read(BackingId(9), extent, Replica::DeviceOwned)
                .is_none(),
            "the device produced these bytes; nothing may be copied over them"
        );
        assert_eq!(
            content.version_of(BackingId(9), extent),
            Some(ContentVersion(5))
        );

        // And it beats an older write that overlaps it, which a version over
        // no bytes could not have done.
        let mut b = builder(2);
        b.declare_access(AccessIntent {
            output_content_version: Some(ContentVersion(2)),
            ..write_access(9, 0, 0x40)
        });
        assert_eq!(interp.run(&b.finish().expect("frozen")), Outcome::Ran);
        assert!(interp
            .trace()
            .iter()
            .any(|o| matches!(o, Observation::VersionBeaten { landed: 0, .. })));
    }

    /// A control command has no records and no accesses, and the guest can
    /// still observe exactly one thing about it: its completion word. Before
    /// the interpreter took the envelope it could not run one at all, so a
    /// batch that mixed a `CmdNOP` into a schedule had a reference that
    /// silently omitted it.
    #[test]
    fn a_packet_that_is_not_gpu_work_still_publishes_its_completion_word() {
        let mut interp = Interpreter::new();
        let mut nop = control(1);
        nop.completion = Some(CompletionStamp {
            slot: StampSlot(3),
            value: StampValue(7),
        });
        assert_eq!(interp.run(&nop), Outcome::Ran);
        assert_eq!(interp.stamp(StampSlot(3)), Some(StampValue(7)));
        assert_eq!(
            interp.trace(),
            &[Observation::StampPublished {
                slot: StampSlot(3),
                value: StampValue(7),
            }]
        );
    }

    /// A stamp wait is the envelope's, so it holds every class of packet — not
    /// just the one that also has records to state it in. A control command
    /// that waits for a word nothing has published is refused, exactly as an
    /// EXEC is.
    #[test]
    fn a_stamp_wait_holds_a_packet_that_carries_no_records() {
        let mut interp = Interpreter::new();
        let mut waiter = control(1);
        waiter.stamp_waits.push(StampWait {
            slot: StampSlot(4),
            value: StampValue(2),
        });
        assert_eq!(
            interp.run(&waiter),
            Outcome::Refused(Refusal::UnmeetableWait)
        );

        // And it runs once the word is there.
        let mut producer = control(2);
        producer.completion = Some(CompletionStamp {
            slot: StampSlot(4),
            value: StampValue(2),
        });
        assert_eq!(interp.run(&producer), Outcome::Ran);
        assert_eq!(interp.run(&waiter), Outcome::Ran);
    }

    /// A lifetime operation the model declines still publishes its stamp.
    ///
    /// The device says so in as many words at the packet arms that handle these
    /// commands: the guest is waiting on the completion word, and a lifetime
    /// event this device chose not to act on is still an event the guest
    /// performed. A reference that refused the transaction would publish no
    /// stamp where the device publishes one, and the guest would wait forever
    /// for a word that agreed with the reference.
    /// A discard the device did not carry out reaches the trace.
    ///
    /// The regression: `Lifecycle::complete`'s answer --- the discards it
    /// offered and could not take --- was dropped at the call site with
    /// `let _ =` and no comment, so a device that freed nothing the guest
    /// discarded produced a trace identical to one that freed all of it. It is
    /// guest work the device declined, and this crate's rule is that such work
    /// gets a typed reason.
    #[test]
    fn a_discard_the_device_could_not_take_is_observed() {
        use crate::identity::{ObjectListRef, SlotGeneration, TaskId};
        use crate::lifecycle::{LifecycleOp, Storage};

        let task = TaskId(1);
        let backing = BackingId(10);
        let resource = ResourceId {
            slot: ObjectListRef(0),
            generation: SlotGeneration::default().next(),
        };

        let mut interp = Interpreter::new();
        for op in [
            LifecycleOp::DefineTask {
                task,
                kernel: false,
                directory: crate::identity::DirectoryFrame(0x1000),
            },
            LifecycleOp::CreateResource {
                task,
                slot: ObjectListRef(0),
                storage: Storage::Dedicated {
                    backing,
                    extent: ByteRange {
                        offset: 0,
                        length: 256,
                    },
                },
            },
        ] {
            // `Effects` is `#[must_use]`; neither of these owes anything.
            assert_eq!(
                interp.model_mut().apply(&op).expect("resolves"),
                crate::lifecycle::Effects::default()
            );
        }
        // The device wrote the whole resource and nothing else holds a copy,
        // so the discard below has nothing to free.
        interp
            .model_mut()
            .record_write(task, resource, 0, 256, crate::content::Replica::DeviceOwned)
            .expect("inside the resource");

        let tx = DeviceTransaction {
            identity: crate::testing::identity(1, 1),
            stamp_waits: Vec::new(),
            completion: None,
            payload: crate::transaction::Payload::ResourceLifecycle(
                crate::transaction::LifecyclePayload::new(
                    LifecycleOp::Discard {
                        task,
                        resources: vec![resource],
                    },
                    vec![(
                        resource,
                        AccessIntent {
                            domain: ChannelId(1),
                            key: AccessKey::Whole(ResourceKey {
                                backing,
                                heap: None,
                            }),
                            mode: AccessMode::Read,
                            api_stages: 0,
                            input_content_version: None,
                            output_content_version: None,
                        },
                    )],
                )
                .expect("one access for the one resource the op names"),
            ),
        };
        assert_eq!(interp.run(&tx), Outcome::Ran);
        assert_eq!(
            interp.trace(),
            &[Observation::DiscardDeclined {
                ingress: tx.identity.ingress,
                resource,
                backing,
                sole_authority_bytes: 256,
            }],
            "the guest asked to drop 256 bytes nothing else holds, and it did not happen"
        );
    }

    #[test]
    fn a_declined_operation_still_owes_its_completion_word() {
        let mut interp = Interpreter::new();
        // One access, for the one resource the operation names: an empty list
        // is `AccessMismatch::Unaccessed` and cannot be built.
        let mut tx = synchronize(
            1,
            vec![AccessIntent {
                domain: ChannelId(1),
                key: AccessKey::Whole(ResourceKey {
                    backing: BackingId(9),
                    heap: None,
                }),
                mode: AccessMode::Read,
                api_stages: 0,
                input_content_version: None,
                output_content_version: None,
            }],
        );
        let stamp = CompletionStamp {
            slot: StampSlot(3),
            value: StampValue(1),
        };
        tx.completion = Some(stamp);

        // Nothing was ever declared, so the synchronise names a task the model
        // does not have.
        assert_eq!(interp.run(&tx), Outcome::Ran, "the transaction ran");
        assert_eq!(
            interp.trace(),
            &[
                Observation::OperationDeclined {
                    ingress: tx.identity.ingress,
                    reason: crate::lifecycle::Refusal::NoSuchTask {
                        task: crate::identity::TaskId(1),
                    },
                },
                Observation::StampPublished {
                    slot: stamp.slot,
                    value: stamp.value,
                },
            ],
            "declined, and the word the guest is blocked on is still paid"
        );
        assert_eq!(interp.stamp(stamp.slot), Some(stamp.value));
        assert_eq!(interp.census(), (1, 0), "ran, and was not refused");
    }

    /// **A shadow over the reference's own publication rule, across every
    /// payload class that has one.**
    ///
    /// The trace is what implementations are checked against, so a version in
    /// it for bytes nothing wrote is worse here than the same defect in a
    /// device: the *correct* device is the one that gets reported as divergent.
    /// Two arms have had exactly this hole --- a query published its
    /// destination's version whether or not an answer was written, and a
    /// declined lifetime operation published its accesses' versions --- and
    /// each was found on its own, one payload class at a time.
    ///
    /// So the law is written once, over all of them: **a transaction publishes
    /// a content version only if its payload's work happened.** The shadow
    /// decides "happened" from what this sweep set up, before the run, and
    /// never from the interpreter --- a shadow that asked the model whether the
    /// task existed would agree with the code by construction and prove
    /// nothing.
    ///
    /// The three classes with no version to publish are driven too, because
    /// "publishes none" is the claim for them and a sweep that only drove the
    /// writers could not tell a present that published nothing from one that
    /// published something the comparison happened not to look at.
    #[test]
    fn a_content_version_reaches_the_trace_only_when_the_work_behind_it_happened() {
        // What this sweep decided the transaction would be, before running it.
        #[derive(Clone, Copy, Debug)]
        enum Planned {
            /// A lifetime synchronise. Its work happens iff the task exists.
            Lifecycle {
                declared: bool,
            },
            /// A query. Its work happens iff an answer of a fitting length is
            /// available.
            Query {
                answered: bool,
            },
            /// An EXEC. Its records always run, so its accesses always publish.
            Exec,
            Present,
            Control,
        }

        impl Planned {
            /// Whether this payload writes content at all, and whether the
            /// write happened. Both from the plan and neither from the run.
            const fn publishes(self) -> bool {
                match self {
                    Self::Lifecycle { declared } => declared,
                    Self::Query { answered } => answered,
                    Self::Exec => true,
                    Self::Present | Self::Control => false,
                }
            }
        }

        let mut rng: u64 = 0x243F_6A88_85A3_08D3;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        let mut tally = [0usize; 7];
        for _ in 0..2_000 {
            let plan = match next() % 6 {
                0 => Planned::Lifecycle { declared: true },
                1 => Planned::Lifecycle { declared: false },
                2 => Planned::Query { answered: true },
                3 => Planned::Query { answered: false },
                4 => Planned::Exec,
                _ => {
                    if next() % 2 == 0 {
                        Planned::Present
                    } else {
                        Planned::Control
                    }
                }
            };
            tally[match plan {
                Planned::Lifecycle { declared: true } => 0,
                Planned::Lifecycle { declared: false } => 1,
                Planned::Query { answered: true } => 2,
                Planned::Query { answered: false } => 3,
                Planned::Exec => 4,
                Planned::Present => 5,
                Planned::Control => 6,
            }] += 1;

            let mut interp = Interpreter::new();
            interp.content_mut().declare(
                BackingId(9),
                ByteRange {
                    offset: 0,
                    length: 4096,
                },
                Replica::GuestPages,
            );

            let tx = match plan {
                Planned::Lifecycle { declared } => {
                    if declared {
                        declare_resource(&mut interp, BackingId(9), 4096);
                    }
                    synchronize(
                        1,
                        vec![AccessIntent {
                            output_content_version: Some(ContentVersion(4)),
                            ..write_access(9, 0, 128)
                        }],
                    )
                }
                Planned::Query { answered } => {
                    if answered {
                        // A whole number of pairs, inside both bounds.
                        interp.answers_from(Box::new(AnswersWith(
                            reims_vgpu_protocol::info_reply::PAIR_LEN as u64 * 2,
                        )));
                    }
                    query(1, BackingId(9), 4096)
                }
                Planned::Exec => {
                    let mut b = builder(1);
                    b.declare_access(AccessIntent {
                        output_content_version: Some(ContentVersion(4)),
                        ..write_access(9, 0, 128)
                    });
                    b.finish().expect("frozen")
                }
                Planned::Present => present(1, 9),
                Planned::Control => control(1),
            };
            // Every class carries a word, because the second obligation these
            // arms state is the one about hangs: a transaction that ran pays
            // its completion word whatever its payload did or did not do. A
            // declined operation and a stalled query each say so in their own
            // doc, and neither said it about the others.
            let stamp = CompletionStamp {
                slot: StampSlot(5),
                value: StampValue(7),
            };
            let mut tx = tx;
            tx.completion = Some(stamp);

            assert_eq!(
                interp.run(&tx),
                Outcome::Ran,
                "{plan:?} is not a refusal; declining and stalling are things a \
                 transaction that ran does"
            );
            let published = interp
                .trace()
                .iter()
                .filter(|o| matches!(o, Observation::VersionPublished { .. }))
                .count();
            assert_eq!(
                published,
                usize::from(plan.publishes()),
                "{plan:?} published {published} versions; trace {:?}",
                interp.trace()
            );
            // The other half, and the one whose failure is a hang rather than
            // a wrong value: whatever the payload did, the word the guest is
            // blocked on is paid, once, and last.
            assert_eq!(
                interp.stamp(stamp.slot),
                Some(stamp.value),
                "{plan:?} left the guest waiting on its completion word"
            );
            assert_eq!(
                interp.trace().last(),
                Some(&Observation::StampPublished {
                    slot: stamp.slot,
                    value: stamp.value,
                }),
                "{plan:?} published its word before everything it makes visible"
            );
            assert_eq!(
                interp
                    .trace()
                    .iter()
                    .filter(|o| matches!(o, Observation::StampPublished { .. }))
                    .count(),
                1,
                "{plan:?} paid its word more than once"
            );
            // And the ledger agrees with the trace, so a publication that
            // reached one and not the other is caught too. `version_of` is the
            // *highest* version current over a range, so the ranges below are
            // chosen to separate what a payload wrote from what it left.
            let ledger = interp.content_mut();
            let held = |from: u64, to: u64| {
                ledger.version_of(
                    BackingId(9),
                    ByteRange {
                        offset: from,
                        length: to - from,
                    },
                )
            };
            match plan {
                // Both writers cover the whole 128 they named.
                Planned::Lifecycle { declared: true } | Planned::Exec => {
                    assert_eq!(held(0, 128), Some(ContentVersion(4)), "{plan:?}");
                }
                // The answer's own window and not the destination's, which is
                // the rule `Answer::Written` exists to carry: two pairs is 16
                // bytes and the guest handed over a 4096-byte page.
                Planned::Query { answered: true } => {
                    let pairs = reims_vgpu_protocol::info_reply::PAIR_LEN as u64 * 2;
                    assert_eq!(held(0, pairs), Some(ContentVersion(1)), "{plan:?}");
                    assert_eq!(
                        held(pairs, 4096),
                        Some(ContentVersion(0)),
                        "{plan:?} made the bytes past its reply current"
                    );
                }
                _ => assert_eq!(
                    held(0, 128),
                    Some(ContentVersion(0)),
                    "{plan:?} wrote nothing"
                ),
            }
        }

        // Per arm, because one aggregate would let an arm go undriven and still
        // read as covered.
        for (arm, count) in tally.iter().enumerate() {
            assert!(*count > 100, "arm {arm} ran {count} times");
        }
    }

    /// The other half of the same rule, and the one that would have made the
    /// reference certify a lie: a lifecycle operation the model declined moved
    /// no bytes, so it publishes no version.
    ///
    /// Identical to the test below except that nothing declares the task, so
    /// `Lifecycle::apply` refuses. Publishing version four here would tell the
    /// content authority that this backing's first 128 bytes are current at a
    /// version nothing produced, and a later read planned against it would skip
    /// the transfer that would have made it true. The completion word is still
    /// paid, because the guest is blocked on it either way.
    #[test]
    fn a_declined_lifecycle_operation_publishes_no_version_for_bytes_it_did_not_move() {
        let mut interp = Interpreter::new();
        interp.content_mut().declare(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 256,
            },
            Replica::GuestPages,
        );
        let region = AccessKey::Range(
            ResourceKey {
                backing: BackingId(9),
                heap: None,
            },
            ByteRange {
                offset: 0,
                length: 128,
            },
        );
        let mut tx = synchronize(
            1,
            vec![AccessIntent {
                domain: ChannelId(1),
                key: region,
                mode: AccessMode::Write,
                api_stages: 0,
                input_content_version: None,
                output_content_version: Some(ContentVersion(4)),
            }],
        );
        let stamp = CompletionStamp {
            slot: StampSlot(3),
            value: StampValue(1),
        };
        tx.completion = Some(stamp);

        assert_eq!(interp.run(&tx), Outcome::Ran, "declined is not refused");
        assert_eq!(
            interp.trace(),
            &[
                Observation::OperationDeclined {
                    ingress: tx.identity.ingress,
                    reason: crate::lifecycle::Refusal::NoSuchTask {
                        task: crate::identity::TaskId(1),
                    },
                },
                Observation::StampPublished {
                    slot: stamp.slot,
                    value: stamp.value,
                },
            ],
            "the word is owed and the version is not"
        );
        // And the ledger agrees. Still the version the declaration above left
        // --- the guest's own pages as declared --- and not the four this
        // transaction reserved and never wrote.
        assert_eq!(
            interp.content_mut().version_of(
                BackingId(9),
                ByteRange {
                    offset: 0,
                    length: 128,
                },
            ),
            Some(ContentVersion(0))
        );
        assert_eq!(interp.census(), (1, 0), "ran, and was not refused");
    }

    /// Content versions come from the accesses, and the accesses are the
    /// payload's whichever payload it is. A lifecycle synchronise writes bytes
    /// the guest reads back; a reference that only published an EXEC's versions
    /// would disagree with any device that honours one.
    #[test]
    fn a_lifecycle_transaction_publishes_the_versions_its_accesses_produce() {
        let mut interp = Interpreter::new();
        interp.content_mut().declare(
            BackingId(9),
            ByteRange {
                offset: 0,
                length: 256,
            },
            Replica::GuestPages,
        );
        declare_resource(&mut interp, BackingId(9), 256);
        let region = AccessKey::Range(
            ResourceKey {
                backing: BackingId(9),
                heap: None,
            },
            ByteRange {
                offset: 0,
                length: 128,
            },
        );
        let tx = synchronize(
            1,
            vec![AccessIntent {
                domain: ChannelId(1),
                key: region,
                mode: AccessMode::Write,
                api_stages: 0,
                input_content_version: None,
                output_content_version: Some(ContentVersion(4)),
            }],
        );
        assert_eq!(interp.run(&tx), Outcome::Ran);
        assert_eq!(
            interp.trace(),
            &[Observation::VersionPublished {
                backing: BackingId(9),
                region,
                version: ContentVersion(4),
            }]
        );
    }
}
