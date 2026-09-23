//! When an admitted transaction may begin, and how a wait that will never be
//! answered is told from one that has simply not been answered yet.
//!
//! # Three kinds of prerequisite, and only one of them is ordered by arrival
//!
//! A hazard edge points backwards in ingress order by construction, so a
//! transaction's hazard prerequisites are always transactions that already
//! exist. A **stamp wait** is not like that at all: the guest may submit a
//! packet that waits for a value nothing has produced yet, and the packet that
//! will produce it may not have arrived. That is legal and ordinary, and it is
//! why the two are tracked apart — counting them together would make "waiting
//! for work that exists" and "waiting for work that does not" the same state.
//!
//! A **pipeline wait** is the third, and it is here rather than at the draw for
//! the reason [`crate::pipeline`] exists: a draw may not block, wait on a host,
//! or discover work. A transaction whose pipeline is still being built is
//! therefore not ready, and everything that does not want that pipeline is —
//! which is the whole claim, and it is only structurally true if the wait lives
//! in the readiness service beside the other two rather than in a check the
//! draw performs.
//!
//! Its terminal case is unlike the other two. A hazard is discharged by
//! completion and a stamp by publication, but a pipeline can be *refused*, and
//! a transaction waiting on a refused pipeline will never be ready. That is not
//! a stall to diagnose later: [`Scheduler::pipeline_refused`] returns those
//! transactions at once, so the caller withdraws them and names the reason
//! instead of leaving them holding their channel's head forever.
//!
//! # Nothing here blocks a thread
//!
//! Readiness is published, not waited on. A transaction becomes ready when its
//! last prerequisite is discharged, and the caller collects the ready set. No
//! method here parks, sleeps, or spins, because a scheduler that can block is a
//! scheduler that will eventually block the drain.
//!
//! # A wait nobody can answer is a diagnosis, not a timeout
//!
//! [`Scheduler::stalled`] names the transactions whose stamp waits no admitted
//! transaction will publish. That is a different statement from "this is taking
//! a while": it is derived from what has been admitted rather than from a
//! clock, so it is true the instant it becomes true and it stays true. A guest
//! waiting on a stamp nobody will write is a hang with a cause, and this is the
//! cause.

use crate::identity::{
    CompletionStamp, IngressOrdinal, ResourceId, StampSlot, StampValue, StampWait,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One admitted transaction's unfinished business.
#[derive(Debug)]
struct Pending {
    /// Earlier transactions this one must not overtake. Counted rather than
    /// listed: the list is the dependents index, held in the other direction.
    remaining_hazards: usize,
    /// Stamp points that must be published before this may begin. Shrinks as
    /// they are published.
    stamp_waits: Vec<StampWait>,
    /// The distinct slots this transaction was registered under in
    /// `waiters_by_slot`, which [`Scheduler::stamp_waits`] shrinks and this
    /// does not.
    ///
    /// The index cannot be cleaned from `stamp_waits`, because by the time a
    /// transaction is discharged that list is empty and no longer names the
    /// slots it was filed under. Without this, a transaction waiting on two
    /// slots was removed from the set of whichever slot happened to satisfy it
    /// last and left in every other — so those sets grew with the session's
    /// whole history and every publish on one scanned entries for transactions
    /// that completed long ago.
    registered_slots: Vec<StampSlot>,
    /// Pipelines that must become usable before this may begin.
    pipeline_waits: Vec<ResourceId>,
    /// What this transaction publishes when it completes.
    completion: Option<CompletionStamp>,
}

impl Pending {
    /// Whether every prerequisite is discharged. One place, so a new kind of
    /// prerequisite cannot be added and then forgotten in one of the three
    /// release paths.
    fn is_ready(&self) -> bool {
        self.remaining_hazards == 0 && self.stamp_waits.is_empty() && self.pipeline_waits.is_empty()
    }
}

/// The readiness service for one session.
#[derive(Debug, Default)]
pub struct Scheduler {
    pending: BTreeMap<IngressOrdinal, Pending>,
    /// The highest value published into each slot, in the wrapping order
    /// [`StampValue::follows`] compares in.
    published: HashMap<StampSlot, StampValue>,
    /// Transactions blocked on each slot, so publishing does not scan.
    waiters_by_slot: HashMap<StampSlot, BTreeSet<IngressOrdinal>>,
    /// Transactions blocked on each pipeline, so a compilation finishing does
    /// not scan either.
    waiters_by_pipeline: HashMap<ResourceId, BTreeSet<IngressOrdinal>>,
    /// For each transaction, the transactions that must not overtake it.
    dependents: HashMap<IngressOrdinal, Vec<IngressOrdinal>>,
    ready: BTreeSet<IngressOrdinal>,
}

impl Scheduler {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit a transaction with its prerequisites.
    ///
    /// `hazard_waits` are ordinals from the dependency compiler; any that have
    /// already completed are discharged here rather than counted, so a
    /// transaction admitted behind finished work is ready at once.
    ///
    /// `pipeline_waits` are the pipelines whose leases came back pending. A
    /// lease that was already ready is not a wait and must not be passed, or
    /// the transaction would hold for a compilation that has already finished.
    ///
    /// Returns whether it is ready immediately. The ready set is also
    /// accumulated for [`Self::take_ready`], so a caller may use either and
    /// will not see the same transaction twice.
    pub fn admit(
        &mut self,
        ordinal: IngressOrdinal,
        hazard_waits: &[IngressOrdinal],
        stamp_waits: &[StampWait],
        pipeline_waits: &[ResourceId],
        completion: Option<CompletionStamp>,
    ) -> bool {
        let live: Vec<_> = hazard_waits
            .iter()
            .copied()
            .filter(|o| self.pending.contains_key(o))
            .collect();
        let unmet: Vec<StampWait> = stamp_waits
            .iter()
            .copied()
            .filter(|w| !self.is_published(*w))
            .collect();
        for w in &unmet {
            self.waiters_by_slot
                .entry(w.slot)
                .or_default()
                .insert(ordinal);
        }
        for dep in &live {
            self.dependents.entry(*dep).or_default().push(ordinal);
        }
        for pipeline in pipeline_waits {
            self.waiters_by_pipeline
                .entry(*pipeline)
                .or_default()
                .insert(ordinal);
        }
        let ready = live.is_empty() && unmet.is_empty() && pipeline_waits.is_empty();
        let mut registered_slots: Vec<StampSlot> = unmet.iter().map(|w| w.slot).collect();
        registered_slots.sort_unstable();
        registered_slots.dedup();
        self.pending.insert(
            ordinal,
            Pending {
                remaining_hazards: live.len(),
                stamp_waits: unmet,
                pipeline_waits: pipeline_waits.to_vec(),
                registered_slots,
                completion,
            },
        );
        if ready {
            self.ready.insert(ordinal);
        }
        ready
    }

    /// Whether a wait is already discharged by what has been published.
    #[must_use]
    pub fn is_published(&self, wait: StampWait) -> bool {
        self.published
            .get(&wait.slot)
            .is_some_and(|v| wait.satisfied_by(*v))
    }

    /// The value standing in a slot, if anything has published one.
    #[must_use]
    pub fn published_value(&self, slot: StampSlot) -> Option<StampValue> {
        self.published.get(&slot).copied()
    }

    /// Complete a transaction: release what was waiting on it, and hand back
    /// the stamp it now owes.
    ///
    /// It does **not** publish that stamp. When a completion word becomes
    /// readable is [`crate::publish`]'s question, not readiness's: a channel
    /// publishes in its own order, so a transaction that finishes ahead of an
    /// earlier position in its channel owes its stamp and may not yet pay it.
    /// A scheduler that published here would be deciding, silently, that
    /// completion and publication are the same event.
    ///
    /// Hazard dependents are released regardless, because they wait for the
    /// *work* and not for the guest being told about it.
    ///
    /// # Panics
    ///
    /// If `ordinal` was never admitted or has already completed. Completing a
    /// transaction twice would hand its stamp back twice and decrement its
    /// dependents past zero, and both of those are silent corruptions rather
    /// than loud ones.
    pub fn complete(&mut self, ordinal: IngressOrdinal) -> Option<CompletionStamp> {
        let done = self
            .pending
            .remove(&ordinal)
            .expect("completing a transaction that is not pending");
        self.ready.remove(&ordinal);
        // Every slot it was filed under, not the ones it is still waiting on:
        // a transaction discharged by publication has an empty `stamp_waits`
        // and its index entries would otherwise stay for the rest of the
        // session.
        self.unregister_slots(ordinal, &done.registered_slots);
        for pipeline in &done.pipeline_waits {
            let Some(set) = self.waiters_by_pipeline.get_mut(pipeline) else {
                continue;
            };
            set.remove(&ordinal);
            if set.is_empty() {
                self.waiters_by_pipeline.remove(pipeline);
            }
        }
        for dep in self.dependents.remove(&ordinal).unwrap_or_default() {
            if let Some(p) = self.pending.get_mut(&dep) {
                p.remaining_hazards -= 1;
                if p.is_ready() {
                    self.ready.insert(dep);
                }
            }
        }
        done.completion
    }

    /// Publish a stamp value without completing a transaction.
    ///
    /// The guest can advance a timeline itself, and a device that only ever
    /// published from its own completions would hold packets against a value
    /// that had already been written.
    pub fn publish(&mut self, stamp: CompletionStamp) {
        let slot = self.published.entry(stamp.slot).or_insert(stamp.value);
        // Later in the wrapping order, not numerically: a wrapped timeline
        // makes the later value the smaller one, and `max` would then refuse
        // to advance past the wrap for the rest of the boot.
        *slot = slot.later(stamp.value);
        let published = *slot;
        let Some(waiters) = self.waiters_by_slot.get(&stamp.slot).cloned() else {
            return;
        };
        let mut discharged: Vec<(IngressOrdinal, Vec<StampSlot>)> = Vec::new();
        for w in waiters {
            let Some(p) = self.pending.get_mut(&w) else {
                continue;
            };
            p.stamp_waits
                .retain(|wait| wait.slot != stamp.slot || !wait.satisfied_by(published));
            if p.stamp_waits.is_empty() {
                let slots = p.registered_slots.clone();
                if p.is_ready() {
                    self.ready.insert(w);
                }
                discharged.push((w, slots));
            }
        }
        // Out of *every* slot it was filed under. Removing it from this one
        // alone left it in the others for the rest of the session, so those
        // sets grew with the history and each publish scanned transactions that
        // had long since completed.
        for (w, slots) in discharged {
            self.unregister_slots(w, &slots);
        }
    }

    /// Take a transaction out of the slot index it was filed under, and drop a
    /// slot's set when nothing waits on it any more.
    fn unregister_slots(&mut self, ordinal: IngressOrdinal, slots: &[StampSlot]) {
        for slot in slots {
            let Some(set) = self.waiters_by_slot.get_mut(slot) else {
                continue;
            };
            set.remove(&ordinal);
            if set.is_empty() {
                self.waiters_by_slot.remove(slot);
            }
        }
    }

    /// A pipeline became usable: discharge every transaction waiting on it.
    ///
    /// Indexed rather than scanned, so a compilation finishing costs the
    /// waiters it actually had.
    pub fn pipeline_ready(&mut self, pipeline: ResourceId) {
        let Some(waiters) = self.waiters_by_pipeline.remove(&pipeline) else {
            return;
        };
        for w in waiters {
            let Some(p) = self.pending.get_mut(&w) else {
                continue;
            };
            p.pipeline_waits.retain(|id| *id != pipeline);
            if p.is_ready() {
                self.ready.insert(w);
            }
        }
    }

    /// A pipeline will never be usable: name the transactions that can
    /// therefore never be ready.
    ///
    /// They are returned rather than dropped or made ready. Made ready they
    /// would execute against a pipeline that does not exist; dropped they would
    /// hold their channel's publication head forever. The caller withdraws them
    /// and says why on its failure channel — which is the same division of
    /// labour [`Scheduler::stalled`] follows, except that this one is certain
    /// rather than diagnosed.
    #[must_use]
    pub fn pipeline_refused(&mut self, pipeline: ResourceId) -> Vec<IngressOrdinal> {
        let Some(waiters) = self.waiters_by_pipeline.remove(&pipeline) else {
            return Vec::new();
        };
        waiters
            .into_iter()
            .filter(|w| self.pending.contains_key(w))
            .collect()
    }

    /// Transactions waiting on a pipeline that is still being built.
    #[must_use]
    pub fn waiting_on_pipelines(&self) -> usize {
        self.pending
            .values()
            .filter(|p| !p.pipeline_waits.is_empty())
            .count()
    }

    /// Take the transactions that have become ready since the last call.
    ///
    /// Taken, not read: they leave the list, so a caller that drops the answer
    /// has transactions nothing will ever run and a channel head nothing will
    /// ever release. Same obligation as [`Self::pipeline_refused`], arriving
    /// through the success path instead of the refusal one.
    #[must_use = "a transaction taken off the ready list and not run is one that never runs"]
    pub fn take_ready(&mut self) -> Vec<IngressOrdinal> {
        std::mem::take(&mut self.ready).into_iter().collect()
    }

    /// Admitted transactions that have not completed.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Transactions waiting on a stamp point no admitted transaction will ever
    /// publish.
    ///
    /// Derived from what has been admitted rather than from a clock, so it is a
    /// diagnosis rather than a timeout: the answer does not depend on how long
    /// anyone waited. It is not a claim that the guest is wrong — a packet that
    /// will publish the value may still arrive — which is why the caller
    /// decides what to do with it, and why nothing here cancels anything.
    #[must_use]
    pub fn stalled(&self) -> Vec<IngressOrdinal> {
        let mut publishable: BTreeMap<StampSlot, StampValue> = BTreeMap::new();
        for p in self.pending.values() {
            if let Some(c) = p.completion {
                publishable
                    .entry(c.slot)
                    .and_modify(|v| *v = v.later(c.value))
                    .or_insert(c.value);
            }
        }
        self.pending
            .iter()
            .filter(|(_, p)| {
                p.stamp_waits.iter().any(|w| {
                    !publishable
                        .get(&w.slot)
                        .is_some_and(|highest| highest.reached(w.value))
                })
            })
            .map(|(o, _)| *o)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ord(n: u64) -> IngressOrdinal {
        IngressOrdinal(n)
    }
    fn wait(slot: u32, value: u32) -> StampWait {
        StampWait {
            slot: StampSlot(slot),
            value: StampValue(value),
        }
    }
    fn stamp(slot: u32, value: u32) -> CompletionStamp {
        CompletionStamp {
            slot: StampSlot(slot),
            value: StampValue(value),
        }
    }

    fn pipe(slot: u32) -> ResourceId {
        ResourceId {
            slot: crate::identity::ObjectListRef(slot),
            generation: crate::identity::SlotGeneration::default().next(),
        }
    }

    /// **The waiter index holds only live waiters.**
    ///
    /// It is a `HashMap<StampSlot, BTreeSet<_>>` whose whole purpose, in its
    /// own words, is that "publishing does not scan". A transaction waiting on
    /// two slots was filed under both and taken out of only the one that
    /// happened to satisfy it last — because by then its `stamp_waits` was
    /// empty and no longer named where it had been filed. So the other slot's
    /// set grew by one per transaction for the life of the session, and every
    /// publish on it walked every transaction that had ever waited there.
    ///
    /// A timeline slot is published every frame, so this is unbounded in both
    /// memory and per-publish cost.
    #[test]
    fn the_waiter_index_does_not_grow_with_transactions_that_have_finished() {
        let mut s = Scheduler::new();
        for n in 1..=50u64 {
            let value = n as u32;
            assert!(!s.admit(ord(n), &[], &[wait(1, value), wait(2, value)], &[], None));
            s.publish(stamp(1, value));
            s.publish(stamp(2, value));
            assert_eq!(s.take_ready(), vec![ord(n)], "both waits are discharged");
            assert_eq!(s.complete(ord(n)), None);
        }
        assert_eq!(s.pending(), 0);
        assert!(
            s.waiters_by_slot.is_empty(),
            "the index still holds {:?}",
            s.waiters_by_slot
        );
    }

    /// The same, for a transaction that completes while it is still waiting.
    #[test]
    fn a_withdrawn_waiter_leaves_no_index_entry_behind() {
        let mut s = Scheduler::new();
        assert!(!s.admit(ord(1), &[], &[wait(1, 5), wait(2, 5)], &[pipe(3)], None));
        assert_eq!(s.complete(ord(1)), None);
        assert!(s.waiters_by_slot.is_empty(), "slots");
        assert!(s.waiters_by_pipeline.is_empty(), "pipelines");
    }

    #[test]
    fn a_transaction_with_no_prerequisites_is_ready_at_once() {
        let mut s = Scheduler::new();
        assert!(s.admit(ord(1), &[], &[], &[], None));
        assert_eq!(s.take_ready(), vec![ord(1)]);
        assert!(
            s.take_ready().is_empty(),
            "the ready set is taken, not read"
        );
    }

    #[test]
    fn a_hazard_dependent_becomes_ready_when_its_predecessor_completes() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], &[], None);
        assert!(!s.admit(ord(2), &[ord(1)], &[], &[], None));
        assert_eq!(s.take_ready(), vec![ord(1)]);
        s.complete(ord(1));
        assert_eq!(s.take_ready(), vec![ord(2)]);
    }

    /// Work admitted behind work that has already finished is not waiting for
    /// anything. A compiler that reported a hazard against a retired
    /// transaction would otherwise park it forever.
    #[test]
    fn a_hazard_against_a_completed_transaction_is_already_discharged() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], &[], None);
        s.complete(ord(1));
        assert!(s.admit(ord(2), &[ord(1)], &[], &[], None));
    }

    #[test]
    fn a_stamp_wait_holds_until_the_value_is_published() {
        let mut s = Scheduler::new();
        assert!(!s.admit(ord(1), &[], &[wait(4, 10)], &[], None));
        s.publish(stamp(4, 9));
        assert!(s.take_ready().is_empty(), "nine does not reach ten");
        s.publish(stamp(4, 10));
        assert_eq!(s.take_ready(), vec![ord(1)]);
    }

    /// A wait already discharged when the packet arrives is not a wait. This is
    /// the case that parks a channel against a value the device has itself
    /// already written.
    #[test]
    fn a_stamp_wait_already_satisfied_at_admission_does_not_hold() {
        let mut s = Scheduler::new();
        s.publish(stamp(4, 12));
        assert!(s.admit(ord(1), &[], &[wait(4, 10)], &[], None));
    }

    /// The slot keeps the later value in the *wrapping* order. A `max` here
    /// would refuse to advance past a wrap for the rest of the boot.
    #[test]
    fn publishing_across_a_wrap_still_advances_the_slot() {
        let mut s = Scheduler::new();
        s.publish(stamp(0, u32::MAX - 1));
        s.publish(stamp(0, 3));
        assert_eq!(s.published_value(StampSlot(0)), Some(StampValue(3)));
        assert!(s.admit(ord(1), &[], &[wait(0, 2)], &[], None));
    }

    /// And an out-of-order publication does not walk the slot backwards.
    #[test]
    fn publishing_an_older_value_does_not_retract_the_slot() {
        let mut s = Scheduler::new();
        s.publish(stamp(0, 20));
        s.publish(stamp(0, 5));
        assert_eq!(s.published_value(StampSlot(0)), Some(StampValue(20)));
    }

    /// Completion hands the stamp back; it does not publish it. A hazard
    /// dependent waits for the work and is released at once; a stamp waiter
    /// waits for the guest-visible word and is not.
    #[test]
    fn completion_releases_hazard_dependents_and_owes_its_stamp() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], &[], Some(stamp(7, 1)));
        s.admit(ord(2), &[ord(1)], &[], &[], None);
        s.admit(ord(3), &[], &[wait(7, 1)], &[], None);
        let _ = s.take_ready();
        let owed = s.complete(ord(1));
        assert_eq!(owed, Some(stamp(7, 1)));
        assert_eq!(
            s.take_ready(),
            vec![ord(2)],
            "the stamp waiter waits for publication, which has not happened"
        );
        s.publish(owed.expect("a stamp"));
        assert_eq!(s.take_ready(), vec![ord(3)]);
    }

    #[test]
    fn a_transaction_with_both_kinds_of_prerequisite_waits_for_both() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], &[], None);
        s.admit(ord(2), &[ord(1)], &[wait(3, 5)], &[], None);
        let _ = s.take_ready();
        s.complete(ord(1));
        assert!(s.take_ready().is_empty(), "the stamp is still unpublished");
        s.publish(stamp(3, 5));
        assert_eq!(s.take_ready(), vec![ord(2)]);
    }

    /// The diagnosis: a wait no admitted transaction can answer.
    #[test]
    fn a_wait_nobody_will_publish_is_named() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[wait(2, 100)], &[], None);
        assert_eq!(s.stalled(), vec![ord(1)]);
        // A packet that *will* publish it arrives: no longer stalled, and the
        // answer changed without a clock being involved.
        s.admit(ord(2), &[], &[], &[], Some(stamp(2, 100)));
        assert!(s.stalled().is_empty());
        let owed = s.complete(ord(2)).expect("a stamp");
        s.publish(owed);
        assert_eq!(s.take_ready(), vec![ord(1)]);
    }

    /// A publisher that will not publish *far enough* is still not an answer.
    #[test]
    fn a_publisher_that_stops_short_does_not_discharge_the_wait() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[wait(2, 100)], &[], None);
        s.admit(ord(2), &[], &[], &[], Some(stamp(2, 99)));
        assert_eq!(s.stalled(), vec![ord(1)]);
    }

    /// Two transactions each waiting for the other's stamp. Both are named,
    /// because neither can be the one that moves.
    #[test]
    fn a_mutual_wait_names_both_sides() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[wait(1, 5)], &[], Some(stamp(2, 5)));
        s.admit(ord(2), &[], &[wait(2, 5)], &[], Some(stamp(1, 5)));
        assert!(
            s.stalled().is_empty(),
            "each side's stamp is publishable by the other, so this is a \
             deadlock in the wait-for graph rather than an unanswerable wait — \
             a different diagnosis, and not this one's to make"
        );
        assert_eq!(s.pending(), 2);
        assert!(s.take_ready().is_empty());
    }

    #[test]
    #[should_panic(expected = "not pending")]
    fn completing_twice_is_loud() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], &[], None);
        s.complete(ord(1));
        s.complete(ord(1));
    }
    /// The plan's claim, and the reason the wait lives here rather than at the
    /// draw: slow pipeline compilation blocks only the transactions that want
    /// that pipeline.
    #[test]
    fn a_compiling_pipeline_blocks_only_what_wants_it() {
        let mut s = Scheduler::new();
        assert!(
            !s.admit(ord(1), &[], &[], &[pipe(4)], None),
            "the transaction that wants the pipeline is not ready"
        );
        assert!(
            s.admit(ord(2), &[], &[], &[], None),
            "and independent work is"
        );
        assert!(
            !s.admit(ord(3), &[], &[], &[pipe(9)], None),
            "as is a transaction wanting a different pipeline — separately"
        );
        assert_eq!(s.take_ready(), vec![ord(2)]);
        assert_eq!(s.waiting_on_pipelines(), 2);

        s.pipeline_ready(pipe(4));
        assert_eq!(s.take_ready(), vec![ord(1)]);
        assert_eq!(s.waiting_on_pipelines(), 1, "the other one still waits");
        s.pipeline_ready(pipe(9));
        assert_eq!(s.take_ready(), vec![ord(3)]);
        assert_eq!(s.waiting_on_pipelines(), 0);
    }

    /// A pipeline is one prerequisite among three, and discharging it alone
    /// does not make a transaction ready.
    #[test]
    fn a_pipeline_wait_is_not_the_only_prerequisite() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], &[], None);
        assert!(!s.admit(ord(2), &[ord(1)], &[wait(0, 5)], &[pipe(4)], None));
        let _ = s.take_ready();

        s.pipeline_ready(pipe(4));
        assert!(s.take_ready().is_empty(), "the hazard and the stamp remain");
        s.complete(ord(1));
        assert!(s.take_ready().is_empty(), "the stamp remains");
        s.publish(stamp(0, 5));
        assert_eq!(s.take_ready(), vec![ord(2)]);
    }

    /// And the same from each other direction: whichever prerequisite is
    /// discharged last is the one that makes it ready, so no release path may
    /// forget to ask about the others.
    #[test]
    fn the_pipeline_may_be_the_last_prerequisite_or_the_first() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], &[], None);
        s.admit(ord(2), &[ord(1)], &[wait(0, 5)], &[pipe(4)], None);
        let _ = s.take_ready();
        s.complete(ord(1));
        assert!(s.take_ready().is_empty());
        s.publish(stamp(0, 5));
        assert!(
            s.take_ready().is_empty(),
            "the pipeline is the last one standing"
        );
        s.pipeline_ready(pipe(4));
        assert_eq!(s.take_ready(), vec![ord(2)]);
    }

    /// A refused pipeline is certain rather than diagnosed. Its waiters can
    /// never be ready, so they come back at once — made ready they would
    /// execute against a pipeline that does not exist, and dropped they would
    /// hold their channel's publication head forever.
    #[test]
    fn a_refused_pipeline_names_the_transactions_that_can_never_run() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], &[pipe(4)], None);
        s.admit(ord(2), &[], &[], &[pipe(4)], None);
        s.admit(ord(3), &[], &[], &[pipe(7)], None);
        assert_eq!(s.pipeline_refused(pipe(4)), vec![ord(1), ord(2)]);
        assert!(s.take_ready().is_empty(), "and none of them was made ready");
        assert_eq!(
            s.pipeline_refused(pipe(4)),
            vec![],
            "asking twice names nobody twice"
        );
        assert_eq!(
            s.waiting_on_pipelines(),
            3,
            "they are still pending until the caller withdraws them"
        );
        assert_eq!(s.pipeline_refused(pipe(7)), vec![ord(3)]);
    }

    /// A transaction that completed is no longer a waiter, so a compilation
    /// finishing afterwards finds nothing and says nothing.
    #[test]
    fn a_completed_transaction_stops_waiting_on_its_pipeline() {
        let mut s = Scheduler::new();
        s.admit(ord(1), &[], &[], &[pipe(4)], None);
        s.pipeline_ready(pipe(4));
        let _ = s.take_ready();
        s.complete(ord(1));
        assert_eq!(s.waiting_on_pipelines(), 0);
        s.pipeline_ready(pipe(4));
        assert!(s.take_ready().is_empty());
        assert_eq!(s.pipeline_refused(pipe(4)), vec![]);
    }

    struct Rng(u64);

    impl Rng {
        const fn new(seed: u64) -> Self {
            Self(seed ^ 0x9E37_79B9_7F4A_7C15)
        }

        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: u64) -> u64 {
            if bound == 0 {
                return 0;
            }
            self.next() % bound
        }
    }

    /// One pending transaction as the shadow holds it: its ordinal, and the
    /// three kinds of prerequisite it has not met yet.
    type Waiting = (
        IngressOrdinal,
        Vec<IngressOrdinal>,
        Vec<StampWait>,
        Vec<ResourceId>,
    );

    /// Tiny pools, so slots and pipelines are shared constantly and one
    /// publication discharges several waiters at once.
    const SLOTS: u64 = 3;
    const PIPES: u64 = 3;

    /// **Readiness is a function of what is outstanding, and the indexes hold
    /// exactly the live waiters.**
    ///
    /// The shadow is a flat list of pending transactions with their unmet
    /// prerequisites and a published map — no indexes at all. Every readiness
    /// answer is recomputed from it, so an index that discharged a transaction
    /// early, missed one, or kept an entry for one that had finished cannot
    /// agree.
    ///
    /// The last of those is the one the indexes existed to get right and did
    /// not: they are what makes publishing cost its waiters rather than the
    /// session's history, so the sweep asserts their exact contents after every
    /// step rather than only the readiness they produce.
    #[test]
    fn the_indexes_hold_exactly_the_transactions_still_waiting() {
        let mut admitted = 0usize;
        let mut ready_at_once = 0usize;
        let mut discharged_by_publish = 0usize;
        let mut discharged_by_pipeline = 0usize;
        let mut discharged_by_hazard = 0usize;
        let mut refused_pipelines = 0usize;

        for seed in 0..384u64 {
            let mut rng = Rng::new(seed);
            let mut s = Scheduler::new();
            // Shadow: what is pending, and what has been published.
            let mut pending: Vec<Waiting> = Vec::new();
            let mut published: BTreeMap<StampSlot, StampValue> = BTreeMap::new();
            let mut taken: Vec<IngressOrdinal> = Vec::new();
            let mut next = 1u64;

            for _ in 0..48 {
                match rng.below(10) {
                    // Admit, with prerequisites drawn from what exists.
                    0..=3 => {
                        let ordinal = ord(next);
                        next += 1;
                        let hazards: Vec<IngressOrdinal> = if pending.is_empty() {
                            Vec::new()
                        } else {
                            (0..rng.below(3))
                                .map(|_| pending[rng.below(pending.len() as u64) as usize].0)
                                .collect()
                        };
                        let stamps: Vec<StampWait> = (0..rng.below(3))
                            .map(|_| wait(rng.below(SLOTS) as u32 + 1, rng.below(4) as u32 + 1))
                            .collect();
                        let pipes: Vec<ResourceId> = (0..rng.below(3))
                            .map(|_| pipe(rng.below(PIPES) as u32 + 1))
                            .collect();
                        let got = s.admit(ordinal, &hazards, &stamps, &pipes, None);

                        // The shadow keeps only the prerequisites that are not
                        // already discharged, which is the same rule stated
                        // without an index.
                        let live_hazards: Vec<IngressOrdinal> = hazards
                            .iter()
                            .copied()
                            .filter(|h| pending.iter().any(|(o, ..)| o == h))
                            .collect();
                        let unmet: Vec<StampWait> = stamps
                            .iter()
                            .copied()
                            .filter(|w| !published.get(&w.slot).is_some_and(|v| w.satisfied_by(*v)))
                            .collect();
                        let expected =
                            live_hazards.is_empty() && unmet.is_empty() && pipes.is_empty();
                        assert_eq!(got, expected, "seed {seed}: admit readiness");
                        if expected {
                            ready_at_once += 1;
                            taken.push(ordinal);
                        }
                        pending.push((ordinal, live_hazards, unmet, pipes));
                        admitted += 1;
                    }
                    // Publish a stamp value the guest or a completion wrote.
                    4..=5 => {
                        let st = stamp(rng.below(SLOTS) as u32 + 1, rng.below(6) as u32 + 1);
                        s.publish(st);
                        let at = published
                            .entry(st.slot)
                            .and_modify(|v| *v = v.later(st.value))
                            .or_insert(st.value);
                        let at = *at;
                        for (ordinal, hz, waits, pipes) in &mut pending {
                            let was = waits.is_empty();
                            waits.retain(|w| w.slot != st.slot || !w.satisfied_by(at));
                            if !was && waits.is_empty() && hz.is_empty() && pipes.is_empty() {
                                taken.push(*ordinal);
                                discharged_by_publish += 1;
                            }
                        }
                    }
                    // A pipeline became usable.
                    6..=7 => {
                        let id = pipe(rng.below(PIPES) as u32 + 1);
                        s.pipeline_ready(id);
                        for (ordinal, hz, waits, pipes) in &mut pending {
                            let was = pipes.is_empty();
                            pipes.retain(|p| *p != id);
                            if !was && pipes.is_empty() && hz.is_empty() && waits.is_empty() {
                                taken.push(*ordinal);
                                discharged_by_pipeline += 1;
                            }
                        }
                    }
                    // A pipeline will never build.
                    8 => {
                        let id = pipe(rng.below(PIPES) as u32 + 1);
                        let mut expected: Vec<IngressOrdinal> = pending
                            .iter()
                            .filter(|(_, _, _, pipes)| pipes.contains(&id))
                            .map(|(o, ..)| *o)
                            .collect();
                        let mut got = s.pipeline_refused(id);
                        expected.sort_unstable();
                        got.sort_unstable();
                        assert_eq!(got, expected, "seed {seed}: pipeline_refused");
                        refused_pipelines += got.len();
                        // The caller withdraws them; here that is a completion
                        // with the stamp dropped, which is what `withdraw` does.
                        for ordinal in got {
                            complete_shadow(
                                &mut s,
                                &mut pending,
                                &mut taken,
                                ordinal,
                                &mut discharged_by_hazard,
                            );
                        }
                    }
                    // A transaction finished.
                    _ => {
                        if pending.is_empty() {
                            continue;
                        }
                        let ordinal = pending[rng.below(pending.len() as u64) as usize].0;
                        complete_shadow(
                            &mut s,
                            &mut pending,
                            &mut taken,
                            ordinal,
                            &mut discharged_by_hazard,
                        );
                    }
                }

                // The ready set is exactly what the shadow says has no unmet
                // prerequisite and has not been taken yet.
                let mut expected_ready: Vec<IngressOrdinal> = taken
                    .iter()
                    .copied()
                    .filter(|o| pending.iter().any(|(p, ..)| p == o))
                    .collect();
                expected_ready.sort_unstable();
                expected_ready.dedup();
                let mut got = s.take_ready();
                got.sort_unstable();
                assert_eq!(got, expected_ready, "seed {seed}: ready set");
                taken.retain(|o| !expected_ready.contains(o));

                // The indexes hold exactly the live waiters, and no empty set.
                assert_eq!(s.pending(), pending.len(), "seed {seed}: pending");
                for (slot, set) in &s.waiters_by_slot {
                    assert!(!set.is_empty(), "seed {seed}: an empty slot set was kept");
                    for w in set {
                        assert!(
                            pending.iter().any(|(o, ..)| o == w),
                            "seed {seed}: slot {slot:?} holds finished {w:?}"
                        );
                    }
                }
                for (id, set) in &s.waiters_by_pipeline {
                    assert!(!set.is_empty(), "seed {seed}: an empty pipeline set");
                    for w in set {
                        assert!(
                            pending.iter().any(|(o, ..)| o == w),
                            "seed {seed}: pipeline {id:?} holds finished {w:?}"
                        );
                    }
                }
                assert_eq!(
                    s.waiting_on_pipelines(),
                    pending
                        .iter()
                        .filter(|(_, _, _, pipes)| !pipes.is_empty())
                        .count(),
                    "seed {seed}: waiting_on_pipelines"
                );
            }

            // Nothing survives the last transaction leaving.
            let remaining: Vec<IngressOrdinal> = pending.iter().map(|(o, ..)| *o).collect();
            for ordinal in remaining {
                complete_shadow(
                    &mut s,
                    &mut pending,
                    &mut taken,
                    ordinal,
                    &mut discharged_by_hazard,
                );
            }
            assert_eq!(s.pending(), 0, "seed {seed}");
            assert!(
                s.waiters_by_slot.is_empty(),
                "seed {seed}: slot index leaked"
            );
            assert!(
                s.waiters_by_pipeline.is_empty(),
                "seed {seed}: pipeline index leaked"
            );
            assert!(s.dependents.is_empty(), "seed {seed}: dependents leaked");
        }

        // Non-vacuity: every shape an assertion above depends on reaching.
        assert!(admitted > 5_000, "transactions admitted: {admitted}");
        assert!(ready_at_once > 400, "ready on admission: {ready_at_once}");
        assert!(
            discharged_by_publish > 200,
            "discharged by a publication: {discharged_by_publish}"
        );
        assert!(
            discharged_by_pipeline > 400,
            "discharged by a compilation: {discharged_by_pipeline}"
        );
        assert!(
            discharged_by_hazard > 1_000,
            "discharged by an earlier transaction finishing: {discharged_by_hazard}"
        );
        assert!(
            refused_pipelines > 800,
            "transactions named by a refused pipeline: {refused_pipelines}"
        );
    }

    /// Complete one transaction in both the scheduler and the shadow, and
    /// release whatever was waiting on it.
    fn complete_shadow(
        s: &mut Scheduler,
        pending: &mut Vec<Waiting>,
        taken: &mut Vec<IngressOrdinal>,
        ordinal: IngressOrdinal,
        discharged_by_hazard: &mut usize,
    ) {
        if !pending.iter().any(|(o, ..)| *o == ordinal) {
            return;
        }
        let _ = s.complete(ordinal);
        pending.retain(|(o, ..)| *o != ordinal);
        for (other, hz, waits, pipes) in pending.iter_mut() {
            let was = hz.is_empty();
            hz.retain(|h| *h != ordinal);
            if !was && hz.is_empty() && waits.is_empty() && pipes.is_empty() {
                taken.push(*other);
                *discharged_by_hazard += 1;
            }
        }
    }
}
