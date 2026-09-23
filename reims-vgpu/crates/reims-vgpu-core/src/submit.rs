//! Which physical queue a transaction lands on, and the one ordering rule that
//! is about the queue rather than about the work.
//!
//! # Ready is not the same as submittable
//!
//! [`crate::ready`] answers whether a transaction's prerequisites are
//! discharged. That is a question about the work. This answers a different one:
//! whether submitting it *now*, on the path it will actually take, can strand
//! the thing it is waiting for.
//!
//! Vulkan timeline semaphores permit wait-before-signal, and that is a real
//! feature — a submission may wait for a value nothing has signalled yet. But
//! if the waiter and its producer land on the same physical queue and the
//! implementation has only one path through that queue, submitting the waiter
//! first puts a submission that will never proceed in front of the one that
//! would have released it. Nothing in Vulkan forbids it and nothing in the API
//! contract requires it; it is exactly the ownership cycle the architecture
//! exists to make unrepresentable, arriving through the queue rather than
//! through a lock.
//!
//! So the logical-to-physical queue mapping is an input here, and the rule is:
//! **a same-physical-queue producer is submitted before its waiter.** A waiter
//! whose producer is on a *different* queue may go at once, because that
//! producer has its own path and the wait will be released from it.
//!
//! # Which waits reach this gate
//!
//! A wait the device satisfies on the CPU — held until the producer publishes,
//! then released — never reaches here, because [`crate::ready`] already will not
//! call the waiter ready until the producer has *completed*, which is strictly
//! stronger than waiting for it to be submitted. The waits this gate is for are
//! the ones forwarded to the device as timeline waits, where the waiter is
//! submittable while its producer is still running and the GPU does the waiting.
//! That is the arrangement the plan asks for — cross-queue dependencies become
//! timeline waits — and it is precisely the arrangement in which submitting the
//! waiter first can strand it.
//!
//! Which waits are forwarded is an executor decision and not this module's, and
//! it is expressed by where the caller puts them: a CPU-held wait goes to
//! [`crate::session::Packet::stamp_waits`], and a forwarded one goes to
//! [`SubmitGate::admit`] as a producer. A wait passed to both is merely
//! redundant, not wrong.
//!
//! # Holding is not blocking
//!
//! [`SubmitGate::admit`] returns an answer; it never waits. A held transaction
//! is named along with the producer it is held for, so the caller submits the
//! producer and asks again. A gate that blocked would be the first of the three
//! conditions in the deadlock class — ownership held while waiting — inside the
//! very thing built to prevent it.
//!
//! # A mapping that is not known is not assumed
//!
//! A channel with no physical queue assigned is [`Placement::Unassigned`], and
//! an unassigned waiter is held rather than submitted. Guessing "probably a
//! different queue" is the optimistic reading, and the cost of being wrong is
//! the hang above; the cost of being conservative is one extra round through
//! the gate.

use crate::identity::{ChannelId, IngressOrdinal};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One actual submission path on the host.
///
/// Not a queue family and not a queue index: two logical channels may share one
/// physical queue, and whether they do is the executor's arrangement. What
/// matters here is only whether two transactions share a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalQueue(pub u32);

/// Where a channel's work is submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    On(PhysicalQueue),
    /// No physical queue has been assigned to this channel. Treated as
    /// possibly-shared with everything, which is the conservative reading.
    Unassigned,
}

/// Which physical queue each submission domain lands on.
#[derive(Debug, Default)]
pub struct QueueMap {
    by_channel: HashMap<ChannelId, PhysicalQueue>,
}

impl QueueMap {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign a channel to a physical queue.
    ///
    /// Reassignment is allowed and is what a device recreation does: the
    /// channels are the same and the paths under them are new.
    pub fn assign(&mut self, channel: ChannelId, queue: PhysicalQueue) {
        self.by_channel.insert(channel, queue);
    }

    #[must_use]
    pub fn placement(&self, channel: ChannelId) -> Placement {
        self.by_channel
            .get(&channel)
            .copied()
            .map_or(Placement::Unassigned, Placement::On)
    }

    /// Whether two channels could share one submission path.
    ///
    /// True when they are assigned to the same queue, and true when either is
    /// unassigned — see the module docs for why the unknown case answers the
    /// conservative way.
    #[must_use]
    pub fn may_share_path(&self, a: ChannelId, b: ChannelId) -> bool {
        match (self.placement(a), self.placement(b)) {
            (Placement::On(x), Placement::On(y)) => x == y,
            _ => true,
        }
    }
}

/// What the gate said about a transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "a transaction that is neither submitted nor held is one nothing will ever run"]
pub enum Admission {
    /// Submit it.
    Go,
    /// Hold it: a producer it waits for shares its submission path and has not
    /// been submitted. Submit that producer and ask again.
    HoldForProducer {
        producer: IngressOrdinal,
        queue: Placement,
    },
}

/// The submission-order gate for one device.
///
/// Holds who has been submitted and what each pending transaction is waiting
/// for. It owns no queue, calls nothing, and cannot block.
#[derive(Debug, Default)]
pub struct SubmitGate {
    map: QueueMap,
    /// The channel each admitted transaction belongs to.
    channel: BTreeMap<IngressOrdinal, ChannelId>,
    /// Producers each transaction waits for, from the explicit wait graph.
    producers: BTreeMap<IngressOrdinal, BTreeSet<IngressOrdinal>>,
    submitted: BTreeSet<IngressOrdinal>,
    held: usize,
}

impl SubmitGate {
    #[must_use]
    pub fn new(map: QueueMap) -> Self {
        Self {
            map,
            channel: BTreeMap::new(),
            producers: BTreeMap::new(),
            submitted: BTreeSet::new(),
            held: 0,
        }
    }

    #[must_use]
    pub const fn map(&self) -> &QueueMap {
        &self.map
    }

    pub fn map_mut(&mut self) -> &mut QueueMap {
        &mut self.map
    }

    /// Record a transaction, its channel, and the producers of its explicit
    /// waits.
    ///
    /// `producers` come from [`crate::prereq::WaitGraph::edges`]; a wait with
    /// no admitted producer is not a submission-order question — nothing is
    /// being overtaken — and is [`crate::prereq::Diagnosis::Unproduced`]'s.
    pub fn admit(
        &mut self,
        ordinal: IngressOrdinal,
        channel: ChannelId,
        producers: &[IngressOrdinal],
    ) {
        self.channel.insert(ordinal, channel);
        self.producers
            .insert(ordinal, producers.iter().copied().collect());
    }

    /// Whether this transaction may be submitted now.
    ///
    /// Names the first same-path producer that has not been submitted, in
    /// ingress order, so repeated calls make progress in a stable order rather
    /// than reporting a different blocker each time.
    pub fn gate(&mut self, ordinal: IngressOrdinal) -> Admission {
        let Some(&channel) = self.channel.get(&ordinal) else {
            // Nothing was recorded about it, so it waits for nothing this gate
            // knows about and there is no path it can strand.
            return Admission::Go;
        };
        let empty = BTreeSet::new();
        let producers = self.producers.get(&ordinal).unwrap_or(&empty);
        for producer in producers {
            if self.submitted.contains(producer) {
                continue;
            }
            let Some(&producer_channel) = self.channel.get(producer) else {
                continue;
            };
            if self.map.may_share_path(channel, producer_channel) {
                self.held += 1;
                return Admission::HoldForProducer {
                    producer: *producer,
                    queue: self.map.placement(channel),
                };
            }
        }
        Admission::Go
    }

    /// Record that a transaction has been submitted.
    ///
    /// Submission, not completion: the rule is about occupying the path, and a
    /// producer that has entered the queue can no longer be overtaken by a
    /// waiter behind it.
    pub fn submitted(&mut self, ordinal: IngressOrdinal) {
        self.submitted.insert(ordinal);
    }

    /// Forget a transaction that has completed or been withdrawn.
    pub fn retire(&mut self, ordinal: IngressOrdinal) {
        self.channel.remove(&ordinal);
        self.producers.remove(&ordinal);
        self.submitted.remove(&ordinal);
        for set in self.producers.values_mut() {
            set.remove(&ordinal);
        }
    }

    /// How many times the gate has held a transaction.
    ///
    /// The number that says whether the mapping is costing submission
    /// concurrency. A gate that never holds is one whose channels do not share
    /// paths, or one whose guest writes no cross-packet waits.
    #[must_use]
    pub const fn holds(&self) -> usize {
        self.held
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ord(n: u64) -> IngressOrdinal {
        IngressOrdinal(n)
    }
    const A: ChannelId = ChannelId(1);
    const B: ChannelId = ChannelId(2);
    const Q0: PhysicalQueue = PhysicalQueue(0);
    const Q1: PhysicalQueue = PhysicalQueue(1);

    fn shared() -> SubmitGate {
        let mut map = QueueMap::new();
        map.assign(A, Q0);
        map.assign(B, Q0);
        SubmitGate::new(map)
    }

    fn split() -> SubmitGate {
        let mut map = QueueMap::new();
        map.assign(A, Q0);
        map.assign(B, Q1);
        SubmitGate::new(map)
    }

    /// The rule. A waiter that shares its producer's path is held until the
    /// producer has entered the queue, or it occupies the only path through it
    /// and the producer can never release it.
    #[test]
    fn a_same_path_waiter_is_held_until_its_producer_is_submitted() {
        let mut g = shared();
        g.admit(ord(1), A, &[]);
        g.admit(ord(2), B, &[ord(1)]);
        assert_eq!(
            g.gate(ord(2)),
            Admission::HoldForProducer {
                producer: ord(1),
                queue: Placement::On(Q0),
            }
        );
        assert_eq!(g.gate(ord(1)), Admission::Go, "the producer itself is free");
        g.submitted(ord(1));
        assert_eq!(g.gate(ord(2)), Admission::Go);
        assert_eq!(g.holds(), 1);
    }

    /// A producer on another queue has its own path, so the wait will be
    /// released from it and the waiter may go at once. Holding here would be
    /// giving up the wait-before-signal Vulkan actually offers.
    #[test]
    fn a_cross_path_waiter_goes_at_once() {
        let mut g = split();
        g.admit(ord(1), A, &[]);
        g.admit(ord(2), B, &[ord(1)]);
        assert_eq!(g.gate(ord(2)), Admission::Go);
        assert_eq!(g.holds(), 0);
    }

    /// The optimistic reading of an unknown mapping costs a hang; the
    /// conservative one costs a round through the gate.
    #[test]
    fn an_unassigned_channel_is_treated_as_possibly_shared() {
        let mut map = QueueMap::new();
        map.assign(A, Q0);
        let mut g = SubmitGate::new(map);
        assert_eq!(g.map().placement(B), Placement::Unassigned);
        g.admit(ord(1), B, &[]);
        g.admit(ord(2), A, &[ord(1)]);
        assert_eq!(
            g.gate(ord(2)),
            Admission::HoldForProducer {
                producer: ord(1),
                queue: Placement::On(Q0),
            }
        );
        // Assigning it to a different queue lifts the hold, without the
        // transaction having to be re-admitted.
        g.map_mut().assign(B, Q1);
        assert_eq!(g.gate(ord(2)), Admission::Go);
    }

    /// Submission and not completion: a producer that has entered the queue can
    /// no longer be overtaken, so the waiter goes without waiting for it to
    /// finish.
    #[test]
    fn a_submitted_producer_releases_its_waiter_before_completing() {
        let mut g = shared();
        g.admit(ord(1), A, &[]);
        g.admit(ord(2), A, &[ord(1)]);
        g.submitted(ord(1));
        assert_eq!(g.gate(ord(2)), Admission::Go);
        assert_eq!(g.holds(), 0, "it was never held");
    }

    /// Repeated calls name the same blocker until it moves, so a caller
    /// draining the gate makes progress in a stable order.
    #[test]
    fn the_first_unsubmitted_producer_in_ingress_order_is_the_one_named() {
        let mut g = shared();
        g.admit(ord(1), A, &[]);
        g.admit(ord(2), A, &[]);
        g.admit(ord(3), B, &[ord(2), ord(1)]);
        for _ in 0..3 {
            assert_eq!(
                g.gate(ord(3)),
                Admission::HoldForProducer {
                    producer: ord(1),
                    queue: Placement::On(Q0),
                },
                "the lower ordinal, every time"
            );
        }
        g.submitted(ord(1));
        assert_eq!(
            g.gate(ord(3)),
            Admission::HoldForProducer {
                producer: ord(2),
                queue: Placement::On(Q0),
            }
        );
        g.submitted(ord(2));
        assert_eq!(g.gate(ord(3)), Admission::Go);
    }

    /// A mutual wait is a `prereq::Diagnosis::Cycle` and not this gate's to
    /// resolve — but the gate must not pretend it can be submitted either.
    /// Both are held, which is the honest answer, and the caller's cycle
    /// diagnosis is what breaks it.
    #[test]
    fn two_transactions_waiting_on_each_other_are_both_held() {
        let mut g = shared();
        g.admit(ord(1), A, &[ord(2)]);
        g.admit(ord(2), A, &[ord(1)]);
        assert!(matches!(g.gate(ord(1)), Admission::HoldForProducer { .. }));
        assert!(matches!(g.gate(ord(2)), Admission::HoldForProducer { .. }));
    }

    /// A transaction the gate knows nothing about strands no path.
    #[test]
    fn an_unrecorded_transaction_may_go() {
        let mut g = shared();
        assert_eq!(g.gate(ord(99)), Admission::Go);
    }

    /// Retiring a producer takes it out of what its waiters are held for, or a
    /// completed producer would hold them forever.
    #[test]
    fn retiring_a_producer_releases_what_was_held_for_it() {
        let mut g = shared();
        g.admit(ord(1), A, &[]);
        g.admit(ord(2), A, &[ord(1)]);
        assert!(matches!(g.gate(ord(2)), Admission::HoldForProducer { .. }));
        g.retire(ord(1));
        assert_eq!(g.gate(ord(2)), Admission::Go);
    }

    /// A device recreation gives the same channels new paths.
    #[test]
    fn a_channel_may_be_reassigned() {
        let mut map = QueueMap::new();
        map.assign(A, Q0);
        map.assign(A, Q1);
        assert_eq!(map.placement(A), Placement::On(Q1));
        assert!(!map.may_share_path(A, B) || map.placement(B) == Placement::Unassigned);
    }
}
