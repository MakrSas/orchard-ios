//! Ordered guest publication: when a completion stamp becomes something the
//! guest may read.
//!
//! # Completion is not publication
//!
//! Work finishing and the guest being told it finished are two events, and the
//! gap between them is the whole point of this module. A transaction's content
//! versions, event signals and fence updates become visible when its work
//! completes — out of order, as fast as the host finishes them. Its completion
//! stamp does not: a stamp is the word the guest polls to decide that
//! everything up to that point in its channel is done, so publishing it while
//! an earlier position in the same channel is still outstanding would tell the
//! guest a lie it has no way to detect.
//!
//! So each publication domain is a FIFO, and completion may release only the
//! ready positions at its head.
//!
//! # An independent domain stays free
//!
//! Head-of-line blocking within a domain is the contract. Head-of-line
//! blocking *between* domains is a bug, and the way to not have that bug is to
//! not have a structure that could express it: there is no shared queue here,
//! no global head, and no ordering between domains at all. A domain whose head
//! is stuck holds its own finished positions and nothing else's, and
//! [`Publisher::blocked`] counts exactly that so the cost is visible rather
//! than inferred.
//!
//! # A position that will never finish must leave
//!
//! A refused, cancelled or reset position is still a position, and leaving it
//! at the head would stall its domain forever. [`Publisher::withdraw`] removes
//! it and releases whatever was queued behind it — which is the only way a
//! refusal is *quiet* about ordering while being loud on the failure channel.

use crate::identity::{ChannelId, ChannelSequence, CompletionStamp};
use std::collections::{HashMap, VecDeque};

/// What a released position publishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a release nobody publishes is a completion word the guest polls forever"]
pub struct Release {
    pub sequence: ChannelSequence,
    /// The stamp the guest may now read, if this position carries one. A
    /// position with no stamp still holds the FIFO, because the positions
    /// behind it are ordered against it and not against its stamp.
    pub stamp: Option<CompletionStamp>,
}

/// Why a channel could not be retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetireRefusal {
    /// Positions are still admitted and unreleased. Retiring would drop
    /// publication the guest is owed.
    LivePositions { outstanding: usize },
}

impl RetireRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::LivePositions { .. } => "publish_live_positions",
        }
    }
}

/// One publication domain's FIFO.
#[derive(Debug, Default)]
struct Domain {
    /// Admitted and not yet released, in admission order. This is the FIFO;
    /// the sequence values need not be contiguous, because a channel may
    /// refuse a packet without admitting a position for it.
    order: VecDeque<ChannelSequence>,
    /// Positions whose work has completed, waiting for the head to reach them.
    finished: HashMap<ChannelSequence, Option<CompletionStamp>>,
    /// The highest position ever admitted into this lifetime of the channel.
    ///
    /// Kept apart from `order`, which holds only what is still outstanding: a
    /// released position leaves the FIFO, so the FIFO's back is not the
    /// channel's high-water mark and comparing against it would let a channel
    /// rewind through every position it had already published.
    admitted_through: Option<ChannelSequence>,
}

/// Ordered guest publication for every channel.
#[derive(Debug, Default)]
pub struct Publisher {
    domains: HashMap<ChannelId, Domain>,
    released: usize,
}

impl Publisher {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a position in a channel's publication order.
    ///
    /// # Panics
    ///
    /// If `sequence` is not greater than the last position admitted into this
    /// channel — *ever*, and not merely the last one still outstanding.
    /// Publication order *is* channel order; admitting out of it would make the
    /// FIFO a queue of whatever arrived rather than a statement about the
    /// guest's channel.
    ///
    /// A channel that has published everything it admitted has an empty FIFO,
    /// so the outstanding positions cannot answer this: compared against them,
    /// a channel could rewind to any position it had already published and
    /// publish it a second time, and the guest would read a stamp for work that
    /// was reported done long before. The high-water mark answers it, and
    /// [`Self::retire`] is what clears it — a later definition of the same
    /// channel is a new lifetime and starts at position one.
    pub fn admit(&mut self, domain: ChannelId, sequence: ChannelSequence) {
        let queue = self.domains.entry(domain).or_default();
        assert!(
            queue.admitted_through.is_none_or(|last| sequence > last),
            "publication order is channel order; {sequence:?} was admitted after a later position"
        );
        queue.admitted_through = Some(sequence);
        queue.order.push_back(sequence);
    }

    /// Record that a position's work has completed, and return everything the
    /// channel may now publish, in order.
    ///
    /// Returns empty when the position is not at the head — that is the
    /// ordinary case for out-of-order completion and is not an error.
    ///
    /// # Panics
    ///
    /// If the position was never admitted, or has already completed.
    /// Completing twice would publish a stamp twice.
    #[must_use = "the released positions are stamps the guest is waiting to read"]
    pub fn complete(
        &mut self,
        domain: ChannelId,
        sequence: ChannelSequence,
        stamp: Option<CompletionStamp>,
    ) -> Vec<Release> {
        let queue = self
            .domains
            .get_mut(&domain)
            .expect("completing a position in a channel that has none");
        assert!(
            queue.order.contains(&sequence),
            "completing {sequence:?}, which is not an outstanding position"
        );
        assert!(
            queue.finished.insert(sequence, stamp).is_none(),
            "completing {sequence:?} twice"
        );
        let released = Self::drain(queue);
        self.released += released.len();
        released
    }

    /// Remove a position that will never publish, and release whatever was
    /// queued behind it.
    ///
    /// # Panics
    ///
    /// If the position was never admitted, or has already completed —
    /// withdrawing completed work would discard publication the guest is owed
    /// rather than one it never will be.
    #[must_use = "the positions behind a withdrawal are stamps the guest is waiting to read"]
    pub fn withdraw(&mut self, domain: ChannelId, sequence: ChannelSequence) -> Vec<Release> {
        let queue = self
            .domains
            .get_mut(&domain)
            .expect("withdrawing a position from a channel that has none");
        assert!(
            !queue.finished.contains_key(&sequence),
            "withdrawing {sequence:?}, which has already completed"
        );
        let at = queue
            .order
            .iter()
            .position(|s| *s == sequence)
            .expect("withdrawing a position that is not outstanding");
        queue.order.remove(at);
        let released = Self::drain(queue);
        self.released += released.len();
        released
    }

    /// Release the finished prefix of a domain's FIFO.
    ///
    /// `#[must_use]` on the method as well as on [`Release`]: the lint does not
    /// look inside a `Vec`, so the annotation on the element says nothing about
    /// a call whose result is dropped — and that call is the one that publishes
    /// nothing while the FIFO has moved on.
    #[must_use = "the released positions are stamps the guest is waiting to read"]
    fn drain(queue: &mut Domain) -> Vec<Release> {
        let mut out = Vec::new();
        while let Some(head) = queue.order.front().copied() {
            let Some(stamp) = queue.finished.remove(&head) else {
                break;
            };
            queue.order.pop_front();
            out.push(Release {
                sequence: head,
                stamp,
            });
        }
        out
    }

    /// Positions admitted into a channel and not yet released.
    #[must_use]
    pub fn outstanding(&self, domain: ChannelId) -> usize {
        self.domains.get(&domain).map_or(0, |q| q.order.len())
    }

    /// The position a channel is waiting on, if it is waiting on one.
    #[must_use]
    pub fn head(&self, domain: ChannelId) -> Option<ChannelSequence> {
        self.domains
            .get(&domain)
            .and_then(|q| q.order.front().copied())
    }

    /// Positions that have finished and are held behind an unfinished head,
    /// per channel.
    ///
    /// The cost of ordered publication, measured rather than assumed. A number
    /// that grows without the corresponding head ever finishing is the shape
    /// of a stuck channel, and it is a channel's own number: nothing here can
    /// make one domain's head hold another domain's work.
    #[must_use]
    pub fn blocked(&self) -> Vec<(ChannelId, usize)> {
        let mut out: Vec<_> = self
            .domains
            .iter()
            .filter(|(_, q)| !q.finished.is_empty())
            .map(|(d, q)| (*d, q.finished.len()))
            .collect();
        out.sort_unstable();
        out
    }

    /// Positions released across every channel.
    #[must_use]
    pub const fn released(&self) -> usize {
        self.released
    }

    /// End a channel's publication lifetime.
    ///
    /// # Errors
    ///
    /// If the channel still holds unreleased positions. A later definition of
    /// the same channel starts at position one and must not join the former
    /// lifetime's queue, so the former lifetime has to be empty before it can
    /// end.
    pub fn retire(&mut self, domain: ChannelId) -> Result<(), RetireRefusal> {
        let outstanding = self.outstanding(domain);
        if outstanding > 0 {
            return Err(RetireRefusal::LivePositions { outstanding });
        }
        self.domains.remove(&domain);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{StampSlot, StampValue};

    fn stamp(value: u32) -> Option<CompletionStamp> {
        Some(CompletionStamp {
            slot: StampSlot(1),
            value: StampValue(value),
        })
    }

    fn seq(n: u64) -> ChannelSequence {
        ChannelSequence(n)
    }

    /// The rule: work may finish in any order, and the guest is told in one.
    #[test]
    fn a_position_that_finishes_early_publishes_when_its_predecessor_does() {
        let mut p = Publisher::new();
        for n in 1..=3 {
            p.admit(ChannelId(1), seq(n));
        }
        assert!(
            p.complete(ChannelId(1), seq(3), stamp(3)).is_empty(),
            "third finished first and published nothing"
        );
        assert!(p.complete(ChannelId(1), seq(2), stamp(2)).is_empty());
        assert_eq!(p.blocked(), vec![(ChannelId(1), 2)]);
        assert_eq!(
            p.complete(ChannelId(1), seq(1), stamp(1)),
            vec![
                Release {
                    sequence: seq(1),
                    stamp: stamp(1)
                },
                Release {
                    sequence: seq(2),
                    stamp: stamp(2)
                },
                Release {
                    sequence: seq(3),
                    stamp: stamp(3)
                },
            ],
            "the head finishing releases the whole ready prefix, in order"
        );
        assert_eq!(p.outstanding(ChannelId(1)), 0);
        assert!(p.blocked().is_empty());
    }

    /// The structural claim: one domain's head holds its own work and nothing
    /// else's.
    #[test]
    fn a_stuck_channel_does_not_hold_another_channels_publication() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        p.admit(ChannelId(1), seq(2));
        p.admit(ChannelId(2), seq(1));
        assert!(p.complete(ChannelId(1), seq(2), stamp(2)).is_empty());
        assert_eq!(
            p.complete(ChannelId(2), seq(1), stamp(9)),
            vec![Release {
                sequence: seq(1),
                stamp: stamp(9)
            }],
            "channel two's head is its own"
        );
        assert_eq!(p.blocked(), vec![(ChannelId(1), 1)]);
        assert_eq!(p.head(ChannelId(1)), Some(seq(1)));
        assert_eq!(p.head(ChannelId(2)), None);
    }

    /// A refusal is loud on the failure channel and quiet about ordering.
    #[test]
    fn withdrawing_a_position_releases_what_was_queued_behind_it() {
        let mut p = Publisher::new();
        for n in 1..=3 {
            p.admit(ChannelId(1), seq(n));
        }
        assert!(p.complete(ChannelId(1), seq(2), stamp(2)).is_empty());
        assert!(p.complete(ChannelId(1), seq(3), stamp(3)).is_empty());
        assert_eq!(
            p.withdraw(ChannelId(1), seq(1)),
            vec![
                Release {
                    sequence: seq(2),
                    stamp: stamp(2)
                },
                Release {
                    sequence: seq(3),
                    stamp: stamp(3)
                },
            ],
            "a position that will never publish must not stall the ones behind it"
        );
        assert_eq!(p.released(), 2);
    }

    /// A position without a stamp is still a position.
    #[test]
    fn a_position_that_publishes_no_stamp_still_holds_the_order() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        p.admit(ChannelId(1), seq(2));
        assert!(p.complete(ChannelId(1), seq(2), stamp(2)).is_empty());
        assert_eq!(
            p.complete(ChannelId(1), seq(1), None),
            vec![
                Release {
                    sequence: seq(1),
                    stamp: None
                },
                Release {
                    sequence: seq(2),
                    stamp: stamp(2)
                },
            ]
        );
    }

    /// Sequences need not be contiguous: a channel may refuse a packet without
    /// ever admitting a position for it.
    #[test]
    fn gaps_in_the_sequence_do_not_stall_the_head() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        p.admit(ChannelId(1), seq(7));
        assert_eq!(p.complete(ChannelId(1), seq(1), None).len(), 1);
        assert_eq!(p.complete(ChannelId(1), seq(7), None).len(), 1);
    }

    #[test]
    fn a_channel_with_live_positions_cannot_end_its_lifetime() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        assert_eq!(
            p.retire(ChannelId(1)),
            Err(RetireRefusal::LivePositions { outstanding: 1 })
        );
        let _ = p.complete(ChannelId(1), seq(1), None);
        assert_eq!(p.retire(ChannelId(1)), Ok(()));
        // And a later definition starts at position one without joining the
        // lifetime that just ended.
        p.admit(ChannelId(1), seq(1));
        assert_eq!(p.outstanding(ChannelId(1)), 1);
    }

    #[test]
    #[should_panic(expected = "publication order is channel order")]
    fn admitting_a_position_out_of_channel_order_is_a_contract_violation() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(5));
        p.admit(ChannelId(1), seq(2));
    }

    #[test]
    #[should_panic(expected = "twice")]
    fn completing_a_position_twice_would_publish_its_stamp_twice() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(1));
        p.admit(ChannelId(1), seq(2));
        let _ = p.complete(ChannelId(1), seq(2), None);
        let _ = p.complete(ChannelId(1), seq(2), None);
    }

    /// A channel that has published everything it admitted still may not go
    /// back: the FIFO is empty, and the emptiness is not permission.
    #[test]
    #[should_panic(expected = "publication order is channel order")]
    fn a_channel_that_published_everything_cannot_rewind_behind_it() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(5));
        assert_eq!(p.complete(ChannelId(1), seq(5), stamp(5)).len(), 1);
        assert_eq!(p.outstanding(ChannelId(1)), 0, "the FIFO is empty");
        p.admit(ChannelId(1), seq(3));
    }

    /// Retiring is what makes position one legal again, and it is the only
    /// thing that does.
    #[test]
    fn a_retired_channel_starts_its_next_lifetime_at_position_one() {
        let mut p = Publisher::new();
        p.admit(ChannelId(1), seq(5));
        assert_eq!(p.complete(ChannelId(1), seq(5), stamp(5)).len(), 1);
        p.retire(ChannelId(1)).expect("nothing outstanding");
        p.admit(ChannelId(1), seq(1));
        assert_eq!(p.head(ChannelId(1)), Some(seq(1)));
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

    /// Three channels, so head-of-line blocking has somewhere to leak to if it
    /// can.
    const CHANNELS: u64 = 3;

    /// **The rule, driven over histories: each domain publishes its own
    /// admission order and nothing else's.**
    ///
    /// The shadow is a plain `Vec` per channel with a finished-set beside it —
    /// no `VecDeque`, no shared structure — so a release the module makes that
    /// the shadow's own prefix rule does not license is a release out of order,
    /// and a domain that answers for another domain's head cannot even be
    /// expressed on the shadow side.
    ///
    /// What it asserts after every step: the releases are exactly the finished
    /// prefix, in ascending order; no position is released twice or lost; and
    /// `outstanding`, `head`, `blocked` and `released` all agree.
    #[test]
    fn every_channel_publishes_its_own_order_and_holds_nothing_elses() {
        let mut releases = 0usize;
        let mut held_behind_a_head = 0usize;
        let mut withdrawals = 0usize;
        let mut withdrawals_that_released = 0usize;
        let mut retires = 0usize;
        let mut retires_refused = 0usize;

        for seed in 0..512u64 {
            let mut rng = Rng::new(seed);
            let mut p = Publisher::new();
            // Shadow: admission order, what has finished, and the high-water
            // mark that says which position may be admitted next.
            let mut order: Vec<Vec<ChannelSequence>> = vec![Vec::new(); CHANNELS as usize];
            let mut finished: Vec<HashMap<ChannelSequence, Option<CompletionStamp>>> =
                vec![HashMap::new(); CHANNELS as usize];
            let mut next: Vec<u64> = vec![1; CHANNELS as usize];
            let mut published: Vec<Vec<ChannelSequence>> = vec![Vec::new(); CHANNELS as usize];
            // This publisher's own tally. `releases` is the whole sweep's, for
            // the non-vacuity floor, and a fresh publisher has released nothing.
            let mut released_here = 0usize;

            for _ in 0..48 {
                let c = rng.below(CHANNELS) as usize;
                let id = ChannelId(c as u32);
                match rng.below(10) {
                    // Admit the next position. Sequences skip, because a
                    // channel may refuse a packet without admitting one.
                    0..=3 => {
                        let s = seq(next[c]);
                        next[c] += 1 + rng.below(3);
                        p.admit(id, s);
                        order[c].push(s);
                    }
                    // Complete an outstanding position that has not finished.
                    4..=7 => {
                        let pending: Vec<ChannelSequence> = order[c]
                            .iter()
                            .copied()
                            .filter(|s| !finished[c].contains_key(s))
                            .collect();
                        if pending.is_empty() {
                            continue;
                        }
                        let s = pending[rng.below(pending.len() as u64) as usize];
                        let st = (rng.below(2) == 0).then(|| stamp(s.0 as u32).expect("some"));
                        let got = p.complete(id, s, st);
                        finished[c].insert(s, st);
                        let expected = drain_shadow(&mut order[c], &mut finished[c]);
                        assert_eq!(got, expected, "seed {seed}: complete released wrongly");
                        releases += got.len();
                        released_here += got.len();
                        published[c].extend(got.iter().map(|r| r.sequence));
                    }
                    // Withdraw one that will never publish.
                    8 => {
                        let live: Vec<ChannelSequence> = order[c]
                            .iter()
                            .copied()
                            .filter(|s| !finished[c].contains_key(s))
                            .collect();
                        if live.is_empty() {
                            continue;
                        }
                        let s = live[rng.below(live.len() as u64) as usize];
                        let got = p.withdraw(id, s);
                        order[c].retain(|held| *held != s);
                        let expected = drain_shadow(&mut order[c], &mut finished[c]);
                        assert_eq!(got, expected, "seed {seed}: withdraw released wrongly");
                        withdrawals += 1;
                        if !got.is_empty() {
                            withdrawals_that_released += 1;
                        }
                        releases += got.len();
                        released_here += got.len();
                        published[c].extend(got.iter().map(|r| r.sequence));
                    }
                    // End the channel's lifetime.
                    _ => {
                        let outstanding = order[c].len();
                        match p.retire(id) {
                            Ok(()) => {
                                assert_eq!(outstanding, 0, "seed {seed}: retired a live channel");
                                retires += 1;
                                // A new lifetime starts at position one.
                                next[c] = 1;
                                published[c].clear();
                                finished[c].clear();
                            }
                            Err(RetireRefusal::LivePositions { outstanding: n }) => {
                                assert_eq!(n, outstanding, "seed {seed}");
                                assert!(n > 0);
                                retires_refused += 1;
                            }
                        }
                    }
                }

                // Every observer agrees with the shadow after every step.
                for c in 0..CHANNELS as usize {
                    let id = ChannelId(c as u32);
                    assert_eq!(
                        p.outstanding(id),
                        order[c].len(),
                        "seed {seed}: outstanding"
                    );
                    assert_eq!(p.head(id), order[c].first().copied(), "seed {seed}: head");
                    // Published in ascending order, each position once.
                    assert!(
                        published[c].windows(2).all(|w| w[0] < w[1]),
                        "seed {seed}: channel {c} published out of order"
                    );
                }
                let mut expected_blocked: Vec<(ChannelId, usize)> = (0..CHANNELS as usize)
                    .filter(|c| !finished[*c].is_empty())
                    .map(|c| (ChannelId(c as u32), finished[c].len()))
                    .collect();
                expected_blocked.sort_unstable();
                assert_eq!(p.blocked(), expected_blocked, "seed {seed}: blocked");
                held_behind_a_head += expected_blocked.iter().map(|(_, n)| n).sum::<usize>();
                assert_eq!(p.released(), released_here, "seed {seed}: released");
            }
        }

        // Non-vacuity: every shape an assertion above depends on reaching.
        assert!(releases > 3_000, "positions published: {releases}");
        assert!(
            held_behind_a_head > 3_000,
            "finished positions held behind a head: {held_behind_a_head}"
        );
        assert!(withdrawals > 400, "withdrawals: {withdrawals}");
        assert!(
            withdrawals_that_released > 100,
            "withdrawals that unblocked something: {withdrawals_that_released}"
        );
        assert!(retires > 300, "channel lifetimes ended: {retires}");
        assert!(
            retires_refused > 300,
            "retires refused for live positions: {retires_refused}"
        );
    }

    /// The shadow's own release rule: the finished prefix of the admission
    /// order, stated as a prefix and not as a queue operation.
    fn drain_shadow(
        order: &mut Vec<ChannelSequence>,
        finished: &mut HashMap<ChannelSequence, Option<CompletionStamp>>,
    ) -> Vec<Release> {
        let mut out = Vec::new();
        while let Some(head) = order.first().copied() {
            let Some(stamp) = finished.remove(&head) else {
                break;
            };
            order.remove(0);
            out.push(Release {
                sequence: head,
                stamp,
            });
        }
        out
    }
}
