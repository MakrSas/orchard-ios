//! What happens to a timeline point between reserving it and the GPU reaching
//! it.
//!
//! # A reserved point that is never signalled is a leak with no symptom
//!
//! Everything deferred in this rail is deferred against a timeline point:
//! command-buffer slots ([`crate::pools`]), descriptor sets
//! ([`crate::descriptor`]), swapchain images ([`crate::frames`]), native object
//! destruction. All of them wait for the counter to reach a value.
//!
//! So a point that is reserved and then never signalled — because the
//! submission it was for was refused, or cancelled, or the recording was
//! abandoned — strands every one of those forever. The ring never recycles, the
//! set is never rewritten, the image is never reacquired, the handle is never
//! destroyed. Nothing errors; the device simply stops making progress in a way
//! that looks like the GPU being slow.
//!
//! This module makes that unrepresentable. [`Reserved`] is not `Clone`, has no
//! public constructor, and every exit consumes it — and the refusal exit does
//! not merely drop the point, it hands back a [`Skip`] obligation that is
//! `#[must_use]` and discharged by host-signalling the semaphore to that value.
//! `vkSignalSemaphore` is exactly the operation for this: a timeline value that
//! no queue will ever produce is produced from the host instead, and everything
//! waiting on it proceeds.
//!
//! # A refused point is retired, never reused
//!
//! The other half of the same rule. After a refusal the caller reserves a
//! *new* point for the retry rather than resubmitting the old one, because the
//! old one may already have been host-signalled and a second submission
//! signalling it again is a counter that goes backwards — which
//! [`crate::timeline`] refuses rather than clamps, for its own good reasons.
//!
//! # Points are consumed in the order they were reserved
//!
//! A timeline counter only goes up, so signalling point 5 makes 4 reached
//! whether or not the work for 4 ever ran. Submitting out of reserved order
//! therefore publishes a completion that has not happened. The ledger admits
//! outcomes in reserved order and refuses anything else without changing state,
//! so two independently prepared callers cannot reorder each other's signals.
//!
//! # The first driver failure is latched
//!
//! A queue that has returned `VK_ERROR_DEVICE_LOST` returns it again, and every
//! subsequent call is noise on top of the one fact that matters. The submitter
//! keeps the first result and reports it; a caller that wants to know whether
//! this device is finished asks once rather than interpreting a stream of
//! secondary errors.
//!
//! Driver acceptance and timeline completion stay separate throughout.
//! `vkQueueSubmit` returning `VK_SUCCESS` means the driver took the work, not
//! that it ran — [`Receipt`] says the first and [`crate::timeline`] says the
//! second.

use reims_vgpu_core::identity::TimelinePoint;

/// A reserved timeline point that has not yet been submitted or given up.
///
/// Not `Clone`, no public constructor, and consumed by both exits — so the
/// point cannot be submitted twice, cannot be both submitted and skipped, and
/// cannot be silently dropped without the compiler saying so.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a reserved point that is neither submitted nor skipped strands everything deferred against it"]
pub struct Reserved {
    point: TimelinePoint,
}

impl Reserved {
    /// The value this submission will signal.
    #[must_use]
    pub const fn point(&self) -> TimelinePoint {
        self.point
    }
}

/// The driver took the work.
///
/// Says nothing about whether it ran. That is the timeline's answer, and
/// keeping the two apart is why this is a distinct type from a completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Receipt {
    point: TimelinePoint,
}

impl Receipt {
    /// The point the accepted submission will signal.
    #[must_use]
    pub const fn point(&self) -> TimelinePoint {
        self.point
    }
}

/// A reserved point whose work will never reach the queue.
///
/// The caller must host-signal the semaphore to [`Self::value`]
/// (`vkSignalSemaphore`) and then call [`Submitter::discharged`]. Until that
/// happens, everything deferred against this point and every point before it is
/// waiting for a counter that no queue is going to produce.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "an undischarged skip leaves the timeline permanently short of this value"]
pub struct Skip {
    point: TimelinePoint,
}

impl Skip {
    /// The value to host-signal the semaphore to.
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.point.0
    }

    #[must_use]
    pub const fn point(&self) -> TimelinePoint {
        self.point
    }
}

/// Why an outcome was not accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutOfOrder {
    /// An outcome arrived for a point that is not the oldest outstanding one.
    /// Nothing was changed.
    NotNext {
        offered: TimelinePoint,
        expected: TimelinePoint,
    },
    /// An outcome arrived with nothing outstanding at all.
    NothingOutstanding { offered: TimelinePoint },
}

impl OutOfOrder {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NotNext { .. } => "vk_submit_out_of_order",
            Self::NothingOutstanding { .. } => "vk_submit_nothing_outstanding",
        }
    }
}

impl std::fmt::Display for OutOfOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotNext { offered, expected } => write!(
                f,
                "{} offered={} expected={}",
                self.slug(),
                offered.0,
                expected.0
            ),
            Self::NothingOutstanding { offered } => {
                write!(f, "{} offered={}", self.slug(), offered.0)
            }
        }
    }
}

/// What the ledger has seen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub reserved: usize,
    pub accepted: usize,
    /// Points given up and host-signalled instead.
    pub skipped: usize,
    /// Outcomes refused for arriving out of reserved order.
    pub out_of_order: usize,
}

/// The ledger of reserved points and what became of each.
///
/// Holds no Vulkan object: the point arithmetic and the ordering rule are what
/// can be wrong, and they are testable without a queue. The caller holds the
/// `VkQueue` — through a [`crate::queues::QueueOwner`], which is what makes the
/// submission order singular — and the semaphore.
///
/// **Not `Clone`.** [`crate::queues::QueueOwner`] is not `Clone` so that a
/// `VkQueue` cannot have two submission orders; a ledger that recorded that
/// order and could be copied puts the second one back. Two copies of a ledger
/// carrying the same outstanding points each believe they are the one that
/// decides what is next, which is the "two independently prepared callers"
/// case the module documentation says cannot reorder each other's signals —
/// and their `undischarged` counts diverge, so the one number that says the
/// timeline is stuck is read off whichever copy the caller happens to hold.
/// [`Reserved`] and [`Skip`] are not `Clone` for the same reason; the ledger
/// they come from is the thing it would have been cheapest to leave copyable
/// and the least obvious to have got wrong.
#[derive(Debug, PartialEq, Eq)]
pub struct Submitter {
    /// Points reserved and not yet resolved, oldest first.
    outstanding: std::collections::VecDeque<TimelinePoint>,
    /// Skips handed out and not yet host-signalled.
    undischarged: usize,
    latched: Option<i32>,
    census: Census,
}

impl Default for Submitter {
    fn default() -> Self {
        Self::new()
    }
}

impl Submitter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            outstanding: std::collections::VecDeque::new(),
            undischarged: 0,
            latched: None,
            census: Census::default(),
        }
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Points reserved and not yet resolved.
    #[must_use]
    pub fn outstanding(&self) -> usize {
        self.outstanding.len()
    }

    /// Skips handed out and not host-signalled yet.
    ///
    /// Non-zero means the timeline is short of a value nothing will produce,
    /// and everything at or after it is waiting.
    #[must_use]
    pub const fn undischarged(&self) -> usize {
        self.undischarged
    }

    /// Take a point reserved from the timeline cursor into this ledger.
    ///
    /// The point comes from [`crate::timeline::TimelineCursor::reserve`],
    /// because that is what owns the counter; this owns what happens to it
    /// afterwards.
    pub fn reserve(&mut self, point: TimelinePoint) -> Reserved {
        self.outstanding.push_back(point);
        self.census.reserved += 1;
        Reserved { point }
    }

    /// The driver accepted the submission.
    ///
    /// # Errors
    ///
    /// [`OutOfOrder`] when this is not the oldest outstanding point. Nothing is
    /// changed, so the caller still owns the reservation — which it gets back
    /// alongside the error.
    pub fn accepted(&mut self, reserved: Reserved) -> Result<Receipt, (Reserved, OutOfOrder)> {
        if let Err(refusal) = self.check_next(reserved.point) {
            self.census.out_of_order += 1;
            return Err((reserved, refusal));
        }
        self.outstanding.pop_front();
        self.census.accepted += 1;
        Ok(Receipt {
            point: reserved.point,
        })
    }

    /// The work for this point will never reach the queue.
    ///
    /// Returns the obligation to host-signal the semaphore to it. `result` is
    /// the driver's, when there was one — a submission the driver refused
    /// latches it here so a caller can ask once whether this device is
    /// finished, rather than reading a stream of secondary errors.
    ///
    /// # Errors
    ///
    /// [`OutOfOrder`] as [`Self::accepted`], with the reservation handed back.
    pub fn refused(
        &mut self,
        reserved: Reserved,
        result: Option<i32>,
    ) -> Result<Skip, (Reserved, OutOfOrder)> {
        if let Err(refusal) = self.check_next(reserved.point) {
            self.census.out_of_order += 1;
            return Err((reserved, refusal));
        }
        self.outstanding.pop_front();
        self.census.skipped += 1;
        self.undischarged += 1;
        if self.latched.is_none() {
            self.latched = result;
        }
        Ok(Skip {
            point: reserved.point,
        })
    }

    /// The semaphore has been host-signalled to the skipped value.
    pub fn discharged(&mut self, skip: Skip) -> TimelinePoint {
        self.undischarged -= 1;
        skip.point
    }

    /// The first driver failure this submitter saw, if any.
    ///
    /// Latched: a queue that has returned `VK_ERROR_DEVICE_LOST` returns it
    /// again, and only the first one is a fact about what happened.
    #[must_use]
    pub const fn latched(&self) -> Option<i32> {
        self.latched
    }

    /// Whether the device has failed and no further submission should be made.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.latched.is_some()
    }

    /// The next point an outcome may be offered for.
    #[must_use]
    pub fn expected(&self) -> Option<TimelinePoint> {
        self.outstanding.front().copied()
    }

    fn check_next(&self, offered: TimelinePoint) -> Result<(), OutOfOrder> {
        match self.outstanding.front() {
            None => Err(OutOfOrder::NothingOutstanding { offered }),
            Some(&expected) if expected != offered => {
                Err(OutOfOrder::NotNext { offered, expected })
            }
            Some(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::TimelineCursor;

    fn at(n: u64) -> TimelinePoint {
        TimelinePoint(n)
    }

    #[test]
    fn an_accepted_submission_yields_a_receipt_and_nothing_more() {
        let mut submitter = Submitter::new();
        let reserved = submitter.reserve(at(1));
        assert_eq!(reserved.point(), at(1));
        assert_eq!(submitter.outstanding(), 1);

        let receipt = submitter.accepted(reserved).expect("the oldest point");
        assert_eq!(receipt.point(), at(1));
        assert_eq!(submitter.outstanding(), 0);
        assert_eq!(submitter.census().accepted, 1);
        // Acceptance is not completion: nothing here says the GPU ran it.
        assert!(!submitter.is_finished());
    }

    /// The claim the module exists for.
    #[test]
    fn a_refused_point_becomes_an_obligation_and_not_a_dropped_value() {
        let mut submitter = Submitter::new();
        let reserved = submitter.reserve(at(4));
        let skip = submitter
            .refused(reserved, Some(-4))
            .expect("the oldest point");
        assert_eq!(skip.value(), 4);
        assert_eq!(
            submitter.undischarged(),
            1,
            "the timeline is short of a value nothing will produce"
        );
        assert_eq!(submitter.latched(), Some(-4));
        assert!(submitter.is_finished());

        let point = submitter.discharged(skip);
        assert_eq!(point, at(4));
        assert_eq!(submitter.undischarged(), 0);
        assert_eq!(submitter.census().skipped, 1);
    }

    /// Host-signalling the skipped value is what makes every deferral against
    /// it proceed — including deferrals the skipped work never touched.
    #[test]
    fn a_discharged_skip_lets_the_cursor_reach_the_value() {
        let mut cursor = TimelineCursor::new();
        let mut submitter = Submitter::new();

        let first = submitter.reserve(cursor.reserve());
        let second = submitter.reserve(cursor.reserve());
        assert_eq!(first.point(), at(1));
        assert_eq!(second.point(), at(2));

        // The first submission is refused, so nothing on the queue will ever
        // produce 1 — and 2, submitted after it, would.
        let skip = submitter.refused(first, None).expect("oldest");
        let receipt = submitter.accepted(second).expect("now oldest");

        // The host signals 1. A cursor that observes it has reached the value
        // everything deferred against 1 was waiting for.
        submitter.discharged(skip);
        assert_eq!(cursor.observe(1).expect("forwards"), at(1));
        assert!(cursor.has_reached(at(1)));
        assert!(!cursor.has_reached(receipt.point()));
    }

    #[test]
    fn an_outcome_out_of_reserved_order_changes_nothing() {
        let mut submitter = Submitter::new();
        let first = submitter.reserve(at(1));
        let second = submitter.reserve(at(2));

        let (returned, refusal) = submitter
            .accepted(second)
            .expect_err("the older point has not been resolved");
        assert_eq!(
            refusal,
            OutOfOrder::NotNext {
                offered: at(2),
                expected: at(1)
            }
        );
        assert_eq!(refusal.slug(), "vk_submit_out_of_order");
        assert!(refusal.to_string().contains("expected=1"));

        // Nothing moved, and the caller still owns both reservations.
        assert_eq!(submitter.outstanding(), 2);
        assert_eq!(submitter.census().accepted, 0);
        assert_eq!(submitter.census().out_of_order, 1);

        // In order, both land.
        submitter.accepted(first).expect("oldest");
        submitter.accepted(returned).expect("now oldest");
        assert_eq!(submitter.outstanding(), 0);
    }

    #[test]
    fn an_outcome_with_nothing_outstanding_is_its_own_reading() {
        let mut submitter = Submitter::new();
        let reserved = submitter.reserve(at(9));
        submitter.accepted(reserved).expect("oldest");

        // Fabricate the situation the type system otherwise prevents: a second
        // ledger's reservation offered here.
        let mut other = Submitter::new();
        let stranger = other.reserve(at(9));
        let (_returned, refusal) = submitter
            .accepted(stranger)
            .expect_err("this ledger has nothing outstanding");
        assert_eq!(refusal, OutOfOrder::NothingOutstanding { offered: at(9) });
        assert_eq!(refusal.slug(), "vk_submit_nothing_outstanding");
    }

    #[test]
    fn only_the_first_driver_failure_is_latched() {
        let mut submitter = Submitter::new();
        for (point, result) in [(1u64, Some(-4)), (2, Some(-2)), (3, None)] {
            let reserved = submitter.reserve(at(point));
            let skip = submitter.refused(reserved, result).expect("oldest");
            submitter.discharged(skip);
        }
        assert_eq!(
            submitter.latched(),
            Some(-4),
            "the first result is the fact; the rest are its echoes"
        );
        assert_eq!(submitter.census().skipped, 3);
        assert_eq!(submitter.undischarged(), 0);
    }

    /// A refusal with no driver result — a cancellation before the call — does
    /// not make the device look lost.
    #[test]
    fn a_cancellation_is_not_a_driver_failure() {
        let mut submitter = Submitter::new();
        let reserved = submitter.reserve(at(1));
        let skip = submitter.refused(reserved, None).expect("oldest");
        submitter.discharged(skip);
        assert_eq!(submitter.latched(), None);
        assert!(!submitter.is_finished());
        // But the point was still owed and still discharged.
        assert_eq!(submitter.census().skipped, 1);
    }

    /// Every reserved point leaves the ledger through exactly one of the two
    /// exits, whatever order the outcomes are attempted in.
    #[test]
    fn every_reserved_point_is_resolved_exactly_once() {
        let mut cursor = TimelineCursor::new();
        let mut submitter = Submitter::new();
        let mut held: Vec<Reserved> = (0..8)
            .map(|_| submitter.reserve(cursor.reserve()))
            .collect();
        assert_eq!(submitter.outstanding(), 8);

        // Attempt them newest-first, so every round tries several out-of-order
        // outcomes before reaching the one that can land. Progress is possible
        // only in reserved order, which is the claim.
        let mut resolved = Vec::new();
        let mut round = 0usize;
        while !held.is_empty() {
            let mut kept: Vec<Reserved> = Vec::new();
            let mut landed = None;
            while let Some(reservation) = held.pop() {
                // Alternate the two exits so both are exercised.
                let use_accept = round.is_multiple_of(2);
                let outcome = if use_accept {
                    submitter.accepted(reservation).map(|r| r.point())
                } else {
                    submitter.refused(reservation, None).map(|skip| {
                        let point = skip.point();
                        submitter.discharged(skip);
                        point
                    })
                };
                match outcome {
                    Ok(point) => {
                        landed = Some(point);
                        break;
                    }
                    Err((reservation, refusal)) => {
                        assert!(matches!(refusal, OutOfOrder::NotNext { .. }));
                        kept.push(reservation);
                    }
                }
            }
            let point = landed.expect("the oldest point is always resolvable");
            resolved.push(point);
            // The landed one was the oldest, so nothing older was left
            // unpopped and `held` is empty. `kept` is newest-first from the
            // pops, so reversing it restores reserved order.
            debug_assert!(held.is_empty());
            kept.reverse();
            held = kept;
            round += 1;
        }

        assert_eq!(submitter.outstanding(), 0);
        assert_eq!(submitter.undischarged(), 0);
        assert_eq!(resolved.len(), 8);
        // Resolved in reserved order, and every value distinct.
        let values: Vec<u64> = resolved.iter().map(|p| p.0).collect();
        assert_eq!(values, (1..=8).collect::<Vec<u64>>());
        assert_eq!(
            submitter.census().accepted + submitter.census().skipped,
            submitter.census().reserved
        );
        assert!(
            submitter.census().out_of_order > 0,
            "the sweep never actually attempted an out-of-order outcome"
        );
    }

    #[test]
    fn a_retry_after_a_refusal_takes_a_new_point() {
        let mut cursor = TimelineCursor::new();
        let mut submitter = Submitter::new();
        let first = submitter.reserve(cursor.reserve());
        let point = first.point();
        let skip = submitter.refused(first, None).expect("oldest");
        submitter.discharged(skip);

        // The retry reserves again rather than resubmitting the old value: the
        // old one has been host-signalled, and signalling it again is a counter
        // that goes backwards.
        let retry = submitter.reserve(cursor.reserve());
        assert!(retry.point() > point);
        submitter.accepted(retry).expect("oldest");
    }
}
