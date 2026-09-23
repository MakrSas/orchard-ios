//! The two lifetimes a native object lives under, and when it stops living.
//!
//! # Two lifetimes, two questions, never one answer
//!
//! Every native object a transaction holds is named by a [`Lifetime`]: the
//! [`SessionGeneration`] it was created under and the [`DeviceEpoch`] whose
//! handles it is made of. They answer different questions and they end for
//! different reasons.
//!
//! - A **guest reset** closes the semantic generation. The guest may no longer
//!   name what it named; accepted work still has to finish, and the host device
//!   is untouched. Nothing is destroyed here, and a reset that tore down a
//!   healthy device would be a reset that throws away work the host is still
//!   executing.
//! - **Device loss** ends the epoch. Every handle from it is gone at once,
//!   whatever the guest still names, and no timeline will ever advance to say
//!   so — the thing that would advance it is what was lost.
//!
//! Collapsing the two into one counter produces exactly two failures: a reset
//! that destroys a working device, and a device loss that leaves dead handles
//! reachable under a live name. So [`Validity`] has four answers rather than
//! two, and a caller has to say which question it is asking.
//!
//! # Retirement is exact, not swept
//!
//! An object retires when the timeline point of its **last use** is reached,
//! and not before. That is a completion fact the executor delivers, not a
//! deadline, a cache bound, or a periodic sweep — a sweep either destroys
//! objects the GPU is still reading or keeps ones nothing needs, and it cannot
//! tell you which it just did. [`NativeRetirement::reached`] does exactly one
//! thing: it hands back the objects whose last use is now in the past.
//!
//! # Retired and abandoned are different types on purpose
//!
//! An object whose last use completed still has live handles and **must** be
//! destroyed through the device. An object whose epoch was lost has no handles
//! left to destroy and must be dropped without being touched. Calling the
//! wrong one is a driver crash, so they are [`Retired`] and [`Abandoned`] and
//! the type system will not let a caller confuse them.
//!
//! # What the object is, is not this crate's business
//!
//! The queue is generic in the payload. This crate owns *when* a native object
//! dies; the executor owns *what* it is. A `T` that had to be nameable here
//! would be a Vulkan handle in the semantic model.

use crate::identity::{DeviceEpoch, SessionGeneration, TimelinePoint};

/// The pair of lifetimes every native object lease carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lifetime {
    /// Decides whether the guest may still name this.
    pub session: SessionGeneration,
    /// Decides whether its handles may still be touched.
    pub epoch: DeviceEpoch,
}

/// What is left of a lease, against the lifetimes that are current now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Validity {
    /// Both lifetimes are current: the guest may name it and its handles work.
    Live,
    /// The generation closed. The guest may not name it; the handles are still
    /// real and the work holding them still has to finish.
    SemanticallyClosed,
    /// The epoch ended. The handles are gone; a guest that still names it is
    /// naming something that cannot be executed.
    HandlesUnusable,
    /// Both. Nothing is left but the obligation to drop it.
    Gone,
}

impl Validity {
    /// Whether new work may be accepted against this lease.
    #[must_use]
    pub const fn admits_new_work(self) -> bool {
        matches!(self, Self::Live)
    }

    /// Whether a native handle from this lease may be touched — submitted,
    /// recorded into, or destroyed.
    ///
    /// True for a closed generation, which is the whole point of the split:
    /// accepted work outlives the guest's ability to name it.
    #[must_use]
    pub const fn handles_usable(self) -> bool {
        matches!(self, Self::Live | Self::SemanticallyClosed)
    }

    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Live => "lease_live",
            Self::SemanticallyClosed => "lease_semantically_closed",
            Self::HandlesUnusable => "lease_handles_unusable",
            Self::Gone => "lease_gone",
        }
    }
}

impl Lifetime {
    #[must_use]
    pub const fn new(session: SessionGeneration, epoch: DeviceEpoch) -> Self {
        Self { session, epoch }
    }

    /// What is left of this lease against the current lifetimes.
    ///
    /// Two independent comparisons, deliberately not short-circuited into one:
    /// asking "is it stale" with a single test is what makes a reset destroy a
    /// device or a loss leave handles reachable.
    #[must_use]
    pub const fn against(self, session: SessionGeneration, epoch: DeviceEpoch) -> Validity {
        match (
            self.session.get() == session.get(),
            self.epoch.get() == epoch.get(),
        ) {
            (true, true) => Validity::Live,
            (false, true) => Validity::SemanticallyClosed,
            (true, false) => Validity::HandlesUnusable,
            (false, false) => Validity::Gone,
        }
    }
}

/// An object whose last use has completed. Its handles are live and it must be
/// destroyed through the device that made them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a retired object nothing destroys is a handle the device never gets back"]
pub struct Retired<T> {
    pub lifetime: Lifetime,
    pub last_use: TimelinePoint,
    pub object: T,
}

/// An object whose device incarnation ended. Its handles went with it and it
/// must be dropped without being touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "an abandoned object is still the caller's to drop, and to drop without touching"]
pub struct Abandoned<T> {
    pub lifetime: Lifetime,
    pub object: T,
}

#[derive(Clone, Copy, Debug)]
struct Entry<T> {
    lifetime: Lifetime,
    last_use: TimelinePoint,
    object: T,
}

/// Native objects waiting for the point at which they stop being in use.
///
/// Holds nothing but what was handed to it, evicts on no bound of its own, and
/// releases only on an exact completion fact. A cap here would be a cache
/// bound deciding when the GPU is finished with something.
#[derive(Debug)]
pub struct NativeRetirement<T> {
    pending: Vec<Entry<T>>,
    retired: usize,
    abandoned: usize,
}

impl<T> Default for NativeRetirement<T> {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            retired: 0,
            abandoned: 0,
        }
    }
}

impl<T> NativeRetirement<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Hand an object over, with the lifetimes it belongs to and the timeline
    /// point of its last use.
    ///
    /// `last_use` is the point *after* which nothing reads it. A caller that
    /// does not know one has not finished establishing the object's use and
    /// must not be queueing it.
    pub fn queue(&mut self, lifetime: Lifetime, last_use: TimelinePoint, object: T) {
        self.pending.push(Entry {
            lifetime,
            last_use,
            object,
        });
    }

    /// One epoch's timeline reached `at`: take the objects nothing reads any
    /// more.
    ///
    /// Scoped to an epoch because a timeline belongs to one. Two epochs' points
    /// are unrelated numbers, and comparing one against the other would retire
    /// live objects whenever a new device's timeline started behind an old
    /// one's.
    ///
    /// `#[must_use]` on the method and not only on [`Retired`]: the lint does
    /// not look inside a `Vec`, so the annotation on the element type says
    /// nothing at all about a call whose result is dropped — which is the exact
    /// call that leaks every handle it just took out of the queue.
    #[must_use = "objects taken out of the queue and not destroyed are handles nothing will ever destroy"]
    pub fn reached(&mut self, epoch: DeviceEpoch, at: TimelinePoint) -> Vec<Retired<T>> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].lifetime.epoch == epoch && at.reached(self.pending[i].last_use) {
                let entry = self.pending.swap_remove(i);
                out.push(Retired {
                    lifetime: entry.lifetime,
                    last_use: entry.last_use,
                    object: entry.object,
                });
            } else {
                i += 1;
            }
        }
        self.retired += out.len();
        out
    }

    /// An epoch ended: take everything that belonged to it, at once.
    ///
    /// No timeline is consulted, because the thing that would advance it is
    /// what was lost. Waiting for a point a dead device will never reach is how
    /// a device loss becomes a leak and a hang instead of a transition.
    ///
    /// `#[must_use]` for the reason [`Self::reached`] gives.
    #[must_use = "objects taken out of the queue are the caller's to drop"]
    pub fn epoch_lost(&mut self, epoch: DeviceEpoch) -> Vec<Abandoned<T>> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].lifetime.epoch == epoch {
                let entry = self.pending.swap_remove(i);
                out.push(Abandoned {
                    lifetime: entry.lifetime,
                    object: entry.object,
                });
            } else {
                i += 1;
            }
        }
        self.abandoned += out.len();
        out
    }

    /// Objects still in use.
    #[must_use]
    pub fn outstanding(&self) -> usize {
        self.pending.len()
    }

    /// Objects held for one epoch.
    #[must_use]
    pub fn outstanding_in(&self, epoch: DeviceEpoch) -> usize {
        self.pending
            .iter()
            .filter(|e| e.lifetime.epoch == epoch)
            .count()
    }

    /// Objects released so far: destroyed, and abandoned.
    #[must_use]
    pub const fn census(&self) -> (usize, usize) {
        (self.retired, self.abandoned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lifetime(session: u64, epoch: u64) -> Lifetime {
        let mut s = SessionGeneration::FIRST;
        for _ in 1..session {
            s = s.next();
        }
        let mut e = DeviceEpoch::FIRST;
        for _ in 1..epoch {
            e = e.next();
        }
        Lifetime::new(s, e)
    }

    fn at(n: u64) -> TimelinePoint {
        TimelinePoint(n)
    }

    /// The four answers, and the two that are not "stale".
    #[test]
    fn a_lease_is_judged_on_two_lifetimes_and_not_on_one() {
        let l = lifetime(1, 1);
        let (s2, e2) = (lifetime(2, 2).session, lifetime(2, 2).epoch);
        assert_eq!(
            l.against(l.session, l.epoch),
            Validity::Live,
            "both current"
        );
        assert_eq!(
            l.against(s2, l.epoch),
            Validity::SemanticallyClosed,
            "a guest reset closed the generation; the device is untouched"
        );
        assert_eq!(
            l.against(l.session, e2),
            Validity::HandlesUnusable,
            "the device was lost; the guest still names it"
        );
        assert_eq!(l.against(s2, e2), Validity::Gone);
    }

    /// The claim the split exists for: accepted work outlives the guest's
    /// ability to name it.
    #[test]
    fn a_closed_generation_still_has_usable_handles() {
        let l = lifetime(1, 1);
        let closed = l.against(lifetime(2, 1).session, l.epoch);
        assert!(!closed.admits_new_work(), "the guest may not name it");
        assert!(
            closed.handles_usable(),
            "and the submission the host is still executing must finish"
        );
    }

    #[test]
    fn a_lost_epoch_admits_nothing_and_touches_nothing() {
        let l = lifetime(1, 1);
        let lost = l.against(l.session, lifetime(1, 2).epoch);
        assert!(!lost.admits_new_work());
        assert!(!lost.handles_usable());
    }

    /// Exact deferred retirement: at its point, and not before it.
    #[test]
    fn an_object_retires_at_its_last_use_and_not_before() {
        let mut r = NativeRetirement::new();
        r.queue(lifetime(1, 1), at(10), "buffer");
        assert!(
            r.reached(DeviceEpoch::FIRST, at(9)).is_empty(),
            "nine is not ten, and the GPU is still reading it"
        );
        assert_eq!(r.outstanding(), 1);
        let out = r.reached(DeviceEpoch::FIRST, at(10));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].object, "buffer");
        assert_eq!(r.outstanding(), 0);
        assert_eq!(r.census(), (1, 0));
    }

    /// A timeline belongs to one epoch, so a point from another says nothing.
    #[test]
    fn one_epochs_timeline_does_not_retire_another_epochs_objects() {
        let mut r = NativeRetirement::new();
        r.queue(lifetime(1, 1), at(10), "old");
        r.queue(lifetime(1, 2), at(1), "new");
        let out = r.reached(lifetime(1, 2).epoch, at(5));
        assert_eq!(
            out.iter().map(|o| o.object).collect::<Vec<_>>(),
            vec!["new"],
            "the new device's timeline started behind the old one's and says \
             nothing about it"
        );
        assert_eq!(r.outstanding_in(lifetime(1, 1).epoch), 1);
    }

    /// Device loss does not wait for a point the dead device will never reach.
    #[test]
    fn a_lost_epoch_releases_everything_it_held_without_a_timeline() {
        let mut r = NativeRetirement::new();
        r.queue(lifetime(1, 1), at(u64::MAX), "never completes");
        r.queue(lifetime(2, 1), at(u64::MAX), "nor this");
        r.queue(lifetime(1, 2), at(1), "another device");
        let gone = r.epoch_lost(DeviceEpoch::FIRST);
        assert_eq!(gone.len(), 2);
        assert_eq!(
            r.outstanding(),
            1,
            "and the other device's objects are untouched"
        );
        assert_eq!(r.census(), (0, 2));
    }

    /// A reset retires nothing. The device is fine; the work is still running.
    #[test]
    fn closing_a_generation_retires_no_native_object() {
        let mut r = NativeRetirement::new();
        r.queue(lifetime(1, 1), at(10), "buffer");
        // There is no `session_closed` to call, and that is the point: nothing
        // in this queue is keyed on a generation, so a reset has no way to
        // release anything and cannot be made to by accident.
        assert_eq!(r.outstanding(), 1);
        assert_eq!(r.reached(DeviceEpoch::FIRST, at(10)).len(), 1);
    }

    /// Retired and abandoned are different types, so the destroy path cannot
    /// be handed a handle that no longer exists.
    #[test]
    fn what_may_be_destroyed_and_what_may_only_be_dropped_are_different_types() {
        fn destroy(_: Retired<&'static str>) {}
        let mut r = NativeRetirement::new();
        r.queue(lifetime(1, 1), at(1), "live");
        for object in r.reached(DeviceEpoch::FIRST, at(1)) {
            destroy(object);
        }
        r.queue(lifetime(1, 1), at(1), "dead");
        for object in r.epoch_lost(DeviceEpoch::FIRST) {
            // `destroy(object)` does not compile here, which is the test.
            assert_eq!(object.object, "dead");
        }
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

    /// Deliberately few epochs and a short timeline, so an object queued under
    /// one epoch is constantly asked about by another's points and the two
    /// timelines overlap numerically.
    const EPOCHS: u64 = 3;
    const POINTS: u64 = 6;

    /// **Every queued object leaves exactly once, and never before its last
    /// use — on its own epoch's timeline.**
    ///
    /// The shadow is a flat list of `(epoch, last_use, id)` with a released
    /// set beside it, which has no notion of a timeline at all: what may be
    /// taken is stated as the predicate "this epoch, and this point is at or
    /// past the last use", and what a loss takes is stated as "this epoch",
    /// full stop. A queue that consulted a timeline on a device-loss path, or
    /// compared one epoch's point against another's, cannot agree with it.
    #[test]
    fn every_queued_object_leaves_once_at_its_own_epochs_point() {
        let mut retired = 0usize;
        let mut abandoned = 0usize;
        let mut sweeps_that_took_nothing = 0usize;
        let mut losses_that_took_nothing = 0usize;
        let mut queued_after_a_loss = 0usize;

        for seed in 0..512u64 {
            let mut rng = Rng::new(seed);
            let mut r: NativeRetirement<u64> = NativeRetirement::new();
            // Shadow: what is still held, and what has left.
            let mut held: Vec<(DeviceEpoch, TimelinePoint, u64)> = Vec::new();
            let mut left: Vec<u64> = Vec::new();
            let mut lost: Vec<DeviceEpoch> = Vec::new();
            let mut next_id = 0u64;
            let mut retired_here = 0usize;
            let mut abandoned_here = 0usize;

            for _ in 0..48 {
                let e = epoch(rng.below(EPOCHS) + 1);
                match rng.below(6) {
                    // Queue an object.
                    0..=2 => {
                        let point = at(rng.below(POINTS) + 1);
                        let id = next_id;
                        next_id += 1;
                        if lost.contains(&e) {
                            queued_after_a_loss += 1;
                        }
                        r.queue(Lifetime::new(session(rng.below(3) + 1), e), point, id);
                        held.push((e, point, id));
                    }
                    // One epoch's timeline reached a point.
                    3..=4 => {
                        let point = at(rng.below(POINTS + 2));
                        let got = r.reached(e, point);
                        let mut expected: Vec<u64> = held
                            .iter()
                            .filter(|(he, hp, _)| *he == e && point.reached(*hp))
                            .map(|(_, _, id)| *id)
                            .collect();
                        let mut names: Vec<u64> = got.iter().map(|o| o.object).collect();
                        expected.sort_unstable();
                        names.sort_unstable();
                        assert_eq!(names, expected, "seed {seed}: reached({e:?}, {point:?})");
                        for o in &got {
                            assert_eq!(o.lifetime.epoch, e, "seed {seed}: wrong epoch retired");
                            assert!(
                                point.reached(o.last_use),
                                "seed {seed}: {:?} retired before its last use",
                                o.object
                            );
                            assert!(
                                !left.contains(&o.object),
                                "seed {seed}: {} left twice",
                                o.object
                            );
                            left.push(o.object);
                        }
                        held.retain(|(he, hp, _)| !(*he == e && point.reached(*hp)));
                        if got.is_empty() {
                            sweeps_that_took_nothing += 1;
                        }
                        retired += got.len();
                        retired_here += got.len();
                    }
                    // An epoch ended.
                    _ => {
                        let got = r.epoch_lost(e);
                        let mut expected: Vec<u64> = held
                            .iter()
                            .filter(|(he, _, _)| *he == e)
                            .map(|(_, _, id)| *id)
                            .collect();
                        let mut names: Vec<u64> = got.iter().map(|o| o.object).collect();
                        expected.sort_unstable();
                        names.sort_unstable();
                        assert_eq!(names, expected, "seed {seed}: epoch_lost({e:?})");
                        for o in &got {
                            assert_eq!(o.lifetime.epoch, e, "seed {seed}: wrong epoch abandoned");
                            assert!(
                                !left.contains(&o.object),
                                "seed {seed}: {} left twice",
                                o.object
                            );
                            left.push(o.object);
                        }
                        held.retain(|(he, _, _)| *he != e);
                        if got.is_empty() {
                            losses_that_took_nothing += 1;
                        }
                        if !lost.contains(&e) {
                            lost.push(e);
                        }
                        abandoned += got.len();
                        abandoned_here += got.len();
                    }
                }

                // The observers agree with the shadow after every step.
                assert_eq!(r.outstanding(), held.len(), "seed {seed}: outstanding");
                for n in 1..=EPOCHS {
                    let e = epoch(n);
                    assert_eq!(
                        r.outstanding_in(e),
                        held.iter().filter(|(he, _, _)| *he == e).count(),
                        "seed {seed}: outstanding_in({e:?})"
                    );
                }
                assert_eq!(
                    r.census(),
                    (retired_here, abandoned_here),
                    "seed {seed}: census"
                );
                assert_eq!(
                    left.len() + held.len(),
                    next_id as usize,
                    "seed {seed}: an object is neither held nor gone"
                );
            }
        }

        // Non-vacuity: every shape an assertion above depends on reaching.
        assert!(retired > 2_000, "objects retired: {retired}");
        assert!(abandoned > 1_000, "objects abandoned: {abandoned}");
        assert!(
            sweeps_that_took_nothing > 1_000,
            "timeline points that released nothing: {sweeps_that_took_nothing}"
        );
        assert!(
            losses_that_took_nothing > 500,
            "losses of an epoch holding nothing: {losses_that_took_nothing}"
        );
        assert!(
            queued_after_a_loss > 300,
            "objects queued under an epoch already lost: {queued_after_a_loss}"
        );
    }

    fn session(n: u64) -> SessionGeneration {
        let mut s = SessionGeneration::FIRST;
        for _ in 1..n {
            s = s.next();
        }
        s
    }

    fn epoch(n: u64) -> DeviceEpoch {
        let mut e = DeviceEpoch::FIRST;
        for _ in 1..n {
            e = e.next();
        }
        e
    }
}
