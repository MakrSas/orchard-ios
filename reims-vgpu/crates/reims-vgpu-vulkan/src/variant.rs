//! One semantic pipeline, many native pipelines, and who is allowed to compile
//! which.
//!
//! # Why a semantic pipeline is not a native pipeline
//!
//! [`reims_vgpu_core::pipeline`] owns when a guest pipeline object exists and
//! when a transaction may use it. That is one object with one lifetime. A
//! Vulkan pipeline is not: the same guest pipeline drawn into two different
//! attachment sets, at two sample counts, or with two different specialized
//! shader programs is two `VkPipeline` values, and there is no way to know how
//! many until the draws arrive.
//!
//! So a ready semantic pipeline owns a *family* of native variants keyed by
//! everything that has to be baked in. The key is derived where the draw is
//! resolved; this module owns what happens to it afterwards.
//!
//! # One flight per key, and joining is not compiling again
//!
//! Two workers resolving the same draw at the same time must not both compile
//! it: pipeline compilation is the most expensive thing this rail does, and
//! doing it twice also produces two handles where the caller expects one. So a
//! key admits exactly one compile flight. [`VariantFamily::begin_flight`]
//! hands out a [`Flight`] the first time and `None` after, and `None` means
//! "somebody else is on it" rather than "no". The second caller waits on
//! readiness like any other.
//!
//! The flight is unforgeable: not `Clone`, no public constructor, and carrying
//! the family it came from. A flight published into the wrong family is refused
//! **without mutating either one**, because a family that accepted a stranger's
//! compilation would be publishing handles from a device epoch it does not own.
//!
//! # Refused is a state, not an error return
//!
//! A variant this host cannot build stays refused with its reason. A guest
//! drawing it every frame produces one refusal rather than one per frame, and
//! the reason survives to whoever reads the failure channel — the same rule
//! [`reims_vgpu_core::pipeline`] applies one level up, for the same reason.
//!
//! # Nothing live is ever evicted
//!
//! There is no eviction method, no capacity bound and no LRU. A variant the
//! guest can still reach is state that represents guest work, and the contract
//! provides no lawful loss of it; a cache bound here would silently recompile
//! under load, which is the stutter nobody can reproduce. Growth is bounded by
//! the guest's own distinct draw configurations.
//!
//! Retirement is therefore the only way a variant goes away, and even that does
//! not free one that submitted work still names: an acquired [`Variant`] keeps
//! its native objects alive past the family's own end, so a command buffer
//! already recorded against a `VkPipeline` cannot have it destroyed underneath
//! it. [`VariantFamily::collect`] hands back exactly the ones nobody holds.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Which family a flight belongs to.
///
/// Monotone and never reused, so a token from a family that has already ended
/// is recognisably foreign rather than accidentally matching a later one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FamilyId(u64);

impl FamilyId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        // Relaxed: uniqueness is the only requirement, and the ordering a
        // family takes part in is its owner's.
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for FamilyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "family{}", self.0)
    }
}

/// The exclusive right to compile one key for one family.
///
/// Not `Clone`, and constructible only by [`VariantFamily::begin_flight`].
/// Consumed by [`VariantFamily::publish`], so one key cannot be published
/// twice and a compilation cannot be started twice.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a flight taken and not published leaves the key compiling forever"]
pub struct Flight<K> {
    family: FamilyId,
    key: K,
}

impl<K> Flight<K> {
    /// The key being compiled, so the compiler knows what to build.
    pub const fn key(&self) -> &K {
        &self.key
    }

    #[must_use]
    pub const fn family(&self) -> FamilyId {
        self.family
    }
}

/// An acquired native variant.
///
/// Keeps its native objects alive for as long as it is held, including past
/// the family's retirement — which is what makes a recorded command buffer safe
/// against a guest that deleted the pipeline it names.
#[derive(Debug)]
pub struct Variant<V> {
    inner: Arc<V>,
}

impl<V> Variant<V> {
    #[must_use]
    pub fn get(&self) -> &V {
        &self.inner
    }
}

impl<V> Clone for Variant<V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<V> std::ops::Deref for Variant<V> {
    type Target = V;

    fn deref(&self) -> &V {
        &self.inner
    }
}

/// What a caller asking for a key learns.
///
/// `Clone` and `PartialEq` are written out rather than derived: a derive would
/// bound them on `V`, and `V` is a native pipeline that is deliberately neither.
/// A [`Variant`] is cloneable and comparable whatever it wraps, because both
/// are about the allocation and not its contents.
#[derive(Debug)]
pub enum Readiness<V, R> {
    /// Nobody has asked for this key yet. The caller may take a flight.
    Absent,
    /// The family's semantic generation has ended and this key was never
    /// compiled, so no flight will ever be granted for it.
    ///
    /// Distinct from [`Self::Absent`] because the two lead a caller to
    /// opposite actions and [`VariantFamily::begin_flight`] cannot tell them
    /// apart: its `None` means "somebody else is on it" for a key that is
    /// already an entry and "never" for a retired family, and a caller reading
    /// `Absent`, taking no flight and then waiting for readiness waits for a
    /// compile that nothing will start.
    Retired,
    /// A flight is outstanding. The caller waits; it does not start a second.
    Compiling,
    Ready(Variant<V>),
    /// Terminal, with the reason.
    Refused(R),
}

impl<V, R: Clone> Clone for Readiness<V, R> {
    fn clone(&self) -> Self {
        match self {
            Self::Absent => Self::Absent,
            Self::Retired => Self::Retired,
            Self::Compiling => Self::Compiling,
            Self::Ready(v) => Self::Ready(v.clone()),
            Self::Refused(r) => Self::Refused(r.clone()),
        }
    }
}

impl<V, R: PartialEq> PartialEq for Readiness<V, R> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Absent, Self::Absent)
            | (Self::Retired, Self::Retired)
            | (Self::Compiling, Self::Compiling) => true,
            (Self::Ready(a), Self::Ready(b)) => a == b,
            (Self::Refused(a), Self::Refused(b)) => a == b,
            _ => false,
        }
    }
}

impl<V, R: Eq> Eq for Readiness<V, R> {}

impl<V, R> Readiness<V, R> {
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }

    /// Whether a caller should take a flight for this key.
    ///
    /// Only `Absent`. `Compiling` is somebody else's flight, `Refused` is
    /// terminal — a retry there is the per-frame refusal storm the module doc
    /// describes — and `Retired` is terminal for the whole family.
    #[must_use]
    pub const fn wants_a_flight(&self) -> bool {
        matches!(self, Self::Absent)
    }
}

impl<V> PartialEq for Variant<V> {
    fn eq(&self, other: &Self) -> bool {
        // Identity, not contents: two variants are the same variant when they
        // are the same allocation. Comparing contents would call a native
        // handle equal to a different handle with the same fields.
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<V> Eq for Variant<V> {}

/// Publishing into a family the flight did not come from, with everything the
/// caller handed over given back.
///
/// The compiled value comes back because it is a native object nobody else has
/// a name for: dropping it here would leak a `VkPipeline` on a path that is
/// already a caller bug, and a leak is the one failure that does not show up in
/// the run that caused it. The flight comes back for the same reason — it is
/// the right to compile that key, it is not `Clone`, and consuming it would
/// leave the key it names stuck in `Compiling` forever.
#[derive(Debug)]
pub struct Misdirected<K, V, R> {
    pub wrong: WrongFamily,
    pub flight: Flight<K>,
    pub outcome: Result<V, R>,
}

impl<K, V, R> std::fmt::Display for Misdirected<K, V, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.wrong, f)
    }
}

/// Publishing into a family the flight did not come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrongFamily {
    pub flight: FamilyId,
    pub family: FamilyId,
}

impl WrongFamily {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        "vk_variant_wrong_family"
    }
}

impl std::fmt::Display for WrongFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} flight={} family={}",
            self.slug(),
            self.flight,
            self.family
        )
    }
}

/// What the family has been asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Keys that were already ready.
    pub hits: usize,
    /// Keys that were not, so somebody had to compile.
    pub misses: usize,
    /// Requests that found a flight already outstanding. The number that says
    /// what the single-flight rule is saving.
    pub joined: usize,
    pub ready: usize,
    pub refused: usize,
    /// Publications refused for naming another family.
    pub foreign: usize,
}

/// Every native variant of one semantic pipeline generation.
///
/// Not `Clone`: it is the owner of the variants' lifetimes.
#[derive(Debug)]
pub struct VariantFamily<K, V, R> {
    id: FamilyId,
    entries: HashMap<K, Entry<V, R>>,
    retired: bool,
    census: Census,
}

#[derive(Debug)]
enum Entry<V, R> {
    Compiling,
    Ready(Arc<V>),
    Refused(R),
}

impl<K: Eq + Hash + Clone + Debug, V, R: Clone> VariantFamily<K, V, R> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: FamilyId::next(),
            entries: HashMap::new(),
            retired: false,
            census: Census::default(),
        }
    }

    #[must_use]
    pub const fn id(&self) -> FamilyId {
        self.id
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    #[must_use]
    pub const fn is_retired(&self) -> bool {
        self.retired
    }

    /// How many keys the family holds, ready or otherwise.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Ask about a key without changing anything.
    ///
    /// A key with no entry in a retired family answers [`Readiness::Retired`]
    /// and not [`Readiness::Absent`]: `Absent` invites a flight, and
    /// [`Self::begin_flight`] refuses every one a retired family is asked for,
    /// so the caller that followed the protocol would wait on a compile
    /// nothing was ever going to start. Entries the family already holds
    /// answer for themselves — a variant somebody compiled before the
    /// retirement is still the reason a recorded command buffer is safe, and a
    /// refusal's reason still has to reach whoever reads it.
    #[must_use]
    pub fn peek(&self, key: &K) -> Readiness<V, R> {
        match self.entries.get(key) {
            None if self.retired => Readiness::Retired,
            None => Readiness::Absent,
            Some(Entry::Compiling) => Readiness::Compiling,
            Some(Entry::Ready(inner)) => Readiness::Ready(Variant {
                inner: Arc::clone(inner),
            }),
            Some(Entry::Refused(reason)) => Readiness::Refused(reason.clone()),
        }
    }

    /// Ask about a key and count the answer.
    ///
    /// The counting version, for the path a draw takes. A hit and a miss are
    /// the numbers that say whether the key contains too much.
    pub fn request(&mut self, key: &K) -> Readiness<V, R> {
        let answer = self.peek(key);
        match &answer {
            Readiness::Ready(_) => self.census.hits += 1,
            Readiness::Compiling => self.census.joined += 1,
            Readiness::Absent => self.census.misses += 1,
            Readiness::Refused(_) | Readiness::Retired => {}
        }
        answer
    }

    /// Take the right to compile a key.
    ///
    /// `None` when the key is already compiling, ready, or refused — or when
    /// the family has retired, because a variant compiled into a family nobody
    /// can acquire from is work with no consumer.
    pub fn begin_flight(&mut self, key: K) -> Option<Flight<K>> {
        if self.retired || self.entries.contains_key(&key) {
            return None;
        }
        self.entries.insert(key.clone(), Entry::Compiling);
        Some(Flight {
            family: self.id,
            key,
        })
    }

    /// Publish a compilation's outcome, consuming the flight.
    ///
    /// # Errors
    ///
    /// [`Misdirected`] when the flight came from another family. Neither family
    /// is modified, and the flight and the compiled value come back so the
    /// caller can give them to whoever should have had them.
    pub fn publish(
        &mut self,
        flight: Flight<K>,
        outcome: Result<V, R>,
    ) -> Result<Readiness<V, R>, Misdirected<K, V, R>> {
        if flight.family != self.id {
            self.census.foreign += 1;
            return Err(Misdirected {
                wrong: WrongFamily {
                    flight: flight.family,
                    family: self.id,
                },
                flight,
                outcome,
            });
        }
        let Flight { key, .. } = flight;
        let entry = match outcome {
            Ok(value) => {
                self.census.ready += 1;
                Entry::Ready(Arc::new(value))
            }
            Err(reason) => {
                self.census.refused += 1;
                Entry::Refused(reason)
            }
        };
        self.entries.insert(key.clone(), entry);
        Ok(self.peek(&key))
    }

    /// The family's semantic generation has ended.
    ///
    /// No further flight may start, and a key the family never compiled
    /// answers [`Readiness::Retired`] rather than inviting one. Nothing is
    /// destroyed: work already recorded against these variants is still going
    /// to run, and [`Variant`] holders keep theirs alive. Call
    /// [`Self::collect`] once the timeline says that work is retired.
    pub fn retire(&mut self) {
        self.retired = true;
    }

    /// Take back every variant nobody else is holding.
    ///
    /// Returns the values whose only reference was the family's own, so the
    /// caller can destroy their native objects. A variant an outstanding
    /// [`Variant`] still names is **left in place** and returned by a later
    /// call — never destroyed underneath a recorded command buffer.
    ///
    /// Legal before retirement too, where it collects nothing, since every
    /// entry is still the family's to hand out.
    #[must_use = "the collected variants own native objects that need destroying"]
    pub fn collect(&mut self) -> Vec<V> {
        if !self.retired {
            return Vec::new();
        }
        let mut freed = Vec::new();
        let mut kept = HashMap::with_capacity(self.entries.len());
        for (key, entry) in self.entries.drain() {
            match entry {
                Entry::Ready(inner) => match Arc::try_unwrap(inner) {
                    Ok(value) => freed.push(value),
                    Err(still_held) => {
                        kept.insert(key, Entry::Ready(still_held));
                    }
                },
                // A refused key holds no native object, and a flight still
                // outstanding has to be publishable — into a retired family,
                // where it will be collected next time.
                other => {
                    kept.insert(key, other);
                }
            }
        }
        self.entries = kept;
        freed
    }

    /// Variants still held by somebody after a collect.
    #[must_use]
    pub fn outstanding(&self) -> usize {
        self.entries
            .values()
            .filter(|e| matches!(e, Entry::Ready(_)))
            .count()
    }

    /// Flights taken and not yet published.
    ///
    /// Retirement does not end them: [`Self::begin_flight`] grants no new one,
    /// but one already outstanding has to be publishable, and the family that
    /// issued it is the only place its key belongs. So a retired family with
    /// this above zero is still owed something, which is a different question
    /// from [`Self::outstanding`] and from [`Self::is_empty`] — the entries a
    /// [`Self::collect`] deliberately leaves behind make all three disagree.
    #[must_use]
    pub fn compiling(&self) -> usize {
        self.entries
            .values()
            .filter(|e| matches!(e, Entry::Compiling))
            .count()
    }
}

impl<K: Eq + Hash + Clone + Debug, V, R: Clone> Default for VariantFamily<K, V, R> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stands in for a `VkPipeline` and its layout: something with identity
    /// that a test can watch not being destroyed.
    #[derive(Debug, PartialEq, Eq)]
    struct Native(u32);

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Refusal {
        NoDualSourceBlend,
    }

    type Family = VariantFamily<u32, Native, Refusal>;

    #[test]
    fn an_absent_key_wants_a_flight_and_a_compiling_one_does_not() {
        let mut family = Family::new();
        assert!(family.request(&7).wants_a_flight());

        let flight = family.begin_flight(7).expect("the first caller");
        assert_eq!(*flight.key(), 7);
        assert_eq!(flight.family(), family.id());

        assert!(matches!(family.request(&7), Readiness::Compiling));
        assert!(!family.request(&7).wants_a_flight());
        assert!(
            family.begin_flight(7).is_none(),
            "a key admits exactly one flight"
        );
        assert_eq!(family.census().joined, 2);
        assert_eq!(family.census().misses, 1);

        family
            .publish(flight, Ok(Native(1)))
            .expect("its own flight");
        assert!(family.request(&7).is_ready());
        assert_eq!(family.census().hits, 1);
    }

    #[test]
    fn a_ready_key_is_never_recompiled() {
        let mut family = Family::new();
        let flight = family.begin_flight(1).expect("first");
        family.publish(flight, Ok(Native(9))).expect("own");
        assert!(family.begin_flight(1).is_none());
        assert!(family.begin_flight(2).is_some(), "a different key is free");
    }

    #[test]
    fn a_refusal_is_terminal_and_never_retried() {
        let mut family = Family::new();
        let flight = family.begin_flight(4).expect("first");
        let published = family
            .publish(flight, Err(Refusal::NoDualSourceBlend))
            .expect("own");
        assert_eq!(published, Readiness::Refused(Refusal::NoDualSourceBlend));

        // A guest drawing it every frame gets one refusal, not one per frame.
        for _ in 0..10 {
            assert!(!family.request(&4).wants_a_flight());
            assert!(family.begin_flight(4).is_none());
        }
        assert_eq!(family.census().refused, 1);
    }

    /// A flight published into the wrong family changes nothing anywhere, and
    /// hands back everything the caller gave it.
    ///
    /// The value coming back is the point, not a convenience: it is a native
    /// object nobody else has a name for, and dropping it on a path that is
    /// already a caller bug would leak it silently. The flight coming back is
    /// the same claim for the right to compile — it is not `Clone`, so
    /// consuming it would leave the key stuck compiling for the life of the
    /// family.
    #[test]
    fn a_foreign_flight_is_refused_without_mutating_either_family() {
        let mut mine = Family::new();
        let mut theirs = Family::new();
        assert_ne!(mine.id(), theirs.id());

        let flight = theirs.begin_flight(3).expect("their flight");
        let before = mine.len();
        let misdirected = mine
            .publish(flight, Ok(Native(5)))
            .expect_err("not mine to publish");
        assert_eq!(misdirected.wrong.family, mine.id());
        assert_eq!(misdirected.wrong.flight, theirs.id());
        assert_eq!(misdirected.wrong.slug(), "vk_variant_wrong_family");

        assert_eq!(mine.len(), before, "the wrong family gained an entry");
        assert!(
            matches!(theirs.peek(&3), Readiness::Compiling),
            "the right family lost its flight"
        );
        assert_eq!(mine.census().foreign, 1);

        // Everything came back, so the misdirected publication is recoverable
        // rather than a leak plus a stuck key.
        assert!(matches!(misdirected.outcome, Ok(Native(5))));
        assert_eq!(*misdirected.flight.key(), 3);
        let readiness = theirs
            .publish(misdirected.flight, misdirected.outcome)
            .expect("its own family this time");
        assert!(readiness.is_ready());
        assert!(theirs.request(&3).is_ready());
    }

    /// The claim that makes a recorded command buffer safe.
    #[test]
    fn a_held_variant_survives_its_familys_retirement() {
        let mut family = Family::new();
        let flight = family.begin_flight(1).expect("first");
        family.publish(flight, Ok(Native(42))).expect("own");

        let Readiness::Ready(held) = family.request(&1) else {
            panic!("ready");
        };
        family.retire();
        assert!(family.is_retired());
        assert!(
            family.begin_flight(2).is_none(),
            "a retired family starts no work"
        );

        // The holder still reads it, and collecting frees nothing.
        assert_eq!(held.get(), &Native(42));
        assert!(family.collect().is_empty(), "somebody still holds it");
        assert_eq!(family.outstanding(), 1);

        // Only when the last holder is gone does it come back for destruction.
        drop(held);
        let freed = family.collect();
        assert_eq!(freed, vec![Native(42)]);
        assert_eq!(family.outstanding(), 0);
        assert!(family.is_empty());
    }

    /// The livelock this variant exists to prevent. `Absent` tells a caller to
    /// take a flight; a retired family grants none. A caller following the
    /// module's own protocol — read readiness, take a flight if it wants one,
    /// otherwise wait — would have waited on a compile nothing was ever going
    /// to start, because `begin_flight`'s single `None` says "somebody else is
    /// on it" and "never" with the same value.
    #[test]
    fn a_key_a_retired_family_never_compiled_says_so_instead_of_inviting_a_flight() {
        let mut family = Family::new();
        assert_eq!(family.peek(&7), Readiness::Absent);
        assert!(family.peek(&7).wants_a_flight());

        family.retire();

        assert_eq!(family.peek(&7), Readiness::Retired);
        assert!(!family.peek(&7).wants_a_flight());
        assert!(!family.peek(&7).is_ready());
        assert!(
            family.begin_flight(7).is_none(),
            "and the answer matches what a flight would have been given"
        );
        // Terminal, not a miss: nothing is ever going to service it, so it is
        // not counted as work somebody has to do.
        assert_eq!(family.request(&7), Readiness::Retired);
        assert_eq!(family.census().misses, 0);
    }

    /// Retirement does not erase what the family already knows. A variant
    /// compiled before it is still why a recorded command buffer is safe, and
    /// a refusal's reason still has to reach whoever reads the fail channel.
    #[test]
    fn a_retired_family_still_answers_for_the_keys_it_holds() {
        let mut family = Family::new();
        let ready = family.begin_flight(1).expect("first");
        family.publish(ready, Ok(Native(5))).expect("own");
        let refused = family.begin_flight(2).expect("first");
        family
            .publish(refused, Err(Refusal::NoDualSourceBlend))
            .expect("own");
        let outstanding = family.begin_flight(3).expect("first");

        family.retire();

        assert!(family.peek(&1).is_ready());
        assert!(matches!(family.peek(&2), Readiness::Refused(_)));
        assert_eq!(family.peek(&3), Readiness::Compiling, "the flight lands");
        assert_eq!(family.peek(&4), Readiness::Retired, "and nothing else will");
        drop(outstanding);
    }

    #[test]
    fn collecting_before_retirement_frees_nothing() {
        let mut family = Family::new();
        let flight = family.begin_flight(1).expect("first");
        family.publish(flight, Ok(Native(3))).expect("own");
        assert!(
            family.collect().is_empty(),
            "a live family's variants are still its to hand out"
        );
        assert!(family.request(&1).is_ready());
    }

    /// A flight outstanding when the family retires still has to be
    /// publishable — otherwise a compile thread has nowhere to put its result
    /// and its native objects leak.
    #[test]
    fn a_flight_outstanding_at_retirement_can_still_land_and_is_then_collected() {
        let mut family = Family::new();
        let flight = family.begin_flight(8).expect("first");
        family.retire();

        assert!(family.collect().is_empty(), "still compiling");
        let published = family.publish(flight, Ok(Native(11))).expect("own flight");
        assert!(published.is_ready());
        // The `Readiness` handed back is itself a holder, so it has to go
        // before the variant can be collected.
        drop(published);
        assert_eq!(family.collect(), vec![Native(11)]);
    }

    #[test]
    fn a_refused_key_holds_no_native_object_to_collect() {
        let mut family = Family::new();
        let flight = family.begin_flight(2).expect("first");
        family
            .publish(flight, Err(Refusal::NoDualSourceBlend))
            .expect("own");
        family.retire();
        assert!(family.collect().is_empty());
        // And the reason survives collection, so a later reader still learns
        // why.
        assert!(matches!(family.peek(&2), Readiness::Refused(_)));
    }

    /// Nothing bounds the map, so a guest with many distinct configurations
    /// keeps all of them.
    #[test]
    fn a_thousand_live_variants_are_all_retained() {
        let mut family = Family::new();
        let mut held = Vec::new();
        for key in 0..1024u32 {
            let flight = family.begin_flight(key).expect("distinct keys");
            let ready = family.publish(flight, Ok(Native(key))).expect("own");
            let Readiness::Ready(variant) = ready else {
                panic!("ready");
            };
            held.push(variant);
        }
        assert_eq!(family.len(), 1024);
        for (key, variant) in held.iter().enumerate() {
            assert_eq!(variant.get(), &Native(key as u32));
        }
        assert_eq!(family.census().ready, 1024);
        // Every one of them is still there after a thousand more requests.
        for key in 0..1024u32 {
            assert!(family.request(&key).is_ready());
        }
        assert_eq!(family.census().hits, 1024);
        assert_eq!(
            family.census().misses,
            0,
            "the census counts requests, and this loop compiled without asking"
        );
    }

    #[test]
    fn two_variants_are_the_same_variant_only_when_they_are_one_allocation() {
        let mut family = Family::new();
        let one = family.begin_flight(1).expect("first");
        family.publish(one, Ok(Native(5))).expect("own");
        let two = family.begin_flight(2).expect("first");
        family.publish(two, Ok(Native(5))).expect("own");

        let Readiness::Ready(a) = family.peek(&1) else {
            panic!("ready")
        };
        let Readiness::Ready(b) = family.peek(&2) else {
            panic!("ready")
        };
        let Readiness::Ready(a_again) = family.peek(&1) else {
            panic!("ready")
        };
        assert_eq!(a, a_again);
        assert_ne!(a, b, "equal contents are not the same native pipeline");
    }

    #[test]
    fn a_family_id_is_unique_and_names_itself() {
        let a = Family::new();
        let b = Family::new();
        assert_ne!(a.id(), b.id());
        assert!(a.id().to_string().starts_with("family"));
        assert!(b.id() > a.id());
    }
}

/// A driven history of one family, checked against a shadow that knows only
/// the calls.
///
/// The three laws this module exists to keep are all laws about a *history*
/// rather than about one call, so no single-shot test can state them:
///
/// * a key admits **one** compile flight, ever, across every interleaving of
///   requests, publications, retirement and collection;
/// * [`VariantFamily::collect`] never hands back a variant somebody is still
///   holding — the law that stands between a recorded command buffer and a
///   destroyed `VkPipeline`;
/// * a handle taken before a retirement still names the value it was published
///   with afterwards.
///
/// The shadow below is deliberately dumber than the family and differently
/// shaped: it keeps no `Arc`, so it cannot agree with the family by reference
/// counting the same way. It counts held handles itself, in a plain map keyed
/// by the key, and derives every expectation from the arguments of the calls
/// that were made — never from anything the family returned about its own
/// state.
#[cfg(test)]
mod sweep {
    use super::*;
    use std::collections::BTreeMap;

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

    /// Stands in for a `VkPipeline`: something with an identity the sweep can
    /// follow from the publication that created it to the collection that
    /// hands it back, and which is not `Clone`, so a value can exist once.
    #[derive(Debug, PartialEq, Eq)]
    struct Native(u32);

    type Family = VariantFamily<u32, Native, u32>;

    /// What the shadow believes about one key, from the calls alone.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Believed {
        /// No call has put an entry here, or a collect took the one that was.
        Nothing,
        /// A flight was granted and has not been published.
        Flying,
        /// A publication landed a value with this serial.
        Ready(u32),
        /// A publication landed this reason.
        Refused(u32),
    }

    /// Every expectation the sweep has, derived from the calls it made.
    struct Shadow {
        keys: BTreeMap<u32, Believed>,
        retired: bool,
        census: Census,
    }

    impl Shadow {
        fn new() -> Self {
            Self {
                keys: BTreeMap::new(),
                retired: false,
                census: Census::default(),
            }
        }

        fn at(&self, key: u32) -> Believed {
            self.keys.get(&key).copied().unwrap_or(Believed::Nothing)
        }

        fn put(&mut self, key: u32, believed: Believed) {
            self.keys.insert(key, believed);
        }

        /// Entries the family should be holding: everything the calls put
        /// there and no collect took away.
        fn entries(&self) -> usize {
            self.keys
                .values()
                .filter(|b| **b != Believed::Nothing)
                .count()
        }

        fn ready(&self) -> usize {
            self.keys
                .values()
                .filter(|b| matches!(b, Believed::Ready(_)))
                .count()
        }
    }

    /// How often each path was taken, so a floor can say the history reached
    /// the ones the laws live on rather than merely being long.
    #[derive(Default)]
    struct Tally {
        published_ok: u32,
        published_err: u32,
        foreign: u32,
        joined: u32,
        flight_refused_busy: u32,
        flight_refused_retired: u32,
        peeked_retired: u32,
        held_taken: u32,
        held_dropped: u32,
        collected_value: u32,
        collect_kept_held: u32,
        collect_before_retirement: u32,
    }

    /// A handle the driver is holding, beside the serial it was published with.
    struct Held {
        key: u32,
        serial: u32,
        variant: Variant<Native>,
    }

    const KEYS: u64 = 6;
    const HELD_CAP: usize = 8;

    /// Everything that must be true of the family after every single call.
    fn check(family: &Family, shadow: &Shadow, held: &[Held], tally: &mut Tally) {
        assert_eq!(family.len(), shadow.entries(), "entries");
        assert_eq!(family.is_empty(), shadow.entries() == 0, "emptiness");
        assert_eq!(family.outstanding(), shadow.ready(), "ready entries");
        assert_eq!(family.census(), shadow.census, "census");
        assert_eq!(family.is_retired(), shadow.retired, "retirement");

        for key in 0..u32::try_from(KEYS).expect("small") {
            let answer = family.peek(&key);
            match shadow.at(key) {
                Believed::Nothing if shadow.retired => {
                    assert!(
                        matches!(answer, Readiness::Retired),
                        "a key a retired family does not hold answers Retired"
                    );
                    assert!(
                        !answer.wants_a_flight(),
                        "and invites no flight, which is why it is not Absent"
                    );
                    tally.peeked_retired += 1;
                }
                Believed::Nothing => assert!(matches!(answer, Readiness::Absent), "absent"),
                Believed::Flying => assert!(matches!(answer, Readiness::Compiling), "compiling"),
                Believed::Refused(reason) => {
                    assert_eq!(answer, Readiness::Refused(reason), "refusal survives");
                }
                Believed::Ready(serial) => {
                    let Readiness::Ready(variant) = answer else {
                        panic!("a published key is ready");
                    };
                    assert_eq!(
                        *variant.get(),
                        Native(serial),
                        "the value that was published"
                    );
                    // Every handle for one key is one allocation: a family
                    // that re-wrapped a value would hand out a second
                    // lifetime for the same native object.
                    assert_eq!(family.peek(&key), Readiness::Ready(variant.clone()));
                }
            }
        }

        // The law that matters most: nothing the driver holds has been taken
        // away from underneath it, whatever the family did in between.
        for h in held {
            assert_eq!(
                *h.variant.get(),
                Native(h.serial),
                "a held variant is intact"
            );
            assert_eq!(
                shadow.at(h.key),
                Believed::Ready(h.serial),
                "a held key keeps its entry until a collect that may not take it"
            );
        }
    }

    fn drive(seed: u64, steps: usize, tally: &mut Tally) {
        let mut rng = Rng::new(seed);
        let mut family = Family::new();
        let mut donor = Family::new();
        let mut shadow = Shadow::new();
        let mut flights: Vec<Flight<u32>> = Vec::new();
        let mut held: Vec<Held> = Vec::new();
        let mut serial = 1u32;

        for _ in 0..steps {
            let key = u32::try_from(rng.below(KEYS)).expect("small");
            // Weighted rather than uniform: the laws live on publication,
            // holding and collection, and a uniform draw over nine arms left
            // the collect-while-held path at a tenth of a percent of the
            // history.
            match rng.below(100) {
                0..=17 => {
                    let answer = family.request(&key);
                    match shadow.at(key) {
                        Believed::Nothing if shadow.retired => {
                            assert!(matches!(answer, Readiness::Retired));
                            assert!(!answer.wants_a_flight());
                        }
                        Believed::Nothing => {
                            assert!(answer.wants_a_flight());
                            shadow.census.misses += 1;
                        }
                        Believed::Flying => {
                            assert!(matches!(answer, Readiness::Compiling));
                            assert!(!answer.wants_a_flight());
                            shadow.census.joined += 1;
                            tally.joined += 1;
                        }
                        Believed::Ready(_) => {
                            assert!(answer.is_ready());
                            assert!(!answer.wants_a_flight());
                            shadow.census.hits += 1;
                        }
                        Believed::Refused(_) => assert!(!answer.wants_a_flight()),
                    }
                }
                18..=33 => {
                    if held.len() < HELD_CAP {
                        if let (Readiness::Ready(variant), Believed::Ready(serial)) =
                            (family.peek(&key), shadow.at(key))
                        {
                            held.push(Held {
                                key,
                                serial,
                                variant,
                            });
                            tally.held_taken += 1;
                        }
                    }
                }
                34..=41 => {
                    if !held.is_empty() {
                        let at = usize::try_from(rng.below(held.len() as u64)).expect("in range");
                        held.remove(at);
                        tally.held_dropped += 1;
                    }
                }
                42..=61 => {
                    let taken = family.begin_flight(key);
                    match (shadow.retired, shadow.at(key)) {
                        (false, Believed::Nothing) => {
                            let flight = taken.expect("the first caller for a live key");
                            assert_eq!(flight.family(), family.id());
                            assert_eq!(*flight.key(), key);
                            shadow.put(key, Believed::Flying);
                            flights.push(flight);
                        }
                        (true, _) => {
                            assert!(taken.is_none(), "a retired family grants no flight");
                            tally.flight_refused_retired += 1;
                        }
                        (false, _) => {
                            assert!(taken.is_none(), "one flight per key");
                            tally.flight_refused_busy += 1;
                        }
                    }
                }
                62..=79 => {
                    if !flights.is_empty() {
                        let at =
                            usize::try_from(rng.below(flights.len() as u64)).expect("in range");
                        let flight = flights.remove(at);
                        let flown = *flight.key();
                        let outcome = if rng.below(4) == 0 {
                            let reason = 900 + flown;
                            shadow.put(flown, Believed::Refused(reason));
                            shadow.census.refused += 1;
                            tally.published_err += 1;
                            Err(reason)
                        } else {
                            let value = serial;
                            serial += 1;
                            shadow.put(flown, Believed::Ready(value));
                            shadow.census.ready += 1;
                            tally.published_ok += 1;
                            Ok(Native(value))
                        };
                        let landed = family.publish(flight, outcome).expect("its own family");
                        match shadow.at(flown) {
                            Believed::Ready(value) => {
                                let Readiness::Ready(variant) = landed else {
                                    panic!("a landed value is ready")
                                };
                                assert_eq!(*variant.get(), Native(value));
                            }
                            Believed::Refused(reason) => {
                                assert_eq!(landed, Readiness::Refused(reason));
                            }
                            other => panic!("a publication leaves no {other:?}"),
                        }
                    }
                }
                80..=85 => {
                    // A flight from another family, published into this one.
                    // Both the value and the right to compile have to come
                    // back, and this family must not have moved.
                    let Some(stranger) = donor.begin_flight(key) else {
                        continue;
                    };
                    let before = shadow.at(key);
                    let refused = family
                        .publish(stranger, Ok(Native(u32::MAX)))
                        .expect_err("a foreign flight");
                    assert_eq!(
                        refused.wrong,
                        WrongFamily {
                            flight: donor.id(),
                            family: family.id(),
                        }
                    );
                    assert_eq!(*refused.flight.key(), key);
                    assert_eq!(refused.outcome, Ok(Native(u32::MAX)));
                    assert_eq!(shadow.at(key), before, "a stranger changes nothing");
                    shadow.census.foreign += 1;
                    tally.foreign += 1;
                    donor
                        .publish(refused.flight, refused.outcome)
                        .expect("back where it came from");
                }
                86..=89 => {
                    // Gated inside its arm rather than given a narrower draw,
                    // or every history would end at its first few steps.
                    if !shadow.retired && rng.below(4) == 0 {
                        family.retire();
                        shadow.retired = true;
                    }
                }
                90..=99 => {
                    let mut expected: Vec<u32> = Vec::new();
                    if shadow.retired {
                        for (key, believed) in &shadow.keys {
                            if let Believed::Ready(value) = *believed {
                                if held.iter().any(|h| h.key == *key) {
                                    tally.collect_kept_held += 1;
                                } else {
                                    expected.push(value);
                                }
                            }
                        }
                    } else {
                        tally.collect_before_retirement += 1;
                    }
                    let mut freed: Vec<u32> = family.collect().into_iter().map(|n| n.0).collect();
                    freed.sort_unstable();
                    expected.sort_unstable();
                    assert_eq!(freed, expected, "collected exactly the unheld variants");
                    tally.collected_value += u32::try_from(freed.len()).expect("small");
                    for value in freed {
                        let gone = shadow
                            .keys
                            .iter()
                            .find(|(_, b)| **b == Believed::Ready(value))
                            .map(|(k, _)| *k)
                            .expect("a collected value was a believed entry");
                        shadow.put(gone, Believed::Nothing);
                    }
                }
                _ => {}
            }
            check(&family, &shadow, &held, tally);
        }

        // Whatever the history did, a handle still names its own value once
        // everything else is gone.
        drop(family);
        for h in &held {
            assert_eq!(*h.variant.get(), Native(h.serial), "past the family's end");
        }
    }

    #[test]
    fn a_driven_family_grants_one_flight_per_key_and_frees_only_what_nobody_holds() {
        let mut tally = Tally::default();
        for seed in 0..400u64 {
            drive(seed, 150, &mut tally);
        }
        // Floors on each path separately, at roughly half what the history
        // measured: one aggregate count would let a whole arm go quiet.
        assert!(tally.published_ok > 700, "{}", tally.published_ok);
        assert!(tally.published_err > 240, "{}", tally.published_err);
        assert!(tally.foreign > 900, "{}", tally.foreign);
        assert!(tally.joined > 250, "{}", tally.joined);
        assert!(
            tally.flight_refused_busy > 2400,
            "{}",
            tally.flight_refused_busy
        );
        assert!(
            tally.flight_refused_retired > 2500,
            "{}",
            tally.flight_refused_retired
        );
        assert!(tally.peeked_retired > 50_000, "{}", tally.peeked_retired);
        assert!(tally.held_taken > 1400, "{}", tally.held_taken);
        assert!(tally.held_dropped > 900, "{}", tally.held_dropped);
        assert!(tally.collected_value > 340, "{}", tally.collected_value);
        assert!(tally.collect_kept_held > 800, "{}", tally.collect_kept_held);
        assert!(
            tally.collect_before_retirement > 1700,
            "{}",
            tally.collect_before_retirement
        );
    }
}
