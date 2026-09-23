//! Where a backing's current bytes are, and what it costs to read them
//! somewhere else.
//!
//! # The two failures this exists to make impossible
//!
//! **A transfer with no version transition.** If a copy can happen without the
//! content having changed, then nothing bounds how many copies happen, and the
//! device pays for the same bytes every frame. Every transfer here is derived
//! from a replica being behind on specific bytes, and recording it makes it
//! not-behind — so the same transfer cannot be asked for twice without a write
//! in between.
//!
//! **A whole-resource copy to answer a partial question.** A guest writes part
//! of a buffer and a draw reads part of it. Whether a transfer is owed is a
//! question about bytes, and a per-resource dirty flag can only answer it by
//! copying everything. Freshness is a [`RangeSet`] per replica for exactly that
//! reason.
//!
//! # Authority is not placement
//!
//! [`Replica`] says *where the current bytes are*, which is a semantic fact:
//! the guest wrote them, or the device produced them into storage it owns.
//! It says nothing about how that storage is arranged, whether it is imported
//! or copied, or which memory type it lives in — those are an executor's
//! decisions about one capability cell, and a semantic model that knew them
//! would make the same guest stream mean different things on two hosts.

use crate::access::{BackingId, ByteRange, ContentVersion};
use crate::coverage::{Applied, Span, VersionCoverage};
use crate::range_set::RangeSet;
use std::collections::HashMap;

/// Where a copy of a backing's content lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Replica {
    /// The guest's own pages.
    GuestPages,
    /// Storage this device owns. Whether that storage is an import of the
    /// guest's pages or a separate allocation is a placement decision and not
    /// this one.
    DeviceOwned,
}

impl Replica {
    pub const BOTH: [Replica; 2] = [Replica::GuestPages, Replica::DeviceOwned];

    pub const fn name(self) -> &'static str {
        match self {
            Self::GuestPages => "guest_pages",
            Self::DeviceOwned => "device_owned",
        }
    }

    #[must_use]
    pub const fn other(self) -> Replica {
        match self {
            Self::GuestPages => Self::DeviceOwned,
            Self::DeviceOwned => Self::GuestPages,
        }
    }
}

/// A copy that is owed before a read can be served.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transfer {
    pub backing: BackingId,
    pub from: Replica,
    pub to: Replica,
    /// Exactly the bytes the destination is behind on.
    pub bytes: RangeSet,
    /// The version current in each part of `bytes` when the copy was planned —
    /// which is to say, *what* is being copied and not only from where.
    ///
    /// A transfer is planned at admission and recorded at completion, and the
    /// content can move in between. Without this the recording would mark the
    /// destination fresh for bytes it holds an older copy of, and the next read
    /// would find it fresh and serve them. See [`ContentLedger::record_transfer`].
    pub at: Vec<Span>,
}

/// What has happened to one backing's content.
#[derive(Clone, Debug, Default)]
struct Entry {
    /// Which version is current in each part of the backing. See
    /// [`crate::coverage`] for why this is not one number.
    canonical: VersionCoverage,
    /// The extent the declaration named, which is the whole backing.
    ///
    /// `None` for a backing that reached this ledger through a write rather
    /// than a declaration — a test, or a replay that starts mid-stream. It is
    /// what turns "the whole backing" into bytes, and where it is absent that
    /// translation is refused rather than guessed at some default size.
    extent: Option<ByteRange>,
    fresh: HashMap<Replica, RangeSet>,
}

/// The content authority for one session.
#[derive(Debug, Default)]
pub struct ContentLedger {
    backings: HashMap<BackingId, Entry>,
    /// The next version any write may reserve. Monotone, and one counter for
    /// the whole ledger.
    ///
    /// One per *backing* would be enough for the comparisons — every version
    /// is only ever compared against another version of the same backing — but
    /// it would live in the entry, and [`ContentLedger::forget`] drops the
    /// entry. The counter would restart, and a version already carried by a
    /// planned transfer or a reserved write would be handed out again for
    /// different content. [`ContentLedger::reserve`] says why a number must not
    /// be reused; this is where that has to be arranged.
    ///
    /// Per region would be wrong for the reason the coverage is not one number
    /// the other way round: two writers of disjoint ranges must not be handed
    /// one number, or their completions cannot be told apart.
    next_version: ContentVersion,
    census: Census,
}

/// What the ledger has been asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Writes recorded.
    pub writes: usize,
    /// Transfers the ledger said were owed.
    pub transfers_planned: usize,
    /// Bytes in those transfers.
    pub transfer_bytes: u64,
    /// Reads that needed no transfer because the replica was already fresh for
    /// every byte. The number that says how much the per-byte freshness is
    /// buying over a per-resource flag.
    pub reads_already_fresh: usize,
    /// Reads that needed no transfer because no replica held the bytes they
    /// were behind on — content nothing has ever written.
    ///
    /// A separate count and not part of [`Self::reads_already_fresh`], because
    /// the two say opposite things about the ledger. A read that was already
    /// fresh is the per-byte freshness paying for itself; a read over content
    /// nobody has written is a resource being read before it is filled, and a
    /// stream where that number is large is one where the guest is reading
    /// undefined bytes rather than one where this ledger is working well.
    /// Together they account for every read [`ContentLedger::transfer_for_read`]
    /// answered with `None`, so a caller can tell that it has seen all of them.
    pub reads_with_no_source: usize,
    /// Copies that completed after the content they carried had already been
    /// superseded, and so destroyed the destination's bytes instead of
    /// delivering it the current ones.
    ///
    /// Not a ledger fault and not something the ledger can prevent: the copy
    /// was correct when it was planned, and what orders it against a write is
    /// the hazard graph, not this. It is here because the consequence — bytes
    /// that were current in the destination and are now nowhere — is otherwise
    /// visible only as a later read reporting no source, a long way from the
    /// transfer that caused it.
    pub transfers_overtaken: usize,
}

impl ContentLedger {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// The highest version current anywhere in the backing.
    ///
    /// `None` for a backing nothing has written. Distinct from version zero,
    /// which is a version a write can produce — see
    /// [`VersionCoverage::newest_over`].
    ///
    /// A summary, and only ever that. Two writers of disjoint ranges are both
    /// current and this reports one of their versions, so a caller deciding
    /// whether *its own* bytes are current must ask
    /// [`Self::version_of`] about the range it names.
    #[must_use]
    pub fn newest_version(&self, backing: BackingId) -> Option<ContentVersion> {
        self.backings.get(&backing).and_then(|e| {
            e.canonical.newest_over(ByteRange {
                offset: 0,
                length: u64::MAX,
            })
        })
    }

    /// The highest version current over `range`.
    #[must_use]
    pub fn version_of(&self, backing: BackingId, range: ByteRange) -> Option<ContentVersion> {
        self.backings
            .get(&backing)
            .and_then(|e| e.canonical.newest_over(range))
    }

    /// The bytes a record naming "the whole backing" is naming.
    ///
    /// The extent its declaration gave it, and `None` for a backing no
    /// declaration reached. A caller with a whole-backing write must ask this
    /// rather than inventing a range: a write recorded against no bytes
    /// publishes a version over nothing, so a later *older* write is not
    /// beaten by it and a replica that produced the content does not become
    /// fresh for it — which is a transfer that copies stale bytes over what
    /// the device just wrote.
    ///
    /// Not answerable for a subresource, and that is a different fact: image
    /// coordinates need a layout, which is an executor's and not this crate's.
    #[must_use]
    pub fn extent(&self, backing: BackingId) -> Option<ByteRange> {
        self.backings.get(&backing).and_then(|e| e.extent)
    }

    /// Take the next version a write of this backing may produce.
    ///
    /// The planning half of the reservation rule: a version is reserved when
    /// the write is planned and becomes current only when
    /// [`Self::materialize`] records the completion, so a reader planned
    /// against it waits for the work rather than for the plan.
    ///
    /// Advances the counter whatever happens next. A reservation that is never
    /// materialized — a transaction the device refuses after planning it —
    /// leaves a gap in the numbering and nothing else, which is the cheap
    /// answer; reusing the number would let a later write pass for the
    /// abandoned one.
    pub fn reserve(&mut self, backing: BackingId) -> ContentVersion {
        self.backings.entry(backing).or_default();
        let reserved = self.next_version;
        self.next_version = reserved.next();
        reserved
    }

    /// Record that a write of `bytes` at `version` has completed in `replica`.
    ///
    /// Coverage is taken only where nothing at least as new already holds the
    /// bytes, and **freshness follows the coverage**: a replica becomes fresh
    /// for exactly the bytes whose version it won, and the other replica loses
    /// exactly those. A completion that lost the race leaves both replicas as
    /// they were, which is the point — the winner's bytes are the current
    /// content and the loser's must not be readable from anywhere.
    ///
    /// Returns what landed and what was beaten. A caller that ignores the
    /// second half has a transaction whose bytes never became visible and no
    /// record saying so.
    pub fn materialize(
        &mut self,
        backing: BackingId,
        bytes: ByteRange,
        version: ContentVersion,
        replica: Replica,
    ) -> Applied {
        // Keep the counter ahead of anything committed, so content whose
        // versions were assigned elsewhere — a replayed transaction, a test —
        // cannot later reserve a number already in a coverage.
        if version >= self.next_version {
            self.next_version = version.next();
        }
        let e = self.backings.entry(backing).or_default();
        let applied = e.canonical.apply(bytes, version);
        for taken in applied.taken.ranges() {
            for r in Replica::BOTH {
                let set = e.fresh.entry(r).or_default();
                if r == replica {
                    set.insert(*taken);
                } else {
                    set.remove(*taken);
                }
            }
        }
        applied
    }

    /// Declare that a backing's content originates in one replica and is
    /// current there over `extent`.
    ///
    /// This is creation, not a write: a resource whose pages the guest supplied
    /// starts with the guest fresh for all of it and the device owning nothing.
    ///
    /// # A backing this ledger already knows is joined, not reset
    ///
    /// Two live names for one backing is a first-class state, not a corner —
    /// [`crate::namespace::Teardown::HeldByAnotherName`] is a whole variant
    /// about it, and an `IOSurface` reachable from two tasks is the ordinary
    /// case. This used to clear the coverage and overwrite the extent whenever
    /// it was called, so the second name's creation discarded the first name's
    /// content: a range the device had rendered into became "content nothing
    /// has ever written", so a later read planned no transfer and found no
    /// source, and the backing's extent shrank to the newcomer's — which is
    /// what turns a whole-resource participation into bytes, so every
    /// whole-backing access of the older name was then planned over a window
    /// of itself.
    ///
    /// So a known backing takes the declaration as a claim about *those*
    /// bytes: the extent grows to cover both and the named range becomes the
    /// declaring replica's at a new version, leaving every other byte's
    /// authority where it was. That is the same answer a placed resource's
    /// creation already gives for the same reason — "declaring the whole
    /// backing would discard the neighbours' content" — and the same one
    /// [`crate::lifecycle`]'s invalidate gives for "the guest's pages are the
    /// current content".
    ///
    /// The clearing form is therefore reachable only for a backing nothing
    /// knows, where there is nothing to clear. [`Self::forget`] is the one
    /// door that drops what a backing holds.
    pub fn declare(&mut self, backing: BackingId, extent: ByteRange, authority: Replica) {
        if let Some(known) = self.backings.get_mut(&backing) {
            known.extent = Some(match known.extent {
                Some(old) => {
                    let end = old
                        .offset
                        .saturating_add(old.length)
                        .max(extent.offset.saturating_add(extent.length));
                    let offset = old.offset.min(extent.offset);
                    ByteRange {
                        offset,
                        length: end - offset,
                    }
                }
                None => extent,
            });
            self.write(backing, extent, authority);
            return;
        }
        let version = self.reserve(backing);
        let e = self.backings.entry(backing).or_default();
        e.extent = Some(extent);
        e.canonical.clear();
        e.fresh.clear();
        e.fresh.insert(authority, RangeSet::from_range(extent));
        // The coverage is rebuilt rather than applied over the old one: a
        // declaration is not a write that has to beat what was there, it is a
        // statement that what was there is gone. `apply` on a cleared map takes
        // everything, which is the same answer written the way the invariant
        // wants it.
        e.canonical.apply(extent, version);
    }

    /// Record a write of `bytes` performed in `replica`.
    ///
    /// The version advances once per write, and the writing replica becomes
    /// the only one fresh for those bytes. Every other replica loses them,
    /// which is what makes a later read from elsewhere owe a transfer — and
    /// what makes a read from *here* owe nothing.
    pub fn write(&mut self, backing: BackingId, bytes: ByteRange, replica: Replica) {
        self.census.writes += 1;
        let version = self.reserve(backing);
        // A CPU write is planned and completed at once, so its reservation can
        // never lose: nothing else could have taken a higher version in
        // between. `materialize` still decides, rather than this asserting it.
        self.materialize(backing, bytes, version, replica);
    }

    /// The transfer a read of `bytes` from `replica` needs, if any.
    ///
    /// `None` means the replica already holds those bytes at the current
    /// version. It is not a promise that the read is free — that depends on
    /// placement — only that no content has to move for it to be correct.
    #[must_use]
    pub fn transfer_for_read(
        &mut self,
        backing: BackingId,
        bytes: ByteRange,
        replica: Replica,
    ) -> Option<Transfer> {
        // A backing this ledger has never heard of is the sourceless case in
        // its purest form: no replica holds these bytes because nothing has
        // ever written them. Returning `None` without counting it would leave
        // the one number that says "the guest is reading undefined bytes"
        // blind to the reads that say it loudest, and would break the census's
        // own claim that the two counters account for every `None`.
        let Some(e) = self.backings.get(&backing) else {
            self.census.reads_with_no_source += 1;
            return None;
        };
        let empty = RangeSet::new();
        let fresh = e.fresh.get(&replica).unwrap_or(&empty);
        // The steady-state answer, asked without building anything. A replica
        // that already holds every byte is the common case on a warm frame,
        // and computing the owed set only to find it empty put an allocation
        // on the path every read takes.
        if fresh.covers(bytes) {
            self.census.reads_already_fresh += 1;
            return None;
        }
        let owed = fresh.missing_from(bytes);
        // Only the other replica can supply them, and only the parts it is
        // itself fresh for. Bytes neither replica holds have never been
        // written, so there is nothing to move and no version to preserve —
        // copying them would move whatever the source happens to contain and
        // then claim the destination is current.
        let source = replica.other();
        let movable = owed.intersection(e.fresh.get(&source).unwrap_or(&empty));
        if movable.is_empty() {
            self.census.reads_with_no_source += 1;
            return None;
        }
        // What is being copied, taken while the answer is still the plan's.
        // `fresh` is only ever set from bytes `canonical` covers, so every part
        // of `movable` has a version here.
        let at: Vec<Span> = movable
            .ranges()
            .iter()
            .flat_map(|r| e.canonical.over(*r))
            .collect();
        self.census.transfers_planned += 1;
        self.census.transfer_bytes += movable.len();
        Some(Transfer {
            backing,
            from: source,
            to: replica,
            bytes: movable,
            at,
        })
    }

    /// Record that a planned transfer has completed.
    ///
    /// The destination becomes fresh for the bytes that moved *and are still
    /// what the plan said they were*, and the source keeps what it had: a copy
    /// does not change where content is authoritative, only how many places
    /// hold it.
    ///
    /// # Why the version is checked here and not assumed
    ///
    /// A transfer is planned when a transaction is admitted and recorded when
    /// it completes, and the guest may write the same bytes in between — that
    /// is an ordinary sequence, not a race the device can forbid. The copy then
    /// carried the older content, and marking the destination fresh for it
    /// makes the next read of those bytes find a replica that is fresh by the
    /// ledger and stale in fact, with no transfer owed and nothing to say so.
    ///
    /// [`Self::materialize`] already refuses a write beaten by newer content,
    /// for the same reason and in the same words: the loser's bytes must not be
    /// readable from anywhere. A completed copy is the other kind of completion
    /// and was the one not asking.
    ///
    /// # And the bytes it did not deliver are bytes it destroyed
    ///
    /// A copy that arrives late does not arrive as nothing. It writes the
    /// content it carried over whatever the destination had there, so where the
    /// destination held the current bytes and the copy is stale, those bytes
    /// are gone — from the destination's storage, and therefore from the ledger
    /// too. Recording only the half that landed would leave the destination
    /// fresh for content its own copy has just overwritten, which is the same
    /// wrong answer one step further on.
    ///
    /// Where the source still holds them, the next read simply owes a transfer
    /// again. Where it does not, the bytes are genuinely nowhere, and a read
    /// reporting no source is the honest answer rather than a served copy of
    /// something else. [`Census::transfers_overtaken`] counts these, because
    /// the cause and the symptom are otherwise far apart.
    ///
    /// A backing this ledger has forgotten takes nothing: there are no content
    /// facts left to update, and creating an entry for one would publish
    /// freshness over coverage nobody declared.
    pub fn record_transfer(&mut self, transfer: &Transfer) {
        let Some(e) = self.backings.get_mut(&transfer.backing) else {
            return;
        };
        let mut landed = RangeSet::new();
        for planned in &transfer.at {
            for now in e.canonical.over(planned.range) {
                if now.version == planned.version {
                    landed.insert(now.range);
                }
            }
        }
        let mut clobbered = transfer.bytes.clone();
        clobbered.subtract(&landed);
        let set = e.fresh.entry(transfer.to).or_default();
        set.union_with(&landed);
        set.subtract(&clobbered);
        if !clobbered.is_empty() {
            self.census.transfers_overtaken += 1;
        }
    }

    /// Discard a replica's copy of some bytes without changing the content.
    ///
    /// This is what a discard packet means: the bytes are still whatever they
    /// were, and this replica no longer holds them. The version does not
    /// advance, because nothing was written.
    ///
    /// Region-scoped rather than whole-backing, because several resources share
    /// one backing whenever they are placed in one heap, and discarding a
    /// heap-placed resource's copy must not throw away its neighbours'.
    pub fn discard(&mut self, backing: BackingId, bytes: ByteRange, replica: Replica) {
        if let Some(e) = self.backings.get_mut(&backing) {
            if let Some(set) = e.fresh.get_mut(&replica) {
                set.remove(bytes);
            }
        }
    }

    /// The bytes of `range` that only `replica` holds.
    ///
    /// The question a discard has to ask before it takes a hint: content no
    /// other replica is fresh for exists nowhere else, so dropping it is not a
    /// memory saving but a loss of bytes the guest may still read.
    #[must_use]
    pub fn sole_authority(
        &self,
        backing: BackingId,
        range: ByteRange,
        replica: Replica,
    ) -> RangeSet {
        let Some(e) = self.backings.get(&backing) else {
            return RangeSet::new();
        };
        let Some(here) = e.fresh.get(&replica) else {
            return RangeSet::new();
        };
        let mut out = here.intersection(&RangeSet::from_range(range));
        if let Some(elsewhere) = e.fresh.get(&replica.other()) {
            out.subtract(elsewhere);
        }
        out
    }

    /// Whether the ledger holds anything at all about a backing.
    ///
    /// The question [`Self::forget`] answers by making it false, and the one a
    /// session-wide owner asks to check that it has neither dropped a live
    /// backing's history nor kept a dead one's. Presence is not freshness: a
    /// backing that was declared and never written is known here and fresh in
    /// no replica.
    #[must_use]
    pub fn knows(&self, backing: BackingId) -> bool {
        self.backings.contains_key(&backing)
    }

    /// Forget a backing entirely.
    pub fn forget(&mut self, backing: BackingId) {
        self.backings.remove(&backing);
    }

    /// Whether a replica is fresh for every byte of a range.
    #[must_use]
    pub fn is_fresh(&self, backing: BackingId, bytes: ByteRange, replica: Replica) -> bool {
        self.backings
            .get(&backing)
            .and_then(|e| e.fresh.get(&replica))
            .is_some_and(|set| set.covers(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(offset: u64, length: u64) -> ByteRange {
        ByteRange { offset, length }
    }
    const B: BackingId = BackingId(1);

    #[test]
    fn a_declared_backing_is_fresh_where_it_was_declared_and_nowhere_else() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        assert!(c.is_fresh(B, r(0, 256), Replica::GuestPages));
        assert!(!c.is_fresh(B, r(0, 1), Replica::DeviceOwned));
        // A declaration reserves and commits one version over the extent.
        assert_eq!(c.newest_version(B), Some(ContentVersion(0)));
        assert_eq!(c.version_of(B, r(0, 256)), Some(ContentVersion(0)));
    }

    /// The first failure this module exists to prevent: a copy without a
    /// version transition. Asking twice for the same read cannot produce two
    /// transfers.
    #[test]
    fn a_transfer_recorded_is_a_transfer_not_asked_for_again() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .expect("the device holds nothing yet");
        assert_eq!(t.from, Replica::GuestPages);
        assert_eq!(t.bytes.ranges(), &[r(0, 64)]);
        c.record_transfer(&t);
        assert!(c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .is_none());
        assert_eq!(c.census().transfers_planned, 1);
        assert_eq!(c.census().reads_already_fresh, 1);
    }

    /// **A copy that landed after the content moved on is not the content.**
    ///
    /// The plan and the recording are two moments — admission and completion —
    /// and a guest write between them is ordinary. The copy then carried the
    /// older bytes, so the destination must not come out of the recording fresh
    /// for them: a replica fresh by the ledger and stale in fact owes no
    /// transfer, and every later read of those bytes is served the old content
    /// with nothing anywhere saying so.
    #[test]
    fn a_transfer_overtaken_by_a_write_does_not_make_the_destination_fresh() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .expect("the device holds nothing yet");
        assert_eq!(t.bytes.ranges(), &[r(0, 64)]);

        // The guest rewrites the front half while the copy is in flight.
        c.write(B, r(0, 32), Replica::GuestPages);
        c.record_transfer(&t);

        let again = c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .expect("the front half is still owed");
        assert_eq!(
            again.bytes.ranges(),
            &[r(0, 32)],
            "exactly the bytes the copy was overtaken on, and not the ones it \
             delivered"
        );
        assert!(
            c.is_fresh(B, r(32, 32), Replica::DeviceOwned),
            "the back half moved unchallenged and did land"
        );

        // And recording that one, unovertaken, finishes the job.
        c.record_transfer(&again);
        assert!(c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .is_none());
    }

    /// A backing the ledger has forgotten has no content facts to update, and a
    /// late recording must not invent one.
    #[test]
    fn a_transfer_recorded_after_the_backing_was_forgotten_publishes_nothing() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .expect("owed");
        c.forget(B);
        c.record_transfer(&t);
        assert!(!c.knows(B), "a recording is not a declaration");
    }

    /// And a write is what makes it owed again.
    #[test]
    fn a_write_elsewhere_makes_the_transfer_owed_again() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        let before = c.version_of(B, r(0, 16)).expect("declared");
        c.write(B, r(0, 16), Replica::GuestPages);
        assert!(
            c.version_of(B, r(0, 16)).expect("written") > before,
            "a write is a version transition"
        );
        // And only over the bytes it named: the rest of the backing is still
        // at the version the declaration gave it.
        assert_eq!(c.version_of(B, r(16, 240)), Some(before));
        let again = c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .expect("the first sixteen bytes moved under it");
        assert_eq!(
            again.bytes.ranges(),
            &[r(0, 16)],
            "only the bytes that changed are owed, not the whole read"
        );
    }

    /// The second failure: answering a partial question with a whole-resource
    /// copy.
    #[test]
    fn a_partial_read_owes_only_the_bytes_it_is_behind_on() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 4096), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 4096), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        c.write(B, r(1024, 16), Replica::GuestPages);
        c.write(B, r(3000, 8), Replica::GuestPages);
        let owed = c
            .transfer_for_read(B, r(0, 4096), Replica::DeviceOwned)
            .unwrap();
        assert_eq!(owed.bytes.ranges(), &[r(1024, 16), r(3000, 8)]);
        assert_eq!(owed.bytes.len(), 24, "24 bytes, not 4096");
    }

    /// A read from the replica that just wrote owes nothing, whatever else has
    /// happened to the backing.
    #[test]
    fn a_read_from_the_writer_is_always_fresh() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        c.write(B, r(0, 256), Replica::DeviceOwned);
        assert!(c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .is_none());
        let back = c
            .transfer_for_read(B, r(0, 256), Replica::GuestPages)
            .expect("the guest lost them when the device wrote");
        assert_eq!(back.from, Replica::DeviceOwned);
    }

    /// Bytes neither replica has ever held are not a transfer. Copying them
    /// would move whatever the source happens to contain and then claim the
    /// destination is current — which is worse than leaving both undefined,
    /// because the second one is honest.
    #[test]
    fn bytes_nobody_has_written_are_not_copied() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 64), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .expect("the declared part is owed");
        assert_eq!(
            t.bytes.ranges(),
            &[r(0, 64)],
            "only the declared extent moves; the rest was never anyone's"
        );
    }

    /// The two ways a read can owe nothing are two different facts, and a
    /// census that folded them together would report a device whose resources
    /// are read before they are written as a device whose freshness tracking is
    /// working. Every `None` lands in exactly one of them.
    #[test]
    fn a_read_over_content_nobody_wrote_is_counted_apart_from_a_fresh_one() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 64), Replica::GuestPages);
        assert!(
            c.transfer_for_read(B, r(64, 64), Replica::DeviceOwned)
                .is_none(),
            "nothing has written past the extent"
        );
        assert_eq!(c.census().reads_with_no_source, 1);
        assert_eq!(c.census().reads_already_fresh, 0);

        assert!(c
            .transfer_for_read(B, r(0, 64), Replica::GuestPages)
            .is_none());
        assert_eq!(c.census().reads_already_fresh, 1);
        assert_eq!(
            c.census().reads_with_no_source,
            1,
            "a fresh read is not a sourceless one"
        );

        // And a read that owes a transfer is neither.
        assert!(c
            .transfer_for_read(B, r(0, 64), Replica::DeviceOwned)
            .is_some());
        assert_eq!(c.census().reads_already_fresh, 1);
        assert_eq!(c.census().reads_with_no_source, 1);
        assert_eq!(c.census().transfers_planned, 1);

        // A backing the ledger has never heard of is the same fact as reading
        // past a declared extent, and lands in the same counter: it is the
        // read this number exists to make visible, not the one case exempt
        // from it.
        assert!(c
            .transfer_for_read(BackingId(99), r(0, 64), Replica::DeviceOwned)
            .is_none());
        assert_eq!(c.census().reads_with_no_source, 2);
        assert_eq!(c.census().reads_already_fresh, 1);
        assert!(
            !c.knows(BackingId(99)),
            "and asking did not invent a history for it"
        );
    }

    /// A discard drops a copy and not the content, so the version does not
    /// move and the other replica is still authoritative.
    #[test]
    fn a_discard_loses_a_copy_and_not_the_content() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        let version = c.version_of(B, r(0, 256));
        c.discard(B, r(0, 256), Replica::DeviceOwned);
        assert_eq!(c.version_of(B, r(0, 256)), version, "nothing was written");
        assert!(c.is_fresh(B, r(0, 256), Replica::GuestPages));
        assert!(c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .is_some());
    }

    /// Replace-physical re-declares: the guest re-pointed the resource, so
    /// every copy anyone held is of memory this resource no longer names.
    #[test]
    fn redeclaring_a_backing_invalidates_every_copy() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        c.declare(B, r(0, 256), Replica::GuestPages);
        assert!(!c.is_fresh(B, r(0, 1), Replica::DeviceOwned));
        assert_eq!(
            c.version_of(B, r(0, 256)),
            Some(ContentVersion(1)),
            "two declarations, so the second reserved the version after the \
             first; the transfer between them was not a version transition, \
             which is the whole reason it could not repeat"
        );
    }

    #[test]
    fn a_forgotten_backing_answers_nothing() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 256), Replica::GuestPages);
        c.forget(B);
        assert!(c
            .transfer_for_read(B, r(0, 256), Replica::DeviceOwned)
            .is_none());
        assert_eq!(c.newest_version(B), None);
    }
    /// Two resources placed in one heap share a backing, so a discard has to
    /// be about bytes. A whole-backing discard would throw away a neighbour's
    /// copy and charge it for a transfer it did not earn.
    #[test]
    fn discarding_one_placement_leaves_its_neighbour_fresh() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 512), Replica::GuestPages);
        let t = c
            .transfer_for_read(B, r(0, 512), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        c.discard(B, r(0, 256), Replica::DeviceOwned);
        assert!(!c.is_fresh(B, r(0, 256), Replica::DeviceOwned));
        assert!(c.is_fresh(B, r(256, 256), Replica::DeviceOwned));
    }

    /// The question a discard asks before taking a hint. Content only one
    /// replica holds is not a spare copy.
    #[test]
    fn sole_authority_is_the_bytes_nowhere_else_holds() {
        let mut c = ContentLedger::new();
        c.declare(B, r(0, 512), Replica::GuestPages);
        assert!(
            c.sole_authority(B, r(0, 512), Replica::DeviceOwned)
                .is_empty(),
            "the device holds nothing yet"
        );
        assert_eq!(
            c.sole_authority(B, r(0, 512), Replica::GuestPages).ranges(),
            &[r(0, 512)],
            "and the guest holds all of it alone"
        );
        let t = c
            .transfer_for_read(B, r(0, 512), Replica::DeviceOwned)
            .unwrap();
        c.record_transfer(&t);
        assert!(
            c.sole_authority(B, r(0, 512), Replica::GuestPages)
                .is_empty(),
            "a copy is a second holder"
        );
        // A device write of part of it makes that part the device's alone.
        c.write(B, r(128, 64), Replica::DeviceOwned);
        assert_eq!(
            c.sole_authority(B, r(0, 512), Replica::DeviceOwned)
                .ranges(),
            &[r(128, 64)]
        );
        assert_eq!(
            c.sole_authority(B, r(0, 128), Replica::DeviceOwned)
                .ranges(),
            &[],
            "and the question is bounded by the range asked about"
        );
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

    const EXTENT: usize = 24;
    const BACKINGS: u64 = 3;

    /// What each replica physically holds, byte by byte, beside what the
    /// content actually is.
    ///
    /// `truth[i]` is the version of the current content at byte `i`.
    /// `held[r][i]` is the version of the copy replica `r` physically has
    /// there, or `None` for no copy at all. They are the same number only while
    /// that replica's bytes are the content.
    #[derive(Clone)]
    struct Physical {
        truth: Vec<Option<u64>>,
        held: [Vec<Option<u64>>; 2],
    }

    impl Physical {
        fn new() -> Self {
            Self {
                truth: vec![None; EXTENT],
                held: [vec![None; EXTENT], vec![None; EXTENT]],
            }
        }
    }

    const fn index(r: Replica) -> usize {
        match r {
            Replica::GuestPages => 0,
            Replica::DeviceOwned => 1,
        }
    }

    /// **A replica the ledger calls fresh holds the current content.**
    ///
    /// The one claim the whole module is for, and the one no hand-written case
    /// can make: it is a property of every byte after every step of an
    /// arbitrary history, and the histories that break it are the ones where a
    /// plan and its completion are separated by something else.
    ///
    /// The shadow is deliberately *physical*. It tracks the version of the
    /// bytes each replica actually has, which is not what the ledger tracks —
    /// the ledger tracks which of them are current. A copy is modelled as
    /// delivering the source's content **as it was when the transfer was
    /// planned**, because that is the earliest the hardware could have read it
    /// and therefore the worst case the ledger has to be right about.
    ///
    /// A shadow that instead recorded what the ledger said would agree with it
    /// by construction and prove nothing.
    #[test]
    fn a_replica_the_ledger_calls_fresh_holds_the_current_bytes() {
        let mut declares = 0usize;
        let mut writes = 0usize;
        let mut planned = 0usize;
        let mut recorded = 0usize;
        let mut overtaken = 0usize;
        let mut discards = 0usize;
        let mut forgets = 0usize;
        let mut no_source = 0usize;

        for seed in 0..512u64 {
            let mut rng = Rng::new(seed);
            let mut c = ContentLedger::new();
            let mut phys: HashMap<BackingId, Physical> = HashMap::new();
            // Transfers planned and not yet recorded, with the source content
            // they will deliver.
            let mut in_flight: Vec<(Transfer, Vec<Option<u64>>)> = Vec::new();
            let mut clock = 0u64;

            for _ in 0..48 {
                let b = BackingId(rng.below(BACKINGS));
                match rng.below(40) {
                    // Declare, which is creation and resets every copy.
                    0..=3 => {
                        let authority = if rng.below(2) == 0 {
                            Replica::GuestPages
                        } else {
                            Replica::DeviceOwned
                        };
                        clock += 1;
                        c.declare(b, r(0, EXTENT as u64), authority);
                        let p = phys.entry(b).or_insert_with(Physical::new);
                        for i in 0..EXTENT {
                            p.truth[i] = Some(clock);
                            p.held[index(authority)][i] = Some(clock);
                            p.held[index(authority.other())][i] = None;
                        }
                        // A declaration is a statement that what was there is
                        // gone, so a copy still in flight over it delivers
                        // nothing the ledger may believe.
                        declares += 1;
                    }
                    // Write, which completes at once.
                    4..=13 => {
                        let Some(p) = phys.get_mut(&b) else { continue };
                        let offset = rng.below(EXTENT as u64);
                        let length = 1 + rng.below(EXTENT as u64 - offset);
                        let replica = if rng.below(2) == 0 {
                            Replica::GuestPages
                        } else {
                            Replica::DeviceOwned
                        };
                        clock += 1;
                        c.write(b, r(offset, length), replica);
                        for i in offset as usize..(offset + length) as usize {
                            p.truth[i] = Some(clock);
                            p.held[index(replica)][i] = Some(clock);
                        }
                        writes += 1;
                    }
                    // Plan a copy for a read.
                    14..=25 => {
                        let offset = rng.below(EXTENT as u64);
                        let length = 1 + rng.below(EXTENT as u64 - offset);
                        let replica = if rng.below(2) == 0 {
                            Replica::GuestPages
                        } else {
                            Replica::DeviceOwned
                        };
                        match c.transfer_for_read(b, r(offset, length), replica) {
                            Some(t) => {
                                assert_eq!(t.to, replica, "seed {seed}");
                                assert_eq!(t.from, replica.other(), "seed {seed}");
                                assert!(!t.bytes.is_empty(), "seed {seed}");
                                // What the source has right now is what the
                                // copy will carry.
                                let p = &phys[&b];
                                let carried = p.held[index(t.from)].clone();
                                for range in t.bytes.ranges() {
                                    let span = range.offset as usize
                                        ..(range.offset + range.length) as usize;
                                    assert!(
                                        carried[span].iter().all(Option::is_some),
                                        "seed {seed}: planned a copy of bytes the source \
                                         does not have"
                                    );
                                }
                                in_flight.push((t, carried));
                                planned += 1;
                            }
                            None => {
                                no_source += 1;
                            }
                        }
                    }
                    // Record one, in any order and however late.
                    26..=35 if !in_flight.is_empty() => {
                        let which = rng.below(in_flight.len() as u64) as usize;
                        let (t, carried) = in_flight.swap_remove(which);
                        c.record_transfer(&t);
                        if let Some(p) = phys.get_mut(&t.backing) {
                            let mut late = false;
                            for range in t.bytes.ranges() {
                                let span =
                                    range.offset as usize..(range.offset + range.length) as usize;
                                for (i, delivered) in carried[span].iter().enumerate() {
                                    let i = i + range.offset as usize;
                                    // The destination now physically holds what
                                    // the source had when this was planned.
                                    p.held[index(t.to)][i] = *delivered;
                                    if *delivered != p.truth[i] {
                                        late = true;
                                    }
                                }
                            }
                            if late {
                                overtaken += 1;
                            }
                        }
                        recorded += 1;
                    }
                    // Drop a replica's copy without changing the content.
                    36..=37 => {
                        let offset = rng.below(EXTENT as u64);
                        let length = 1 + rng.below(EXTENT as u64 - offset);
                        let replica = if rng.below(2) == 0 {
                            Replica::GuestPages
                        } else {
                            Replica::DeviceOwned
                        };
                        c.discard(b, r(offset, length), replica);
                        if let Some(p) = phys.get_mut(&b) {
                            for i in offset as usize..(offset + length) as usize {
                                p.held[index(replica)][i] = None;
                            }
                        }
                        discards += 1;
                    }
                    38 => {
                        c.forget(b);
                        phys.remove(&b);
                        assert!(!c.knows(b), "seed {seed}");
                        forgets += 1;
                    }
                    // A record whose queue was empty, and the arm that keeps
                    // an empty queue from turning into a forget.
                    _ => {}
                }

                // The claim, over every byte of every backing, after every step.
                for (backing, p) in &phys {
                    for replica in Replica::BOTH {
                        for i in 0..EXTENT {
                            if c.is_fresh(*backing, r(i as u64, 1), replica) {
                                assert_eq!(
                                    p.held[index(replica)][i],
                                    p.truth[i],
                                    "seed {seed}: {backing:?} byte {i} is fresh in \
                                     {} and holds the wrong content",
                                    replica.name()
                                );
                            }
                        }
                    }
                }
            }
        }

        // Non-vacuity: every shape the assertion above depends on reaching.
        assert!(declares > 2_000, "declarations: {declares}");
        assert!(writes > 2_400, "writes: {writes}");
        assert!(planned > 1_500, "copies planned: {planned}");
        assert!(recorded > 1_200, "copies recorded: {recorded}");
        assert!(
            overtaken > 250,
            "copies the content moved on under: {overtaken} --- the histories \
             this sweep exists for"
        );
        assert!(discards > 1_000, "discards: {discards}");
        assert!(forgets > 400, "forgets: {forgets}");
        assert!(no_source > 4_000, "reads with nothing to copy: {no_source}");
    }
}
