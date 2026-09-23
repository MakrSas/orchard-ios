//! The content-authority class, driven: over randomly generated histories, no
//! byte is copied twice without something having happened to it, and no byte
//! the guest can still read stops existing.
//!
//! # What is being claimed
//!
//! [`ContentLedger`](reims_vgpu_core::content::ContentLedger) states two
//! failures it exists to make impossible — a transfer with no version
//! transition, and a whole-resource copy answering a partial question — and
//! tests them at the module on hand-written histories. Both are *global*
//! claims about a sequence, not about one call: a ledger that answered every
//! individual question correctly and still re-copied under some interleaving
//! of declare, write, discard and read would pass every one of those tests.
//!
//! So this drives whole histories and checks the claims against an independent
//! account of the same facts. The shadow is deliberately *not* a second copy of
//! the ledger's design: it tracks one boolean per byte per replica — does this
//! replica hold the current content of this byte — with no version numbers and
//! no coverage map. A defect in the version arithmetic shows up here as a
//! disagreement about holdings, and a defect in the holdings shows up as one
//! about versions, because neither side is computing the other's answer.
//!
//! # The properties
//!
//! 1. **Nothing moves that the destination already holds.** Every byte of every
//!    planned transfer is one the destination lost — to a write elsewhere, or
//!    to a discard — since it last held it. This is the "no transfer without a
//!    version transition" claim in the only form that can be checked over a
//!    history rather than over a pair of calls.
//! 2. **A transfer is bounded by the question.** Its bytes lie inside the range
//!    read, are exactly the bytes the destination is behind on, and are all
//!    bytes the source itself holds. Bytes nobody holds are never moved.
//! 3. **A recorded transfer settles the read.** Repeating the same read
//!    immediately after recording owes nothing.
//! 4. **A write is a version transition.** Every byte a write names comes back
//!    at a strictly higher version than it had, and every byte it does not name
//!    keeps the version it had.
//! 5. **No lawful loss.** At every point in the history, every byte anyone has
//!    written is held by at least one replica — where the only discards issued
//!    are the ones the contract permits, which is those the discarding replica
//!    is not the sole authority for.
//!
//! # Why the discard case is a separate clause and not a hole
//!
//! A discard drops a copy without advancing the version, so property 1 cannot
//! be "a repeat needs a higher version" — that would be false, and correctly
//! so. It is stated as "the destination lost the bytes", and the shadow
//! distinguishes *how* it lost them: a loss to a write is additionally required
//! to have moved the version, which is checked as property 4. Between them
//! nothing is admitted that a version-only rule would have caught.

use reims_vgpu_core::access::{BackingId, ByteRange, ContentVersion};
use reims_vgpu_core::content::{ContentLedger, Replica, Transfer};
use std::collections::HashMap;

/// Deliberately tiny, so a byte-by-byte shadow is affordable and so generated
/// ranges collide constantly. Three backings share nothing; they are there so a
/// per-backing mistake — a lookup that answers about the wrong entry, a forget
/// that clears too much — has somewhere to show up.
const BACKINGS: [BackingId; 3] = [BackingId(1), BackingId(7), BackingId(9)];
const EXTENT: u64 = 96;

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

    /// A range inside the extent, biased towards short ones so that partial
    /// overlap is the common case rather than the exception.
    fn range(&mut self) -> ByteRange {
        let offset = self.below(EXTENT);
        let max = EXTENT - offset;
        let length = match self.below(4) {
            0 => max,
            1 => 1 + self.below(max),
            _ => 1 + self.below(max.min(24)),
        };
        ByteRange { offset, length }
    }
}

/// How a replica came to not hold a byte it once held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lost {
    /// Someone wrote it somewhere else. A version transition is owed.
    ToAWrite,
    /// A discard dropped the copy. The content did not change.
    ToADiscard,
}

/// One replica's independent account of one backing.
///
/// Not `Default`: an empty `Held` is not a backing nobody holds, it is a
/// backing with no bytes, and every read of it would be out of bounds.
#[derive(Clone, Debug)]
struct Held {
    /// Byte is current here.
    holds: Vec<bool>,
    /// For a byte not held, why — `None` if it was never held at all.
    lost: Vec<Option<Lost>>,
}

impl Held {
    fn new() -> Self {
        Self {
            holds: vec![false; EXTENT as usize],
            lost: vec![None; EXTENT as usize],
        }
    }
}

/// The shadow: who holds what, and which bytes anyone has ever written.
struct Shadow {
    guest: HashMap<BackingId, Held>,
    device: HashMap<BackingId, Held>,
    /// Bytes some declaration or write has given content to. A byte outside
    /// this has no content, so no replica owes it and losing it is not a loss.
    written: HashMap<BackingId, Vec<bool>>,
}

impl Shadow {
    fn new() -> Self {
        Self {
            guest: HashMap::new(),
            device: HashMap::new(),
            written: HashMap::new(),
        }
    }

    fn side(&mut self, replica: Replica) -> &mut HashMap<BackingId, Held> {
        match replica {
            Replica::GuestPages => &mut self.guest,
            Replica::DeviceOwned => &mut self.device,
        }
    }

    fn held(&self, backing: BackingId, byte: u64, replica: Replica) -> bool {
        let side = match replica {
            Replica::GuestPages => &self.guest,
            Replica::DeviceOwned => &self.device,
        };
        side.get(&backing).is_some_and(|h| h.holds[byte as usize])
    }

    fn lost_reason(&self, backing: BackingId, byte: u64, replica: Replica) -> Option<Lost> {
        let side = match replica {
            Replica::GuestPages => &self.guest,
            Replica::DeviceOwned => &self.device,
        };
        side.get(&backing).and_then(|h| h.lost[byte as usize])
    }

    fn has_content(&self, backing: BackingId, byte: u64) -> bool {
        self.written.get(&backing).is_some_and(|w| w[byte as usize])
    }

    fn gain(&mut self, backing: BackingId, bytes: ByteRange, replica: Replica) {
        let side = self.side(replica).entry(backing).or_insert_with(Held::new);
        for b in bytes.offset..bytes.offset + bytes.length {
            side.holds[b as usize] = true;
            side.lost[b as usize] = None;
        }
    }

    fn drop_copy(&mut self, backing: BackingId, bytes: ByteRange, replica: Replica, why: Lost) {
        let side = self.side(replica).entry(backing).or_insert_with(Held::new);
        for b in bytes.offset..bytes.offset + bytes.length {
            if side.holds[b as usize] {
                side.lost[b as usize] = Some(why);
            }
            side.holds[b as usize] = false;
        }
    }

    fn mark_written(&mut self, backing: BackingId, bytes: ByteRange) {
        let w = self
            .written
            .entry(backing)
            .or_insert_with(|| vec![false; EXTENT as usize]);
        for b in bytes.offset..bytes.offset + bytes.length {
            w[b as usize] = true;
        }
    }

    fn declare(&mut self, backing: BackingId, extent: ByteRange, authority: Replica) {
        // A backing this shadow already answers for is *joined*: a second name
        // for one backing is a claim about the bytes it names, and the bytes
        // it does not name keep whatever they held. Written from what the
        // guest did — a name appearing over pages another name also holds — and
        // not from what the ledger does with it.
        if self.written.contains_key(&backing) {
            self.write(backing, extent, authority);
            return;
        }
        // A backing nothing knows has nothing to lose: no replica holds
        // anything of it, and nothing outside the new extent has content.
        for side in [Replica::GuestPages, Replica::DeviceOwned] {
            self.side(side).insert(backing, Held::new());
        }
        self.written.insert(backing, vec![false; EXTENT as usize]);
        self.gain(backing, extent, authority);
        self.mark_written(backing, extent);
    }

    fn write(&mut self, backing: BackingId, bytes: ByteRange, replica: Replica) {
        self.gain(backing, bytes, replica);
        self.drop_copy(backing, bytes, replica.other(), Lost::ToAWrite);
        self.mark_written(backing, bytes);
    }

    fn forget(&mut self, backing: BackingId) {
        self.guest.remove(&backing);
        self.device.remove(&backing);
        self.written.remove(&backing);
    }
}

/// How much of each shape the sweep actually produced. A sweep whose generator
/// drifts into never issuing a discard, or into never asking a read that is
/// already fresh, still passes every assertion above while testing much less
/// than it reads as testing.
#[derive(Default)]
struct Counts {
    declares: usize,
    writes: usize,
    reads: usize,
    transfers_planned: usize,
    transfers_recorded: usize,
    reads_already_fresh: usize,
    bytes_moved: u64,
    discards_offered: usize,
    discards_issued: usize,
    discards_refused_as_sole_authority: usize,
    repeat_transfers_after_a_write: usize,
    repeat_transfers_after_a_discard: usize,
    forgets: usize,
    /// Reads naming bytes no replica has ever held. The shape that separates
    /// "copy the bytes the source has" from "copy everything the destination
    /// lacks", and the only reason the distinction is observable at all.
    reads_over_uncreated_bytes: usize,
}

/// The part of `range` that lies inside `bound`, or `None` if none does.
fn clip(range: ByteRange, bound: ByteRange) -> Option<ByteRange> {
    let start = range.offset.max(bound.offset);
    let end = (range.offset + range.length).min(bound.offset + bound.length);
    (start < end).then(|| ByteRange {
        offset: start,
        length: end - start,
    })
}

fn byte_at(offset: u64) -> ByteRange {
    ByteRange { offset, length: 1 }
}

/// Every byte the ledger says `replica` is fresh for, rebuilt one byte at a
/// time. `is_fresh` is the only freshness the ledger exposes, and asking it per
/// byte is what turns it into a set this test can compare against its own.
fn fresh_bytes(ledger: &ContentLedger, backing: BackingId, replica: Replica) -> Vec<bool> {
    (0..EXTENT)
        .map(|b| ledger.is_fresh(backing, byte_at(b), replica))
        .collect()
}

fn versions(ledger: &ContentLedger, backing: BackingId) -> Vec<Option<ContentVersion>> {
    (0..EXTENT)
        .map(|b| ledger.version_of(backing, byte_at(b)))
        .collect()
}

/// Property 2, checked against the ledger's own freshness rather than the
/// shadow's, so that a transfer that is bounded correctly with respect to a
/// wrong freshness is still caught by property 1.
fn check_transfer_is_bounded(
    ledger: &ContentLedger,
    t: &Transfer,
    read: ByteRange,
    replica: Replica,
) {
    assert_eq!(
        t.to, replica,
        "a transfer serves the read that asked for it"
    );
    assert_eq!(t.from, replica.other(), "there are only two replicas");
    assert!(!t.bytes.is_empty(), "an empty transfer is not a transfer");
    for r in t.bytes.ranges() {
        for b in r.offset..r.offset + r.length {
            assert!(
                b >= read.offset && b < read.offset + read.length,
                "byte {b} of backing {:?} moved for a read of {read:?} that did \
                 not name it",
                t.backing
            );
            assert!(
                ledger.is_fresh(t.backing, byte_at(b), t.from),
                "byte {b} moved from a replica that does not hold it, which \
                 copies whatever the source happens to contain"
            );
        }
    }
}

fn run(seed: u64, steps: usize, counts: &mut Counts) {
    let mut ledger = ContentLedger::new();
    let mut shadow = Shadow::new();
    let mut rng = Rng::new(seed);
    // The ledger keeps a census of itself, and it is the number a placement
    // policy would be tuned against, so a census that drifts is a wrong answer
    // even where every transfer is right. Counted per run, because the ledger
    // is per run.
    let (mut asked_and_owed, mut asked_and_free, mut owed_bytes) = (0usize, 0usize, 0u64);
    let mut asked_with_no_source = 0usize;
    let mut writes_here = 0usize;

    // Every backing starts declared, over part of the byte space rather than
    // all of it: a resource is its extent, and the bytes outside it have no
    // content anywhere. Re-declaration is generated below.
    for b in BACKINGS {
        let extent = rng.range();
        ledger.declare(b, extent, Replica::GuestPages);
        shadow.declare(b, extent, Replica::GuestPages);
        counts.declares += 1;
    }

    for _ in 0..steps {
        let backing = BACKINGS[rng.below(BACKINGS.len() as u64) as usize];
        let replica = if rng.below(2) == 0 {
            Replica::GuestPages
        } else {
            Replica::DeviceOwned
        };
        // Reads dominate, because the property under test is about what reads
        // are charged and a history of mostly writes asks the question rarely.
        // Declarations and forgets are rare on purpose: each one throws the
        // backing's whole history away, and a generator that reset every
        // sixteenth step would only ever test shallow ones.
        match rng.below(100) {
            0 => {
                // A second name over pages this backing already answers for,
                // which joins rather than resets.
                let extent = rng.range();
                ledger.declare(backing, extent, replica);
                shadow.declare(backing, extent, replica);
                counts.declares += 1;
                // A join *is* a write: it advances the version over the bytes
                // it names and takes them from the other replica, which is
                // what the census counts and what makes the cost visible.
                writes_here += 1;
            }
            1 => {
                // And the creation the clearing form is for: a backing nothing
                // knows, which is the only way to reach it now that a known
                // one is joined.
                let extent = rng.range();
                ledger.forget(backing);
                shadow.forget(backing);
                counts.forgets += 1;
                ledger.declare(backing, extent, replica);
                shadow.declare(backing, extent, replica);
                counts.declares += 1;
            }
            2 => {
                ledger.forget(backing);
                shadow.forget(backing);
                counts.forgets += 1;
                // And re-declare it, so the rest of the history has a backing
                // to talk about. The forget is still under test: everything it
                // was holding has to be gone.
                assert_eq!(
                    ledger.newest_version(backing),
                    None,
                    "a forgotten backing answers nothing"
                );
                let extent = rng.range();
                ledger.declare(backing, extent, replica);
                shadow.declare(backing, extent, replica);
                counts.declares += 1;
            }
            3..=24 => {
                // Inside the declared extent, because that is what a resource's
                // bytes are. Leaving the rest of the address space uncreated is
                // what lets a read below name bytes no replica has ever held —
                // the case where a transfer must move nothing rather than
                // moving whatever the source happens to contain.
                let Some(bytes) = ledger.extent(backing).and_then(|e| clip(rng.range(), e)) else {
                    continue;
                };
                let before = versions(&ledger, backing);
                ledger.write(backing, bytes, replica);
                shadow.write(backing, bytes, replica);
                counts.writes += 1;
                writes_here += 1;

                // Property 4.
                let after = versions(&ledger, backing);
                for b in 0..EXTENT {
                    let inside = b >= bytes.offset && b < bytes.offset + bytes.length;
                    if inside {
                        let now = after[b as usize].expect("a written byte has a version");
                        assert!(
                            before[b as usize].is_none_or(|was| now > was),
                            "byte {b} was written and its version did not advance"
                        );
                    } else {
                        assert_eq!(
                            before[b as usize], after[b as usize],
                            "byte {b} was not written and its version moved"
                        );
                    }
                }
            }
            25..=34 => {
                // A discard, offered only where the contract permits one: the
                // bytes this replica is the sole authority for are content that
                // exists nowhere else, and dropping them is a loss rather than
                // a saving.
                let bytes = rng.range();
                counts.discards_offered += 1;
                let sole = ledger.sole_authority(backing, bytes, replica);
                if sole.is_empty() {
                    ledger.discard(backing, bytes, replica);
                    shadow.drop_copy(backing, bytes, replica, Lost::ToADiscard);
                    counts.discards_issued += 1;
                } else {
                    counts.discards_refused_as_sole_authority += 1;
                    // And the refusal is about bytes: the part it is not sole
                    // authority for is still droppable.
                    let droppable = sole.missing_from(bytes);
                    for r in droppable.ranges() {
                        ledger.discard(backing, *r, replica);
                        shadow.drop_copy(backing, *r, replica, Lost::ToADiscard);
                    }
                    if !droppable.is_empty() {
                        counts.discards_issued += 1;
                    }
                }
            }
            _ => {
                let bytes = rng.range();
                counts.reads += 1;
                if (bytes.offset..bytes.offset + bytes.length)
                    .any(|b| !shadow.has_content(backing, b))
                {
                    counts.reads_over_uncreated_bytes += 1;
                }
                let planned = ledger.transfer_for_read(backing, bytes, replica);
                let Some(t) = planned else {
                    // The two shapes of "nothing owed" are counted apart,
                    // because the ledger's census counts them apart and the
                    // sweep is what says the split is exhaustive.
                    let behind: Vec<u64> = (bytes.offset..bytes.offset + bytes.length)
                        .filter(|b| !shadow.held(backing, *b, replica))
                        .collect();
                    if behind.is_empty() {
                        counts.reads_already_fresh += 1;
                        asked_and_free += 1;
                    } else {
                        asked_with_no_source += 1;
                    }
                    // Nothing is owed, so either the destination holds every
                    // byte of the read, or the bytes it lacks have no content
                    // anywhere.
                    for b in bytes.offset..bytes.offset + bytes.length {
                        assert!(
                            shadow.held(backing, b, replica)
                                || !shadow.held(backing, b, replica.other()),
                            "byte {b} was readable from the other replica and no \
                             transfer was planned"
                        );
                    }
                    continue;
                };
                counts.transfers_planned += 1;
                counts.bytes_moved += t.bytes.len();
                asked_and_owed += 1;
                owed_bytes += t.bytes.len();
                check_transfer_is_bounded(&ledger, &t, bytes, replica);

                // Property 1, and the reason the shadow records *how* a byte
                // was lost.
                for r in t.bytes.ranges() {
                    for b in r.offset..r.offset + r.length {
                        assert!(
                            !shadow.held(backing, b, replica),
                            "byte {b} was copied to a replica that already held it"
                        );
                        match shadow.lost_reason(backing, b, replica) {
                            Some(Lost::ToAWrite) => counts.repeat_transfers_after_a_write += 1,
                            Some(Lost::ToADiscard) => counts.repeat_transfers_after_a_discard += 1,
                            None => {}
                        }
                    }
                }

                // Property 2's other half: exactly the bytes behind, no fewer.
                // A transfer that under-copies is the failure that leaves a
                // read serving stale bytes, and it is not caught by anything
                // above.
                for b in bytes.offset..bytes.offset + bytes.length {
                    let owed = !shadow.held(backing, b, replica)
                        && shadow.held(backing, b, replica.other());
                    let moving = t.bytes.covers(byte_at(b));
                    assert_eq!(
                        owed, moving,
                        "byte {b}: the shadow says owed={owed} and the transfer \
                         says moving={moving}"
                    );
                }

                // Recording it is what makes it not owed again — sometimes.
                // A planned transfer the device never performs must stay owed,
                // which is the other half of the same rule.
                if rng.below(4) == 0 {
                    let again = ledger.transfer_for_read(backing, bytes, replica);
                    asked_and_owed += 1;
                    owed_bytes += again.as_ref().map_or(0, |a| a.bytes.len());
                    assert_eq!(
                        again.as_ref().map(|a| a.bytes.len()),
                        Some(t.bytes.len()),
                        "an unperformed transfer stopped being owed"
                    );
                    continue;
                }
                ledger.record_transfer(&t);
                for r in t.bytes.ranges() {
                    shadow.gain(backing, *r, replica);
                }
                counts.transfers_recorded += 1;

                // Property 3. Classified the same way as any other read that
                // owed nothing: the bytes outside the declared extent are still
                // not held by anyone, and a recorded transfer does not change
                // that.
                if (bytes.offset..bytes.offset + bytes.length)
                    .all(|b| shadow.held(backing, b, replica))
                {
                    asked_and_free += 1;
                } else {
                    asked_with_no_source += 1;
                }
                assert!(
                    ledger.transfer_for_read(backing, bytes, replica).is_none(),
                    "the same read owed a second transfer immediately after the \
                     first was recorded"
                );
            }
        }

        // Properties 5 and the shadow agreement, after every step.
        for b in BACKINGS {
            let ledger_guest = fresh_bytes(&ledger, b, Replica::GuestPages);
            let ledger_device = fresh_bytes(&ledger, b, Replica::DeviceOwned);
            for byte in 0..EXTENT {
                assert_eq!(
                    ledger_guest[byte as usize],
                    shadow.held(b, byte, Replica::GuestPages),
                    "backing {b:?} byte {byte}: the ledger and the shadow \
                     disagree about the guest"
                );
                assert_eq!(
                    ledger_device[byte as usize],
                    shadow.held(b, byte, Replica::DeviceOwned),
                    "backing {b:?} byte {byte}: the ledger and the shadow \
                     disagree about the device"
                );
                if shadow.has_content(b, byte) {
                    assert!(
                        ledger_guest[byte as usize] || ledger_device[byte as usize],
                        "backing {b:?} byte {byte} has content and no replica \
                         holds it: the guest can still read bytes this device \
                         has lost"
                    );
                }
            }
        }
    }

    let census = ledger.census();
    assert_eq!(census.writes, writes_here, "writes recorded");
    assert_eq!(
        census.transfers_planned, asked_and_owed,
        "transfers planned"
    );
    assert_eq!(
        census.transfer_bytes, owed_bytes,
        "bytes in those transfers"
    );
    assert_eq!(
        census.reads_already_fresh, asked_and_free,
        "reads that were already fresh"
    );
    assert_eq!(
        census.reads_with_no_source, asked_with_no_source,
        "reads whose missing bytes nobody held"
    );
}

#[test]
fn no_byte_is_copied_twice_without_losing_it_first() {
    let mut counts = Counts::default();
    for seed in 0..64u64 {
        run(seed, 600, &mut counts);
    }

    // Non-vacuity. Every one of these is a shape an assertion above depends on
    // having been reached; a generator change that stops producing one turns
    // the corresponding property into a comment.
    assert!(counts.declares > 600, "{}", counts.declares);
    assert!(counts.forgets > 150, "{}", counts.forgets);
    assert!(counts.writes > 2_000, "{}", counts.writes);
    assert!(counts.reads > 12_000, "{}", counts.reads);
    assert!(
        counts.transfers_planned > 2_500,
        "{}",
        counts.transfers_planned
    );
    assert!(
        counts.transfers_recorded > 1_900,
        "{}",
        counts.transfers_recorded
    );
    assert!(
        counts.reads_already_fresh > 1_200,
        "reads that were already fresh: {}",
        counts.reads_already_fresh
    );
    assert!(
        counts.repeat_transfers_after_a_write > 13_000,
        "bytes re-copied because someone wrote them elsewhere: {}",
        counts.repeat_transfers_after_a_write
    );
    assert!(
        counts.repeat_transfers_after_a_discard > 3_000,
        "bytes re-copied because a discard dropped them: {}",
        counts.repeat_transfers_after_a_discard
    );
    assert!(
        counts.reads_over_uncreated_bytes > 10_000,
        "reads naming bytes nobody has ever written: {}",
        counts.reads_over_uncreated_bytes
    );
    assert!(counts.discards_issued > 1_800, "{}", counts.discards_issued);
    assert!(
        counts.discards_refused_as_sole_authority > 400,
        "discards the contract would not permit: {}",
        counts.discards_refused_as_sole_authority
    );

    // And the whole point of the per-byte freshness: the bytes actually moved
    // are a fraction of what a per-resource dirty flag would have moved, which
    // is one whole extent per transfer.
    let whole_resource = counts.transfers_planned as u64 * EXTENT;
    assert!(
        counts.bytes_moved * 4 < whole_resource,
        "{} bytes moved against {whole_resource} for a per-resource flag; the \
         per-byte freshness is not buying anything",
        counts.bytes_moved
    );
}
