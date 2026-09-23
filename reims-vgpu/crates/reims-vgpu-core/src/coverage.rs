//! Which content version is current in each part of a backing.
//!
//! # One version per backing is the wrong number
//!
//! [`crate::content`] already answers "which *bytes* is a replica behind on"
//! per byte, because a whole-resource copy to answer a partial question is the
//! cost that model exists to avoid. What it kept per *backing* was a single
//! version, and that number cannot be right for more than one writer at a time.
//!
//! Two transactions write disjoint ranges of one buffer — a compositor
//! updating two damage rectangles, a heap with two resources placed in it — and
//! both results are current. Under a scalar version they are not: the second
//! writer's reservation is the higher number, so the first writer's completion
//! arrives holding a version the backing has already passed. Either it is
//! rejected, and the bytes it produced are lost while the ledger says the
//! backing is current; or it is accepted, and the backing's version goes
//! backwards over memory the second writer owns. There is no third answer,
//! because the question was asked of the wrong granularity.
//!
//! So the version is a *coverage*: a map from disjoint byte ranges to the
//! version current in each. Disjoint writers cover disjoint ranges and never
//! meet, and two writers that do overlap meet on exactly the bytes they share.
//! The same [`crate::access::AccessKey::may_alias`] that decides whether they
//! need an ordering edge decides whether their versions collide, which is what
//! keeps the hazard graph and the content authority from disagreeing about who
//! overlapped whom.
//!
//! # Newer wins, and the loser says what it lost
//!
//! [`VersionCoverage::apply`] takes coverage only where the range is uncovered
//! or covered at a strictly older version. That is the stale-completion rule:
//! a GPU write that reserved version 5 and completed after a guest write
//! reserved 6 must not replace the guest's bytes, because the guest's are
//! newer and the guest can read them.
//!
//! It returns both halves. A completion that took nothing is not the same
//! event as one that took everything, and an authority that silently discarded
//! the difference would leave "this transaction's bytes never landed" invisible
//! — which for a GPU write means a frame the guest will never see with no line
//! anywhere saying so.
//!
//! # Adjacent equal versions coalesce
//!
//! Not an optimisation with a threshold: it is what keeps the representation a
//! function of the content rather than of the order the writes arrived in. Two
//! halves written separately at one version and one whole written at that
//! version are the same content, and a map that told them apart would make
//! equality, tests and censuses depend on history.

use crate::access::{ByteRange, ContentVersion};
use crate::range_set::RangeSet;

/// One contiguous run of a backing that is current at one version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub range: ByteRange,
    pub version: ContentVersion,
}

/// What [`VersionCoverage::apply`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    /// Bytes this version now covers: they were uncovered, or covered older.
    pub taken: RangeSet,
    /// Bytes it did not, because something at least as new already covers
    /// them. Named rather than dropped — see the module doc.
    pub refused: RangeSet,
}

impl Applied {
    /// Whether the write landed nowhere at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.taken.is_empty()
    }

    /// Whether some part of the write was beaten by newer content.
    #[must_use]
    pub fn was_partly_stale(&self) -> bool {
        !self.refused.is_empty()
    }
}

/// The version current in each part of one backing.
///
/// Sorted by offset, disjoint, and no two adjacent spans share a version. Every
/// operation restores all three before returning, for the reason
/// [`crate::range_set`] gives about its own: an invariant that holds only most
/// of the time is one whose violation shows up as a wrong answer somewhere
/// else.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VersionCoverage {
    spans: Vec<Span>,
}

impl VersionCoverage {
    #[must_use]
    pub const fn new() -> Self {
        Self { spans: Vec::new() }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Every covered run, in offset order.
    #[must_use]
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Forget every version. What the content itself is does not change; this
    /// is the coverage half of a re-declaration.
    pub fn clear(&mut self) {
        self.spans.clear();
    }

    /// The highest version covering any part of `range`.
    ///
    /// `None` where no part of it is covered — which is a backing nothing has
    /// written, not a backing at version zero. The two are different: version
    /// zero is a real version a write can produce, and folding "never written"
    /// onto it would make an unwritten range beat nothing and be beaten by
    /// everything alike.
    #[must_use]
    pub fn newest_over(&self, range: ByteRange) -> Option<ContentVersion> {
        self.overlapping(range).map(|(_, span)| span.version).max()
    }

    /// The versions covering `range`, clipped to it, in offset order.
    ///
    /// The snapshot a read takes: a reader needs the version of each part of
    /// what it reads, because a representation is usable only where its own
    /// coverage matches.
    pub fn over(&self, range: ByteRange) -> impl Iterator<Item = Span> + '_ {
        self.overlapping(range)
            .map(|(clipped, span)| Span {
                range: clipped,
                version: span.version,
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// The parts of `range` no version covers.
    #[must_use]
    pub fn uncovered(&self, range: ByteRange) -> RangeSet {
        // `overlapping` walks the spans in order, so the clipped pieces
        // already arrive ascending.
        let mut covered = RangeSet::new();
        for (clipped, _) in self.overlapping(range) {
            covered.push_ascending(clipped);
        }
        covered.missing_from(range)
    }

    /// Make `version` current over `range`, wherever nothing newer is.
    ///
    /// Uncovered bytes are taken. Bytes covered at a strictly older version are
    /// taken. Bytes covered at the same version or a newer one are refused, and
    /// both halves come back — see the module doc for why the refusal is not
    /// silence.
    ///
    /// Equal versions refuse rather than take. A second claim on a version
    /// already committed is either the same write arriving twice or two writers
    /// that were handed one reservation, and taking it would make those
    /// indistinguishable from the first landing.
    pub fn apply(&mut self, range: ByteRange, version: ContentVersion) -> Applied {
        let mut out = Applied::default();
        if range.length == 0 {
            return out;
        }
        let end = range.offset.saturating_add(range.length);
        let mut cursor = range.offset;
        for span in &self.spans {
            let start = span.range.offset;
            let stop = start.saturating_add(span.range.length);
            if stop <= cursor {
                continue;
            }
            if start >= end {
                break;
            }
            if start > cursor {
                // A gap in the coverage: nothing holds these bytes.
                out.taken.push_ascending(ByteRange {
                    offset: cursor,
                    length: start - cursor,
                });
                cursor = start;
            }
            let overlap_end = end.min(stop);
            if cursor < overlap_end {
                let piece = ByteRange {
                    offset: cursor,
                    length: overlap_end - cursor,
                };
                if span.version < version {
                    out.taken.push_ascending(piece);
                } else {
                    out.refused.push_ascending(piece);
                }
                cursor = overlap_end;
            }
        }
        if cursor < end {
            out.taken.push_ascending(ByteRange {
                offset: cursor,
                length: end - cursor,
            });
        }
        if !out.taken.is_empty() {
            self.write_spans(&out.taken, version);
        }
        out
    }

    /// Every existing span clipped to `range`, with the span it came from.
    fn overlapping(&self, range: ByteRange) -> impl Iterator<Item = (ByteRange, &Span)> + '_ {
        let end = range.offset.saturating_add(range.length);
        self.spans.iter().filter_map(move |span| {
            let start = span.range.offset.max(range.offset);
            let stop = span.range.offset.saturating_add(span.range.length).min(end);
            // `then`, not `then_some`: the argument of `then_some` is
            // evaluated whatever the condition says, and for a span that does
            // not reach the range at all the length below underflows.
            (start < stop).then(|| {
                (
                    ByteRange {
                        offset: start,
                        length: stop - start,
                    },
                    span,
                )
            })
        })
    }

    /// Replace the coverage of `bytes` with `version`, and restore the
    /// invariants.
    fn write_spans(&mut self, bytes: &RangeSet, version: ContentVersion) {
        let mut next: Vec<Span> = Vec::with_capacity(self.spans.len() + bytes.ranges().len());
        for span in &self.spans {
            // What is left of this span once the new version has taken its
            // share. A span fully overwritten leaves nothing and disappears.
            // `missing_from` is that question already: the parts of a range a
            // set does not cover.
            let left = bytes.missing_from(span.range);
            for range in left.ranges() {
                next.push(Span {
                    range: *range,
                    version: span.version,
                });
            }
        }
        for range in bytes.ranges() {
            next.push(Span {
                range: *range,
                version,
            });
        }
        next.sort_unstable_by_key(|s| s.range.offset);
        self.spans = coalesce(next);
    }
}

/// Merge runs that touch and agree. Input must be sorted and disjoint.
fn coalesce(sorted: Vec<Span>) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::with_capacity(sorted.len());
    for span in sorted {
        if span.range.length == 0 {
            continue;
        }
        if let Some(last) = out.last_mut() {
            let last_end = last.range.offset.saturating_add(last.range.length);
            if last_end == span.range.offset && last.version == span.version {
                last.range.length = last.range.length.saturating_add(span.range.length);
                continue;
            }
        }
        out.push(span);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(offset: u64, length: u64) -> ByteRange {
        ByteRange { offset, length }
    }

    fn v(n: u64) -> ContentVersion {
        ContentVersion(n)
    }

    /// Every operation leaves the map sorted, disjoint and coalesced.
    fn invariants(c: &VersionCoverage) {
        let mut previous_end = 0u64;
        let mut previous_version: Option<ContentVersion> = None;
        for (index, span) in c.spans().iter().enumerate() {
            assert!(span.range.length > 0, "empty span at {index}");
            if index > 0 {
                assert!(
                    span.range.offset >= previous_end,
                    "span {index} overlaps or precedes its neighbour"
                );
                if span.range.offset == previous_end {
                    assert_ne!(
                        Some(span.version),
                        previous_version,
                        "span {index} touches its neighbour at the same version and was not coalesced"
                    );
                }
            }
            previous_end = span.range.offset + span.range.length;
            previous_version = Some(span.version);
        }
    }

    /// Nothing has been written, so nothing has a version — and that is not
    /// version zero.
    #[test]
    fn an_empty_coverage_reports_no_version_rather_than_the_first_one() {
        let c = VersionCoverage::new();
        assert!(c.is_empty());
        assert_eq!(c.newest_over(r(0, 100)), None);
        assert_eq!(c.uncovered(r(0, 100)).ranges(), &[r(0, 100)]);
        assert_eq!(c.over(r(0, 100)).count(), 0);
    }

    /// The failure this type exists for: two writers of disjoint ranges are
    /// both current, whatever order their completions arrive in.
    ///
    /// Under one version per backing the earlier reservation loses — either its
    /// bytes are dropped while the ledger claims the backing is current, or the
    /// backing's version goes backwards over memory the later writer owns.
    #[test]
    fn two_disjoint_writers_are_both_current_in_either_completion_order() {
        for late_first in [false, true] {
            let mut c = VersionCoverage::new();
            let (first, second) = if late_first {
                ((r(64, 64), v(6)), (r(0, 64), v(5)))
            } else {
                ((r(0, 64), v(5)), (r(64, 64), v(6)))
            };
            assert!(c.apply(first.0, first.1).refused.is_empty());
            let later = c.apply(second.0, second.1);
            assert!(
                later.refused.is_empty(),
                "disjoint writers must not meet ({late_first})"
            );
            invariants(&c);
            assert_eq!(c.newest_over(r(0, 64)), Some(v(5)));
            assert_eq!(c.newest_over(r(64, 64)), Some(v(6)));
            assert!(c.uncovered(r(0, 128)).is_empty());
        }
    }

    /// The stale-completion rule. A write that reserved an older version and
    /// completed later does not replace the newer content, and says so.
    #[test]
    fn an_older_completion_does_not_replace_newer_content_and_names_what_it_lost() {
        let mut c = VersionCoverage::new();
        c.apply(r(0, 128), v(9));
        let stale = c.apply(r(32, 64), v(4));
        assert!(stale.is_empty(), "nothing older may land");
        assert!(stale.was_partly_stale());
        assert_eq!(stale.refused.ranges(), &[r(32, 64)]);
        invariants(&c);
        assert_eq!(c.newest_over(r(0, 128)), Some(v(9)));
        // And the map is untouched: one span, not three.
        assert_eq!(c.spans().len(), 1);
    }

    /// A completion may be partly stale, and the two halves are separate
    /// answers. The bytes it did land are current at its version.
    #[test]
    fn a_partly_overlapping_completion_lands_where_it_is_the_newest() {
        let mut c = VersionCoverage::new();
        c.apply(r(64, 64), v(9));
        let mixed = c.apply(r(0, 128), v(5));
        assert_eq!(mixed.taken.ranges(), &[r(0, 64)]);
        assert_eq!(mixed.refused.ranges(), &[r(64, 64)]);
        invariants(&c);
        assert_eq!(c.newest_over(r(0, 64)), Some(v(5)));
        assert_eq!(c.newest_over(r(64, 64)), Some(v(9)));
    }

    /// The same version arriving twice is refused, not taken again: a
    /// duplicate landing must not read as a first one.
    #[test]
    fn a_version_already_committed_is_refused_rather_than_reapplied() {
        let mut c = VersionCoverage::new();
        assert_eq!(c.apply(r(0, 64), v(3)).taken.ranges(), &[r(0, 64)]);
        let again = c.apply(r(0, 64), v(3));
        assert!(again.is_empty());
        assert_eq!(again.refused.ranges(), &[r(0, 64)]);
        invariants(&c);
    }

    /// A newer write covering an older one leaves one span, not two: the
    /// representation is a function of the content and not of the history.
    #[test]
    fn a_newer_write_over_an_older_one_leaves_one_span() {
        let mut c = VersionCoverage::new();
        c.apply(r(0, 64), v(1));
        c.apply(r(0, 64), v(2));
        invariants(&c);
        assert_eq!(
            c.spans(),
            &[Span {
                range: r(0, 64),
                version: v(2)
            }]
        );
    }

    /// Two halves written separately and one whole written at once are the
    /// same content, so they are the same map.
    #[test]
    fn adjacent_equal_versions_coalesce_however_they_were_written() {
        let mut halves = VersionCoverage::new();
        halves.apply(r(0, 64), v(7));
        halves.apply(r(64, 64), v(7));
        let mut whole = VersionCoverage::new();
        whole.apply(r(0, 128), v(7));
        invariants(&halves);
        invariants(&whole);
        assert_eq!(halves, whole);
        assert_eq!(halves.spans().len(), 1);
    }

    /// A write through the middle of one span splits it and keeps both ends at
    /// the version they were.
    #[test]
    fn a_write_through_the_middle_of_a_span_keeps_both_ends() {
        let mut c = VersionCoverage::new();
        c.apply(r(0, 300), v(1));
        c.apply(r(100, 100), v(2));
        invariants(&c);
        assert_eq!(
            c.spans(),
            &[
                Span {
                    range: r(0, 100),
                    version: v(1)
                },
                Span {
                    range: r(100, 100),
                    version: v(2)
                },
                Span {
                    range: r(200, 100),
                    version: v(1)
                },
            ]
        );
        let snapshot: Vec<Span> = c.over(r(50, 100)).collect();
        assert_eq!(
            snapshot,
            vec![
                Span {
                    range: r(50, 50),
                    version: v(1)
                },
                Span {
                    range: r(100, 50),
                    version: v(2)
                },
            ]
        );
    }

    /// A gap between two spans is uncovered, and a write across the gap takes
    /// the gap and the older span alike.
    #[test]
    fn a_gap_between_spans_is_uncovered_and_a_write_across_it_fills_it() {
        let mut c = VersionCoverage::new();
        c.apply(r(0, 32), v(1));
        c.apply(r(96, 32), v(4));
        invariants(&c);
        assert_eq!(c.uncovered(r(0, 128)).ranges(), &[r(32, 64)]);
        assert_eq!(c.newest_over(r(0, 128)), Some(v(4)));

        let across = c.apply(r(0, 128), v(3));
        // The gap and the older span; not the newer one at the end.
        assert_eq!(across.taken.ranges(), &[r(0, 96)]);
        assert_eq!(across.refused.ranges(), &[r(96, 32)]);
        invariants(&c);
        assert!(c.uncovered(r(0, 128)).is_empty());
    }

    /// A zero-length write is not a write. Nothing is taken, nothing is
    /// refused, and the map does not gain an empty span that would break the
    /// coalescing rule for its neighbours.
    #[test]
    fn a_zero_length_write_changes_nothing() {
        let mut c = VersionCoverage::new();
        c.apply(r(0, 64), v(1));
        let nothing = c.apply(r(32, 0), v(9));
        assert!(nothing.is_empty());
        assert!(!nothing.was_partly_stale());
        invariants(&c);
        assert_eq!(c.spans().len(), 1);
        assert_eq!(c.newest_over(r(0, 64)), Some(v(1)));
    }

    /// A range past everything covered is uncovered, and asking about it does
    /// not walk off the end.
    #[test]
    fn a_range_past_the_coverage_is_uncovered() {
        let mut c = VersionCoverage::new();
        c.apply(r(0, 64), v(1));
        assert_eq!(c.newest_over(r(1000, 64)), None);
        assert_eq!(c.uncovered(r(1000, 64)).ranges(), &[r(1000, 64)]);
        assert_eq!(c.uncovered(r(32, 64)).ranges(), &[r(64, 32)]);
    }

    /// Clearing forgets the versions and leaves the invariants intact.
    #[test]
    fn clearing_forgets_every_version() {
        let mut c = VersionCoverage::new();
        c.apply(r(0, 64), v(1));
        c.clear();
        invariants(&c);
        assert!(c.is_empty());
        assert_eq!(c.newest_over(r(0, 64)), None);
    }

    /// Many interleaved writes in a random-ish order keep every invariant and
    /// end with each byte at the highest version that ever claimed it.
    ///
    /// The property a hand-written case cannot state: `apply` is order-
    /// independent for the *maximum*, because a write only takes where it is
    /// strictly newer. Two schedules of the same writes must agree.
    #[test]
    fn the_map_is_the_pointwise_maximum_whatever_order_the_writes_arrive_in() {
        let writes: Vec<(ByteRange, ContentVersion)> = (0..24u64)
            .map(|n| {
                let offset = (n * 37) % 200;
                let length = 1 + (n * 13) % 60;
                (r(offset, length), v((n * 7) % 11))
            })
            .collect();

        let mut forwards = VersionCoverage::new();
        for (range, version) in &writes {
            forwards.apply(*range, *version);
            invariants(&forwards);
        }
        let mut backwards = VersionCoverage::new();
        for (range, version) in writes.iter().rev() {
            backwards.apply(*range, *version);
            invariants(&backwards);
        }

        // Byte for byte, both maps hold the highest version any write claimed
        // for that byte.
        for byte in 0..320u64 {
            let one = ByteRange {
                offset: byte,
                length: 1,
            };
            let expected = writes
                .iter()
                .filter(|(range, _)| byte >= range.offset && byte < range.offset + range.length)
                .map(|(_, version)| *version)
                .max();
            assert_eq!(forwards.newest_over(one), expected, "byte {byte} forwards");
            assert_eq!(
                backwards.newest_over(one),
                expected,
                "byte {byte} backwards"
            );
        }
        assert_eq!(forwards, backwards);
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

    /// The byte window the sweep works in, and the version space it draws
    /// from. Both small on purpose: a per-byte shadow has to be affordable, and
    /// collisions — equal versions meeting, a write landing exactly on a span
    /// boundary, three writers overlapping one byte — are the cases the
    /// hand-written ones cannot arrange together.
    const WINDOW: u64 = 40;
    const VERSIONS: u64 = 5;

    /// Every operation, driven, against a per-byte array of `Option<version>`
    /// that knows nothing about spans.
    ///
    /// # What this adds to the cases above
    ///
    /// Two things neither a hand-written case nor
    /// `the_map_is_the_pointwise_maximum_whatever_order_the_writes_arrive_in`
    /// reaches.
    ///
    /// **`Applied` is a partition.** The module's whole argument for returning
    /// two halves is that "this transaction's bytes never landed" must not be
    /// invisible — for a GPU write that is a frame the guest will never see. A
    /// caller can only rely on that if `taken` and `refused` together are
    /// *exactly* the range and never share a byte. Nothing checked it, and a
    /// gap between them is silent: the map would still be right, `taken` would
    /// still be right, and the bytes that were beaten would simply not be
    /// reported by either half.
    ///
    /// **The order-independence is checked over permutations rather than over
    /// one schedule and its reverse.** Forwards-and-backwards agree on any
    /// sequence whose overlaps happen to be nested; the shapes that separate a
    /// pointwise maximum from an order-dependent one need a third order.
    #[test]
    fn every_apply_partitions_its_range_and_the_map_is_the_pointwise_maximum() {
        let mut applies = 0usize;
        let mut wholly_taken = 0usize;
        let mut wholly_refused = 0usize;
        let mut split = 0usize;
        let mut equal_version_refusals = 0usize;

        for seed in 0..192u64 {
            let mut rng = Rng::new(seed);
            let mut coverage = VersionCoverage::new();
            let mut shadow: Vec<Option<ContentVersion>> = vec![None; WINDOW as usize];

            let mut writes: Vec<(ByteRange, ContentVersion)> = Vec::new();
            for _ in 0..24 {
                let offset = rng.below(WINDOW);
                let length = 1 + rng.below(WINDOW - offset);
                let range = r(offset, length);
                let version = v(rng.below(VERSIONS));
                writes.push((range, version));

                let applied = coverage.apply(range, version);
                applies += 1;
                invariants(&coverage);

                // The partition, byte by byte, against the shadow's own answer
                // about which bytes this write was allowed to take.
                for byte in 0..WINDOW {
                    let one = r(byte, 1);
                    let inside = byte >= offset && byte < offset + length;
                    let takes = inside && shadow[byte as usize].is_none_or(|held| held < version);
                    assert_eq!(
                        applied.taken.covers(one),
                        takes,
                        "seed {seed} byte {byte}: taken"
                    );
                    assert_eq!(
                        applied.refused.covers(one),
                        inside && !takes,
                        "seed {seed} byte {byte}: refused"
                    );
                    if takes {
                        shadow[byte as usize] = Some(version);
                    }
                }
                assert_eq!(
                    applied.taken.len() + applied.refused.len(),
                    length,
                    "seed {seed}: the two halves must be the whole range"
                );
                assert_eq!(applied.is_empty(), applied.taken.is_empty());
                assert_eq!(applied.was_partly_stale(), !applied.refused.is_empty());

                match (applied.taken.is_empty(), applied.refused.is_empty()) {
                    (false, true) => wholly_taken += 1,
                    (true, false) => wholly_refused += 1,
                    (false, false) => split += 1,
                    (true, true) => unreachable!("a non-empty range partitions into something"),
                }
                if (offset..offset + length)
                    .any(|b| shadow[b as usize] == Some(version) && applied.refused.covers(r(b, 1)))
                {
                    equal_version_refusals += 1;
                }

                // Every query answers off the same shadow.
                for byte in 0..WINDOW {
                    let one = r(byte, 1);
                    assert_eq!(
                        coverage.newest_over(one),
                        shadow[byte as usize],
                        "seed {seed} byte {byte}: newest_over"
                    );
                    assert_eq!(
                        coverage.uncovered(one).is_empty(),
                        shadow[byte as usize].is_some(),
                        "seed {seed} byte {byte}: uncovered"
                    );
                }
                // And over the whole window at once, which is where a reader
                // that clipped a span wrongly shows up.
                let whole = r(0, WINDOW);
                let mut walked = 0u64;
                for span in coverage.over(whole) {
                    for byte in span.range.offset..span.range.offset + span.range.length {
                        assert_eq!(
                            shadow[byte as usize],
                            Some(span.version),
                            "seed {seed} byte {byte}: over"
                        );
                        walked += 1;
                    }
                }
                assert_eq!(
                    walked,
                    shadow.iter().filter(|s| s.is_some()).count() as u64,
                    "seed {seed}: `over` must reach every covered byte exactly once"
                );
                assert_eq!(
                    coverage.newest_over(whole),
                    shadow.iter().copied().flatten().max(),
                    "seed {seed}: the summary is the maximum over the range"
                );
            }

            // The same writes in three more orders must produce the same map,
            // value for value — which is the coalescing claim as well as the
            // maximum one, since two maps that agree byte for byte are equal
            // only if neither kept a split its history caused.
            for shuffle in 0..3u64 {
                let mut order: Vec<usize> = (0..writes.len()).collect();
                let mut r2 = Rng::new(seed * 31 + shuffle + 1);
                for i in (1..order.len()).rev() {
                    order.swap(i, r2.below(i as u64 + 1) as usize);
                }
                let mut other = VersionCoverage::new();
                for &i in &order {
                    other.apply(writes[i].0, writes[i].1);
                    invariants(&other);
                }
                assert_eq!(
                    other, coverage,
                    "seed {seed} shuffle {shuffle}: the map depends on the order \
                     the writes arrived in"
                );
            }

            coverage.clear();
            invariants(&coverage);
            assert!(coverage.is_empty());
        }

        // Non-vacuity: each of the three outcomes of an apply, and the equal-
        // version refusal specifically — that is the arm which distinguishes
        // "a second claim on a committed version" from a first landing, and a
        // generator drawing versions from a wide enough space would never
        // produce one.
        assert!(applies > 4_000, "{applies}");
        assert!(
            wholly_taken > 500,
            "writes that landed whole: {wholly_taken}"
        );
        assert!(
            wholly_refused > 500,
            "writes beaten everywhere: {wholly_refused}"
        );
        assert!(split > 500, "writes that landed in part: {split}");
        assert!(
            equal_version_refusals > 500,
            "refusals of a version already committed: {equal_version_refusals}"
        );
    }
}
