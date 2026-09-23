//! A set of byte ranges over one backing, kept sorted and disjoint.
//!
//! # Why the model needs one
//!
//! Content authority is not per resource. A guest writes part of a buffer, a
//! blit fills another part, a draw reads a third — and whether a transfer is
//! owed depends on which *bytes* a replica is behind on, not on whether it is
//! behind at all. A per-resource dirty flag answers the second question and
//! then copies the whole resource to answer the first, which is the cost this
//! representation exists to avoid.
//!
//! # Sorted and disjoint is an invariant, not a convention
//!
//! Every operation restores it before returning, and the tests assert it after
//! each one. An overlapping pair here would make [`RangeSet::remove`] leave
//! bytes behind, which reads as a transfer that copied everything it was asked
//! to and quietly did not.

use crate::access::ByteRange;

/// A sorted, disjoint, coalesced set of half-open byte ranges.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RangeSet {
    /// Invariant: sorted by offset, non-empty, disjoint, and non-adjacent —
    /// two ranges that touch are one range, so equality is structural.
    ranges: Vec<ByteRange>,
}

impl RangeSet {
    #[must_use]
    pub const fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    #[must_use]
    pub fn from_range(range: ByteRange) -> Self {
        let mut s = Self::new();
        s.insert(range);
        s
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The ranges, in order.
    #[must_use]
    pub fn ranges(&self) -> &[ByteRange] {
        &self.ranges
    }

    /// Total bytes covered.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.ranges.iter().map(|r| r.length).sum()
    }

    /// Add a range that begins at or after every range already here.
    ///
    /// The constant-time form of [`Self::insert`], for a producer that already
    /// walks in offset order — [`crate::coverage::VersionCoverage::apply`]
    /// scans its spans left to right and emits one piece per span it meets, so
    /// every piece it produces begins where the previous one ended. `insert`
    /// rebuilds the whole member list per call, which makes such a scan
    /// quadratic in the coverage's own size on the path every write takes.
    ///
    /// `pub(crate)` and debug-asserted rather than public: the precondition is
    /// on the caller, and a caller outside this crate that got it wrong would
    /// leave an unsorted member list, which every later answer silently
    /// depends on.
    pub(crate) fn push_ascending(&mut self, range: ByteRange) {
        if range.length == 0 {
            return;
        }
        if let Some(last) = self.ranges.last_mut() {
            debug_assert!(
                range.offset >= last.offset,
                "push_ascending went backwards: {range:?} after {last:?}"
            );
            // Touching counts as overlapping, as everywhere else here.
            if range.offset <= end_of(*last) {
                let end = end_of(*last).max(end_of(range));
                last.length = end - last.offset;
                return;
            }
        }
        self.ranges.push(range);
    }

    /// Add a range, coalescing with anything it touches.
    pub fn insert(&mut self, range: ByteRange) {
        if range.length == 0 {
            return;
        }
        let end = end_of(range);
        let mut merged = range;
        let mut merged_end = end;
        let mut out: Vec<ByteRange> = Vec::with_capacity(self.ranges.len() + 1);
        let mut placed = false;
        for r in self.ranges.drain(..) {
            let r_end = end_of(r);
            if r_end < merged.offset || merged_end < r.offset {
                // Disjoint and not adjacent. Order decides which side it lands.
                if r_end < merged.offset {
                    out.push(r);
                } else {
                    if !placed {
                        out.push(ByteRange {
                            offset: merged.offset,
                            length: merged_end - merged.offset,
                        });
                        placed = true;
                    }
                    out.push(r);
                }
            } else {
                merged.offset = merged.offset.min(r.offset);
                merged_end = merged_end.max(r_end);
            }
        }
        if !placed {
            out.push(ByteRange {
                offset: merged.offset,
                length: merged_end - merged.offset,
            });
        }
        self.ranges = out;
    }

    /// Remove a range.
    pub fn remove(&mut self, range: ByteRange) {
        if range.length == 0 {
            return;
        }
        let cut_start = range.offset;
        let cut_end = end_of(range);
        let mut out = Vec::with_capacity(self.ranges.len() + 1);
        for r in self.ranges.drain(..) {
            let r_end = end_of(r);
            if r_end <= cut_start || cut_end <= r.offset {
                out.push(r);
                continue;
            }
            if r.offset < cut_start {
                out.push(ByteRange {
                    offset: r.offset,
                    length: cut_start - r.offset,
                });
            }
            if cut_end < r_end {
                out.push(ByteRange {
                    offset: cut_end,
                    length: r_end - cut_end,
                });
            }
        }
        self.ranges = out;
    }

    /// Whether every byte of `range` is covered.
    #[must_use]
    pub fn covers(&self, range: ByteRange) -> bool {
        if range.length == 0 {
            return true;
        }
        self.ranges
            .iter()
            .any(|r| r.offset <= range.offset && end_of(range) <= end_of(*r))
    }

    /// The parts of `range` this set does **not** cover.
    ///
    /// This is the transfer question: given what a replica is fresh for, which
    /// bytes of a read does it still owe?
    #[must_use]
    pub fn missing_from(&self, range: ByteRange) -> RangeSet {
        let mut want = RangeSet::from_range(range);
        want.subtract(self);
        want
    }

    /// The bytes both sets hold.
    ///
    /// # Why this is here and not written at the call site
    ///
    /// Two places in [`crate::content`] used to intersect a freshness set with
    /// another by nesting one member loop inside the other and inserting each
    /// overlap. That is quadratic in the member lists *and* pays an `insert`
    /// per overlap, and it is on the path a read takes — a backing written in
    /// many small pieces makes both lists long, which is exactly the case the
    /// per-byte representation exists to serve well.
    ///
    /// Both lists are sorted and disjoint, so one merge walk finds every
    /// overlap in linear time. The overlaps come out sorted, and already
    /// non-adjacent: two adjacent pieces would need two adjacent members on
    /// one side, which coalescing forbids. So the result is built directly
    /// rather than inserted into.
    #[must_use]
    pub fn intersection(&self, other: &RangeSet) -> RangeSet {
        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() && j < other.ranges.len() {
            let (a, b) = (self.ranges[i], other.ranges[j]);
            let (a_end, b_end) = (end_of(a), end_of(b));
            let start = a.offset.max(b.offset);
            let end = a_end.min(b_end);
            if start < end {
                out.push(ByteRange {
                    offset: start,
                    length: end - start,
                });
            }
            // Retire whichever member ends first; the other may still meet the
            // next one on that side.
            if a_end < b_end {
                i += 1;
            } else {
                j += 1;
            }
        }
        RangeSet { ranges: out }
    }

    /// Drop everything `other` holds.
    ///
    /// Linear for the reason [`Self::intersection`] is, and the reason
    /// [`Self::missing_from`] is written in terms of it: removing `other`'s
    /// members one at a time rebuilds the whole member list per member.
    pub fn subtract(&mut self, other: &RangeSet) {
        if self.ranges.is_empty() || other.ranges.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(self.ranges.len());
        let mut j = 0;
        for a in self.ranges.drain(..) {
            let a_end = end_of(a);
            let mut cur = a.offset;
            // Members of `other` that end at or before this one begins can
            // never meet a later one either, both lists being sorted.
            while j < other.ranges.len() && end_of(other.ranges[j]) <= cur {
                j += 1;
            }
            let mut k = j;
            while k < other.ranges.len() && other.ranges[k].offset < a_end {
                let b = other.ranges[k];
                if b.offset > cur {
                    out.push(ByteRange {
                        offset: cur,
                        length: b.offset - cur,
                    });
                }
                cur = cur.max(end_of(b));
                if cur >= a_end {
                    break;
                }
                k += 1;
            }
            if cur < a_end {
                out.push(ByteRange {
                    offset: cur,
                    length: a_end - cur,
                });
            }
        }
        self.ranges = out;
    }

    /// Everything in `other` as well as everything here.
    ///
    /// One merge walk rather than an `insert` per member, each of which
    /// rebuilds the whole list.
    pub fn union_with(&mut self, other: &RangeSet) {
        if other.ranges.is_empty() {
            return;
        }
        if self.ranges.is_empty() {
            self.ranges = other.ranges.clone();
            return;
        }
        let mut out: Vec<ByteRange> = Vec::with_capacity(self.ranges.len() + other.ranges.len());
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() || j < other.ranges.len() {
            let take_self = match (self.ranges.get(i), other.ranges.get(j)) {
                (Some(a), Some(b)) => a.offset <= b.offset,
                (Some(_), None) => true,
                _ => false,
            };
            let next = if take_self {
                i += 1;
                self.ranges[i - 1]
            } else {
                j += 1;
                other.ranges[j - 1]
            };
            match out.last_mut() {
                // Touching counts as overlapping: the invariant is that two
                // ranges which meet are one range.
                Some(last) if next.offset <= end_of(*last) => {
                    let end = end_of(*last).max(end_of(next));
                    last.length = end - last.offset;
                }
                _ => out.push(next),
            }
        }
        self.ranges = out;
    }

    /// Drop everything.
    pub fn clear(&mut self) {
        self.ranges.clear();
    }
}

const fn end_of(r: ByteRange) -> u64 {
    r.offset.saturating_add(r.length)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(offset: u64, length: u64) -> ByteRange {
        ByteRange { offset, length }
    }

    /// The invariant every operation restores. Checked after each one rather
    /// than trusted, because an overlapping pair makes `remove` leave bytes
    /// behind — which reads as a transfer that copied everything it was asked
    /// to and quietly did not.
    fn assert_sorted_disjoint_coalesced(s: &RangeSet) {
        for w in s.ranges().windows(2) {
            assert!(
                end_of(w[0]) < w[1].offset,
                "ranges must be ordered, disjoint and non-adjacent: {:?}",
                s.ranges()
            );
        }
        for x in s.ranges() {
            assert!(x.length > 0, "an empty range is not a member");
        }
    }

    #[test]
    fn inserting_touching_ranges_makes_one() {
        let mut s = RangeSet::new();
        s.insert(r(0, 4));
        s.insert(r(4, 4));
        assert_sorted_disjoint_coalesced(&s);
        assert_eq!(s.ranges(), &[r(0, 8)]);
        assert_eq!(s.len(), 8);
    }

    #[test]
    fn inserting_a_gap_keeps_two() {
        let mut s = RangeSet::new();
        s.insert(r(0, 4));
        s.insert(r(8, 4));
        assert_sorted_disjoint_coalesced(&s);
        assert_eq!(s.ranges(), &[r(0, 4), r(8, 4)]);
        // And a range spanning the gap collapses all three.
        s.insert(r(3, 6));
        assert_sorted_disjoint_coalesced(&s);
        assert_eq!(s.ranges(), &[r(0, 12)]);
    }

    #[test]
    fn inserting_out_of_order_still_sorts() {
        let mut s = RangeSet::new();
        for range in [r(100, 4), r(0, 4), r(50, 4), r(20, 4)] {
            s.insert(range);
            assert_sorted_disjoint_coalesced(&s);
        }
        assert_eq!(s.ranges(), &[r(0, 4), r(20, 4), r(50, 4), r(100, 4)]);
    }

    #[test]
    fn removing_the_middle_splits_a_range() {
        let mut s = RangeSet::from_range(r(0, 16));
        s.remove(r(4, 4));
        assert_sorted_disjoint_coalesced(&s);
        assert_eq!(s.ranges(), &[r(0, 4), r(8, 8)]);
    }

    #[test]
    fn removing_the_edges_trims_rather_than_splits() {
        let mut s = RangeSet::from_range(r(8, 8));
        s.remove(r(0, 10));
        assert_eq!(s.ranges(), &[r(10, 6)]);
        s.remove(r(14, 100));
        assert_eq!(s.ranges(), &[r(10, 4)]);
        assert_sorted_disjoint_coalesced(&s);
    }

    #[test]
    fn a_zero_length_range_is_neither_inserted_nor_removed() {
        let mut s = RangeSet::from_range(r(0, 8));
        s.insert(r(4, 0));
        s.remove(r(4, 0));
        assert_eq!(s.ranges(), &[r(0, 8)]);
        assert!(
            s.covers(r(4, 0)),
            "a range naming no byte is always covered"
        );
    }

    /// The transfer question, and the one that has to be exactly right: a
    /// replica fresh for part of a read owes only the rest.
    #[test]
    fn missing_from_names_only_the_bytes_that_are_owed() {
        let mut fresh = RangeSet::new();
        fresh.insert(r(0, 16));
        fresh.insert(r(32, 16));
        let owed = fresh.missing_from(r(0, 64));
        assert_sorted_disjoint_coalesced(&owed);
        assert_eq!(owed.ranges(), &[r(16, 16), r(48, 16)]);
        assert_eq!(owed.len(), 32);
        // Fully covered reads owe nothing.
        assert!(fresh.missing_from(r(0, 16)).is_empty());
        assert!(fresh.covers(r(4, 8)));
        assert!(!fresh.covers(r(8, 32)));
    }

    #[test]
    fn a_read_of_nothing_owes_nothing() {
        let fresh = RangeSet::new();
        assert!(fresh.missing_from(r(0, 0)).is_empty());
        assert_eq!(fresh.missing_from(r(0, 8)).ranges(), &[r(0, 8)]);
    }

    #[test]
    fn union_is_insertion_of_every_member() {
        let mut a = RangeSet::from_range(r(0, 8));
        let mut b = RangeSet::from_range(r(8, 8));
        b.insert(r(100, 4));
        a.union_with(&b);
        assert_sorted_disjoint_coalesced(&a);
        assert_eq!(a.ranges(), &[r(0, 16), r(100, 4)]);
    }

    /// The ascending fast path must build the same value `insert` does, or a
    /// set assembled by a scan and one assembled by a caller stop comparing
    /// equal --- and equality here is how a caller asks "did this change".
    #[test]
    fn pushing_in_order_builds_what_inserting_would_have() {
        let mut rng = Rng::new(7);
        for _ in 0..2_000 {
            let mut pushed = RangeSet::new();
            let mut inserted = RangeSet::new();
            let mut cursor = 0u64;
            for _ in 0..1 + rng.below(6) {
                cursor += rng.below(4);
                let length = rng.below(5);
                let x = r(cursor, length);
                cursor += length;
                pushed.push_ascending(x);
                inserted.insert(x);
            }
            assert_sorted_disjoint_coalesced(&pushed);
            assert_eq!(pushed, inserted);
        }
    }

    /// Coalescing makes equality structural, which is what lets a caller ask
    /// "did this change" without walking the set.
    #[test]
    fn two_sets_covering_the_same_bytes_are_equal_however_they_were_built() {
        let mut a = RangeSet::new();
        a.insert(r(0, 4));
        a.insert(r(4, 4));
        let b = RangeSet::from_range(r(0, 8));
        assert_eq!(a, b);
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

    /// The byte window the sweep works in. Small on purpose: a bitmap shadow
    /// has to be affordable, and every interesting case — a range that spans a
    /// gap, one that lands exactly against an edge, one that swallows several
    /// members at once — needs the members to be close enough together to
    /// collide.
    const WINDOW: u64 = 48;

    fn bitmap(s: &RangeSet) -> Vec<bool> {
        let mut out = vec![false; WINDOW as usize];
        for x in s.ranges() {
            for b in x.offset..end_of(*x) {
                out[b as usize] = true;
            }
        }
        out
    }

    /// A range inside the window, including zero-length ones and ones that end
    /// exactly at the edge.
    fn gen_range(rng: &mut Rng) -> ByteRange {
        let offset = rng.below(WINDOW + 1);
        let length = match rng.below(8) {
            0 => 0,
            1 => WINDOW - offset,
            _ => rng.below(WINDOW - offset + 1),
        };
        r(offset, length)
    }

    /// Every operation, driven, against a bitmap that knows nothing about
    /// ranges.
    ///
    /// # Why this is not covered by the cases above
    ///
    /// The cases above each build one shape and check it. What they cannot
    /// reach is the *combination*: an insert that merges two members while a
    /// third sits adjacent to the result, a remove that splits one member and
    /// deletes the next in the same call, a union whose members interleave
    /// with the receiver's. Those are where a sorted-and-disjoint
    /// representation goes wrong, and where the failure is silent — an
    /// overlapping pair still answers `covers` correctly and only shows up
    /// later, as a `remove` that leaves bytes behind and a transfer that
    /// copied everything it was asked to and quietly did not.
    ///
    /// The shadow is a bitmap, so it shares no code and no idea with the thing
    /// under test. Structural equality is asserted too, not just byte
    /// equality: coalescing is what makes `==` mean "the same bytes", and two
    /// sets that agree byte for byte while differing in their member list
    /// would break every caller that asks "did this change" without walking.
    #[test]
    fn every_operation_agrees_with_a_bitmap_that_knows_nothing_about_ranges() {
        let mut inserts = 0usize;
        let mut removes = 0usize;
        let mut unions = 0usize;
        let mut splits = 0usize;
        let mut merges = 0usize;
        let mut covered_queries = 0usize;
        let mut uncovered_queries = 0usize;
        let mut intersections = 0usize;
        let mut subtractions = 0usize;

        for seed in 0..256u64 {
            let mut rng = Rng::new(seed);
            let mut set = RangeSet::new();
            let mut shadow = vec![false; WINDOW as usize];

            for _ in 0..64 {
                let before = set.ranges().len();
                match rng.below(8) {
                    0..=3 => {
                        let x = gen_range(&mut rng);
                        set.insert(x);
                        for b in x.offset..x.offset + x.length {
                            shadow[b as usize] = true;
                        }
                        inserts += 1;
                        if set.ranges().len() < before {
                            merges += 1;
                        }
                    }
                    4..=5 => {
                        let x = gen_range(&mut rng);
                        set.remove(x);
                        for b in x.offset..x.offset + x.length {
                            shadow[b as usize] = false;
                        }
                        removes += 1;
                        if set.ranges().len() > before {
                            splits += 1;
                        }
                    }
                    6 => {
                        // A union with a set built independently, which is the
                        // only operation that can interleave two member lists.
                        let mut other = RangeSet::new();
                        for _ in 0..1 + rng.below(4) {
                            let x = gen_range(&mut rng);
                            other.insert(x);
                            for b in x.offset..x.offset + x.length {
                                shadow[b as usize] = true;
                            }
                        }
                        set.union_with(&other);
                        unions += 1;
                    }
                    _ => {
                        set.clear();
                        shadow.iter_mut().for_each(|b| *b = false);
                    }
                }

                assert_sorted_disjoint_coalesced(&set);
                assert_eq!(bitmap(&set), shadow, "members: {:?}", set.ranges());
                assert_eq!(
                    set.len(),
                    shadow.iter().filter(|b| **b).count() as u64,
                    "len counts bytes, not members"
                );
                assert_eq!(set.is_empty(), shadow.iter().all(|b| !*b));

                // Structural equality: a set rebuilt one byte at a time has to
                // *be* this set, member list and all.
                let mut rebuilt = RangeSet::new();
                for (b, held) in shadow.iter().enumerate() {
                    if *held {
                        rebuilt.insert(r(b as u64, 1));
                    }
                }
                assert_eq!(
                    rebuilt, set,
                    "two sets over the same bytes must be one value"
                );

                // The two questions the content model actually asks, against
                // the bitmap rather than against the set that answered them.
                let q = gen_range(&mut rng);
                let want: Vec<u64> = (q.offset..q.offset + q.length).collect();
                let all_held = want.iter().all(|b| shadow[*b as usize]);
                assert_eq!(set.covers(q), all_held, "covers({q:?})");
                if all_held {
                    covered_queries += 1;
                } else {
                    uncovered_queries += 1;
                }

                let missing = set.missing_from(q);
                assert_sorted_disjoint_coalesced(&missing);
                let missing_bytes: Vec<u64> = missing
                    .ranges()
                    .iter()
                    .flat_map(|x| x.offset..end_of(*x))
                    .collect();
                let owed: Vec<u64> = want
                    .iter()
                    .copied()
                    .filter(|b| !shadow[*b as usize])
                    .collect();
                assert_eq!(missing_bytes, owed, "missing_from({q:?})");

                // The two set-against-set operations, against an independently
                // built partner and a bitmap that does not know what a member
                // is. Neither one changes `set`, so they ride along on every
                // step rather than needing arms of their own.
                let mut partner = RangeSet::new();
                let mut partner_bits = vec![false; WINDOW as usize];
                for _ in 0..rng.below(4) {
                    let x = gen_range(&mut rng);
                    partner.insert(x);
                    for b in x.offset..end_of(x) {
                        partner_bits[b as usize] = true;
                    }
                }

                let both = set.intersection(&partner);
                assert_sorted_disjoint_coalesced(&both);
                assert_eq!(
                    bitmap(&both),
                    shadow
                        .iter()
                        .zip(&partner_bits)
                        .map(|(a, b)| *a && *b)
                        .collect::<Vec<bool>>(),
                    "intersection"
                );
                assert_eq!(
                    both,
                    partner.intersection(&set),
                    "and it does not depend on which side asks"
                );
                if !both.is_empty() {
                    intersections += 1;
                }

                let mut without = set.clone();
                without.subtract(&partner);
                assert_sorted_disjoint_coalesced(&without);
                assert_eq!(
                    bitmap(&without),
                    shadow
                        .iter()
                        .zip(&partner_bits)
                        .map(|(a, b)| *a && !*b)
                        .collect::<Vec<bool>>(),
                    "subtract"
                );
                if without != set {
                    subtractions += 1;
                }
            }
        }

        // Non-vacuity: the shapes the combination cases depend on having been
        // reached. A generator that stopped producing merges and splits would
        // still pass every assertion above while testing only insertion into
        // empty space.
        assert!(merges > 250, "inserts that swallowed a member: {merges}");
        assert!(splits > 400, "removes that split a member: {splits}");
        assert!(inserts > 4_000, "{inserts}");
        assert!(removes > 2_000, "{removes}");
        assert!(unions > 1_000, "{unions}");
        assert!(covered_queries > 3_000, "{covered_queries}");
        assert!(uncovered_queries > 4_000, "{uncovered_queries}");
        assert!(
            intersections > 4_000,
            "intersections that met: {intersections}"
        );
        assert!(
            subtractions > 4_000,
            "subtractions that took: {subtractions}"
        );
    }

    /// The one range the model asks for that is not a range: `newest_over`
    /// spells "the whole backing" as offset zero and length `u64::MAX`, so
    /// arithmetic on the end of a member has to survive it.
    ///
    /// Saturating rather than wrapping is the whole answer here, and it is
    /// asserted rather than assumed because the wrapping form is silently
    /// wrong in the direction that matters: an end that wraps to zero makes a
    /// member compare as ending before it starts, and the range drops out of
    /// every subsequent answer.
    #[test]
    fn a_range_that_reaches_the_end_of_the_address_space_still_answers() {
        let whole = r(0, u64::MAX);
        let mut s = RangeSet::from_range(whole);
        assert_eq!(s.ranges(), &[whole]);
        assert!(s.covers(r(0, 16)));
        assert!(s.covers(whole));
        assert!(s.missing_from(whole).is_empty());

        // A member at the top of the space is not lost by the end computation.
        s.clear();
        s.insert(r(u64::MAX - 8, 8));
        assert!(s.covers(r(u64::MAX - 8, 8)));
        assert!(!s.covers(r(u64::MAX - 9, 1)));
        s.remove(r(u64::MAX - 4, 4));
        assert_eq!(s.ranges(), &[r(u64::MAX - 8, 4)]);

        // And a length that would overflow the offset is held at the top of
        // the space rather than wrapping past it, so the member still names
        // every byte from its offset upwards.
        s.clear();
        s.insert(r(u64::MAX - 4, u64::MAX));
        assert_eq!(
            s.ranges(),
            &[r(u64::MAX - 4, 4)],
            "the part of the range that exists is kept and the rest does not \
             wrap to the bottom of the space"
        );
    }
}
