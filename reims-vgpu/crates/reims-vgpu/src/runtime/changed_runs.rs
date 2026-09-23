//! The byte runs in which a newly rendered row differs from the row already in
//! the guest's pages.
//!
//! # Why this is a type and not the loop it replaced
//!
//! `mapping_write::write_rgba8_image_changed` writes only the bytes of a frame
//! that changed, which is the whole point of holding a seed: a compositor
//! repaints a window and most of the surface is the frame before it. Finding
//! those bytes was a nested `while` — skip equal, take differing, copy, repeat —
//! **hand-inlined twice** in that one function, once for the mapped-pointer arm
//! and once for the mapper-write arm. Two copies of one algorithm that had to
//! agree about a byte offset, with no test that could reach either.
//!
//! # The block compare, and what it did not buy
//!
//! Both copies compared **one byte at a time**, and the scan's whole job is to
//! walk the *unchanged* majority of the frame and find nothing — 8.3 MB a flush
//! on a 1920x1080 BGRA surface, two bounds-checked loads per byte.
//! [`ChangedRuns`] skips agreement a **block at a time** instead, through a
//! slice comparison the compiler lowers to `bcmp`, narrowing to the exact byte
//! only inside the one block that disagreed. Same runs, same offsets.
//!
//! **That changed nothing measurable, and the measurement is the point.** The
//! enclosing `surface_changed_rows_us` was 90.9 % of `store_us` on a driven
//! macos-13 Metal boot before this and 89.9 % after — one ratio, taken inside
//! each boot, because totals across boots are not comparable here (the same
//! drive script produced 590 and 468 chains, and `sampled_us` differed 8.6x).
//!
//! So the 5 ms/draw inside that span is **not** this scan. It is one of the row
//! convert, this scan, or the landing write, and nothing yet divides them; do
//! not assume which from the shape of the code, which is the mistake
//! `backend::metal::render`'s `render_core_mrt` doc records having carried for a
//! year. This type is committed for the duplication it removes and the test it
//! makes possible, not for a speedup it did not deliver.
//!
//! # Why only the equal side is blocked
//!
//! The two sides of the scan are not symmetric and are deliberately treated
//! differently.
//!
//! Skipping equal bytes is pure overhead: nothing is copied for them, so every
//! byte the scan touches there is work the frame did not need. That side gets
//! the block compare, and it is where the megabytes are.
//!
//! Walking the differing bytes is bounded by the bytes the caller is about to
//! copy anyway, so blocking it would buy a constant factor on a term that is
//! already proportional to real work. It would also cost exactness: a block that
//! is *not wholly equal* may still contain equal bytes, so extending the run by
//! whole blocks would report a run wider than the bytes that actually changed.
//! That is not merely imprecise — the runs name what this device writes into
//! guest pages, and widening them writes bytes the guest did not ask this device
//! to touch. The runs are maximal and exact, and
//! [`the_runs_are_exactly_the_differing_bytes`] holds them to it against a
//! byte-at-a-time oracle over exhaustive small inputs.

use std::ops::Range;

/// Bytes compared per step of the equal-skip.
///
/// A slice comparison of this length lowers to `bcmp`, which moves a cache line
/// per iteration instead of a byte. The value is not tuned against a target's
/// vector width on purpose: the win is going from one byte per step to many, and
/// every plausible width divides this one.
const BLOCK: usize = 64;

/// The maximal runs of bytes in which `new` differs from `old`.
///
/// Yields disjoint, ascending, non-empty ranges. Two adjacent runs are never
/// yielded — a run ends only at a byte that is equal, or at the end of the
/// comparable region — so `next` returning `None` means every remaining byte
/// agrees.
///
/// # The comparable region
///
/// Bytes past the shorter of the two slices are not reported. There is nothing
/// to compare them against, and a caller that means "the whole row changed"
/// should say so with a full-row write rather than by handing in a short seed.
/// Every caller in this crate slices both sides to the row's tight length first.
pub struct ChangedRuns<'a> {
    new: &'a [u8],
    old: &'a [u8],
    at: usize,
}

impl<'a> ChangedRuns<'a> {
    /// Scan `new` against `old` from the start.
    pub fn new(new: &'a [u8], old: &'a [u8]) -> Self {
        Self { new, old, at: 0 }
    }

    /// The region both slices cover.
    fn end(&self) -> usize {
        self.new.len().min(self.old.len())
    }
}

impl Iterator for ChangedRuns<'_> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        let end = self.end();
        // Skip agreement a block at a time. The slices are re-sliced rather than
        // indexed so the comparison is one `bcmp` and not `BLOCK` bounds-checked
        // loads, which is the whole difference between this and the loop it
        // replaced.
        while self.at + BLOCK <= end
            && self.new[self.at..self.at + BLOCK] == self.old[self.at..self.at + BLOCK]
        {
            self.at += BLOCK;
        }
        // Narrow to the exact first differing byte. Bounded by `BLOCK` when the
        // loop above stopped on a disagreeing block, and by `BLOCK` again for
        // the final partial block, so this is never the frame-sized walk.
        while self.at < end && self.new[self.at] == self.old[self.at] {
            self.at += 1;
        }
        if self.at >= end {
            return None;
        }
        let start = self.at;
        // Exact, byte at a time — see the module doc on why this side is not
        // blocked.
        while self.at < end && self.new[self.at] != self.old[self.at] {
            self.at += 1;
        }
        Some(start..self.at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte-at-a-time definition the blocked scan has to reproduce. This is
    /// the loop `write_rgba8_image_changed` carried in two copies, kept here as
    /// the oracle rather than as the implementation.
    fn runs_byte_at_a_time(new: &[u8], old: &[u8]) -> Vec<Range<usize>> {
        let end = new.len().min(old.len());
        let mut out = Vec::new();
        let mut x = 0;
        while x < end {
            while x < end && new[x] == old[x] {
                x += 1;
            }
            if x >= end {
                break;
            }
            let start = x;
            while x < end && new[x] != old[x] {
                x += 1;
            }
            out.push(start..x);
        }
        out
    }

    /// The runs are exactly the differing bytes, against the oracle, over every
    /// difference pattern of a length that straddles the block boundary.
    ///
    /// # Why exhaustive rather than sampled
    ///
    /// The block skip has three seams — the block-aligned stride, the narrowing
    /// walk after a block disagrees, and the final partial block — and an
    /// off-by-one at any of them is a run that starts or ends one byte early.
    /// A run that ends one byte early leaves a stale byte in the guest's frame;
    /// one that starts one byte late does the same. Neither is visible in a
    /// spot check, and both are visible here.
    #[test]
    fn the_runs_are_exactly_the_differing_bytes() {
        // 17 bits over a 17-byte row would not reach a block boundary; this
        // walks patterns *placed* across BLOCK instead, at every offset.
        for len in [0usize, 1, 63, 64, 65, 127, 128, 129, 200] {
            for pattern in 0u32..(1 << 9) {
                for offset in [0usize, 1, 60, 64, 120] {
                    if offset >= len && len > 0 {
                        continue;
                    }
                    let old = vec![0xa5u8; len];
                    let mut new = old.clone();
                    for bit in 0..9 {
                        if pattern & (1 << bit) != 0 {
                            if let Some(slot) = new.get_mut(offset + bit) {
                                *slot ^= 0xff;
                            }
                        }
                    }
                    let got: Vec<_> = ChangedRuns::new(&new, &old).collect();
                    assert_eq!(
                        got,
                        runs_byte_at_a_time(&new, &old),
                        "len {len} pattern {pattern:#x} offset {offset}"
                    );
                }
            }
        }
    }

    /// A whole row that differs is one run, not one run per block.
    ///
    /// # Why
    ///
    /// The caller turns each run into one `copy_nonoverlapping` or one
    /// `write_mapping_bytes`. A scan that split a changed row into 64-byte runs
    /// would be correct and would replace one memcpy of a row with 120 calls,
    /// which is the failure mode a blocked *differing* side would have had.
    #[test]
    fn a_wholly_changed_row_is_one_run() {
        let old = vec![0u8; 512];
        let new = vec![1u8; 512];
        let got: Vec<_> = ChangedRuns::new(&new, &old).collect();
        assert_eq!(got, vec![0..512]);
    }

    /// Two changes separated by a single equal byte stay two runs.
    ///
    /// # Why
    ///
    /// This is the case a block-extended differing side would have merged. The
    /// runs name bytes this device writes into the guest's pages, so a merge
    /// writes a byte the guest did not ask for.
    #[test]
    fn a_one_byte_gap_separates_two_runs() {
        let old = vec![0u8; 8];
        let mut new = old.clone();
        new[2] = 1;
        new[4] = 1;
        let got: Vec<_> = ChangedRuns::new(&new, &old).collect();
        assert_eq!(got, vec![2..3, 4..5]);
    }

    /// Identical rows yield nothing, however long they are.
    #[test]
    fn identical_rows_yield_no_runs() {
        for len in [0usize, 1, 64, 65, 4096] {
            let row = vec![7u8; len];
            assert_eq!(ChangedRuns::new(&row, &row).count(), 0, "len {len}");
        }
    }

    /// A change past the shorter slice is outside the comparable region.
    ///
    /// # Why
    ///
    /// The type reports what it can compare, so a caller cannot get a run that
    /// names bytes only one side has. See the struct doc.
    #[test]
    fn the_shorter_slice_bounds_the_scan() {
        let old = vec![0u8; 4];
        let new = vec![1u8; 16];
        assert_eq!(
            ChangedRuns::new(&new, &old).collect::<Vec<_>>(),
            vec![0..4],
            "the run stops where the comparison does"
        );
        assert_eq!(
            ChangedRuns::new(&old, &new).collect::<Vec<_>>(),
            vec![0..4],
            "and does so whichever side is short"
        );
    }

    /// The runs are ascending, disjoint and non-empty, so a caller can write
    /// them in order without checking.
    #[test]
    fn the_runs_are_ascending_disjoint_and_non_empty() {
        let old: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let new: Vec<u8> = old
            .iter()
            .enumerate()
            .map(|(i, &b)| if i % 7 < 3 { b ^ 0x5a } else { b })
            .collect();
        let mut last = 0usize;
        let mut any = false;
        for run in ChangedRuns::new(&new, &old) {
            assert!(run.start < run.end, "empty run {run:?}");
            assert!(run.start >= last, "run {run:?} overlaps or precedes {last}");
            last = run.end;
            any = true;
        }
        assert!(any, "this input does differ");
    }
}
