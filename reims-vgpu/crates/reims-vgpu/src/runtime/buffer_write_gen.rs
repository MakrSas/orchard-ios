//! Guest-declared write generations for task-local GVA resources.
//!
//! # Why this state is separate
//!
//! [`crate::runtime::resource_validity::apply`] takes the guest's validity quad
//! for one object id and applies it to `DeviceState::mappings`. Buffers have no
//! mapping, while GVA render resources are owned by a task-local texture
//! reference. Their guest-write declarations therefore need a generation keyed
//! by `(task, object)` rather than the mapping table's validity state.
//!
//! A Store stamps the generation beside a host-authoritative GVA image. If the
//! guest later declares that object written, the changed stamp abandons that
//! image instead of copying its older pixels over the guest's newer bytes.
//! [`crate::runtime::writeback_debt`] owns both the stamp and the decision.
//!
//! # The bound
//!
//! One entry per `(task, object)` the guest has declared a write to. A task's
//! entries go when the task does, which is the same lifetime `bound_buffers`
//! retires on and the only announcement this device gets. Past [`BufferWriteGens::MAX`] the
//! map is **cleared** rather than partially evicted: forgetting one object would
//! make its next comparison read as clean, which is the direction that reports a
//! cache hit for a window whose bytes moved. A clear makes every comparison read
//! as *unknown*, which is the safe direction, and `buffer_write_gen_reset` says
//! it happened.

use std::collections::HashMap;

/// Per-object write generations for objects this device holds no mapping for.
#[derive(Default, Debug)]
pub struct BufferWriteGens {
    gens: HashMap<(u32, u32), u64>,
    /// Bumped on every drop of an entry — the whole-map clear and the per-task
    /// retire alike — so a comparison spanning one is not mistaken for a
    /// comparison that found the same generation twice. A debt stores this
    /// beside the generation and treats a change in it as "unknown".
    epoch: u64,
}

impl BufferWriteGens {
    /// The most `(task, object)` pairs tracked before the map resets.
    ///
    /// A driven boot's `bound_buffers` registry holds ~700 resolutions across
    /// **22** distinct `(task, reference)` pairs, and this is keyed the same way
    /// — by object, not by window — so 4 096 is two orders of magnitude above
    /// the only related number anyone has measured. It bounds a guest that
    /// creates and writes objects without bound, and `buffer_write_gen_reset`
    /// is what says whether it ever binds.
    pub const MAX: usize = 4096;

    /// Record that the guest declared a write to `object_id` under `task_id`.
    ///
    /// Called only for the ids [`crate::runtime::resource_validity::apply`]
    /// found no mapping for: an object with a mapping already has
    /// `content_generation`, and stamping it twice would be two spellings of one
    /// fact.
    pub fn note_write(&mut self, task_id: u32, object_id: u32) {
        if self.gens.len() >= Self::MAX && !self.gens.contains_key(&(task_id, object_id)) {
            self.gens.clear();
            self.epoch = self.epoch.wrapping_add(1);
            crate::runtime::drain::note_store_route("buffer_write_gen_reset");
        }
        let slot = self.gens.entry((task_id, object_id)).or_insert(0);
        *slot = slot.wrapping_add(1);
        // Keep the decoded write rate visible beside GVA debt abandonment. A
        // zero here with live GVA Stores means the task/object namespace is not
        // reaching the authority check at all.
        crate::runtime::drain::note_store_route("buffer_write_gen_bump");
    }

    /// What a reader records beside a copy it has just taken, and compares
    /// against later.
    ///
    /// The epoch travels with the generation so a clear cannot be read as
    /// "unchanged": an object with no entry reads `(epoch, 0)`, and after a
    /// clear the epoch differs from every stamp taken before it.
    pub fn stamp(&self, task_id: u32, object_id: u32) -> BufferWriteStamp {
        BufferWriteStamp {
            epoch: self.epoch,
            gen: self.gens.get(&(task_id, object_id)).copied().unwrap_or(0),
        }
    }

    /// Forget one task's objects, because the task's ids no longer name them.
    ///
    /// Retiring by task rather than by object for the reason
    /// [`crate::runtime::bound_buffers`] states about its own registry: mapping
    /// an object id back to what resolved through it is machinery bought with
    /// nothing, and task teardown is rare.
    ///
    /// This bumps the epoch for the same reason the clear does, and the case is
    /// not hypothetical: a guest reuses task ids. Drop `(5, 7)` at generation 3,
    /// let a *different* task 5 create a *different* object 7 and declare three
    /// writes to it, and a stamp taken before the retire compares equal to one
    /// taken after — same epoch, same generation, unrelated bytes. That is the
    /// one direction this whole type exists to refuse. Bumped only when an entry
    /// actually went, so retiring a task that declared no writes costs no
    /// reader's stamp.
    pub fn retire_task(&mut self, task_id: u32) {
        let before = self.gens.len();
        self.gens.retain(|&(task, _), _| task != task_id);
        if self.gens.len() != before {
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    /// Entries held.
    ///
    /// Named for what it counts rather than `len`, because this is not a
    /// collection anything iterates and a `len`/`is_empty` pair would suggest it
    /// is.
    pub fn tracked(&self) -> usize {
        self.gens.len()
    }
}

/// One object's write generation as a reader saw it.
///
/// Two of these are comparable only when their epochs agree; see
/// [`BufferWriteGens::stamp`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferWriteStamp {
    epoch: u64,
    gen: u64,
}

impl BufferWriteStamp {
    /// Whether the guest has declared no write to this object between the two
    /// stamps.
    ///
    /// `false` for a pair that straddles a map clear, which is the unknown case
    /// answered in the safe direction — see this module's doc on [`BufferWriteGens::MAX`].
    pub fn quiet_since(self, earlier: Self) -> bool {
        self.epoch == earlier.epoch && self.gen == earlier.gen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generation moves only when the guest says it wrote, so a debt stays
    /// authoritative across unrelated activity.
    #[test]
    fn a_declared_write_moves_the_generation_and_nothing_else_does() {
        let mut g = BufferWriteGens::default();
        let before = g.stamp(1, 7);
        assert!(g.stamp(1, 7).quiet_since(before), "no write, no move");
        g.note_write(1, 7);
        assert!(!g.stamp(1, 7).quiet_since(before));
        let after = g.stamp(1, 7);
        assert!(g.stamp(1, 7).quiet_since(after), "still no further write");
    }

    /// A write to one object must not invalidate another's stamp, or an
    /// unrelated resource would abandon host-authoritative pixels.
    #[test]
    fn a_write_to_another_object_leaves_this_one_quiet() {
        let mut g = BufferWriteGens::default();
        let before = g.stamp(1, 7);
        g.note_write(1, 8);
        g.note_write(2, 7);
        assert!(g.stamp(1, 7).quiet_since(before));
    }

    /// A stamp taken before a clear must not compare equal to one taken after
    /// it. Forgetting an object silently must abandon, not preserve, authority.
    #[test]
    fn a_stamp_that_straddles_a_reset_is_not_quiet() {
        let mut g = BufferWriteGens::default();
        g.note_write(1, 7);
        let before = g.stamp(1, 7);
        for object in 0..(BufferWriteGens::MAX as u32 + 1) {
            g.note_write(9, object);
        }
        assert!(
            !g.stamp(1, 7).quiet_since(before),
            "the map was cleared under this reader, so it cannot say the object was quiet"
        );
    }

    /// A task that goes takes its objects with it, so a later task reusing an
    /// id cannot inherit a stamp that was about something else.
    #[test]
    fn retiring_a_task_forgets_its_objects() {
        let mut g = BufferWriteGens::default();
        g.note_write(1, 7);
        g.note_write(2, 7);
        let before = g.stamp(1, 7);
        g.retire_task(1);
        assert_eq!(g.tracked(), 1, "only task 2's entry remains");
        assert!(
            !g.stamp(1, 7).quiet_since(before),
            "the entry is gone, so its generation reads as 0 and cannot match"
        );
    }

    /// The retire is a forgetting, so it has to move the epoch exactly as the
    /// clear does. A guest reuses task ids, and a generation that climbs back to
    /// the value a reader recorded is the one shape where the entry going is not
    /// enough on its own.
    #[test]
    fn a_stamp_that_straddles_a_retire_is_not_quiet() {
        let mut g = BufferWriteGens::default();
        for _ in 0..3 {
            g.note_write(5, 7);
        }
        let before = g.stamp(5, 7);
        g.retire_task(5);
        // A different task 5, a different object 7, back to generation 3.
        for _ in 0..3 {
            g.note_write(5, 7);
        }
        assert!(
            !g.stamp(5, 7).quiet_since(before),
            "the tracked object was retired under this reader, so nothing it \
             stamped is comparable across the gap however the count reads"
        );
    }

    /// Retiring a task nothing was tracked for must not move anyone's stamp, or
    /// task teardown alone would report every window as dirty.
    #[test]
    fn retiring_an_untracked_task_leaves_every_stamp_alone() {
        let mut g = BufferWriteGens::default();
        g.note_write(1, 7);
        let before = g.stamp(1, 7);
        g.retire_task(2);
        assert!(g.stamp(1, 7).quiet_since(before));
    }

    /// The map stops at its bound rather than tracking a guest that creates
    /// objects without end.
    #[test]
    fn the_map_stops_at_its_bound() {
        let mut g = BufferWriteGens::default();
        for object in 0..(BufferWriteGens::MAX as u32 + 5) {
            g.note_write(1, object);
        }
        assert!(g.tracked() <= BufferWriteGens::MAX);
    }
}
