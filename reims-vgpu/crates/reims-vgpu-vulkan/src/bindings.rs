//! What a draw has to re-emit, and — far more often — what it does not.
//!
//! # The gate this exists for
//!
//! A steady-state draw must not rebuild its bindings. A guest that binds the
//! same forty textures every draw is the ordinary case, and an emitter that
//! wrote forty descriptors each time would spend the frame writing values that
//! were already there. So the table remembers what it emitted, a bind that does
//! not change a slot marks nothing, and [`BindingTable::take_dirty`] hands back
//! only the slots whose contents actually moved.
//!
//! # Equality is the whole mechanism, so the values have to be exact
//!
//! "Same binding" is decided by comparing the semantic binding — the resource
//! identity with its generation, the offset, the length — and never by
//! comparing native handles. Two different resources may occupy one handle
//! across a retire and a recreate, and a slot compared by handle would then
//! report clean while pointing at somebody else's memory. The identities carry
//! generations for exactly this reason, so the comparison is theirs.
//!
//! # Vulkan disturbs bindings, and pretending otherwise is a validation error
//!
//! Binding a pipeline whose layout is incompatible with the previous one
//! disturbs every descriptor set from the first incompatible index onward, and
//! a fresh command buffer starts with no bindings at all. Neither is something
//! the tracker can observe, so both are told to it:
//! [`BindingTable::disturb_all`] is the command-buffer boundary and the
//! incompatible-layout case. A tracker that quietly kept believing its own
//! contents across either would emit nothing and the draw would read whatever
//! the driver had.
//!
//! # An unbound slot is a change
//!
//! A guest unbinds by naming no object. That is a different state from "still
//! holds the previous object", and a tracker that treated `None` as "no news"
//! would leave a descriptor pointing at a resource the guest has released.

use reims_vgpu_core::bind::{BufferBinding, ObjectBinding};

/// A set of slot indices, one bit each.
///
/// Words rather than a `Vec<bool>`: a draw asks "is anything dirty" and "which
/// slots" on every emission, and both are word operations here. A hundred and
/// twenty-eight texture slots are two words.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotMask {
    words: Vec<u64>,
}

impl SlotMask {
    #[must_use]
    pub fn with_capacity(slots: usize) -> Self {
        Self {
            words: vec![0; slots.div_ceil(64)],
        }
    }

    pub fn insert(&mut self, slot: usize) {
        let word = slot / 64;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << (slot % 64);
    }

    /// Whether `slot` is in the mask.
    ///
    /// For a caller asking about one slot; [`Self::slots`] is what an emitter
    /// walks, and is therefore what the driven sweep drives. Inverting this
    /// is not observable through that sweep, so the tests that name single
    /// slots are the ones that hold it.
    #[must_use]
    pub fn contains(&self, slot: usize) -> bool {
        self.words
            .get(slot / 64)
            .is_some_and(|w| w >> (slot % 64) & 1 == 1)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }

    /// Make room for `slots` without disturbing the bits already held.
    ///
    /// Redundant against every mask this module makes — they are all built by
    /// [`Self::with_capacity`] at the table's own width — so no mutation of it
    /// is observable and the driven sweep cannot catch one. Kept because what
    /// it prevents is an index out of range in [`mark_bound`], which is a panic
    /// rather than a wrong answer, and because the alternative is a width
    /// assumption two functions have to keep agreeing about.
    fn reserve(&mut self, slots: usize) {
        let words = slots.div_ceil(64);
        if self.words.len() < words {
            self.words.resize(words, 0);
        }
    }

    /// The slots in the mask, ascending.
    ///
    /// An iterator and not a `Vec`. This is the door an emitter walks on every
    /// draw that rebinds anything, once per class, and the walk keeps no state
    /// past a word index and the residue of that word --- so the `Vec` was
    /// three allocations per rebinding draw buying nothing. A caller that
    /// genuinely wants one still collects; what it cannot do is get one
    /// without asking.
    pub fn slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(word, bits)| {
            let mut bits = *bits;
            core::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros() as usize;
                // Clear the lowest set bit, so the residue is what is left to
                // report and the loop terminates on the population count
                // rather than on the word width.
                bits &= bits - 1;
                Some(word * 64 + bit)
            })
        })
    }
}

/// The slots one emission has to write.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[must_use = "slots reported dirty and not written leave the descriptor stale"]
pub struct Dirty {
    pub buffers: SlotMask,
    pub textures: SlotMask,
    pub samplers: SlotMask,
}

impl Dirty {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty() && self.textures.is_empty() && self.samplers.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.buffers.len() + self.textures.len() + self.samplers.len()
    }
}

/// What the table has been asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Binds that changed a slot.
    pub changed: usize,
    /// Binds that named the value already there. The number that says what the
    /// tracker is saving.
    pub redundant: usize,
    /// Slots handed to an emission.
    pub emitted: usize,
    /// Times everything was invalidated by a command-buffer boundary or an
    /// incompatible pipeline layout.
    pub disturbances: usize,
    /// Binds that named a slot past the table and were dropped.
    ///
    /// Counted rather than left to be inferred from the others. A dropped
    /// bind is neither a change nor a redundancy, so without this the three
    /// counters silently stop adding up to the binds the guest issued, and the
    /// only evidence of guest work this table refused is a shader reading a
    /// descriptor nobody wrote. Non-zero means either the guest contradicted
    /// its own argument-table sizes or this table was built with the wrong
    /// ones — and those are worth telling apart, which needs the number to
    /// exist.
    pub ignored: usize,
}

/// One shader-visible table's contents and its dirty set.
#[derive(Clone, Debug)]
pub struct BindingTable {
    buffers: Vec<Option<BufferBinding>>,
    textures: Vec<Option<ObjectBinding>>,
    samplers: Vec<Option<ObjectBinding>>,
    dirty: Dirty,
    census: Census,
}

impl BindingTable {
    /// A table with the guest's argument-table sizes, everything unbound.
    ///
    /// Nothing is dirty at construction: an unbound slot has no descriptor to
    /// write, and marking them all would make the first draw of every command
    /// buffer emit the whole table.
    #[must_use]
    pub fn new(buffers: usize, textures: usize, samplers: usize) -> Self {
        Self {
            buffers: vec![None; buffers],
            textures: vec![None; textures],
            samplers: vec![None; samplers],
            dirty: Dirty {
                buffers: SlotMask::with_capacity(buffers),
                textures: SlotMask::with_capacity(textures),
                samplers: SlotMask::with_capacity(samplers),
            },
            census: Census::default(),
        }
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Bind a buffer slot, or unbind it with `None`.
    ///
    /// Returns whether anything changed. A slot outside the table is ignored
    /// and reported as no change: the argument-table sizes are the guest's own
    /// capacity hints, and a record past them is a record about a slot no shader
    /// on this pipeline can read.
    pub fn bind_buffer(&mut self, slot: usize, binding: Option<BufferBinding>) -> bool {
        let Some(current) = self.buffers.get_mut(slot) else {
            self.census.ignored += 1;
            return false;
        };
        if *current == binding {
            self.census.redundant += 1;
            return false;
        }
        *current = binding;
        self.dirty.buffers.insert(slot);
        self.census.changed += 1;
        true
    }

    /// Bind a texture slot, or unbind it with `None`.
    pub fn bind_texture(&mut self, slot: usize, binding: Option<ObjectBinding>) -> bool {
        let Some(current) = self.textures.get_mut(slot) else {
            self.census.ignored += 1;
            return false;
        };
        if *current == binding {
            self.census.redundant += 1;
            return false;
        }
        *current = binding;
        self.dirty.textures.insert(slot);
        self.census.changed += 1;
        true
    }

    /// Bind a sampler slot, or unbind it with `None`.
    pub fn bind_sampler(&mut self, slot: usize, binding: Option<ObjectBinding>) -> bool {
        let Some(current) = self.samplers.get_mut(slot) else {
            self.census.ignored += 1;
            return false;
        };
        if *current == binding {
            self.census.redundant += 1;
            return false;
        }
        *current = binding;
        self.dirty.samplers.insert(slot);
        self.census.changed += 1;
        true
    }

    #[must_use]
    pub fn buffer(&self, slot: usize) -> Option<BufferBinding> {
        self.buffers.get(slot).copied().flatten()
    }

    #[must_use]
    pub fn texture(&self, slot: usize) -> Option<ObjectBinding> {
        self.textures.get(slot).copied().flatten()
    }

    #[must_use]
    pub fn sampler(&self, slot: usize) -> Option<ObjectBinding> {
        self.samplers.get(slot).copied().flatten()
    }

    /// Whether an emission would write anything.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.dirty.is_empty()
    }

    /// Take the slots this emission has to write, clearing the dirty set.
    ///
    /// The contents are unchanged: what was bound is still bound, and the next
    /// draw with no rebinds in between takes nothing.
    pub fn take_dirty(&mut self) -> Dirty {
        let taken = std::mem::take(&mut self.dirty);
        self.dirty = Dirty {
            buffers: SlotMask::with_capacity(self.buffers.len()),
            textures: SlotMask::with_capacity(self.textures.len()),
            samplers: SlotMask::with_capacity(self.samplers.len()),
        };
        self.census.emitted += taken.len();
        taken
    }

    /// Everything the driver believed is gone: a new command buffer, or a
    /// pipeline layout incompatible with the previous one.
    ///
    /// Every *bound* slot becomes dirty. Unbound slots do not, because there is
    /// nothing to write for them — which is also why a fresh table's first draw
    /// emits nothing rather than the whole table.
    pub fn disturb_all(&mut self) {
        self.census.disturbances += 1;
        mark_bound(&self.buffers, &mut self.dirty.buffers);
        mark_bound(&self.textures, &mut self.dirty.textures);
        mark_bound(&self.samplers, &mut self.dirty.samplers);
    }

    /// Forget every binding as well as every belief about them.
    ///
    /// For a semantic reset, where the objects the slots named are gone. Not
    /// the same as [`Self::disturb_all`], which keeps the contents because the
    /// guest has not unbound anything.
    pub fn reset(&mut self) {
        self.buffers.iter_mut().for_each(|s| *s = None);
        self.textures.iter_mut().for_each(|s| *s = None);
        self.samplers.iter_mut().for_each(|s| *s = None);
        self.dirty.buffers.clear();
        self.dirty.textures.clear();
        self.dirty.samplers.clear();
    }
}

/// Add every bound slot to `mask`, keeping what it already holds.
///
/// A union and not a replacement. A slot the guest unbound since the last
/// emission is dirty and *not* bound, so a mask built from the bound set alone
/// would clear it — and the emission that followed would write every bound
/// slot into a live set and leave the unbound one naming the resource the
/// guest just released. A disturbance is a claim about what the driver
/// believes, never permission to forget what the guest did.
///
/// A word at a time so the cost is the same as writing the whole mask.
fn mark_bound<T>(slots: &[Option<T>], mask: &mut SlotMask) {
    mask.reserve(slots.len());
    for (index, chunk) in slots.chunks(64).enumerate() {
        let mut bound = 0u64;
        for (bit, entry) in chunk.iter().enumerate() {
            if entry.is_some() {
                bound |= 1 << bit;
            }
        }
        mask.words[index] |= bound;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};

    /// Consume an emission whose contents this test is not asserting about.
    /// `Dirty` is `#[must_use]` because slots reported and not written leave the
    /// descriptor stale, so a test drops one deliberately or not at all.
    fn emit(t: &mut BindingTable) -> usize {
        t.take_dirty().len()
    }

    fn res(slot: u32, generation: u64) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(generation),
        }
    }

    fn buffer(slot: u32, offset: u64) -> Option<BufferBinding> {
        Some(BufferBinding {
            buffer: Some(res(slot, 1)),
            offset,
            stride: None,
        })
    }

    fn texture(slot: u32) -> Option<ObjectBinding> {
        Some(ObjectBinding {
            object: Some(res(slot, 1)),
            lod_clamps: None,
        })
    }

    fn object(slot: u32, generation: u64) -> Option<ObjectBinding> {
        Some(ObjectBinding {
            object: Some(res(slot, generation)),
            lod_clamps: None,
        })
    }

    fn table() -> BindingTable {
        BindingTable::new(32, 128, 16)
    }

    /// The gate: a guest that binds the same things every draw emits nothing
    /// after the first.
    #[test]
    fn a_steady_state_draw_emits_nothing() {
        let mut t = table();
        for slot in 0..40 {
            t.bind_texture(slot, texture(slot as u32));
        }
        assert_eq!(t.take_dirty().len(), 40, "the first draw writes them all");
        for _ in 0..100 {
            for slot in 0..40 {
                assert!(
                    !t.bind_texture(slot, texture(slot as u32)),
                    "rebinding the same value is not a change"
                );
            }
            assert!(t.is_clean());
            assert!(t.take_dirty().is_empty());
        }
        assert_eq!(
            t.census().emitted,
            40,
            "forty writes across a hundred draws"
        );
        assert_eq!(t.census().redundant, 4000);
    }

    /// Only what moved.
    #[test]
    fn only_the_slots_that_changed_are_emitted() {
        let mut t = table();
        for slot in 0..8 {
            t.bind_buffer(slot, buffer(slot as u32, 0));
        }
        emit(&mut t);
        t.bind_buffer(3, buffer(3, 256));
        t.bind_buffer(5, buffer(99, 0));
        let dirty = t.take_dirty();
        assert_eq!(dirty.buffers.slots().collect::<Vec<_>>(), vec![3, 5]);
        assert!(dirty.textures.is_empty());
    }

    /// A slot's offset is part of its binding, so the same resource at a
    /// different offset is a different binding.
    #[test]
    fn a_different_window_of_one_buffer_is_a_change() {
        let mut t = table();
        t.bind_buffer(0, buffer(1, 0));
        emit(&mut t);
        assert!(t.bind_buffer(0, buffer(1, 64)));
        assert_eq!(t.take_dirty().buffers.slots().collect::<Vec<_>>(), vec![0]);
    }

    /// Two different resources may occupy one native handle across a retire and
    /// a recreate. Comparing by identity is what stops a slot reading clean
    /// while pointing at somebody else's memory.
    #[test]
    fn a_reused_slot_at_a_new_generation_is_a_change() {
        let mut t = table();
        t.bind_texture(0, object(7, 1));
        emit(&mut t);
        assert!(
            t.bind_texture(0, object(7, 2)),
            "the same object-list slot at a later generation is a different resource"
        );
    }

    /// A tracker that read `None` as "no news" would leave a descriptor
    /// pointing at a resource the guest has released.
    #[test]
    fn unbinding_a_slot_is_a_change_and_binding_nothing_twice_is_not() {
        let mut t = table();
        t.bind_texture(4, texture(1));
        emit(&mut t);
        assert!(t.bind_texture(4, None));
        assert_eq!(t.take_dirty().textures.slots().collect::<Vec<_>>(), vec![4]);
        assert!(t.texture(4).is_none());
        assert!(!t.bind_texture(4, None), "it was already unbound");
    }

    /// A fresh command buffer has no bindings, so everything bound has to be
    /// written again — and nothing unbound does.
    #[test]
    fn a_disturbance_dirties_the_bound_slots_and_only_those() {
        let mut t = table();
        t.bind_buffer(0, buffer(1, 0));
        t.bind_buffer(7, buffer(2, 0));
        t.bind_texture(31, texture(3));
        t.bind_texture(64, texture(4));
        emit(&mut t);
        assert!(t.is_clean());

        t.disturb_all();
        let dirty = t.take_dirty();
        assert_eq!(dirty.buffers.slots().collect::<Vec<_>>(), vec![0, 7]);
        assert_eq!(
            dirty.textures.slots().collect::<Vec<_>>(),
            vec![31, 64],
            "and the mask spans more than one word"
        );
        assert!(dirty.samplers.is_empty(), "nothing was bound to a sampler");
        assert_eq!(t.census().disturbances, 1);
    }

    /// A fresh table's first draw writes nothing: an unbound slot has no
    /// descriptor to write, and marking them all would make every command
    /// buffer start by emitting a full table.
    #[test]
    fn a_fresh_table_is_clean() {
        let mut t = table();
        assert!(t.is_clean());
        t.disturb_all();
        assert!(t.is_clean(), "nothing is bound, so nothing is stale");
    }

    /// A reset is the guest's objects going away, not the driver's belief.
    #[test]
    fn a_reset_forgets_the_bindings_and_a_disturbance_does_not() {
        let mut t = table();
        t.bind_buffer(1, buffer(1, 0));
        emit(&mut t);

        let mut disturbed = t.clone();
        disturbed.disturb_all();
        assert_eq!(disturbed.buffer(1), buffer(1, 0), "still bound");
        assert!(!disturbed.is_clean(), "and owed a write");

        t.reset();
        assert_eq!(t.buffer(1), None);
        assert!(t.is_clean(), "there is nothing left to write");
    }

    /// A driven history of the table against a shadow that knows only what the
    /// *calls* said.
    ///
    /// The shadow is two plain arrays per class and no state machine: `held` is
    /// what the guest last bound, updated from the bind arguments, and
    /// `written` is what the descriptor set contains, updated only when an
    /// emission hands the slot over. Nothing in it consults the table, so the
    /// two cannot be wrong together.
    ///
    /// The claim under test is the one the module exists for and the one a
    /// stale descriptor breaks: **after an emission, the set contains exactly
    /// what the guest bound.** Not "the bound slots are right" — a slot the
    /// guest *unbound* must stop naming the resource it named, because that
    /// resource can then be freed, and a weaker invariant would pass the very
    /// history that goes wrong.
    ///
    /// `disturb_all` therefore leaves `written` alone. It is a claim about what
    /// the driver *believes*, not about what the set holds: an incompatible
    /// pipeline layout unbinds the set and changes none of its descriptors, and
    /// the emission that follows may reuse that same live set. Modelling it as
    /// "the set is empty now" would describe only the fresh-set path and would
    /// pass the reuse path, which is where the fault is.
    ///
    /// `reset` is the one place the sweep assumes a pairing the table cannot
    /// enforce: it means the objects are gone, and the caller resets the
    /// descriptor pool with it — `SetRing::reset` is the other half. The shadow
    /// clears both sides to match, and that assumption is stated here rather
    /// than hidden in the arithmetic.
    #[test]
    fn a_driven_history_never_leaves_the_set_disagreeing_with_the_guest() {
        #[derive(Clone, PartialEq, Eq, Debug)]
        struct Shadow {
            buffers: Vec<Option<BufferBinding>>,
            textures: Vec<Option<ObjectBinding>>,
            samplers: Vec<Option<ObjectBinding>>,
        }

        impl Shadow {
            fn new(b: usize, t: usize, s: usize) -> Self {
                Self {
                    buffers: vec![None; b],
                    textures: vec![None; t],
                    samplers: vec![None; s],
                }
            }
        }

        let mut rng: u64 = 0x243f_6a88_85a3_08d3;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        // Counted so a floor can say the interesting paths were driven rather
        // than assumed. `stale_window` is the fault's exact shape: an unbind
        // that a disturbance sat on top of before it was emitted.
        let (mut changed, mut redundant, mut ignored) = (0u64, 0u64, 0u64);
        let (mut unbinds, mut emissions, mut disturbances, mut resets) = (0u64, 0u64, 0u64, 0u64);
        let mut stale_window = 0u64;

        for (buffers, textures, samplers) in [(6usize, 5usize, 4usize), (3, 70, 2)] {
            for _ in 0..300 {
                let mut table = BindingTable::new(buffers, textures, samplers);
                let mut held = Shadow::new(buffers, textures, samplers);
                let mut written = Shadow::new(buffers, textures, samplers);
                // Slots unbound since the last emission, so the counter below
                // can say the fault's window was actually entered.
                let mut pending_unbind = false;
                let mut binds = 0u64;

                for _ in 0..60 {
                    // Steered: three quarters of the operations are binds, so
                    // the table spends its history in the state it exists for
                    // rather than in emissions over nothing.
                    match next() % 16 {
                        0..=8 => {
                            binds += 1;
                            let class = next() % 3;
                            // Occasionally past the table, which is the third
                            // census answer and must not be a change.
                            let width = match class {
                                0 => buffers,
                                1 => textures,
                                _ => samplers,
                            };
                            let slot = if next() % 8 == 0 {
                                width + (next() as usize % 4)
                            } else {
                                next() as usize % width.max(1)
                            };
                            // A small value space, so a rebind of the same
                            // value is frequent and the redundant path is real.
                            let name = (next() % 3) as u32;
                            let unbind = next() % 4 == 0;
                            let before = table.census();
                            let was_clean = table.is_clean();
                            // Whether the bind names what is already there.
                            // Read off the shadow, so the claim below is not
                            // the table agreeing with itself.
                            let names_the_same;
                            match class {
                                0 => {
                                    let value = if unbind { None } else { buffer(name, 0) };
                                    names_the_same = held.buffers.get(slot) == Some(&value);
                                    table.bind_buffer(slot, value);
                                    if let Some(entry) = held.buffers.get_mut(slot) {
                                        if *entry != value && value.is_none() {
                                            pending_unbind = true;
                                        }
                                        *entry = value;
                                    }
                                }
                                1 => {
                                    let value = if unbind { None } else { texture(name) };
                                    names_the_same = held.textures.get(slot) == Some(&value);
                                    table.bind_texture(slot, value);
                                    if let Some(entry) = held.textures.get_mut(slot) {
                                        if *entry != value && value.is_none() {
                                            pending_unbind = true;
                                        }
                                        *entry = value;
                                    }
                                }
                                _ => {
                                    let value = if unbind { None } else { texture(name) };
                                    names_the_same = held.samplers.get(slot) == Some(&value);
                                    table.bind_sampler(slot, value);
                                    if let Some(entry) = held.samplers.get_mut(slot) {
                                        if *entry != value && value.is_none() {
                                            pending_unbind = true;
                                        }
                                        *entry = value;
                                    }
                                }
                            }
                            if unbind && slot < width {
                                unbinds += 1;
                            }
                            // The gate the whole module exists for: a guest
                            // binding the forty textures it already had must
                            // leave a clean table clean, or the steady-state
                            // draw rebuilds its bindings every frame. Asserted
                            // per bind rather than only through the census
                            // floors, which aggregate the three classes and so
                            // let one of them stop detecting redundancy while
                            // the other two cover for it.
                            if was_clean && names_the_same {
                                assert!(
                                    table.is_clean(),
                                    "a bind that changed nothing dirtied the table"
                                );
                            }
                            let after = table.census();
                            changed += (after.changed - before.changed) as u64;
                            redundant += (after.redundant - before.redundant) as u64;
                            ignored += (after.ignored - before.ignored) as u64;
                        }
                        9..=10 => {
                            // An emission. Hands over exactly the slots the
                            // table reports, and nothing else.
                            if table.is_clean() {
                                // The claim a clean draw rests on: it writes
                                // nothing, so the set must already agree.
                                assert_eq!(written, held, "clean and disagreeing");
                            }
                            let dirty = table.take_dirty();
                            emissions += 1;
                            for slot in dirty.buffers.slots() {
                                written.buffers[slot] = held.buffers[slot];
                            }
                            for slot in dirty.textures.slots() {
                                written.textures[slot] = held.textures[slot];
                            }
                            for slot in dirty.samplers.slots() {
                                written.samplers[slot] = held.samplers[slot];
                            }
                            pending_unbind = false;
                            assert_eq!(
                                written, held,
                                "the set disagrees with the guest after an emission"
                            );
                            // And the emission took everything: a table still
                            // owed a write after handing one over would emit
                            // the same slots on the next draw forever.
                            assert!(table.is_clean(), "an emission left the table dirty");
                        }
                        11..=13 => {
                            if pending_unbind {
                                stale_window += 1;
                            }
                            table.disturb_all();
                            disturbances += 1;
                            // Deliberately not touching `written`: see the doc.
                        }
                        // Gated inside the arm rather than given a narrower
                        // range: a reset truncates a history, and too many of
                        // them leave every history too short to reach the
                        // states the sweep is for.
                        14 if next() % 4 == 0 => {
                            table.reset();
                            resets += 1;
                            held = Shadow::new(buffers, textures, samplers);
                            written = Shadow::new(buffers, textures, samplers);
                            assert!(table.is_clean());
                        }
                        _ => {
                            if table.is_clean() {
                                assert_eq!(written, held, "clean and disagreeing");
                            }
                        }
                    }
                }

                let census = table.census();
                assert_eq!(
                    census.changed + census.redundant + census.ignored,
                    binds as usize,
                    "every bind lands in exactly one census answer"
                );
            }
        }

        // Floors on each path separately, never on an aggregate: one busy path
        // cannot cover for another that never ran.
        assert!(changed > 2_000, "changed={changed}");
        assert!(redundant > 2_000, "redundant={redundant}");
        assert!(ignored > 200, "ignored={ignored}");
        assert!(unbinds > 1_000, "unbinds={unbinds}");
        assert!(emissions > 3_000, "emissions={emissions}");
        assert!(disturbances > 3_000, "disturbances={disturbances}");
        assert!(resets > 400, "resets={resets}");
        // The fault's own window, without which the sweep would prove only
        // that the ordinary path works.
        assert!(stale_window > 1_000, "stale_window={stale_window}");
    }

    /// A slot the guest unbound is a change, and a disturbance is a claim
    /// about the driver's belief rather than permission to forget it. The
    /// disturbance built its mask from the *bound* slots and overwrote the
    /// dirty set with it, so the unbind's bit was cleared; the emission that
    /// followed took a live holder, wrote every bound slot into it, and left
    /// the unbound slot's descriptor naming the released resource.
    #[test]
    fn a_disturbance_does_not_swallow_an_unbind_that_has_not_been_emitted() {
        let mut t = table();
        t.bind_texture(5, texture(9));
        t.bind_texture(6, texture(10));
        emit(&mut t);
        assert!(t.is_clean());

        assert!(t.bind_texture(5, None), "an unbind is a change");
        t.disturb_all();

        let dirty = t.take_dirty();
        assert!(dirty.textures.contains(5), "the unbind still owes a write");
        assert!(
            dirty.textures.contains(6),
            "and so does the slot still bound"
        );
    }

    /// The union must not invent slots either: a disturbance still owes
    /// nothing for a slot that was never bound.
    #[test]
    fn a_disturbance_still_marks_only_what_is_bound_or_already_dirty() {
        let mut t = BindingTable::new(4, 4, 4);
        t.bind_texture(2, texture(1));
        emit(&mut t);
        t.disturb_all();

        let dirty = t.take_dirty();
        assert_eq!(dirty.textures.slots().collect::<Vec<_>>(), vec![2]);
        assert!(dirty.buffers.is_empty());
        assert!(dirty.samplers.is_empty());
    }

    /// The argument-table sizes are the guest's own capacity hints. A record
    /// past them names a slot no shader on this pipeline can read.
    #[test]
    fn a_slot_past_the_table_is_not_a_change() {
        let mut t = BindingTable::new(4, 4, 4);
        assert!(!t.bind_buffer(4, buffer(1, 0)));
        assert!(t.is_clean());
        assert_eq!(t.buffer(4), None);
    }

    /// Dropped and counted. A bind the table refuses is neither a change nor a
    /// redundancy, so a census that did not name it would report three
    /// counters that no longer add up to the binds the guest issued.
    #[test]
    fn a_dropped_bind_is_counted_apart_from_a_redundant_one() {
        let mut t = BindingTable::new(4, 4, 4);
        t.bind_buffer(0, buffer(1, 0));
        t.bind_buffer(0, buffer(1, 0));
        t.bind_buffer(9, buffer(1, 0));
        t.bind_texture(9, texture(1));
        t.bind_sampler(9, texture(1));

        let census = t.census();
        assert_eq!(census.changed, 1);
        assert_eq!(census.redundant, 1);
        assert_eq!(census.ignored, 3, "one per table, and none of them silent");
        assert_eq!(
            census.changed + census.redundant + census.ignored,
            5,
            "every bind is accounted to exactly one of the three"
        );
    }

    #[test]
    fn the_mask_reports_exactly_the_slots_it_holds() {
        let mut m = SlotMask::with_capacity(8);
        assert!(m.is_empty());
        m.insert(0);
        m.insert(63);
        m.insert(64);
        m.insert(200);
        assert_eq!(m.slots().collect::<Vec<_>>(), vec![0, 63, 64, 200]);
        assert_eq!(m.len(), 4);
        assert!(m.contains(64));
        assert!(!m.contains(65));
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.slots().collect::<Vec<_>>(), Vec::<usize>::new());
    }
}
