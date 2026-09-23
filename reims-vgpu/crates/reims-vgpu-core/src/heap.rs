//! Heaps: one piece of storage, several resources placed in it, and a lifetime
//! that outlives the guest's delete.
//!
//! # A placement is a coordinate, not a new piece of memory
//!
//! A heap-allocated resource does not own storage. It occupies a window of the
//! heap's storage at an offset the guest chose, and two windows the guest chose
//! to overlap *are* the same bytes — that is what a heap is for. So a placement
//! does not get its own [`BackingId`]. It gets the heap's, and a byte range
//! within it.
//!
//! That single decision is what makes partial aliasing come out right for free.
//! [`crate::access::AccessKey::Range`] already conflicts on overlapping ranges of one backing,
//! so two resources sharing the front half of a heap window conflict on the
//! front half and not on the rest, with no alias closure to compute, no second
//! storage identity to keep consistent, and nothing that has to be told which
//! resources were declared aliasable. A design that minted a backing per
//! placement would have had to reconstruct all of that, and would have had to
//! be *told* about the overlap by something that measured it.
//!
//! # The arithmetic has one checked place
//!
//! A resource's own coordinates are relative to its placement, and the graph's
//! are relative to the heap. Converting between them is an addition, and an
//! addition performed at call sites is the class of slip that reaches a
//! neighbouring resource's bytes with the GPU's write access. There is exactly
//! one conversion, [`HeapPlacement::within`], it is checked against the
//! placement's own length, and there is no public field a caller could add to
//! something itself: `region` is the heap-relative window and reading it gives
//! no license to offset into it.
//!
//! # Deletion is the namespace rule, applied to storage
//!
//! [`crate::namespace`] establishes that a delete stops resolution and does not
//! stop work. A heap adds a second holder: its own allocations. Guest code
//! releases a heap while resources placed in it are still alive, and the heap's
//! storage is those resources' storage — freeing it on the delete frees memory
//! a live resource is reading.
//!
//! So the delete is accepted, the heap stops resolving, and the storage is
//! handed back at whichever comes last: the delete, or the removal of the final
//! allocation. [`Retirement`] is that answer, and it is `#[must_use]` because a
//! caller that dropped it either frees storage under a live resource or leaks
//! the heap. The wait has no bound and no eviction: there is no lawful loss
//! here, and a heap that outlives its delete for a long time is a guest that
//! kept its allocations, not a leak to be capped.
//!
//! # A resource is placed once, and that is a claim about the resource
//!
//! A placement *is* the resource's storage: the [`BackingId`] it carries is
//! the answer to where the resource's bytes are. Two placements are two
//! answers, and the second one leaves the first's compiled coordinates naming
//! bytes nothing agrees about — in one heap they are the wrong offset, and
//! across two heaps they are the wrong storage entirely. So
//! [`Refusal::AlreadyPlaced`] is checked against every heap here, live or
//! retiring behind a delete, and not against the one being placed into. It is
//! the same cross-heap closure [`Refusal::StorageInUse`] applies from the
//! storage's side.
//!
//! `placed` is the index that makes the check a lookup rather than a scan of
//! every placement in every heap, which a resource declaration would otherwise
//! pay on the way in. It is maintained at the two operations that change
//! membership and checked against the heaps themselves after every step of the
//! driven sweep.
//!
//! # Membership is recorded, and it is not the aliasing question
//!
//! Placing or removing a resource advances the heap's membership generation,
//! and [`Heaps::membership`] stamps the generation into the [`HeapId`] a
//! command records. That says which set a `useHeap` record was written
//! against. It is deliberately not what decides whether two accesses meet —
//! see [`HeapId::same_heap`], which exists because a resource that never moved
//! must keep meeting a declaration written before its neighbour arrived.

use crate::access::{BackingId, ByteRange, HeapId, ResourceKey};
use crate::identity::ResourceId;
use std::collections::HashMap;

/// Why a heap operation did not happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No live heap has this number. It was never created, or it was deleted:
    /// a delete stops the number resolving, and after it there is nothing here
    /// to tell the two apart with that a later `create` would not invalidate.
    NoSuchHeap { heap: u64 },
    /// The heap number is already in use by a live heap. A guest that wants it
    /// back deletes first; silently replacing would strand the allocations
    /// placed in the previous occupant.
    HeapExists { heap: u64 },
    /// The window does not fit inside the heap, or its end overflows.
    OutOfHeap {
        heap: u64,
        offset: u64,
        length: u64,
        heap_length: u64,
    },
    /// The window does not fit inside the placement, or its end overflows.
    OutOfPlacement {
        offset: u64,
        length: u64,
        placement_length: u64,
    },
    /// This resource is already placed, in the storage named. A second
    /// placement would leave the first one's compiled coordinates naming bytes
    /// nothing agrees about — and across two heaps the two placements would not
    /// even agree which storage the resource is in, so this is a claim about the
    /// resource and not about one heap. See the module doc.
    AlreadyPlaced {
        resource: ResourceId,
        backing: BackingId,
    },
    /// This resource is not placed in the storage the placement names.
    NotPlaced { resource: ResourceId },
    /// The resource is placed in this storage, but not at the window the
    /// placement names — so the placement is a stale one, minted before an
    /// earlier removal, and the window it names now belongs to a later
    /// placement somebody else holds. Honouring it would remove that one.
    StalePlacement {
        resource: ResourceId,
        named: ByteRange,
        placed_at: ByteRange,
    },
    /// The placement names storage no heap and no retirement holds. Either it
    /// was already removed, or it was minted by a heap whose number has since
    /// been reused for different storage.
    StaleStorage { backing: BackingId },
    /// A live heap, or storage still retiring behind a deleted one, already
    /// holds this backing.
    ///
    /// Storage is what identifies a heap once its number is gone — a retiring
    /// heap is reachable only through the placements handed out over it, and
    /// those name the backing. Two heaps over one backing therefore have one
    /// retirement between them: the second delete replaces the first's set of
    /// live allocations, and the last removal of *that* set reports the storage
    /// free while the first heap's resources are still in it. That is the exact
    /// failure this module's deletion rule exists to prevent, so it is refused
    /// at the point the second heap is created rather than diagnosed later.
    ///
    /// It would also make the aliasing answer wrong in the other direction:
    /// a placement is a coordinate in its heap's backing, so two placements at
    /// one offset in two heaps over one backing *are* the same bytes, while
    /// [`HeapId::same_heap`] says they are in different heaps and their accesses
    /// never meet.
    StorageInUse { backing: BackingId },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoSuchHeap { .. } => "heap_no_such_heap",
            Self::HeapExists { .. } => "heap_exists",
            Self::OutOfHeap { .. } => "heap_out_of_heap",
            Self::OutOfPlacement { .. } => "heap_out_of_placement",
            Self::AlreadyPlaced { .. } => "heap_already_placed",
            Self::NotPlaced { .. } => "heap_not_placed",
            Self::StalePlacement { .. } => "heap_stale_placement",
            Self::StaleStorage { .. } => "heap_stale_storage",
            Self::StorageInUse { .. } => "heap_storage_in_use",
        }
    }
}

/// What a delete or a removal left of the heap's storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "storage nothing reports as free is either freed under a live resource or leaked"]
pub enum Retirement {
    /// The heap is still named, or still holds allocations. Its storage stays.
    Held { allocations: usize, named: bool },
    /// The heap's number is gone and its last allocation is gone. Nothing can
    /// reach the storage again, so it may be torn down now.
    StorageFree { backing: BackingId },
}

/// A resource's window of a heap.
///
/// Carries the heap's [`BackingId`] rather than one of its own, because that is
/// what it is: the resource occupies the heap's storage. `region` is
/// heap-relative, which is the coordinate system the dependency graph compares
/// in, and [`HeapPlacement::within`] is the only way to turn a resource-local
/// window into one.
///
/// The backing travels with it for a second reason. A heap number may be
/// deleted and reused, and a holder that presented the *number* back would
/// remove its allocation from whatever now answers to it. [`Heaps::remove`]
/// takes the whole placement, so a stale one refuses instead — and a placement
/// goes stale two ways, not one. Its heap's number can be reused for other
/// storage, which `backing` answers; and the resource can be removed and placed
/// in the same heap again, which only `region` answers, because the later
/// placement equals the earlier one in every other field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapPlacement {
    pub heap: u64,
    pub backing: BackingId,
    pub region: ByteRange,
}

impl HeapPlacement {
    /// The access key for this placement at a membership generation.
    #[must_use]
    pub const fn key(self, membership: HeapId) -> ResourceKey {
        ResourceKey {
            backing: self.backing,
            heap: Some(membership),
        }
    }

    /// A window of this resource, in the heap's coordinates.
    ///
    /// `offset` and `length` are relative to the resource, which is how every
    /// command that names part of a buffer expresses itself. The result is
    /// relative to the heap, which is what the dependency graph compares.
    ///
    /// # Errors
    ///
    /// If the window runs past the end of the placement, or its end overflows.
    /// Both are the same defect — an access naming a neighbouring resource's
    /// bytes — and neither is clamped, because a clamped access silently
    /// executes against memory the guest did not name.
    pub fn within(self, offset: u64, length: u64) -> Result<ByteRange, Refusal> {
        if offset
            .checked_add(length)
            .is_none_or(|end| end > self.region.length)
        {
            return Err(Refusal::OutOfPlacement {
                offset,
                length,
                placement_length: self.region.length,
            });
        }
        Ok(ByteRange {
            // Cannot overflow: `offset` is at most the placement's length, and
            // the placement's end was checked against the heap's when it was
            // created.
            offset: self.region.offset.saturating_add(offset),
            length,
        })
    }

    /// The whole resource, in the heap's coordinates.
    #[must_use]
    pub const fn whole(self) -> ByteRange {
        self.region
    }
}

#[derive(Debug)]
struct Heap {
    backing: BackingId,
    length: u64,
    membership_generation: u64,
    placements: HashMap<ResourceId, ByteRange>,
}

/// Storage whose heap number is gone, kept alive by the allocations still in it.
#[derive(Debug)]
struct Retiring {
    /// The same map a live heap keeps, windows included. A retiring heap
    /// refuses a stale placement for the reason a live one does — see
    /// [`Heaps::remove`] — and the window is half of what makes a placement
    /// stale.
    allocations: HashMap<ResourceId, ByteRange>,
}

/// One session generation's heaps.
///
/// `heaps` holds only live ones: a delete removes the number, and storage the
/// delete could not free moves to `retiring`, where nothing can name it and
/// only the placements already handed out can retire it. Recreating the number
/// therefore cannot reach the previous occupant's storage — the two do not
/// share a key.
#[derive(Debug, Default)]
pub struct Heaps {
    heaps: HashMap<u64, Heap>,
    retiring: HashMap<BackingId, Retiring>,
    /// Where each placed resource's bytes are, across every heap and every
    /// retirement. The index behind [`Refusal::AlreadyPlaced`]; see the module
    /// doc for why the law is global and why this is not a scan.
    placed: HashMap<ResourceId, BackingId>,
}

impl Heaps {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a heap over a piece of storage.
    ///
    /// # Errors
    ///
    /// If a live heap already has this number, or any heap — live or retiring
    /// behind a delete — already holds this storage.
    pub fn create(&mut self, heap: u64, backing: BackingId, length: u64) -> Result<(), Refusal> {
        if self.heaps.contains_key(&heap) {
            return Err(Refusal::HeapExists { heap });
        }
        // Storage is the key a retirement is held under, and the identity a
        // placement carries once the number is gone, so one backing may belong
        // to at most one heap at a time. See [`Refusal::StorageInUse`].
        if self.holds_storage(backing) {
            return Err(Refusal::StorageInUse { backing });
        }
        self.heaps.insert(
            heap,
            Heap {
                backing,
                length,
                // Advances on every placement and every removal, so a command
                // records which membership set it was written against.
                membership_generation: 0,
                placements: HashMap::new(),
            },
        );
        Ok(())
    }

    /// The heap's identity at this command point.
    ///
    /// # Errors
    ///
    /// If no live heap has this number.
    pub fn membership(&self, heap: u64) -> Result<HeapId, Refusal> {
        let h = self.live(heap)?;
        Ok(HeapId {
            id: heap,
            membership_generation: h.membership_generation,
        })
    }

    /// The storage a live heap is over.
    ///
    /// # Errors
    ///
    /// If no live heap has this number.
    pub fn backing_of(&self, heap: u64) -> Result<BackingId, Refusal> {
        Ok(self.live(heap)?.backing)
    }

    /// Place a resource in the heap.
    ///
    /// # Errors
    ///
    /// If no live heap has this number, the window does not fit, or this
    /// resource is already placed here.
    /// Whether this heap would admit a window of this shape.
    ///
    /// The half of [`Self::place`] that does not need a name, so a caller can
    /// ask before it publishes one. [`crate::lifecycle::Lifecycle`] has to: a
    /// declaration displaces whatever its slot already held and that is not
    /// undoable, so a placement still able to refuse afterwards would strand
    /// the displaced occupant's teardown on an error path that carries no
    /// effects.
    ///
    /// `place` calls this rather than restating it, so there is one window
    /// rule and not two that can drift.
    ///
    /// # Errors
    ///
    /// [`Refusal::NoSuchHeap`] when no live heap has this number, and
    /// [`Refusal::OutOfHeap`] for a window the heap does not contain.
    pub fn admits(&self, heap: u64, offset: u64, length: u64) -> Result<(), Refusal> {
        let h = self.live(heap)?;
        if offset.checked_add(length).is_none_or(|end| end > h.length) {
            return Err(Refusal::OutOfHeap {
                heap,
                offset,
                length,
                heap_length: h.length,
            });
        }
        Ok(())
    }

    pub fn place(
        &mut self,
        heap: u64,
        resource: ResourceId,
        offset: u64,
        length: u64,
    ) -> Result<HeapPlacement, Refusal> {
        self.admits(heap, offset, length)?;
        if let Some(&backing) = self.placed.get(&resource) {
            return Err(Refusal::AlreadyPlaced { resource, backing });
        }
        let region = ByteRange { offset, length };
        let h = self.heaps.get_mut(&heap).expect("live above");
        h.placements.insert(resource, region);
        h.membership_generation += 1;
        let backing = h.backing;
        self.placed.insert(resource, backing);
        Ok(HeapPlacement {
            heap,
            backing,
            region,
        })
    }

    /// The placement of a resource already in the heap.
    ///
    /// # Errors
    ///
    /// If no live heap has this number, or the resource is not placed in it.
    pub fn placement(&self, heap: u64, resource: ResourceId) -> Result<HeapPlacement, Refusal> {
        let h = self.live(heap)?;
        let region = *h
            .placements
            .get(&resource)
            .ok_or(Refusal::NotPlaced { resource })?;
        Ok(HeapPlacement {
            heap,
            backing: h.backing,
            region,
        })
    }

    /// Remove a resource from the storage its placement names, and say what
    /// that left of the storage.
    ///
    /// Takes the placement rather than the heap number, so it reaches the
    /// storage the caller was actually given — whether that heap is still named
    /// or is retiring behind a reused number.
    ///
    /// The window is part of what it was given. A resource removed from a heap
    /// may be placed in it again at a different offset, and the two placements
    /// differ in nothing else, so a holder of the first one presenting it back
    /// would remove the second — leaving its holder unable to retire it and the
    /// index saying the resource is nowhere. That is the failure the backing
    /// already prevents for a reused heap number, one field along.
    ///
    /// # Errors
    ///
    /// If nothing holds that storage any more, the resource is not placed in
    /// it, or it is placed at a different window than the one named.
    pub fn remove(
        &mut self,
        placement: HeapPlacement,
        resource: ResourceId,
    ) -> Result<Retirement, Refusal> {
        if let Some(h) = self.heaps.get_mut(&placement.heap) {
            if h.backing == placement.backing {
                match h.placements.get(&resource) {
                    None => return Err(Refusal::NotPlaced { resource }),
                    Some(&placed_at) if placed_at != placement.region => {
                        return Err(Refusal::StalePlacement {
                            resource,
                            named: placement.region,
                            placed_at,
                        })
                    }
                    Some(_) => {}
                }
                h.placements.remove(&resource);
                h.membership_generation += 1;
                self.placed.remove(&resource);
                return Ok(Retirement::Held {
                    allocations: h.placements.len(),
                    named: true,
                });
            }
        }
        let Some(r) = self.retiring.get_mut(&placement.backing) else {
            return Err(Refusal::StaleStorage {
                backing: placement.backing,
            });
        };
        match r.allocations.get(&resource) {
            None => return Err(Refusal::NotPlaced { resource }),
            Some(&placed_at) if placed_at != placement.region => {
                return Err(Refusal::StalePlacement {
                    resource,
                    named: placement.region,
                    placed_at,
                })
            }
            Some(_) => {}
        }
        r.allocations.remove(&resource);
        self.placed.remove(&resource);
        if r.allocations.is_empty() {
            self.retiring.remove(&placement.backing);
            return Ok(Retirement::StorageFree {
                backing: placement.backing,
            });
        }
        Ok(Retirement::Held {
            allocations: r.allocations.len(),
            named: false,
        })
    }

    /// Delete the heap.
    ///
    /// The number stops resolving immediately and becomes available again. The
    /// storage is returned now only if nothing is placed in it; otherwise the
    /// last [`Heaps::remove`] returns it, with no bound on how long that takes
    /// and no eviction — a heap outliving its delete is a guest that kept its
    /// allocations, and there is no lawful loss to substitute for waiting.
    ///
    /// # Errors
    ///
    /// If no live heap has this number.
    pub fn delete(&mut self, heap: u64) -> Result<Retirement, Refusal> {
        let Some(h) = self.heaps.remove(&heap) else {
            return Err(Refusal::NoSuchHeap { heap });
        };
        if h.placements.is_empty() {
            return Ok(Retirement::StorageFree { backing: h.backing });
        }
        let allocations = h.placements;
        let held = allocations.len();
        self.retiring.insert(h.backing, Retiring { allocations });
        Ok(Retirement::Held {
            allocations: held,
            named: false,
        })
    }

    /// How many allocations are placed in a live heap.
    #[must_use]
    pub fn allocations(&self, heap: u64) -> usize {
        self.heaps.get(&heap).map_or(0, |h| h.placements.len())
    }

    /// Whether this storage is still held by a heap or a retirement.
    #[must_use]
    pub fn holds_storage(&self, backing: BackingId) -> bool {
        self.retiring.contains_key(&backing) || self.heaps.values().any(|h| h.backing == backing)
    }

    /// Every live heap number, sorted, so a teardown's order is a property of
    /// the heaps and not of a hash seed.
    #[must_use]
    pub fn live_heaps(&self) -> Vec<u64> {
        let mut out: Vec<u64> = self.heaps.keys().copied().collect();
        out.sort_unstable();
        out
    }

    /// How many pieces of storage are outliving their heap number.
    #[must_use]
    pub fn retiring_storage(&self) -> usize {
        self.retiring.len()
    }

    fn live(&self, heap: u64) -> Result<&Heap, Refusal> {
        self.heaps.get(&heap).ok_or(Refusal::NoSuchHeap { heap })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessKey, HeapId};
    use crate::identity::{ObjectListRef, SlotGeneration};
    use std::collections::HashSet;

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration::default().next(),
        }
    }

    fn heaps_with(length: u64) -> Heaps {
        let mut h = Heaps::new();
        h.create(7, BackingId(100), length).expect("first create");
        h
    }

    /// The reason a placement is a coordinate and not a piece of memory: two
    /// windows the guest chose to overlap are the same bytes, and the ordinary
    /// range conflict already says so. Nothing had to be told they alias.
    #[test]
    fn two_placements_alias_exactly_where_their_windows_meet() {
        let mut heaps = heaps_with(4096);
        let a = heaps.place(7, res(1), 0, 512).expect("a fits");
        let b = heaps.place(7, res(2), 256, 512).expect("b fits");
        let c = heaps.place(7, res(3), 2048, 512).expect("c fits");
        let m = heaps.membership(7).expect("live");
        assert_eq!(a.backing, b.backing, "one heap is one piece of storage");
        let key = |p: HeapPlacement, r: ByteRange| AccessKey::Range(p.key(m), r);
        assert!(
            key(a, a.whole()).may_alias(key(b, b.whole())),
            "overlapping windows are the same bytes"
        );
        assert!(
            !key(a, a.whole()).may_alias(key(c, c.whole())),
            "disjoint windows are not"
        );
        // And the overlap is exact, not whole-resource: the front of `a` and
        // the back of `b` never meet.
        let front_of_a = a.within(0, 128).expect("inside a");
        let back_of_b = b.within(384, 128).expect("inside b");
        assert!(!key(a, front_of_a).may_alias(key(b, back_of_b)));
        // While the tail of `a` and the stretch of `b` sitting under it do.
        let tail_of_a = a.within(384, 128).expect("inside a");
        let under_it = b.within(128, 128).expect("inside b");
        assert_eq!(tail_of_a, under_it, "the same heap bytes, named two ways");
        assert!(key(a, tail_of_a).may_alias(key(b, under_it)));
    }

    /// The one conversion between a resource's coordinates and the heap's, and
    /// the reason it is the only one: an addition at a call site reaches the
    /// neighbour's bytes.
    #[test]
    fn a_window_of_a_placement_cannot_leave_it() {
        let mut heaps = heaps_with(4096);
        let p = heaps.place(7, res(1), 1024, 256).expect("fits");
        assert_eq!(
            p.within(16, 32),
            Ok(ByteRange {
                offset: 1040,
                length: 32
            })
        );
        assert_eq!(
            p.within(0, 256),
            Ok(ByteRange {
                offset: 1024,
                length: 256
            })
        );
        assert_eq!(
            p.within(240, 32),
            Err(Refusal::OutOfPlacement {
                offset: 240,
                length: 32,
                placement_length: 256,
            }),
            "one byte past the end is not clamped to the end"
        );
        assert_eq!(
            p.within(8, u64::MAX),
            Err(Refusal::OutOfPlacement {
                offset: 8,
                length: u64::MAX,
                placement_length: 256,
            }),
            "and an overflowing end is the same defect, not a wrap"
        );
    }

    #[test]
    fn a_placement_cannot_leave_the_heap() {
        let mut heaps = heaps_with(1024);
        assert_eq!(
            heaps.place(7, res(1), 768, 512),
            Err(Refusal::OutOfHeap {
                heap: 7,
                offset: 768,
                length: 512,
                heap_length: 1024,
            })
        );
        assert_eq!(
            heaps.place(7, res(1), 1, u64::MAX),
            Err(Refusal::OutOfHeap {
                heap: 7,
                offset: 1,
                length: u64::MAX,
                heap_length: 1024,
            })
        );
        assert_eq!(
            heaps.allocations(7),
            0,
            "a refused placement is not a member"
        );
        assert_eq!(
            heaps.membership(7).map(|m| m.membership_generation),
            Ok(0),
            "and did not advance membership"
        );
    }

    #[test]
    fn placing_and_removing_both_advance_membership() {
        let mut heaps = heaps_with(4096);
        let generation = |h: &Heaps| h.membership(7).expect("live").membership_generation;
        assert_eq!(generation(&heaps), 0);
        let a = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_eq!(generation(&heaps), 1);
        heaps.place(7, res(2), 64, 64).expect("fits");
        assert_eq!(generation(&heaps), 2);
        assert_eq!(
            heaps.remove(a, res(1)),
            Ok(Retirement::Held {
                allocations: 1,
                named: true
            })
        );
        assert_eq!(generation(&heaps), 3);
    }

    #[test]
    fn a_resource_is_placed_once_and_removed_once() {
        let mut heaps = heaps_with(4096);
        let a = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_eq!(
            heaps.place(7, res(1), 128, 64),
            Err(Refusal::AlreadyPlaced {
                resource: res(1),
                backing: BackingId(100)
            }),
            "a second placement would leave the first one's coordinates unowned"
        );
        // **And the law is about the resource, not about one heap.** A second
        // heap is different storage, so the two placements would not even agree
        // which bytes the resource is made of.
        heaps
            .create(8, BackingId(101), 4096)
            .expect("a second heap");
        assert_eq!(
            heaps.place(8, res(1), 0, 64),
            Err(Refusal::AlreadyPlaced {
                resource: res(1),
                backing: BackingId(100)
            }),
            "the refusal names the storage the resource is already in"
        );
        // It outlives the heap's number, too: a retiring heap still holds the
        // allocation, and that placement is still the caller's to present.
        assert_eq!(
            heaps.delete(7),
            Ok(Retirement::Held {
                allocations: 1,
                named: false
            })
        );
        assert_eq!(
            heaps.place(8, res(1), 0, 64),
            Err(Refusal::AlreadyPlaced {
                resource: res(1),
                backing: BackingId(100)
            })
        );
        assert_eq!(
            heaps.remove(a, res(1)),
            Ok(Retirement::StorageFree {
                backing: BackingId(100)
            })
        );
        let b = heaps.place(8, res(1), 0, 64).expect("nothing holds it now");
        assert_eq!(b.backing, BackingId(101));
        assert_eq!(
            heaps.remove(b, res(1)),
            Ok(Retirement::Held {
                allocations: 0,
                named: true
            })
        );

        heaps.create(7, BackingId(100), 4096).expect("free again");
        let a = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_eq!(heaps.placement(7, res(1)), Ok(a));
        assert_eq!(
            heaps.placement(7, res(2)),
            Err(Refusal::NotPlaced { resource: res(2) })
        );
        assert_eq!(
            heaps.remove(a, res(1)),
            Ok(Retirement::Held {
                allocations: 0,
                named: true
            })
        );
        assert_eq!(
            heaps.remove(a, res(1)),
            Err(Refusal::NotPlaced { resource: res(1) })
        );
    }

    /// The whole reason heaps have a lifetime module of their own: the guest
    /// releases the heap while resources placed in it are alive, and the heap's
    /// storage *is* their storage.
    #[test]
    fn deleting_a_heap_with_allocations_hands_the_storage_to_them() {
        let mut heaps = heaps_with(4096);
        let a = heaps.place(7, res(1), 0, 64).expect("fits");
        let b = heaps.place(7, res(2), 64, 64).expect("fits");
        assert_eq!(
            heaps.delete(7),
            Ok(Retirement::Held {
                allocations: 2,
                named: false
            })
        );
        assert_eq!(
            heaps.membership(7),
            Err(Refusal::NoSuchHeap { heap: 7 }),
            "the delete stopped the number resolving"
        );
        assert!(
            heaps.holds_storage(BackingId(100)),
            "and did not free storage"
        );
        assert_eq!(
            heaps.remove(a, res(1)),
            Ok(Retirement::Held {
                allocations: 1,
                named: false
            })
        );
        assert!(heaps.holds_storage(BackingId(100)));
        assert_eq!(
            heaps.remove(b, res(2)),
            Ok(Retirement::StorageFree {
                backing: BackingId(100)
            }),
            "the last allocation completes the handoff"
        );
        assert!(!heaps.holds_storage(BackingId(100)));
        assert_eq!(heaps.retiring_storage(), 0);
    }

    #[test]
    fn deleting_an_empty_heap_frees_its_storage_now() {
        let mut heaps = heaps_with(4096);
        assert_eq!(
            heaps.delete(7),
            Ok(Retirement::StorageFree {
                backing: BackingId(100)
            })
        );
        assert_eq!(heaps.delete(7), Err(Refusal::NoSuchHeap { heap: 7 }));
        assert!(!heaps.holds_storage(BackingId(100)));
    }

    /// The sibling of the reused-number case below, and the one the backing
    /// cannot answer: a resource removed from a heap may be placed in it again,
    /// and the second placement differs from the first only in its window. A
    /// holder of the first presenting it back would remove the second, leaving
    /// its holder unable to retire it and the index saying the resource is in
    /// no heap at all.
    #[test]
    fn a_placement_superseded_by_a_later_one_cannot_remove_it() {
        let mut heaps = heaps_with(4096);
        let first = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_eq!(
            heaps.remove(first, res(1)),
            Ok(Retirement::Held {
                allocations: 0,
                named: true
            })
        );
        let second = heaps
            .place(7, res(1), 1024, 64)
            .expect("unplaced, so it fits");
        assert_eq!(first.heap, second.heap);
        assert_eq!(first.backing, second.backing);
        assert_eq!(
            heaps.remove(first, res(1)),
            Err(Refusal::StalePlacement {
                resource: res(1),
                named: ByteRange {
                    offset: 0,
                    length: 64
                },
                placed_at: ByteRange {
                    offset: 1024,
                    length: 64
                },
            })
        );
        assert_eq!(
            heaps.allocations(7),
            1,
            "and the live placement is still placed"
        );
        assert_eq!(
            heaps.remove(second, res(1)),
            Ok(Retirement::Held {
                allocations: 0,
                named: true
            }),
            "so its holder can still retire it"
        );
    }

    /// Storage retiring behind a deleted number refuses a superseded placement
    /// for the same reason a live heap does. It is reachable only through the
    /// placements handed out over it, so accepting the wrong one is the last
    /// removal of storage a live allocation is still in.
    #[test]
    fn a_retiring_heap_refuses_a_superseded_placement_too() {
        let mut heaps = heaps_with(4096);
        let first = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_eq!(
            heaps.remove(first, res(1)),
            Ok(Retirement::Held {
                allocations: 0,
                named: true
            })
        );
        let second = heaps.place(7, res(1), 1024, 64).expect("unplaced");
        assert_eq!(
            heaps.delete(7),
            Ok(Retirement::Held {
                allocations: 1,
                named: false
            })
        );
        assert_eq!(
            heaps.remove(first, res(1)),
            Err(Refusal::StalePlacement {
                resource: res(1),
                named: ByteRange {
                    offset: 0,
                    length: 64
                },
                placed_at: ByteRange {
                    offset: 1024,
                    length: 64
                },
            }),
            "the storage is not free while `second` is in it"
        );
        assert!(heaps.holds_storage(BackingId(100)));
        assert_eq!(
            heaps.remove(second, res(1)),
            Ok(Retirement::StorageFree {
                backing: BackingId(100)
            })
        );
    }

    /// A heap number is a guest object reference and gets reused. The retiring
    /// storage behind it must be unreachable from the number, or a holder of an
    /// old placement removes its allocation from the new heap and the old
    /// storage never retires.
    #[test]
    fn a_reused_number_cannot_reach_the_previous_heaps_storage() {
        let mut heaps = heaps_with(4096);
        let old = heaps.place(7, res(1), 0, 64).expect("fits");
        assert_eq!(
            heaps.delete(7),
            Ok(Retirement::Held {
                allocations: 1,
                named: false
            })
        );
        heaps
            .create(7, BackingId(200), 4096)
            .expect("the number is available again");
        // A different resource, because `res(1)` is still placed in the storage
        // retiring behind the old number and a resource is placed once — see
        // [`Refusal::AlreadyPlaced`]. What is being tested is that the *number*
        // does not reach the old storage, and the old placement is what carries
        // it there.
        let new = heaps.place(7, res(2), 0, 64).expect("fits");
        assert_ne!(old.backing, new.backing);
        assert_eq!(
            heaps.remove(old, res(1)),
            Ok(Retirement::StorageFree {
                backing: BackingId(100)
            }),
            "the old placement retires the old storage"
        );
        assert_eq!(
            heaps.allocations(7),
            1,
            "and left the new heap's membership alone"
        );
        assert_eq!(
            heaps.remove(new, res(2)),
            Ok(Retirement::Held {
                allocations: 0,
                named: true
            }),
            "the new heap is still named, so its storage is still held"
        );
        assert!(heaps.holds_storage(BackingId(200)));
    }

    #[test]
    fn a_live_number_cannot_be_created_over() {
        let mut heaps = heaps_with(4096);
        assert_eq!(
            heaps.create(7, BackingId(200), 4096),
            Err(Refusal::HeapExists { heap: 7 })
        );
    }

    #[test]
    fn a_placement_naming_storage_nothing_holds_refuses() {
        let mut heaps = heaps_with(4096);
        let stranger = HeapPlacement {
            heap: 9,
            backing: BackingId(999),
            region: ByteRange {
                offset: 0,
                length: 64,
            },
        };
        assert_eq!(
            heaps.remove(stranger, res(1)),
            Err(Refusal::StaleStorage {
                backing: BackingId(999)
            })
        );
    }

    /// Membership stamps which set a record was written against; it is not what
    /// decides whether two accesses meet. `access::HeapId::same_heap` owns that,
    /// and this is the heap side of the same claim.
    #[test]
    fn a_key_recorded_at_one_membership_still_meets_a_later_declaration() {
        let mut heaps = heaps_with(4096);
        let a = heaps.place(7, res(1), 0, 64).expect("fits");
        let early = heaps.membership(7).expect("live");
        heaps.place(7, res(2), 2048, 64).expect("fits");
        let late = heaps.membership(7).expect("live");
        assert_ne!(early.membership_generation, late.membership_generation);
        assert!(
            AccessKey::Range(a.key(early), a.whole()).may_alias(AccessKey::Heap(HeapId { ..late }))
        );
    }

    /// One backing belongs to at most one heap, and the second heap over it is
    /// refused rather than admitted.
    ///
    /// Admitted, it produced the failure the deletion rule exists to prevent.
    /// Two heaps over backing `B`, one resource placed in each, both deleted:
    /// the second delete's `retiring.insert(B, ...)` replaced the first's set
    /// of live allocations, so removing the *second* heap's resource emptied
    /// what was left and reported `StorageFree` while the first heap's resource
    /// was still in that storage — and the first resource's own removal then
    /// refused as `StaleStorage`, which is the model saying the memory a live
    /// placement names is gone.
    ///
    /// Refused at creation rather than diagnosed at deletion, because by the
    /// time the second delete runs there is nothing left to refuse: both heaps
    /// were legal, both placements were legal, and the retirement map has one
    /// key for two sets.
    #[test]
    fn two_heaps_cannot_be_created_over_one_piece_of_storage() {
        let mut h = Heaps::new();
        let b = BackingId(9);
        h.create(1, b, 100)
            .expect("the first heap over this storage");
        assert_eq!(
            h.create(2, b, 100),
            Err(Refusal::StorageInUse { backing: b }),
            "a second heap over one backing gives the two one retirement"
        );
        // A different backing under a different number is unaffected.
        h.create(2, BackingId(10), 100).expect("its own storage");

        // And the storage is free again once the first heap is gone, so the
        // refusal is about overlap in time and not a permanent claim on a
        // number.
        assert!(matches!(
            h.delete(1).expect("live"),
            Retirement::StorageFree { backing } if backing == b
        ));
        h.create(3, b, 100).expect("the storage was handed back");
    }

    /// The refusal outlives the delete for as long as the storage does: a heap
    /// retiring behind its allocations still holds its backing, and a new heap
    /// over it would inherit the retirement it cannot see.
    #[test]
    fn storage_outliving_its_heap_number_is_still_in_use() {
        let mut h = Heaps::new();
        let b = BackingId(9);
        h.create(1, b, 100).expect("live");
        let p = h.place(1, res(10), 0, 10).expect("placed");
        assert!(matches!(
            h.delete(1).expect("live"),
            Retirement::Held { .. }
        ));
        assert_eq!(
            h.create(2, b, 100),
            Err(Refusal::StorageInUse { backing: b }),
            "the number is free and the storage is not"
        );
        assert!(matches!(
            h.remove(p, res(10)).expect("placed"),
            Retirement::StorageFree { backing } if backing == b
        ));
        h.create(2, b, 100)
            .expect("the last allocation handed it back");
    }

    #[test]
    fn every_refusal_has_its_own_slug() {
        let all = [
            Refusal::NoSuchHeap { heap: 1 },
            Refusal::HeapExists { heap: 1 },
            Refusal::OutOfHeap {
                heap: 1,
                offset: 0,
                length: 0,
                heap_length: 0,
            },
            Refusal::OutOfPlacement {
                offset: 0,
                length: 0,
                placement_length: 0,
            },
            Refusal::AlreadyPlaced {
                resource: res(1),
                backing: BackingId(1),
            },
            Refusal::NotPlaced { resource: res(1) },
            Refusal::StaleStorage {
                backing: BackingId(1),
            },
            Refusal::StorageInUse {
                backing: BackingId(1),
            },
        ];
        let mut slugs: Vec<&str> = all.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count);
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

    /// Deliberately tiny pools, so numbers and storage are reused constantly
    /// and the same resource is placed, removed and placed again. Everything
    /// interesting here is about a name outliving the thing it named.
    const NUMBERS: u64 = 4;
    const BACKINGS: u64 = 3;
    const RESOURCES: u64 = 6;
    const HEAP_LEN: u64 = 64;

    /// Whole heap histories, driven, against a shadow that tracks only who
    /// holds what.
    ///
    /// # The property the hand-written cases cannot state
    ///
    /// **Storage is handed back exactly once, and never while a placement still
    /// names it.** Both halves are claims about a *sequence*: a double free is
    /// two `StorageFree` reports for one backing separated by any amount of
    /// history, and an early free is a report issued while some placement handed
    /// out earlier is still outstanding. Every case above drives one heap
    /// through one shape and can see neither.
    ///
    /// This is the sweep that would have caught two heaps sharing one backing —
    /// the defect `two_heaps_cannot_be_created_over_one_piece_of_storage` now
    /// refuses at creation — and it is here so the next one of that family is
    /// caught by a test rather than by reading the code.
    ///
    /// # Stale placements are part of the history, not an error case
    ///
    /// Removals are drawn from every placement ever handed out, including ones
    /// whose heap was deleted and whose number has since been reused. That is
    /// the ordinary shape of this module — a placement outlives its heap's
    /// number by construction — so a sweep that only removed live placements
    /// would be testing the easy half.
    #[test]
    fn storage_is_handed_back_exactly_once_and_never_under_a_live_placement() {
        let mut frees = 0usize;
        let mut refused_number = 0usize;
        let mut refused_storage = 0usize;
        let mut stale_removals = 0usize;
        let mut deletes_that_waited = 0usize;
        let mut removals_of_a_retiring_heap = 0usize;
        let mut removals_of_a_superseded_window = 0usize;

        for seed in 0..384u64 {
            let mut rng = Rng::new(seed);
            let mut heaps = Heaps::new();
            // Shadow: which storage each live number is over, and every
            // placement still outstanding against a piece of storage.
            let mut number_storage: HashMap<u64, BackingId> = HashMap::new();
            let mut outstanding: HashMap<BackingId, HashSet<ResourceId>> = HashMap::new();
            let mut handed_out: Vec<(HeapPlacement, ResourceId)> = Vec::new();
            // Where each placed resource is, across every heap and every
            // retirement: the law `AlreadyPlaced` states, stated globally.
            let mut placed_where: HashMap<ResourceId, BackingId> = HashMap::new();
            let mut live_placement: HashMap<ResourceId, HeapPlacement> = HashMap::new();

            for _ in 0..64 {
                match rng.below(10) {
                    0..=2 => {
                        let number = rng.below(NUMBERS);
                        let backing = BackingId(rng.below(BACKINGS));
                        let result = heaps.create(number, backing, HEAP_LEN);
                        match result {
                            Ok(()) => {
                                assert!(
                                    !number_storage.contains_key(&number),
                                    "seed {seed}: number {number} was already live"
                                );
                                assert!(
                                    !number_storage.values().any(|b| *b == backing)
                                        && !outstanding.contains_key(&backing),
                                    "seed {seed}: {backing:?} was already held"
                                );
                                number_storage.insert(number, backing);
                            }
                            Err(Refusal::HeapExists { .. }) => {
                                refused_number += 1;
                                assert!(number_storage.contains_key(&number));
                            }
                            Err(Refusal::StorageInUse { .. }) => {
                                refused_storage += 1;
                                assert!(
                                    number_storage.values().any(|b| *b == backing)
                                        || outstanding.contains_key(&backing),
                                    "seed {seed}: refused storage nothing holds"
                                );
                            }
                            Err(other) => panic!("seed {seed}: create refused as {other:?}"),
                        }
                    }
                    3..=5 => {
                        let number = rng.below(NUMBERS);
                        let resource = res(rng.below(RESOURCES) as u32);
                        let offset = rng.below(HEAP_LEN);
                        let length = rng.below(HEAP_LEN + 8 - offset);
                        let membership_before = heaps.membership(number).ok();
                        match heaps.place(number, resource, offset, length) {
                            Ok(p) => {
                                // A placement is a change of membership, and a
                                // command records which set it was written
                                // against — so a place that did not advance it
                                // would let a later declaration claim it was
                                // written after this resource arrived.
                                assert!(
                                    heaps
                                        .membership(number)
                                        .expect("live")
                                        .membership_generation
                                        > membership_before.expect("live").membership_generation,
                                    "seed {seed}: a placement did not advance membership"
                                );
                                let backing = number_storage[&number];
                                assert_eq!(p.backing, backing, "seed {seed}");
                                assert_eq!(p.region.offset, offset);
                                assert_eq!(p.region.length, length);
                                assert!(
                                    offset + length <= HEAP_LEN,
                                    "seed {seed}: a placement left its heap"
                                );
                                assert!(
                                    outstanding.entry(backing).or_default().insert(resource),
                                    "seed {seed}: placed twice"
                                );
                                assert!(
                                    placed_where.insert(resource, backing).is_none(),
                                    "seed {seed}: {resource:?} was placed in a second heap"
                                );
                                live_placement.insert(resource, p);
                                handed_out.push((p, resource));
                            }
                            Err(Refusal::NoSuchHeap { .. }) => {
                                assert!(!number_storage.contains_key(&number))
                            }
                            Err(Refusal::OutOfHeap { .. }) => {
                                assert!(offset + length > HEAP_LEN, "seed {seed}")
                            }
                            Err(Refusal::AlreadyPlaced {
                                resource: named,
                                backing,
                            }) => {
                                assert_eq!(named, resource, "seed {seed}");
                                assert_eq!(
                                    placed_where.get(&resource),
                                    Some(&backing),
                                    "seed {seed}: refused a resource nothing holds, or named \
                                     the wrong storage"
                                );
                                assert!(outstanding[&backing].contains(&resource));
                            }
                            Err(other) => panic!("seed {seed}: place refused as {other:?}"),
                        }
                    }
                    6..=8 if !handed_out.is_empty() => {
                        let which = rng.below(handed_out.len() as u64) as usize;
                        let (placement, resource) = handed_out[which];
                        // Which placement is live is the shadow's own
                        // bookkeeping: the last one `place` handed out for this
                        // resource and did not take back. Deriving it instead
                        // from where the resource can be *found* — its backing
                        // is still named, or still retiring — is the module's
                        // lookup rule rather than a property, and a shadow that
                        // restates the lookup agrees with the module about
                        // every placement of one resource in one heap at once.
                        // The sweep hands those out: a resource removed and
                        // placed again differs only in its window.
                        let named = number_storage.get(&placement.heap) == Some(&placement.backing);
                        let held = live_placement.get(&resource) == Some(&placement);
                        let superseded = !held
                            && live_placement.get(&resource).is_some_and(|live| {
                                live.heap == placement.heap && live.backing == placement.backing
                            });
                        let retiring_here =
                            held && !number_storage.values().any(|b| *b == placement.backing);
                        if !held {
                            stale_removals += 1;
                        }
                        if superseded {
                            removals_of_a_superseded_window += 1;
                        }
                        if retiring_here {
                            removals_of_a_retiring_heap += 1;
                        }
                        let membership_before = heaps.membership(placement.heap).ok();
                        match heaps.remove(placement, resource) {
                            Ok(Retirement::StorageFree { backing }) => {
                                assert_eq!(backing, placement.backing, "seed {seed}");
                                let set = outstanding
                                    .get_mut(&backing)
                                    .expect("a free needs something to have been held");
                                set.remove(&resource);
                                assert!(
                                    set.is_empty(),
                                    "seed {seed}: {backing:?} freed while {} placements still \
                                     name it",
                                    set.len()
                                );
                                assert!(
                                    !number_storage.values().any(|b| *b == backing),
                                    "seed {seed}: {backing:?} freed while a number still names it"
                                );
                                // A second free of this storage cannot pass:
                                // the shadow now holds nothing under it, so the
                                // `expect` above fires unless a create hands it
                                // out again first.
                                outstanding.remove(&backing);
                                placed_where.remove(&resource);
                                live_placement.remove(&resource);
                                frees += 1;
                            }
                            Ok(Retirement::Held {
                                allocations,
                                named: n,
                            }) => {
                                assert!(held, "seed {seed}: a stale placement was accepted");
                                assert_eq!(n, named, "seed {seed}: wrong `named`");
                                if n {
                                    assert!(
                                        heaps
                                            .membership(placement.heap)
                                            .expect("named")
                                            .membership_generation
                                            > membership_before
                                                .expect("named")
                                                .membership_generation,
                                        "seed {seed}: a removal did not advance membership"
                                    );
                                }
                                let set = outstanding.get_mut(&placement.backing).expect("held");
                                set.remove(&resource);
                                placed_where.remove(&resource);
                                live_placement.remove(&resource);
                                if n {
                                    // A named heap reports its own allocations,
                                    // which are the ones in *this* storage.
                                    assert_eq!(allocations, set.len(), "seed {seed}");
                                } else {
                                    assert_eq!(allocations, set.len(), "seed {seed}");
                                    assert!(!set.is_empty(), "seed {seed}: held nothing");
                                }
                            }
                            Err(
                                e @ (Refusal::NotPlaced { .. }
                                | Refusal::StaleStorage { .. }
                                | Refusal::StalePlacement { .. }),
                            ) => {
                                assert!(
                                    !held,
                                    "seed {seed}: a live placement was refused as {e:?}; \
                                     placement={placement:?} resource={resource:?} \
                                     numbers={number_storage:?} outstanding={outstanding:?}"
                                );
                            }
                            Err(other) => panic!("seed {seed}: remove refused as {other:?}"),
                        }
                    }
                    _ => {
                        let number = rng.below(NUMBERS);
                        match heaps.delete(number) {
                            Ok(Retirement::StorageFree { backing }) => {
                                assert_eq!(number_storage.remove(&number), Some(backing));
                                assert!(
                                    outstanding.get(&backing).is_none_or(|s| s.is_empty()),
                                    "seed {seed}: an empty-heap delete freed storage a \
                                     placement names"
                                );
                                outstanding.remove(&backing);
                                frees += 1;
                            }
                            Ok(Retirement::Held { allocations, named }) => {
                                let backing = number_storage
                                    .remove(&number)
                                    .expect("a delete that held needs a live heap");
                                assert!(!named, "seed {seed}: a delete leaves nothing named");
                                assert_eq!(allocations, outstanding[&backing].len(), "seed {seed}");
                                assert!(allocations > 0);
                                deletes_that_waited += 1;
                            }
                            Err(Refusal::NoSuchHeap { .. }) => {
                                assert!(!number_storage.contains_key(&number));
                            }
                            Err(other) => panic!("seed {seed}: delete refused as {other:?}"),
                        }
                    }
                }

                // The observers agree with the shadow after every step.
                assert_eq!(
                    heaps.live_heaps(),
                    {
                        let mut v: Vec<u64> = number_storage.keys().copied().collect();
                        v.sort_unstable();
                        v
                    },
                    "seed {seed}: live_heaps"
                );
                for n in 0..NUMBERS {
                    let expected = number_storage
                        .get(&n)
                        .map_or(0, |b| outstanding.get(b).map_or(0, |s| s.len()));
                    assert_eq!(
                        heaps.allocations(n),
                        expected,
                        "seed {seed}: allocations({n})"
                    );
                }
                for b in 0..BACKINGS {
                    let backing = BackingId(b);
                    let expected = number_storage.values().any(|v| *v == backing)
                        || outstanding.get(&backing).is_some_and(|s| !s.is_empty());
                    assert_eq!(
                        heaps.holds_storage(backing),
                        expected,
                        "seed {seed}: holds_storage({backing:?})"
                    );
                }
                assert_eq!(
                    heaps.retiring_storage(),
                    outstanding
                        .iter()
                        .filter(|(b, s)| {
                            !s.is_empty() && !number_storage.values().any(|v| *v == **b)
                        })
                        .count(),
                    "seed {seed}: retiring_storage"
                );
                // The index behind `AlreadyPlaced` is not a second source of
                // truth: it says exactly what the heaps and the retirements
                // say, at every step and not only at the end.
                assert_eq!(heaps.placed, placed_where, "seed {seed}: the placed index");
                let derived: HashMap<ResourceId, BackingId> = heaps
                    .heaps
                    .values()
                    .flat_map(|h| h.placements.keys().map(|r| (*r, h.backing)))
                    .chain(
                        heaps
                            .retiring
                            .iter()
                            .flat_map(|(b, r)| r.allocations.keys().map(|res| (*res, *b))),
                    )
                    .collect();
                assert_eq!(heaps.placed, derived, "seed {seed}: the index drifted");
            }
        }

        // Non-vacuity: every shape an assertion above depends on reaching.
        assert!(frees > 700, "storage handed back: {frees}");
        assert!(
            deletes_that_waited > 150,
            "deletes that could not free yet: {deletes_that_waited}"
        );
        assert!(
            removals_of_a_retiring_heap > 120,
            "removals from storage whose number is gone: {removals_of_a_retiring_heap}"
        );
        assert!(
            removals_of_a_superseded_window > 40,
            "removals naming a window a later placement of the same resource in the same heap \
             has taken over: {removals_of_a_superseded_window}"
        );
        assert!(
            stale_removals > 1_500,
            "removals of a placement nothing holds: {stale_removals}"
        );
        assert!(
            refused_number > 1_500,
            "creates over a live number: {refused_number}"
        );
        assert!(
            refused_storage > 1_000,
            "creates over held storage: {refused_storage}"
        );
    }
}
