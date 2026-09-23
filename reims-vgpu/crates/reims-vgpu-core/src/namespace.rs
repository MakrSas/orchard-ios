//! The object namespace: what a guest name resolves to, and what keeps the
//! thing it resolved to alive.
//!
//! # A name is resolved once, and then never again
//!
//! A guest names an object by its slot in an object list. Slots are reused, so
//! a slot number alone is not an identity: work that carried a slot number and
//! looked it up again later would find whatever now occupies it. Resolution
//! happens once, at admission, and produces a [`ResourceId`] carrying the
//! slot's generation; everything downstream carries that, and a slot reused
//! under it resolves to a different generation and refuses.
//!
//! # Deletion stops resolution; it does not stop work
//!
//! This is the invariant the whole module exists for. A guest deletes an
//! object while a submission that uses it is still executing — routinely, and
//! legally. Deleting must therefore do exactly two things: make *new*
//! resolution fail, and leave everything already accepted alone. Teardown
//! happens when the last accepted use retires, not when the guest asks.
//!
//! An implementation that freed on the delete has a use-after-free that only
//! appears under load. One that refused the delete while uses are outstanding
//! turns an ordinary guest sequence into a stall. So the delete is accepted,
//! the slot stops resolving, and [`Teardown`] says whether the object may be
//! torn down now or is owed to a later release.
//!
//! # Replacing the memory under a name is the same problem
//!
//! `replace-physical` swaps the backing a live object reads from. Work already
//! accepted against the old backing must still read the old backing — it was
//! planned against those bytes and its hazard edges were compiled against that
//! [`BackingId`]. So the replacement takes effect for resolution immediately
//! and the old backing is handed back with the same [`Teardown`] answer the
//! delete gives.
//!
//! # A name may name no storage, and most of them do not
//!
//! Every object in a guest's object list resolves through here — a bind names a
//! sampler by its slot exactly as it names a texture by one — and a sampler,
//! a function, a pipeline state, a depth-stencil state and a view over another
//! object's bytes all own no memory of their own. Their names still have to
//! resolve, still carry a generation, and still stop resolving when the guest
//! deletes them.
//!
//! So a slot's backing is optional, and the alternative was not available: a
//! [`BackingId`] minted for an object that names no storage would have to come
//! from the object's *name*, which is the one derivation
//! [`crate::access::BackingId`] rules out — it makes every sharing relationship
//! wrong in the direction that drops hazard edges, and it would compile. The
//! device says so in its own vocabulary: `BackingIdRefusal::NamesNoStorage` is
//! the arm a sampler takes, and it is a statement about the object rather than
//! a gap in what the device can read.
//!
//! Nothing else changes shape. A name with no storage takes no claim that holds
//! anything, is held by no other name, and hands nothing back when it goes —
//! [`Teardown::NoStorage`] — so it draws no hazard edge, which is correct,
//! because there is no memory for it to race over.
//!
//! # What this does not own
//!
//! Where the bytes are ([`crate::content`]), when a native object dies
//! ([`crate::retire`]), and whether a task's address space maps an address.
//!
//! A slot once carried a `mapped` flag, on the reading that the map and unmap
//! packets name an object. They do not: both carry a task and a 64-bit
//! interval of that task's GPU virtual address space and no object ref at all,
//! which is what [`crate::lifecycle::map_notice`] now decodes. The flag was a
//! per-object state with no command that could set it, and a name it would have
//! been set for is whichever object shared the low half of an address.

use crate::access::BackingId;
use crate::identity::{ObjectListRef, ResourceId, SlotGeneration};
use std::collections::HashMap;

/// Why a namespace operation did not happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Nothing has ever been declared in this slot.
    NotDeclared { slot: ObjectListRef },
    /// The slot holds a different generation. The name is stale: whatever the
    /// caller meant was deleted and the slot reused.
    StaleGeneration {
        slot: ObjectListRef,
        named: SlotGeneration,
        current: SlotGeneration,
    },
    /// The object is deleted. It may still be executing; it may not be named.
    Deleted { id: ResourceId },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NotDeclared { .. } => "namespace_not_declared",
            Self::StaleGeneration { .. } => "namespace_stale_generation",
            Self::Deleted { .. } => "namespace_deleted",
        }
    }
}

/// What a delete or a physical replacement left behind.
///
/// Distinct variants rather than an `Option` with a comment, because the caller
/// has to do something different in each case and a caller that ignored the
/// difference would either free memory the GPU is reading or leak it.
///
/// The three answers are the three reasons a backing may not be freed yet:
/// nothing is holding it (free it), work is holding it (wait for the release),
/// or *another live name* is holding it (do nothing at all — that name will
/// answer for it when it goes). The third is not a zero-outstanding
/// [`Self::WhenUsesRetire`]: no release is owed to this caller, and one spelled
/// that way would be a handback that never arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a backing that is owed teardown is one nothing else will free"]
pub enum Teardown {
    /// Nothing accepted still reads this backing and no live name resolves to
    /// it. It may be torn down now.
    Now { backing: BackingId },
    /// Accepted work still reads it. [`Namespace::release`] returns it when the
    /// last use retires.
    WhenUsesRetire {
        backing: BackingId,
        outstanding: usize,
    },
    /// A different live name still resolves to this backing. The caller owns no
    /// teardown for it: the storage stays, and the last name to leave will
    /// carry it out.
    HeldByAnotherName { backing: BackingId },
    /// The name owned no storage — a sampler, a function, a pipeline state, a
    /// view over another object's bytes. There is nothing to tear down.
    ///
    /// Its own variant rather than a [`Self::Now`] with a placeholder backing,
    /// because the caller's action is different in kind: `Now` says *free these
    /// bytes*, and a caller handed a placeholder would free whatever that
    /// number happened to name.
    NoStorage,
}

impl Teardown {
    /// The storage this teardown concerns, or `None` when the name owned none.
    #[must_use]
    pub const fn backing(self) -> Option<BackingId> {
        match self {
            Self::Now { backing }
            | Self::WhenUsesRetire { backing, .. }
            | Self::HeldByAnotherName { backing } => Some(backing),
            Self::NoStorage => None,
        }
    }
}

/// What a declaration produced: the new name, and what the slot's previous
/// occupant left behind.
///
/// The teardown is a field rather than a second return value because the two
/// are one event. A caller that took the id and dropped the teardown would have
/// a live name and a backing nothing else will ever free — which is why this
/// type is `#[must_use]` as well as [`Teardown`]: a `Teardown` inside a struct
/// is not one the compiler asks about.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "the displaced occupant's backing is one nothing else will free"]
pub struct Declared {
    pub id: ResourceId,
    /// The slot's previous *live* occupant, if it had one.
    ///
    /// `None` when the slot was empty, and `None` when its occupant had been
    /// deleted: [`Namespace::delete`] already answered for that backing, and
    /// answering twice hands the same storage back twice.
    pub displaced: Option<Displaced>,
}

/// The occupant a declaration wrote over.
///
/// Both halves, because a caller needs both and neither is derivable from the
/// other: the [`Teardown`] says what is owed for the storage, and the id is
/// what everything the caller kept about that object is keyed by. A
/// declaration that reported only the teardown would leave the previous
/// occupant's residency, placement and content authority behind under a name
/// that no longer resolves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "the displaced occupant's backing is one nothing else will free"]
pub struct Displaced {
    /// The name it answered to. Every id equal to this one now resolves
    /// [`Refusal::StaleGeneration`].
    pub id: ResourceId,
    pub teardown: Teardown,
}

/// A resolved name, and the right to take one claim on the memory it named.
///
/// Carries the generation, so it cannot be confused with a later occupant of
/// the same slot, and the backing it resolved to, so work planned against those
/// bytes keeps reading those bytes even after the name is pointed elsewhere.
/// The backing is `None` for a name that owns no storage, which is most of
/// them — see this module's own doc.
///
/// Not `Clone` and not constructible outside this module. Resolving twice is
/// how a caller gets two of these, and two resolutions are two claims — a
/// forged or copied one would be a claim [`Namespace::release`] pays off
/// without [`Namespace::acquire`] ever having taken it.
#[derive(Debug, PartialEq, Eq)]
pub struct Lease {
    id: ResourceId,
    backing: Option<BackingId>,
}

impl Lease {
    /// Which object, at which generation, this resolved to.
    #[must_use]
    pub const fn id(&self) -> ResourceId {
        self.id
    }

    /// The memory it resolved to, which stays the same afterwards even if the
    /// name is pointed elsewhere. `None` when the object owns no memory.
    #[must_use]
    pub const fn backing(&self) -> Option<BackingId> {
        self.backing
    }
}

/// One accepted use of a backing, outstanding until the work retires.
///
/// The thing [`Namespace::release`] pays off, and the reason it can be paid off
/// exactly once. Not `Clone`, not `Copy`, no public constructor, and consumed
/// by `release` — because `release` is what decides a backing has no uses left
/// and hands it to the caller to free. A second release of one claim retires a
/// use that never existed, and the backing goes back to the caller while work
/// that took the *other* claim is still reading it. That is a use-after-free
/// with no failing call in it: every step returns what it always returns.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a claim that is never released keeps its backing alive forever"]
pub struct Claim {
    id: ResourceId,
    backing: Option<BackingId>,
}

impl Claim {
    #[must_use]
    pub const fn id(&self) -> ResourceId {
        self.id
    }

    /// The memory this use holds alive, or `None` when the name owns none —
    /// in which case the claim keeps nothing alive and releasing it frees
    /// nothing, which is exactly right.
    #[must_use]
    pub const fn backing(&self) -> Option<BackingId> {
        self.backing
    }
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: SlotGeneration,
    /// The storage this name answers for, or `None` when the object owns none.
    backing: Option<BackingId>,
    deleted: bool,
    /// Accepted uses of the *current* backing that have not retired.
    outstanding: usize,
}

/// One session generation's object namespace.
#[derive(Debug, Default)]
pub struct Namespace {
    slots: HashMap<ObjectListRef, Slot>,
    /// Backings detached from a slot by a delete or a replacement, still held
    /// by accepted work, counted *per backing* rather than per detachment.
    ///
    /// Several names may legitimately resolve to one backing — every resource
    /// placed in a heap is declared with the heap's — and a slot may be pointed
    /// back at a backing it detached earlier. Claims kept per detachment would
    /// be several counters over one piece of storage, each reaching zero on its
    /// own and each handing the same storage back.
    retiring: HashMap<BackingId, usize>,
    /// Live names per backing, so "does another name still answer for this
    /// storage" is a lookup rather than a scan of every slot.
    ///
    /// Derived from `slots` and never read from anywhere else, so the two
    /// cannot drift into two answers: an entry appears when a declaration
    /// points a slot at a backing and goes when the slot is deleted or
    /// repointed, which are the only three ways the set can change. A backing
    /// with no live name holds no entry at all rather than an entry of zero —
    /// the count is the membership, and one spelled the other way would make
    /// "present" and "held" different questions.
    live_by_backing: HashMap<BackingId, usize>,
    resolved: usize,
    refused: usize,
}

impl Namespace {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare an object into a slot.
    ///
    /// A slot's generation advances on every declaration, including the first,
    /// so no live identity ever carries the default generation and a zeroed
    /// structure cannot read as a valid name.
    ///
    /// # A live slot may be redeclared, because on this interface it is
    ///
    /// This used to refuse a slot that still held an undeleted object, on the
    /// reading that a guest wanting a slot back deletes first. **That is not
    /// the contract.** There is no creation command on this wire at all: an
    /// object comes into existence when the guest writes a record into its own
    /// object-list page, and it is replaced when the guest writes over that
    /// record — in its own memory, with no packet, and with nothing that
    /// obliges it to delete first. A model that refuses the overwrite cannot
    /// describe what the guest actually does, and a device built on it would
    /// keep resolving the previous occupant: a reference that named a small
    /// texture early in a boot and a full-screen video plane later would bind
    /// the small one forever, which is a wrong surface rather than a missing
    /// one.
    ///
    /// So the overwrite is lawful, and it is exactly the event the generation
    /// exists for. The name moves to a new generation, so every [`ResourceId`]
    /// naming the previous occupant resolves to
    /// [`Refusal::StaleGeneration`] instead of to the new object's bytes. Work
    /// already accepted keeps the backing it was planned against, because a
    /// [`Claim`] holds a backing and not a slot. And the displaced backing is
    /// detached exactly as [`Self::delete`] detaches one — which is what stops
    /// the silent replacement this refusal was guarding against, without
    /// refusing anything.
    ///
    /// It cannot fail, which is why it returns no `Result`: every slot is
    /// declarable, and the only question a declaration answers is what the
    /// previous occupant left behind.
    pub fn declare(&mut self, slot: ObjectListRef, backing: Option<BackingId>) -> Declared {
        let existing = self.slots.get(&slot).copied();
        let generation =
            existing.map_or_else(|| SlotGeneration::default().next(), |e| e.generation.next());
        // Only a *live* occupant is displaced. A deleted one already handed its
        // backing over.
        let displaced = existing.filter(|e| !e.deleted);
        if let Some(old) = displaced {
            self.leave(old.backing);
        }
        self.slots.insert(
            slot,
            Slot {
                generation,
                backing,
                deleted: false,
                outstanding: 0,
            },
        );
        self.enter(backing);
        // Detached *after* the new occupant has entered, as
        // [`Self::replace_physical`] does and for the same reason: a
        // redeclaration that points the slot back at the backing it already
        // held would otherwise be told to free the storage its own new
        // occupant is using.
        Declared {
            id: ResourceId { slot, generation },
            displaced: displaced.map(|old| Displaced {
                id: ResourceId {
                    slot,
                    generation: old.generation,
                },
                teardown: self.detach(old.backing, old.outstanding),
            }),
        }
    }

    /// Resolve a guest name.
    ///
    /// # Errors
    ///
    /// The one check that refused: never declared, deleted, or a different
    /// generation than the caller named.
    pub fn resolve(&mut self, name: ResourceId) -> Result<Lease, Refusal> {
        let refusal = match self.slots.get(&name.slot) {
            None => Refusal::NotDeclared { slot: name.slot },
            Some(slot) if slot.generation != name.generation => Refusal::StaleGeneration {
                slot: name.slot,
                named: name.generation,
                current: slot.generation,
            },
            Some(slot) if slot.deleted => Refusal::Deleted { id: name },
            Some(slot) => {
                self.resolved += 1;
                return Ok(Lease {
                    id: name,
                    backing: slot.backing,
                });
            }
        };
        self.refused += 1;
        Err(refusal)
    }

    /// Resolve by slot alone, for a guest name that carries no generation.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve`], minus the generation check there is no input for.
    pub fn resolve_slot(&mut self, slot: ObjectListRef) -> Result<Lease, Refusal> {
        let Some(generation) = self.slots.get(&slot).map(|s| s.generation) else {
            // Counted here, because [`Self::resolve`] never sees it. A name
            // this door refused is still a name refused, and a census that
            // dropped it would report a clean run over a slot nothing declared.
            self.refused += 1;
            return Err(Refusal::NotDeclared { slot });
        };
        self.resolve(ResourceId { slot, generation })
    }

    /// Record that accepted work holds this lease, and hand back the claim.
    ///
    /// Keeps the backing alive past a delete. A caller that resolved and then
    /// did not acquire has a name it may read and no claim on the memory, which
    /// is the shape of a use-after-free — so the lease is consumed here and the
    /// [`Claim`] is the only thing [`Self::release`] takes. A release that
    /// never acquired, and a second release of one acquisition, are both
    /// unrepresentable rather than merely discouraged.
    pub fn acquire(&mut self, lease: Lease) -> Claim {
        let claim = Claim {
            id: lease.id,
            backing: lease.backing,
        };
        if let Some(slot) = self.slots.get_mut(&lease.id.slot) {
            if slot.generation == lease.id.generation
                && !slot.deleted
                && slot.backing == lease.backing
            {
                slot.outstanding += 1;
                return claim;
            }
        }
        // The name moved between the resolve and the acquire — deleted,
        // repointed, or redeclared. The claim is still a claim, so it is
        // counted against the *storage*, which is what it was taken on and what
        // must not be freed under it.
        //
        // **Entered rather than only incremented.** When a detachment left uses
        // outstanding there is an entry to add to; when it left none there is
        // not, because [`Self::detach`] removes an empty one — and the arm that
        // only incremented dropped the claim on the floor in exactly that case.
        // The backing then read as held by nothing, [`Self::holds`] said so, and
        // [`Self::release`] returned `None` for a claim that was the last thing
        // reading the storage. That is a claim taken and never paid off, which
        // the rest of this module spends [`Claim`]'s whole type design making
        // unspellable.
        if let Some(backing) = lease.backing {
            *self.retiring.entry(backing).or_insert(0) += 1;
        }
        claim
    }

    /// Record that work holding this claim has retired.
    ///
    /// Returns the backing if this was the last use of one whose name has
    /// already moved on — the deferred half of [`Teardown::WhenUsesRetire`].
    pub fn release(&mut self, claim: Claim) -> Option<BackingId> {
        if let Some(slot) = self.slots.get_mut(&claim.id.slot) {
            if slot.generation == claim.id.generation
                && !slot.deleted
                && slot.backing == claim.backing
                && slot.outstanding > 0
            {
                slot.outstanding -= 1;
                return None;
            }
        }
        // A claim on a name that owns no storage holds nothing alive, so there
        // is nothing here to pay off and nothing to hand back. It still had to
        // be a `Claim`: the caller cannot know which kind it holds without
        // asking, and a resolve that handed back no claim for some names would
        // make "acquired" and "acquired nothing" the same shape.
        let backing = claim.backing?;
        let claims = self.retiring.get_mut(&backing)?;
        *claims -= 1;
        if *claims > 0 {
            return None;
        }
        self.retiring.remove(&backing);
        // Nothing is owed any more, but the storage is only the caller's to
        // free if no name still resolves to it.
        (!self.named_by_a_live_slot(backing)).then_some(backing)
    }

    /// Delete an object: stop it resolving, and leave accepted work alone.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve`]. Deleting twice is a refusal rather than a silent
    /// success, because the second delete's caller believes it owns a teardown
    /// the first one already answered for.
    pub fn delete(&mut self, name: ResourceId) -> Result<Teardown, Refusal> {
        let lease = self.resolve(name)?;
        let slot = self.slots.get_mut(&name.slot).expect("just resolved");
        slot.deleted = true;
        let outstanding = slot.outstanding;
        slot.outstanding = 0;
        self.leave(lease.backing);
        Ok(self.detach(lease.backing, outstanding))
    }

    /// Point a live name at different memory.
    ///
    /// Resolution moves at once; work already accepted against the old backing
    /// keeps reading the old backing, because it was planned against those
    /// bytes and its hazard edges were compiled against that identity.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve`].
    pub fn replace_physical(
        &mut self,
        name: ResourceId,
        backing: BackingId,
    ) -> Result<Teardown, Refusal> {
        let lease = self.resolve(name)?;
        let slot = self.slots.get_mut(&name.slot).expect("just resolved");
        let outstanding = slot.outstanding;
        slot.backing = Some(backing);
        slot.outstanding = 0;
        self.leave(lease.backing);
        self.enter(Some(backing));
        Ok(self.detach(lease.backing, outstanding))
    }

    /// Detach a backing from its slot, keeping it alive if anything still holds
    /// it — accepted work, or another live name.
    ///
    /// Called after the slot has already been deleted or repointed, so the
    /// departing name is not among the live ones.
    fn detach(&mut self, backing: Option<BackingId>, outstanding: usize) -> Teardown {
        // A name that owned no storage leaves nothing behind. Its outstanding
        // uses are not carried anywhere, and they hold nothing: `acquire` never
        // counted them against a backing, because there was none to count them
        // against.
        let Some(backing) = backing else {
            return Teardown::NoStorage;
        };
        let claims = self.retiring.entry(backing).or_insert(0);
        *claims += outstanding;
        let claims = *claims;
        if claims > 0 {
            return Teardown::WhenUsesRetire {
                backing,
                outstanding: claims,
            };
        }
        self.retiring.remove(&backing);
        if self.named_by_a_live_slot(backing) {
            return Teardown::HeldByAnotherName { backing };
        }
        Teardown::Now { backing }
    }

    /// Record that a live slot now names this backing, if it names one.
    fn enter(&mut self, backing: Option<BackingId>) {
        let Some(backing) = backing else { return };
        *self.live_by_backing.entry(backing).or_insert(0) += 1;
    }

    /// Record that a slot has stopped naming it.
    fn leave(&mut self, backing: Option<BackingId>) {
        let Some(backing) = backing else { return };
        if let Some(count) = self.live_by_backing.get_mut(&backing) {
            *count -= 1;
            if *count == 0 {
                self.live_by_backing.remove(&backing);
            }
        }
    }

    /// Whether any undeleted slot still resolves to this backing.
    fn named_by_a_live_slot(&self, backing: BackingId) -> bool {
        self.live_by_backing.contains_key(&backing)
    }

    /// Whether anything in this namespace still answers for a backing — a live
    /// name, or a detachment whose accepted uses have not retired.
    ///
    /// The question a *session-wide* owner has to ask before dropping anything
    /// keyed by a backing, and it is not the same question a [`Teardown`]
    /// answers. A teardown says what *this* namespace owes; one namespace is
    /// one task, and a backing an IOSurface supplies is reachable from several.
    /// See [`crate::lifecycle::Lifecycle`], which asks every task before it
    /// forgets a backing's content.
    #[must_use]
    pub fn holds(&self, backing: BackingId) -> bool {
        self.named_by_a_live_slot(backing) || self.retiring.contains_key(&backing)
    }

    /// Accepted uses of a name's current backing that have not retired.
    #[must_use]
    pub fn outstanding(&self, name: ResourceId) -> usize {
        self.slots
            .get(&name.slot)
            .filter(|s| s.generation == name.generation)
            .map_or(0, |s| s.outstanding)
    }

    /// Every live name, with its exact generation.
    ///
    /// The complete input a teardown needs, taken without releasing or
    /// republishing anything: a teardown that discovered membership by asking
    /// each value owner separately would have to trust that they agree, and a
    /// teardown that released names as it found them could not refuse partway
    /// through without having already destroyed some of them.
    ///
    /// Sorted by slot, so a caller's teardown order is a property of the
    /// namespace and not of a hash seed.
    #[must_use]
    pub fn live_names(&self) -> Vec<ResourceId> {
        let mut out: Vec<ResourceId> = self
            .slots
            .iter()
            .filter(|(_, s)| !s.deleted)
            .map(|(slot, s)| ResourceId {
                slot: *slot,
                generation: s.generation,
            })
            .collect();
        out.sort_unstable_by_key(|id| id.slot.0);
        out
    }

    /// Distinct backings detached from a slot and still held by accepted work.
    ///
    /// Counted per backing, not per detachment: two names over one piece of
    /// storage owe one handback between them, not two.
    #[must_use]
    pub fn awaiting_teardown(&self) -> usize {
        self.retiring.len()
    }

    /// Names resolved, and names refused.
    #[must_use]
    pub const fn census(&self) -> (usize, usize) {
        (self.resolved, self.refused)
    }
}

/// **The join between a guest's object-list ref and the identity work carries.**
///
/// [`crate::resolve`] and [`crate::lifecycle`] both take a
/// [`RefResolver`](crate::resolve::RefResolver) and this module holds the slots,
/// and nothing implemented the trait — so the only resolvers in the crate were
/// test stubs, and the whole resolution path could be driven by nothing that
/// knew what was actually declared.
///
/// # Identity, and not a claim
///
/// This answers *which* object a ref names. It does not acquire anything, and
/// that is deliberate rather than an omission: a record resolves before its
/// transaction is admitted, and a claim taken at resolution would be a claim on
/// work that may still be refused. [`Namespace::resolve_slot`] is the other
/// door — it returns a [`Lease`], which is the thing [`Namespace::acquire`]
/// takes — and the two questions belong to different moments. The refusal
/// vocabulary already says so: `ResolveRefusal::NeedsStorage` exists precisely
/// because a `RefResolver` answers identity only.
///
/// Deleted slots answer `None`, which is the module's own invariant: deletion
/// stops *new* resolution and leaves accepted work alone.
impl crate::resolve::RefResolver for Namespace {
    fn resource(&self, object_ref: u32) -> Option<ResourceId> {
        let slot = ObjectListRef(object_ref);
        let held = self.slots.get(&slot)?;
        (!held.deleted).then_some(ResourceId {
            slot,
            generation: held.generation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve::RefResolver as _;

    fn slot(n: u32) -> ObjectListRef {
        ObjectListRef(n)
    }

    fn backing(n: u64) -> BackingId {
        BackingId(n)
    }

    /// A declaration into a slot nothing lived in, which is what most of these
    /// tests set up. Asserts the emptiness rather than assuming it, so a test
    /// that grew a live occupant by accident fails here instead of quietly
    /// dropping the teardown it displaced.
    fn fresh(declared: Declared) -> ResourceId {
        assert_eq!(
            declared.displaced, None,
            "the slot already held a live object"
        );
        declared.id
    }

    /// A name that owns no storage resolves, is claimed, is deleted, and hands
    /// nothing back.
    ///
    /// Most of a guest's object list is this: samplers, functions, pipeline
    /// states, depth-stencil states, views over another object's bytes. They are
    /// bound by slot exactly as a texture is, so they must resolve here — and a
    /// namespace that demanded a `BackingId` for them would have forced one to
    /// be derived from the *name*, which is the derivation
    /// [`crate::access::BackingId`] rules out.
    #[test]
    fn a_name_that_owns_no_storage_resolves_and_hands_nothing_back() {
        let mut ns = Namespace::new();
        let id = fresh(ns.declare(slot(1), None));

        let lease = ns.resolve(id).expect("a sampler is a live name");
        assert_eq!(lease.id(), id, "it resolves like any other name");
        assert_eq!(
            lease.backing(),
            None,
            "and it names no memory, which is a statement about the object \
             rather than a failure to find its pages"
        );

        let claim = ns.acquire(lease);
        assert_eq!(claim.backing(), None);
        assert_eq!(
            ns.outstanding(id),
            1,
            "the use is counted: the name is in use whether or not memory is"
        );

        let teardown = ns.delete(id).expect("live");
        assert_eq!(
            teardown,
            Teardown::NoStorage,
            "there is nothing to free, and a `Now` with a placeholder backing \
             would have told the caller to free whatever that number named"
        );
        assert_eq!(teardown.backing(), None);
        assert_eq!(
            ns.delete(id),
            Err(Refusal::Deleted { id }),
            "deletion still stops resolution"
        );
        assert_eq!(
            ns.release(claim),
            None,
            "and the claim pays off nothing, because it was holding nothing"
        );
        assert_eq!(ns.awaiting_teardown(), 0, "nothing is owed to anyone");
    }

    /// A slot that held memory and then holds none, and the reverse.
    ///
    /// The redeclaration path is where the two kinds meet: the displaced
    /// occupant's teardown is the *previous* occupant's question, and reading it
    /// off the new one would free a texture's pages because a sampler landed on
    /// top of them.
    #[test]
    fn a_slot_that_stops_owning_storage_still_hands_back_what_it_held() {
        let mut ns = Namespace::new();
        let held = fresh(ns.declare(slot(1), Some(backing(10))));

        let over = ns.declare(slot(1), None);
        let displaced = over.displaced.expect("the slot held a live object");
        assert_eq!(displaced.id, held);
        assert_eq!(
            displaced.teardown,
            Teardown::Now {
                backing: backing(10)
            },
            "the pages the texture held are freed even though what replaced it \
             owns none"
        );
        assert_eq!(
            ns.resolve(over.id).expect("live").backing(),
            None,
            "and the new occupant owns nothing"
        );
        assert!(
            !ns.holds(backing(10)),
            "nothing answers for those bytes now"
        );

        // ...and back the other way.
        let back = ns.declare(slot(1), Some(backing(11)));
        assert_eq!(
            back.displaced.expect("the sampler was live").teardown,
            Teardown::NoStorage,
            "the sampler leaves nothing behind"
        );
        assert!(ns.holds(backing(11)));
    }

    /// The invariant the module exists for.
    #[test]
    fn deleting_stops_resolution_and_leaves_accepted_work_alone() {
        let mut ns = Namespace::new();
        let id = fresh(ns.declare(slot(1), Some(backing(10))));
        let lease = ns.resolve(id).expect("live");
        let lease = ns.acquire(lease);

        assert_eq!(
            ns.delete(id),
            Ok(Teardown::WhenUsesRetire {
                backing: backing(10),
                outstanding: 1
            }),
            "the delete is accepted; the memory is not freed under the work"
        );
        assert_eq!(
            ns.resolve(id),
            Err(Refusal::Deleted { id }),
            "and nothing new may name it"
        );
        assert_eq!(
            ns.release(lease),
            Some(backing(10)),
            "the last use retiring is what makes it free-able"
        );
        assert_eq!(ns.awaiting_teardown(), 0);
    }

    #[test]
    fn deleting_an_unused_object_frees_it_at_once() {
        let mut ns = Namespace::new();
        let id = fresh(ns.declare(slot(1), Some(backing(10))));
        assert_eq!(
            ns.delete(id),
            Ok(Teardown::Now {
                backing: backing(10)
            })
        );
    }

    /// **A claim is counted whatever the name did in the meantime.**
    ///
    /// `acquire` takes a lease resolved a moment earlier, and between the two
    /// the name may have been deleted, repointed or redeclared. When the
    /// detachment left uses outstanding there was a per-backing counter to add
    /// to and the claim landed in it. When it left none there was not — a
    /// detachment with nothing outstanding removes its entry — and the claim
    /// went nowhere: the storage read as held by nothing, and the release of
    /// the only thing still using it returned nothing to free.
    #[test]
    fn a_claim_taken_after_its_name_went_still_holds_the_storage() {
        // Deleted with nothing outstanding, so the namespace kept no counter.
        let mut ns = Namespace::new();
        let id = fresh(ns.declare(slot(1), Some(backing(10))));
        let lease = ns.resolve(id).expect("live");
        assert_eq!(
            ns.delete(id),
            Ok(Teardown::Now {
                backing: backing(10)
            })
        );
        assert!(!ns.holds(backing(10)), "nothing had taken a claim yet");
        let claim = ns.acquire(lease);
        assert!(
            ns.holds(backing(10)),
            "a claim on the storage is the namespace answering for it"
        );
        assert_eq!(ns.awaiting_teardown(), 1);
        assert_eq!(
            ns.release(claim),
            Some(backing(10)),
            "and the last claim is what hands it back"
        );
        assert!(!ns.holds(backing(10)));
        assert_eq!(ns.awaiting_teardown(), 0);

        // Redeclared into the same storage: the claim is still owed, and the
        // handback is not, because a live name answers for those bytes.
        let mut ns = Namespace::new();
        let first = fresh(ns.declare(slot(1), Some(backing(10))));
        let lease = ns.resolve(first).expect("live");
        assert_eq!(
            ns.delete(first),
            Ok(Teardown::Now {
                backing: backing(10)
            })
        );
        let claim = ns.acquire(lease);
        let _second = fresh(ns.declare(slot(1), Some(backing(10))));
        assert_eq!(
            ns.release(claim),
            None,
            "the new occupant answers for the storage"
        );
        assert!(ns.holds(backing(10)));
    }

    /// Slot reuse produces a new generation, and the old name does not follow
    /// it.
    #[test]
    fn a_reused_slot_refuses_the_name_that_used_to_live_there() {
        let mut ns = Namespace::new();
        let first = fresh(ns.declare(slot(1), Some(backing(10))));
        assert_eq!(
            ns.delete(first).expect("live"),
            Teardown::Now {
                backing: backing(10)
            }
        );
        let second = fresh(ns.declare(slot(1), Some(backing(11))));
        assert_ne!(first.generation, second.generation);
        assert_eq!(
            ns.resolve(first),
            Err(Refusal::StaleGeneration {
                slot: slot(1),
                named: first.generation,
                current: second.generation,
            })
        );
        assert_eq!(ns.resolve(second).expect("live").backing, Some(backing(11)));
    }

    /// **The guest overwrites a live slot, with no packet, and this is what
    /// that is.** There is no creation command on this wire; an object appears
    /// when the guest writes a record into its own object-list page and is
    /// replaced when it writes over that record. A model that refused would
    /// keep resolving the previous occupant forever.
    #[test]
    fn a_live_slot_is_redeclared_and_its_occupant_is_displaced() {
        let mut ns = Namespace::new();
        let first = fresh(ns.declare(slot(1), Some(backing(10))));

        let over = ns.declare(slot(1), Some(backing(11)));
        assert_eq!(
            over.displaced,
            Some(Displaced {
                id: first,
                // Nothing acquired the previous occupant, so its storage is
                // owed to nobody and is the caller's to free.
                teardown: Teardown::Now {
                    backing: backing(10)
                },
            }),
            "the previous occupant's backing is owed to whoever declared over it"
        );

        // The name moved on, which is what stops the silent replacement the
        // old refusal was guarding against.
        assert_ne!(over.id, first);
        assert_eq!(
            ns.resolve(first),
            Err(Refusal::StaleGeneration {
                slot: slot(1),
                named: first.generation,
                current: over.id.generation,
            })
        );
        assert_eq!(
            ns.resolve(over.id).expect("live").backing(),
            Some(backing(11))
        );
    }

    /// Accepted work keeps the bytes it was planned against, exactly as it does
    /// across a delete or a replacement — a claim holds a backing, not a slot.
    #[test]
    fn a_redeclaration_leaves_accepted_work_reading_the_old_backing() {
        let mut ns = Namespace::new();
        let first = fresh(ns.declare(slot(1), Some(backing(10))));
        let held = ns.resolve(first).expect("live");
        let held = ns.acquire(held);
        assert_eq!(held.backing(), Some(backing(10)));

        let over = ns.declare(slot(1), Some(backing(11)));
        assert_eq!(
            over.displaced.map(|d| d.teardown),
            Some(Teardown::WhenUsesRetire {
                backing: backing(10),
                outstanding: 1,
            }),
            "one accepted use is still reading it"
        );
        assert!(ns.holds(backing(10)), "and the namespace says so");
        assert_eq!(
            ns.release(held),
            Some(backing(10)),
            "the last use retires and the old bytes come back"
        );
        assert!(!ns.holds(backing(10)));
    }

    /// A slot redeclared at the memory it already held is not a handback. The
    /// detachment happens after the new occupant has entered, so the storage
    /// its own declaration is using is never reported free.
    #[test]
    fn redeclaring_a_slot_at_the_same_backing_frees_nothing() {
        let mut ns = Namespace::new();
        let first = fresh(ns.declare(slot(1), Some(backing(10))));
        let over = ns.declare(slot(1), Some(backing(10)));
        assert_ne!(over.id, first, "a new declaration is a new generation");
        assert_eq!(
            over.displaced.map(|d| d.teardown),
            Some(Teardown::HeldByAnotherName {
                backing: backing(10)
            }),
            "the name that holds it is this slot's own new occupant"
        );
        assert!(ns.holds(backing(10)));
        assert_eq!(
            ns.resolve(over.id).expect("live").backing(),
            Some(backing(10))
        );
    }

    /// A slot whose occupant was deleted owes nothing on the next declaration:
    /// the delete answered for that backing, and answering twice would hand
    /// the same storage back to two callers.
    #[test]
    fn redeclaring_over_a_deleted_slot_displaces_nothing() {
        let mut ns = Namespace::new();
        let first = fresh(ns.declare(slot(1), Some(backing(10))));
        assert_eq!(
            ns.delete(first),
            Ok(Teardown::Now {
                backing: backing(10)
            })
        );
        let over = ns.declare(slot(1), Some(backing(11)));
        assert_eq!(over.displaced, None);
        assert_ne!(over.id, first);
    }

    /// Work planned against the old bytes keeps reading the old bytes.
    #[test]
    fn replacing_the_memory_moves_the_name_and_not_the_accepted_work() {
        let mut ns = Namespace::new();
        let id = fresh(ns.declare(slot(1), Some(backing(10))));
        let old = ns.resolve(id).expect("live");
        let old = ns.acquire(old);

        assert_eq!(
            ns.replace_physical(id, backing(20)),
            Ok(Teardown::WhenUsesRetire {
                backing: backing(10),
                outstanding: 1
            })
        );
        assert_eq!(
            ns.resolve(id).expect("still live").backing(),
            Some(backing(20)),
            "the name resolves to the new memory at once"
        );
        assert_eq!(
            old.backing(),
            Some(backing(10)),
            "and the lease already taken still names the old"
        );
        assert_eq!(ns.release(old), Some(backing(10)));
    }

    /// Two uses of one backing, released one at a time.
    #[test]
    fn a_detached_backing_survives_until_its_last_use_retires() {
        let mut ns = Namespace::new();
        let id = fresh(ns.declare(slot(1), Some(backing(10))));
        let a = ns.resolve(id).expect("live");
        let a = ns.acquire(a);
        let b = ns.resolve(id).expect("live");
        let b = ns.acquire(b);
        assert_eq!(ns.outstanding(id), 2);
        assert_eq!(
            ns.delete(id).expect("live"),
            Teardown::WhenUsesRetire {
                backing: backing(10),
                outstanding: 2
            }
        );
        assert_eq!(ns.release(a), None, "one use left");
        assert_eq!(ns.release(b), Some(backing(10)));
    }

    /// Deleting twice would hand two callers the same teardown.
    #[test]
    fn deleting_twice_is_refused() {
        let mut ns = Namespace::new();
        let id = fresh(ns.declare(slot(1), Some(backing(10))));
        assert_eq!(
            ns.delete(id).expect("live").backing(),
            Some(backing(10)),
            "the first delete owns the teardown"
        );
        assert_eq!(ns.delete(id), Err(Refusal::Deleted { id }));
    }

    #[test]
    fn a_name_that_was_never_declared_refuses_by_its_own_reason() {
        let mut ns = Namespace::new();
        let name = ResourceId {
            slot: slot(4),
            generation: SlotGeneration(1),
        };
        assert_eq!(
            ns.resolve(name),
            Err(Refusal::NotDeclared { slot: slot(4) })
        );
        assert_eq!(
            ns.resolve_slot(slot(4)),
            Err(Refusal::NotDeclared { slot: slot(4) })
        );
        // Both doors resolve a name and both refused, so the census counts
        // two. `resolve_slot`'s undeclared path returns before `resolve` sees
        // it, and a census that let it through would report a clean run over a
        // slot nothing ever declared. The uncounted door is
        // `RefResolver::resource`, which asks identity and takes nothing —
        // see `resolving_for_identity_is_not_a_lease`.
        assert_eq!(ns.census(), (0, 2), "both doors refused a name");
    }

    /// No live identity carries the default generation, so a zeroed structure
    /// cannot read as a valid name.
    #[test]
    fn the_first_declaration_does_not_use_the_default_generation() {
        let mut ns = Namespace::new();
        let id = fresh(ns.declare(slot(1), Some(backing(10))));
        assert_ne!(id.generation, SlotGeneration::default());
    }

    /// Resolving a ref for identity gives the live occupant, the current
    /// generation, and nothing for a slot the guest deleted.
    ///
    /// The generation half is the whole reason a slot number is not an
    /// identity: a slot reused after a delete answers with a *different*
    /// generation, so work still carrying the old id no longer names it.
    #[test]
    fn a_ref_resolves_to_the_slots_live_occupant_and_its_generation() {
        let mut ns = Namespace::new();
        assert_eq!(ns.resource(1), None, "nothing has been declared");
        let first = fresh(ns.declare(slot(1), Some(backing(10))));
        assert_eq!(ns.resource(1), Some(first));

        assert_eq!(
            ns.delete(first).expect("declared"),
            Teardown::Now {
                backing: backing(10)
            },
            "nothing was acquired, so nothing is owed"
        );
        assert_eq!(
            ns.resource(1),
            None,
            "a delete stops new resolution, which is the module's invariant"
        );

        let second = fresh(ns.declare(slot(1), Some(backing(11))));
        assert_eq!(ns.resource(1), Some(second));
        assert_ne!(second, first, "a reused slot is not the same name");
    }

    /// Resolving for identity takes no claim, and the census says so.
    ///
    /// A record resolves before its transaction is admitted, and a claim taken
    /// at resolution would be a claim held for work that may still be refused.
    /// `resolve_slot` is the door that produces a `Lease`; this one does not,
    /// and a reader has to be able to tell them apart.
    #[test]
    fn resolving_for_identity_is_not_a_lease() {
        let mut ns = Namespace::new();
        let id = fresh(ns.declare(slot(1), Some(backing(10))));
        let before = ns.census();
        assert_eq!(ns.resource(1), Some(id));
        assert_eq!(ns.census(), before, "identity is not a resolution taken");
        assert_eq!(ns.outstanding(id), 0);

        let lease = ns.resolve_slot(slot(1)).expect("live");
        assert_eq!(ns.census(), (before.0 + 1, before.1));
        let claim = ns.acquire(lease);
        assert_eq!(
            ns.outstanding(id),
            1,
            "the lease is where the claim comes from"
        );
        assert_eq!(claim.id(), id);
    }

    /// Several live names over one backing is the ordinary path —
    /// `lifecycle::create_resource` declares every heap-placed resource with
    /// the heap's backing — so a delete of one of them must hand nothing back.
    #[test]
    fn a_backing_two_names_share_is_not_handed_back_when_one_of_them_goes() {
        let mut ns = Namespace::new();
        let b = backing(7);
        let a = fresh(ns.declare(slot(1), Some(b)));
        let c = fresh(ns.declare(slot(2), Some(b)));
        let la = ns.resolve(a).expect("live");
        let la = ns.acquire(la);
        let lc = ns.resolve(c).expect("live");
        let lc = ns.acquire(lc);

        assert_eq!(
            ns.delete(a),
            Ok(Teardown::WhenUsesRetire {
                backing: b,
                outstanding: 1,
            })
        );
        assert_eq!(
            ns.release(la),
            None,
            "storage handed back while another name still resolves to it"
        );
        assert_eq!(ns.resolve(c).expect("still live").backing(), Some(b));
        assert_eq!(ns.outstanding(c), 1);

        // Only the last name out carries it.
        assert_eq!(
            ns.delete(c),
            Ok(Teardown::WhenUsesRetire {
                backing: b,
                outstanding: 1
            })
        );
        assert_eq!(ns.release(lc), Some(b));
    }

    /// A name whose backing nothing else holds, deleted while a *live* name
    /// still resolves to that backing, is owed no teardown at all — and saying
    /// so as a zero-outstanding `WhenUsesRetire` would promise a handback no
    /// release can ever deliver.
    #[test]
    fn a_delete_another_live_name_answers_for_says_so() {
        let mut ns = Namespace::new();
        let b = backing(7);
        let a = fresh(ns.declare(slot(1), Some(b)));
        let c = fresh(ns.declare(slot(2), Some(b)));

        assert_eq!(ns.delete(a), Ok(Teardown::HeldByAnotherName { backing: b }));
        assert_eq!(ns.awaiting_teardown(), 0, "nothing is owed");
        assert_eq!(ns.delete(c), Ok(Teardown::Now { backing: b }));
    }

    /// A slot pointed back at a backing it detached earlier reaches the
    /// namespace twice over one piece of storage. Claims kept per detachment
    /// would be two counters over it, each reaching zero and each handing the
    /// same storage back.
    #[test]
    fn a_backing_a_name_returns_to_is_handed_back_once() {
        let mut ns = Namespace::new();
        let b = backing(7);
        let id = fresh(ns.declare(slot(1), Some(b)));
        let lease = ns.resolve(id).expect("live");
        let lease = ns.acquire(lease);

        assert_eq!(
            ns.replace_physical(id, backing(8)),
            Ok(Teardown::WhenUsesRetire {
                backing: b,
                outstanding: 1,
            })
        );
        // Back to `b`, which is still owed a handback, and away again.
        assert_eq!(
            ns.replace_physical(id, b),
            Ok(Teardown::Now {
                backing: backing(8)
            })
        );
        assert_eq!(
            ns.replace_physical(id, backing(9)),
            Ok(Teardown::WhenUsesRetire {
                backing: b,
                outstanding: 1,
            }),
            "the two visits to this storage owe one handback between them"
        );
        assert_eq!(ns.awaiting_teardown(), 1);
        assert_eq!(ns.release(lease), Some(b));
        assert_eq!(ns.awaiting_teardown(), 0);
    }

    /// A slot redeclared over the same backing is a different occupancy. Its
    /// claims are its own, and a release from the previous occupant must not
    /// pay them off.
    #[test]
    fn a_redeclared_slot_does_not_absorb_the_previous_occupants_release() {
        let mut ns = Namespace::new();
        let b = backing(7);
        let first = fresh(ns.declare(slot(1), Some(b)));
        let lease = ns.resolve(first).expect("live");
        let lease = ns.acquire(lease);
        assert_eq!(
            ns.delete(first),
            Ok(Teardown::WhenUsesRetire {
                backing: b,
                outstanding: 1,
            })
        );

        let second = fresh(ns.declare(slot(1), Some(b)));
        let fresh = ns.resolve(second).expect("live");
        let _fresh = ns.acquire(fresh);
        assert_eq!(ns.outstanding(second), 1);

        assert_eq!(
            ns.release(lease),
            None,
            "the backing is still named by the slot's new occupant"
        );
        assert_eq!(
            ns.outstanding(second),
            1,
            "a stale lease paid off the new occupant's claim"
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

    /// Deliberately tiny pools, so several live names share one backing
    /// constantly and slots are redeclared under leases still in flight.
    /// Everything interesting here is about storage outliving one of its names.
    const SLOTS: u64 = 4;
    const BACKINGS: u64 = 3;

    #[derive(Clone, Copy, Debug)]
    struct ShadowSlot {
        generation: SlotGeneration,
        /// `None` for a name that owns no storage, exactly as the real slot.
        backing: Option<BackingId>,
        deleted: bool,
        /// Claims taken through *this* occupancy of the slot.
        claims: usize,
    }

    /// **The invariant the module exists for, driven over histories.**
    ///
    /// A backing is handed back exactly once, and never while anything still
    /// holds it — accepted work, or another live name. `create_resource`
    /// declares every heap-placed resource with the *heap's* backing, so
    /// several names over one piece of storage is the ordinary path and not a
    /// corner; a slot may also be pointed back at a backing it detached
    /// earlier, which is the same storage reaching the namespace twice.
    ///
    /// The shadow counts claims per *occupancy* — a slot at one generation
    /// pointing at one backing — and moves them to a per-backing detached pool
    /// when that occupancy lets go. That split is the one thing it has to state
    /// the way the module does, because two leases naming the same slot,
    /// generation and backing are genuinely indistinguishable and something has
    /// to say which counter a late release pays. Everything the sweep actually
    /// asserts — exactly-once, never-early, and every observer — is checked
    /// against the shadow's own bookkeeping.
    #[test]
    fn a_backing_is_handed_back_exactly_once_and_never_under_a_live_name() {
        let mut handbacks_now = 0usize;
        let mut handbacks_on_release = 0usize;
        let mut deletes_that_waited = 0usize;
        let mut held_by_another_name = 0usize;
        let mut redeclarations = 0usize;
        let mut refused_stale = 0usize;
        let mut refused_deleted = 0usize;
        let mut refused_undeclared = 0usize;
        let mut replacements = 0usize;
        let mut late_releases = 0usize;
        let mut late_acquires = 0usize;

        for seed in 0..384u64 {
            let mut rng = Rng::new(seed);
            let mut ns = Namespace::new();
            let mut shadow: HashMap<ObjectListRef, ShadowSlot> = HashMap::new();
            // Claims whose occupancy has let go of them: per backing, because
            // several names may have shared it.
            let mut detached: HashMap<BackingId, usize> = HashMap::new();
            let mut names: Vec<ResourceId> = Vec::new();
            let mut leases: Vec<Claim> = Vec::new();
            // Leases resolved and not yet acquired. A lease is uncounted by
            // construction — see `RefResolver` — so holding one across a
            // delete, a replacement or a redeclaration is the shape where the
            // slot the claim would have landed in is gone by the time it is
            // taken.
            let mut pending: Vec<Lease> = Vec::new();
            let mut resolved = 0usize;
            let mut refused = 0usize;

            // A backing nothing names and nothing holds.
            let unowned = |shadow: &HashMap<ObjectListRef, ShadowSlot>,
                           detached: &HashMap<BackingId, usize>,
                           b: BackingId| {
                !shadow.values().any(|s| !s.deleted && s.backing == Some(b))
                    && !shadow
                        .values()
                        .any(|s| !s.deleted && s.claims > 0 && s.backing == Some(b))
                    && detached.get(&b).copied().unwrap_or(0) == 0
            };
            let named_live = |shadow: &HashMap<ObjectListRef, ShadowSlot>, b: Option<BackingId>| {
                shadow.values().any(|s| !s.deleted && s.backing == b)
            };

            for _ in 0..64 {
                match rng.below(13) {
                    // Declare, over a free slot, a deleted one, or a live one
                    // the guest is writing over.
                    0..=2 => {
                        let sl = slot(rng.below(SLOTS) as u32);
                        // One declaration in four owns no storage — a sampler,
                        // a function, a pipeline state. Frequent enough that a
                        // history mixes them with the storage-bearing names
                        // over the same slots, which is the case that matters:
                        // a slot that held memory and now does not, and the
                        // reverse.
                        let b = (rng.below(4) != 0).then(|| backing(rng.below(BACKINGS)));
                        let previous = shadow.get(&sl).copied();
                        let generation = previous.map_or_else(
                            || SlotGeneration::default().next(),
                            |s| s.generation.next(),
                        );
                        // Only a live occupant is displaced. A deleted one
                        // handed its backing over at the delete.
                        let displaced = previous.filter(|s| !s.deleted);
                        let declared = ns.declare(sl, b);
                        assert_eq!(declared.id.slot, sl, "seed {seed}");
                        assert_eq!(declared.id.generation, generation, "seed {seed}");
                        // The new occupant lands before the old backing is
                        // detached, so a slot redeclared at the memory it
                        // already held is never reported free.
                        shadow.insert(
                            sl,
                            ShadowSlot {
                                generation,
                                backing: b,
                                deleted: false,
                                claims: 0,
                            },
                        );
                        let expected = displaced.map(|old| Displaced {
                            id: ResourceId {
                                slot: sl,
                                generation: old.generation,
                            },
                            teardown: detach_expectation(
                                &shadow,
                                &mut detached,
                                old.backing,
                                old.claims,
                            ),
                        });
                        assert_eq!(declared.displaced, expected, "seed {seed}: declare");
                        if let Some(d) = expected {
                            redeclarations += 1;
                            account(
                                d.teardown,
                                &mut handbacks_now,
                                &mut deletes_that_waited,
                                &mut held_by_another_name,
                            );
                            if matches!(d.teardown, Teardown::Now { .. }) {
                                let freed = d.teardown.backing().expect("`Now` names storage");
                                assert!(
                                    unowned(&shadow, &detached, freed),
                                    "seed {seed}: {freed:?} freed under a live holder"
                                );
                            }
                        }
                        names.push(declared.id);
                    }
                    // Resolve a name, possibly a stale one, and acquire it.
                    3..=5 => {
                        let sl = slot(rng.below(SLOTS) as u32);
                        match ns.resolve_slot(sl) {
                            Ok(lease) => {
                                resolved += 1;
                                let sh = shadow.get_mut(&sl).expect("resolved");
                                assert!(!sh.deleted, "seed {seed}: a deleted slot resolved");
                                assert_eq!(lease.backing(), sh.backing, "seed {seed}");
                                assert_eq!(lease.id().generation, sh.generation, "seed {seed}");
                                // Acquired now, or held for a later step so the
                                // name can move underneath it first.
                                if rng.below(3) == 0 {
                                    pending.push(lease);
                                } else {
                                    sh.claims += 1;
                                    leases.push(ns.acquire(lease));
                                }
                            }
                            Err(Refusal::NotDeclared { .. }) => {
                                refused += 1;
                                refused_undeclared += 1;
                                assert!(!shadow.contains_key(&sl), "seed {seed}");
                            }
                            Err(Refusal::Deleted { .. }) => {
                                refused += 1;
                                refused_deleted += 1;
                                assert!(shadow[&sl].deleted, "seed {seed}");
                            }
                            Err(other) => panic!("seed {seed}: resolve refused as {other:?}"),
                        }
                    }
                    // Release a lease, in any order, however late.
                    6..=8 if !leases.is_empty() => {
                        let which = rng.below(leases.len() as u64) as usize;
                        let claim = leases.swap_remove(which);
                        let (id, held) = (claim.id(), claim.backing());
                        let live_here = shadow.get_mut(&id.slot).filter(|s| {
                            s.generation == id.generation
                                && !s.deleted
                                && s.backing == held
                                && s.claims > 0
                        });
                        let expected = if let Some(sh) = live_here {
                            sh.claims -= 1;
                            None
                        } else if let Some(held) = held {
                            late_releases += 1;
                            let pool = detached
                                .get_mut(&held)
                                .expect("a released claim was taken somewhere");
                            *pool -= 1;
                            if *pool > 0 {
                                None
                            } else {
                                detached.remove(&held);
                                (!named_live(&shadow, Some(held))).then_some(held)
                            }
                        } else {
                            // A claim on a name that owned no storage. It held
                            // nothing, so a late release of it hands nothing
                            // back — and it is not a `late_release` either,
                            // because there is no pool it could have been
                            // taken from.
                            None
                        };
                        assert_eq!(ns.release(claim), expected, "seed {seed}: release");
                        if let Some(held) = expected {
                            assert!(
                                unowned(&shadow, &detached, held),
                                "seed {seed}: {held:?} handed back under a live holder",
                            );
                            handbacks_on_release += 1;
                        }
                    }
                    // Take a claim on a lease the name may have moved out
                    // from under. Wherever it lands, it is a claim and the
                    // storage is not free until it is released.
                    11 if !pending.is_empty() => {
                        let lease = pending.swap_remove(rng.below(pending.len() as u64) as usize);
                        let (id, held) = (lease.id(), lease.backing());
                        let still_here = shadow.get_mut(&id.slot).filter(|s| {
                            s.generation == id.generation && !s.deleted && s.backing == held
                        });
                        if let Some(sh) = still_here {
                            sh.claims += 1;
                        } else if let Some(held) = held {
                            *detached.entry(held).or_insert(0) += 1;
                            late_acquires += 1;
                        }
                        leases.push(ns.acquire(lease));
                        // A claim on a name that owns no storage holds nothing,
                        // so there is nothing for `holds` to say — which is the
                        // assertion, not an exemption from it.
                        if let Some(held) = held {
                            assert!(
                                ns.holds(held),
                                "seed {seed}: {held:?} is claimed and the namespace does not say so"
                            );
                        }
                    }
                    // Delete a name, possibly a stale one.
                    9..=10 if !names.is_empty() => {
                        let id = names[rng.below(names.len() as u64) as usize];
                        let expected = expected_resolution(&shadow, id);
                        match ns.delete(id) {
                            Ok(teardown) => {
                                assert!(expected.is_ok(), "seed {seed}: deleted a stale name");
                                resolved += 1;
                                let sh = shadow.get_mut(&id.slot).expect("live");
                                let b = sh.backing;
                                let moved = sh.claims;
                                sh.claims = 0;
                                sh.deleted = true;
                                assert_eq!(
                                    teardown,
                                    detach_expectation(&shadow, &mut detached, b, moved),
                                    "seed {seed}: delete teardown"
                                );
                                account(
                                    teardown,
                                    &mut handbacks_now,
                                    &mut deletes_that_waited,
                                    &mut held_by_another_name,
                                );
                                if matches!(teardown, Teardown::Now { .. }) {
                                    let b = b.expect("`Now` names storage");
                                    assert!(
                                        unowned(&shadow, &detached, b),
                                        "seed {seed}: {b:?} freed under a live holder"
                                    );
                                }
                            }
                            Err(refusal) => {
                                refused += 1;
                                assert_eq!(Err(refusal), expected, "seed {seed}: delete");
                                count_refusal(
                                    refusal,
                                    &mut refused_stale,
                                    &mut refused_deleted,
                                    &mut refused_undeclared,
                                );
                            }
                        }
                    }
                    // Point a name at different memory.
                    _ if !names.is_empty() => {
                        let id = names[rng.below(names.len() as u64) as usize];
                        let to = backing(rng.below(BACKINGS));
                        let expected = expected_resolution(&shadow, id);
                        match ns.replace_physical(id, to) {
                            Ok(teardown) => {
                                assert!(expected.is_ok(), "seed {seed}: replaced a stale name");
                                resolved += 1;
                                replacements += 1;
                                let sh = shadow.get_mut(&id.slot).expect("live");
                                let old = sh.backing;
                                let moved = sh.claims;
                                sh.claims = 0;
                                sh.backing = Some(to);
                                assert_eq!(
                                    teardown,
                                    detach_expectation(&shadow, &mut detached, old, moved),
                                    "seed {seed}: replace teardown"
                                );
                                account(
                                    teardown,
                                    &mut handbacks_now,
                                    &mut deletes_that_waited,
                                    &mut held_by_another_name,
                                );
                                if matches!(teardown, Teardown::Now { .. }) {
                                    let old = old.expect("`Now` names storage");
                                    assert!(
                                        unowned(&shadow, &detached, old),
                                        "seed {seed}: {old:?} freed under a live holder"
                                    );
                                }
                            }
                            Err(refusal) => {
                                refused += 1;
                                assert_eq!(Err(refusal), expected, "seed {seed}: replace");
                                count_refusal(
                                    refusal,
                                    &mut refused_stale,
                                    &mut refused_deleted,
                                    &mut refused_undeclared,
                                );
                            }
                        }
                    }
                    _ => {}
                }

                // Every observer agrees with the shadow after every step.
                assert_eq!(
                    ns.live_names(),
                    {
                        let mut v: Vec<ResourceId> = shadow
                            .iter()
                            .filter(|(_, s)| !s.deleted)
                            .map(|(sl, s)| ResourceId {
                                slot: *sl,
                                generation: s.generation,
                            })
                            .collect();
                        v.sort_unstable_by_key(|id| id.slot.0);
                        v
                    },
                    "seed {seed}: live_names"
                );
                assert_eq!(
                    ns.awaiting_teardown(),
                    detached.len(),
                    "seed {seed}: awaiting_teardown"
                );
                assert_eq!(ns.census(), (resolved, refused), "seed {seed}: census");
                // The per-backing membership index against the scan it
                // replaced. `holds` is the session-wide owner's question, so a
                // count that drifted from `slots` would let a backing another
                // name still resolves to be dropped out from under it.
                for n in 0..BACKINGS {
                    let b = backing(n);
                    assert_eq!(
                        ns.holds(b),
                        named_live(&shadow, Some(b)) || detached.get(&b).copied().unwrap_or(0) > 0,
                        "seed {seed}: holds {b:?}"
                    );
                }
                for id in &names {
                    let expected = shadow
                        .get(&id.slot)
                        .filter(|s| s.generation == id.generation)
                        .map_or(0, |s| s.claims);
                    assert_eq!(ns.outstanding(*id), expected, "seed {seed}: outstanding");
                }
            }
        }

        // Non-vacuity: every shape an assertion above depends on reaching.
        assert!(handbacks_now > 700, "immediate handbacks: {handbacks_now}");
        assert!(
            handbacks_on_release > 200,
            "handbacks owed to a release: {handbacks_on_release}"
        );
        assert!(
            deletes_that_waited > 400,
            "detachments that could not free yet: {deletes_that_waited}"
        );
        assert!(
            held_by_another_name > 200,
            "detachments another live name answers for: {held_by_another_name}"
        );
        assert!(
            late_releases > 400,
            "releases after a detach: {late_releases}"
        );
        assert!(
            late_acquires > 200,
            "claims taken after the name moved: {late_acquires}"
        );
        assert!(replacements > 300, "replacements: {replacements}");
        assert!(
            redeclarations > 200,
            "live slots written over: {redeclarations}"
        );
        assert!(refused_stale > 200, "stale generation: {refused_stale}");
        assert!(refused_deleted > 200, "deleted: {refused_deleted}");
        assert!(
            refused_undeclared > 100,
            "not declared: {refused_undeclared}"
        );
    }

    /// What a resolution of this name must answer, from the shadow alone.
    fn expected_resolution(
        shadow: &HashMap<ObjectListRef, ShadowSlot>,
        id: ResourceId,
    ) -> Result<Option<BackingId>, Refusal> {
        match shadow.get(&id.slot) {
            None => Err(Refusal::NotDeclared { slot: id.slot }),
            Some(s) if s.generation != id.generation => Err(Refusal::StaleGeneration {
                slot: id.slot,
                named: id.generation,
                current: s.generation,
            }),
            Some(s) if s.deleted => Err(Refusal::Deleted { id }),
            Some(s) => Ok(s.backing),
        }
    }

    /// What a detachment of `backing`, carrying `moved` claims away from the
    /// occupancy that is letting go, must answer — and the shadow's own
    /// bookkeeping for it.
    fn detach_expectation(
        shadow: &HashMap<ObjectListRef, ShadowSlot>,
        detached: &mut HashMap<BackingId, usize>,
        backing: Option<BackingId>,
        moved: usize,
    ) -> Teardown {
        // A name that owned nothing leaves nothing behind, and its claims were
        // never pooled because there was no backing to pool them against.
        let Some(backing) = backing else {
            return Teardown::NoStorage;
        };
        let pool = detached.entry(backing).or_insert(0);
        *pool += moved;
        if *pool > 0 {
            return Teardown::WhenUsesRetire {
                backing,
                outstanding: *pool,
            };
        }
        detached.remove(&backing);
        if shadow
            .values()
            .any(|s| !s.deleted && s.backing == Some(backing))
        {
            Teardown::HeldByAnotherName { backing }
        } else {
            Teardown::Now { backing }
        }
    }

    fn account(teardown: Teardown, now: &mut usize, waited: &mut usize, held: &mut usize) {
        match teardown {
            Teardown::Now { .. } => *now += 1,
            Teardown::WhenUsesRetire { .. } => *waited += 1,
            Teardown::HeldByAnotherName { .. } => *held += 1,
            Teardown::NoStorage => {}
        }
    }

    fn count_refusal(
        refusal: Refusal,
        stale: &mut usize,
        deleted: &mut usize,
        undeclared: &mut usize,
    ) {
        match refusal {
            Refusal::StaleGeneration { .. } => *stale += 1,
            Refusal::Deleted { .. } => *deleted += 1,
            Refusal::NotDeclared { .. } => *undeclared += 1,
        }
    }
}
