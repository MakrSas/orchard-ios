//! The names, and the arithmetic that goes with them.
//!
//! # Guest ordinals are parsed once
//!
//! A guest sends 32-bit numbers: task ids, object refs, channel ids, stamp
//! slots, stamp values. Every one of them means something different, and a
//! `u32` parameter list lets any two of them be swapped at a call site without
//! the compiler noticing. Worse, two of them here name *different namespaces
//! that overlap numerically* — the serializer's per-kind object refs and the
//! kernel object-list refs — and equal integers across those spaces are
//! unrelated. That is not a hypothetical: acting on one as the other was
//! measured, and its only possible effect was destroying an unrelated object
//! that happened to share a number.
//!
//! So each one is its own type, and the crossings that used to be free now have
//! to be written out.
//!
//! # Resolution produces a generation
//!
//! A guest name is a slot, and slots are reused. Work that holds a slot number
//! holds a promise the guest can break by deleting and recreating; work that
//! holds a [`ResourceId`] holds a slot *and* the generation it was resolved in,
//! so a reused slot cannot answer for the object that used to be in it. Nothing
//! in the model carries a raw name past resolution.

use core::num::NonZeroU64;

/// One attached vGPU.
///
/// Distinct from a [`SessionGeneration`]: the device outlives a guest reset,
/// and the semantic lifetime does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(pub u32);

/// A semantic lifetime, opened by attach or by guest reset.
///
/// Every accepted lease carries the generation it was accepted in. Closing a
/// generation stops new resolution; it does not invalidate work already
/// accepted, which stays retained until its own terminal point. That
/// distinction is the whole reason this is a separate identity from
/// [`SessionId`] — a reset that invalidated accepted work would be a reset that
/// can drop a submission the host is still executing.
///
/// Non-zero so that "no generation" is representable as `Option` with no niche
/// cost and so that a zeroed structure cannot read as generation 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionGeneration(NonZeroU64);

impl SessionGeneration {
    /// The generation an attach opens.
    pub const FIRST: SessionGeneration = SessionGeneration(NonZeroU64::MIN);

    /// The generation a reset opens after this one.
    ///
    /// Saturating rather than wrapping: a wrapped generation would make a stale
    /// lease from 2^64 resets ago compare equal to a live one, and the failure
    /// would be a use-after-free that looks like a valid name. Saturating at
    /// the ceiling instead makes every later generation compare equal to the
    /// ceiling, which stops the counter rather than corrupting it — and 2^64
    /// guest resets is not a run this device will see.
    #[must_use]
    pub const fn next(self) -> SessionGeneration {
        match NonZeroU64::new(self.0.get().saturating_add(1)) {
            Some(n) => SessionGeneration(n),
            // Unreachable: a saturating add on a non-zero value is non-zero.
            None => SessionGeneration::FIRST,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// The host device incarnation: the lifetime whose objects are invalidated
/// together when the host device is lost or deliberately recreated.
///
/// A **separate** lifetime from [`SessionGeneration`], and the separation is
/// the point. A guest reset closes the semantic lifetime and says nothing about
/// the host device, which may be perfectly healthy and must not be torn down
/// for it. Host device loss ends every handle at once and says nothing about
/// what the guest still names. Every native object lease carries both: the
/// generation decides whether the guest may still name it, the epoch decides
/// whether its handles may still be touched. Collapsing them into one counter
/// makes a reset destroy a working device or a device loss leave dead handles
/// reachable under a live name, and those are the two failures this device
/// cannot have.
///
/// What an epoch *contains* is an executor's business and is not nameable from
/// here. What it *is* — a lifetime that ends all at once — is the contract, and
/// that is what this is.
///
/// Non-zero for the same reason [`SessionGeneration`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceEpoch(NonZeroU64);

impl DeviceEpoch {
    /// The incarnation an attach creates.
    pub const FIRST: DeviceEpoch = DeviceEpoch(NonZeroU64::MIN);

    /// The incarnation created after this one is lost.
    ///
    /// Saturating for the same reason [`SessionGeneration::next`] is: a wrapped
    /// epoch would make a lease from a dead device compare equal to a live one,
    /// and that failure is a use-after-free wearing a valid name.
    #[must_use]
    pub const fn next(self) -> DeviceEpoch {
        match NonZeroU64::new(self.0.get().saturating_add(1)) {
            Some(n) => DeviceEpoch(n),
            // Unreachable: a saturating add on a non-zero value is non-zero.
            None => DeviceEpoch::FIRST,
        }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// A point on a host completion timeline.
///
/// Not a [`StampValue`]. The guest's completion word is a 32-bit value that
/// wraps and is compared in a wrapping order; a host timeline is a 64-bit
/// monotone counter that does not wrap in any run this device will see. They
/// are different contracts with different comparison rules, so a single type
/// for both would mean one of the two comparisons is silently wrong — and the
/// wrong one is whichever the reader was not thinking about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimelinePoint(pub u64);

impl TimelinePoint {
    /// Whether a timeline that has reached `self` has passed `other`.
    #[must_use]
    pub const fn reached(self, other: TimelinePoint) -> bool {
        self.0 >= other.0
    }

    #[must_use]
    pub const fn next(self) -> TimelinePoint {
        TimelinePoint(self.0.wrapping_add(1))
    }
}

/// What a present names as the thing to show.
///
/// **A third namespace, numerically overlapping the other two and unrelated to
/// them.** The device's own state keeps this apart from object-list refs in as
/// many words — "surface_id namespace only, never texture_ref (object list ids
/// collide)" — because a host render cache keyed by one and fed by the other
/// serves a frame from whatever texture happened to share a number. That is the
/// hazard this module exists to make a type error, so a present's target is its
/// own type rather than the `u32` it arrives as.
///
/// Not `NonZero`. Zero is a value the guest sends and it means "nothing to
/// show": a present carrying it is a well-formed packet whose completion is
/// owed in full, not a malformed one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MappingId(pub u32);

/// A guest task ordinal, as it arrives on the wire.
///
/// Task 0 is the kernel task and is a legal value, so this is not `NonZero`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskId(pub u32);

/// The guest page frame a task's page-table root sits at.
///
/// A task's address space *is* this number: two definitions of one task id with
/// the same root are the same space re-declared, and two with different roots
/// are different spaces under one name — which is why the model keeps it rather
/// than only the id. It is a page frame and not an address; what the frame
/// contains, and how a walk of it translates anything, belongs to the layer
/// that can read guest pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DirectoryFrame(pub u32);

/// A name in the kernel **object-list** ref space: what a task's object list is
/// keyed by, and what a decoded command resolves its objects out of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectListRef(pub u32);

/// A name in the **serializer's per-kind** ref space, which is a different
/// namespace from [`ObjectListRef`] and overlaps it numerically.
///
/// Kept apart by type because they were once kept apart by attention. The
/// object-destroy packet carries one of these, and using it against the object
/// table could only ever have retired an unrelated object that shared the
/// integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SerializerRef {
    /// The record opcode that names the object *kind*. Refs are per kind, so a
    /// ref without its kind is not a name.
    pub kind: u32,
    pub value: u32,
}

/// A resolved resource: the slot the guest named and the generation it was
/// resolved in.
///
/// Work holds one of these and never a bare slot. A guest that deletes an
/// object and creates another in the same slot produces a new generation, so
/// work still holding the old id resolves to nothing rather than to the new
/// object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId {
    pub slot: ObjectListRef,
    pub generation: SlotGeneration,
}

/// How many times a namespace slot has been filled.
///
/// Wrapping is not a concern the way it is for a completion stamp: this counts
/// creations of one object slot, and it is 64 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SlotGeneration(pub u64);

impl SlotGeneration {
    #[must_use]
    pub const fn next(self) -> SlotGeneration {
        SlotGeneration(self.0.wrapping_add(1))
    }
}

/// A FIFO channel: the submission ordering domain every packet on it belongs
/// to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChannelId(pub u32);

impl ChannelId {
    /// The domain that exists because the device does.
    ///
    /// Every other domain is opened by a `CmdDefineFifo` the guest sends, and
    /// **this is the domain it sends it on**. There is no packet that could
    /// open it: a model requiring one would refuse the command that opens
    /// everything else, along with every task definition, object-list bind and
    /// device query the guest puts on the root FIFO. So
    /// [`crate::session::SessionModel::new`] opens it, and
    /// [`crate::session::SessionModel::retire_channel`] refuses to close it —
    /// the root FIFO's publication lifetime is the device's.
    pub const ROOT: Self = Self(0);
}

/// A packet's position within its channel.
///
/// Channel-local and monotonic. Two packets on different channels have no
/// ordering from this alone — that is what [`IngressOrdinal`] is for, and the
/// two are deliberately not the same number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ChannelSequence(pub u64);

impl ChannelSequence {
    #[must_use]
    pub const fn next(self) -> ChannelSequence {
        ChannelSequence(self.0 + 1)
    }
}

/// A packet's position in the device's single arrival order, across all
/// channels.
///
/// The dependency compiler consumes transactions in increasing ingress order
/// and every hazard edge it creates points at a *lower* ordinal. That is the
/// property that makes the graph acyclic by construction, and it is only true
/// because this number is assigned once at ingress and never re-derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct IngressOrdinal(pub u64);

impl IngressOrdinal {
    #[must_use]
    pub const fn next(self) -> IngressOrdinal {
        IngressOrdinal(self.0 + 1)
    }
}

/// Where one accepted packet sits, in every order the device keeps.
///
/// # Assigned once, by the one service that observes arrival
///
/// The four numbers are not independent: a packet's channel decides which
/// sequence counter advances, and both are consumed under the same arrival that
/// produced the ingress ordinal. Anything that could state them separately
/// could state them inconsistently — a payload claiming ingress 7 inside an
/// envelope that arrived at 8 is representable the moment two structures each
/// carry their own copy, and no reader could say which one the dependency graph
/// meant.
///
/// So identity is one value, and the service that observes FIFO arrival is the
/// only thing that makes one. A builder resolving a packet's contents does not
/// know where the packet sits and must not be able to say; it produces work,
/// and admission stamps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransactionIdentity {
    /// The semantic lifetime this was accepted in.
    pub session: SessionGeneration,
    /// The submission ordering domain.
    pub domain: ChannelId,
    /// Position within that domain.
    pub domain_sequence: ChannelSequence,
    /// Position in the device's single arrival order.
    pub ingress: IngressOrdinal,
}

/// Which of the guest's completion-stamp words a stamp names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StampSlot(pub u32);

/// A point on a channel's completion timeline.
///
/// # It wraps, and the type knows it
///
/// The timeline is 32 bits and the guest lets it wrap. So "has this point been
/// reached" is not `>=`: after a wrap the later value is numerically smaller,
/// and a plain comparison parks a channel forever on a stamp the device has
/// already written. The answer is the sign of the wrapping difference, which is
/// correct as long as the two points are within 2^31 of each other — which the
/// contract guarantees, because a channel advances its timeline once per
/// lowered EXEC and cannot have 2^31 of them outstanding.
///
/// That arithmetic was spelled inline at each reader. It is one line and it is
/// the kind of one line that gets written as `>=` by the next person, so it is
/// a method here and the readers ask instead of remembering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct StampValue(pub u32);

impl StampValue {
    /// Whether `self` is at or past `other` on the wrapping timeline.
    #[must_use]
    pub const fn reached(self, other: StampValue) -> bool {
        self.0.wrapping_sub(other.0) as i32 >= 0
    }

    /// Whether `self` is strictly past `other`.
    #[must_use]
    pub const fn follows(self, other: StampValue) -> bool {
        (self.0.wrapping_sub(other.0) as i32) > 0
    }

    /// The later of two points, in the same order [`Self::follows`] compares
    /// in.
    #[must_use]
    pub const fn later(self, other: StampValue) -> StampValue {
        if self.follows(other) {
            self
        } else {
            other
        }
    }

    /// The next point on the timeline. One lowered EXEC advances it once.
    #[must_use]
    pub const fn next(self) -> StampValue {
        StampValue(self.0.wrapping_add(1))
    }
}

/// A completion obligation: the value a packet owes into a slot when its work
/// has completed.
///
/// Carried as a pair because the slot is not implied by the channel — a packet
/// names the word its own completion is published into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompletionStamp {
    pub slot: StampSlot,
    pub value: StampValue,
}

/// A prerequisite: a point another packet must have published before this one
/// may begin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StampWait {
    pub slot: StampSlot,
    pub value: StampValue,
}

impl StampWait {
    /// Whether a published value discharges this wait.
    #[must_use]
    pub const fn satisfied_by(self, published: StampValue) -> bool {
        published.reached(self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole reason [`StampValue`] is not a `u32`.
    #[test]
    fn a_wrapped_timeline_point_still_compares_as_later() {
        let before = StampValue(u32::MAX - 2);
        let after = before.next().next().next().next();
        assert_eq!(after, StampValue(1), "the timeline wrapped, as it may");
        assert!(
            after.follows(before),
            "a plain `>` here reads 1 > 4294967293 as false and parks the \
             channel forever on a stamp already written"
        );
        assert!(!before.follows(after));
        assert_eq!(after.later(before), after);
        assert_eq!(before.later(after), after);
    }

    #[test]
    fn a_point_has_reached_itself_but_does_not_follow_itself() {
        let v = StampValue(7);
        assert!(v.reached(v));
        assert!(!v.follows(v));
        assert!(StampWait {
            slot: StampSlot(0),
            value: v
        }
        .satisfied_by(v));
    }

    /// The comparison is only correct inside half the space, and the contract
    /// is what keeps it there. Pinned so that a change to the timeline's width
    /// or to how often a channel advances it has to come back here.
    #[test]
    fn the_wrapping_order_holds_across_half_the_space() {
        let base = StampValue(0x8000_0000);
        for step in [1u32, 2, 1000, 0x4000_0000, 0x7fff_ffff] {
            let later = StampValue(base.0.wrapping_add(step));
            assert!(later.follows(base), "step {step:#x}");
            assert!(!base.follows(later), "step {step:#x}");
        }
    }

    /// A generation counter that wrapped would let a lease from an ancient
    /// reset answer for a live one, and the failure would look like a valid
    /// name rather than like a bug.
    #[test]
    fn the_session_generation_stops_rather_than_wrapping() {
        let ceiling = SessionGeneration(NonZeroU64::new(u64::MAX).unwrap());
        assert_eq!(ceiling.next(), ceiling);
        assert_eq!(SessionGeneration::FIRST.get(), 1);
        assert!(SessionGeneration::FIRST.next() > SessionGeneration::FIRST);
    }

    /// Two namespaces that overlap numerically must not be interchangeable, and
    /// this is the compiler's version of that claim: the two types share no
    /// conversion, so the only way to cross them is to write the crossing out.
    #[test]
    fn a_resolved_id_carries_the_generation_it_was_resolved_in() {
        let slot = ObjectListRef(32);
        let first = ResourceId {
            slot,
            generation: SlotGeneration::default(),
        };
        let after_reuse = ResourceId {
            slot,
            generation: SlotGeneration::default().next(),
        };
        assert_ne!(
            first, after_reuse,
            "the guest deleted and recreated in one slot; work holding the old \
             id must not resolve to the new object"
        );
        assert_eq!(first.slot, after_reuse.slot);
    }
}
