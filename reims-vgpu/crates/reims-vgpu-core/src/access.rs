//! What a transaction touches, how precisely that is known, and what that
//! implies about ordering.
//!
//! # Precision is a ladder, and the rung is part of the answer
//!
//! An operation's access can be known at four different precisions, and the
//! model has to carry which one it got rather than flattening them:
//!
//! 1. the exact byte range or image subresource, when the command names it;
//! 2. the whole backing, when a resource table names participation but no
//!    range — or the whole heap, when a heap-use record declares indirect
//!    access to everything allocated from it without naming a resource;
//! 3. the submission domain alone, when participation is incomplete;
//! 4. a typed refusal, when the *operation* cannot be executed correctly.
//!
//! Rung 4 is the one that is easy to reach for and wrong. Lack of a concurrency
//! proof is not a reason to reject valid guest work: an operation whose access
//! is imprecisely known still executes, at rung 3, ordered by its domain. Only
//! an operation the device cannot perform is refused, and that decision belongs
//! to the closure ledger rather than to this compiler.
//!
//! # Coarse and fine keys have to meet
//!
//! The rungs are not separate namespaces. A draw that names a level of a
//! texture and a resource-lifecycle packet that names the whole backing are
//! talking about the same memory, and a conflict test that compared them by
//! variant would let them past each other. So conflict is decided on the
//! memory, not on the precision: two keys conflict when the memory they could
//! refer to overlaps, and a coarser key overlaps everything inside it.
//!
//! # Read against read is the only free pair
//!
//! Within one key, a proven read depends on the preceding writer; a proven
//! write depends on the preceding writer *and* every preceding reader; two
//! proven reads create no edge. An access whose mode is not established
//! conflicts with everything, visibly — [`AccessMode::Unknown`] is a distinct
//! variant precisely so a census can count how much ordering is being bought
//! with ignorance.

use crate::identity::{ChannelId, ResourceId};

/// The canonical identity of a piece of backing memory.
///
/// Aliasing is decided from contract-declared backing relationships, and this
/// is that decision's result: two resources that share backing share a
/// `BackingId`. Resource names alone never prove or disprove aliasing, so
/// nothing here is derived from a name.
///
/// # What a producer of these has to establish first
///
/// This crate consumes `BackingId`s and cannot mint one: minting requires
/// reading a guest descriptor and a page table, which is the device's. A survey
/// of what this interface's objects actually name found that the device could
/// not mint one correctly either, and the reason was a contract question rather
/// than missing code. Recorded here, at the type that demands the value,
/// because the failure mode is silent: an id that is *too distinct* — two names
/// for one piece of storage getting different ids — makes
/// [`crate::depend`] find no conflict between them and drop the hazard edge
/// that was ordering a real read against a real write. Nothing refuses, nothing
/// logs, and the frame is wrong intermittently.
///
/// Two of the three joins are settled, and the device mints against both of
/// them now — an address-named window through one entry point, a mapping's
/// surface through the other, interned from one counter so the two key spaces
/// are one identity space. The join that is not settled is named below with
/// what would settle it: a value somebody has to supply, not a question
/// somebody has to think about. An object whose identity turns on it is refused
/// by name and never approximated.
///
/// ## Settled: every name this device can see is a window of one address space
///
/// Some objects name a mapping the device tracks; the rest name pages by a
/// handle in the owning task. Those are *not* two namespaces. Both are
/// guest-virtual page numbers at the device's own page shift, resolved through
/// the same task's page directory, and the device states it at both ends — a
/// linear descriptor's backing is its handle shifted, and a mapping's page
/// number is documented in the attach path as a guest-virtual page walked
/// through that task's directory. So two names are the same storage exactly
/// when their windows overlap, which is a test rather than a rule that has to
/// be supplied.
///
/// The cross-task worry that leaves — one imported surface being a different
/// address in each address space — does not arise, and this is the load-bearing
/// part. A surface's backing is registered in the accelerator's kernel task,
/// whose id is fixed; the guest says so on the wire in the ref-texture view's
/// owner field; and a client naming that surface reaches it through the mapping
/// rather than through its own address space. The device holds itself to it
/// with two always-on instruments: one counts how many tasks claim a surface
/// id and reports anything above one, the other reports a ref-texture view
/// whose owner field is not the kernel task. Either reading would mean this
/// paragraph has stopped being true, which is why they are failures and not
/// counters.
///
/// So a window in the owning task is a sufficient canonical identity for
/// everything this device can name today — buffers, textures, shader blobs,
/// indirect command memory and imported surfaces alike.
///
/// ## Settled: a window plus an incarnation, and the device counts both halves
///
/// [`crate::namespace::Namespace::declare`] wants an id at declaration, and on
/// this interface a declaration is the guest writing a record into its own
/// object-list page — before it has necessarily finished mapping the backing.
/// A descriptor-derived window is available then and a page-derived one is not,
/// so the settled join above says the descriptor is the side to derive from.
///
/// A window alone is not enough. A physical replacement re-points the *same*
/// guest-virtual window at different host frames. Work already accepted was
/// planned against the old frames and must keep reading them — which is
/// [`crate::namespace::Namespace::replace_physical`]'s whole contract — so the
/// two incarnations must not share an id. Derived from the window alone they
/// would, and then a [`Claim`] on the old frames would be satisfied by the new
/// ones and the old storage would be handed back while something is still
/// reading it. That is false *equality*, the direction this type's other risk
/// is not, and it is the one a window invites.
///
/// So the id is a window and an incarnation, and the device now counts one for
/// each of the two ways it can reach storage. A mapping bumps a generation
/// whenever its page list changes — a map, an unmap, a replacement, a reattach,
/// or a page-table refresh that moves frames. Storage named by an address in a
/// task carries a pair: a count advanced by a re-point packet, and a per-task
/// epoch advanced when the task's whole address space ends. Two scopes because
/// the events have two scopes, and an epoch rather than a walk because most
/// references are ones the guest published and this device never touched — they
/// have no per-name entry for a walk to find, and the epoch covers them without
/// having to have seen them.
///
/// ## The count is on the window, and it was on the reference
///
/// It was on the reference, because a re-point packet names a reference and
/// nothing else — which is canonical exactly as long as one window has one
/// live name. **It does not.** A driven macos-15 boot examined 74 collisions
/// on that claim and one of them was a genuine alias: two live references in
/// one task naming a single 8 294 400-byte window, which is 1920×1080×4 — the
/// compositor's own scanout allocation, and so the most hazard-critical
/// backing on the device.
///
/// A per-reference count would give that framebuffer two identities. A
/// re-point through one reference advances that reference's count and leaves
/// the other naming the old incarnation, so a [`Claim`] held under the second
/// name goes on claiming frames the first has already replaced, and the edge
/// between the two names is never drawn. That is false distinctness, and it is
/// on the one buffer where it matters most.
///
/// So the re-point resolves the reference it names to that reference's window
/// and advances the count *there*, and both names move together. Releasing a
/// name advances nothing: a name is not storage, and a reference reused for
/// different storage names a different window and is distinct already.
///
/// The other 73 collisions were one name after another over a recycled
/// allocation, which is not an alias — telling those apart needs the holder's
/// *current* object-list record re-read out of guest memory, because the guest
/// frees an object by writing over that record and sends no packet at all.
///
/// [`Claim`]: crate::namespace::Claim
///
/// ## Open: a heap's extent — but not, any longer, its identity
///
/// [`crate::heap`] takes a placement as its heap's [`BackingId`] and a byte
/// range, because two windows the guest chose to overlap are the same bytes.
/// A heap-placed texture names a heap reference and an offset, and neither had
/// anything behind it. The identity half is no longer the hard part: a heap
/// reference is a reference in the packet's own task, in the same object list
/// every other reference in that record resolves through — the record puts it
/// in the same `u32` form immediately after the resource's own — and the
/// device's resource constructor already accepts heap object tags. So a heap
/// is an object-list object, and an object-list object's canonical identity is
/// the address-named form the join above settles: the same window, the same
/// per-reference incarnation.
///
/// That claim is under test rather than asserted. The device probes what the
/// reference names on the always-on failure channel, and a reading of *nothing
/// listed* is what would falsify it — the reading is the point of the
/// instrument, and no driven boot has produced one yet because none placed a
/// heap texture.
///
/// What is still missing is the heap's **extent**, which [`crate::heap`] wants
/// as a length and no packet has yet been shown to carry. Until it is
/// recovered, a placement cannot be bounded, and an unbounded heap cannot say
/// which two offsets overlap — so two placements in one heap would still come
/// out distinct. That is the whole of what remains, and it is a decode question
/// with a named place to look: the descriptor at the slot the reference names.
///
/// ## Not this
///
/// An id derived from a resource's own name — a per-resource counter, a slot
/// number — is the tempting shape and the forbidden one: it is what the first
/// paragraph of this type rules out, it gets every sharing relationship wrong
/// in the dangerous direction, and it would compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackingId(pub u64);

/// A heap, and the generation of its membership at the point the record was
/// decoded.
///
/// The generation is carried because it is a decoded fact worth reporting —
/// it says *which* membership set the declaration was written against — but it
/// is deliberately not part of the aliasing question. A resource leaves a heap
/// only by being destroyed, so a declaration recorded at generation *N* and an
/// access recorded at generation *N+1* can still name the very same bytes;
/// requiring the generations to match would drop that edge, and a dropped
/// hazard edge is a race. [`HeapId::same_heap`] is therefore what the conflict
/// test asks, and the cost of asking it is over-ordering against resources
/// placed after the declaration — the sound direction, and the same
/// conservatism the heap rung already carries by naming no usage at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeapId {
    pub id: u64,
    pub membership_generation: u64,
}

impl HeapId {
    /// Whether these name the same heap, whatever either one's membership was
    /// when it was recorded.
    #[must_use]
    pub const fn same_heap(self, other: HeapId) -> bool {
        self.id == other.id
    }
}

/// A half-open byte range within one backing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

impl ByteRange {
    /// Whether two ranges share a byte.
    ///
    /// A zero-length range overlaps nothing, including itself: it names no byte,
    /// so an operation carrying one touches nothing and cannot conflict.
    #[must_use]
    pub const fn overlaps(self, other: ByteRange) -> bool {
        if self.length == 0 || other.length == 0 {
            return false;
        }
        let self_end = self.offset.saturating_add(self.length);
        let other_end = other.offset.saturating_add(other.length);
        self.offset < other_end && other.offset < self_end
    }
}

/// A half-open interval of one task's GPU virtual address space.
///
/// **Not a [`ByteRange`], and the distinction is the point.** A `ByteRange` is
/// an offset into one backing — a place inside content this device holds. A
/// `GuestSpan` is an address in the guest's own translation space, which names
/// whatever pages that space currently maps and names *different* pages once
/// the guest remaps it. Adding one to the other, or resolving one where the
/// other is expected, reads bytes at an offset that has no relationship to the
/// address that was meant.
///
/// It is a name and never a pointer. Nothing in this crate follows it; the
/// operations that carry one say *which interval the guest changed*, and the
/// executor that owns translation is what turns it into pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GuestSpan {
    pub base: u64,
    pub length: u64,
}

impl GuestSpan {
    /// Whether two intervals of the same task's space share an address.
    ///
    /// Zero length overlaps nothing, for [`ByteRange::overlaps`]'s reason: an
    /// interval naming no address cannot name one in common with another.
    #[must_use]
    pub const fn overlaps(self, other: Self) -> bool {
        if self.length == 0 || other.length == 0 {
            return false;
        }
        let self_end = self.base.saturating_add(self.length);
        let other_end = other.base.saturating_add(other.length);
        self.base < other_end && other.base < self_end
    }
}

/// A half-open window of an image's levels and slices.
///
/// Plane is exact rather than a range: a plane is a separate memory layout, not
/// a coordinate within one, so "planes 0..2" is two accesses and not one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubresourceRange {
    pub base_level: u32,
    pub level_count: u32,
    pub base_slice: u32,
    pub slice_count: u32,
    pub plane: u32,
}

impl SubresourceRange {
    /// The single level and slice a record names.
    ///
    /// Plane is zero, and that is a contract statement rather than a default:
    /// no record on this wire carries a plane, so naming one here would be
    /// inventing a field — a planar format's second plane is a separate
    /// resource in the guest's own model.
    ///
    /// Written once because four records spell this same shape — a blit
    /// endpoint, a `slice:level:` content directive, and an attachment's
    /// target and resolve target — and a fifth spelling it by hand is where a
    /// `level_count` of zero, or a plane taken from somewhere, gets in.
    #[must_use]
    pub const fn one(slice: u32, level: u32) -> Self {
        Self {
            base_level: level,
            level_count: 1,
            base_slice: slice,
            slice_count: 1,
            plane: 0,
        }
    }

    #[must_use]
    pub const fn overlaps(self, other: SubresourceRange) -> bool {
        if self.plane != other.plane {
            return false;
        }
        span_overlaps(
            self.base_level,
            self.level_count,
            other.base_level,
            other.level_count,
        ) && span_overlaps(
            self.base_slice,
            self.slice_count,
            other.base_slice,
            other.slice_count,
        )
    }
}

const fn span_overlaps(a_base: u32, a_count: u32, b_base: u32, b_count: u32) -> bool {
    if a_count == 0 || b_count == 0 {
        return false;
    }
    let a_end = a_base.saturating_add(a_count);
    let b_end = b_base.saturating_add(b_count);
    a_base < b_end && b_base < a_end
}

/// A resource's backing, and the heap it was allocated from if it has one.
///
/// The heap travels with the key because heap-use participation names a heap
/// and not its members: without it, a heap declaration and a resource access
/// have nothing to compare and the coarser rung would silently order against
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceKey {
    pub backing: BackingId,
    pub heap: Option<HeapId>,
}

/// How precisely an access is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessKey {
    /// The command named an exact byte range.
    Range(ResourceKey, ByteRange),
    /// The command named exact levels, slices and a plane.
    Subresource(ResourceKey, SubresourceRange),
    /// Participation is named but no range is: the whole backing.
    Whole(ResourceKey),
    /// A heap-use record: indirect access to everything allocated from this
    /// heap, with no per-resource usage named.
    Heap(HeapId),
    /// Participation is incomplete. Nothing about which memory is touched is
    /// established, so ordering comes from the submission domain alone.
    ///
    /// Not a refusal. An operation here still executes.
    DomainOnly,
}

impl AccessKey {
    /// Which rung of the precision ladder this key sits on, for the census that
    /// prices how much ordering is being bought with imprecision.
    #[must_use]
    pub const fn rung(self) -> u8 {
        match self {
            Self::Range(..) | Self::Subresource(..) => 1,
            Self::Whole(_) | Self::Heap(_) => 2,
            Self::DomainOnly => 3,
        }
    }

    const fn resource(self) -> Option<ResourceKey> {
        match self {
            Self::Range(r, _) | Self::Subresource(r, _) | Self::Whole(r) => Some(r),
            Self::Heap(_) | Self::DomainOnly => None,
        }
    }

    /// Whether the memory these two keys could refer to overlaps.
    ///
    /// Deliberately decided on the memory rather than on the variant: a draw
    /// naming one level of a texture and a lifecycle packet naming the whole
    /// backing are talking about the same bytes, and comparing them by shape
    /// would let them past each other.
    #[must_use]
    pub fn may_alias(self, other: AccessKey) -> bool {
        // Incomplete participation could be anything, so it meets everything.
        if matches!(self, Self::DomainOnly) || matches!(other, Self::DomainOnly) {
            return true;
        }
        match (self, other) {
            // Two declarations of one heap meet, whichever membership each
            // was recorded against.
            (Self::Heap(a), Self::Heap(b)) => a.same_heap(b),
            // A heap declaration meets every resource allocated from it.
            (Self::Heap(h), key) | (key, Self::Heap(h)) => key
                .resource()
                .is_some_and(|r| r.heap.is_some_and(|rh| rh.same_heap(h))),
            (a, b) => {
                let (Some(ra), Some(rb)) = (a.resource(), b.resource()) else {
                    return false;
                };
                if ra.backing != rb.backing {
                    return false;
                }
                match (a, b) {
                    (Self::Range(_, x), Self::Range(_, y)) => x.overlaps(y),
                    (Self::Subresource(_, x), Self::Subresource(_, y)) => x.overlaps(y),
                    // A byte range and a subresource window are two coordinate
                    // systems over one backing, and nothing here relates them.
                    // The honest answer is that they may alias — narrowing it
                    // would need the image's layout, which is a decision the
                    // executor owns and this crate cannot see.
                    _ => true,
                }
            }
        }
    }
}

/// What an access does to the memory it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
    /// The direction is not established.
    ///
    /// A separate variant rather than `ReadWrite`, even though it orders the
    /// same way: the census has to be able to say how many edges exist because
    /// something is genuinely read-modify-write and how many exist because
    /// nobody knows, and a conservative answer that cannot be counted is a
    /// conservative answer nobody will ever narrow.
    Unknown,
}

impl AccessMode {
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite | Self::Unknown)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::ReadWrite => "read_write",
            Self::Unknown => "unknown",
        }
    }
}

/// A resource's content version.
///
/// Content transfers are keyed to a version transition: no transfer happens
/// without one, and none repeats for the same key. An access declares the
/// version it consumes and, when it writes, the version it will produce — and
/// the produced version is *reserved* during planning and committed only after
/// the work completes, so a reader planned against it waits for the completion
/// rather than for the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ContentVersion(pub u64);

impl ContentVersion {
    #[must_use]
    pub const fn next(self) -> ContentVersion {
        ContentVersion(self.0.wrapping_add(1))
    }
}

/// One access an operation declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessIntent {
    /// The submission ordering domain the access belongs to. Carried on the
    /// access rather than looked up, because a conflict test that had to reach
    /// for it is a conflict test that can be called without it.
    pub domain: ChannelId,
    pub key: AccessKey,
    pub mode: AccessMode,
    /// The API stages the access is declared for, as the wire carries them.
    /// Translated into host stage masks by an executor and never here.
    pub api_stages: u32,
    /// The version this access consumes, when one is established.
    pub input_content_version: Option<ContentVersion>,
    /// The version this access will produce. Reserved at planning, committed at
    /// completion; `None` for a pure read.
    pub output_content_version: Option<ContentVersion>,
}

/// What an operation says it touches, before the resource is resolved.
///
/// An operation record names a *ref* and a region; it does not name a backing,
/// a heap, or a content version. Those come from resolution, which needs the
/// resource registry — a thing this module cannot and should not see. So the
/// operation's own claim is this, and [`Participation::resolve`] is the single
/// step that turns it into an [`AccessIntent`].
///
/// The split matters beyond tidiness: it is what makes "an operation declares
/// its exact participation" checkable at the operation, where the record's
/// fields are, rather than after a registry lookup has already had a chance to
/// widen it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Participation {
    pub resource: ResourceId,
    pub extent: ParticipationExtent,
    pub mode: AccessMode,
    /// The API stages the record declares, as the wire carries them.
    ///
    /// Zero for a transfer: a copy record names no shader stage, and the
    /// transfer stage a host needs is a translation an executor performs. A
    /// non-zero value here always came from a record that carried one.
    pub api_stages: u32,
}

/// How much of a resource an operation named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticipationExtent {
    /// An exact byte range.
    Range(ByteRange),
    /// Exact levels, slices and a plane.
    Subresource(SubresourceRange),
    /// The record named the resource and nothing narrower.
    ///
    /// Not the same as unknown participation: the resource *is* named, so this
    /// still conflicts only with that resource's memory. It is the honest
    /// answer for a record like `generateMipmapsForTexture:`, whose extent is
    /// the texture's whole pyramid and whose level count the record does not
    /// carry.
    Whole,
}

impl Participation {
    /// Attach the resolved backing, submission domain and content versions.
    ///
    /// The versions are the caller's because they are the content authority's:
    /// this type knows the operation reads or writes, and the authority knows
    /// which version that is.
    #[must_use]
    pub const fn resolve(
        &self,
        domain: ChannelId,
        resource: ResourceKey,
        input_content_version: Option<ContentVersion>,
        output_content_version: Option<ContentVersion>,
    ) -> AccessIntent {
        AccessIntent {
            domain,
            key: match self.extent {
                ParticipationExtent::Range(r) => AccessKey::Range(resource, r),
                ParticipationExtent::Subresource(s) => AccessKey::Subresource(resource, s),
                ParticipationExtent::Whole => AccessKey::Whole(resource),
            },
            mode: self.mode,
            api_stages: self.api_stages,
            input_content_version,
            output_content_version,
        }
    }

    /// The participation over a resource that owns no bytes.
    ///
    /// A resource declared [`crate::lifecycle::Storage::NoBytes`] resolves to a
    /// live name with no backing, no extent and no content authority — there
    /// are no bytes for any of the three to be about. So there is no key to
    /// compare against another access's key, and the honest answer is
    /// [`AccessKey::DomainOnly`]: participation is real and incomplete, and
    /// ordering comes from the submission domain alone.
    ///
    /// **Never a missing edge.** `DomainOnly` meets everything in
    /// [`AccessKey::may_alias`], so this over-orders rather than under-orders,
    /// and [`AccessKey::rung`] prices exactly that. The alternative — refusing
    /// the transaction — would drop guest work whose only fault is naming an
    /// object whose bytes this device cannot address, which is not a fault the
    /// contract names.
    ///
    /// The versions are `None` on both sides for the same reason there is no
    /// key: a write to bytes that do not exist reserves nothing, and a read of
    /// them consumes nothing.
    #[must_use]
    pub const fn resolve_no_bytes(&self, domain: ChannelId) -> AccessIntent {
        AccessIntent {
            domain,
            key: AccessKey::DomainOnly,
            mode: self.mode,
            api_stages: self.api_stages,
            input_content_version: None,
            output_content_version: None,
        }
    }
}

/// Up to two participations, without an allocation.
///
/// Two, because that is the widest thing any *record* declares by itself: a
/// draw's index buffer and its indirect arguments, a copy's source and its
/// destination, an ICB and its argument buffer. A pass descriptor names more,
/// and that is exactly why it is not a record's own claim — it lives in the
/// transaction's arena and is aggregated in [`crate::exec::ResolvedOperation`],
/// where the arena is in scope.
///
/// An inline array rather than a `Vec` because this is answered once per
/// record of every stream. A heap allocation per record is a cost the shape of
/// the answer does not require, and the two operations that used to return
/// `Vec` were paying it for at most two items.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Participations {
    /// Both slots are always initialized — `len` says how many are the
    /// answer. A slot past `len` is a copy of an earlier one and never read,
    /// which is what lets this be `Copy` with no `Option` per element.
    items: [Option<Participation>; 2],
}

impl Participations {
    /// The record names no memory of its own.
    ///
    /// A real answer and not an absence: every operation class answers this
    /// question, and the classes that touch nothing say so rather than being
    /// skipped by a caller that knows which ones they are.
    pub const NONE: Self = Self { items: [None; 2] };

    #[must_use]
    pub const fn one(a: Participation) -> Self {
        Self {
            items: [Some(a), None],
        }
    }

    #[must_use]
    pub const fn two(a: Participation, b: Participation) -> Self {
        Self {
            items: [Some(a), Some(b)],
        }
    }

    /// One participation, or none, from an `Option` — the shape a record with
    /// a single optional read has.
    #[must_use]
    pub const fn maybe(a: Option<Participation>) -> Self {
        Self { items: [a, None] }
    }

    /// The two optional slots, in record order, with the gaps closed.
    ///
    /// A draw may name arguments and no index buffer, so the slots are
    /// independently optional and the answer must not have a hole in it.
    #[must_use]
    pub fn pair(a: Option<Participation>, b: Option<Participation>) -> Self {
        match (a, b) {
            (Some(a), Some(b)) => Self::two(a, b),
            (Some(only), None) | (None, Some(only)) => Self::one(only),
            (None, None) => Self::NONE,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Participation> {
        self.items.iter().flatten()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl core::ops::Index<usize> for Participations {
    type Output = Participation;

    /// The `index`th participation, in the order the record names them.
    ///
    /// Indexable because the order is part of the answer — a copy's source is
    /// first and its destination second — and because the slots are packed:
    /// [`Participations::pair`] closes the gap, so a present slot never
    /// follows an absent one and `p[1]` cannot mean "the second slot, which
    /// happens to be empty".
    ///
    /// # Panics
    ///
    /// Past the answer's length, like any slice.
    fn index(&self, index: usize) -> &Participation {
        self.items
            .get(index)
            .and_then(Option::as_ref)
            .expect("participation index past the answer")
    }
}

impl IntoIterator for Participations {
    type Item = Participation;
    type IntoIter = core::iter::Flatten<core::array::IntoIter<Option<Participation>, 2>>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter().flatten()
    }
}

impl Participation {
    /// A read of a buffer window a record named, with the extent it
    /// established.
    ///
    /// `length: None` is a record that named an offset and no size — an
    /// indirect argument block whose layout is not established, say — and it
    /// widens to [`ParticipationExtent::Whole`] rather than narrowing to a
    /// guessed span. The offset is not lost: it is not *carried*, because a
    /// range starting at an offset and running to an unknown end is exactly
    /// the whole resource from the hazard test's point of view, and a shorter
    /// claim is an edge that does not get built, which is a race.
    ///
    /// No shader stage: a record that names a buffer window in its own fields
    /// carries no stage mask. A participation with stages always came from a
    /// record that had them.
    #[must_use]
    pub const fn buffer_read(resource: ResourceId, offset: u64, length: Option<u64>) -> Self {
        Self {
            resource,
            extent: match length {
                Some(length) => ParticipationExtent::Range(ByteRange { offset, length }),
                None => ParticipationExtent::Whole,
            },
            mode: AccessMode::Read,
            api_stages: 0,
        }
    }
}

/// What turns a record's own participation claim into the access a scheduler
/// can order.
///
/// The step [`Participation`]'s doc names, given an owner. A record names a
/// *ref* and a region; an access names a backing, a heap membership, the
/// region in that backing's coordinates and the content versions it consumes
/// and produces. Every one of those comes from a registry this module cannot
/// see, and there is exactly one implementation that has them all —
/// [`crate::lifecycle::Lifecycle`], which owns the names, the heaps and the
/// content authority together.
///
/// It is a trait rather than a concrete parameter so that
/// [`crate::exec::ExecBuilder`] can require one without depending on the
/// lifecycle owner, and so a model test can state the accesses it means to
/// exercise. There is a blanket implementation for any `FnMut` of the same
/// shape, which is what a test uses; a closure is not a second semantic model,
/// it is the same model with the registry stubbed.
pub trait AccessSource {
    /// The access this participation is.
    ///
    /// # Errors
    ///
    /// Where the name no longer resolves or the window leaves the resource.
    /// Both refuse the whole transaction rather than dropping the access: an
    /// operation missing from the access list is a hazard edge that does not
    /// get built.
    fn access(&mut self, participation: &Participation) -> Result<AccessIntent, AccessRefusal>;
}

impl<F> AccessSource for F
where
    F: FnMut(&Participation) -> Result<AccessIntent, AccessRefusal>,
{
    fn access(&mut self, participation: &Participation) -> Result<AccessIntent, AccessRefusal> {
        self(participation)
    }
}

/// Why a participation could not become an access.
///
/// Carries the owner's own reason string rather than re-encoding it: every
/// refusal in this crate is greppable by the slug of the check that produced
/// it, and a second enum here would give the same failure two names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessRefusal {
    pub resource: ResourceId,
    pub reason: &'static str,
}

/// A stub registry for the in-crate tests whose subject is the stream rather
/// than the resolution.
///
/// Every name places onto a backing named after its slot, with no heap and no
/// content versions, so a test about encoder admission or scheduling does not
/// have to stand up a [`crate::lifecycle::Lifecycle`] to say what a record
/// touched. Not reachable outside the test build: a production caller that
/// wanted this would be one inventing a backing.
#[cfg(test)]
pub(crate) struct StubRegistry(pub ChannelId);

#[cfg(test)]
impl AccessSource for StubRegistry {
    fn access(&mut self, participation: &Participation) -> Result<AccessIntent, AccessRefusal> {
        Ok(participation.resolve(
            self.0,
            ResourceKey {
                backing: BackingId(u64::from(participation.resource.slot.0)),
                heap: None,
            },
            None,
            None,
        ))
    }
}

/// Whether an earlier access and a later one require an ordering edge.
///
/// Read against read is the only free pair, and only when both directions are
/// established: an [`AccessMode::Unknown`] on either side conflicts, because
/// what it does not know might be a write.
///
/// Cross-domain accesses produce no edge from conflict alone. That is not an
/// oversight and not an optimisation: the contract leaves separate submission
/// domains unordered, and manufacturing an edge here would repair an
/// application data race into a guarantee this API does not make. Cross-domain
/// visibility comes from explicit synchronisation, and a host-safety
/// serialisation an executor needs is a separate kind of edge that cannot order
/// guest-visible publication.
#[must_use]
pub fn requires_edge(earlier: &AccessIntent, later: &AccessIntent) -> bool {
    if earlier.domain != later.domain {
        return false;
    }
    if !earlier.mode.writes() && !later.mode.writes() {
        return false;
    }
    earlier.key.may_alias(later.key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(backing: u64) -> ResourceKey {
        ResourceKey {
            backing: BackingId(backing),
            heap: None,
        }
    }

    fn intent(key: AccessKey, mode: AccessMode) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(1),
            key,
            mode,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    #[test]
    fn a_zero_length_range_names_no_byte_and_so_conflicts_with_nothing() {
        let empty = ByteRange {
            offset: 0,
            length: 0,
        };
        assert!(!empty.overlaps(empty));
        assert!(!empty.overlaps(ByteRange {
            offset: 0,
            length: 16
        }));
    }

    #[test]
    fn adjacent_ranges_do_not_overlap_and_touching_ones_do() {
        let a = ByteRange {
            offset: 0,
            length: 16,
        };
        assert!(!a.overlaps(ByteRange {
            offset: 16,
            length: 16
        }));
        assert!(a.overlaps(ByteRange {
            offset: 15,
            length: 1
        }));
    }

    /// The failure this exists to prevent: a fine key and a coarse key over one
    /// backing compared by shape rather than by memory, so a resource delete
    /// naming the whole backing passes a draw naming one level of it.
    #[test]
    fn a_whole_backing_meets_every_precise_access_inside_it() {
        let whole = AccessKey::Whole(key(7));
        let sub = AccessKey::Subresource(
            key(7),
            SubresourceRange {
                base_level: 3,
                level_count: 1,
                base_slice: 0,
                slice_count: 1,
                plane: 0,
            },
        );
        let range = AccessKey::Range(
            key(7),
            ByteRange {
                offset: 4096,
                length: 64,
            },
        );
        assert!(whole.may_alias(sub));
        assert!(sub.may_alias(whole));
        assert!(whole.may_alias(range));
        // And a different backing still does not meet it.
        assert!(!whole.may_alias(AccessKey::Whole(key(8))));
    }

    /// Two coordinate systems over one backing, with nothing here relating
    /// them. The honest answer is that they may alias; narrowing it needs the
    /// image layout, which is the executor's to know.
    #[test]
    fn a_byte_range_and_a_subresource_over_one_backing_may_alias() {
        let range = AccessKey::Range(
            key(1),
            ByteRange {
                offset: 0,
                length: 4,
            },
        );
        let sub = AccessKey::Subresource(
            key(1),
            SubresourceRange {
                base_level: 9,
                level_count: 1,
                base_slice: 0,
                slice_count: 1,
                plane: 0,
            },
        );
        assert!(range.may_alias(sub));
    }

    #[test]
    fn different_planes_are_different_memory() {
        let plane = |p| {
            AccessKey::Subresource(
                key(2),
                SubresourceRange {
                    base_level: 0,
                    level_count: 1,
                    base_slice: 0,
                    slice_count: 1,
                    plane: p,
                },
            )
        };
        assert!(!plane(0).may_alias(plane(1)));
        assert!(plane(1).may_alias(plane(1)));
    }

    /// A heap declaration names no resource, so it can only meet one through
    /// membership.
    #[test]
    fn a_heap_declaration_meets_its_members_and_not_a_stranger() {
        let heap = HeapId {
            id: 5,
            membership_generation: 2,
        };
        let member = AccessKey::Whole(ResourceKey {
            backing: BackingId(11),
            heap: Some(heap),
        });
        let stranger = AccessKey::Whole(key(11));
        assert!(AccessKey::Heap(heap).may_alias(member));
        assert!(member.may_alias(AccessKey::Heap(heap)));
        assert!(
            !AccessKey::Heap(heap).may_alias(stranger),
            "a resource with no heap is not in one"
        );
        let other_heap = HeapId {
            id: 6,
            membership_generation: 2,
        };
        assert!(
            !AccessKey::Heap(heap).may_alias(AccessKey::Heap(other_heap)),
            "two heaps are two sets of memory"
        );
        assert!(
            !AccessKey::Heap(other_heap).may_alias(member),
            "and a member of one is not a member of the other"
        );
    }

    /// Placing a resource in a heap advances that heap's membership, and the
    /// resource that was already there did not move. A declaration recorded
    /// before the placement and an access recorded after it are talking about
    /// the same bytes, so they must still meet: the generation says which set
    /// was declared, not which memory exists.
    #[test]
    fn a_membership_change_does_not_dissolve_a_heap_hazard() {
        let declared = HeapId {
            id: 5,
            membership_generation: 2,
        };
        let after_a_placement = HeapId {
            id: 5,
            membership_generation: 3,
        };
        assert!(AccessKey::Heap(declared).may_alias(AccessKey::Heap(after_a_placement)));
        let member_now = AccessKey::Whole(ResourceKey {
            backing: BackingId(11),
            heap: Some(after_a_placement),
        });
        assert!(AccessKey::Heap(declared).may_alias(member_now));
        assert!(member_now.may_alias(AccessKey::Heap(declared)));
    }

    /// Incomplete participation could be anything, so it meets everything —
    /// and it is still not a refusal.
    #[test]
    fn incomplete_participation_meets_everything() {
        for other in [
            AccessKey::Whole(key(1)),
            AccessKey::Heap(HeapId {
                id: 1,
                membership_generation: 0,
            }),
            AccessKey::DomainOnly,
        ] {
            assert!(AccessKey::DomainOnly.may_alias(other));
            assert!(other.may_alias(AccessKey::DomainOnly));
        }
        assert_eq!(AccessKey::DomainOnly.rung(), 3);
    }

    #[test]
    fn read_against_read_is_the_only_free_pair() {
        let k = AccessKey::Whole(key(3));
        let r = intent(k, AccessMode::Read);
        let w = intent(k, AccessMode::Write);
        let rw = intent(k, AccessMode::ReadWrite);
        let u = intent(k, AccessMode::Unknown);
        assert!(!requires_edge(&r, &r));
        assert!(requires_edge(&r, &w));
        assert!(requires_edge(&w, &r));
        assert!(requires_edge(&w, &w));
        assert!(requires_edge(&rw, &r));
        assert!(
            requires_edge(&r, &u) && requires_edge(&u, &r),
            "an unestablished direction might be a write, so it is not free"
        );
    }

    /// Conflict alone does not cross submission domains. Manufacturing that
    /// edge would repair an application data race into a guarantee this API
    /// does not make.
    #[test]
    fn a_conflict_across_domains_creates_no_edge() {
        let k = AccessKey::Whole(key(4));
        let mut a = intent(k, AccessMode::Write);
        let mut b = intent(k, AccessMode::Read);
        assert!(requires_edge(&a, &b), "same domain, write then read");
        a.domain = ChannelId(1);
        b.domain = ChannelId(2);
        assert!(!requires_edge(&a, &b));
    }

    #[test]
    fn the_rungs_are_ordered_from_exact_to_domain_only() {
        let k = key(1);
        assert_eq!(
            AccessKey::Range(
                k,
                ByteRange {
                    offset: 0,
                    length: 1
                }
            )
            .rung(),
            1
        );
        assert_eq!(AccessKey::Whole(k).rung(), 2);
        assert_eq!(
            AccessKey::Heap(HeapId {
                id: 0,
                membership_generation: 0
            })
            .rung(),
            2
        );
        assert_eq!(AccessKey::DomainOnly.rung(), 3);
    }

    // ---- The alias relation's algebra, enumerated ------------------------
    //
    // `crate::depend`'s all-pairs oracle decides conflict by calling
    // `requires_edge`, which calls `may_alias` — the shadow and the thing it
    // shadows share this function, so a wrong answer here agrees with itself
    // and the sweep there cannot see it. The key space is small, so the
    // properties the dependency graph relies on are enumerated rather than
    // sampled.

    fn heap_of(id: u64, membership_generation: u64) -> HeapId {
        HeapId {
            id,
            membership_generation,
        }
    }

    /// Two backings, and every heap membership a resource on one can record:
    /// none, a heap at two different membership generations, and a second heap.
    fn every_resource_key() -> Vec<ResourceKey> {
        let mut out = Vec::new();
        for backing in [1_u64, 2] {
            for heap in [
                None,
                Some(heap_of(1, 0)),
                Some(heap_of(1, 1)),
                Some(heap_of(2, 0)),
            ] {
                out.push(ResourceKey {
                    backing: BackingId(backing),
                    heap,
                });
            }
        }
        out
    }

    /// Every variant over that pool, including the degenerate ranges that name
    /// no memory — they are the ones the relation is least obviously total on.
    fn every_key() -> Vec<AccessKey> {
        let ranges = [
            ByteRange {
                offset: 0,
                length: 0,
            },
            ByteRange {
                offset: 0,
                length: 64,
            },
            ByteRange {
                offset: 64,
                length: 64,
            },
        ];
        let subresources = [
            SubresourceRange {
                base_level: 0,
                level_count: 0,
                base_slice: 0,
                slice_count: 1,
                plane: 0,
            },
            SubresourceRange {
                base_level: 0,
                level_count: 1,
                base_slice: 0,
                slice_count: 1,
                plane: 0,
            },
            SubresourceRange {
                base_level: 1,
                level_count: 1,
                base_slice: 0,
                slice_count: 1,
                plane: 0,
            },
            SubresourceRange {
                base_level: 0,
                level_count: 2,
                base_slice: 0,
                slice_count: 1,
                plane: 1,
            },
        ];
        let mut out = vec![AccessKey::DomainOnly];
        for heap in [heap_of(1, 0), heap_of(1, 1), heap_of(2, 0)] {
            out.push(AccessKey::Heap(heap));
        }
        for resource in every_resource_key() {
            out.push(AccessKey::Whole(resource));
            for range in ranges {
                out.push(AccessKey::Range(resource, range));
            }
            for subresource in subresources {
                out.push(AccessKey::Subresource(resource, subresource));
            }
        }
        out
    }

    /// Whether a key names at least one byte. A zero-length range and a
    /// zero-count subresource window deliberately name nothing, so they are
    /// the exceptions to reflexivity rather than counterexamples to it.
    fn names_memory(key: AccessKey) -> bool {
        match key {
            AccessKey::Range(_, range) => range.length > 0,
            AccessKey::Subresource(_, window) => window.level_count > 0 && window.slice_count > 0,
            AccessKey::Whole(_) | AccessKey::Heap(_) | AccessKey::DomainOnly => true,
        }
    }

    /// An asymmetric alias relation makes the dependency graph's answer depend
    /// on which access arrived first, which is exactly the property a hazard
    /// compiler may not have.
    #[test]
    fn the_alias_relation_is_symmetric_and_reflexive_on_keys_that_name_memory() {
        let keys = every_key();
        let (mut alias, mut disjoint) = (0_u32, 0_u32);
        for &a in &keys {
            for &b in &keys {
                assert_eq!(
                    a.may_alias(b),
                    b.may_alias(a),
                    "alias is asymmetric: {a:?} vs {b:?}"
                );
                if a.may_alias(b) {
                    alias += 1;
                } else {
                    disjoint += 1;
                }
            }
            assert_eq!(
                a.may_alias(a),
                names_memory(a),
                "a key aliases itself exactly when it names memory: {a:?}"
            );
        }
        assert!(alias > 0 && disjoint > 0, "vacuous: {alias} vs {disjoint}");
    }

    /// The safety property of the precision ladder: replacing a key with a
    /// coarser one may only *add* edges. Losing one is a race, and every rung
    /// above the exact one exists precisely because the exact answer was not
    /// available.
    #[test]
    fn coarsening_a_key_never_loses_an_alias() {
        let keys = every_key();
        let mut strict_gains = 0_u32;
        for &fine in &keys {
            let coarser = match fine {
                AccessKey::Range(resource, _) | AccessKey::Subresource(resource, _) => {
                    vec![AccessKey::Whole(resource), AccessKey::DomainOnly]
                }
                AccessKey::Whole(_) | AccessKey::Heap(_) => vec![AccessKey::DomainOnly],
                AccessKey::DomainOnly => vec![],
            };
            for coarse in coarser {
                for &other in &keys {
                    if fine.may_alias(other) {
                        assert!(
                            coarse.may_alias(other),
                            "coarsening {fine:?} to {coarse:?} lost its alias with {other:?}"
                        );
                    } else if coarse.may_alias(other) {
                        strict_gains += 1;
                    }
                }
            }
        }
        assert!(
            strict_gains > 0,
            "vacuous: coarsening never widened anything"
        );
    }

    /// Distinct backings are distinct memory, and sharing a heap does not
    /// change that — heap membership is what lets a *heap* declaration meet a
    /// resource, not what lets two resources meet each other.
    #[test]
    fn resource_keys_over_different_backings_never_alias_however_they_are_placed() {
        let keys = every_key();
        let mut checked = 0_u32;
        for &a in &keys {
            for &b in &keys {
                let (Some(ra), Some(rb)) = (a.resource(), b.resource()) else {
                    continue;
                };
                if ra.backing == rb.backing {
                    continue;
                }
                assert!(!a.may_alias(b), "{a:?} and {b:?} are separate backings");
                checked += 1;
            }
        }
        assert!(checked > 0, "vacuous: no cross-backing pair in the pool");
    }

    /// A heap declaration meets a key exactly when that key records membership
    /// in the same heap, whatever generation either side was recorded against.
    /// It is deliberately *not* enough to sit on the heap's storage backing:
    /// that fact is not in the key.
    #[test]
    fn a_heap_declaration_meets_exactly_the_keys_that_record_its_membership() {
        let keys = every_key();
        let (mut met, mut missed) = (0_u32, 0_u32);
        for heap in [heap_of(1, 0), heap_of(1, 1), heap_of(2, 0)] {
            let declaration = AccessKey::Heap(heap);
            for &other in &keys {
                let expected = match other {
                    AccessKey::DomainOnly => true,
                    AccessKey::Heap(theirs) => theirs.same_heap(heap),
                    _ => other
                        .resource()
                        .and_then(|r| r.heap)
                        .is_some_and(|theirs| theirs.same_heap(heap)),
                };
                assert_eq!(
                    declaration.may_alias(other),
                    expected,
                    "heap {heap:?} against {other:?}"
                );
                if expected {
                    met += 1;
                } else {
                    missed += 1;
                }
            }
        }
        assert!(met > 0 && missed > 0, "vacuous: {met} vs {missed}");
    }

    /// Bytes and image subresources are two coordinate systems over one
    /// backing and this crate cannot relate them, so every such pair over one
    /// backing must answer conservatively — including the degenerate windows,
    /// where "names nothing" is a claim about one coordinate system that says
    /// nothing about the other.
    #[test]
    fn a_byte_range_and_a_subresource_window_over_one_backing_always_meet() {
        let keys = every_key();
        let mut bridged = 0_u32;
        for &a in &keys {
            for &b in &keys {
                let bridge = matches!(
                    (a, b),
                    (AccessKey::Range(..), AccessKey::Subresource(..))
                        | (AccessKey::Subresource(..), AccessKey::Range(..))
                );
                let (Some(ra), Some(rb)) = (a.resource(), b.resource()) else {
                    continue;
                };
                if !bridge || ra.backing != rb.backing {
                    continue;
                }
                assert!(a.may_alias(b), "{a:?} and {b:?} cannot be proven disjoint");
                bridged += 1;
            }
        }
        assert!(bridged > 0, "vacuous: no bridge pair in the pool");
    }
}
