//! The resource-lifecycle transaction: every packet in the class, and what it
//! does to the owners of names, storage and content.
//!
//! # Why the class needs a vocabulary of its own
//!
//! [`crate::transaction::PayloadClass::ResourceLifecycle`] is one of five
//! payload classes and the only one whose members do genuinely different
//! things: a delete retires a name, a synchronise owes a copy, a discard
//! *offers* to drop one. Collapsing them into "the lifecycle packet" and
//! branching on the opcode at the point of use is how the replaced
//! architecture ended up with a partial mechanism per command. So the class has
//! an exhaustive resolved vocabulary — [`LifecycleOp`] — and
//! [`LifecycleKind::of`] is checked against the closure ledger so that "every
//! lifecycle packet has an operation" is a test rather than a claim.
//!
//! # Each command's effect is the ledger's reading of it, and not more
//!
//! - `CmdInvalidateResources` says the device's cached view of the guest's
//!   pages is stale. That is not a discard of a copy: it means the guest's
//!   pages are the current content and ours is not, which is exactly a write
//!   recorded in [`Replica::GuestPages`]. Recording it as a discard would leave
//!   the device replica able to claim freshness at the old version.
//! - `CmdSynchronizeResources` says the guest is about to read the pages with
//!   its CPU, so every write already submitted must have executed *before the
//!   packet completes*. It therefore owes copies, and those copies are the
//!   transaction's completion obligation rather than something a later read
//!   discovers.
//! - `CmdDiscardResources` releases a transfer copy that a later prepare or
//!   synchronise recreates. It is a hint: the ledger's reading is that ignoring
//!   it costs memory and not correctness. So a discard whose bytes exist in no
//!   other replica is *declined* and named, because taking it would not free a
//!   spare copy — it would destroy content.
//! - `CmdSynchronizeAndDiscardResources` is the two of them in that order, and
//!   the order is the whole point: after the synchronise the guest holds the
//!   bytes, so the discard has a second holder and is never the declined kind.
//!
//! # A discard happens at completion, because submission is not completion
//!
//! Both discard forms come back in [`Effects::at_completion`] rather than being
//! applied when the packet is admitted. A discard applied at admission would
//! mark the device replica stale while the copies that make that safe are still
//! owed, and a read admitted in between would be planned against freshness that
//! does not exist yet. [`Lifecycle::complete`] is where content authority
//! actually moves, and it re-asks the sole-authority question at that point
//! rather than trusting an answer computed earlier.
//!
//! # Nothing is half-applied
//!
//! The four content commands name a list of resources. Every name in the list
//! is resolved before any of them takes effect, so a list with one stale name
//! refuses whole. A caller that saw a refusal after three of five resources had
//! been invalidated would have no way to describe the state it was left in.

use crate::access::{
    AccessIntent, AccessMode, AccessRefusal, AccessSource, BackingId, ByteRange, GuestSpan,
    Participation, ParticipationExtent, ResourceKey,
};
use crate::content::{ContentLedger, Replica, Transfer};
use crate::heap::{self, HeapPlacement, Heaps, Retirement};
use crate::identity::{ChannelId, DirectoryFrame, ObjectListRef, ResourceId, TaskId};
use crate::namespace::{self, Namespace, Teardown};
use crate::transaction::{classify, PayloadClass};
use reims_vgpu_protocol::fifo;
use reims_vgpu_protocol::packets::Channel;
use std::collections::HashMap;

/// Which lifecycle command a packet is.
///
/// Exhaustive over the packet classes [`classify`] calls
/// [`PayloadClass::ResourceLifecycle`], which is what the totality test checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LifecycleKind {
    DefineTask,
    DeleteTask,
    SetObjectList,
    DeleteResource,
    MapMemory,
    UnmapMemory,
    ReplacePhysical,
    Invalidate,
    Synchronize,
    SynchronizeAndDiscard,
    Discard,
    DeleteBacking,
    /// The serializer's per-kind destroy, carried in its own packet.
    ///
    /// The odd one of the thirteen: its payload is a task word followed by a
    /// self-describing record, so *which* object ends is in the record's opcode
    /// and not in this kind. See [`object_destroy`].
    DeleteObject,
}

impl LifecycleKind {
    /// The lifecycle command a packet is, or `None` if the packet is not a
    /// lifecycle packet at all.
    #[must_use]
    pub fn of(channel: Channel, opcode: u16) -> Option<Self> {
        if classify(channel, opcode) != Some(PayloadClass::ResourceLifecycle) {
            return None;
        }
        use Channel::{Child, Root};
        Some(match (channel, opcode) {
            // Root and child are two views of one flat opcode space; the shared
            // numbers appear twice on purpose rather than through a
            // fallthrough, so adding one to a single channel is a visible edit.
            (Root | Child, 0x38) => Self::DefineTask,
            (Root | Child, 0x20) => Self::DeleteTask,
            (Root | Child, 0x33) => Self::SetObjectList,
            (Child, 0x25) => Self::DeleteResource,
            (Child, 0x39) => Self::MapMemory,
            (Child, 0x22) => Self::UnmapMemory,
            (Child, 0x3c) => Self::ReplacePhysical,
            (Child, 0x34) => Self::Invalidate,
            (Child, 0x35) => Self::Synchronize,
            (Child, 0x3e) => Self::SynchronizeAndDiscard,
            (Child, 0x3f) => Self::Discard,
            (Child, 0x36) => Self::DeleteBacking,
            (Child, 0x28) => Self::DeleteObject,
            _ => return None,
        })
    }

    /// The length of one record in this command's counted resource list, or
    /// `None` when the command carries no such list.
    ///
    /// **Four of the twelve carry one, and they do not all use the same record
    /// length.** `Invalidate` carries the guest's validity quad beside each
    /// object ref, so its record is eight bytes; the three that only name
    /// objects use four. Nothing else here has a list at all: a task
    /// definition, an object-list bind, a single delete, a map, an unmap, a
    /// re-point and a backing retirement each name one thing.
    ///
    /// The map lives here rather than in [`reims_vgpu_protocol::fifo`] because
    /// that module holds the layouts and deliberately holds no opcodes — a
    /// second table of the same numbers cannot be kept honest by anything in
    /// the toolchain. This is the one table, and it is keyed on the kind rather
    /// than on the opcode for the same reason: [`Self::of`] already turned the
    /// opcode into a kind.
    ///
    /// Before it, the choice of decoder was made at each call site in the
    /// device's drain — three arms picked the four-byte decoder and one picked
    /// the eight-byte one, and nothing could compare them.
    #[must_use]
    pub const fn resource_list_record_len(self) -> Option<u32> {
        match self {
            Self::Invalidate => Some(fifo::CHILD_INVALIDATE_RECORD_LEN),
            Self::Synchronize | Self::SynchronizeAndDiscard | Self::Discard => {
                Some(fifo::CHILD_SYNCHRONIZE_RECORD_LEN)
            }
            Self::DefineTask
            | Self::DeleteTask
            | Self::SetObjectList
            | Self::DeleteResource
            | Self::MapMemory
            | Self::UnmapMemory
            | Self::ReplacePhysical
            | Self::DeleteBacking
            | Self::DeleteObject => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DefineTask => "define_task",
            Self::DeleteTask => "delete_task",
            Self::SetObjectList => "set_object_list",
            Self::DeleteResource => "delete_resource",
            Self::MapMemory => "map_memory",
            Self::UnmapMemory => "unmap_memory",
            Self::ReplacePhysical => "replace_physical",
            Self::Invalidate => "invalidate",
            Self::Synchronize => "synchronize",
            Self::SynchronizeAndDiscard => "synchronize_and_discard",
            Self::Discard => "discard",
            Self::DeleteBacking => "delete_backing",
            Self::DeleteObject => "delete_object",
        }
    }
}

/// Where a resource's bytes live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// Guest pages the resource names by itself.
    Dedicated {
        backing: BackingId,
        extent: ByteRange,
    },
    /// A window of a heap, at an offset the guest chose. The resource does not
    /// own storage; see [`crate::heap`].
    Placed { heap: u64, offset: u64, length: u64 },
    /// The object has no bytes of its own at all: a sampler, a function, a
    /// pipeline state, a depth-stencil state, a view over another object's
    /// bytes.
    ///
    /// **Most of an object list is this**, and the walk that produces these
    /// operations sees every slot, not only the ones that own memory. Such a
    /// name still resolves, still carries a generation and still stops
    /// resolving when the guest deletes it — see [`crate::namespace`], where a
    /// slot's backing is optional for this reason — but it has no residency, no
    /// heap window and no content authority, because there are no bytes for any
    /// of the three to be about.
    ///
    /// Not expressible as a `Dedicated` with an empty extent: that would put a
    /// [`BackingId`] on an object that owns no storage, and the only place such
    /// an id could come from is the object's own name — the one derivation
    /// [`crate::access::BackingId`] rules out.
    NoBytes,
}

/// One resolved resource-lifecycle operation.
///
/// Every variant carries the task it acts in, because an object-list name is
/// task-local and a resolution that reached across tasks would find whatever
/// shared the integer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifecycleOp {
    DefineTask {
        task: TaskId,
        /// Whether the guest registered this as its kernel task.
        ///
        /// Carried because the packet's first word is `(task_id << 1) |
        /// is_kernel_task` and dropping the low bit makes the kernel task —
        /// whose own id is `0` — indistinguishable from a user task in slot
        /// zero. They are not the same registration and their teardowns do not
        /// cost the same.
        kernel: bool,
        /// The page frame the task's page table is rooted at.
        ///
        /// **What makes a redefinition of a live task answerable.** A guest
        /// does redefine one, and the question that decides what survives is
        /// whether the root moved: a definition at the same root re-declares
        /// the space it already had, and one at a different root means every
        /// address this device resolved through the old table now translates
        /// somewhere else — including the object list itself, which reads back
        /// as zeros through the new one.
        ///
        /// The packet's fourth word, the address-space length, is **not** here.
        /// No decision in this crate is a function of it — the model holds
        /// nothing keyed by a guest address, which is the same reason
        /// [`Self::MapMemory`] is an obligation rather than a table — and a
        /// field nothing reads is one that quietly acquires a wrong meaning. It
        /// stays on [`reims_vgpu_protocol::fifo::DefineTaskCommand`] for the
        /// layer that walks the space.
        directory: DirectoryFrame,
    },
    DeleteTask {
        task: TaskId,
    },
    /// The guest has said where this task's object list is and how many slots
    /// it has.
    ///
    /// **The operation is the binding, not the walk, and that is a correction a
    /// measurement forced.** This command used to have no operation at all:
    /// A refusal named `NeedsGuestTable` said the operation was "the
    /// per-entry result of reading the table", which is a walk of guest memory
    /// this crate cannot address. A driven boot then measured what such a walk
    /// would cost — every `SetObjectList` in it declares **1 048 576** entries,
    /// eighteen times over a boot, and about twelve hundred slots are ever
    /// named. The count is the table's capacity and not its population, so the
    /// walk was never the operation: it was a gigabyte of reads to find what a
    /// resolution finds one slot at a time.
    ///
    /// So the binding is the operation, and [`Self::CreateResource`] stays what
    /// it always was — one slot's declaration, produced when something names
    /// that slot rather than when the table is bound.
    ///
    /// A rebind does **not** end the objects already declared in the task. That
    /// is the contract this device implements and it is deliberate: a new table
    /// changes what an *unresolved* reference will construct, and typed objects
    /// already built keep their own delete and their task's lifetime.
    BindObjectList {
        task: TaskId,
        list: ObjectList,
    },
    /// The object-list walk's per-object result: this slot now holds a
    /// resource with this storage.
    CreateResource {
        task: TaskId,
        slot: ObjectListRef,
        storage: Storage,
    },
    /// The guest is ending one object-list slot's resource.
    ///
    /// # The name is optional for the reason a destroy's is
    ///
    /// This carried a bare `ResourceId` and an unresolvable reference refused
    /// the whole packet, which is the same wrong answer
    /// [`Self::DeleteObject::resource`] records and for the same reason: the
    /// guest clears its own object-list slot *before* it sends the delete, so a
    /// slot this device never named answers nothing when the delete arrives.
    /// A driven macos-15 boot refused 25 packets of guest work on it.
    ///
    /// A refusal there loses more than the name. Every rail's teardown for this
    /// command — the resolution memo, the GVA resource, the bound-buffer
    /// retirement — is keyed by `(task, ref)` and not by a name, so refusing
    /// the packet leaves the device holding the very state the delete came to
    /// drop. So the reference travels beside the name, exactly as a destroy's
    /// does, and `None` orders the operation against nothing: a resource this
    /// model cannot name is one no other operation in it can name either.
    DeleteResource {
        task: TaskId,
        /// The guest's own number, which the rail's `(task, ref)`-keyed
        /// teardown needs whether or not the model has a name.
        object_ref: u32,
        /// The name, when the object namespace has one.
        resource: Option<ResourceId>,
    },
    /// The guest has given an interval of this task's GPU virtual address
    /// space pages, and says so after it has already applied the change to its
    /// own page table.
    ///
    /// **It names an address interval and no object.** The packet carries a
    /// task, a 64-bit base and a 64-bit length, and the closure ledger reads
    /// its opcode as establishing *a task GPU-VA mapping over guest pages* —
    /// there is no object ref in it to resolve. A model that took it for a
    /// per-object mapping would be asserting something the wire never says,
    /// and the resource it named would be whichever one shared the low half of
    /// an address.
    MapMemory {
        task: TaskId,
        span: GuestSpan,
    },
    /// The same interval, taken away. Arrives after the guest has removed the
    /// translations, so the addresses in it no longer resolve.
    UnmapMemory {
        task: TaskId,
        span: GuestSpan,
    },
    /// Re-point a resource at different guest pages.
    ReplacePhysical {
        task: TaskId,
        resource: ResourceId,
        backing: BackingId,
        extent: ByteRange,
    },
    Invalidate {
        task: TaskId,
        resources: Vec<ResourceId>,
    },
    Synchronize {
        task: TaskId,
        resources: Vec<ResourceId>,
    },
    SynchronizeAndDiscard {
        task: TaskId,
        resources: Vec<ResourceId>,
    },
    Discard {
        task: TaskId,
        resources: Vec<ResourceId>,
    },
    /// Retire a backing and the resources that named it.
    /// The guest is retiring one surface's host backing, and with it every
    /// resource that named those bytes.
    ///
    /// # The backing is optional for the reason a delete's name is
    ///
    /// The record's first word is a **mapping** id, and a mapping that no
    /// longer resolves is one this model never had a backing for — so there is
    /// nothing here to retire, and there never was. Refusing on it threw away
    /// the packet, which is the same wrong answer
    /// [`Self::DeleteResource::resource`] records: a driven macos-15 boot
    /// refused one a boot that way, and the rail's own mapping-keyed teardown
    /// was the half with work left.
    ///
    /// The mapping travels beside the backing so that half has what it needs
    /// whether or not this model can name anything.
    DeleteBacking {
        task: TaskId,
        /// The guest's own mapping id, which the rail's teardown is keyed by.
        mapping: crate::identity::MappingId,
        /// The bytes, when the mapping still resolves to some.
        backing: Option<BackingId>,
    },
    /// End one serializer object: the name stops resolving, and the rail drops
    /// whatever it holds for that kind.
    ///
    /// # Why the kind travels and `DeleteResource` was not reused
    ///
    /// The name half is identical — a slot stops resolving, and
    /// [`crate::namespace`] owns that either way. What is not identical is what
    /// the executor owes: a sampler destroy drops a task-local sampler
    /// registry entry, a fence destroy drops the generations a wait is decided
    /// against, and a function destroy drops nothing at all because that kind
    /// is held by content. An operation that lost the kind would leave the
    /// executor to re-derive it from bytes the model no longer carries.
    ///
    /// They are also two different commands on the wire — `0x25` names
    /// `{task, object}` in the packet and this one carries a record — so
    /// folding them would be one kind with two payload layouts, which
    /// [`LifecycleKind::of`] could not tell apart.
    DeleteObject {
        task: TaskId,
        /// The guest's own number, which is what every rail's retirement is
        /// keyed by — `(task, ref)`, not a name.
        ///
        /// Carried beside the name rather than derived from it because for one
        /// measured kind there *is* no name: a fence comes into being when an
        /// exec stream first signals it, so no path reads a fence out of the
        /// guest's object list and the namespace has nothing to give.
        object_ref: u32,
        /// The name, when the object namespace has one.
        ///
        /// # `None` is an answer, not a failure
        ///
        /// This used to be a `ResourceId` and an unresolvable reference
        /// refused the whole packet. That is wrong for a *destroy*, and a
        /// driven boot priced it: seven a boot arrive with neither a name in
        /// the model nor a live entry in the guest's list —
        /// `delete_unnamed_and_unresolvable`, of which two are fences. Refusing
        /// those loses the retirement, and for a fence that is a wrong answer
        /// rather than a dropped one: `retire_fence` clears the stored
        /// generation, and a generation outliving its fence makes the next
        /// fence to reuse that reference start life already signalled.
        ///
        /// Nothing is lost by ordering it against nothing. A resource the
        /// namespace cannot name is one no other operation in this model can
        /// name either, so there is no access for this delete to conflict with
        /// — see [`Self::resources`], which returns an empty slice for exactly
        /// this case.
        resource: Option<ResourceId>,
        kind: reims_vgpu_protocol::destroy::DestroyKind,
    },
}

impl LifecycleOp {
    #[must_use]
    pub const fn kind(&self) -> LifecycleKind {
        match self {
            Self::DefineTask { .. } => LifecycleKind::DefineTask,
            Self::DeleteTask { .. } => LifecycleKind::DeleteTask,
            Self::BindObjectList { .. } | Self::CreateResource { .. } => {
                LifecycleKind::SetObjectList
            }
            Self::DeleteResource { .. } => LifecycleKind::DeleteResource,
            Self::MapMemory { .. } => LifecycleKind::MapMemory,
            Self::UnmapMemory { .. } => LifecycleKind::UnmapMemory,
            Self::ReplacePhysical { .. } => LifecycleKind::ReplacePhysical,
            Self::Invalidate { .. } => LifecycleKind::Invalidate,
            Self::Synchronize { .. } => LifecycleKind::Synchronize,
            Self::SynchronizeAndDiscard { .. } => LifecycleKind::SynchronizeAndDiscard,
            Self::Discard { .. } => LifecycleKind::Discard,
            Self::DeleteBacking { .. } => LifecycleKind::DeleteBacking,
            Self::DeleteObject { .. } => LifecycleKind::DeleteObject,
        }
    }

    /// The resources this operation names, in the order it names them.
    ///
    /// Empty is a claim about the *operation*, not about the transaction: a
    /// task teardown names no resource and still tears down everything in the
    /// task, and a map notice names an address interval. What it means is that
    /// the operation itself makes no per-resource statement, so nothing here
    /// constrains the transaction's access list.
    ///
    /// Non-empty is the opposite, and it is what
    /// [`crate::transaction::LifecyclePayload`] holds the envelope to: these
    /// resources *are* the operation's statement about what it touches, so an
    /// envelope naming a different set is a hazard edge built against the wrong
    /// memory.
    #[must_use]
    pub fn resources(&self) -> &[ResourceId] {
        match self {
            Self::Invalidate { resources, .. }
            | Self::Synchronize { resources, .. }
            | Self::SynchronizeAndDiscard { resources, .. }
            | Self::Discard { resources, .. } => resources,
            Self::ReplacePhysical { resource, .. } => std::slice::from_ref(resource),
            // Empty when the namespace has no name for it, which is the honest
            // answer rather than a missing one: a resource this model cannot
            // name is one nothing else in it names, so a delete of it conflicts
            // with nothing.
            Self::DeleteObject { resource, .. } | Self::DeleteResource { resource, .. } => {
                resource.as_slice()
            }
            // A task's own lifetime, an address interval, a slot declaration
            // and a backing retirement each name something that is not a
            // resource id.
            Self::DefineTask { .. }
            | Self::DeleteTask { .. }
            | Self::BindObjectList { .. }
            | Self::CreateResource { .. }
            | Self::MapMemory { .. }
            | Self::UnmapMemory { .. }
            | Self::DeleteBacking { .. } => &[],
        }
    }

    /// What every resource this operation names is subject to, or `None` when
    /// it names none.
    ///
    /// # Why a mode per operation and not per resource
    ///
    /// A lifecycle command's resource list is a list of *operands of one verb*:
    /// a synchronise synchronises all of them, a discard discards all of them.
    /// Nothing on this wire distinguishes the operands from one another, so a
    /// per-resource mode would be a field with one value repeated, and the first
    /// caller to fill it differently would be inventing a distinction the packet
    /// does not carry.
    ///
    /// # `Unknown` is the answer for three of them, deliberately
    ///
    /// [`AccessMode::Unknown`] exists so that "conservative because nobody
    /// knows" is countable and therefore narrowable, and that is exactly what
    /// three of these are. It orders identically to `ReadWrite`, so nothing is
    /// under-ordered by saying it; what it buys is that a census can say how
    /// many of this device's lifecycle edges are bought with imprecision, and a
    /// later closure can show the number falling.
    ///
    /// The split, and what each side rests on:
    ///
    /// * `Invalidate`, `Synchronize` and `SynchronizeAndDiscard` are
    ///   [`AccessMode::Write`], and that is established rather than assumed. A
    ///   synchronise stores rendered content **into the resource's own guest
    ///   pages**, and an invalidate is the guest declaring it CPU-authored those
    ///   pages — content authority moving, outside the GPU timeline. Both
    ///   produce content on the memory the operation names; neither reads it.
    /// * `Discard` releases the transfer staging of the resources it names while
    ///   preserving their identity and host texture. Which memory that is a
    ///   statement about is *not* established: the resource's guest bytes are
    ///   not what it touches, and the model has no key for a transfer staging.
    ///   So the direction is unestablished over the memory the access is keyed
    ///   on, which is what `Unknown` says.
    /// * `DeleteResource` and `ReplacePhysical` end or re-point the storage the
    ///   name answered for. [`crate::namespace`] already owns the *name* half —
    ///   deletion stops resolution, teardown waits for the last accepted use —
    ///   so what is left here is only the hazard edge against work still
    ///   touching those bytes. That edge must exist in both directions and the
    ///   operation reads nothing, so neither `Read` nor `Write` is the term;
    ///   `Unknown` orders it and says the term is open.
    ///
    /// # It is `None` exactly when [`Self::resources`] is empty
    ///
    /// Same arms, same file, one screen apart, and
    /// `an_operation_declares_an_access_exactly_when_it_names_resources` holds
    /// them to it — because a mode without a resource orders against memory the
    /// operation never named, and a resource without a mode is a packet
    /// [`crate::transaction::LifecyclePayload`] refuses.
    #[must_use]
    pub const fn declared_access(&self) -> Option<AccessMode> {
        match self {
            Self::Invalidate { .. }
            | Self::Synchronize { .. }
            | Self::SynchronizeAndDiscard { .. } => Some(AccessMode::Write),
            Self::Discard { .. }
            | Self::DeleteResource { .. }
            | Self::ReplacePhysical { .. }
            | Self::DeleteObject { .. } => Some(AccessMode::Unknown),
            // Each names something that is not a resource, so there is no
            // per-resource claim for a mode to be about.
            Self::DefineTask { .. }
            | Self::DeleteTask { .. }
            | Self::BindObjectList { .. }
            | Self::CreateResource { .. }
            | Self::MapMemory { .. }
            | Self::UnmapMemory { .. }
            | Self::DeleteBacking { .. } => None,
        }
    }

    #[must_use]
    pub const fn task(&self) -> TaskId {
        match self {
            Self::DefineTask { task, .. }
            | Self::DeleteTask { task }
            | Self::BindObjectList { task, .. }
            | Self::CreateResource { task, .. }
            | Self::DeleteResource { task, .. }
            | Self::MapMemory { task, .. }
            | Self::UnmapMemory { task, .. }
            | Self::ReplacePhysical { task, .. }
            | Self::Invalidate { task, .. }
            | Self::Synchronize { task, .. }
            | Self::SynchronizeAndDiscard { task, .. }
            | Self::Discard { task, .. }
            | Self::DeleteBacking { task, .. }
            | Self::DeleteObject { task, .. } => *task,
        }
    }
}

/// A copy the device offered to drop and did not, with the reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a declined discard nobody reports is guest work the device dropped in silence"]
pub struct Declined {
    pub resource: ResourceId,
    pub backing: BackingId,
    /// Bytes that exist in no other replica. Dropping them would not free a
    /// spare copy; it would destroy content the guest may still read.
    pub sole_authority_bytes: u64,
}

/// A content-authority change owed to a transaction's completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeferredDiscard {
    pub resource: ResourceId,
    pub backing: BackingId,
    pub bytes: ByteRange,
}

/// What a lifecycle transaction obliges and leaves behind.
#[derive(Clone, Debug, Default, PartialEq)]
#[must_use = "the transfers a synchronise owes and the storage a delete freed are both obligations"]
pub struct Effects {
    /// Copies that must have executed before the transaction's completion
    /// stamp may publish.
    pub transfers: Vec<Transfer>,
    /// What the namespace said about each backing detached by this
    /// transaction.
    pub teardowns: Vec<Teardown>,
    /// Heap storage whose last allocation went with this transaction.
    pub storage_freed: Vec<BackingId>,
    /// Copies offered for release, evaluated when the transfers above have
    /// executed. See the module docs for why this is not immediate.
    pub at_completion: Vec<DeferredDiscard>,
    /// Address intervals whose translations the guest has changed under this
    /// device.
    pub remapped: Vec<Remap>,
    /// Task address spaces this transaction replaced under a live id.
    pub redefined: Vec<Redefinition>,
}

impl Effects {
    /// Take everything `other` obliges into this one.
    ///
    /// Written as a destructure rather than a list of `extend` calls so that a
    /// new obligation field is a compile error here instead of a silent drop.
    /// The callers are the two commands that are lists of smaller deletes ---
    /// [`Lifecycle::delete_task`] and [`Lifecycle::delete_backing`] --- and
    /// each was reaching into two of the six fields by name. That is correct
    /// only for as long as the delete they call fills exactly those two, which
    /// is not a fact either call site can see, and the failure it produces is
    /// an obligation that was owed and then discarded on the way out. That is
    /// the loss this type's `#[must_use]` exists to make impossible, so the
    /// merge is where it has to be caught.
    fn absorb(&mut self, other: Self) {
        let Self {
            transfers,
            teardowns,
            storage_freed,
            at_completion,
            remapped,
            redefined,
        } = other;
        self.transfers.extend(transfers);
        self.teardowns.extend(teardowns);
        self.storage_freed.extend(storage_freed);
        self.at_completion.extend(at_completion);
        self.remapped.extend(remapped);
        self.redefined.extend(redefined);
    }
}

/// A live task redefined: the whole address space replaced under one id.
///
/// An obligation of the same kind [`Remap`] is, and a wider one. A remap moves
/// an interval; this replaces the table every interval was resolved through, so
/// *every* resolution held for the task is answered by the wrong pages
/// afterwards. The teardowns the redefinition performed travel in the same
/// [`Effects`] — nothing is orphaned — and this is what says the survivors'
/// cached addresses are not survivors.
///
/// `root_moved` is the distinction that decides how bad it is. A redefinition
/// at the same root re-declares the space the task already had, and what the
/// guest published into it is still there; one at a different root means the
/// object list itself is now a different page, and everything published into
/// the old one reads back as whatever the new pages hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Redefinition {
    pub task: TaskId,
    pub previous: DirectoryFrame,
    pub directory: DirectoryFrame,
}

impl Redefinition {
    /// Whether the object list itself is now a different page.
    ///
    /// Derived rather than carried: it is exactly `previous != directory`, and
    /// a field holding the same answer is a second copy that a construction
    /// site can set the other way. There is one source, and it is the pair of
    /// frames this already reports.
    #[must_use]
    pub const fn root_moved(&self) -> bool {
        // `PartialEq` is not `const`, and a `DirectoryFrame` is one `u32`.
        self.previous.0 != self.directory.0
    }
}

/// An interval of a task's address space whose translations have changed.
///
/// An obligation and not a record: every resolution held over the interval was
/// computed against pages the guest has since moved, so a cache that keeps one
/// answers a later question with the wrong pages. This crate caches none, which
/// is exactly why the obligation has to leave it named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Remap {
    pub task: TaskId,
    pub span: GuestSpan,
    /// Whether the interval has pages behind it now.
    ///
    /// It does not decide the invalidation — both directions invalidate, since
    /// the pages under a re-established mapping are new ones. It decides what
    /// may be read *after*: an unmapped interval has no translation at all, and
    /// a reader that treated the two alike would follow one the guest removed.
    pub established: bool,
}

/// Why a lifecycle packet's bytes did not become an operation.
///
/// Separate from [`Refusal`], which is why a well-formed operation could not be
/// *applied*. A payload that cannot be read and a task that does not exist are
/// different failures with different fixes, and folding them would make the log
/// unable to say which one happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveRefusal {
    /// The command carries no counted resource list, so there is nothing for
    /// [`resource_list`] to read. Not a malformed packet — a caller asking the
    /// wrong question.
    NotAResourceList { kind: LifecycleKind },
    /// The payload disagrees with itself: too short for its header, or
    /// declaring more records than it carries.
    Payload(fifo::ResourceListDecodeError),
    /// A ref in the list names no live object.
    ///
    /// Carries the guest's number, because the number is what a log line has to
    /// contain for anyone to find which object the guest thought it had.
    UnknownRef { object_ref: u32 },
    /// The command carries no `{task, object}` record, so there is nothing for
    /// [`object_reference`] to read.
    NotAnObjectReference { kind: LifecycleKind },
    /// The record resolved, and the operation still cannot be built: it names
    /// the pages behind the object, and a [`crate::resolve::RefResolver`]
    /// answers identity only.
    ///
    /// `ReplacePhysical` is the case. Its packet is a bare `{task, object}` —
    /// the guest re-points a resource at host frames it has already wired, at
    /// the *same* GPU-VA — so the new backing and extent are not on the wire at
    /// all. They come from whatever holds the object's storage, which is a
    /// registry this crate does not have. Named rather than approximated,
    /// because an operation carrying the *old* backing would re-point nothing
    /// while reporting success.
    ///
    /// **It carries the half that did resolve.** The task and the resource were
    /// settled before this refusal was reached — the ref is looked up in its own
    /// task's namespace and a dead one refuses as [`Self::UnknownRef`] — so a
    /// caller that can supply the storage has everything else already, and does
    /// not have to re-decode the packet or resolve the ref a second time to use
    /// it. That matters because the second resolution is the one that could
    /// disagree: a caller pairing *its* idea of which resource the packet names
    /// with the model's storage would build an operation about a resource the
    /// packet does not name, and nothing downstream could tell.
    NeedsStorage {
        kind: LifecycleKind,
        task: TaskId,
        resource: ResourceId,
    },
    /// The command is not a backing retirement, so there is nothing for
    /// [`backing_retirement`] to read.
    NotABackingRetirement { kind: LifecycleKind },
    /// The command is not a serializer destroy, so there is nothing for
    /// [`object_destroy`] to read.
    NotAnObjectDestroy { kind: LifecycleKind },
    /// The bytes inside a destroy command are not a destroy record.
    ///
    /// Carried whole from the layer that judged them, which already tells a
    /// number in the destroy span that nothing has measured from a record of
    /// some other family entirely.
    Destroy(reims_vgpu_protocol::destroy::DestroyRefusal),
    /// The record is a destroy of a kind whose ledger row is not settled.
    ///
    /// Five of the eleven kinds have never been observed on a driven boot, and
    /// acting on one would be acting on no evidence. The refusal is per
    /// *packet* rather than per class, which is the difference between holding
    /// five unobserved kinds out of the model and holding 5.5% of a real
    /// guest's stream out of it.
    UnsettledDestroy {
        kind: reims_vgpu_protocol::destroy::DestroyKind,
    },
    /// The command is not a map-or-unmap notice, so there is no interval for
    /// [`map_notice`] to read. [`NotAResourceList`](Self::NotAResourceList)'s
    /// sibling, and a caller asking the wrong question rather than a malformed
    /// packet.
    NotAMapNotice { kind: LifecycleKind },
    /// The command is neither task definition nor task deletion, so there is
    /// nothing for [`task_lifetime`] to read.
    ///
    /// The last of the family to get its own name. This join answered with
    /// [`NotAnObjectReference`](Self::NotAnObjectReference), so a caller that
    /// asked the wrong question here reached the failure channel under the
    /// name of a *different* join and the reader went to look at the
    /// `{task, object}` record, which the packet does not carry either.
    NotATaskLifetime { kind: LifecycleKind },
    /// The notice's payload cannot hold its three fields.
    ShortNotice(fifo::ShortPayload),
    /// An `Invalidate` record asks for a content-authority transition this
    /// device has no established meaning for.
    ///
    /// The record's four validity-op bytes are independent on the wire, and
    /// exactly one combination has been established: clear hostValid together
    /// with set guestValid, which is the guest saying it CPU-wrote the
    /// resource. Anything else is a transition no capture has shown, and the
    /// model has two honest answers — recover it, or refuse. Applying the
    /// guest-write transition regardless is the third answer and it is a wrong
    /// one: it moves authority the guest did not move, which is a stale draw or
    /// a lost CPU write with a clean log.
    ///
    /// Carries the guest's raw dword, because the whole point of the refusal is
    /// that someone reading the log can see which combination arrived.
    UnestablishedValidityOps { object_ref: u32, ops: u32 },
}

impl ResolveRefusal {
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::NotAResourceList { .. } => "lifecycle_not_a_resource_list",
            Self::Payload(inner) => inner.slug(),
            Self::UnknownRef { .. } => "lifecycle_unknown_ref",
            Self::NotAnObjectReference { .. } => "lifecycle_not_an_object_reference",
            Self::NeedsStorage { .. } => "lifecycle_needs_storage",
            Self::NotABackingRetirement { .. } => "lifecycle_not_a_backing_retirement",
            Self::NotAMapNotice { .. } => "lifecycle_not_a_map_notice",
            Self::NotATaskLifetime { .. } => "lifecycle_not_a_task_lifetime",
            Self::ShortNotice(_) => fifo::ShortPayload::SLUG,
            Self::UnestablishedValidityOps { .. } => "lifecycle_unestablished_validity_ops",
            Self::NotAnObjectDestroy { .. } => "lifecycle_not_an_object_destroy",
            Self::Destroy(inner) => inner.reason(),
            Self::UnsettledDestroy { .. } => "lifecycle_unsettled_destroy",
        }
    }
}

/// Turn one resource-list packet's payload into the operation it names.
///
/// **The link between "the wire has a resource list" and "the model has a
/// lifecycle operation".** [`reims_vgpu_protocol::fifo`] could read the list
/// and [`LifecycleOp`] could describe the result, and nothing joined them: the
/// only code that did was the device's drain, which acted on the decoded
/// records directly and produced no operation at all.
///
/// Every ref resolves or the whole packet refuses. A partial list is not an
/// option here — `Invalidate` says a set of resources went stale together, and
/// applying it to the subset that happened to resolve claims the others are
/// still fresh.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind with no list, a payload that disagrees with
/// itself, or a ref naming nothing live.
pub fn resource_list(
    kind: LifecycleKind,
    payload: &[u8],
    namespaces: &(impl crate::resolve::TaskNamespaces + ?Sized),
) -> Result<LifecycleOp, ResolveRefusal> {
    let Some(record_len) = kind.resource_list_record_len() else {
        return Err(ResolveRefusal::NotAResourceList { kind });
    };
    // Which decoder is which record length, asked once. The eight-byte record
    // carries the guest's validity quad; the four-byte one is object refs
    // alone. `resource_list_record_len` is the only thing that decides.
    let (task, refs) = if record_len == fifo::CHILD_INVALIDATE_RECORD_LEN {
        let cmd = fifo::decode_invalidate_resources(payload).map_err(ResolveRefusal::Payload)?;
        // The quad beside each ref, checked rather than dropped. This is the
        // one list command whose records carry a transition of their own, and
        // the operation below states one transition for the whole packet — so
        // a record asking for a different one has to refuse here, or the
        // packet applies an authority move the guest did not ask for.
        let mut refs = Vec::with_capacity(cmd.records.len());
        for record in &cmd.records {
            if !record.ops.is_guest_write() {
                return Err(ResolveRefusal::UnestablishedValidityOps {
                    object_ref: record.object_id,
                    ops: record.flags,
                });
            }
            refs.push(record.object_id);
        }
        (cmd.task_id, refs)
    } else {
        let cmd = fifo::decode_synchronize_resources(payload).map_err(ResolveRefusal::Payload)?;
        (cmd.task_id, cmd.object_ids)
    };

    // The namespace is bound to the task this packet just named, not to one a
    // caller chose. Refs are per-task slot numbers, so another task's namespace
    // would not refuse — it would resolve, and the operation would name
    // somebody else's resources.
    let task = TaskId(task);
    let resolver = crate::resolve::InTask::new(namespaces, task);
    let mut resources = Vec::with_capacity(refs.len());
    for object_ref in refs {
        let resolved = crate::resolve::RefResolver::resource(&resolver, object_ref)
            .ok_or(ResolveRefusal::UnknownRef { object_ref })?;
        resources.push(resolved);
    }

    Ok(match kind {
        LifecycleKind::Invalidate => LifecycleOp::Invalidate { task, resources },
        LifecycleKind::Synchronize => LifecycleOp::Synchronize { task, resources },
        LifecycleKind::SynchronizeAndDiscard => {
            LifecycleOp::SynchronizeAndDiscard { task, resources }
        }
        LifecycleKind::Discard => LifecycleOp::Discard { task, resources },
        // Unreachable: `resource_list_record_len` answered `Some` for exactly
        // the four above, and a `None` returned before this point. Stated as a
        // refusal rather than a panic because the two functions are separate,
        // and a fifth kind gaining a record length without gaining an arm here
        // must be a caller-visible refusal rather than a lost packet.
        other => return Err(ResolveRefusal::NotAResourceList { kind: other }),
    })
}

/// The content-authority moves an EXEC packet declares in its own resource
/// table.
///
/// **A GPU-work packet carries a lifecycle statement, and it is the same
/// statement.** `EXEC_INDIRECT2` writes one 24-byte record per live object-list
/// entry, and the four bytes after each object ref are byte-identical to a
/// `CmdInvalidateResources` record's validity quad. The guest says "I CPU-wrote
/// this resource" here, with the submission that reads it, exactly as often as
/// it says it in a standalone packet — and if the two produced different
/// operations there would be two content-authority models, one of which is
/// reached only by whichever packet the guest happened to use.
///
/// So this returns the same [`LifecycleOp::Invalidate`], and the caller applies
/// it before the transaction's work reads anything. The task is the header's,
/// not a caller's guess: a table of object refs with someone else's namespace
/// resolves to someone else's resources.
///
/// **A zero quad is the normal record here and it is not a refusal.** Unlike a
/// standalone invalidate, whose whole purpose is to move authority, this table
/// lists every resource the submission touches — and most of them were not
/// CPU-written. A record asking for nothing contributes nothing, so an EXEC
/// that wrote none produces an operation over no resources.
///
/// # Errors
///
/// [`ResolveRefusal::ShortNotice`] for a payload too short for the header,
/// [`ResolveRefusal::Payload`] where the declared table is not there,
/// [`ResolveRefusal::UnestablishedValidityOps`] for a quad that is neither the
/// guest-write move nor nothing, and [`ResolveRefusal::UnknownRef`] for a ref
/// naming nothing live.
pub fn exec_resource_table(
    payload: &[u8],
    namespaces: &(impl crate::resolve::TaskNamespaces + ?Sized),
) -> Result<LifecycleOp, ResolveRefusal> {
    let header = fifo::decode_exec_header(payload).map_err(ResolveRefusal::ShortNotice)?;
    let table = fifo::decode_exec_resource_table(payload).map_err(ResolveRefusal::Payload)?;
    // The header's task, bound here rather than chosen by the caller — which is
    // what this function's own contract says and what its signature used to
    // leave to discipline.
    let task = TaskId(header.task_id);
    let resolver = crate::resolve::InTask::new(namespaces, task);
    let mut resources = Vec::new();
    for record in &table {
        if record.ops == fifo::InvalidateValidityOps::default() {
            continue;
        }
        if !record.ops.is_guest_write() {
            return Err(ResolveRefusal::UnestablishedValidityOps {
                object_ref: record.object_id,
                ops: record.ops.to_le_dword(),
            });
        }
        resources.push(
            crate::resolve::RefResolver::resource(&resolver, record.object_id).ok_or(
                ResolveRefusal::UnknownRef {
                    object_ref: record.object_id,
                },
            )?,
        );
    }
    Ok(LifecycleOp::Invalidate { task, resources })
}

/// Turn any lifecycle packet's payload into the operation it names.
///
/// **The one place a kind picks its join.** The five joins below each refuse
/// every kind that is not their own, which makes a wrong pairing a value rather
/// than a misread — but nothing said which one a given kind belongs to, so every
/// caller with a packet had to know, and a caller that knew wrongly would read
/// one command's offsets out of another's payload. `every_lifecycle_kind_
/// reaches_at_most_one_join` could see that no kind is read by two; it could not
/// make the choice for anyone.
///
/// The match is exhaustive over [`LifecycleKind`], so a thirteenth command is a
/// compile error here rather than a packet that quietly reaches whichever arm
/// came last.
///
/// Both resolvers are taken because the twelve commands need both namespaces
/// and they are different namespaces — an object-list ref and a mapping id
/// arrive as `u32`s that overlap numerically and name unrelated things, which is
/// why [`crate::resolve::RefResolver`] and
/// [`crate::resolve::MappingResolver`] are two traits.
///
/// The object side is a [`crate::resolve::TaskNamespaces`] and not one
/// namespace: which task's namespace a lifecycle packet resolves in is stated in
/// the packet, and only the join that decoded it knows. See that trait for why a
/// caller-chosen namespace is a wrong answer rather than a refusal.
///
/// # Errors
///
/// [`ResolveRefusal`] from the join that owns the kind, or — for the one
/// command that resolves and still cannot become an operation —
/// [`ResolveRefusal::NeedsStorage`], which names what is missing rather than
/// approximating it.
pub fn operation(
    kind: LifecycleKind,
    payload: &[u8],
    objects: &(impl crate::resolve::TaskNamespaces + ?Sized),
    mappings: &(impl crate::resolve::MappingResolver + ?Sized),
) -> Result<LifecycleOp, ResolveRefusal> {
    match kind {
        LifecycleKind::DefineTask | LifecycleKind::DeleteTask => task_lifetime(kind, payload),
        LifecycleKind::DeleteResource | LifecycleKind::ReplacePhysical => {
            object_reference(kind, payload, objects)
        }
        LifecycleKind::MapMemory | LifecycleKind::UnmapMemory => map_notice(kind, payload),
        LifecycleKind::DeleteBacking => backing_retirement(kind, payload, mappings),
        LifecycleKind::Invalidate
        | LifecycleKind::Synchronize
        | LifecycleKind::SynchronizeAndDiscard
        | LifecycleKind::Discard => resource_list(kind, payload, objects),
        LifecycleKind::SetObjectList => object_list_bind(kind, payload),
        LifecycleKind::DeleteObject => object_destroy(kind, payload, objects),
    }
}

/// Which task a lifecycle packet names, without resolving anything in it.
///
/// **The half of [`operation`] that needs no namespace, and the term
/// [`Refusal::NoSuchTask`] is decided on.** Every one of the twelve commands
/// carries its task in its own record, and eight of them carry nothing else
/// that has to resolve — so "would the lifecycle owner hold this task" is
/// answerable from the packet's bytes alone, before the owner is production
/// state and before any ref is looked up. That is what a device measuring the
/// group's cutover floor needs: an operation it cannot build yet still names a
/// task, and a task the model does not hold is a refusal of the whole packet
/// whatever the refs do.
///
/// It is not a second decode table. Each arm calls the same
/// [`reims_vgpu_protocol::fifo`] decoder [`operation`]'s join for that kind
/// calls, so the offsets have one owner; what is dropped is the resolution, not
/// the layout. `task_named_agrees_with_the_operation_it_would_build` holds the
/// two to the same answer over every kind.
///
/// # Errors
///
/// [`ResolveRefusal::ShortNotice`] or [`ResolveRefusal::Payload`] when the
/// payload is too short to hold the record its kind names. There is no
/// resolver-shaped refusal here, which is the point.
pub fn task_named(kind: LifecycleKind, payload: &[u8]) -> Result<TaskId, ResolveRefusal> {
    let id = match kind {
        LifecycleKind::DefineTask => {
            fifo::decode_define_task(payload)
                .map_err(ResolveRefusal::ShortNotice)?
                .id
                .task_id
        }
        LifecycleKind::DeleteTask => {
            fifo::decode_delete_task(payload).map_err(ResolveRefusal::ShortNotice)?
        }
        LifecycleKind::DeleteResource | LifecycleKind::ReplacePhysical => {
            fifo::decode_task_object(payload)
                .map_err(ResolveRefusal::ShortNotice)?
                .task_id
        }
        LifecycleKind::MapMemory | LifecycleKind::UnmapMemory => {
            fifo::decode_map_memory(payload)
                .map_err(ResolveRefusal::ShortNotice)?
                .task_id
        }
        LifecycleKind::DeleteBacking => {
            fifo::decode_delete_backing(payload)
                .map_err(ResolveRefusal::ShortNotice)?
                .task_id
        }
        LifecycleKind::SetObjectList => {
            fifo::decode_set_object_list(payload)
                .map_err(ResolveRefusal::ShortNotice)?
                .task_id
        }
        // The task word is the command's, ahead of a record whose framing this
        // question does not need: a destroy that is too short to hold a record
        // still names the task it would have acted in.
        LifecycleKind::DeleteObject => {
            fifo::decode_delete_object(payload)
                .map_err(|error| ResolveRefusal::ShortNotice(error.short()))?
                .task_id
        }
        // The two record lengths, told apart by the same question
        // [`resource_list`] asks: the eight-byte record is the validity quad's
        // and the four-byte one is refs alone, and the task word is not at the
        // same place in the two headers.
        LifecycleKind::Invalidate => {
            fifo::decode_invalidate_resources(payload)
                .map_err(ResolveRefusal::Payload)?
                .task_id
        }
        LifecycleKind::Synchronize
        | LifecycleKind::SynchronizeAndDiscard
        | LifecycleKind::Discard => {
            fifo::decode_synchronize_resources(payload)
                .map_err(ResolveRefusal::Payload)?
                .task_id
        }
    };
    Ok(TaskId(id))
}

/// Turn a `SetObjectList` packet's payload into the binding it names.
///
/// Takes no resolver, as [`task_lifetime`] does not: the packet carries a task
/// id, a page frame and a count, and nothing about any of the three is a name
/// device state could answer differently.
///
/// **This used to be a `NeedsGuestTable` refusal**, on the reading that
/// the operation was the per-entry result of walking the table. See
/// [`LifecycleOp::BindObjectList`] for the measurement that overturned it: the
/// count is the table's capacity, a driven boot declares 1 048 576 of them
/// eighteen times, and about twelve hundred slots are ever named. The walk was
/// never the operation.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind that is not `SetObjectList`, or a payload too
/// short for the three words. A short one is refused rather than defaulted,
/// because a zeroed page frame is a valid frame number and a zeroed task id is
/// the kernel task.
pub fn object_list_bind(
    kind: LifecycleKind,
    payload: &[u8],
) -> Result<LifecycleOp, ResolveRefusal> {
    if kind != LifecycleKind::SetObjectList {
        return Err(ResolveRefusal::NotATaskLifetime { kind });
    }
    let command = fifo::decode_set_object_list(payload).map_err(ResolveRefusal::ShortNotice)?;
    Ok(LifecycleOp::BindObjectList {
        task: TaskId(command.task_id),
        list: ObjectList {
            page: DirectoryFrame(command.pfn),
            capacity: command.count,
        },
    })
}

/// Turn a task-lifetime packet's payload into the operation it names.
///
/// Reached through [`operation`], which is what decides that this is the join
/// a task-lifetime kind belongs to.
///
/// Takes no resolver, as [`map_notice`] does not: a task is named by its own
/// slot number and there is nothing about that number device state could answer
/// differently.
///
/// The two commands do **not** carry the id the same way. A definition's first
/// word is `(task_id << 1) | is_kernel_task`; a deletion's is a plain slot.
/// Reading either at the other's convention indexes every slot at twice or half
/// its number, which is why the shift lives in
/// [`reims_vgpu_protocol::fifo::DefineTaskId`] and not at a call site.
///
/// The definition's address-space extent and page-table directory are not
/// carried into the operation. They are the address-resolution owner's, this
/// crate resolves no address, and an operation holding a page frame it never
/// reads is a field that goes stale unnoticed.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind that is neither of the two, or a payload too
/// short. A short deletion is refused rather than defaulted, because slot `0`
/// is the kernel task.
pub fn task_lifetime(kind: LifecycleKind, payload: &[u8]) -> Result<LifecycleOp, ResolveRefusal> {
    match kind {
        LifecycleKind::DefineTask => {
            let command = fifo::decode_define_task(payload).map_err(ResolveRefusal::ShortNotice)?;
            Ok(LifecycleOp::DefineTask {
                task: TaskId(command.id.task_id),
                kernel: command.id.kernel,
                directory: DirectoryFrame(command.directory_pfn),
            })
        }
        LifecycleKind::DeleteTask => {
            let task = fifo::decode_delete_task(payload).map_err(ResolveRefusal::ShortNotice)?;
            Ok(LifecycleOp::DeleteTask { task: TaskId(task) })
        }
        other => Err(ResolveRefusal::NotATaskLifetime { kind: other }),
    }
}

/// Turn a `{task, object}` packet's payload into the operation it names.
///
/// Two commands carry that record. Only one of them becomes an operation from
/// it: a delete names what to retire and nothing else, while a re-point names
/// pages that are not on the wire — see [`ResolveRefusal::NeedsStorage`].
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind carrying no such record, a payload too short, a
/// ref naming nothing live, or a kind whose operation needs storage this
/// resolver cannot name.
pub fn object_reference(
    kind: LifecycleKind,
    payload: &[u8],
    namespaces: &(impl crate::resolve::TaskNamespaces + ?Sized),
) -> Result<LifecycleOp, ResolveRefusal> {
    if !matches!(
        kind,
        LifecycleKind::DeleteResource | LifecycleKind::ReplacePhysical
    ) {
        return Err(ResolveRefusal::NotAnObjectReference { kind });
    }
    let command = fifo::decode_task_object(payload).map_err(ResolveRefusal::ShortNotice)?;
    // The record's own task names the namespace its object ref is a slot in.
    let task = TaskId(command.task_id);
    let resolver = crate::resolve::InTask::new(namespaces, task);
    // The ref is resolved before the kind is judged, so a re-point naming a
    // dead object refuses as a dead object rather than as unfinished work here.
    // The two kinds differ in what an unresolvable one *means*, which is why the
    // refusal is each arm's and not this line's: a delete whose slot the guest
    // already cleared is ordinary, and a re-point with no resource to re-point
    // is not a re-point.
    let resource = crate::resolve::RefResolver::resource(&resolver, command.object_id);
    match kind {
        LifecycleKind::DeleteResource => Ok(LifecycleOp::DeleteResource {
            task,
            object_ref: command.object_id,
            resource,
        }),
        // `ReplacePhysical`, and nothing else reaches here. The two terms it
        // did resolve travel with the refusal; see `NeedsStorage`.
        other => Err(match resource {
            Some(resource) => ResolveRefusal::NeedsStorage {
                kind: other,
                task,
                resource,
            },
            None => ResolveRefusal::UnknownRef {
                object_ref: command.object_id,
            },
        }),
    }
}

/// Turn a serializer destroy packet's payload into the operation it names.
///
/// # The kind is in the record and the task is in the command
///
/// `CmdDeleteObject`'s payload is a task word followed by one self-describing
/// destroy record, so this is the one lifetime join that reads two layers:
/// [`reims_vgpu_protocol::fifo::decode_delete_object`] frames the record inside
/// the payload, and [`reims_vgpu_protocol::destroy::decode`] says which of the
/// eleven kinds it is and which reference it names.
///
/// # An unsettled kind refuses, and that is what makes the row affordable
///
/// The eleven kinds do not share a closure. Six are settled on their own
/// serializer ledger rows; five — buffer, texture, heap, rasterization rate map
/// and indirect command buffer — have never been observed on any driven boot
/// and are unresolved. Refusing the whole *packet class* on their account was
/// what held 5.5% of a real guest's stream out of the model; refusing the
/// individual packet is the same honesty at 1/2000 the cost, and it costs the
/// channel nothing because a refused packet's completion word is
/// [`crate::session::SessionModel::admit`]'s caller's to stamp.
///
/// # The reference is an object-list name
///
/// Measured, not assumed. A sampler is constructed by looking this same number
/// up in the guest's object list and requiring a serializer-object entry there.
/// A driven boot finding that most destroys resolve to no *live* entry is a
/// fact about time, not about namespaces — the guest clears its slot before it
/// sends the destroy, and the name the device took at construction outlives the
/// slot. That is why this resolves through [`crate::resolve::TaskNamespaces`]
/// like every other object reference, and why it can.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind that is not a destroy command, a payload too
/// short for the record, a record that is not a destroy, or a kind the ledger
/// has not settled.
///
/// **Not** a reference naming nothing live. That used to refuse here and it
/// was wrong for a destroy: the effect is keyed by `(task, ref)` and the
/// reference is right there, so refusing loses a retirement this device can
/// still perform. See [`LifecycleOp::DeleteObject::resource`].
pub fn object_destroy(
    kind: LifecycleKind,
    payload: &[u8],
    namespaces: &(impl crate::resolve::TaskNamespaces + ?Sized),
) -> Result<LifecycleOp, ResolveRefusal> {
    if kind != LifecycleKind::DeleteObject {
        return Err(ResolveRefusal::NotAnObjectDestroy { kind });
    }
    let command = fifo::decode_delete_object(payload)
        .map_err(|error| ResolveRefusal::ShortNotice(error.short()))?;
    let record =
        reims_vgpu_protocol::destroy::decode(command.record).map_err(ResolveRefusal::Destroy)?;
    if !record.kind.settled() {
        return Err(ResolveRefusal::UnsettledDestroy { kind: record.kind });
    }
    let task = TaskId(command.task_id);
    let resolver = crate::resolve::InTask::new(namespaces, task);
    // Not `ok_or(UnknownRef)`. A destroy is the one lifecycle command whose
    // effect must not be lost to an unresolvable reference — see
    // [`LifecycleOp::DeleteObject::resource`] — and the reference itself is
    // carried so the executor's `(task, ref)`-keyed retirement runs either way.
    let resource = crate::resolve::RefResolver::resource(&resolver, record.object_ref);
    Ok(LifecycleOp::DeleteObject {
        task,
        object_ref: record.object_ref,
        resource,
        kind: record.kind,
    })
}

/// Turn a backing-retirement packet's payload into the operation it names.
///
/// The one join that needs a [`crate::resolve::MappingResolver`] rather than a
/// [`crate::resolve::RefResolver`]. Its record's first word is a **mapping**
/// id — the surface whose host backing is being retired — and mapping ids and
/// object-list refs are numerically overlapping namespaces for unrelated
/// things, so resolving this one through the object list would retire whatever
/// backing happened to share the integer.
///
/// The record is also the reverse of the `{task, object}` pair two other
/// commands carry: `{object, task}`. Nothing about the bytes distinguishes
/// them, which is why the offsets are
/// [`reims_vgpu_protocol::fifo::decode_delete_backing`]'s and not restated here.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind that is not a retirement, a payload too short, or
/// a mapping naming no live surface.
pub fn backing_retirement(
    kind: LifecycleKind,
    payload: &[u8],
    resolver: &(impl crate::resolve::MappingResolver + ?Sized),
) -> Result<LifecycleOp, ResolveRefusal> {
    if kind != LifecycleKind::DeleteBacking {
        return Err(ResolveRefusal::NotABackingRetirement { kind });
    }
    let command = fifo::decode_delete_backing(payload).map_err(ResolveRefusal::ShortNotice)?;
    let mapping = crate::identity::MappingId(command.object_id);
    // Not `ok_or(UnknownMapping)`. A retirement's effect must not be lost to a
    // mapping this model never held — see [`LifecycleOp::DeleteBacking`] — and
    // the mapping itself is carried so the executor's own teardown runs either
    // way.
    let backing = resolver.backing(mapping);
    Ok(LifecycleOp::DeleteBacking {
        task: TaskId(command.task_id),
        mapping,
        backing,
    })
}

/// Turn one map-or-unmap packet's payload into the operation it names.
///
/// **The join the resolver interface could not be the missing piece of.** Every
/// other lifecycle command that needed a byte-to-operation path needed a
/// [`crate::resolve::RefResolver`] to say which object a `u32` names; this pair
/// takes none, because the packet carries no ref. The interval it carries is
/// already total — a task, a base and a length — and there is nothing about it
/// that device state could answer differently.
///
/// # Errors
///
/// [`ResolveRefusal`]: a kind that is not one of the two, or a payload too
/// short to hold the interval. A short notice is refused rather than read with
/// a zero-filled tail: a length read short retires the wrong pages, and a base
/// read short names an address in the wrong half of the task's space.
pub fn map_notice(kind: LifecycleKind, payload: &[u8]) -> Result<LifecycleOp, ResolveRefusal> {
    if !matches!(kind, LifecycleKind::MapMemory | LifecycleKind::UnmapMemory) {
        return Err(ResolveRefusal::NotAMapNotice { kind });
    }
    let notice = fifo::decode_map_memory(payload).map_err(ResolveRefusal::ShortNotice)?;
    let task = TaskId(notice.task_id);
    let span = GuestSpan {
        base: notice.gva,
        length: notice.length,
    };
    Ok(match kind {
        LifecycleKind::UnmapMemory => LifecycleOp::UnmapMemory { task, span },
        // `MapMemory`, and every other kind returned above.
        _ => LifecycleOp::MapMemory { task, span },
    })
}

/// Why a lifecycle operation did not happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NoSuchTask {
        task: TaskId,
    },

    Namespace {
        task: TaskId,
        refusal: namespace::Refusal,
    },
    Heap {
        task: TaskId,
        refusal: heap::Refusal,
    },
    /// A heap-placed resource has no pages of its own, so there is nothing for
    /// a physical replacement to re-point. Re-pointing the heap's storage under
    /// it would move every other resource in the heap too.
    PlacedResourceHasNoPhysical {
        resource: ResourceId,
    },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoSuchTask { .. } => "lifecycle_no_such_task",
            Self::Namespace { .. } => "lifecycle_namespace",
            Self::Heap { .. } => "lifecycle_heap",
            Self::PlacedResourceHasNoPhysical { .. } => "lifecycle_placed_resource_has_no_physical",
        }
    }
}

/// One task's records, in one submission domain, as an [`AccessSource`].
///
/// A borrow of the owner rather than a copy of anything out of it: the content
/// authority reserves versions while the transaction is being built, so a
/// source that held a snapshot would hand two records of one packet the same
/// reservation.
pub struct TaskAccess<'a> {
    lifecycle: &'a mut Lifecycle,
    task: TaskId,
    domain: ChannelId,
}

impl AccessSource for TaskAccess<'_> {
    fn access(&mut self, participation: &Participation) -> Result<AccessIntent, AccessRefusal> {
        self.lifecycle
            .access(self.task, self.domain, participation)
            .map_err(|refusal| AccessRefusal {
                resource: participation.resource,
                reason: refusal.slug(),
            })
    }
}

/// Where a live resource's bytes are, derived once at creation.
#[derive(Clone, Copy, Debug)]
enum Resident {
    Dedicated {
        backing: BackingId,
        extent: ByteRange,
    },
    Placed(HeapPlacement),
}

impl Resident {
    const fn backing(self) -> BackingId {
        match self {
            Self::Dedicated { backing, .. } => backing,
            Self::Placed(p) => p.backing,
        }
    }

    const fn extent(self) -> ByteRange {
        match self {
            Self::Dedicated { extent, .. } => extent,
            Self::Placed(p) => p.region,
        }
    }

    /// A window of the resource, in its backing's coordinates.
    ///
    /// Both storage shapes need the same check and neither may clamp: a
    /// resource placed in a heap has a neighbour one byte past its end, and a
    /// resource with its own pages has the next allocation there.
    fn window(self, offset: u64, length: u64) -> Result<ByteRange, heap::Refusal> {
        match self {
            Self::Placed(p) => p.within(offset, length),
            Self::Dedicated { extent, .. } => {
                if offset
                    .checked_add(length)
                    .is_none_or(|end| end > extent.length)
                {
                    return Err(heap::Refusal::OutOfPlacement {
                        offset,
                        length,
                        placement_length: extent.length,
                    });
                }
                Ok(ByteRange {
                    offset: extent.offset.saturating_add(offset),
                    length,
                })
            }
        }
    }
}

#[derive(Debug)]
struct Task {
    /// The page frame its page table is rooted at. See
    /// [`crate::identity::DirectoryFrame`].
    directory: DirectoryFrame,
    /// Where this task's object list lives and how many slots it has, once a
    /// `SetObjectList` has said so.
    ///
    /// `None` until then, which is a real state and not a default: a task whose
    /// list is unbound has no table for a reference to be an index into, so
    /// every ref in it names nothing — and that is different from a bound list
    /// whose slot happens to be empty.
    object_list: Option<ObjectList>,
    namespace: Namespace,
    heaps: Heaps,
    resident: HashMap<ResourceId, Resident>,
}

/// Where a task's object list is and how big the guest said it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectList {
    /// The guest page frame the table starts at.
    pub page: DirectoryFrame,
    /// How many slots the table has room for.
    ///
    /// **Capacity, not population, and the difference is not academic.** A
    /// driven boot's `SetObjectList` declares 1 048 576 entries eighteen times
    /// over, and about twelve hundred of them are ever named. An owner that
    /// treated this as the number of objects to walk would read a gigabyte of
    /// guest memory per bind to find them.
    pub capacity: u32,
}

impl Task {
    fn new(directory: DirectoryFrame) -> Self {
        Self {
            directory,
            object_list: None,
            namespace: Namespace::new(),
            heaps: Heaps::default(),
            resident: HashMap::new(),
        }
    }
}

/// The resource-lifecycle owner: every task's names and heaps, and the one
/// session-wide content authority.
///
/// The authority is session-wide and not per task because content is a
/// property of a backing, and an IOSurface backing is reachable from more than
/// one task, so a per-task ledger would have two answers to where the current
/// bytes are.
///
/// # What ends this, and what does not
///
/// Deleting a task ends its half — [`Self::apply`]'s `DeleteTask` retires
/// every name, frees the storage no other task holds, and hands both back in
/// the [`Effects`]. That is the only door, and it is per task on purpose.
///
/// There is deliberately **no** generation door. This used to be described as
/// the owner "for one session generation", which claimed a scope nothing here
/// enforces: a reset advances [`crate::identity::SessionGeneration`] and
/// leaves every task, name, heap and content entry exactly where it was.
/// Whether that is right is not this module's to decide —
/// `.agents/paravirt-vulkan-replacement-architecture-plan.md` puts a reset's
/// "completion and resource consequences" among the terms that must be
/// *learned* from the ParaVirt contract, and observing that the current device
/// clears guest-derived state on reset is listed there as evidence for
/// separating the two lifetimes rather than as proof of the survival rule. A
/// door that tore down every task on reset would be that guess, and a guest
/// that resets and keeps using its objects would lose them to it.
///
/// Device loss is the other one and is a different question, because there the
/// plan *is* explicit: content the lost device held is undefined and must not
/// be promoted to a successful result or copied back. Nothing here drops a
/// [`crate::content::Replica::DeviceOwned`] claim when an epoch dies, so a
/// later synchronize can still find the device the only holder of bytes that
/// no longer exist. What the guest is owed for those bytes is the unrecovered
/// half, and inventing an answer would put plausible content in guest memory
/// — which is the one thing the plan says this device does not do.
#[derive(Debug, Default)]
pub struct Lifecycle {
    tasks: HashMap<TaskId, Task>,
    content: ContentLedger,
}

impl Lifecycle {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The session's content authority, for the executor that has to ask where
    /// bytes are without going through a lifecycle operation to do it.
    #[must_use]
    pub const fn content(&self) -> &ContentLedger {
        &self.content
    }

    /// The same authority, writable.
    ///
    /// **Not a second authority, and the distinction is what makes this
    /// legitimate.** A lifecycle operation's own content effects go through
    /// [`Self::apply`] and [`Self::complete`]; what this is for is the *other*
    /// event that changes where current bytes are — a transaction's writes
    /// landing at completion — which is not a lifecycle operation and has no
    /// operation to go through. Both events belong to one ledger, which is
    /// exactly the property a second ledger beside this one would break.
    ///
    /// [`crate::interpret::Interpreter`] is the caller: it holds this model and
    /// materialises the versions a completed transaction published. A caller
    /// that wanted to *declare* a backing before a run reaches the same way.
    pub const fn content_mut(&mut self) -> &mut ContentLedger {
        &mut self.content
    }

    /// Declare a heap into a task.
    ///
    /// Not a [`LifecycleOp`]: the ledger judges heap-backed *texture* creation
    /// and leaves heap destruction unresolved, so there is no established
    /// packet-level heap lifetime to give an operation to. This is the owner
    /// interface the creation route will use once that route is closed, and
    /// naming it here keeps [`Storage::Placed`] reachable without inventing a
    /// command.
    ///
    /// # Errors
    ///
    /// If the task does not exist, a live heap already has the number, or any
    /// heap of this task already holds the storage — see
    /// [`crate::heap::Refusal::StorageInUse`], which forwards through
    /// [`Refusal::Heap`].
    pub fn declare_heap(
        &mut self,
        task: TaskId,
        heap: u64,
        backing: BackingId,
        length: u64,
    ) -> Result<(), Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        t.heaps
            .create(heap, backing, length)
            .map_err(|refusal| Refusal::Heap { task, refusal })
    }

    /// Resolve a task-local name to the backing and extent it currently holds.
    ///
    /// # Errors
    ///
    /// If the task or the name does not resolve.
    pub fn resolve(&mut self, task: TaskId, resource: ResourceId) -> Result<ByteRange, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        t.namespace
            .resolve(resource)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        Ok(t.resident.get(&resource).copied().map_or(
            ByteRange {
                offset: 0,
                length: 0,
            },
            Resident::extent,
        ))
    }

    /// This owner as an [`AccessSource`] for one task's records, in one
    /// submission domain.
    ///
    /// Both are properties of the *packet* and not of any one participation,
    /// which is why they are bound here rather than passed to
    /// [`Self::access`] by a caller that would have to repeat them per record.
    pub fn task_access(&mut self, task: TaskId, domain: ChannelId) -> TaskAccess<'_> {
        TaskAccess {
            lifecycle: self,
            task,
            domain,
        }
    }

    /// Place one participation: which backing it names, where in that
    /// backing's coordinates, and which content versions it consumes and
    /// produces.
    ///
    /// The single step from what a record said to what a scheduler can order,
    /// and it is here because this is the only owner that holds all three
    /// registries at once — the task's names, its heaps, and the session's
    /// content authority. Anywhere else it would be three lookups a caller
    /// could get out of step with each other.
    ///
    /// # The extent is translated, and a heap placement is not widened
    ///
    /// A record names a window of a *resource*; the dependency graph compares
    /// windows of a *backing*. For a resource with its own pages those are the
    /// same coordinates; for one placed in a heap they are not, and the single
    /// checked conversion is [`crate::heap::HeapPlacement::within`].
    ///
    /// A whole-resource participation is therefore two different keys. A
    /// dedicated resource *is* its backing, so it stays
    /// [`crate::access::AccessKey::Whole`] — the record named no range and the
    /// precision census should keep saying so. A placed resource is a window of
    /// a heap someone else's resource also sits in, so "the whole resource" is
    /// an exact range and naming the whole backing would alias every
    /// neighbour into it. The model knows that window exactly; claiming less
    /// would buy ordering the record never asked for.
    ///
    /// A subresource is not translated. Relating image coordinates to bytes
    /// needs a layout, which is an executor's — so it travels as the record
    /// wrote it and `may_alias` compares it against the backing.
    ///
    /// # Versions
    ///
    /// The input version is what is current over the memory named, and it is
    /// `None` where nothing has written it. A write reserves the next version
    /// now and commits it at completion, which is the reservation rule
    /// [`crate::coverage`] states: a reader planned against it waits for the
    /// work rather than for the plan.
    ///
    /// # Errors
    ///
    /// If the task or the name does not resolve, or the window leaves the
    /// resource. Nothing is reserved on a refusal.
    pub fn access(
        &mut self,
        task: TaskId,
        domain: ChannelId,
        participation: &Participation,
    ) -> Result<AccessIntent, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        t.namespace
            .resolve(participation.resource)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        // The name resolved a line above, so an absent residency is not "no
        // such slot" — it is exactly and only a declaration of
        // `Storage::NoBytes`, which is the one arm of `create_resource` that
        // inserts nothing here. This used to answer `NotDeclared` for it, which
        // was a lie about a live name and refused the whole transaction: an
        // indirect command buffer is such an object, so one ICB record refused
        // every draw in its packet.
        //
        // What a resource with no bytes participates as is
        // `AccessKey::DomainOnly` — see `Participation::resolve_no_bytes`. It
        // over-orders and never under-orders, and it is the same vocabulary an
        // unresolvable present target already uses.
        let Some(&resident) = t.resident.get(&participation.resource) else {
            return Ok(participation.resolve_no_bytes(domain));
        };
        let key = ResourceKey {
            backing: resident.backing(),
            heap: match resident {
                // The membership generation the participation is recorded
                // against, so a `useHeap` written before a neighbour arrived
                // still meets the resource that never moved — see
                // `HeapId::same_heap`.
                Resident::Placed(p) => t
                    .heaps
                    .membership(p.heap)
                    .map_err(|refusal| Refusal::Heap { task, refusal })
                    .map(Some)?,
                Resident::Dedicated { .. } => None,
            },
        };
        // The whole of this resource, in whichever vocabulary its storage shape
        // uses. Named once because two arms below need it: a participation that
        // says "the whole thing" and one whose window this resource cannot hold.
        let whole = match resident {
            Resident::Dedicated { .. } => ParticipationExtent::Whole,
            Resident::Placed(p) => ParticipationExtent::Range(p.region),
        };
        let extent = match participation.extent {
            // **A window that does not fit widens; it does not refuse.**
            //
            // A record's `Range` is an *upper bound on what the record touches*
            // and not an assertion that those bytes exist.
            // `ImagePitch::span_bytes` says so in its own doc — it multiplies a
            // pitch by a height or a depth and its caller widens to the whole
            // buffer rather than guess a packed layout, "because a packed guess
            // is a shorter span than the copy really touches and a short span is
            // a missed hazard edge". A bound built to err long, checked as if it
            // were exact, refuses the copies it was made safe for.
            //
            // It was measured. A driven macos-15 boot refused seven exec packets
            // a boot here, every one of them a `0x12c`
            // `copyFromBuffer:…toTexture:…` reading its source buffer:
            // `offset=65536 length=196608` of a resource whose whole extent is
            // 196 608 bytes, the span being `bytes_per_row × height` over a
            // sub-rect that starts partway in. Each refusal threw away a whole
            // packet of guest work.
            //
            // So the answer is the one three lines up, for the same reason:
            // widening over-orders and never under-orders, exactly as
            // `resolve_no_bytes`' `AccessKey::DomainOnly` does. Narrowing —
            // clamping the window to the extent — is the one answer that is
            // wrong, because it claims fewer bytes than the record may touch and
            // a short claim is an edge that does not get built.
            //
            // The caller can see it happened: it asked with a `Range` and got a
            // key that is not one.
            ParticipationExtent::Range(range) => resident
                .window(range.offset, range.length)
                .map_or(whole, ParticipationExtent::Range),
            ParticipationExtent::Subresource(range) => ParticipationExtent::Subresource(range),
            ParticipationExtent::Whole => whole,
        };
        let bytes = match extent {
            ParticipationExtent::Range(range) => Some(range),
            // The whole of a dedicated resource is the extent it was declared
            // with, which the content authority already knows.
            ParticipationExtent::Whole => self.content.extent(key.backing),
            ParticipationExtent::Subresource(_) => None,
        };
        let input = bytes.map_or_else(
            || self.content.newest_version(key.backing),
            |range| self.content.version_of(key.backing, range),
        );
        let output = participation
            .mode
            .writes()
            .then(|| self.content.reserve(key.backing));
        Ok(Participation {
            extent,
            ..*participation
        }
        .resolve(domain, key, input, output))
    }

    /// Apply one lifecycle operation.
    ///
    /// # Errors
    ///
    /// If the task, a name, or a heap does not resolve. A refused operation
    /// changes nothing, including a multi-resource one that refuses on its last
    /// name.
    pub fn apply(&mut self, op: &LifecycleOp) -> Result<Effects, Refusal> {
        match op {
            LifecycleOp::DefineTask {
                task, directory, ..
            } => self.define_task(*task, *directory),
            LifecycleOp::DeleteTask { task } => self.delete_task(*task),
            LifecycleOp::BindObjectList { task, list } => self.bind_object_list(*task, *list),
            LifecycleOp::CreateResource {
                task,
                slot,
                storage,
            } => self.create_resource(*task, *slot, *storage),
            LifecycleOp::DeleteResource { task, resource, .. } => match resource {
                Some(resource) => self.delete_resource(*task, *resource),
                // The same `None` [`LifecycleOp::DeleteObject`]'s arm answers,
                // for the same reason: this model never named the slot, so it
                // holds nothing to retire, and the rail's `(task, ref)`-keyed
                // teardown is the half with work left.
                None => Ok(Effects::default()),
            },
            LifecycleOp::MapMemory { task, span } => self.remap(*task, *span, true),
            LifecycleOp::UnmapMemory { task, span } => self.remap(*task, *span, false),
            LifecycleOp::ReplacePhysical {
                task,
                resource,
                backing,
                extent,
            } => self.replace_physical(*task, *resource, *backing, *extent),
            LifecycleOp::Invalidate { task, resources } => self.invalidate(*task, resources),
            LifecycleOp::Synchronize { task, resources } => {
                self.synchronize(*task, resources, false)
            }
            LifecycleOp::SynchronizeAndDiscard { task, resources } => {
                self.synchronize(*task, resources, true)
            }
            LifecycleOp::Discard { task, resources } => self.discard(*task, resources),
            LifecycleOp::DeleteBacking { task, backing, .. } => match backing {
                Some(backing) => self.delete_backing(*task, *backing),
                // No bytes were ever named for this mapping, so no resource in
                // this model can be resident on them. The rail's mapping-keyed
                // teardown is the half with work left.
                None => Ok(Effects::default()),
            },
            // The name half, which is the only half this model owns. Which
            // per-kind registry the rail drops is the executor's, and it is
            // carried on the operation so the executor does not have to
            // re-derive it from bytes the model no longer holds.
            LifecycleOp::DeleteObject { task, resource, .. } => match resource {
                Some(resource) => self.delete_resource(*task, *resource),
                // Nothing in this model was ever named for it, so there is
                // nothing here to drop. The rail's own `(task, ref)`-keyed
                // retirement is the half that still has work to do, and it has
                // the reference.
                None => Ok(Effects::default()),
            },
        }
    }

    /// Record that a transaction's transfers have executed, and take the
    /// content-authority changes it deferred to that point.
    ///
    /// Returns the discards that were offered and not taken. The sole-authority
    /// question is asked here rather than when the operation was admitted,
    /// because the transfers this transaction owed are exactly what may have
    /// changed the answer.
    #[must_use = "the declined discards belong on the always-on failure channel"]
    pub fn complete(&mut self, effects: &Effects) -> Vec<Declined> {
        for transfer in &effects.transfers {
            self.content.record_transfer(transfer);
        }
        let mut declined = Vec::new();
        for d in &effects.at_completion {
            let sole = self
                .content
                .sole_authority(d.backing, d.bytes, Replica::DeviceOwned);
            if sole.is_empty() {
                self.content
                    .discard(d.backing, d.bytes, Replica::DeviceOwned);
            } else {
                declined.push(Declined {
                    resource: d.resource,
                    backing: d.backing,
                    sole_authority_bytes: sole.len(),
                });
            }
        }
        declined
    }

    /// Forget a backing's content authority, if nothing still answers for it.
    ///
    /// **The check a [`Teardown`] cannot make, and the reason this is here.**
    /// A [`Namespace`] is per task and the content ledger is per session —
    /// deliberately, because an IOSurface backing is reachable from more than
    /// one task — so `Teardown::Now` means "no name in *this* task" and says
    /// nothing about the others. Forgetting on it drops the authority out from
    /// under a live name in a second task, and it is worse than losing
    /// freshness: a forgotten backing's versions restart at zero, so a reader
    /// already ordered against version two would be satisfied by an unrelated
    /// later write.
    ///
    /// A task's heaps are asked too. Heap storage is declared per task and the
    /// same storage may be a heap's in one task and a resource's in another,
    /// which is the same question with a different registry answering it.
    ///
    /// This is also what keeps a deferred teardown deferred. A backing whose
    /// accepted uses have not retired is still held, so the answer is no, and
    /// a caller that retires the last use asks again.
    /// Returns whether it was forgotten — which is the same fact as "no name
    /// and no heap in this session answers for this storage any more", and so
    /// is the only thing that may be reported as freed storage. A `Heaps` is
    /// per task like a `Namespace` is, so `Retirement::StorageFree` is the
    /// same per-task answer a `Teardown::Now` is, and reporting it as a
    /// session-wide free is the split this exists to close.
    fn forget_backing(&mut self, backing: BackingId) -> bool {
        let held = self
            .tasks
            .values()
            .any(|t| t.namespace.holds(backing) || t.heaps.holds_storage(backing));
        if !held {
            self.content.forget(backing);
        }
        !held
    }

    /// Record that work wrote a window of a resource in one replica.
    ///
    /// The one place a resource-relative window becomes a backing-relative one
    /// for content purposes, and the reason it is here rather than at the
    /// executor: a heap-placed resource's neighbour begins one byte past its
    /// end, so an unchecked addition performed where the write is reported
    /// would claim freshness over content the writer never touched.
    ///
    /// # Errors
    ///
    /// If the task or the name does not resolve, or the window leaves the
    /// resource.
    pub fn record_write(
        &mut self,
        task: TaskId,
        resource: ResourceId,
        offset: u64,
        length: u64,
        replica: Replica,
    ) -> Result<(), Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        t.namespace
            .resolve(resource)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        let resident = *t.resident.get(&resource).ok_or(Refusal::Namespace {
            task,
            refusal: namespace::Refusal::NotDeclared {
                slot: resource.slot,
            },
        })?;
        let bytes = resident
            .window(offset, length)
            .map_err(|refusal| Refusal::Heap { task, refusal })?;
        self.content.write(resident.backing(), bytes, replica);
        Ok(())
    }

    /// Install a task's address space, replacing one already under the id.
    ///
    /// **A live task is redefined, and refusing it loses the packet.** The
    /// previous position here was that silently replacing would orphan every
    /// object the old definition owns — which is true, and is an argument
    /// against replacing *silently* rather than against replacing. The guest
    /// does this: one macOS release never redefines a live task and a later one
    /// does, so a model that refuses it refuses an ordinary packet on the
    /// second.
    ///
    /// So the replacement goes through the same teardown a delete does, and
    /// nothing is orphaned: every resource retires by name, every heap that
    /// held the last allocation frees its storage, and all of it travels in the
    /// returned effects. What the caller additionally gets is a
    /// [`Redefinition`], because the objects are only half of it — every
    /// address resolved through the old page table is answered by the wrong
    /// pages now, and this crate holds no such resolution to invalidate.
    fn define_task(&mut self, task: TaskId, directory: DirectoryFrame) -> Result<Effects, Refusal> {
        let effects = match self.tasks.get(&task).map(|t| t.directory) {
            Some(previous) => {
                let mut effects = self.delete_task(task)?;
                effects.redefined.push(Redefinition {
                    task,
                    previous,
                    directory,
                });
                effects
            }
            None => Effects::default(),
        };
        self.tasks.insert(task, Task::new(directory));
        Ok(effects)
    }

    /// Bind a task's object list.
    ///
    /// Nothing is retired here, and that is the contract rather than an
    /// omission. A new table changes what an *unresolved* reference will
    /// construct; the objects already declared in this task keep their own
    /// deletes and their task's lifetime, which is what the device this models
    /// does. Entries whose construction input is no longer valid are the
    /// caller's to retire — they are host caches and resolved windows, neither
    /// of which is a name.
    ///
    /// # Errors
    ///
    /// [`Refusal::NoSuchTask`] for a list bound into a task no definition
    /// opened. Refused rather than creating the task, because a table with no
    /// page directory has no address space its page frame means anything in.
    fn bind_object_list(&mut self, task: TaskId, list: ObjectList) -> Result<Effects, Refusal> {
        let Some(t) = self.tasks.get_mut(&task) else {
            return Err(Refusal::NoSuchTask { task });
        };
        t.object_list = Some(list);
        Ok(Effects::default())
    }

    fn delete_task(&mut self, task: TaskId) -> Result<Effects, Refusal> {
        if !self.tasks.contains_key(&task) {
            return Err(Refusal::NoSuchTask { task });
        }
        let mut effects = Effects::default();
        for name in self.tasks[&task].namespace.live_names() {
            effects.absorb(self.delete_resource(task, name)?);
        }
        // Every allocation is gone, so each remaining heap frees its storage
        // now. A heap that still held one would have said so, and this would
        // be the leak that answer exists to prevent.
        let t = self.tasks.get_mut(&task).expect("checked above");
        let mut released = Vec::new();
        for heap in t.heaps.live_heaps() {
            if let Ok(Retirement::StorageFree { backing }) = t.heaps.delete(heap) {
                released.push(backing);
            }
        }
        // Asked after the task is gone, because forgetting asks every task
        // whether it still answers for the backing and this one no longer
        // does. Reported only where the answer is yes: a heap's retirement is
        // its own task's answer, and another task's heap may hold the same
        // storage.
        self.tasks.remove(&task);
        for backing in released {
            if self.forget_backing(backing) {
                effects.storage_freed.push(backing);
            }
        }
        Ok(effects)
    }

    fn create_resource(
        &mut self,
        task: TaskId,
        slot: ObjectListRef,
        storage: Storage,
    ) -> Result<Effects, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        // The heap window is checked before a name is published, and *all* of
        // it is: `admits` is the half of `place` that does not need a name. A
        // declaration displaces whatever the slot already held, which is not
        // undoable, so a placement still able to refuse afterwards would strand
        // the displaced occupant's teardown on an error path that carries no
        // effects at all.
        let backing = match storage {
            Storage::Dedicated { backing, .. } => Some(backing),
            Storage::Placed {
                heap,
                offset,
                length,
            } => {
                t.heaps
                    .admits(heap, offset, length)
                    .map_err(|refusal| Refusal::Heap { task, refusal })?;
                Some(
                    t.heaps
                        .backing_of(heap)
                        .map_err(|refusal| Refusal::Heap { task, refusal })?,
                )
            }
            // Nothing to admit and nothing to look up: the object owns no bytes.
            Storage::NoBytes => None,
        };
        let crate::namespace::Declared { id, displaced } = t.namespace.declare(slot, backing);
        // The guest wrote over a slot that still held an object. There is no
        // delete packet for that — see `Namespace::declare` — so the delete's
        // obligations are this declaration's, and they are exactly the ones
        // `delete_resource` discharges once the namespace has answered.
        let displaced = displaced.map(|d| self.retire_resident(task, d.id, d.teardown));
        let t = self.tasks.get_mut(&task).expect("resolved above");
        let resident = match storage {
            Storage::Dedicated { backing, extent } => {
                // Its own pages: the guest supplied them and holds all of the
                // content, so this is a declaration and not a write.
                self.content.declare(backing, extent, Replica::GuestPages);
                Resident::Dedicated { backing, extent }
            }
            Storage::Placed {
                heap,
                offset,
                length,
            } => {
                // Cannot refuse. `admits` cleared the window before the name
                // was published, and a freshly declared generation has never
                // been placed — which is what makes the undo this arm used to
                // perform unnecessary, and its loss of the displaced
                // occupant's teardown impossible.
                let placement = t.heaps.place(heap, id, offset, length).expect(
                    "the window was admitted before the name was published, and a fresh \
                     generation is unplaced",
                );
                // A window of a heap, so the authority is about those bytes
                // alone: declaring the whole backing would discard the
                // neighbours' content.
                self.content
                    .write(placement.backing, placement.region, Replica::GuestPages);
                Resident::Placed(placement)
            }
            // No residency entry at all, rather than an empty one. `Resident`
            // answers "where are this name's bytes", and every reader of the
            // table — `resolve_all`, the validity quad, `retire_resident` —
            // already treats an absent entry as "this name has none", which is
            // exactly the truth here. An entry standing for no bytes would make
            // those readers ask a question with no answer.
            Storage::NoBytes => return Ok(displaced.unwrap_or_default()),
        };
        let t = self.tasks.get_mut(&task).expect("resolved above");
        t.resident.insert(id, resident);
        Ok(displaced.unwrap_or_default())
    }

    fn delete_resource(&mut self, task: TaskId, resource: ResourceId) -> Result<Effects, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        let teardown = t
            .namespace
            .delete(resource)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        Ok(self.retire_resident(task, resource, teardown))
    }

    /// Everything a name's departure owes once the namespace has answered for
    /// it: its residency, its heap window, and its content authority.
    ///
    /// Split out of [`Self::delete_resource`] because a delete is no longer the
    /// only way a name departs. A guest that writes over a live object-list
    /// slot replaces its occupant with no packet at all, and that occupant owes
    /// exactly this — see [`crate::namespace::Declared`]. Left inline, the
    /// redeclaration path would have had to restate it, and a restatement that
    /// dropped the heap window would leak a heap allocation per overwritten
    /// slot while `delete_resource` beside it did not.
    ///
    /// Takes the teardown rather than producing it: the two callers reach it
    /// differently, and a function that decided it as well could not be used by
    /// the one whose slot has already moved on.
    fn retire_resident(
        &mut self,
        task: TaskId,
        resource: ResourceId,
        teardown: Teardown,
    ) -> Effects {
        let t = self
            .tasks
            .get_mut(&task)
            .expect("the caller holds the task");
        let resident = t.resident.remove(&resource);
        let mut effects = Effects {
            teardowns: vec![teardown],
            ..Effects::default()
        };
        match resident {
            Some(Resident::Placed(placement)) => {
                if let Ok(Retirement::StorageFree { backing }) = t.heaps.remove(placement, resource)
                {
                    // Reported as freed only if the session agrees. The
                    // retirement is this task's heaps answering, and another
                    // task's heap may hold the same storage --- `Heaps::create`
                    // refuses storage a heap of *its own* task holds and knows
                    // nothing of the others.
                    if self.forget_backing(backing) {
                        effects.storage_freed.push(backing);
                    }
                }
            }
            // Its own pages: nothing else names them, so the content goes with
            // the backing the namespace just handed back.
            Some(Resident::Dedicated { backing, .. }) => match teardown {
                // Nothing in this task reads it and nothing in this task names
                // it. Whether that is true of the session is the question
                // `forget_backing` asks.
                // A dedicated backing is the guest's own pages and not a
                // heap's storage, so nothing is reported freed either way.
                Teardown::Now { .. } => {
                    let _ = self.forget_backing(backing);
                }
                // Someone else still answers for these bytes. Forgetting here
                // would drop the content authority out from under a reader or
                // a live name.
                // `NoStorage` cannot arrive on this arm — a dedicated resident
                // is a name the namespace holds a backing for — and it is
                // grouped here rather than asserted, because the two ways of
                // being wrong are not equal: an unreachable arm that fires
                // does nothing, and a panic in the model's own teardown path
                // takes the session down over a name that owns nothing.
                Teardown::WhenUsesRetire { .. }
                | Teardown::HeldByAnotherName { .. }
                | Teardown::NoStorage => {}
            },
            None => {}
        }
        effects
    }

    /// Record that a task's translations over `span` have changed.
    ///
    /// This crate holds nothing keyed by a guest address, so there is nothing
    /// here to invalidate — the obligation belongs to whoever caches a
    /// resolution, and it is stated as an effect rather than performed. A live
    /// task is still required: an interval in a task that does not exist names
    /// no address space, and accepting it would let a stale channel retire
    /// another task's resolutions.
    fn remap(
        &mut self,
        task: TaskId,
        span: GuestSpan,
        established: bool,
    ) -> Result<Effects, Refusal> {
        if !self.tasks.contains_key(&task) {
            return Err(Refusal::NoSuchTask { task });
        }
        Ok(Effects {
            remapped: vec![Remap {
                task,
                span,
                established,
            }],
            ..Effects::default()
        })
    }

    fn replace_physical(
        &mut self,
        task: TaskId,
        resource: ResourceId,
        backing: BackingId,
        extent: ByteRange,
    ) -> Result<Effects, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        if matches!(t.resident.get(&resource), Some(Resident::Placed(_))) {
            return Err(Refusal::PlacedResourceHasNoPhysical { resource });
        }
        let teardown = t
            .namespace
            .replace_physical(resource, backing)
            .map_err(|refusal| Refusal::Namespace { task, refusal })?;
        t.resident
            .insert(resource, Resident::Dedicated { backing, extent });
        // The new pages are the guest's and every copy of the old ones is of
        // memory this resource no longer names.
        self.content.declare(backing, extent, Replica::GuestPages);
        match teardown {
            Teardown::Now { backing: old } => {
                let _ = self.forget_backing(old);
            }
            // The old bytes are still read by accepted work, or still named by
            // another slot, or there were no old bytes at all. In none of the
            // three is there an authority here to drop.
            Teardown::WhenUsesRetire { .. }
            | Teardown::HeldByAnotherName { .. }
            | Teardown::NoStorage => {}
        }
        Ok(Effects {
            teardowns: vec![teardown],
            ..Effects::default()
        })
    }

    fn invalidate(&mut self, task: TaskId, resources: &[ResourceId]) -> Result<Effects, Refusal> {
        for (_, resident) in self.resolve_all(task, resources)? {
            // The guest's pages are the current content and ours is not, which
            // is a write recorded in the guest's replica. Recorded as a discard
            // instead, the device replica could still claim freshness at the
            // old version.
            self.content
                .write(resident.backing(), resident.extent(), Replica::GuestPages);
        }
        Ok(Effects::default())
    }

    fn synchronize(
        &mut self,
        task: TaskId,
        resources: &[ResourceId],
        then_discard: bool,
    ) -> Result<Effects, Refusal> {
        let resolved = self.resolve_all(task, resources)?;
        let mut effects = Effects::default();
        for (resource, resident) in resolved {
            let (backing, extent) = (resident.backing(), resident.extent());
            if let Some(transfer) =
                self.content
                    .transfer_for_read(backing, extent, Replica::GuestPages)
            {
                effects.transfers.push(transfer);
            }
            if then_discard {
                effects.at_completion.push(DeferredDiscard {
                    resource,
                    backing,
                    bytes: extent,
                });
            }
        }
        Ok(effects)
    }

    fn discard(&mut self, task: TaskId, resources: &[ResourceId]) -> Result<Effects, Refusal> {
        let resolved = self.resolve_all(task, resources)?;
        Ok(Effects {
            at_completion: resolved
                .into_iter()
                .map(|(resource, resident)| DeferredDiscard {
                    resource,
                    backing: resident.backing(),
                    bytes: resident.extent(),
                })
                .collect(),
            ..Effects::default()
        })
    }

    fn delete_backing(&mut self, task: TaskId, backing: BackingId) -> Result<Effects, Refusal> {
        let t = self.tasks.get(&task).ok_or(Refusal::NoSuchTask { task })?;
        // The contract retires the backing *and* the resources that named it,
        // so this is not a refusal when they exist — it is a delete of each.
        let naming: Vec<ResourceId> = t
            .namespace
            .live_names()
            .into_iter()
            .filter(|id| t.resident.get(id).is_some_and(|r| r.backing() == backing))
            .collect();
        let mut effects = Effects::default();
        for resource in naming {
            effects.absorb(self.delete_resource(task, resource)?);
        }
        self.forget_backing(backing);
        Ok(effects)
    }

    /// Resolve every name in a list before any of them takes effect.
    fn resolve_all(
        &mut self,
        task: TaskId,
        resources: &[ResourceId],
    ) -> Result<Vec<(ResourceId, Resident)>, Refusal> {
        let t = self
            .tasks
            .get_mut(&task)
            .ok_or(Refusal::NoSuchTask { task })?;
        let mut out = Vec::with_capacity(resources.len());
        for resource in resources {
            t.namespace
                .resolve(*resource)
                .map_err(|refusal| Refusal::Namespace { task, refusal })?;
            if let Some(resident) = t.resident.get(resource) {
                out.push((*resource, *resident));
            }
        }
        Ok(out)
    }
}

/// The lifecycle owner answering what a guest reference names.
///
/// # Why the owner and not the namespace
///
/// `Namespace` already implements [`crate::resolve::RefResolver`], and one
/// namespace is one task. A caller that has not yet decoded which task a packet
/// is about cannot pick one — which is the whole reason
/// [`crate::resolve::TaskNamespaces`] exists — and the holder of *every* task's
/// namespace is this type. So the routing lives here, and it is routing and
/// nothing else: the generation in the answer is the namespace's, and nothing
/// on this path mints, leases or retires.
///
/// A task this owner does not hold answers `None`, which is the same answer a
/// slot holding nothing gives and is correct for both: a reference into an
/// address space that does not exist names nothing.
impl crate::resolve::TaskNamespaces for Lifecycle {
    fn resource(&self, task: TaskId, object_ref: u32) -> Option<ResourceId> {
        use crate::resolve::RefResolver as _;
        self.tasks.get(&task)?.namespace.resource(object_ref)
    }
}

/// The lifecycle owner answering where a named resource's bytes are.
///
/// # It can carry the heap term, and nothing else can
///
/// [`crate::access::ResourceKey`] holds a backing *and* the heap it was
/// allocated from, so that a heap-use record and a member's access have
/// something to compare — without it the coarser rung silently orders against
/// nothing. A device that mints identities from an allocation base has no heap
/// term at all and must answer `None`; this owner placed the resource itself, so
/// it has the placement and the heap's membership generation, and answers with
/// them.
///
/// The membership generation is asked for at the moment of the question rather
/// than stored in the placement, because that is what a `HeapId` is: two
/// accesses to one heap taken either side of a placement are not comparable, and
/// a key carrying a stale generation would say they were.
///
/// `None` when the task is not held, the name does not resolve in it, or the
/// name resolves and owns no bytes. All three are "there is no key for this",
/// which is what the trait asks; a caller turns it into
/// `AccessKey::DomainOnly` rather than dropping the access.
impl crate::resolve::ResourceStorage for Lifecycle {
    fn storage(&self, task: TaskId, resource: ResourceId) -> Option<ResourceKey> {
        use crate::resolve::RefResolver as _;
        let t = self.tasks.get(&task)?;
        // The name must still be the slot's own, at its own generation: a stale
        // name whose slot has been redeclared would otherwise be keyed onto the
        // new occupant's storage.
        if t.namespace.resource(resource.slot.0) != Some(resource) {
            return None;
        }
        Some(match *t.resident.get(&resource)? {
            Resident::Dedicated { backing, .. } => ResourceKey {
                backing,
                heap: None,
            },
            Resident::Placed(placement) => placement.key(t.heaps.membership(placement.heap).ok()?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::SlotGeneration;
    use reims_vgpu_protocol::packets::LEDGER;
    use std::collections::HashSet;

    const TASK: TaskId = TaskId(1);

    fn range(offset: u64, length: u64) -> ByteRange {
        ByteRange { offset, length }
    }

    fn dedicated(backing: u64, length: u64) -> Storage {
        Storage::Dedicated {
            backing: BackingId(backing),
            extent: range(0, length),
        }
    }

    /// A task with one dedicated resource in slot 0, and the name it got.
    fn with_one_resource(length: u64) -> (Lifecycle, ResourceId) {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(0),
                storage: dedicated(10, length),
            },
        );
        (l, name(0))
    }

    /// Apply an operation that owes nothing, and say so rather than dropping
    /// the answer: `Effects` is `#[must_use]` because an ignored one is an
    /// obligation nobody holds.
    fn apply_inert(l: &mut Lifecycle, op: &LifecycleOp) {
        assert_eq!(l.apply(op).expect("resolves"), Effects::default());
    }

    fn name(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration::default().next(),
        }
    }

    /// One operation of every kind, so a sweep over the vocabulary is a sweep
    /// over values and not over a second list of kinds.
    fn one_of_every_kind() -> Vec<LifecycleOp> {
        let span = GuestSpan {
            base: 0x1000,
            length: 0x1000,
        };
        vec![
            LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
            LifecycleOp::DeleteTask { task: TASK },
            LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(0),
                storage: dedicated(10, 0x100),
            },
            LifecycleOp::DeleteResource {
                task: TASK,
                object_ref: 0,
                resource: Some(name(0)),
            },
            LifecycleOp::MapMemory { task: TASK, span },
            LifecycleOp::UnmapMemory { task: TASK, span },
            LifecycleOp::ReplacePhysical {
                task: TASK,
                resource: name(0),
                backing: BackingId(11),
                extent: range(0, 0x100),
            },
            LifecycleOp::Invalidate {
                task: TASK,
                resources: vec![name(0)],
            },
            LifecycleOp::Synchronize {
                task: TASK,
                resources: vec![name(0)],
            },
            LifecycleOp::SynchronizeAndDiscard {
                task: TASK,
                resources: vec![name(0)],
            },
            LifecycleOp::Discard {
                task: TASK,
                resources: vec![name(0)],
            },
            LifecycleOp::DeleteBacking {
                task: TASK,
                mapping: crate::identity::MappingId(10),
                backing: Some(BackingId(10)),
            },
            LifecycleOp::DeleteObject {
                task: TASK,
                object_ref: 0,
                resource: Some(name(0)),
                kind: reims_vgpu_protocol::destroy::DestroyKind::SamplerState,
            },
        ]
    }

    /// The sweep is over kinds and not over the vector's length, so a
    /// thirteenth operation added without a row here fails rather than being
    /// silently absent from every assertion the vector feeds.
    #[test]
    fn the_kind_sweep_covers_the_whole_vocabulary() {
        let mut kinds: Vec<LifecycleKind> =
            one_of_every_kind().iter().map(LifecycleOp::kind).collect();
        let total = kinds.len();
        kinds.sort_by_key(|k| format!("{k:?}"));
        kinds.dedup();
        assert_eq!(
            kinds.len(),
            total,
            "two operations in the sweep have the same kind, so one kind is unswept"
        );
        for p in LEDGER {
            if let Some(kind) = LifecycleKind::of(p.channel, p.opcode) {
                assert!(
                    kinds.contains(&kind),
                    "{kind:?} is a lifecycle command with no operation in the sweep"
                );
            }
        }
    }

    /// `LifecycleOp::declared_access` and `LifecycleOp::resources` are two
    /// matches over one enum, and this is what stops them drifting.
    ///
    /// Either direction is a defect and neither is caught by the compiler: a
    /// mode with no resource orders the operation against memory it never
    /// named, and a resource with no mode is a packet
    /// `crate::transaction::LifecyclePayload` refuses outright — which is a
    /// whole command class silently lost rather than a wrong edge.
    #[test]
    fn an_operation_declares_an_access_exactly_when_it_names_resources() {
        for op in one_of_every_kind() {
            assert_eq!(
                op.declared_access().is_some(),
                !op.resources().is_empty(),
                "{:?} disagrees with itself about whether it names resources",
                op.kind()
            );
        }
    }

    /// No lifecycle command may declare a pure read of the resources it names.
    ///
    /// Not a restatement of the table: it is the property the table has to have.
    /// Every one of these either produces content on the memory it names or
    /// ends it, so a mode that did not order against a concurrent reader would
    /// let a synchronise publish over bytes a draw is still sampling. `Read` is
    /// the one mode that would do that, and `AccessMode::writes` is the
    /// question the dependency compiler actually asks.
    #[test]
    fn no_lifecycle_operation_declares_a_pure_read() {
        for op in one_of_every_kind() {
            if let Some(mode) = op.declared_access() {
                assert!(
                    mode.writes(),
                    "{:?} declares {} of the resources it names, which orders against no reader",
                    op.kind(),
                    mode.name()
                );
            }
        }
    }

    /// The owner answers both of the model's read doors, and answers them about
    /// the same object.
    ///
    /// Two traits and one test, because the failure worth catching is not either
    /// answer alone: it is the pair disagreeing about which object a reference
    /// names. A `TaskNamespaces` that resolved a slot and a `ResourceStorage`
    /// that keyed a *different* generation's bytes would build a hazard edge
    /// against memory the packet did not name, and each door read on its own
    /// would look right.
    #[test]
    fn the_owner_answers_a_reference_and_its_storage_about_one_object() {
        use crate::resolve::{ResourceStorage as _, TaskNamespaces as _};

        let (l, name) = with_one_resource(0x100);
        assert_eq!(l.resource(TASK, name.slot.0), Some(name));
        assert_eq!(
            l.storage(TASK, name),
            Some(ResourceKey {
                backing: BackingId(10),
                heap: None,
            }),
            "a dedicated allocation carries no heap term, because it is in no heap"
        );

        // A task nobody defined, and a slot nothing was declared into: both
        // doors say nothing rather than defaulting.
        assert_eq!(l.resource(TaskId(9), 0), None);
        assert_eq!(l.storage(TaskId(9), name), None);
        assert_eq!(l.resource(TASK, 5), None);
    }

    /// A name the slot no longer holds is keyed to nothing, not to its
    /// successor.
    ///
    /// The redeclaration case `Namespace::declare` exists for. A guest that
    /// writes over an object-list slot moves the name to a new generation, and a
    /// storage door that keyed the *old* name would hand an access the new
    /// occupant's bytes — the wrong surface, ordered against the wrong work.
    #[test]
    fn a_stale_name_is_keyed_to_nothing_rather_than_to_the_slots_new_occupant() {
        use crate::resolve::ResourceStorage as _;

        let (mut l, first) = with_one_resource(0x100);
        let effects = l
            .apply(&LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(0),
                storage: dedicated(11, 0x200),
            })
            .expect("a live slot may be redeclared");
        assert!(
            !effects.teardowns.is_empty(),
            "the displaced occupant's teardown is owed, or this test is not \
             exercising a redeclaration"
        );
        assert_eq!(
            l.storage(TASK, first),
            None,
            "the previous generation's name keys nothing"
        );
    }

    /// A rebind changes what an unresolved reference will construct and ends
    /// nothing already declared.
    ///
    /// The half a walk-shaped operation could not have expressed. `SetObjectList`
    /// arrives eighteen times a boot on this interface, each declaring a
    /// million-entry capacity; if a bind retired the task's objects, every one
    /// of those would tear down a namespace the guest still holds names in.
    #[test]
    fn rebinding_a_tasks_object_list_leaves_its_declared_names_alone() {
        let (mut l, name) = with_one_resource(0x100);
        let list = ObjectList {
            page: DirectoryFrame(0x9000),
            capacity: 1 << 20,
        };
        apply_inert(&mut l, &LifecycleOp::BindObjectList { task: TASK, list });
        apply_inert(
            &mut l,
            &LifecycleOp::BindObjectList {
                task: TASK,
                list: ObjectList {
                    page: DirectoryFrame(0xA000),
                    capacity: 1 << 20,
                },
            },
        );
        assert_eq!(
            l.tasks[&TASK].namespace.live_names(),
            vec![name],
            "two binds and the name declared before them both still answers"
        );
    }

    /// A bind into a task no definition opened is refused rather than creating
    /// one: a table's page frame means nothing without an address space.
    #[test]
    fn binding_an_object_list_into_an_undefined_task_is_refused() {
        let mut l = Lifecycle::new();
        assert_eq!(
            l.apply(&LifecycleOp::BindObjectList {
                task: TaskId(9),
                list: ObjectList {
                    page: DirectoryFrame(0x9000),
                    capacity: 4,
                },
            }),
            Err(Refusal::NoSuchTask { task: TaskId(9) })
        );
    }

    /// The bind carries the packet's own three words, at the protocol's
    /// offsets, and the count is carried as the capacity it is.
    #[test]
    fn an_object_list_bind_carries_the_packets_own_words() {
        use reims_vgpu_protocol::fifo::{
            SET_OBJECT_LIST_COUNT, SET_OBJECT_LIST_LEN, SET_OBJECT_LIST_PFN,
            SET_OBJECT_LIST_TASK_ID,
        };
        let mut payload = vec![0u8; SET_OBJECT_LIST_LEN];
        payload[SET_OBJECT_LIST_TASK_ID..SET_OBJECT_LIST_TASK_ID + 4]
            .copy_from_slice(&7u32.to_le_bytes());
        payload[SET_OBJECT_LIST_PFN..SET_OBJECT_LIST_PFN + 4]
            .copy_from_slice(&0x1234u32.to_le_bytes());
        payload[SET_OBJECT_LIST_COUNT..SET_OBJECT_LIST_COUNT + 4]
            .copy_from_slice(&(1u32 << 20).to_le_bytes());
        assert_eq!(
            object_list_bind(LifecycleKind::SetObjectList, &payload),
            Ok(LifecycleOp::BindObjectList {
                task: TaskId(7),
                list: ObjectList {
                    page: DirectoryFrame(0x1234),
                    capacity: 1 << 20,
                },
            })
        );
        // One byte short of the record is refused, not defaulted: a zeroed page
        // frame is a valid frame and a zeroed task id is the kernel task.
        payload.pop();
        assert!(matches!(
            object_list_bind(LifecycleKind::SetObjectList, &payload),
            Err(ResolveRefusal::ShortNotice(_))
        ));
    }

    /// The claim the module docs make and cannot check by being read: the
    /// vocabulary is total over the payload class.
    #[test]
    fn every_lifecycle_packet_has_exactly_one_operation() {
        let mut seen: Vec<LifecycleKind> = Vec::new();
        for p in LEDGER {
            let kind = LifecycleKind::of(p.channel, p.opcode);
            let is_lifecycle =
                classify(p.channel, p.opcode) == Some(PayloadClass::ResourceLifecycle);
            assert_eq!(
                kind.is_some(),
                is_lifecycle,
                "{} {:#04x} is classified {:?} and resolves to {:?}",
                p.channel.name(),
                p.opcode,
                classify(p.channel, p.opcode),
                kind
            );
            if let Some(k) = kind {
                seen.push(k);
            }
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            13,
            "every kind is a packet the ledger judged, and every judged \
             lifecycle packet is a kind"
        );
    }

    #[test]
    fn a_packet_that_is_not_lifecycle_has_no_kind() {
        assert_eq!(LifecycleKind::of(Channel::Child, 0x37), None, "the EXEC");
        assert_eq!(LifecycleKind::of(Channel::Child, 0x06), None, "a present");
        assert_eq!(LifecycleKind::of(Channel::Root, 0x00), None, "unjudged");
    }

    /// An invalidate says the device's cached view is stale. Recorded as a
    /// discard of a copy it would leave the device replica able to claim
    /// freshness at the old version; recorded as a guest write it cannot.
    #[test]
    fn an_invalidate_makes_the_guests_pages_the_current_content() {
        let (mut l, id) = with_one_resource(256);
        let sync = l
            .apply(&LifecycleOp::Synchronize {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert!(sync.transfers.is_empty(), "the guest already holds them");
        // The device produced the whole resource, so it is the only holder.
        let backing = BackingId(10);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        assert!(l
            .content()
            .is_fresh(backing, range(0, 256), Replica::DeviceOwned));
        let version_before = l.content().version_of(backing, range(0, 256));
        apply_inert(
            &mut l,
            &LifecycleOp::Invalidate {
                task: TASK,
                resources: vec![id],
            },
        );
        assert!(
            l.content().version_of(backing, range(0, 256)) != version_before,
            "the content changed under us, so the version moved"
        );
        assert!(l
            .content()
            .is_fresh(backing, range(0, 256), Replica::GuestPages));
        assert!(!l
            .content()
            .is_fresh(backing, range(0, 1), Replica::DeviceOwned));
    }

    /// A synchronise owes exactly the bytes the guest is behind on, once.
    #[test]
    fn a_synchronise_owes_the_missing_bytes_and_does_not_repeat() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        // The device produced the back half.
        l.record_write(TASK, id, 128, 128, Replica::DeviceOwned)
            .expect("inside the resource");
        let _ = backing;
        let effects = l
            .apply(&LifecycleOp::Synchronize {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert_eq!(effects.transfers.len(), 1);
        assert_eq!(effects.transfers[0].bytes.ranges(), &[range(128, 128)]);
        assert_eq!(effects.transfers[0].to, Replica::GuestPages);
        assert!(
            l.complete(&effects).is_empty(),
            "a synchronise declines nothing"
        );
        let again = l
            .apply(&LifecycleOp::Synchronize {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert!(
            again.transfers.is_empty(),
            "no write in between, so nothing moved and nothing is owed"
        );
    }

    /// The reading the ledger records: the discard is a hint, and taking it
    /// when the bytes exist nowhere else would destroy content rather than
    /// free a copy.
    #[test]
    fn a_discard_of_content_nothing_else_holds_is_declined_and_named() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        let effects = l
            .apply(&LifecycleOp::Discard {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert!(
            l.content()
                .is_fresh(backing, range(0, 256), Replica::DeviceOwned),
            "admission does not move content authority"
        );
        assert_eq!(
            l.complete(&effects),
            vec![Declined {
                resource: id,
                backing,
                sole_authority_bytes: 256,
            }]
        );
        assert!(
            l.content()
                .is_fresh(backing, range(0, 256), Replica::DeviceOwned),
            "and the declined hint left the copy alone"
        );
    }

    /// The order in the combined packet is the whole point: the synchronise
    /// gives the bytes a second holder, so the discard is always lawful.
    #[test]
    fn a_synchronise_and_discard_is_never_the_declined_kind() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        let effects = l
            .apply(&LifecycleOp::SynchronizeAndDiscard {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        assert_eq!(
            effects.transfers.len(),
            1,
            "the guest is behind on all of it"
        );
        assert_eq!(
            l.complete(&effects),
            vec![],
            "the transfer this transaction owed is what made the hint lawful"
        );
        assert!(
            !l.content()
                .is_fresh(backing, range(0, 1), Replica::DeviceOwned),
            "so the copy was released"
        );
        assert!(
            l.content()
                .is_fresh(backing, range(0, 256), Replica::GuestPages),
            "and the content survived in the guest's pages"
        );
    }

    /// The same claim from the other side: had the discard been applied when
    /// the packet was admitted, the transfer it owed would have been planned
    /// against freshness that no longer existed.
    #[test]
    fn a_deferred_discard_is_evaluated_against_the_state_its_transfers_left() {
        let (mut l, id) = with_one_resource(256);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        let effects = l
            .apply(&LifecycleOp::SynchronizeAndDiscard {
                task: TASK,
                resources: vec![id],
            })
            .expect("resolves");
        // Complete without recording the transfers this transaction owed: the
        // guest never got the bytes, so the discard is the destroying kind.
        let without_transfers = Effects {
            transfers: Vec::new(),
            ..effects
        };
        assert_eq!(
            l.complete(&without_transfers).len(),
            1,
            "the answer is asked at completion and not cached from admission"
        );
    }

    /// A merge that names its fields one by one is only correct for as long as
    /// nobody adds a field, and the failure mode is an obligation dropped in
    /// silence. This holds `absorb` to totality with a value in every field.
    #[test]
    fn absorbing_effects_keeps_every_obligation_and_not_only_the_two_a_delete_makes() {
        let full = Effects {
            transfers: vec![Transfer {
                backing: BackingId(7),
                from: Replica::GuestPages,
                to: Replica::DeviceOwned,
                bytes: crate::range_set::RangeSet::default(),
                at: Vec::new(),
            }],
            teardowns: vec![Teardown::Now {
                backing: BackingId(7),
            }],
            storage_freed: vec![BackingId(9)],
            at_completion: vec![DeferredDiscard {
                resource: name(3),
                backing: BackingId(7),
                bytes: range(0, 16),
            }],
            remapped: vec![Remap {
                task: TASK,
                span: GuestSpan {
                    base: 0,
                    length: 16,
                },
                established: true,
            }],
            redefined: vec![Redefinition {
                task: TASK,
                previous: DirectoryFrame(1),
                directory: DirectoryFrame(2),
            }],
        };

        let mut into = Effects::default();
        into.absorb(full.clone());
        assert_eq!(into, full, "a field was dropped on the way through");

        // And it accumulates rather than replaces, which is what the two
        // list-shaped deletes need from it.
        into.absorb(full.clone());
        for len in [
            into.transfers.len(),
            into.teardowns.len(),
            into.storage_freed.len(),
            into.at_completion.len(),
            into.remapped.len(),
            into.redefined.len(),
        ] {
            assert_eq!(len, 2);
        }
    }

    #[test]
    fn a_list_with_one_stale_name_changes_nothing() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        let version = l.content().version_of(backing, range(0, 256));
        let stale = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration::default().next(),
        };
        let refusal = l
            .apply(&LifecycleOp::Invalidate {
                task: TASK,
                resources: vec![id, stale],
            })
            .expect_err("the second name never resolved");
        assert_eq!(
            refusal,
            Refusal::Namespace {
                task: TASK,
                refusal: namespace::Refusal::NotDeclared {
                    slot: ObjectListRef(9)
                },
            }
        );
        assert_eq!(
            l.content().version_of(backing, range(0, 256)),
            version,
            "the first resource was not invalidated on the way to the refusal"
        );
    }

    /// The module's "nothing is half-applied" claim, over every content
    /// command and every position a bad name can sit in.
    ///
    /// `a_list_with_one_stale_name_changes_nothing` drives one command with the
    /// stale name last, which is the position where a loop that applied as it
    /// resolved would still look right for the *first* resource it touched only
    /// if it touched none --- and says nothing about the other three commands
    /// or about a name in the middle. The claim is over all four and over every
    /// position, so it is driven that way rather than from a hand-picked case:
    /// a hand-picked case is the second copy of the rule.
    ///
    /// The observation is the whole content ledger reachable through its own
    /// doors, per backing, so a command that moved a version, a freshness run
    /// or an extent anywhere is caught rather than only one that moved the
    /// version this test happened to name.
    #[test]
    fn no_content_command_applies_any_of_a_list_it_refuses() {
        const LENGTH: u64 = 256;
        let build = || {
            let mut l = Lifecycle::new();
            apply_inert(
                &mut l,
                &LifecycleOp::DefineTask {
                    task: TASK,
                    kernel: false,
                    directory: DirectoryFrame(0x1000),
                },
            );
            for slot in 0..3u32 {
                apply_inert(
                    &mut l,
                    &LifecycleOp::CreateResource {
                        task: TASK,
                        slot: ObjectListRef(slot),
                        storage: dedicated(u64::from(slot) + 10, LENGTH),
                    },
                );
            }
            // The device produced the back half of each. Without a write there
            // is nothing to synchronise and nothing to discard, and three of
            // the four commands would be inert for a reason that has nothing to
            // do with the rule under test.
            for slot in 0..3u32 {
                l.record_write(
                    TASK,
                    name(slot),
                    LENGTH / 2,
                    LENGTH / 2,
                    Replica::DeviceOwned,
                )
                .expect("inside the resource");
            }
            l
        };
        // Everything the ledger will say about the three backings, through its
        // own doors: the newest version, the declared extent, the current
        // version of the whole span and of each half, and which replicas hold
        // which bytes fresh.
        let snapshot = |l: &Lifecycle| -> Vec<String> {
            // The census leads, because a synchronise's admission leaves no
            // mark on the versions at all --- what it does is ask
            // `transfer_for_read`, which counts. Without it a synchronise that
            // resolved and asked one resource at a time would refuse having
            // already asked about the ones before the bad name, and every
            // version would still read as untouched.
            core::iter::once(format!("{:?}", l.content().census()))
                .chain((10..13u64).map(|backing| {
                    let backing = BackingId(backing);
                    let content = l.content();
                    format!(
                        "{backing:?} newest={:?} extent={:?} whole={:?} lo={:?} hi={:?} {:?}",
                        content.newest_version(backing),
                        content.extent(backing),
                        content.version_of(backing, range(0, LENGTH)),
                        content.version_of(backing, range(0, LENGTH / 2)),
                        content.version_of(backing, range(LENGTH / 2, LENGTH / 2)),
                        Replica::BOTH.map(|replica| (
                            replica,
                            content.is_fresh(backing, range(0, LENGTH), replica),
                            content.is_fresh(backing, range(0, LENGTH / 2), replica),
                        )),
                    )
                }))
                .collect()
        };

        let stale = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration::default().next(),
        };
        let expected = Refusal::Namespace {
            task: TASK,
            refusal: namespace::Refusal::NotDeclared {
                slot: ObjectListRef(9),
            },
        };
        let commands: [fn(TaskId, Vec<ResourceId>) -> LifecycleOp; 4] = [
            |task, resources| LifecycleOp::Invalidate { task, resources },
            |task, resources| LifecycleOp::Synchronize { task, resources },
            |task, resources| LifecycleOp::SynchronizeAndDiscard { task, resources },
            |task, resources| LifecycleOp::Discard { task, resources },
        ];
        for command in commands {
            // Each command is first driven with a list that resolves whole, so
            // a version that never moves because the command does nothing at
            // all cannot be read as a command that refused cleanly.
            let mut moves = build();
            let before = snapshot(&moves);
            let effects = moves
                .apply(&command(TASK, (0..3).map(name).collect()))
                .expect("three live names resolve");
            let kind = command(TASK, Vec::new()).kind();
            assert!(
                snapshot(&moves) != before || effects != Effects::default(),
                "{kind:?} over three live resources did nothing observable, so \
                 the refusing runs below prove nothing"
            );

            for position in 0..4 {
                let mut resources: Vec<ResourceId> = (0..3).map(name).collect();
                resources.insert(position, stale);
                let mut l = build();
                let before = snapshot(&l);
                assert_eq!(
                    l.apply(&command(TASK, resources))
                        .expect_err("one name never resolved"),
                    expected,
                    "{kind:?} with the stale name at {position}"
                );
                assert_eq!(
                    snapshot(&l),
                    before,
                    "{kind:?} applied part of a list it refused, with the stale \
                     name at {position}"
                );
            }
        }
    }

    /// A heap-placed resource does not own pages, so there is nothing to
    /// re-point — and re-pointing the heap's storage would move its neighbours.
    #[test]
    fn a_placed_resource_refuses_a_physical_replacement() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        l.declare_heap(TASK, 3, BackingId(50), 4096).expect("heap");
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(0),
                storage: Storage::Placed {
                    heap: 3,
                    offset: 0,
                    length: 256,
                },
            },
        );
        assert_eq!(
            l.apply(&LifecycleOp::ReplacePhysical {
                task: TASK,
                resource: name(0),
                backing: BackingId(60),
                extent: range(0, 256),
            }),
            Err(Refusal::PlacedResourceHasNoPhysical { resource: name(0) })
        );
    }

    /// The dedicated arm's half of the same rule.
    ///
    /// Two live names for one backing is a first-class state — an `IOSurface`
    /// reachable from two tasks is the ordinary case, and `Namespace::delete`
    /// carries a whole variant for it. Creating the second one used to reset
    /// the backing's authority: the first name's rendered bytes became content
    /// nothing had ever written, so a read planned no transfer and found no
    /// source, and the backing's extent shrank to the newcomer's — which is
    /// what turns a whole-resource participation into bytes.
    #[test]
    fn a_second_name_for_one_backing_leaves_the_first_ones_content_alone() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        let declare = |l: &mut Lifecycle, slot: u32, extent: ByteRange| {
            apply_inert(
                l,
                &LifecycleOp::CreateResource {
                    task: TASK,
                    slot: ObjectListRef(slot),
                    storage: Storage::Dedicated {
                        backing: BackingId(10),
                        extent,
                    },
                },
            );
        };
        declare(&mut l, 0, range(0, 256));
        l.record_write(TASK, name(0), 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");

        // A second name over the first sixty-four bytes of the same pages.
        declare(&mut l, 1, range(0, 64));

        assert!(
            l.content()
                .is_fresh(BackingId(10), range(64, 192), Replica::DeviceOwned),
            "the bytes the second name did not claim are still the device's"
        );
        assert!(
            l.content()
                .is_fresh(BackingId(10), range(0, 64), Replica::GuestPages),
            "and the ones it did are the guest's"
        );
        assert_eq!(
            l.content().extent(BackingId(10)),
            Some(range(0, 256)),
            "the backing is still as big as the name that declared it"
        );
    }

    /// Freed storage is a session-wide fact and a heap's retirement is a
    /// per-task answer.
    ///
    /// `Heaps` is per task exactly as `Namespace` is, and `Heaps::create`
    /// refuses storage a heap of *its own* task already holds — it knows
    /// nothing of the others. So two tasks may each declare a heap over one
    /// backing, and one of them retiring its last placement says only that
    /// *it* is finished. Reported as `storage_freed`, that told the caller to
    /// tear down an allocation the other task's heap still holds, while
    /// `forget_backing` — asked the session-wide question — correctly kept the
    /// content entry. The model said both at once.
    #[test]
    fn storage_two_tasks_hold_is_not_reported_freed_when_one_lets_go() {
        let mut l = Lifecycle::new();
        for task in [TaskId(1), TaskId(2)] {
            apply_inert(
                &mut l,
                &LifecycleOp::DefineTask {
                    task,
                    kernel: false,
                    directory: DirectoryFrame(0x1000),
                },
            );
        }
        l.declare_heap(TaskId(1), 3, BackingId(50), 4096)
            .expect("a heap");
        l.declare_heap(TaskId(2), 7, BackingId(50), 4096)
            .expect("the same storage, in another task's registry");

        // Task one ends, and its heaps retire their storage. That answer is
        // task one's alone.
        let effects = l
            .apply(&LifecycleOp::DeleteTask { task: TaskId(1) })
            .expect("resolves");
        assert!(
            effects.storage_freed.is_empty(),
            "task two's heap still holds this storage"
        );

        // Once the last holder lets go, the storage is free and is reported
        // once.
        let effects = l
            .apply(&LifecycleOp::DeleteTask { task: TaskId(2) })
            .expect("resolves");
        assert_eq!(effects.storage_freed, vec![BackingId(50)]);
    }

    /// Two resources in one heap share a backing, so creating the second must
    /// not declare the backing's authority — that would discard the first's
    /// content.
    #[test]
    fn placing_a_second_resource_leaves_the_first_ones_content_alone() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        l.declare_heap(TASK, 3, BackingId(50), 4096).expect("heap");
        let place = |l: &mut Lifecycle, slot: u32, offset: u64| {
            apply_inert(
                l,
                &LifecycleOp::CreateResource {
                    task: TASK,
                    slot: ObjectListRef(slot),
                    storage: Storage::Placed {
                        heap: 3,
                        offset,
                        length: 256,
                    },
                },
            );
        };
        place(&mut l, 0, 0);
        l.record_write(TASK, name(0), 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        place(&mut l, 1, 256);
        assert!(
            l.content()
                .is_fresh(BackingId(50), range(0, 256), Replica::DeviceOwned),
            "the neighbour's copy survived"
        );
        assert!(
            l.content()
                .is_fresh(BackingId(50), range(256, 256), Replica::GuestPages),
            "and the new window is the guest's"
        );
    }

    #[test]
    fn deleting_a_task_retires_its_objects_and_frees_its_heap_storage() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        l.declare_heap(TASK, 3, BackingId(50), 4096).expect("heap");
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(0),
                storage: Storage::Placed {
                    heap: 3,
                    offset: 0,
                    length: 256,
                },
            },
        );
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(1),
                storage: dedicated(10, 128),
            },
        );
        let effects = l
            .apply(&LifecycleOp::DeleteTask { task: TASK })
            .expect("live task");
        assert_eq!(effects.teardowns.len(), 2, "both objects were retired");
        assert_eq!(
            effects.storage_freed,
            vec![BackingId(50)],
            "the heap's last allocation went with the task"
        );
        assert_eq!(
            l.apply(&LifecycleOp::DeleteTask { task: TASK }),
            Err(Refusal::NoSuchTask { task: TASK })
        );
    }

    /// A live task is redefined by the guest, and the model replaces it rather
    /// than refusing the packet.
    ///
    /// Refusing was the previous answer, on the ground that replacing silently
    /// would orphan the objects the old definition owns. It does — so the
    /// replacement is the delete's own teardown path, and the orphans arrive as
    /// effects instead of being invented away.
    #[test]
    fn a_live_task_is_redefined_and_its_objects_retire_by_name() {
        let (mut l, id) = with_one_resource(256);
        let effects = l
            .apply(&LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x2000),
            })
            .expect("a guest redefines a live task");
        assert_eq!(
            effects.redefined,
            vec![Redefinition {
                task: TASK,
                previous: DirectoryFrame(0x1000),
                directory: DirectoryFrame(0x2000),
            }]
        );
        assert!(effects.redefined[0].root_moved());
        assert!(
            !effects.teardowns.is_empty(),
            "the resource the old definition owned retired by name"
        );
        // The name is the previous space's and does not resolve in the new one.
        assert_eq!(
            l.resolve(TASK, id),
            Err(Refusal::Namespace {
                task: TASK,
                refusal: namespace::Refusal::NotDeclared { slot: id.slot },
            })
        );
    }

    /// A redefinition at the same root is still a redefinition — the objects go
    /// either way — and it says the root did not move, because what the guest
    /// published into that page is still in it.
    #[test]
    fn a_redefinition_at_the_same_root_says_so() {
        let (mut l, _) = with_one_resource(256);
        let effects = l
            .apply(&LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            })
            .expect("redefined");
        assert_eq!(
            effects.redefined,
            vec![Redefinition {
                task: TASK,
                previous: DirectoryFrame(0x1000),
                directory: DirectoryFrame(0x1000),
            }]
        );
        assert!(
            !effects.redefined[0].root_moved(),
            "what the guest published into that page is still in it"
        );
    }

    /// A first definition replaces nothing and says nothing about a space that
    /// was not there.
    #[test]
    fn a_first_definition_is_not_a_redefinition() {
        let mut l = Lifecycle::new();
        let effects = l
            .apply(&LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            })
            .expect("a fresh task");
        assert_eq!(effects, Effects::default());
    }

    /// The contract retires the backing *and* the resources that named it, so
    /// this is not a refusal when they exist.
    #[test]
    fn deleting_a_backing_retires_the_resources_that_named_it() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        for slot in 0..2 {
            apply_inert(
                &mut l,
                &LifecycleOp::CreateResource {
                    task: TASK,
                    slot: ObjectListRef(slot),
                    storage: dedicated(10, 128),
                },
            );
        }
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(2),
                storage: dedicated(11, 128),
            },
        );
        let effects = l
            .apply(&LifecycleOp::DeleteBacking {
                task: TASK,
                mapping: crate::identity::MappingId(10),
                backing: Some(BackingId(10)),
            })
            .expect("live task");
        assert_eq!(effects.teardowns.len(), 2, "and only those two");
        assert_eq!(
            l.resolve(TASK, name(2)),
            Ok(range(0, 128)),
            "the resource on the other backing is untouched"
        );
        assert!(matches!(
            l.resolve(TASK, name(0)),
            Err(Refusal::Namespace { .. })
        ));
    }

    /// A retirement's first word is a mapping and its second is the task, and
    /// the mapping resolves through the mapper rather than the object list.
    ///
    /// The record is the reverse of the `{task, object}` pair, and the two
    /// numbers here are different so a swap cannot pass: read backwards this
    /// would retire task 4's number as a mapping.
    #[test]
    fn a_backing_retirement_names_a_mapping_first_and_a_task_second() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&9u32.to_le_bytes());
        payload.extend_from_slice(&4u32.to_le_bytes());
        assert_eq!(
            backing_retirement(LifecycleKind::DeleteBacking, &payload, &EveryMapping),
            Ok(LifecycleOp::DeleteBacking {
                task: TaskId(4),
                mapping: crate::identity::MappingId(9),
                backing: Some(BackingId(1_009)),
            })
        );
    }

    /// A mapping the mapper holds no surface for still retires.
    ///
    /// **A retirement is never refused for a name nothing resolves**, which is
    /// the rule [`LifecycleOp::DeleteObject`] established and
    /// [`LifecycleOp::DeleteResource`] joined. This is the third command under
    /// it, and the one whose id is a *mapping*: a mapping this model holds no
    /// surface for is one no resource in it can be resident on, so there is
    /// nothing to retire here and never was — while the rail's own
    /// mapping-keyed teardown still has work, and only the mapping id can
    /// reach it.
    #[test]
    fn a_retirement_naming_no_live_surface_still_retires() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&9u32.to_le_bytes());
        payload.extend_from_slice(&4u32.to_le_bytes());
        let op = backing_retirement(LifecycleKind::DeleteBacking, &payload, &NoMapping)
            .expect("a retirement is not lost to an unheld mapping");
        assert_eq!(
            op,
            LifecycleOp::DeleteBacking {
                task: TaskId(4),
                mapping: crate::identity::MappingId(9),
                backing: None,
            }
        );
        assert!(
            op.resources().is_empty(),
            "no backing, so no resource of this model's is named"
        );
    }

    /// A definition's word is doubled and a deletion's is not, and the join
    /// reads each the way its own command carries it.
    ///
    /// The pair that makes it matter: a definition of user task 1 sends `2`,
    /// and a deletion of user task 1 sends `1`. Swap the conventions and the
    /// definition registers task 2 while the deletion retires the kernel task.
    #[test]
    fn a_task_definition_is_doubled_and_a_deletion_is_not() {
        let mut define = vec![0u8; fifo::DEFINE_TASK_LEN];
        define[..4].copy_from_slice(
            &fifo::DefineTaskId {
                task_id: 1,
                kernel: false,
            }
            .to_raw()
            .to_le_bytes(),
        );
        // The page-table root, at its own offset. A definition that read it
        // from anywhere else would name a page the guest did not.
        define[fifo::DEFINE_TASK_DIRECTORY_PFN..fifo::DEFINE_TASK_DIRECTORY_PFN + 4]
            .copy_from_slice(&0x1000u32.to_le_bytes());
        assert_eq!(
            task_lifetime(LifecycleKind::DefineTask, &define),
            Ok(LifecycleOp::DefineTask {
                task: TaskId(1),
                kernel: false,
                directory: DirectoryFrame(0x1000),
            })
        );
        assert_eq!(
            task_lifetime(LifecycleKind::DeleteTask, &1u32.to_le_bytes()),
            Ok(LifecycleOp::DeleteTask { task: TaskId(1) })
        );

        // And the registration a dropped class bit would hide: the kernel task
        // and user task zero are two registrations of slot zero.
        let mut kernel = vec![0u8; fifo::DEFINE_TASK_LEN];
        kernel[..4].copy_from_slice(
            &fifo::DefineTaskId {
                task_id: 0,
                kernel: true,
            }
            .to_raw()
            .to_le_bytes(),
        );
        assert_eq!(
            task_lifetime(LifecycleKind::DefineTask, &kernel),
            Ok(LifecycleOp::DefineTask {
                task: TaskId(0),
                kernel: true,
                directory: DirectoryFrame(0),
            })
        );
    }

    /// A deletion with no id refuses. It does not name slot `0`, which is the
    /// kernel task.
    #[test]
    fn a_short_task_deletion_does_not_name_the_kernel_task() {
        assert_eq!(
            task_lifetime(LifecycleKind::DeleteTask, &[0u8; 3]),
            Err(ResolveRefusal::ShortNotice(fifo::ShortPayload {
                plen: 3,
                need: fifo::DELETE_TASK_LEN,
            }))
        );
        assert_eq!(
            task_lifetime(LifecycleKind::DefineTask, &[0u8; fifo::DEFINE_TASK_LEN - 1]),
            Err(ResolveRefusal::ShortNotice(fifo::ShortPayload {
                plen: fifo::DEFINE_TASK_LEN - 1,
                need: fifo::DEFINE_TASK_LEN,
            }))
        );
    }

    /// A delete names an object and becomes an operation; a re-point names the
    /// same two words and cannot.
    ///
    /// The re-point's new backing and extent are nowhere on the wire — the
    /// guest wired the pages itself and re-committed them at the same GPU-VA —
    /// so the refusal names what is missing rather than letting an operation
    /// carrying the *old* backing report a re-point that moved nothing.
    #[test]
    fn a_delete_resolves_from_its_ref_and_a_repoint_needs_more_than_one() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&TASK.0.to_le_bytes());
        payload.extend_from_slice(&7u32.to_le_bytes());
        let resource = ResourceId {
            slot: ObjectListRef(7),
            generation: SlotGeneration(1),
        };
        assert_eq!(
            object_reference(LifecycleKind::DeleteResource, &payload, &Everything),
            Ok(LifecycleOp::DeleteResource {
                task: TASK,
                object_ref: 7,
                resource: Some(resource),
            })
        );
        assert_eq!(
            object_reference(LifecycleKind::ReplacePhysical, &payload, &Everything),
            Err(ResolveRefusal::NeedsStorage {
                kind: LifecycleKind::ReplacePhysical,
                task: TASK,
                resource,
            })
        );
    }

    /// The real namespace resolves a delete, and stops resolving it once the
    /// guest has deleted it.
    ///
    /// Every other test here uses a stub that answers about everything or
    /// nothing. This one drives `operation` from `crate::namespace::Namespace`,
    /// which is what the trait is for: the resolution path had no implementation
    /// but stubs, so nothing checked that what the namespace actually holds is
    /// what a command resolves against.
    #[test]
    fn a_delete_resolves_against_the_namespace_that_declared_the_object() {
        let mut names = crate::namespace::Namespace::new();
        let declared = names.declare(ObjectListRef(7), Some(crate::access::BackingId(10)));
        assert_eq!(declared.displaced, None, "a free slot");
        let id = declared.id;
        let mut payload = Vec::new();
        payload.extend_from_slice(&TASK.0.to_le_bytes());
        payload.extend_from_slice(&7u32.to_le_bytes());

        assert_eq!(
            operation(
                LifecycleKind::DeleteResource,
                &payload,
                &crate::resolve::SameForEveryTask(&names),
                &EveryMapping
            ),
            Ok(LifecycleOp::DeleteResource {
                task: TASK,
                object_ref: 7,
                resource: Some(id),
            })
        );

        // The generation is the object's own, not a number this test chose: a
        // second declaration in the same slot is a different name, and work
        // still carrying the first one no longer resolves to it.
        assert_eq!(
            names.delete(id).expect("declared"),
            crate::namespace::Teardown::Now {
                backing: crate::access::BackingId(10)
            },
            "nothing here acquired the lease, so the backing is owed to nobody"
        );
        assert_eq!(
            operation(
                LifecycleKind::DeleteResource,
                &payload,
                &crate::resolve::SameForEveryTask(&names),
                &EveryMapping
            ),
            Ok(LifecycleOp::DeleteResource {
                task: TASK,
                object_ref: 7,
                resource: None,
            }),
            "a deleted slot stops *naming* — and the delete is still the \
             operation, because the rail's teardown is keyed by the reference"
        );
        let redeclared = names.declare(ObjectListRef(7), Some(crate::access::BackingId(11)));
        assert_eq!(
            redeclared.displaced, None,
            "the slot's occupant was deleted above, so this declaration displaces nothing"
        );
        let again = redeclared.id;
        assert_ne!(again, id);
        assert_eq!(
            operation(
                LifecycleKind::DeleteResource,
                &payload,
                &crate::resolve::SameForEveryTask(&names),
                &EveryMapping
            ),
            Ok(LifecycleOp::DeleteResource {
                task: TASK,
                object_ref: 7,
                resource: Some(again),
            })
        );
    }

    /// A destroy names the kind that ends, and an unsettled kind refuses the
    /// packet rather than the class.
    ///
    /// The whole reason this join reads two layers. Six of the eleven kinds are
    /// settled on their own serializer rows and five have never been observed;
    /// holding the *class* out of the model on their account kept 5.5% of a
    /// real guest's stream from ever reaching it, which is what
    /// `CmdDeleteObject`'s packet row said and what it stopped saying.
    #[test]
    fn a_destroy_carries_its_kind_and_an_unsettled_kind_refuses_this_packet_alone() {
        use crate::resolve::TaskNamespaces as _;
        use reims_vgpu_protocol::destroy::DestroyKind;

        let payload = |kind_opcode: u32, object_ref: u32| {
            let rec = fifo::DELETE_OBJECT_RECORD;
            let mut out = vec![0u8; rec + 12];
            out[fifo::DELETE_OBJECT_TASK_ID..fifo::DELETE_OBJECT_TASK_ID + 4]
                .copy_from_slice(&TASK.0.to_le_bytes());
            out[rec..rec + 4].copy_from_slice(&kind_opcode.to_le_bytes());
            out[rec + 4..rec + 8].copy_from_slice(&12u32.to_le_bytes());
            out[rec + 8..rec + 12].copy_from_slice(&object_ref.to_le_bytes());
            out
        };

        // A settled kind. The operation carries it, because which per-kind
        // registry the rail drops is not re-derivable from a model that kept
        // only the name.
        assert_eq!(
            object_destroy(
                LifecycleKind::DeleteObject,
                &payload(DestroyKind::SamplerState.opcode(), 9),
                &Everything
            ),
            Ok(LifecycleOp::DeleteObject {
                task: TASK,
                object_ref: 9,
                resource: Some(Everything.resource(TASK, 9).expect("everything resolves")),
                kind: DestroyKind::SamplerState,
            })
        );

        // A reference nothing names. **Not refused**: a destroy is the one
        // lifecycle command whose effect must not be lost to an unresolvable
        // reference, because every rail's retirement is keyed by `(task, ref)`
        // and the reference is right there. A driven macos-15 boot sends seven
        // of these — `delete_unnamed_and_unresolvable`, of which two are fences
        // — and losing a fence's retirement is a *wrong answer* rather than a
        // dropped one: the stored generation outlives the fence and the next
        // fence to reuse that reference starts life already signalled.
        assert_eq!(
            object_destroy(
                LifecycleKind::DeleteObject,
                &payload(DestroyKind::Fence.opcode(), 9),
                &Nothing
            ),
            Ok(LifecycleOp::DeleteObject {
                task: TASK,
                object_ref: 9,
                resource: None,
                kind: DestroyKind::Fence,
            })
        );
        // And it is ordered against nothing, which is the honest answer: a
        // resource this model cannot name is one no other operation in it names
        // either, so there is no access for the delete to conflict with.
        let unnamed = LifecycleOp::DeleteObject {
            task: TASK,
            object_ref: 9,
            resource: None,
            kind: DestroyKind::Fence,
        };
        assert!(unnamed.resources().is_empty());
        assert_eq!(unnamed.task(), TASK);

        // An unsettled one. Refused by name, and by *its* name rather than the
        // packet's — a reader has to be able to tell which of the eleven is
        // holding the row open.
        assert_eq!(
            object_destroy(
                LifecycleKind::DeleteObject,
                &payload(DestroyKind::Heap.opcode(), 9),
                &Everything
            ),
            Err(ResolveRefusal::UnsettledDestroy {
                kind: DestroyKind::Heap
            }),
            "a kind no driven boot has ever sent must not be acted on"
        );

        // Bytes that are not a destroy at all keep the decoder's own reason,
        // which already separates a number inside the destroy span that nothing
        // has measured from a record of another family.
        assert_eq!(
            object_destroy(LifecycleKind::DeleteObject, &payload(1, 9), &Everything),
            Err(ResolveRefusal::Destroy(
                reims_vgpu_protocol::destroy::DestroyRefusal::NotADestroy { opcode: 1 }
            ))
        );

        // And the join refuses a kind that is not its own, like every other.
        assert_eq!(
            object_destroy(
                LifecycleKind::DeleteResource,
                &payload(DestroyKind::SamplerState.opcode(), 9),
                &Everything
            ),
            Err(ResolveRefusal::NotAnObjectDestroy {
                kind: LifecycleKind::DeleteResource
            })
        );
    }

    /// A ref naming nothing live is a refusal for the re-point and an answer
    /// for the delete, and the two kinds share one record.
    ///
    /// **The asymmetry is the point.** A re-point with no resource to re-point
    /// is not a re-point: the operation's whole content is moving one named
    /// resource's storage, so an unresolvable ref leaves nothing to carry. A
    /// delete's effect is keyed by the reference — every rail's teardown for it
    /// is `(task, ref)` and not a name — and the guest clears its own
    /// object-list slot before sending the delete, so refusing on the name
    /// throws away the packet for the ordinary case. See
    /// [`LifecycleOp::DeleteResource::resource`], and
    /// [`LifecycleOp::DeleteObject::resource`] for the destroy that established
    /// this shape.
    #[test]
    fn an_unnamed_ref_refuses_a_repoint_and_still_deletes() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&TASK.0.to_le_bytes());
        payload.extend_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            object_reference(LifecycleKind::ReplacePhysical, &payload, &Nothing),
            Err(ResolveRefusal::UnknownRef { object_ref: 7 })
        );
        let deleted = object_reference(LifecycleKind::DeleteResource, &payload, &Nothing);
        assert_eq!(
            deleted,
            Ok(LifecycleOp::DeleteResource {
                task: TASK,
                object_ref: 7,
                resource: None,
            })
        );
        // And it orders against nothing, which is what makes admitting it safe:
        // a resource this model cannot name is one no other operation in it can
        // name either.
        assert!(deleted.expect("an operation").resources().is_empty());
    }

    /// Each join refuses every kind that is not its own, and the kinds that
    /// still have no join are named.
    ///
    /// Two claims in one test because they are the same census. A kind read by
    /// two joins would have one command's offsets applied to another's payload;
    /// a kind read by none is unfinished work, and listing them here is what
    /// keeps "which lifecycle commands can still not become an operation" an
    /// answer rather than a search.
    #[test]
    fn every_lifecycle_kind_reaches_at_most_one_join() {
        // Every kind the packet ledger classifies is in the list, so a
        // thirteenth cannot skip the sweep below.
        let mut classified: Vec<LifecycleKind> = LEDGER
            .iter()
            .filter_map(|p| LifecycleKind::of(p.channel, p.opcode))
            .collect();
        classified.sort_unstable();
        classified.dedup();
        let mut named = ALL_KINDS.to_vec();
        named.sort_unstable();
        assert_eq!(classified, named);

        // Each kind's own well-formed payload, so a join that refuses refuses
        // on the *kind* and not on bytes that happen not to be its record. A
        // single all-zero buffer used to serve here, and it stopped serving
        // when a command arrived whose payload contains a second framed record:
        // zeros are a valid `{task, object}` and are not a valid destroy, so
        // the sweep would have read "no join" where the join exists and the
        // fixture was wrong.
        let payload = vec![0u8; 64];
        let mut unjoined = Vec::new();
        for kind in ALL_KINDS {
            let own = one_payload_per_kind(kind);
            let joins = [
                task_lifetime(kind, &own).is_ok(),
                object_reference(kind, &own, &Everything).is_ok(),
                map_notice(kind, &own).is_ok(),
                backing_retirement(kind, &own, &EveryMapping).is_ok(),
                resource_list(kind, &own, &Everything).is_ok(),
                object_list_bind(kind, &own).is_ok(),
                object_destroy(kind, &own, &Everything).is_ok(),
            ];
            let reached = joins.iter().filter(|ok| **ok).count();
            assert!(
                reached <= 1,
                "{} is read by more than one join",
                kind.name()
            );
            if reached == 0 {
                unjoined.push(kind);
            }
        }
        assert_eq!(
            unjoined,
            vec![
                // The last one. Its record resolves and its operation still
                // needs storage the wire does not carry — see
                // `ResolveRefusal::NeedsStorage`.
                //
                // `SetObjectList` left this list when its operation stopped
                // being the table's walk and became the table's *binding*; see
                // `LifecycleOp::BindObjectList` for the measurement that
                // decided it.
                LifecycleKind::ReplacePhysical,
            ],
            "the lifecycle commands that still cannot become an operation"
        );

        // Each join names *itself* when it is the one asked the wrong
        // question. They are read from one dispatcher, so a refusal that named
        // a different join sent a reader to look for a record the packet does
        // not carry either — which is what `task_lifetime` did, borrowing
        // `object_reference`'s name for want of its own.
        assert_eq!(
            task_lifetime(LifecycleKind::MapMemory, &payload),
            Err(ResolveRefusal::NotATaskLifetime {
                kind: LifecycleKind::MapMemory
            })
        );
        assert_eq!(
            object_reference(LifecycleKind::DefineTask, &payload, &Everything),
            Err(ResolveRefusal::NotAnObjectReference {
                kind: LifecycleKind::DefineTask
            })
        );
        assert_eq!(
            map_notice(LifecycleKind::DefineTask, &payload),
            Err(ResolveRefusal::NotAMapNotice {
                kind: LifecycleKind::DefineTask
            })
        );

        // And the dispatcher agrees with the sweep, kind for kind. The sweep
        // above says no kind is read by two joins; this says which one each is
        // read by, which is the half no caller could answer for itself.
        for kind in ALL_KINDS {
            let own = one_payload_per_kind(kind);
            let direct = [
                task_lifetime(kind, &own),
                object_reference(kind, &own, &Everything),
                map_notice(kind, &own),
                backing_retirement(kind, &own, &EveryMapping),
                resource_list(kind, &own, &Everything),
                object_list_bind(kind, &own),
                object_destroy(kind, &own, &Everything),
            ]
            .into_iter()
            .find(Result::is_ok);
            let through = operation(kind, &own, &Everything, &EveryMapping);
            match direct {
                Some(op) => assert_eq!(through, op, "{}", kind.name()),
                // The one with no join names what is missing rather than
                // reaching a join that would read another command's offsets.
                None => assert!(
                    matches!(through, Err(ResolveRefusal::NeedsStorage { .. })),
                    "{} became {through:?}",
                    kind.name()
                ),
            }
        }
    }

    /// The notice's fields as the packet carries them, and the direction the
    /// opcode carries.
    ///
    /// The interval is the whole payload: a `u32` task and two `u64`s. Reading
    /// the base as a `u32` — the shape the model asserted when it called this
    /// command a per-object mapping — would put the length word inside the
    /// address, so the high half is non-zero here.
    #[test]
    fn a_map_notice_is_a_task_and_an_interval_and_a_direction() {
        let mut payload = vec![0u8; fifo::MAP_MEMORY_LEN];
        payload[fifo::MAP_MEMORY_TASK_ID..fifo::MAP_MEMORY_TASK_ID + 4]
            .copy_from_slice(&TASK.0.to_le_bytes());
        payload[fifo::MAP_MEMORY_GVA..fifo::MAP_MEMORY_GVA + 8]
            .copy_from_slice(&0x0000_7f12_3456_1000u64.to_le_bytes());
        payload[fifo::MAP_MEMORY_LENGTH..fifo::MAP_MEMORY_LENGTH + 8]
            .copy_from_slice(&0x01c3_e000u64.to_le_bytes());
        let span = GuestSpan {
            base: 0x0000_7f12_3456_1000,
            length: 0x01c3_e000,
        };
        assert_eq!(
            map_notice(LifecycleKind::MapMemory, &payload),
            Ok(LifecycleOp::MapMemory { task: TASK, span })
        );
        assert_eq!(
            map_notice(LifecycleKind::UnmapMemory, &payload),
            Ok(LifecycleOp::UnmapMemory { task: TASK, span }),
            "the same record; the opcode is the whole difference"
        );
    }

    /// Every other lifecycle kind refuses by name rather than reading this
    /// record's offsets out of a payload that is not one.
    #[test]
    fn only_the_two_map_opcodes_carry_a_notice() {
        let payload = vec![0u8; fifo::MAP_MEMORY_LEN];
        for kind in [
            LifecycleKind::DefineTask,
            LifecycleKind::DeleteResource,
            LifecycleKind::ReplacePhysical,
            LifecycleKind::Invalidate,
            LifecycleKind::DeleteBacking,
        ] {
            assert_eq!(
                map_notice(kind, &payload),
                Err(ResolveRefusal::NotAMapNotice { kind })
            );
        }
    }

    /// A notice too short to hold its interval refuses; it does not become an
    /// interval with a zero-filled tail.
    #[test]
    fn a_short_notice_is_refused_and_not_zero_filled() {
        let payload = vec![0u8; fifo::MAP_MEMORY_LEN - 1];
        let refusal = map_notice(LifecycleKind::UnmapMemory, &payload)
            .expect_err("one byte short of the interval");
        assert_eq!(
            refusal,
            ResolveRefusal::ShortNotice(fifo::ShortPayload {
                plen: fifo::MAP_MEMORY_LEN - 1,
                need: fifo::MAP_MEMORY_LEN,
            })
        );
        assert_eq!(refusal.slug(), fifo::ShortPayload::SLUG);
    }

    /// Both directions leave the same obligation, and it names the interval the
    /// guest changed.
    ///
    /// The model holds nothing keyed by a guest address, so there is nothing
    /// here to invalidate — which is exactly why the obligation has to leave
    /// the crate named. A version that quietly returned no effects would be
    /// indistinguishable from one that had already discharged it.
    #[test]
    fn a_remap_leaves_the_interval_as_an_obligation() {
        let (mut l, _) = with_one_resource(256);
        let span = GuestSpan {
            base: 0x7f00_0000,
            length: 0x4000,
        };
        for (op, established) in [
            (LifecycleOp::MapMemory { task: TASK, span }, true),
            (LifecycleOp::UnmapMemory { task: TASK, span }, false),
        ] {
            let effects = l.apply(&op).expect("a live task");
            assert_eq!(
                effects.remapped,
                vec![Remap {
                    task: TASK,
                    span,
                    established,
                }]
            );
            assert!(
                effects.transfers.is_empty() && effects.teardowns.is_empty(),
                "a translation change moves no bytes and retires no backing"
            );
        }
    }

    /// An interval in a task that does not exist names no address space.
    #[test]
    fn a_remap_of_a_dead_task_is_refused() {
        let (mut l, _) = with_one_resource(256);
        let task = TaskId(TASK.0 + 1);
        assert_eq!(
            l.apply(&LifecycleOp::MapMemory {
                task,
                span: GuestSpan {
                    base: 0x1000,
                    length: 0x1000,
                },
            }),
            Err(Refusal::NoSuchTask { task })
        );
    }

    /// Two intervals of one task's space overlap on an address, and a
    /// zero-length one names none.
    #[test]
    fn an_empty_interval_overlaps_nothing_including_itself() {
        let a = GuestSpan {
            base: 0x1000,
            length: 0x1000,
        };
        assert!(a.overlaps(GuestSpan {
            base: 0x1fff,
            length: 1
        }));
        assert!(!a.overlaps(GuestSpan {
            base: 0x2000,
            length: 1
        }));
        let empty = GuestSpan {
            base: 0x1000,
            length: 0,
        };
        assert!(!empty.overlaps(empty));
        assert!(!a.overlaps(empty));
    }

    #[test]
    fn a_physical_replacement_moves_the_content_authority_with_the_pages() {
        let (mut l, id) = with_one_resource(256);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        let effects = l
            .apply(&LifecycleOp::ReplacePhysical {
                task: TASK,
                resource: id,
                backing: BackingId(20),
                extent: range(0, 256),
            })
            .expect("resolves");
        assert_eq!(
            effects.teardowns,
            vec![Teardown::Now {
                backing: BackingId(10)
            }]
        );
        assert!(
            l.content()
                .is_fresh(BackingId(20), range(0, 256), Replica::GuestPages),
            "the new pages are the guest's"
        );
        assert_eq!(
            l.resolve(TASK, id),
            Ok(range(0, 256)),
            "and the name resolves to them"
        );
    }

    #[test]
    fn every_refusal_has_its_own_slug() {
        let all = [
            Refusal::NoSuchTask { task: TASK },
            Refusal::Namespace {
                task: TASK,
                refusal: namespace::Refusal::NotDeclared {
                    slot: ObjectListRef(0),
                },
            },
            Refusal::Heap {
                task: TASK,
                refusal: heap::Refusal::NoSuchHeap { heap: 0 },
            },
            Refusal::PlacedResourceHasNoPhysical { resource: name(0) },
        ];
        let mut slugs: Vec<&str> = all.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count);
    }
    /// The check `record_write` exists for. A heap-placed resource's
    /// neighbour begins one byte past its end.
    #[test]
    fn a_recorded_write_cannot_leave_its_resource() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        l.declare_heap(TASK, 3, BackingId(50), 4096).expect("heap");
        for (slot, offset) in [(0u32, 0u64), (1, 256)] {
            apply_inert(
                &mut l,
                &LifecycleOp::CreateResource {
                    task: TASK,
                    slot: ObjectListRef(slot),
                    storage: Storage::Placed {
                        heap: 3,
                        offset,
                        length: 256,
                    },
                },
            );
        }
        assert_eq!(
            l.record_write(TASK, name(0), 128, 256, Replica::DeviceOwned),
            Err(Refusal::Heap {
                task: TASK,
                refusal: heap::Refusal::OutOfPlacement {
                    offset: 128,
                    length: 256,
                    placement_length: 256,
                },
            }),
            "a write that ran into the neighbour is refused, not clamped"
        );
        assert!(
            l.content()
                .is_fresh(BackingId(50), range(256, 256), Replica::GuestPages),
            "and the neighbour still holds its own content"
        );
        // The same window inside the resource lands at the heap offset.
        l.record_write(TASK, name(0), 128, 128, Replica::DeviceOwned)
            .expect("inside the resource");
        assert!(l
            .content()
            .is_fresh(BackingId(50), range(128, 128), Replica::DeviceOwned));
        assert!(
            l.content()
                .is_fresh(BackingId(50), range(256, 256), Replica::GuestPages),
            "and stopped at its own end"
        );
    }

    // ------------------------------------------- bytes to a lifecycle op

    /// Every lifecycle kind, so a test over the set cannot silently miss one
    /// that is added later.
    ///
    /// Held against [`LifecycleKind::of`] by
    /// `every_lifecycle_kind_reaches_at_most_one_join`, which requires every
    /// kind the packet ledger classifies to appear here — so a thirteenth kind
    /// cannot quietly skip a join test.
    const ALL_KINDS: [LifecycleKind; 13] = [
        LifecycleKind::DefineTask,
        LifecycleKind::DeleteTask,
        LifecycleKind::SetObjectList,
        LifecycleKind::DeleteResource,
        LifecycleKind::MapMemory,
        LifecycleKind::UnmapMemory,
        LifecycleKind::ReplacePhysical,
        LifecycleKind::Invalidate,
        LifecycleKind::Synchronize,
        LifecycleKind::SynchronizeAndDiscard,
        LifecycleKind::Discard,
        LifecycleKind::DeleteBacking,
        LifecycleKind::DeleteObject,
    ];

    /// One well-formed payload per kind, each naming task [`TASK_UNDER_TEST`].
    ///
    /// Written per kind rather than once, because the task word is not at one
    /// offset: a definition doubles it and puts a class bit under it, and a
    /// backing retirement carries `{object, task}` while a delete carries
    /// `{task, object}` in the same two `u32`s. A single builder would be the
    /// second decode table `task_named` exists not to be.
    fn one_payload_per_kind(kind: LifecycleKind) -> Vec<u8> {
        let task = TASK_UNDER_TEST;
        match kind {
            LifecycleKind::DefineTask => {
                let mut out = vec![0u8; fifo::DEFINE_TASK_LEN];
                out[..4].copy_from_slice(
                    &fifo::DefineTaskId {
                        task_id: task,
                        kernel: false,
                    }
                    .to_raw()
                    .to_le_bytes(),
                );
                out
            }
            LifecycleKind::DeleteTask => task.to_le_bytes().to_vec(),
            // A task word, then a whole destroy record — the one payload here
            // with a second layer inside it. The kind is the record's opcode, so
            // the fixture picks a settled one; an unsettled kind is a refusal
            // this payload deliberately does not exercise.
            LifecycleKind::DeleteObject => {
                use reims_vgpu_protocol::destroy::DestroyKind;
                let mut out = vec![0u8; fifo::DELETE_OBJECT_RECORD + 12];
                out[fifo::DELETE_OBJECT_TASK_ID..fifo::DELETE_OBJECT_TASK_ID + 4]
                    .copy_from_slice(&task.to_le_bytes());
                let record = fifo::DELETE_OBJECT_RECORD;
                out[record..record + 4]
                    .copy_from_slice(&DestroyKind::SamplerState.opcode().to_le_bytes());
                out[record + 4..record + 8].copy_from_slice(&12u32.to_le_bytes());
                out[record + 8..record + 12].copy_from_slice(&7u32.to_le_bytes());
                out
            }
            LifecycleKind::SetObjectList => {
                let mut out = vec![0u8; fifo::SET_OBJECT_LIST_LEN];
                out[fifo::SET_OBJECT_LIST_TASK_ID..fifo::SET_OBJECT_LIST_TASK_ID + 4]
                    .copy_from_slice(&task.to_le_bytes());
                out
            }
            LifecycleKind::DeleteResource | LifecycleKind::ReplacePhysical => {
                let mut out = Vec::new();
                out.extend_from_slice(&task.to_le_bytes());
                out.extend_from_slice(&7u32.to_le_bytes());
                out
            }
            LifecycleKind::MapMemory | LifecycleKind::UnmapMemory => {
                let mut out = vec![0u8; fifo::MAP_MEMORY_LEN];
                out[fifo::MAP_MEMORY_TASK_ID..fifo::MAP_MEMORY_TASK_ID + 4]
                    .copy_from_slice(&task.to_le_bytes());
                out
            }
            LifecycleKind::DeleteBacking => {
                let mut out = vec![0u8; fifo::DELETE_BACKING_LEN];
                out[fifo::DELETE_BACKING_TASK_ID..fifo::DELETE_BACKING_TASK_ID + 4]
                    .copy_from_slice(&task.to_le_bytes());
                out
            }
            LifecycleKind::Invalidate => list_bytes(task, &[7], fifo::CHILD_INVALIDATE_RECORD_LEN),
            LifecycleKind::Synchronize
            | LifecycleKind::SynchronizeAndDiscard
            | LifecycleKind::Discard => list_bytes(task, &[7], 4),
        }
    }

    /// The task a packet names is the task the operation it would build names.
    ///
    /// `task_named` drops the resolution and keeps the layout, and the failure
    /// that matters is it keeping the *wrong* layout — a retirement read at the
    /// delete's offsets answers the object id as a task, and every arm of the
    /// census built on it would then ask whether a task numbered like a surface
    /// was defined. One task id, every kind, both readers.
    ///
    /// `ReplacePhysical` is here too, and it is the reason this function exists
    /// separately at all: its operation refuses for want of storage the wire
    /// does not carry, and its *task* is on the wire like everyone else's.
    #[test]
    fn task_named_agrees_with_the_operation_it_would_build() {
        for kind in ALL_KINDS {
            let payload = one_payload_per_kind(kind);
            let named = task_named(kind, &payload)
                .unwrap_or_else(|refusal| panic!("{} names a task: {refusal:?}", kind.name()));
            assert_eq!(
                named,
                TaskId(TASK_UNDER_TEST),
                "{} read its task word from the wrong offset",
                kind.name()
            );
            match operation(kind, &payload, &Everything, &EveryMapping) {
                Ok(op) => assert_eq!(
                    op.task(),
                    named,
                    "{} builds an operation about a different task than it names",
                    kind.name()
                ),
                // The one kind with no operation. Its task is still answered
                // above, which is the whole point — and the refusal now carries
                // that same task, so the caller that supplies the storage never
                // has to resolve it a second time.
                Err(ResolveRefusal::NeedsStorage {
                    kind: refused_kind,
                    task,
                    ..
                }) => {
                    assert_eq!(
                        (kind, refused_kind),
                        (
                            LifecycleKind::ReplacePhysical,
                            LifecycleKind::ReplacePhysical
                        )
                    );
                    assert_eq!(
                        task, named,
                        "the refusal carries a different task than the packet names"
                    );
                }
                Err(other) => panic!("{} refused as {other:?}", kind.name()),
            }
        }
    }

    /// A number no kind's other field carries, so an arm reading the wrong
    /// offset answers something else and the test above sees it.
    const TASK_UNDER_TEST: u32 = 5;

    /// A resolver that answers every ref, so a resolution test is about the
    /// list and not about whether the objects exist.
    struct Everything;

    impl crate::resolve::RefResolver for Everything {
        fn resource(&self, object_ref: u32) -> Option<ResourceId> {
            Some(ResourceId {
                slot: ObjectListRef(object_ref),
                generation: SlotGeneration(1),
            })
        }
    }

    /// A mapper that resolves every mapping, and one that resolves none.
    ///
    /// Deliberately answers a *different* backing than [`Everything`]'s slot
    /// numbers would suggest: a test that used one number for both namespaces
    /// would pass against a resolver that confused them.
    struct EveryMapping;

    impl crate::resolve::MappingResolver for EveryMapping {
        fn backing(&self, mapping: crate::identity::MappingId) -> Option<BackingId> {
            Some(BackingId(u64::from(mapping.0) + 1_000))
        }
    }

    struct NoMapping;

    impl crate::resolve::MappingResolver for NoMapping {
        fn backing(&self, _mapping: crate::identity::MappingId) -> Option<BackingId> {
            None
        }
    }

    /// A resolver whose slots are all empty.
    struct Nothing;

    impl crate::resolve::RefResolver for Nothing {
        fn resource(&self, _object_ref: u32) -> Option<ResourceId> {
            None
        }
    }

    // These two stand for every task, and say so: a suite about list framing or
    // record stride has one namespace and no routing question in it.
    impl crate::resolve::TaskNamespaces for Nothing {
        fn resource(&self, _task: TaskId, _object_ref: u32) -> Option<ResourceId> {
            None
        }
    }

    impl crate::resolve::TaskNamespaces for Everything {
        fn resource(&self, _task: TaskId, object_ref: u32) -> Option<ResourceId> {
            crate::resolve::RefResolver::resource(self, object_ref)
        }
    }

    /// Two tasks' namespaces, in which one ref number names a different
    /// resource in each — which is the ordinary case, since a ref is an index
    /// into the naming task's own object list.
    ///
    /// The generation carries the task that answered, so a test can say *which*
    /// namespace resolved a ref rather than only that one did.
    struct PerTask;

    impl crate::resolve::TaskNamespaces for PerTask {
        fn resource(&self, task: TaskId, object_ref: u32) -> Option<ResourceId> {
            Some(ResourceId {
                slot: ObjectListRef(object_ref),
                generation: SlotGeneration(u64::from(task.0) + 1),
            })
        }
    }

    /// A lifecycle packet's refs resolve in the namespace of the task the packet
    /// names, and the caller does not get to choose.
    ///
    /// **The wrong namespace does not refuse.** Object-list refs are per-task
    /// slot numbers, so resolving ref 7 in another task's list succeeds and
    /// returns another task's resource — and the operation then retires,
    /// invalidates or synchronizes storage the packet never named. Nothing
    /// downstream can catch that: by then it is a well-formed operation over a
    /// live resource.
    ///
    /// So the binding is not a caller's discipline. The join decodes the task
    /// and binds the namespace itself, and this asserts that the answer moves
    /// with the packet's task word.
    #[test]
    fn a_lifecycle_packets_refs_resolve_in_the_task_the_packet_names() {
        const REF: u32 = 7;
        let named = |task: u32| {
            let mut payload = Vec::new();
            payload.extend_from_slice(&task.to_le_bytes());
            payload.extend_from_slice(&REF.to_le_bytes());
            object_reference(LifecycleKind::DeleteResource, &payload, &PerTask)
                .expect("well formed")
        };

        assert_eq!(
            named(5),
            LifecycleOp::DeleteResource {
                task: TaskId(5),
                object_ref: REF,
                resource: Some(ResourceId {
                    slot: ObjectListRef(REF),
                    generation: SlotGeneration(6),
                }),
            }
        );
        assert_eq!(
            named(9),
            LifecycleOp::DeleteResource {
                task: TaskId(9),
                object_ref: REF,
                resource: Some(ResourceId {
                    slot: ObjectListRef(REF),
                    generation: SlotGeneration(10),
                }),
            },
            "the same ref number, a different packet task, a different resource \u{2014}              which is what makes a caller-chosen namespace a wrong answer rather              than a refusal"
        );

        // The same, through the counted-list join, whose task word is in a
        // different place in a different record.
        let mut list = Vec::new();
        list.extend_from_slice(&11u32.to_le_bytes());
        list.extend_from_slice(&1u32.to_le_bytes());
        list.extend_from_slice(&REF.to_le_bytes());
        let LifecycleOp::Synchronize { task, resources } =
            resource_list(LifecycleKind::Synchronize, &list, &PerTask).expect("well formed")
        else {
            panic!("a synchronize builds a synchronize");
        };
        assert_eq!(task, TaskId(11));
        assert_eq!(
            resources,
            vec![ResourceId {
                slot: ObjectListRef(REF),
                generation: SlotGeneration(12),
            }],
            "the list's refs resolve in the list's own task"
        );
    }

    /// A payload too short to hold even a resource-list header.
    fn alloc_short() -> Vec<u8> {
        vec![0u8; 3]
    }

    /// A resource-list payload: `{task, count}` then `count` records of
    /// `record_len`, with the object ref in the first word of each.
    fn list_bytes(task: u32, refs: &[u32], record_len: u32) -> Vec<u8> {
        list_bytes_with(task, refs, record_len, PAGEON_DWORD)
    }

    /// The guest-write quad, as `pageBacking` writes it.
    const PAGEON_DWORD: u32 = 0x0100_0001;

    /// [`list_bytes`], with the eight-byte record's validity quad stated.
    ///
    /// The quad is the record's, not the packet's, so a fixture that left it
    /// zero was feeding a transition the guest never asks for.
    fn list_bytes_with(task: u32, refs: &[u32], record_len: u32, ops: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&task.to_le_bytes());
        out.extend_from_slice(&u32::try_from(refs.len()).expect("small").to_le_bytes());
        for r in refs {
            out.extend_from_slice(&r.to_le_bytes());
            if record_len == 8 {
                out.extend_from_slice(&ops.to_le_bytes());
            } else {
                out.resize(out.len() + record_len as usize - 4, 0);
            }
        }
        out
    }

    /// Exactly four of the twelve lifecycle commands carry a counted resource
    /// list, and the two record lengths are not interchangeable: `Invalidate`
    /// puts the guest's validity quad beside each ref. Reading one command's
    /// list at the other's stride walks off the records.
    #[test]
    fn exactly_the_four_list_commands_have_a_record_length() {
        let with: Vec<LifecycleKind> = LEDGER
            .iter()
            .filter_map(|p| LifecycleKind::of(p.channel, p.opcode))
            .filter(|k| k.resource_list_record_len().is_some())
            .collect();
        assert_eq!(
            with,
            vec![
                LifecycleKind::Invalidate,
                LifecycleKind::Synchronize,
                LifecycleKind::SynchronizeAndDiscard,
                LifecycleKind::Discard,
            ]
        );
        assert_eq!(
            LifecycleKind::Invalidate.resource_list_record_len(),
            Some(8)
        );
        for kind in [
            LifecycleKind::Synchronize,
            LifecycleKind::SynchronizeAndDiscard,
            LifecycleKind::Discard,
        ] {
            assert_eq!(kind.resource_list_record_len(), Some(4), "{}", kind.name());
        }
    }

    /// The join this function exists to be: a guest's bytes become the
    /// operation the model names, with every ref resolved.
    #[test]
    fn a_resource_list_payload_becomes_the_operation_its_command_names() {
        for kind in [
            LifecycleKind::Invalidate,
            LifecycleKind::Synchronize,
            LifecycleKind::SynchronizeAndDiscard,
            LifecycleKind::Discard,
        ] {
            let stride = kind.resource_list_record_len().expect("a list command");
            let bytes = list_bytes(7, &[11, 12, 13], stride);
            let op = resource_list(kind, &bytes, &Everything).expect("well formed");
            assert_eq!(op.kind(), kind, "the op is the command's own");
            assert_eq!(op.task(), TaskId(7));
            let resources = match &op {
                LifecycleOp::Invalidate { resources, .. }
                | LifecycleOp::Synchronize { resources, .. }
                | LifecycleOp::SynchronizeAndDiscard { resources, .. }
                | LifecycleOp::Discard { resources, .. } => resources.clone(),
                other => panic!("{other:?} is not a list operation"),
            };
            assert_eq!(
                resources.iter().map(|r| r.slot.0).collect::<Vec<_>>(),
                vec![11, 12, 13],
                "{}",
                kind.name()
            );
        }
    }

    /// An `EXEC_INDIRECT2` payload: the three header words, then one 24-byte
    /// record per `(object_ref, quad)`.
    fn exec_bytes(task: u32, records: &[(u32, u32)]) -> Vec<u8> {
        use reims_vgpu_protocol::fifo::{
            CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN, CHILD_EXEC_RESOURCE_TAIL_LEN,
        };
        let mut out = Vec::new();
        out.extend_from_slice(&task.to_le_bytes());
        out.extend_from_slice(&u32::try_from(records.len()).expect("small").to_le_bytes());
        // The command-buffer count. Not this function's table, and stated so a
        // record is never read out of a word that was meant to be a count.
        out.extend_from_slice(&0u32.to_le_bytes());
        for (object_ref, ops) in records {
            out.extend_from_slice(&object_ref.to_le_bytes());
            out.extend_from_slice(&ops.to_le_bytes());
            out.resize(out.len() + CHILD_EXEC_RESOURCE_TAIL_LEN as usize, 0);
        }
        assert_eq!(
            out.len(),
            12 + records.len() * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize
        );
        out
    }

    /// An EXEC packet's own resource table carries the same guest-write
    /// statement a standalone invalidate does, and it becomes the same
    /// operation.
    ///
    /// Two content-authority models — one for each packet the guest could have
    /// used to say the same thing — is the shape this avoids.
    #[test]
    fn an_exec_tables_guest_write_becomes_the_same_invalidate() {
        let bytes = exec_bytes(1, &[(11, PAGEON_DWORD), (12, PAGEON_DWORD)]);
        assert_eq!(
            exec_resource_table(&bytes, &Everything),
            Ok(LifecycleOp::Invalidate {
                task: TaskId(1),
                resources: vec![name(11), name(12)],
            })
        );
    }

    /// A record asking for no move is the normal record in this table: it lists
    /// every resource the submission touches, and most were not CPU-written.
    ///
    /// Refusing them, or treating them as guest writes, are both wrong in the
    /// same direction — one loses every EXEC, the other declares a device
    /// replica stale on every submission and charges the rebuild to the next
    /// reader.
    #[test]
    fn an_exec_record_asking_for_nothing_moves_nothing() {
        let bytes = exec_bytes(1, &[(11, 0), (12, PAGEON_DWORD), (13, 0)]);
        assert_eq!(
            exec_resource_table(&bytes, &Everything),
            Ok(LifecycleOp::Invalidate {
                task: TaskId(1),
                resources: vec![name(12)],
            })
        );
        // A table of nothing but used resources declares no move at all.
        let quiet = exec_bytes(1, &[(11, 0)]);
        assert_eq!(
            exec_resource_table(&quiet, &Everything),
            Ok(LifecycleOp::Invalidate {
                task: TaskId(1),
                resources: Vec::new(),
            })
        );
    }

    /// The same unestablished quad refuses here as in the standalone packet.
    #[test]
    fn an_exec_record_asking_for_an_unestablished_move_refuses() {
        let bytes = exec_bytes(1, &[(11, 0x0100_0101)]);
        assert_eq!(
            exec_resource_table(&bytes, &Everything),
            Err(ResolveRefusal::UnestablishedValidityOps {
                object_ref: 11,
                ops: 0x0100_0101,
            })
        );
    }

    /// The task is the header's. A table of refs resolved in another task's
    /// namespace names another task's resources.
    #[test]
    fn an_exec_tables_task_is_the_headers() {
        let bytes = exec_bytes(9, &[]);
        assert_eq!(
            exec_resource_table(&bytes, &Everything).map(|op| op.task()),
            Ok(TaskId(9))
        );
    }

    /// A payload too short for the header refuses on the header, and one whose
    /// declared table is not there refuses on the table.
    #[test]
    fn an_exec_payload_that_disagrees_with_itself_refuses_by_name() {
        assert!(matches!(
            exec_resource_table(&[0u8; 4], &Everything),
            Err(ResolveRefusal::ShortNotice(_))
        ));
        let mut bytes = exec_bytes(1, &[(11, PAGEON_DWORD)]);
        bytes.truncate(bytes.len() - 1);
        assert!(matches!(
            exec_resource_table(&bytes, &Everything),
            Err(ResolveRefusal::Payload(_))
        ));
    }

    /// The operation the table produces is one `Lifecycle` applies: the guest's
    /// pages become the current content of exactly the resource the record
    /// named.
    #[test]
    fn an_exec_tables_invalidate_applies_like_any_other() {
        let (mut l, id) = with_one_resource(256);
        let backing = BackingId(10);
        l.record_write(TASK, id, 0, 256, Replica::DeviceOwned)
            .expect("inside the resource");
        assert!(l
            .content()
            .is_fresh(backing, range(0, 256), Replica::DeviceOwned));

        let bytes = exec_bytes(TASK.0, &[(0, PAGEON_DWORD)]);
        let op = exec_resource_table(&bytes, &Everything).expect("well formed");
        assert_eq!(l.apply(&op).expect("resolves"), Effects::default());
        assert!(l
            .content()
            .is_fresh(backing, range(0, 256), Replica::GuestPages));
        assert!(!l
            .content()
            .is_fresh(backing, range(0, 1), Replica::DeviceOwned));
    }

    /// A record asking for a transition this device has not established refuses
    /// the packet, naming the ref and the quad that arrived.
    ///
    /// The four validity-op bytes were decoded and dropped: the operation
    /// states one transition for the whole packet and every record got it,
    /// whatever it asked for. A quad of zeros asks for no move at all and a
    /// packet of them would have marked every resource guest-written — a
    /// device replica declared stale that nothing had written.
    #[test]
    fn an_invalidate_record_asking_for_an_unestablished_move_refuses() {
        for ops in [0x0000_0000, 0x0000_0001, 0x0100_0000, 0x0001_0100] {
            let bytes = list_bytes_with(7, &[11], 8, ops);
            assert_eq!(
                resource_list(LifecycleKind::Invalidate, &bytes, &Everything),
                Err(ResolveRefusal::UnestablishedValidityOps {
                    object_ref: 11,
                    ops
                }),
                "{ops:#010x}"
            );
        }
    }

    /// One bad quad refuses the whole packet, for the reason one unresolvable
    /// ref does: the list says a set of resources moved together, and applying
    /// it to the ones that happened to be well formed claims the rest did not.
    #[test]
    fn one_unestablished_quad_refuses_the_whole_list() {
        let mut bytes = list_bytes_with(7, &[11, 12], 8, PAGEON_DWORD);
        let second_ops = bytes.len() - 4;
        bytes[second_ops..].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            resource_list(LifecycleKind::Invalidate, &bytes, &Everything),
            Err(ResolveRefusal::UnestablishedValidityOps {
                object_ref: 12,
                ops: 0
            })
        );
    }

    /// The refusal names the check that produced it.
    #[test]
    fn the_unestablished_quad_refusal_has_its_own_slug() {
        assert_eq!(
            ResolveRefusal::UnestablishedValidityOps {
                object_ref: 11,
                ops: 0
            }
            .slug(),
            "lifecycle_unestablished_validity_ops"
        );
    }

    /// The stride is the command's, and reading a list at the other command's
    /// stride is not a near miss — it walks off the records entirely.
    #[test]
    fn a_list_read_at_the_other_commands_stride_does_not_agree() {
        // Three eight-byte records, read as if they were four-byte ones.
        let bytes = list_bytes(7, &[11, 12, 13], 8);
        let wide =
            resource_list(LifecycleKind::Invalidate, &bytes, &Everything).expect("its own stride");
        let narrow =
            resource_list(LifecycleKind::Synchronize, &bytes, &Everything).expect("long enough");
        assert_ne!(
            format!("{wide:?}").replace("Invalidate", "X"),
            format!("{narrow:?}").replace("Synchronize", "X"),
            "the two strides must not read the same refs out of one payload"
        );
    }

    /// One unresolvable ref refuses the whole packet. `Invalidate` says a set
    /// of resources went stale together; applying it to the subset that
    /// happened to resolve claims the others are still fresh.
    #[test]
    fn one_unknown_ref_refuses_the_whole_list() {
        let bytes = list_bytes(7, &[11, 12], 4);
        assert_eq!(
            resource_list(LifecycleKind::Synchronize, &bytes, &Nothing),
            Err(ResolveRefusal::UnknownRef { object_ref: 11 })
        );
        assert_eq!(
            resource_list(LifecycleKind::Synchronize, &bytes, &Nothing)
                .expect_err("refused")
                .slug(),
            "lifecycle_unknown_ref"
        );
    }

    /// A payload that disagrees with itself refuses with the decoder's own
    /// reason, forwarded rather than restated.
    #[test]
    fn a_payload_that_cannot_be_read_forwards_the_decoders_reason() {
        let short = resource_list(LifecycleKind::Synchronize, &[0u8; 3], &Everything)
            .expect_err("no header");
        assert_eq!(
            short,
            ResolveRefusal::Payload(fifo::ResourceListDecodeError::ShortHeader { plen: 3 })
        );
        assert_eq!(short.slug(), "resource_list_short_header");

        // A count the payload cannot carry is the other half.
        let mut bytes = list_bytes(7, &[11], 4);
        bytes[4..8].copy_from_slice(&9u32.to_le_bytes());
        let truncated = resource_list(LifecycleKind::Synchronize, &bytes, &Everything)
            .expect_err("nine records in one record's worth of bytes");
        assert_eq!(truncated.slug(), "resource_list_truncated");
    }

    /// The eight commands that carry no list say so, rather than being read as
    /// an empty one — an empty `Invalidate` is a real packet and means nothing
    /// went stale, which is not what a `DefineTask` means.
    #[test]
    fn a_command_with_no_list_refuses_rather_than_reading_an_empty_one() {
        for kind in [
            LifecycleKind::DefineTask,
            LifecycleKind::DeleteTask,
            LifecycleKind::SetObjectList,
            LifecycleKind::DeleteResource,
            LifecycleKind::MapMemory,
            LifecycleKind::UnmapMemory,
            LifecycleKind::ReplacePhysical,
            LifecycleKind::DeleteBacking,
        ] {
            // Both a payload long enough to look like a list header and one
            // too short to be one. The reason has to be "this command has no
            // list" either way: a `DefineTask` refused for a short *list*
            // sends the reader looking for a list that was never there.
            for bytes in [list_bytes(7, &[], 4), alloc_short()] {
                assert_eq!(
                    resource_list(kind, &bytes, &Everything),
                    Err(ResolveRefusal::NotAResourceList { kind }),
                    "{} at {} bytes",
                    kind.name(),
                    bytes.len()
                );
            }
        }
        // And an empty list on a command that has one is a real operation.
        let empty = resource_list(LifecycleKind::Discard, &list_bytes(7, &[], 4), &Everything)
            .expect("an empty discard is well formed");
        assert_eq!(
            empty,
            LifecycleOp::Discard {
                task: TaskId(7),
                resources: Vec::new(),
            }
        );
    }

    const SECOND: TaskId = TaskId(2);

    /// A task holding one dedicated resource on `backing`, added to a model
    /// that already has one.
    fn second_task_on(l: &mut Lifecycle, task: TaskId, backing: u64, length: u64) -> ResourceId {
        apply_inert(
            l,
            &LifecycleOp::DefineTask {
                task,
                kernel: false,
                directory: DirectoryFrame(0x2000),
            },
        );
        apply_inert(
            l,
            &LifecycleOp::CreateResource {
                task,
                slot: ObjectListRef(0),
                storage: dedicated(backing, length),
            },
        );
        name(0)
    }

    /// **A name in one task does not take another task's content with it.**
    ///
    /// The teardown a namespace gives is a per-task answer, because a namespace
    /// is per task; the content ledger is per session, because an IOSurface
    /// backing is reachable from several. Forgetting on `Teardown::Now` reads
    /// the first answer as the second.
    ///
    /// And what is lost is not only freshness. A forgotten backing loses its
    /// version counter too, so the next write reserves a version already
    /// issued, and a reader ordered against that number is satisfied by a write
    /// that has nothing to do with it.
    #[test]
    fn deleting_a_name_in_one_task_leaves_a_second_tasks_content_alone() {
        let (mut l, first) = with_one_resource(64);
        let _second = second_task_on(&mut l, SECOND, 10, 64);
        let reserved = l.content_mut().reserve(BackingId(10));

        let effects = l
            .apply(&LifecycleOp::DeleteResource {
                task: TASK,
                object_ref: first.slot.0,
                resource: Some(first),
            })
            .expect("resolves");
        assert_eq!(
            effects.teardowns,
            vec![Teardown::Now {
                backing: BackingId(10)
            }],
            "the first task owes the teardown, because no name of *its* holds it"
        );
        assert_eq!(
            l.content().extent(BackingId(10)),
            Some(range(0, 64)),
            "the second task still names the backing"
        );
        assert_ne!(
            l.content_mut().reserve(BackingId(10)),
            reserved,
            "a version already issued was issued again"
        );
    }

    /// The same for a physical replacement, which hands back the old backing by
    /// the same per-task answer.
    #[test]
    fn repointing_a_name_in_one_task_leaves_a_second_tasks_content_alone() {
        let (mut l, first) = with_one_resource(64);
        let _second = second_task_on(&mut l, SECOND, 10, 64);
        let reserved = l.content_mut().reserve(BackingId(10));

        let effects = l
            .apply(&LifecycleOp::ReplacePhysical {
                task: TASK,
                resource: first,
                backing: BackingId(11),
                extent: range(0, 64),
            })
            .expect("resolves");
        assert_eq!(
            effects.teardowns,
            vec![Teardown::Now {
                backing: BackingId(10)
            }]
        );
        assert_eq!(l.content().extent(BackingId(10)), Some(range(0, 64)));
        assert_ne!(l.content_mut().reserve(BackingId(10)), reserved);
    }

    /// Deleting a backing is the same question asked about every task at once.
    #[test]
    fn retiring_a_backing_in_one_task_leaves_a_second_tasks_content_alone() {
        let (mut l, _first) = with_one_resource(64);
        let _second = second_task_on(&mut l, SECOND, 10, 64);
        let reserved = l.content_mut().reserve(BackingId(10));

        let effects = l
            .apply(&LifecycleOp::DeleteBacking {
                task: TASK,
                mapping: crate::identity::MappingId(10),
                backing: Some(BackingId(10)),
            })
            .expect("resolves");
        assert_eq!(effects.teardowns.len(), 1);
        assert_eq!(
            l.content().extent(BackingId(10)),
            Some(range(0, 64)),
            "the retirement is of the first task's names, not of the storage"
        );
        assert_ne!(l.content_mut().reserve(BackingId(10)), reserved);

        // And once the second task lets go, nothing holds it and the authority
        // does go.
        let effects = l
            .apply(&LifecycleOp::DeleteTask { task: SECOND })
            .expect("resolves");
        assert_eq!(
            effects.teardowns,
            vec![Teardown::Now {
                backing: BackingId(10)
            }]
        );
        assert_eq!(l.content().extent(BackingId(10)), None);
    }

    /// Deleting a task takes its content with it when nothing else holds it —
    /// the check narrows what is forgotten and must not stop it happening.
    #[test]
    fn deleting_the_last_task_that_names_a_backing_does_forget_it() {
        let (mut l, _first) = with_one_resource(64);
        assert_eq!(l.content().extent(BackingId(10)), Some(range(0, 64)));
        let effects = l
            .apply(&LifecycleOp::DeleteTask { task: TASK })
            .expect("resolves");
        assert_eq!(
            effects.teardowns,
            vec![Teardown::Now {
                backing: BackingId(10)
            }]
        );
        assert_eq!(l.content().extent(BackingId(10)), None);
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

    /// Deliberately tiny pools, so tasks share backings constantly and a
    /// cross-task holder is the common case rather than a corner.
    const TASKS: u64 = 3;
    const SLOTS: u64 = 4;
    const BACKINGS: u64 = 4;
    const HEAPS: u64 = 2;
    const HEAP_LENGTH: u64 = 256;
    const RESOURCE_LENGTH: u64 = 64;

    /// One task, projected to what a caller can observe of it.
    #[derive(Debug, PartialEq, Eq)]
    struct TaskShot {
        task: TaskId,
        directory: DirectoryFrame,
        names: Vec<ResourceId>,
        resident: Vec<(ResourceId, BackingId, ByteRange)>,
        heaps: Vec<u64>,
    }

    /// The whole model, projected. Compared before and after a refusal, because
    /// "nothing is half-applied" is a claim about everything and not about the
    /// registry the refusal came from.
    #[derive(Debug, PartialEq, Eq)]
    struct Shot {
        tasks: Vec<TaskShot>,
        content: Vec<(BackingId, bool, Option<ByteRange>)>,
    }

    fn shot(l: &Lifecycle) -> Shot {
        let mut tasks: Vec<TaskShot> = l
            .tasks
            .iter()
            .map(|(task, t)| {
                let mut resident: Vec<(ResourceId, BackingId, ByteRange)> = t
                    .resident
                    .iter()
                    .map(|(id, r)| (*id, r.backing(), r.extent()))
                    .collect();
                resident.sort_unstable_by_key(|(id, _, _)| (id.slot.0, id.generation));
                TaskShot {
                    task: *task,
                    directory: t.directory,
                    names: t.namespace.live_names(),
                    resident,
                    heaps: t.heaps.live_heaps(),
                }
            })
            .collect();
        tasks.sort_unstable_by_key(|s| s.task.0);
        let content = (0..BACKINGS)
            .map(|n| {
                let b = BackingId(n);
                (b, l.content().knows(b), l.content().extent(b))
            })
            .collect();
        Shot { tasks, content }
    }

    /// A name the sweep hands out, remembered whether or not it is still live,
    /// so stale names keep reaching the operations that must refuse them.
    #[derive(Clone, Copy)]
    struct Handed {
        task: TaskId,
        id: ResourceId,
    }

    #[derive(Default)]
    struct Census {
        applied: u32,
        refused: u32,
        /// Deletes, replacements and backing retirements that resolved, which
        /// are the routes that forget content.
        deletes: u32,
        replacements: u32,
        retired_backings: u32,
        deleted_tasks: u32,
        redefinitions: u32,
        placed: u32,
        /// Steps where a backing was named by live residents of two tasks at
        /// once, which is the shape the per-task teardown answer cannot see.
        shared_backings: u32,
        declines: u32,
    }

    /// **The content authority holds exactly the backings the model still
    /// answers for, and a refused operation changes nothing.**
    ///
    /// Two claims, and the first needs a shadow the model does not have. Where
    /// a backing's bytes are is session-wide; who still names it is per task,
    /// split across two registries — a task's namespace and a task's heaps —
    /// and no single owner in the model is asked the whole question in the
    /// course of ordinary work. The shadow asks it the dumbest possible way,
    /// by scanning every task after every step, and holds the ledger to it in
    /// both directions: nothing a live name reaches may be forgotten, and
    /// nothing the model has let go of may be kept.
    ///
    /// The second claim needs no shadow at all, only a projection: a refusal
    /// compares the whole model against the whole model, so a half-applied
    /// list, a name published before a placement was checked, or a content
    /// write recorded before a refusal disagrees wherever it happened.
    ///
    /// The resident map is checked against a shadow built independently too,
    /// because two operations delete resources the caller never named — a
    /// backing retirement deletes everything that named it, and a task
    /// definition over a live task deletes everything the task had. A cascade
    /// computed wrong would otherwise only show up as a leak much later.
    #[test]
    fn the_content_authority_holds_exactly_what_the_model_still_answers_for() {
        let mut census = Census::default();
        for seed in 0..400u64 {
            let mut rng = Rng::new(seed + 1);
            let mut l = Lifecycle::new();
            // The shadow: live residents per task, and heap storage per task.
            // Flat maps, updated from the operation the sweep issued and never
            // from the model's own bookkeeping.
            let mut live: HashMap<TaskId, HashMap<ObjectListRef, (ResourceId, BackingId)>> =
                HashMap::new();
            let mut storage: HashMap<TaskId, HashMap<u64, BackingId>> = HashMap::new();
            let mut generations: HashMap<(TaskId, ObjectListRef), SlotGeneration> = HashMap::new();
            let mut handed: Vec<Handed> = Vec::new();
            // Backings the ledger knew at the previous step. A backing
            // something still holds may not stop being known.
            let mut knew: HashSet<BackingId> = HashSet::new();

            for _ in 0..64 {
                let task = TaskId(rng.below(TASKS) as u32 + 1);
                let slot = ObjectListRef(rng.below(SLOTS) as u32);
                let backing = BackingId(rng.below(BACKINGS));
                let before = shot(&l);

                // A heap declaration and a recorded write are owner calls
                // rather than packets, so they are driven apart from the
                // operation vocabulary. The write is what puts bytes in the
                // device's replica, which is the only state in which a discard
                // can be the declined kind.
                if rng.below(10) == 0 {
                    let heap = rng.below(HEAPS);
                    if l.declare_heap(task, heap, backing, HEAP_LENGTH).is_ok() {
                        storage.entry(task).or_default().insert(heap, backing);
                    } else {
                        assert_eq!(shot(&l), before, "seed {seed}: refused heap declaration");
                    }
                    check(seed, &l, &live, &storage, &mut knew, &mut census);
                    continue;
                }
                if rng.below(5) == 0 {
                    if let Some(h) = a_live_name(&live, &mut rng) {
                        let replica = if rng.below(3) == 0 {
                            Replica::GuestPages
                        } else {
                            Replica::DeviceOwned
                        };
                        if l.record_write(h.task, h.id, 0, RESOURCE_LENGTH, replica)
                            .is_err()
                        {
                            assert_eq!(shot(&l), before, "seed {seed}: refused write");
                        }
                    }
                    check(seed, &l, &live, &storage, &mut knew, &mut census);
                    continue;
                }

                let op = match rng.below(22) {
                    0 => LifecycleOp::DefineTask {
                        task,
                        kernel: false,
                        directory: DirectoryFrame(0x1000 * (rng.below(2) as u32 + 1)),
                    },
                    1 => LifecycleOp::DeleteTask { task },
                    2..=10 => LifecycleOp::CreateResource {
                        task,
                        // Mostly a slot the shadow says is free: a redeclared
                        // live slot refuses, and a driver that spent most of
                        // its creates on one would leave the model nearly
                        // empty.
                        slot: a_free_slot(&live, task, &mut rng).unwrap_or(slot),
                        storage: if rng.below(2) == 0 {
                            Storage::Placed {
                                heap: storage
                                    .get(&task)
                                    .and_then(|h| {
                                        let mut keys: Vec<u64> = h.keys().copied().collect();
                                        keys.sort_unstable();
                                        keys.get(rng.below(keys.len().max(1) as u64) as usize)
                                            .copied()
                                    })
                                    .unwrap_or_else(|| rng.below(HEAPS)),
                                // Reaches past the heap, so a placement that
                                // must be refused after its name was already
                                // published is a case the sweep drives.
                                offset: rng.below(6) * RESOURCE_LENGTH,
                                length: RESOURCE_LENGTH,
                            }
                        } else {
                            Storage::Dedicated {
                                backing,
                                extent: range(0, RESOURCE_LENGTH),
                            }
                        },
                    },
                    11..=13 => {
                        let Some(h) = a_name(&live, &handed, &mut rng) else {
                            continue;
                        };
                        LifecycleOp::DeleteResource {
                            task: h.task,
                            object_ref: h.id.slot.0,
                            resource: Some(h.id),
                        }
                    }
                    14..=16 => {
                        let Some(h) = a_name(&live, &handed, &mut rng) else {
                            continue;
                        };
                        LifecycleOp::ReplacePhysical {
                            task: h.task,
                            resource: h.id,
                            backing,
                            extent: range(0, RESOURCE_LENGTH),
                        }
                    }
                    17 => LifecycleOp::DeleteBacking {
                        task,
                        mapping: crate::identity::MappingId(backing.0 as u32),
                        backing: Some(backing),
                    },
                    18 => LifecycleOp::MapMemory {
                        task,
                        span: GuestSpan {
                            base: 0x1_0000,
                            length: 0x1000,
                        },
                    },
                    19 | 20 => {
                        let resources: Vec<ResourceId> = (0..rng.below(3))
                            .map(|_| {
                                a_name(&live, &handed, &mut rng).map_or_else(|| name(0), |h| h.id)
                            })
                            .collect();
                        // Weighted towards the discard, because it is the
                        // only one of the four whose answer depends on the
                        // state its transfers left — the declined kind needs
                        // bytes the device holds alone, which the driven
                        // writes above are what produce.
                        match rng.below(6) {
                            0 => LifecycleOp::Invalidate { task, resources },
                            1 => LifecycleOp::Synchronize { task, resources },
                            2 => LifecycleOp::SynchronizeAndDiscard { task, resources },
                            _ => LifecycleOp::Discard { task, resources },
                        }
                    }
                    _ => LifecycleOp::UnmapMemory {
                        task,
                        span: GuestSpan {
                            base: 0x1_0000,
                            length: 0x1000,
                        },
                    },
                };

                match l.apply(&op) {
                    Err(_) => {
                        census.refused += 1;
                        assert_eq!(
                            shot(&l),
                            before,
                            "seed {seed}: a refused {op:?} changed the model"
                        );
                        // Every refusal is now a whole no-op, including the
                        // one that was not. A placement used to be checked
                        // after its name had been published, so a refused one
                        // withdrew the name and left the generation spent —
                        // invisible to the snapshot above and visible to the
                        // slot's next occupant. `Heaps::admits` clears the
                        // window first, so there is nothing to withdraw.
                    }
                    Ok(effects) => {
                        census.applied += 1;
                        census.declines += l.complete(&effects).len() as u32;
                        follow(
                            &op,
                            &mut live,
                            &mut storage,
                            &mut generations,
                            &mut handed,
                            &mut census,
                        );
                    }
                }
                check(seed, &l, &live, &storage, &mut knew, &mut census);
            }
        }

        assert!(census.applied > 3000, "{}", census.applied);
        assert!(census.refused > 8000, "{}", census.refused);
        assert!(census.deletes > 300, "{}", census.deletes);
        assert!(census.replacements > 250, "{}", census.replacements);
        assert!(census.retired_backings > 120, "{}", census.retired_backings);
        assert!(census.deleted_tasks > 120, "{}", census.deleted_tasks);
        assert!(census.redefinitions > 120, "{}", census.redefinitions);
        assert!(census.placed > 200, "{}", census.placed);
        // Thin on purpose rather than by accident: a declined discard needs a
        // device write over the resource with no invalidate, synchronise or
        // guest write between, and every one of those restores the guest's
        // authority. The named test is where the case is pinned; this floor
        // only says the sweep reaches it.
        assert!(census.declines > 5, "{}", census.declines);
        assert!(
            census.shared_backings > 300,
            "no backing was ever named by two tasks at once: {}",
            census.shared_backings
        );
    }

    /// A name the shadow believes is live, so the driver spends most of its
    /// steps on work that can apply rather than on refusals.
    fn a_live_name(
        live: &HashMap<TaskId, HashMap<ObjectListRef, (ResourceId, BackingId)>>,
        rng: &mut Rng,
    ) -> Option<Handed> {
        let mut all: Vec<Handed> = live
            .iter()
            .flat_map(|(task, r)| {
                r.values().map(move |(id, _)| Handed {
                    task: *task,
                    id: *id,
                })
            })
            .collect();
        if all.is_empty() {
            return None;
        }
        // Sorted, so which name a seed picks is a property of the seed and not
        // of a hash order.
        all.sort_unstable_by_key(|h| (h.task.0, h.id.slot.0, h.id.generation));
        Some(all[rng.below(all.len() as u64) as usize])
    }

    /// A slot this task has nothing live in, so a declaration can land.
    fn a_free_slot(
        live: &HashMap<TaskId, HashMap<ObjectListRef, (ResourceId, BackingId)>>,
        task: TaskId,
        rng: &mut Rng,
    ) -> Option<ObjectListRef> {
        if rng.below(4) == 0 {
            return None;
        }
        let taken = live.get(&task);
        let free: Vec<ObjectListRef> = (0..SLOTS)
            .map(|n| ObjectListRef(n as u32))
            .filter(|slot| taken.is_none_or(|t| !t.contains_key(slot)))
            .collect();
        (!free.is_empty()).then(|| free[rng.below(free.len() as u64) as usize])
    }

    /// Mostly a live name, sometimes one the sweep handed out long ago — which
    /// is how a stale generation, a deleted name and a name from a task that
    /// has since been redefined all keep reaching the operations that have to
    /// refuse them.
    fn a_name(
        live: &HashMap<TaskId, HashMap<ObjectListRef, (ResourceId, BackingId)>>,
        handed: &[Handed],
        rng: &mut Rng,
    ) -> Option<Handed> {
        if rng.below(4) != 0 {
            if let Some(h) = a_live_name(live, rng) {
                return Some(h);
            }
        }
        if handed.is_empty() {
            return None;
        }
        Some(handed[rng.below(handed.len() as u64) as usize])
    }

    /// Update the shadow from the operation the sweep issued, computing the
    /// two cascades — a backing retirement and a redefinition — for itself.
    fn follow(
        op: &LifecycleOp,
        live: &mut HashMap<TaskId, HashMap<ObjectListRef, (ResourceId, BackingId)>>,
        storage: &mut HashMap<TaskId, HashMap<u64, BackingId>>,
        generations: &mut HashMap<(TaskId, ObjectListRef), SlotGeneration>,
        handed: &mut Vec<Handed>,
        census: &mut Census,
    ) {
        match op {
            LifecycleOp::DefineTask { task, .. } => {
                if live.insert(*task, HashMap::new()).is_some() {
                    census.redefinitions += 1;
                }
                // A new namespace and new heaps: the generations restart with
                // them, which is what makes a name from the previous
                // definition refuse rather than resolve to its successor.
                storage.insert(*task, HashMap::new());
                generations.retain(|(t, _), _| t != task);
            }
            LifecycleOp::DeleteTask { task } => {
                census.deleted_tasks += 1;
                live.remove(task);
                storage.remove(task);
                generations.retain(|(t, _), _| t != task);
            }
            LifecycleOp::CreateResource {
                task,
                slot,
                storage: what,
            } => {
                let generation = generations
                    .get(&(*task, *slot))
                    .map_or_else(|| SlotGeneration::default().next(), |g| g.next());
                generations.insert((*task, *slot), generation);
                let id = ResourceId {
                    slot: *slot,
                    generation,
                };
                let backing = match what {
                    Storage::Dedicated { backing, .. } => *backing,
                    Storage::Placed { heap, .. } => {
                        census.placed += 1;
                        storage[task][heap]
                    }
                    // Not generated by this history. It is about content
                    // authority and teardown across tasks, and a name that owns
                    // no bytes has neither — `namespace`'s own randomized
                    // history is where those are driven. The arm is spelled out
                    // rather than defaulted so that growing one here is a
                    // compile-time decision, instead of a storage-less name
                    // landing in `live` where a later step could pick it as a
                    // re-point target with no memory to re-point.
                    Storage::NoBytes => unreachable!("no storage-less names in this history"),
                };
                live.entry(*task).or_default().insert(*slot, (id, backing));
                handed.push(Handed { task: *task, id });
            }
            LifecycleOp::DeleteResource { task, resource, .. } => {
                census.deletes += 1;
                if let Some(resource) = resource {
                    live.entry(*task).or_default().remove(&resource.slot);
                }
            }
            // A destroy the namespace could not name removes nothing from this
            // history, because nothing named it was ever put in.
            LifecycleOp::DeleteObject { task, resource, .. } => {
                census.deletes += 1;
                if let Some(resource) = resource {
                    live.entry(*task).or_default().remove(&resource.slot);
                }
            }
            LifecycleOp::ReplacePhysical {
                task,
                resource,
                backing,
                ..
            } => {
                census.replacements += 1;
                if let Some(entry) = live.entry(*task).or_default().get_mut(&resource.slot) {
                    entry.1 = *backing;
                }
            }
            LifecycleOp::DeleteBacking { task, backing, .. } => {
                census.retired_backings += 1;
                if let Some(backing) = backing {
                    live.entry(*task)
                        .or_default()
                        .retain(|_, (_, b)| b != backing);
                }
            }
            // A bind changes what an unresolved reference will construct and
            // ends nothing already declared, so the shadow of live names does
            // not move. Listed by name rather than swept into a wildcard: an
            // operation that *did* move names and reached this arm silently
            // would make the shadow agree with the model by not looking.
            LifecycleOp::BindObjectList { .. }
            | LifecycleOp::MapMemory { .. }
            | LifecycleOp::UnmapMemory { .. }
            | LifecycleOp::Invalidate { .. }
            | LifecycleOp::Synchronize { .. }
            | LifecycleOp::SynchronizeAndDiscard { .. }
            | LifecycleOp::Discard { .. } => {}
        }
    }

    /// Hold the model to the shadow, and the content ledger to both.
    fn check(
        seed: u64,
        l: &Lifecycle,
        live: &HashMap<TaskId, HashMap<ObjectListRef, (ResourceId, BackingId)>>,
        storage: &HashMap<TaskId, HashMap<u64, BackingId>>,
        knew: &mut HashSet<BackingId>,
        census: &mut Census,
    ) {
        // The resident map, against a shadow that computed both cascades for
        // itself.
        for (task, t) in &l.tasks {
            let mut mine: Vec<(ObjectListRef, ResourceId, BackingId)> = t
                .resident
                .iter()
                .map(|(id, r)| (id.slot, *id, r.backing()))
                .collect();
            mine.sort_unstable_by_key(|(slot, _, _)| slot.0);
            let mut theirs: Vec<(ObjectListRef, ResourceId, BackingId)> = live
                .get(task)
                .into_iter()
                .flatten()
                .map(|(slot, (id, b))| (*slot, *id, *b))
                .collect();
            theirs.sort_unstable_by_key(|(slot, _, _)| slot.0);
            assert_eq!(mine, theirs, "seed {seed}: residents of {task:?}");
        }
        assert_eq!(
            l.tasks.len(),
            live.len(),
            "seed {seed}: the model and the shadow hold different tasks"
        );

        for n in 0..BACKINGS {
            let b = BackingId(n);
            let namers: Vec<TaskId> = live
                .iter()
                .filter(|(_, r)| r.values().any(|(_, x)| *x == b))
                .map(|(t, _)| *t)
                .collect();
            let stored = storage.values().any(|h| h.values().any(|x| *x == b));
            if namers.len() > 1 {
                census.shared_backings += 1;
            }
            if !namers.is_empty() {
                assert!(
                    l.content().knows(b),
                    "seed {seed}: {b:?} is named by {namers:?} and the ledger forgot it"
                );
            }
            // A heap's storage may be known without a live name — a placement
            // wrote it and the heap still holds the storage. So the two
            // remaining claims ask both registries: nothing held may stop
            // being known, and nothing unheld may stay known.
            if (!namers.is_empty() || stored) && knew.contains(&b) {
                assert!(
                    l.content().knows(b),
                    "seed {seed}: {b:?} is still held and the ledger forgot it"
                );
            }
            if namers.is_empty() && !stored {
                assert!(
                    !l.content().knows(b),
                    "seed {seed}: nothing answers for {b:?} and the ledger kept it"
                );
            }
            if l.content().knows(b) {
                knew.insert(b);
            } else {
                knew.remove(&b);
            }
        }
    }
    /// A resource the guest declared with no bytes participates, and does not
    /// refuse.
    ///
    /// **The sharpest case is the indirect command buffer.** Its create
    /// descriptor names no allocation and no window — its commands are
    /// device-side storage the guest never addresses — so it is declared
    /// `Storage::NoBytes`, and `IcbOp::participations` claims a participation
    /// on it. Answering `NotDeclared` refused the whole exec transaction, which
    /// is every draw in the packet, for a name that resolves perfectly well.
    #[test]
    fn a_participation_over_a_resource_with_no_bytes_is_domain_only() {
        let mut l = Lifecycle::new();
        apply_inert(
            &mut l,
            &LifecycleOp::DefineTask {
                task: TASK,
                kernel: false,
                directory: DirectoryFrame(0x1000),
            },
        );
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(0),
                storage: Storage::NoBytes,
            },
        );

        let intent = l
            .access(
                TASK,
                ChannelId(2),
                &Participation {
                    resource: name(0),
                    extent: ParticipationExtent::Whole,
                    mode: AccessMode::ReadWrite,
                    api_stages: 0,
                },
            )
            .expect("a live name with no bytes is not an undeclared slot");

        assert_eq!(intent.key, crate::access::AccessKey::DomainOnly);
        assert_eq!(intent.domain, ChannelId(2));
        assert_eq!(intent.mode, AccessMode::ReadWrite);
        // No bytes, so nothing to version on either side. A write here that
        // reserved a version would be authority over memory that does not
        // exist.
        assert_eq!(intent.input_content_version, None);
        assert_eq!(intent.output_content_version, None);
    }

    /// It over-orders rather than under-orders: the key meets every other
    /// access, so nothing the record might really have touched slips past it.
    #[test]
    fn a_no_bytes_access_meets_everything_in_its_domain() {
        let (mut l, dedicated_name) = with_one_resource(0x1000);
        apply_inert(
            &mut l,
            &LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(1),
                storage: Storage::NoBytes,
            },
        );

        let bytes = l
            .access(
                TASK,
                ChannelId(2),
                &Participation {
                    resource: dedicated_name,
                    extent: ParticipationExtent::Whole,
                    mode: AccessMode::Write,
                    api_stages: 0,
                },
            )
            .expect("a dedicated resource");
        let none = l
            .access(
                TASK,
                ChannelId(2),
                &Participation {
                    resource: name(1),
                    extent: ParticipationExtent::Whole,
                    mode: AccessMode::Read,
                    api_stages: 0,
                },
            )
            .expect("a resource with no bytes");

        assert!(none.key.may_alias(bytes.key));
        assert!(bytes.key.may_alias(none.key));
        assert_eq!(none.key.rung(), 3, "and the imprecision is priced");
    }

    /// A window past the resource's end widens to the whole of it and still
    /// meets every access over it.
    ///
    /// **The claim is a bound, not an assertion.** A record's `Range` is the
    /// most bytes it could touch — `ImagePitch::span_bytes` builds one by
    /// multiplying a pitch by a height, deliberately long — so checking it as if
    /// it were exact refuses the copies the long side was for. A driven macos-15
    /// boot threw away seven whole exec packets a boot on it, every one a
    /// `0x12c` `copyFromBuffer:…toTexture:…` reading `offset=65536
    /// length=196608` of a 196 608-byte source buffer.
    ///
    /// The widened access is compared against an exact one over the same
    /// resource, because the property that makes widening safe is not that it is
    /// coarse — it is that it still *meets*. A clamped window would be finer and
    /// wrong: it would claim fewer bytes than the record may touch, and the edge
    /// that does not get built is a race.
    #[test]
    fn a_window_past_the_extent_widens_to_the_whole_resource() {
        let (mut l, name) = with_one_resource(0x1000);
        let over = l
            .access(
                TASK,
                ChannelId(2),
                &Participation {
                    resource: name,
                    // One byte longer than the resource, from its own start.
                    extent: ParticipationExtent::Range(range(0, 0x1001)),
                    mode: AccessMode::Read,
                    api_stages: 0,
                },
            )
            .expect("a bound past the end is not a refusal");
        assert!(
            !matches!(over.key, crate::access::AccessKey::Range(..)),
            "the window this model could not hold is not reported as one: {:?}",
            over.key
        );

        let inside = l
            .access(
                TASK,
                ChannelId(2),
                &Participation {
                    resource: name,
                    extent: ParticipationExtent::Range(range(0x40, 0x40)),
                    mode: AccessMode::Write,
                    api_stages: 0,
                },
            )
            .expect("a window inside the extent");
        assert!(
            over.key.may_alias(inside.key) && inside.key.may_alias(over.key),
            "widening must not lose the edge the record's own bytes would have drawn"
        );
    }

    /// A slot nothing declared still refuses, which is what the absent
    /// residency used to be mistaken for.
    #[test]
    fn a_slot_nothing_declared_still_refuses_the_transaction() {
        let (mut l, _) = with_one_resource(0x1000);

        let refusal = l
            .access(
                TASK,
                ChannelId(2),
                &Participation {
                    resource: name(9),
                    extent: ParticipationExtent::Whole,
                    mode: AccessMode::Read,
                    api_stages: 0,
                },
            )
            .expect_err("slot 9 was never declared");
        assert!(matches!(refusal, Refusal::Namespace { .. }));
    }
}
