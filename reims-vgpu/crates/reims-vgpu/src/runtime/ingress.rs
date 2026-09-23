//! From a packet this device drained to a packet the semantic model admits.
//!
//! # What this is for
//!
//! [`crate::runtime::drain`] takes bytes out of a ring and produces a
//! [`drain::Packet`]: an opcode, a record array, a completion word and a
//! payload. [`reims_vgpu_core::session::SessionModel`] admits a
//! [`reims_vgpu_core::session::Packet`]: an ordering domain, a semantic
//! lifetime, typed stamps and a resolved [`Payload`]. Those are the two ends of
//! the cutover and **nothing joined them** — the model could describe every
//! packet this device receives and could be handed none of them.
//!
//! This is that join. It touches no guest memory and mutates nothing;
//! everything it cannot answer it returns as a named [`Gap`] rather than
//! approximating, so the set of classes that can cross today is a value a test
//! can assert rather than a claim a reader has to believe.
//!
//! It is **not** a pure function of the drained packet, and stopped being one
//! the moment the first namespace gap closed. A ref, a mapping id and a page
//! frame are numbers whose meaning lives in a namespace, and a bridge that
//! could not consult one could only ever carry the classes that name nothing —
//! which is what [`Gap`] was a list of. So the namespaces arrive as
//! [`Resolvers`]: a bundle rather than a parameter each, because every
//! remaining gap closes by *adding a resolver*, and a bundle makes that a field
//! rather than another signature change rippling through every caller.
//!
//! # The gaps are closed, and what closed them is worth keeping
//!
//! What crosses used to be "what resolves from the packet's own bytes", and
//! [`Gap`] was the list of classes that did not. That list is now empty of
//! everything but an unsettled ledger row.
//!
//! **The class was never the unit, and that is what naming the gaps bought.**
//! The resource-lifetime class was three different partitions in three commits,
//! and each coarser name was hiding a member that did not belong to it.
//! "Lifetime commands need a namespace" was false of the two that name a task
//! and nothing inside it; once the namespaces landed it was false of all
//! twelve, and what the rest were short of was the *access* each named resource
//! takes — a different missing thing with a different owner. A gap that
//! over-claims is one nobody thinks to close; a gap that closes and leaves its
//! name behind is one everybody thinks is closed.
//!
//! The four that closed, and what each of them turned out to need:
//!
//! - **Query** resolves its reply destination through [`Resolvers::replies`],
//!   in whichever address space its own question uses. Every route this device
//!   interns on draws from one monotone counter, so a page frame identified as
//!   a page frame can equal no window and no mapping by construction. See
//!   [`query_payload`].
//! - **Present** resolves its target as a mapping, and the frame it shows is
//!   the whole of that mapping's surface — so the access is an
//!   `AccessKey::Whole` over the resolved backing and nothing about it is
//!   approximated. See [`present_payload`] for the one case that is still
//!   imprecise and why the imprecision is the honest answer there.
//! - **Exec** needed three inputs and not the two its walk's signature shows:
//!   an object-list resolver, an access source, and the command stream *bytes*,
//!   which live at guest addresses the payload names rather than in the
//!   payload. The first two are `reims_vgpu_core::lifecycle::Lifecycle`'s and
//!   became production state with the resource-lifecycle group's cutover; the
//!   third is a page-table walk and stays the caller's, because it is
//!   guest-memory work and not translation. See [`ExecStreams`].
//! - **The re-point** — the twelfth lifetime command, and the last of all — is
//!   short of pages that are not in its packet at all. Its identity resolves in
//!   the model and its storage arrives from the device that has just moved the
//!   storage incarnation, which is why those two halves come from two places.
//!   See [`ReplacementStorage`].
//!
//! Both of the last two arrive as [`DeviceInputs`], which is the shape the
//! remaining work has: not a namespace this bridge could consult, but a value
//! whose *ordering* against this device's own state is the caller's to get
//! right. What is left in [`Gap`] is [`Gap::Unresolved`] alone — a ledger row
//! whose contract is not settled — which is a question about the guest's
//! interface and not about this join.
//!
//! # Every packet carries a completion word
//!
//! [`reims_vgpu_core::session::Packet::completion`] is an `Option`, because a
//! model may have packets that publish nothing. **This interface does not**:
//! the drain writes the header's completion word into the channel's stamp slot
//! for every packet it processes, and a packet that does not advance the fence
//! repeats the slot's current value rather than leaving it alone. Repeating is
//! idempotent and a wait decided against a repeat is decided the same way it
//! was before, so "does not signal" and "signals the value already there" are
//! the same event on the wire. This bridge therefore always produces `Some`,
//! and `None` stays available for a model that has the other case.
//!
//! # The slot is the channel's and the value is the packet's
//!
//! A `drain::Packet` carries a completion *value* and no slot. The slot belongs
//! to the FIFO — the root's is slot 0 and a child's is read once per drain from
//! its register block — so it arrives here as an argument. Wait records carry
//! their own raw index, masked to a slot by
//! [`stamp_slot_index`], which is the one place that mask is applied.

use crate::model::{is_child_channel, stamp_slot_index};
use crate::runtime::drain;
use reims_vgpu_core::control;
use reims_vgpu_core::identity::{
    ChannelId, CompletionStamp, SessionGeneration, StampSlot, StampValue, StampWait,
};
use reims_vgpu_core::resolve::{MappingResolver, ResourceStorage, TaskNamespaces};
use reims_vgpu_core::session::Packet;
use reims_vgpu_core::transaction::{classify, Payload, PayloadClass};
use reims_vgpu_protocol::packets::Channel;

/// Which FIFO a packet was drained from.
///
/// **One value, because the two questions have one answer.**
/// [`reims_vgpu_core::session::Packet`] carries both a [`Channel`] — which
/// dispatch table the opcode is read against — and a [`ChannelId`] — which
/// ordering domain the packet joins. They are not independent: this device
/// numbers the root FIFO 0 and its children 1..[`MAX_CHANNELS`], which is the
/// rule [`is_child_channel`] states and the rule
/// [`reims_vgpu_core::control::ControlOp::Channel`] hands back a domain under.
/// A bridge taking the pair separately could be given `Channel::Root` with a
/// child's domain, and the packet would then be classified against the root's
/// opcode table and ordered on a child's channel. Deriving the channel from the
/// id makes that unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fifo(ChannelId);

impl Fifo {
    /// The device's own channel, numbered 0.
    pub const ROOT: Fifo = Fifo(ChannelId(0));

    /// A guest-defined channel, or `None` when the id names no FIFO this
    /// device has. The bound is [`is_child_channel`]'s and not restated here.
    #[must_use]
    pub fn child(channel_id: u32) -> Option<Fifo> {
        is_child_channel(channel_id).then_some(Fifo(ChannelId(channel_id)))
    }

    /// The ordering domain packets on this FIFO join.
    #[must_use]
    pub const fn domain(self) -> ChannelId {
        self.0
    }

    /// Which opcode table this FIFO's packets are read against.
    #[must_use]
    pub const fn channel(self) -> Channel {
        if self.0 .0 == 0 {
            Channel::Root
        } else {
            Channel::Child
        }
    }
}

/// An input the model needs that this bridge is not given.
///
/// Not a refusal: the guest's packet is well formed and this device answers it
/// today. Each variant names the one thing missing, so a suite can assert the
/// partition — which classes cross and which do not — instead of a reader
/// having to infer it from what the code happens to handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gap {
    /// The ledger has not closed this row, so [`classify`] gives it no class
    /// and the model may not claim it. Production answers it alone.
    Unresolved,
    /// An exec packet reached [`device_packet`] without the command buffers it
    /// names.
    ///
    /// The streams live in the task's address space, so reading them is a
    /// page-table walk the caller performs before admission — see
    /// [`ExecStreams`]. Every reason that walk produces nothing is already
    /// named on the always-on channel by `crate::runtime::exec::read_submission`
    /// where it was refused, so this says only that they did not arrive, which
    /// is the one thing this bridge can state about them.
    ///
    /// A gap and not a refusal: the packet's own bytes are well formed, and
    /// what is missing is memory the model was not given.
    ExecStreamsUnread,
}

impl Gap {
    /// The name this reaches the failure channel under.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unresolved => "ingress_row_unresolved",
            Self::ExecStreamsUnread => "ingress_exec_streams_unread",
        }
    }
}

/// The guest's bytes not being the command its opcode names, as the resolver
/// that judged them said it.
///
/// Two vocabularies and not one string, because two resolvers judge two classes
/// here and they refuse different things — a control packet that is not control,
/// a lifetime payload too short for the record its opcode implies. Folded into
/// one name, a reading could not say which layer decided, and the two have
/// different fixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refused {
    Control(control::ResolveRefusal),
    Lifecycle(reims_vgpu_core::lifecycle::ResolveRefusal),
    /// A present packet's bytes are not a present, or are too short for the
    /// trailer its form has.
    Present(reims_vgpu_core::present::ResolveRefusal),
    /// A query packet's request words are not the layout its question uses, or
    /// are too short for it.
    Query(reims_vgpu_core::query::WordsRefusal),
    /// A query packet's request resolved and its reply shape did not agree with
    /// the words it was read from.
    ///
    /// Unreachable by construction today — [`query_payload`] reads the words at
    /// the kind's own layout and hands the same kind to the resolver — and
    /// carried for [`Self::PresentAccesses`]'s reason.
    QueryShape(reims_vgpu_core::query::ResolveRefusal),
    /// A query packet does not carry the reply address its own layout puts
    /// there.
    ///
    /// The resolver's `None`, and it is the guest's bytes rather than this
    /// device's incompleteness: an implementor reaching the same decoders
    /// `request_words` reaches answers `None` only for a payload too short to
    /// hold the address. An address it *did* find always resolves — storage
    /// with no allocation is still storage.
    QueryDestination,
    /// The accesses built for a lifetime command do not describe the resources
    /// its operation named.
    ///
    /// Unreachable by construction today — [`lifecycle_accesses`] walks
    /// `LifecycleOp::resources` itself and pairs each entry with exactly one
    /// access — and carried for [`Self::PresentAccesses`]'s reason: the
    /// constructor is what holds an operation and its accesses together, and a
    /// bridge that unwrapped it would be asserting a property it had stopped
    /// letting anything check.
    LifecycleAccesses(reims_vgpu_core::transaction::AccessMismatch),
    /// The accesses built for a present do not describe the frame its packet
    /// asked for.
    ///
    /// Unreachable by construction today — [`present_payload`] builds exactly
    /// one read of the packet's own target — and carried anyway, because the
    /// constructor is what holds the packet and its accesses together and a
    /// bridge that unwrapped it would be asserting a property it had stopped
    /// letting anything check.
    PresentAccesses(reims_vgpu_core::transaction::PresentMismatch),
    /// A command stream did not become a transaction.
    ///
    /// The one refusal here that is not about a *packet's* bytes: an exec
    /// packet's payload is a table of addresses, and what refuses is the stream
    /// at one of them. Carried whole from
    /// [`reims_vgpu_core::walk::WalkRefusal`] — which already names the segment
    /// and offset — because a bridge-level re-wording would lose the site and
    /// the owner that judged it.
    Exec(reims_vgpu_core::walk::WalkRefusal),
    /// A re-pointed reference names no storage this device can describe.
    ///
    /// The device's own answer, carried whole. `ReplacePhysical` is the one
    /// command whose operation needs a term that is nowhere on its wire — the
    /// guest re-points a resource at frames it has already wired, at the same
    /// GPU-VA — so the storage arrives from the caller and this is what it says
    /// when it has none. A driven macos-15 boot puts every route of it at zero:
    /// all 242 of that boot's re-points named describable storage once a
    /// surface-backing object stopped being read as owning no bytes.
    Repoint(crate::runtime::objects::RepointStorageRefusal),
}

impl Refused {
    /// The name this reaches the failure channel under, from the refusal's own
    /// owner rather than restated here.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::Control(inner) => inner.slug(),
            Self::Lifecycle(inner) => inner.slug(),
            Self::Query(inner) => inner.slug(),
            Self::QueryShape(inner) => inner.slug(),
            Self::QueryDestination => "ingress_query_carries_no_reply_address",
            Self::LifecycleAccesses(inner) => inner.slug(),
            Self::Present(inner) => inner.slug(),
            Self::PresentAccesses(inner) => inner.slug(),
            Self::Exec(inner) => inner.reason(),
            Self::Repoint(inner) => inner.route(),
        }
    }
}

/// The namespaces this bridge resolves the guest's numbers in.
///
/// **A bundle and not a parameter each.** Three of the model's inputs are still
/// gaps and every one of them closes by giving this bridge another namespace to
/// consult; carried as separate arguments, each closure would be a signature
/// change at every call site, and a call site would be free to pass the mapping
/// namespace where the object namespace was wanted. Carried as one value with
/// named fields, a closure adds a field and the compiler names every caller that
/// has to answer for it.
///
/// Borrowed rather than owned, and `dyn` rather than generic: the device is the
/// only implementor there will ever be, the call is once per packet against a
/// `BTreeMap` lookup, and a generic here would put a type parameter on
/// [`packet`] that every caller and every test would have to spell.
#[derive(Clone, Copy)]
pub struct Resolvers<'a> {
    /// Which backing a mapping's surface currently occupies.
    pub mappings: &'a dyn MappingResolver,
    /// Every task's object namespace, asked for by the task a packet names.
    ///
    /// Not one namespace: which task's slots a lifecycle packet's refs are
    /// indices into is stated inside that packet, so only the join that decoded
    /// it can bind the namespace. See `reims_vgpu_core::resolve::TaskNamespaces`.
    pub objects: &'a dyn TaskNamespaces,
    /// Where a named resource's bytes are, which is what an access is keyed on.
    ///
    /// Beside [`Self::objects`] rather than inside it: what a ref *names* and
    /// where the named thing *lives* are two questions with two holders, and
    /// `reims_vgpu_core::resolve::ResourceStorage`'s own doc says why folding
    /// them would make "no such slot" and "no storage term" one answer.
    pub storage: &'a dyn ResourceStorage,
    /// Where each question's answer goes, resolved to a backing and a window.
    ///
    /// A third door rather than a field on either of the others, because a
    /// reply destination is resolved in whichever address space its own
    /// question uses — and one of the four uses none of the spaces the other
    /// resolvers answer for. See `reims_vgpu_core::query::Destinations`.
    pub replies: &'a dyn reims_vgpu_core::query::Destinations,
}

/// The three inputs an EXEC packet costs beyond its own bytes.
///
/// # Why this is a separate argument and not a fifth resolver
///
/// [`Resolvers`] is `Copy` and holds shared references on purpose, and two of
/// the three here cannot live in it. The access source is a `&mut` borrow —
/// turning a record's participation claim into an access *records* the
/// participation — and the command streams are not references into anything
/// this bridge can see.
///
/// # The third input is guest memory, and it is why exec was the last class
///
/// An `EXEC_INDIRECT2` payload carries a header, a resource table and
/// `cmdbuf_count` descriptors of `{gva, length}`. The streams themselves live
/// in the task's *address space*, so reading them is a page-table walk per
/// buffer — which is guest-memory work, not translation, and belongs to the
/// caller. `crate::runtime::exec::load_command_streams` is that walk. This
/// bridge therefore cannot be a pure function of the drained packet for this
/// class the way it is for the other four; it is a pure function of the drained
/// packet *and the streams it names*, which is the honest shape.
///
/// A driven macos-15 boot measured what a caller actually has to supply:
/// `exec_cmdbufs_1 = 20 006` with every other bucket silent, so this guest
/// sends exactly one command buffer per exec packet. [`Self::streams`] is a
/// slice regardless — the 17-buffer submission is in this repository's history,
/// and a table walked into one transaction is the contract whatever this rail's
/// count happens to be.
pub struct ExecStreams<'a> {
    /// The task whose object list the refs inside those streams index.
    ///
    /// From the exec header, not from the FIFO: a channel carries packets for
    /// whichever task submitted them, and resolving a ref in the wrong task
    /// finds whatever shared the integer.
    pub task: reims_vgpu_core::identity::TaskId,
    /// What executors have published about the pipelines these streams bind.
    ///
    /// Supplied by the caller rather than read here, for the reason `accesses`
    /// is: this bridge is a pure function of the packet and the things the
    /// device hands it, and reaching into the ordering plane from inside the
    /// walk would be a second owner of the answer. See
    /// [`crate::model::DeviceState::pipeline_usage`].
    pub usages: &'a dyn reims_vgpu_core::pipeline::UsageSource,
    /// One entry per command buffer the packet's table names, in table order,
    /// already read out of the task's address space.
    ///
    /// All of them walk into **one** transaction: a packet is one ordering
    /// position, one completion word and one access list, whatever its buffer
    /// count. See `reims_vgpu_core::walk::command_buffer`.
    pub streams: &'a [Vec<u8>],
    /// Where each record's participation claim becomes an access.
    ///
    /// `reims_vgpu_core::lifecycle::Lifecycle::task_access` is the one
    /// implementation with the terms — it owns names, heaps and content
    /// authority together — and since the resource-lifecycle group's cutover it
    /// is production state, held by `DeviceState`.
    pub accesses: &'a mut dyn reims_vgpu_core::access::AccessSource,
}

/// The storage a re-pointed reference names *now*.
///
/// The identity half is not here on purpose. `ReplacePhysical`'s task and
/// resource are resolved by the model — they are on the wire and the ref is
/// looked up in its own task's namespace — and they travel back on
/// `reims_vgpu_core::lifecycle::ResolveRefusal::NeedsStorage`. Only the bytes
/// come from the device. A caller that supplied its own idea of which resource
/// the packet named would be offering a second resolution of the same ref, and
/// the operation built from the pair would be about whichever of the two was
/// wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplacementStorage {
    pub backing: reims_vgpu_core::access::BackingId,
    pub extent: reims_vgpu_core::access::ByteRange,
}

/// What this bridge cannot compute from the drained packet, supplied by the
/// device that can.
///
/// # Two classes, one argument, and why it is not more resolvers
///
/// [`Resolvers`] answers *namespace* questions: what does this number name.
/// Both members here are something else — work the device performed or memory
/// it read — and neither fits a `Copy` bundle of shared references. Bundled
/// together for [`Resolvers`]'s own reason: a third such input should become a
/// field rather than another signature change through every caller.
///
/// Both are ordered facts, and that is the substance rather than the plumbing.
/// The command streams are read out of the task's address space *before* the
/// packet is admitted; the replacement storage is read *after* this device has
/// moved the storage incarnation, because the identity of the new pages is the
/// next incarnation and nothing has bumped it before the handler runs. Each is
/// a value by the time it arrives here, which is what keeps this bridge a
/// function that mutates nothing.
pub struct DeviceInputs<'a> {
    /// The exec class's command buffers and the task their refs index.
    pub exec: ExecStreams<'a>,
    /// What a re-pointed reference names now, or why the device cannot say.
    ///
    /// `Err` refuses the packet under the device's own reason rather than
    /// building an operation carrying the *old* backing, which would re-point
    /// nothing while reporting success.
    pub repointed: Result<ReplacementStorage, crate::runtime::objects::RepointStorageRefusal>,
}

/// Why a drained packet did not become a model packet.
///
/// The two arms are different obligations and are deliberately not one type. A
/// [`Self::Gap`] is this device's own incompleteness and closes when its owner
/// lands; a [`Self::Refused`] is the guest's bytes not being the command its
/// opcode names, and closes never.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blocked {
    Gap {
        channel: Channel,
        opcode: u16,
        gap: Gap,
    },
    Refused(Refused),
}

impl Blocked {
    /// The name this reaches the failure channel under.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::Gap { gap, .. } => gap.slug(),
            Self::Refused(refusal) => refusal.slug(),
        }
    }

    /// The gap, for a caller deciding whether to fall back to the legacy path.
    #[must_use]
    pub const fn gap(self) -> Option<Gap> {
        match self {
            Self::Gap { gap, .. } => Some(gap),
            Self::Refused(_) => None,
        }
    }
}

/// Walk one exec packet's command buffers into the one transaction they are.
///
/// Split out of [`packet`] so the buffer-count contract lives in a named place:
/// every buffer the table declares walks into a **single** [`ExecBuilder`],
/// which is finished once. A packet is one ordering position, one completion
/// word and one access list — a builder per buffer would give a two-buffer
/// packet two of each and order them against each other, which no field on the
/// wire asks for.
///
/// # Errors
///
/// [`Refused::Exec`], carrying the walk's own refusal with its stream site. The
/// transaction is all-or-nothing: a single unresolvable record refuses the whole
/// packet rather than yielding a transaction missing a hazard edge.
fn exec_payload(
    objects: &dyn TaskNamespaces,
    exec: ExecStreams<'_>,
) -> Result<reims_vgpu_core::exec::ExecWork, Blocked> {
    let ExecStreams {
        task,
        usages,
        streams,
        accesses,
    } = exec;
    let resolver = reims_vgpu_core::resolve::InTask::new(objects, task);
    // One usage answer for the whole packet, which is what `with_usage`'s own
    // doc requires: a packet is one transaction and one access list, so two of
    // its records resolved against two answers would describe two pipelines'
    // shaders in one footprint.
    let mut builder = reims_vgpu_core::exec::ExecBuilder::with_usage(usages);
    for stream in streams {
        reims_vgpu_core::walk::command_buffer(stream, &resolver, accesses, &mut builder)
            .map_err(|refusal| Blocked::Refused(Refused::Exec(refusal)))?;
    }
    builder.finish().map_err(|refusal| {
        Blocked::Refused(Refused::Exec(
            reims_vgpu_core::walk::WalkRefusal::Unfinished { refusal },
        ))
    })
}

/// Build the model packet one drained packet describes.
///
/// `session` is the semantic lifetime the packet was **read under**, which is
/// the reader's fact and not this function's — see
/// [`reims_vgpu_core::session::Packet::session`]. `completion_slot` is the
/// FIFO's stamp slot, already masked by whoever read it.
///
/// # Errors
///
/// [`Blocked::Gap`] for every class this bridge cannot yet build, and
/// [`Blocked::Refused`] when a control packet's bytes are not its command.
pub fn packet(
    fifo: Fifo,
    session: SessionGeneration,
    completion_slot: StampSlot,
    drained: &drain::Packet,
    resolvers: Resolvers<'_>,
    inputs: DeviceInputs<'_>,
) -> Result<Packet, Blocked> {
    let channel = fifo.channel();
    let opcode = drained.opcode;
    let blocked = |gap| Blocked::Gap {
        channel,
        opcode,
        gap,
    };

    // Exhaustive over `PayloadClass`, so a sixth class is a compile error here
    // rather than a packet that quietly reaches whichever arm came last. The
    // gap each class names is stated once, in this match, and nowhere else.
    let DeviceInputs { exec, repointed } = inputs;
    let payload = match classify(channel, opcode) {
        None => return Err(blocked(Gap::Unresolved)),
        Some(PayloadClass::Exec) => Payload::Exec(exec_payload(resolvers.objects, exec)?),
        Some(PayloadClass::ResourceLifecycle) => Payload::ResourceLifecycle(resource_lifetime(
            fifo,
            channel,
            opcode,
            &drained.payload,
            resolvers,
            repointed,
        )?),
        Some(PayloadClass::Query) => Payload::Query(query_payload(
            fifo,
            channel,
            opcode,
            &drained.payload,
            resolvers,
        )?),
        Some(PayloadClass::Present) => Payload::Present(present_payload(
            fifo,
            channel,
            opcode,
            &drained.payload,
            resolvers,
        )?),
        Some(PayloadClass::Control) => Payload::Control(
            control::resolve(channel, opcode, &drained.payload)
                .map_err(|refusal| Blocked::Refused(Refused::Control(refusal)))?,
        ),
    };

    Ok(Packet {
        channel,
        domain: fifo.domain(),
        session,
        opcode,
        stamp_waits: drained
            .stamp_waits
            .iter()
            .map(|wait| StampWait {
                slot: StampSlot(stamp_slot_index(wait.index)),
                value: StampValue(wait.value),
            })
            .collect(),
        completion: Some(CompletionStamp {
            slot: completion_slot,
            value: StampValue(drained.completion_stamp),
        }),
        payload,
    })
}

/// What the caller read out of guest memory for one packet, which
/// [`device_packet`] will not read for it.
///
/// Both members are guest-memory work performed at a moment this bridge cannot
/// choose. The command buffers are read *before* admission, out of the task's
/// address space; the replacement storage is read *after* this device has moved
/// a re-pointed resource's incarnation, because an earlier answer is the old
/// one. A door that performed either would be a bridge that mutates guest
/// state, and one that performed them at its own convenience would answer with
/// the wrong bytes.
///
/// One value rather than two arguments for [`Resolvers`]'s reason: a third such
/// input should become a field rather than a signature change through every
/// caller.
pub struct PacketReads<'a> {
    /// The exec class's command buffers, already read. `None` for every other
    /// class, and for an exec packet whose streams this device could not read —
    /// see [`Gap::ExecStreamsUnread`].
    pub submission: Option<&'a crate::runtime::exec::ExecSubmission>,
    /// What a re-pointed reference names now, or why the device cannot say.
    pub repointed: Result<ReplacementStorage, crate::runtime::objects::RepointStorageRefusal>,
}

/// [`packet`] with this device supplying every namespace it asks for.
///
/// # Why the assembly is a function and not a call site
///
/// [`packet`]'s inputs are five borrows of four different device tables, and
/// three of them have to be constructed rather than looked up:
/// `crate::runtime::objects::TaskNames` is `(state, host)`, the access source is
/// `crate::model::DeviceState::task_access` bound to *this packet's* task and
/// domain, and the mapping resolver is the device itself. Spelled at a call
/// site, the pairing is the caller's to get right — and the pairing that can go
/// wrong silently is the access source's, which takes a task and a domain that
/// are properties of the packet and would resolve someone else's participations
/// against this packet's ordering if they were taken from anywhere else.
///
/// Here the task comes from the submission's own header and the domain from the
/// `Fifo` the packet was drained out of, which are the two facts the packet
/// itself carries.
///
/// # The two inputs the caller still supplies, and why they are not read here
///
/// [`ExecStreams::streams`] is a page-table walk per command buffer and
/// [`ReplacementStorage`] is read *after* this device has moved a re-pointed
/// resource's storage incarnation. Both are the caller's work by contract —
/// see `crate::runtime::exec::ExecSubmission` for the first and
/// [`ReplacementStorage`] for the second — and a door that performed them
/// would be a bridge that mutates guest state.
///
/// # Errors
///
/// Whatever [`packet`] refuses, plus [`Gap::ExecStreamsUnread`] for an exec
/// packet whose command buffers this device did not read.
pub fn device_packet<M: crate::runtime::host::HostMemory>(
    state: &crate::model::DeviceState,
    host: &M,
    fifo: Fifo,
    session: SessionGeneration,
    completion_slot: StampSlot,
    drained: &drain::Packet,
    reads: PacketReads<'_>,
) -> Result<Packet, Blocked> {
    let PacketReads {
        submission,
        repointed,
    } = reads;
    let channel = fifo.channel();
    let class = classify(channel, drained.opcode);
    // Asked before anything is built, because the answer decides whether the
    // exec inputs below are the packet's or a stand-in: an exec packet with no
    // streams would otherwise walk an empty table into an empty transaction,
    // which admits and executes nothing while reporting success.
    if class == Some(PayloadClass::Exec) && submission.is_none() {
        return Err(Blocked::Gap {
            channel,
            opcode: drained.opcode,
            gap: Gap::ExecStreamsUnread,
        });
    }

    let names = crate::runtime::objects::TaskNames::new(state, host);
    let resolvers = Resolvers {
        mappings: state,
        objects: &names,
        storage: &names,
        replies: &names,
    };
    // `TaskId(0)` for every class that is not exec: the field is the task an
    // exec packet's *records* index, which only an exec header states, and the
    // access source is untouched by the other four classes. It is not a
    // default that could be wrong — a class that never asks cannot be answered
    // for the wrong task.
    let task = submission.map_or(reims_vgpu_core::identity::TaskId(0), |s| {
        reims_vgpu_core::identity::TaskId(s.task_id())
    });
    let mut accesses = state.task_access(task, fifo.domain());
    // Taken here and dropped when this returns, so the ordering plane's lock is
    // held for exactly one packet's walk. Safe to hold across it because every
    // other owner the walk reaches — the namespaces and the access source — is
    // the `lifecycle` owner's and not this one's.
    let usages = state.pipeline_usage();
    let no_streams: [Vec<u8>; 0] = [];
    packet(
        fifo,
        session,
        completion_slot,
        drained,
        resolvers,
        DeviceInputs {
            exec: ExecStreams {
                task,
                usages: &usages,
                streams: submission.map_or(&no_streams, |s| s.streams()),
                accesses: &mut accesses,
            },
            repointed,
        },
    )
}

/// What a present asks to show, and the one access that orders it.
///
/// # A present reads the whole of one surface
///
/// The packet names a mapping and a form; it names no byte range, no level and
/// no slice, because showing a frame is not a subresource operation — the
/// display reads the surface. So the access is `AccessKey::Whole` over the
/// backing the mapping resolves to, which is the precision ladder's rung 2 and
/// is *exact* here rather than approximate: there is no finer answer the packet
/// contains and none that would order differently, since anything writing any
/// part of that backing must be ordered before the frame is shown.
///
/// It is a read. `reims_vgpu_core::transaction::PresentPayload` enforces that,
/// and its doc says why: modelled as a writer, a present would reserve a
/// content version, beat the real writer's, and publish bytes nothing produced.
///
/// # The one imprecise case, and why it is not the `DomainOnly` this bridge
/// refused to use
///
/// A mapping the device has no live surface for resolves to no backing. There
/// is then nothing to name, and `AccessKey::DomainOnly` — "participation is
/// incomplete, ordering comes from the submission domain alone" — is the
/// vocabulary for precisely that. That is a different act from handing *every*
/// present a `DomainOnly`, which is what this gap was deliberately not closed
/// with: that would have under-ordered every frame on the device while reading,
/// from outside, as a gap that had closed. Here the imprecision is the target's
/// and it is counted, so a boot says how many frames were shown against a
/// mapping this device could not resolve rather than leaving it to inference.
///
/// # Errors
///
/// [`Blocked::Refused`] when the payload is not a present or is too short for
/// its own trailer.
fn present_payload(
    fifo: Fifo,
    channel: Channel,
    opcode: u16,
    payload: &[u8],
    resolvers: Resolvers<'_>,
) -> Result<reims_vgpu_core::transaction::PresentPayload, Blocked> {
    use reims_vgpu_core::access::{AccessIntent, AccessKey, AccessMode, ResourceKey};
    use reims_vgpu_core::transaction::PresentPayload;

    let packet = reims_vgpu_core::present::resolve(channel, opcode, payload)
        .map_err(|refusal| Blocked::Refused(Refused::Present(refusal)))?;
    let key = match resolvers.mappings.backing(packet.mapping) {
        Some(backing) => {
            crate::runtime::drain::note_store_route("ingress_present_target_resolved");
            AccessKey::Whole(ResourceKey {
                backing,
                heap: None,
            })
        }
        None => {
            // Counted, and counted here rather than left to the rung census, so
            // the reading distinguishes "this rail shows frames the device
            // cannot resolve" from "this rail shows no frames".
            crate::runtime::drain::note_store_route("ingress_present_target_unresolved");
            AccessKey::DomainOnly
        }
    };
    let access = AccessIntent {
        domain: fifo.domain(),
        key,
        mode: AccessMode::Read,
        // A present names no shader stage. Zero always means the record carried
        // none, never that one was dropped.
        api_stages: 0,
        input_content_version: None,
        output_content_version: None,
    };
    PresentPayload::new(packet, vec![(packet.mapping, access)])
        .map_err(|mismatch| Blocked::Refused(Refused::PresentAccesses(mismatch)))
}

/// A question, resolved to the write its answer will make.
///
/// # Three terms, and the one that was missing was the destination
///
/// A query payload is a request and one access. The request's words are the
/// packet's own, at the layout its question uses — `reims_vgpu_core::query`
/// owns that and this bridge could always ask it. The access is derived from
/// the destination by `QueryPayload::new`. What was missing between them was
/// the destination itself: a `BackingId` and a window, neither on the wire, and
/// the four questions do not share an address space to derive one from.
///
/// [`Resolvers::replies`] is that door, and the reason it closed is that a page
/// frame turned out to need no comparison against the other key spaces. This
/// device interns every route on one monotone counter, so a frame identified as
/// a frame can equal no window and no mapping by construction — the property
/// that had to hold, holding structurally rather than by measurement. Two
/// driven boots agree with it: `device_info_reply_before_any_identity` on both,
/// and `query_reply_inside_an_allocation` zero across every boot measured.
///
/// # No content version is reserved here
///
/// `QueryPayload::new`'s `output` is `None`. The reply is a write and the
/// content authority is `reims_vgpu_core::content`'s; this bridge holds no
/// authority to reserve a version with, and `None` says the term is absent
/// rather than claiming one. It is the same answer [`lifecycle_accesses`]
/// gives, for the same reason.
///
/// # Errors
///
/// [`Blocked::Refused`] for a payload that is not the request its opcode names,
/// or that does not carry the reply address its layout puts there.
fn query_payload(
    fifo: Fifo,
    channel: Channel,
    opcode: u16,
    payload: &[u8],
    resolvers: Resolvers<'_>,
) -> Result<reims_vgpu_core::transaction::QueryPayload, Blocked> {
    use reims_vgpu_core::query::{self, QueryKind};
    use reims_vgpu_core::transaction::QueryPayload;

    let Some(kind) = QueryKind::of(channel, opcode) else {
        // Unreachable through [`packet`]: `classify` said `Query` and
        // `QueryKind::of` asks that same table. Answered rather than unwrapped,
        // for `resource_lifetime`'s reason — a disagreement between two
        // functions must be a value a reader can find in a log.
        return Err(Blocked::Gap {
            channel,
            opcode,
            gap: Gap::Unresolved,
        });
    };
    let words = query::request_words(kind, payload)
        .map_err(|refusal| Blocked::Refused(Refused::Query(refusal)))?;
    let destination = resolvers
        .replies
        .destination(kind, payload)
        .ok_or(Blocked::Refused(Refused::QueryDestination))?;
    let request = query::resolve(kind, words, destination)
        .map_err(|refusal| Blocked::Refused(Refused::QueryShape(refusal)))?;
    Ok(QueryPayload::new(request, fifo.domain(), None))
}

/// Every resource-lifetime command, resolved in the namespaces the packet names.
///
/// # The namespaces are here, and they were not the only thing missing
///
/// This function was two commands wide: a task definition and a task deletion
/// name a task and nothing inside it, so they resolved from their own bytes,
/// and the other ten waited on a namespace nobody had given this bridge. The
/// namespaces are now here — `reims_vgpu_core::lifecycle::operation` is the one
/// join that picks which of its five sub-joins a kind belongs to, and it takes
/// both — and **ten of the twelve now produce an operation**.
///
/// An operation is not a payload. `reims_vgpu_core::transaction::LifecyclePayload`
/// holds an operation together with the accesses its named resources take, and
/// refuses a list that disagrees with the operation's own resource set. That
/// refusal is what partitions the ten: four of them name no resource, so the
/// empty list is not a shortfall but the correct statement, and the rest name
/// resources and owe each one an access. Both halves of that access are now
/// answerable — the mode from the operation, the key from
/// [`Resolvers::storage`] — so all ten cross.
///
/// **The partition is the type's and not a list here.** [`lifecycle_accesses`]
/// asks the operation which resources it names and what it declares of them; it
/// does not enumerate which kinds have a list. A thirteenth command, or a kind
/// that gains a resource list, moves itself.
///
/// # One command is short of neither a namespace nor an access
///
/// `ReplacePhysical`'s operation names pages that are not on the wire at all. It
/// resolves its ref and still cannot be built, which is why the core names it
/// with its own refusal rather than with `UnknownRef` — and why it was named
/// separately rather than folded into one gap called "namespaces", which would
/// have read as closed the day the namespaces landed. It closes on
/// [`ReplacementStorage`], which is the caller's half of that refusal.
///
/// `SetObjectList` used to be beside it and is not any more. Its operation was
/// held to be the per-entry result of walking the guest's table; a driven boot
/// measured the walk at 1 048 576 declared entries per bind against about
/// twelve hundred slots ever named, and the count turned out to be the table's
/// capacity. So the operation is the *binding*, the per-slot declaration is
/// produced by resolution as this device already produces it, and the command
/// crosses.
///
/// # A ref that names nothing is the guest's, not this device's
///
/// [`reims_vgpu_core::lifecycle::ResolveRefusal::UnknownRef`] stays a
/// [`Blocked::Refused`], and that is a claim worth stating because it was not
/// true before `crate::runtime::objects::name_resource`. While this device named
/// a reference only when something constructed it, an unresolved ref meant "no
/// draw has touched this yet" — device incompleteness — and 49 well-formed
/// packets a boot would have been refused under it. Resolution now reads the
/// guest's own object list, so a ref that still resolves to nothing is an empty
/// slot or an unreadable descriptor: the packet's, and not a gap that closes
/// later.
///
/// # Errors
///
/// [`Blocked::Gap`] for the commands whose payload this bridge cannot yet
/// complete, and [`Blocked::Refused`] when a lifetime packet's bytes are not the
/// command its opcode names.
fn resource_lifetime(
    fifo: Fifo,
    channel: Channel,
    opcode: u16,
    payload: &[u8],
    resolvers: Resolvers<'_>,
    repointed: Result<ReplacementStorage, crate::runtime::objects::RepointStorageRefusal>,
) -> Result<reims_vgpu_core::transaction::LifecyclePayload, Blocked> {
    use reims_vgpu_core::lifecycle::{self, LifecycleKind, ResolveRefusal};
    use reims_vgpu_core::transaction::LifecyclePayload;
    let gap = |gap| Blocked::Gap {
        channel,
        opcode,
        gap,
    };
    // Asked of the same table `classify` read, rather than of the opcode again:
    // the class said this is a lifetime command and the kind says which one.
    let Some(kind) = LifecycleKind::of(channel, opcode) else {
        // Unreachable through [`packet`], which reached this arm because
        // `classify` said `ResourceLifecycle` and `LifecycleKind::of` asks that
        // same table. Answered rather than unwrapped, because the two are
        // separate functions and a disagreement between them must be a value a
        // reader can find in a log.
        return Err(gap(Gap::Unresolved));
    };
    let op = match lifecycle::operation(kind, payload, resolvers.objects, resolvers.mappings) {
        Ok(op) => op,
        // The one command whose operation the wire cannot complete. The model
        // resolved its task and its resource and stopped there, so the identity
        // is the model's and only the bytes are the caller's — see
        // [`ReplacementStorage`] for why the halves are split that way.
        Err(ResolveRefusal::NeedsStorage { task, resource, .. }) => match repointed {
            Ok(ReplacementStorage { backing, extent }) => lifecycle::LifecycleOp::ReplacePhysical {
                task,
                resource,
                backing,
                extent,
            },
            Err(refusal) => return Err(Blocked::Refused(Refused::Repoint(refusal))),
        },
        Err(other) => return Err(Blocked::Refused(Refused::Lifecycle(other))),
    };
    // The empty access list, offered to the type that knows whether it is the
    // truth. It is, for an operation that names no resource; for one that does,
    // the constructor says which resource is unaccounted for and that is the
    // gap.
    let accesses = lifecycle_accesses(fifo, &op, resolvers);
    LifecyclePayload::new(op, accesses)
        .map_err(|mismatch| Blocked::Refused(Refused::LifecycleAccesses(mismatch)))
}

/// The access every resource a lifetime operation names is subject to.
///
/// # The mode is the operation's and the key is the device's
///
/// Two terms with two owners, joined here and derived in neither place. The
/// direction — what the command does to the memory it names — is a property of
/// the command, so `reims_vgpu_core::lifecycle::LifecycleOp::declared_access`
/// states it once for the whole model; a table here would be that statement's
/// second copy, and the copy a device kept would be the one that drifted. Where
/// those bytes are is not the model's to know at all, so it arrives through
/// `reims_vgpu_core::resolve::ResourceStorage`.
///
/// # An unresolvable target is `DomainOnly`, counted, and not dropped
///
/// A resource whose storage this device has no key for still takes part in the
/// operation, and an access list missing it is a list
/// `reims_vgpu_core::transaction::LifecyclePayload` refuses — which would lose
/// the whole packet over one resource. `AccessKey::DomainOnly` is the
/// vocabulary for exactly this: participation is real, the memory is not
/// established, ordering comes from the submission domain alone. It is the same
/// judgement [`present_payload`] makes for a mapping with no live surface, and
/// it is counted for the same reason — so a boot says how many lifetime edges
/// were bought at the coarse rung rather than leaving it to inference.
///
/// The whole backing, because the operation names no narrower extent: a
/// synchronise names a resource, and even an invalidate's validity quad is not
/// carried into the operation. Rung 2 is the finest answer the packet contains.
fn lifecycle_accesses(
    fifo: Fifo,
    op: &reims_vgpu_core::lifecycle::LifecycleOp,
    resolvers: Resolvers<'_>,
) -> Vec<(
    reims_vgpu_core::identity::ResourceId,
    reims_vgpu_core::access::AccessIntent,
)> {
    use reims_vgpu_core::access::{AccessIntent, AccessKey};
    // Asked of the operation rather than of the kind: an operation that names
    // no resource declares no access, and the two answers are one match in one
    // file so they cannot disagree about which commands those are.
    let Some(mode) = op.declared_access() else {
        return Vec::new();
    };
    let task = op.task();
    op.resources()
        .iter()
        .map(|&resource| {
            let key = match resolvers.storage.storage(task, resource) {
                Some(key) => {
                    crate::runtime::drain::note_store_route("ingress_lifecycle_target_resolved");
                    AccessKey::Whole(key)
                }
                None => {
                    crate::runtime::drain::note_store_route("ingress_lifecycle_target_unresolved");
                    AccessKey::DomainOnly
                }
            };
            (
                resource,
                AccessIntent {
                    domain: fifo.domain(),
                    key,
                    mode,
                    // A lifetime command names no shader stage. Zero always
                    // means the record carried none, never that one was dropped.
                    api_stages: 0,
                    // No content version is established for a lifecycle access.
                    // The content authority is `reims_vgpu_core::content`'s and
                    // it is not consulted here; `None` says the term is absent,
                    // which is what it is, rather than reserving a version this
                    // bridge has no authority to hand out.
                    input_content_version: None,
                    output_content_version: None,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    // The channel bound, read only by the test that sweeps it: `Fifo::child`
    // asks `is_child_channel` and never the constant behind it.
    use crate::model::MAX_CHANNELS;
    use reims_vgpu_core::identity::TaskId;
    use reims_vgpu_protocol::packets::LEDGER;

    /// A payload long enough for every control command's own layout and every
    /// present trailer. Only the two channel-lifetime commands and the three
    /// present forms read one at all.
    const ROOMY: usize = 64;

    /// A mapping namespace a test states outright, so what a present resolves
    /// to is the test's choice and not a device's state.
    struct Mappings(Vec<u32>);

    impl reims_vgpu_core::resolve::MappingResolver for Mappings {
        fn backing(
            &self,
            mapping: reims_vgpu_core::identity::MappingId,
        ) -> Option<reims_vgpu_core::access::BackingId> {
            self.0
                .contains(&mapping.0)
                .then(|| reims_vgpu_core::access::BackingId(u64::from(mapping.0) | 1 << 40))
        }
    }

    /// A namespace nothing is live in, for the rows whose class never asks.
    const NOTHING_MAPPED: Mappings = Mappings(Vec::new());

    /// A mapping namespace that answers for every id, so a ledger row that
    /// gapped because a fixture held no mapping would not be reporting on the
    /// fixture.
    struct EveryMapping;

    impl reims_vgpu_core::resolve::MappingResolver for EveryMapping {
        fn backing(
            &self,
            mapping: reims_vgpu_core::identity::MappingId,
        ) -> Option<reims_vgpu_core::access::BackingId> {
            Some(reims_vgpu_core::access::BackingId(
                u64::from(mapping.0) | 1 << 41,
            ))
        }
    }

    /// An object namespace that answers for every ref in every task.
    ///
    /// The ledger sweep is about which join a row reaches and what it is still
    /// short of, so a row that gapped because a fixture happened not to hold an
    /// object would be reporting on the fixture.
    struct EveryObject;

    impl reims_vgpu_core::resolve::TaskNamespaces for EveryObject {
        fn resource(
            &self,
            task: reims_vgpu_core::identity::TaskId,
            object_ref: u32,
        ) -> Option<reims_vgpu_core::identity::ResourceId> {
            Some(reims_vgpu_core::identity::ResourceId {
                slot: reims_vgpu_core::identity::ObjectListRef(object_ref),
                generation: reims_vgpu_core::identity::SlotGeneration(
                    u64::from(task.0).saturating_add(1),
                ),
            })
        }
    }

    /// A storage source that answers for every resource, keyed on the name so
    /// two different names never collide on one backing.
    ///
    /// Paired with [`EveryObject`] for the ledger sweep's reason: a row that
    /// came out at the coarse rung because a fixture held no storage would be
    /// reporting on the fixture.
    struct EveryStorage;

    impl reims_vgpu_core::resolve::ResourceStorage for EveryStorage {
        fn storage(
            &self,
            task: reims_vgpu_core::identity::TaskId,
            resource: reims_vgpu_core::identity::ResourceId,
        ) -> Option<reims_vgpu_core::access::ResourceKey> {
            Some(reims_vgpu_core::access::ResourceKey {
                backing: reims_vgpu_core::access::BackingId(
                    u64::from(task.0) << 32 | u64::from(resource.slot.0) | 1 << 42,
                ),
                heap: None,
            })
        }
    }

    /// A storage source with no term for anything, for the test that asserts
    /// what a lifetime packet does when its target cannot be keyed.
    struct NoStorage;

    impl reims_vgpu_core::resolve::ResourceStorage for NoStorage {
        fn storage(
            &self,
            _task: reims_vgpu_core::identity::TaskId,
            _resource: reims_vgpu_core::identity::ResourceId,
        ) -> Option<reims_vgpu_core::access::ResourceKey> {
            None
        }
    }

    /// A reply-destination source that answers for every question, keyed on the
    /// kind so two questions never share a destination.
    ///
    /// The window is a fixed page, which no assertion here reads: what the
    /// ledger sweep asks is which rows cross, and a row that gapped because a
    /// fixture had no destination would be reporting on the fixture.
    struct EveryReply;

    impl reims_vgpu_core::query::Destinations for EveryReply {
        fn destination(
            &self,
            kind: reims_vgpu_core::query::QueryKind,
            _payload: &[u8],
        ) -> Option<reims_vgpu_core::query::ReplyDestination> {
            Some(reims_vgpu_core::query::ReplyDestination {
                backing: reims_vgpu_core::access::BackingId(kind as u64 | 1 << 43),
                bytes: reims_vgpu_core::access::ByteRange {
                    offset: 0,
                    length: 0x1000,
                },
            })
        }
    }

    fn resolvers(mappings: &dyn MappingResolver) -> Resolvers<'_> {
        Resolvers {
            mappings,
            objects: &EveryObject,
            storage: &EveryStorage,
            replies: &EveryReply,
        }
    }

    fn drained(opcode: u16) -> drain::Packet {
        drain::Packet {
            opcode,
            stamp_waits: Vec::new(),
            total_size: 0,
            completion_stamp: 0,
            payload: vec![0u8; ROOMY],
            next_head: 0,
        }
    }

    /// The same packet, with a one-record resource list for the four commands
    /// that carry one.
    ///
    /// A zero-filled payload decodes those four as a list of **no** resources,
    /// and an operation naming no resource legitimately crosses — so a ledger
    /// sweep on the bare fixture would report the list commands as crossing and
    /// never exercise the path where a named resource owes an access. The list
    /// is one record long, and the invalidate's record carries the guest-write
    /// quad because a zero quad is a refusal about the guest's bytes rather than
    /// a statement about this bridge.
    fn drained_with_list(channel: Channel, opcode: u16) -> drain::Packet {
        use reims_vgpu_core::lifecycle::LifecycleKind;
        use reims_vgpu_protocol::fifo::{
            InvalidateValidityOps, CHILD_INVALIDATE_RECORD_LEN, CHILD_SYNCHRONIZE_RECORD_LEN,
        };
        let mut packet = drained(opcode);
        if let Some(kind) = reims_vgpu_core::query::QueryKind::of(channel, opcode) {
            fill_query_request(&mut packet, kind);
            return packet;
        }
        // The destroy command is the one whose payload contains a second framed
        // record, so zeros are not a well-formed one of it. Filled with a kind
        // the ledger has settled: an unsettled kind is a refusal this fixture
        // deliberately does not exercise, because it is about the *kind* and
        // this sweep is about the class.
        if LifecycleKind::of(channel, opcode) == Some(LifecycleKind::DeleteObject) {
            use reims_vgpu_protocol::destroy::DestroyKind;
            use reims_vgpu_protocol::fifo::DELETE_OBJECT_RECORD as REC;
            packet.payload[REC..REC + 4]
                .copy_from_slice(&DestroyKind::SamplerState.opcode().to_le_bytes());
            packet.payload[REC + 4..REC + 8].copy_from_slice(&12u32.to_le_bytes());
            packet.payload[REC + 8..REC + 12].copy_from_slice(&9u32.to_le_bytes());
            packet.payload.truncate(REC + 12);
            return packet;
        }
        let Some(record_len) =
            LifecycleKind::of(channel, opcode).and_then(LifecycleKind::resource_list_record_len)
        else {
            return packet;
        };
        // `{u32 task, u32 count}` then `count` records. The count's offset is
        // the header's own, and the record's shape is the length's.
        packet.payload[4..8].copy_from_slice(&1u32.to_le_bytes());
        packet.payload[8..12].copy_from_slice(&9u32.to_le_bytes());
        assert!(
            record_len == CHILD_INVALIDATE_RECORD_LEN || record_len == CHILD_SYNCHRONIZE_RECORD_LEN
        );
        if record_len == CHILD_INVALIDATE_RECORD_LEN {
            packet.payload[12..16]
                .copy_from_slice(&InvalidateValidityOps::PAGEON.to_le_dword().to_le_bytes());
        }
        packet
    }

    /// Make a query row's payload a request its own layout accepts.
    ///
    /// The zeroed fixture is not one for every question: the heap-texture
    /// request refuses a null reply address and a window too small for an
    /// `MTLSizeAndAlign`, and it frames an embedded record whose declared
    /// length must match the payload's. A sweep run on the bare fixture would
    /// report that row as refused and never reach the destination resolver at
    /// all, which is the half this class was gapped on.
    fn fill_query_request(packet: &mut drain::Packet, kind: reims_vgpu_core::query::QueryKind) {
        use reims_vgpu_core::query::QueryKind;
        use reims_vgpu_protocol::fifo::{
            HEAP_TEXTURE_REPLY_GVA, HEAP_TEXTURE_REPLY_LEN, HEAP_TEXTURE_REPLY_LENGTH,
            HEAP_TEXTURE_REQUEST_HEADER_LEN, HEAP_TEXTURE_SERIALIZED_LEN,
            HEAP_TEXTURE_SERIALIZED_TAG, HEAP_TEXTURE_SERIALIZER_LENGTH,
        };
        match kind {
            QueryKind::HeapTextureSizeAndAlign => {
                packet.payload =
                    vec![0u8; HEAP_TEXTURE_REQUEST_HEADER_LEN + HEAP_TEXTURE_SERIALIZED_LEN];
                packet.payload[HEAP_TEXTURE_REPLY_GVA..HEAP_TEXTURE_REPLY_GVA + 8]
                    .copy_from_slice(&0x4000u64.to_le_bytes());
                packet.payload[HEAP_TEXTURE_REPLY_LENGTH..HEAP_TEXTURE_REPLY_LENGTH + 8]
                    .copy_from_slice(&(HEAP_TEXTURE_REPLY_LEN as u64).to_le_bytes());
                packet.payload[HEAP_TEXTURE_SERIALIZER_LENGTH..HEAP_TEXTURE_SERIALIZER_LENGTH + 4]
                    .copy_from_slice(&(HEAP_TEXTURE_SERIALIZED_LEN as u32).to_le_bytes());
                // The embedded record's own head: the selector's opcode and the
                // record's length, both from the protocol's constants rather
                // than written again here.
                let record = HEAP_TEXTURE_REQUEST_HEADER_LEN;
                packet.payload[record..record + 4]
                    .copy_from_slice(&HEAP_TEXTURE_SERIALIZED_TAG.to_le_bytes());
                packet.payload[record + 4..record + 8]
                    .copy_from_slice(&(HEAP_TEXTURE_SERIALIZED_LEN as u32).to_le_bytes());
            }
            // The other three accept the zeroed fixture: their words are counts
            // and addresses, and a zero is a valid one of each for a decode.
            QueryKind::DeviceInfo | QueryKind::DeviceInfoLegacy | QueryKind::ComputeInfo => {}
        }
    }

    /// An access source for the call sites whose packet is not an exec.
    ///
    /// It refuses by panicking rather than by answering, so a fixture that
    /// silently started carrying a command stream would fail here instead of
    /// building a transaction out of a stub's answers.
    struct NoAccesses;

    impl reims_vgpu_core::access::AccessSource for NoAccesses {
        fn access(
            &mut self,
            _: &reims_vgpu_core::access::Participation,
        ) -> Result<reims_vgpu_core::access::AccessIntent, reims_vgpu_core::access::AccessRefusal>
        {
            unreachable!("this fixture's packet names no command stream")
        }
    }

    /// The storage a fixture's re-point names, so the one command whose
    /// operation the wire cannot complete has the half that is not on it.
    ///
    /// A value the test states rather than one the bridge invents: the point of
    /// [`ReplacementStorage`] is that these bytes come from a device that has
    /// just moved the storage incarnation, and a fixture standing in for that
    /// device has to say so out loud.
    const REPLACEMENT: ReplacementStorage = ReplacementStorage {
        backing: reims_vgpu_core::access::BackingId(0x5eed),
        extent: reims_vgpu_core::access::ByteRange {
            offset: 0,
            length: 0x4000,
        },
    };

    /// The device inputs for a packet that carries no command buffers, with a
    /// re-point's storage available.
    fn device_inputs(accesses: &mut NoAccesses) -> DeviceInputs<'_> {
        DeviceInputs {
            exec: ExecStreams {
                task: reims_vgpu_core::identity::TaskId(0),
                usages: &reims_vgpu_core::pipeline::NoUsage,
                streams: &[],
                accesses,
            },
            repointed: Ok(REPLACEMENT),
        }
    }

    fn fifo_for(channel: Channel) -> Fifo {
        match channel {
            Channel::Root => Fifo::ROOT,
            Channel::Child => Fifo::child(1).expect("channel 1 is a child"),
        }
    }

    /// **The cutover ledger.** Every row the protocol crate judged, put through
    /// this bridge, and the answer asserted against the class the row has.
    ///
    /// A row that changes class, or a gap that closes without this being
    /// updated, fails here rather than becoming a silent change in what crosses
    /// to the model.
    #[test]
    fn every_ledger_row_either_crosses_or_names_what_it_needs() {
        let mut crossed = 0usize;
        let mut gapped = 0usize;
        for row in LEDGER {
            let fifo = fifo_for(row.channel);
            let outcome = packet(
                fifo,
                SessionGeneration::FIRST,
                StampSlot(0),
                &drained_with_list(row.channel, row.opcode),
                resolvers(&EveryMapping),
                device_inputs(&mut NoAccesses),
            );
            let expected = match classify(row.channel, row.opcode) {
                Some(PayloadClass::Control) => None,
                Some(PayloadClass::Exec) => None,
                // The class is not the unit here, and this is the one place
                // that says so. Two of its members — a task definition and a
                // task deletion — name a task and nothing inside it, so they
                // need no namespace and cross; the other ten name objects,
                // mappings or counted lists of them and do not.
                // The class is not the unit, and this is the one place that
                // says how it splits. Asked of the kind table rather than of a
                // list of opcodes written here, so the test cannot drift into
                // agreeing with itself.
                // Every member of the class crosses. The re-point was the last
                // holdout and it crosses on the storage the caller supplies —
                // see `ReplacementStorage`.
                Some(PayloadClass::ResourceLifecycle) => None,
                // Crosses. Its reply destination is resolved in whichever
                // address space its own question uses; see `query_payload`.
                Some(PayloadClass::Query) => None,
                // Crosses, and crosses on a namespace nothing is live in: the
                // target resolves to no backing and the access says so. That
                // the *unresolved* case still crosses is the assertion — a
                // bridge that gapped when it could not resolve a mapping would
                // stop presenting the moment a surface was reconfigured.
                Some(PayloadClass::Present) => None,
                None => Some(Gap::Unresolved),
            };
            match (expected, outcome) {
                (None, Ok(built)) => {
                    crossed += 1;
                    assert_eq!(built.opcode, row.opcode);
                    assert_eq!(built.channel, row.channel);
                    assert_eq!(built.domain, fifo.domain());
                    let right_class = match classify(row.channel, row.opcode) {
                        Some(PayloadClass::Control) => matches!(built.payload, Payload::Control(_)),
                        Some(PayloadClass::ResourceLifecycle) => {
                            matches!(built.payload, Payload::ResourceLifecycle(_))
                        }
                        Some(PayloadClass::Present) => matches!(built.payload, Payload::Present(_)),
                        Some(PayloadClass::Query) => matches!(built.payload, Payload::Query(_)),
                        Some(PayloadClass::Exec) => matches!(built.payload, Payload::Exec(_)),
                        // A row the ledger has not judged has no class and
                        // gaps, so nothing reaches here with `None`.
                        None => false,
                    };
                    assert!(
                        right_class,
                        "{} {:#04x} ({}) built a payload that is not its class",
                        row.channel.name(),
                        row.opcode,
                        row.name
                    );
                }
                (
                    Some(want),
                    Err(Blocked::Gap {
                        gap,
                        channel,
                        opcode,
                    }),
                ) => {
                    gapped += 1;
                    assert_eq!(
                        gap,
                        want,
                        "{} {:#04x} ({}) named the wrong missing input",
                        row.channel.name(),
                        row.opcode,
                        row.name
                    );
                    assert_eq!((channel, opcode), (row.channel, row.opcode));
                }
                (want, got) => panic!(
                    "{} {:#04x} ({}) expected {want:?} and got {got:?}",
                    row.channel.name(),
                    row.opcode,
                    row.name
                ),
            }
        }
        assert_eq!(
            crossed + gapped,
            LEDGER.len(),
            "every row is accounted for exactly once"
        );
        assert!(
            crossed > 0 && gapped > 0,
            "one side of the partition is empty, so the assertions above compared nothing: \
             {crossed} crossed, {gapped} gapped"
        );
    }

    /// **What crosses is every row the ledger has judged, and nothing else.**
    ///
    /// The complement used to be the claim and there is no longer one to state:
    /// the last two gaps closed together, EXEC on [`ExecStreams`] and the
    /// re-point on [`ReplacementStorage`], so what a row needs is now exactly
    /// what its own closure says. A gap here means a ledger row that is not
    /// settled, which is a contract question and not a missing input.
    ///
    /// Both of those cross on inputs a *fixture* supplies, and this test says
    /// only that the envelope is built. That the streams' records and the
    /// re-point's operation carry the right contents is asserted separately,
    /// because an empty transaction and a defaulted operation would satisfy
    /// this one.
    ///
    /// The counts are spelled out because this is the cutover's own scoreboard:
    /// a gap that closes moves a number here, and a row that quietly changed
    /// class moves one without anybody deciding to.
    #[test]
    fn what_crosses_is_every_row_the_ledger_has_judged() {
        let crossing: Vec<_> = LEDGER
            .iter()
            .filter(|row| {
                packet(
                    fifo_for(row.channel),
                    SessionGeneration::FIRST,
                    StampSlot(0),
                    &drained_with_list(row.channel, row.opcode),
                    resolvers(&EveryMapping),
                    device_inputs(&mut NoAccesses),
                )
                .is_ok()
            })
            .map(|row| (row.channel, row.opcode))
            .collect();
        let expected: Vec<_> = LEDGER
            .iter()
            .filter(|row| classify(row.channel, row.opcode).is_some())
            .map(|row| (row.channel, row.opcode))
            .collect();
        assert_eq!(crossing, expected);
        let control = LEDGER
            .iter()
            .filter(|row| classify(row.channel, row.opcode) == Some(PayloadClass::Control))
            .count();
        let present = LEDGER
            .iter()
            .filter(|row| classify(row.channel, row.opcode) == Some(PayloadClass::Present))
            .count();
        assert_eq!(
            (control, present, crossing.len()),
            (23, 3, 47),
            "the ledger's crossing rows changed; what reaches the model is not what the \
             module documentation says it is"
        );
    }

    /// A re-point crosses carrying the model's identity and the device's bytes,
    /// and the two halves come from the two places that own them.
    ///
    /// The failure this guards is specific and silent. `ReplacePhysical`'s
    /// packet is a bare `{task, object}` and its new pages are nowhere on the
    /// wire, so the storage has to arrive from outside. If the caller supplied
    /// the *resource* as well, that would be a second resolution of the same
    /// ref, and an operation built from the pair would re-point whichever of
    /// the two was wrong while reporting success. So the task and the resource
    /// here are the model's own resolution, carried back on
    /// `ResolveRefusal::NeedsStorage`, and only the backing and extent are the
    /// caller's.
    #[test]
    fn a_repoint_crosses_with_the_models_identity_and_the_devices_storage() {
        use reims_vgpu_core::lifecycle::LifecycleOp;
        use reims_vgpu_core::resolve::TaskNamespaces as _;

        // `{u32 task, u32 object}`, at the offsets the protocol states.
        let mut drained = drained(0x3c);
        use reims_vgpu_protocol::fifo::{TASK_OBJECT_OBJECT_ID, TASK_OBJECT_TASK_ID};
        drained.payload[TASK_OBJECT_TASK_ID..TASK_OBJECT_TASK_ID + 4]
            .copy_from_slice(&5u32.to_le_bytes());
        drained.payload[TASK_OBJECT_OBJECT_ID..TASK_OBJECT_OBJECT_ID + 4]
            .copy_from_slice(&11u32.to_le_bytes());

        let built = packet(
            fifo_for(Channel::Child),
            SessionGeneration::FIRST,
            StampSlot(0),
            &drained,
            resolvers(&EveryMapping),
            device_inputs(&mut NoAccesses),
        )
        .expect("the caller supplied the storage the wire does not carry");
        let Payload::ResourceLifecycle(payload) = &built.payload else {
            panic!("a re-point must not build another class's payload");
        };
        assert_eq!(
            payload.op(),
            &LifecycleOp::ReplacePhysical {
                task: TaskId(5),
                resource: EveryObject
                    .resource(TaskId(5), 11)
                    .expect("the fixture resolves every ref"),
                backing: REPLACEMENT.backing,
                extent: REPLACEMENT.extent,
            },
            "the identity is the packet's own words resolved in the packet's own \
             task, and the bytes are the ones the caller handed in"
        );
    }

    /// A re-point whose storage the device cannot describe is refused under the
    /// device's own reason, and is not a gap.
    ///
    /// The alternative was an operation carrying the *old* backing, which
    /// re-points nothing and reports success — a resource whose content
    /// authority stays on pages the guest has already rewired, with nothing to
    /// read. Measured at zero on a driven macos-15 boot once a surface-backing
    /// object stopped being read as owning no bytes, which is what makes
    /// refusing affordable rather than a wall.
    #[test]
    fn a_repoint_the_device_cannot_describe_is_refused_under_the_devices_reason() {
        use crate::runtime::objects::RepointStorageRefusal;

        let mut accesses = NoAccesses;
        let undescribable = RepointStorageRefusal::NoBytes {
            object_type: 6,
            mapping_names_a_surface: false,
        };
        let refusal = packet(
            fifo_for(Channel::Child),
            SessionGeneration::FIRST,
            StampSlot(0),
            &drained_with_list(Channel::Child, 0x3c),
            resolvers(&EveryMapping),
            DeviceInputs {
                exec: ExecStreams {
                    task: reims_vgpu_core::identity::TaskId(0),
                    usages: &reims_vgpu_core::pipeline::NoUsage,
                    streams: &[],
                    accesses: &mut accesses,
                },
                repointed: Err(undescribable),
            },
        )
        .expect_err("the device has no storage for this reference");
        assert_eq!(
            refusal.gap(),
            None,
            "the input is not missing from this bridge; the device looked and \
             there was nothing to name"
        );
        assert_eq!(refusal, Blocked::Refused(Refused::Repoint(undescribable)));
        assert_eq!(
            refusal.slug(),
            undescribable.route(),
            "reported under the route the device already counts it on"
        );
    }

    /// An exec packet's command stream refuses as the *walk's* refusal, with
    /// its site, and not as a gap.
    ///
    /// The distinction is the one [`Blocked`]'s two arms exist for. A gap is
    /// this device's incompleteness and closes when its owner lands; a stream
    /// that does not frame is the guest's bytes not being what its opcode says,
    /// and no owner landing will change it. Exec spent the whole cutover on the
    /// first side of that line, so the assertion that it is now on the second is
    /// worth making explicitly.
    #[test]
    fn an_exec_whose_stream_does_not_frame_is_refused_and_not_gapped() {
        let mut accesses = NoAccesses;
        let streams = vec![vec![0xffu8; 2]];
        let refusal = packet(
            fifo_for(Channel::Child),
            SessionGeneration::FIRST,
            StampSlot(0),
            &drained_with_list(Channel::Child, 0x37),
            resolvers(&EveryMapping),
            DeviceInputs {
                exec: ExecStreams {
                    task: reims_vgpu_core::identity::TaskId(1),
                    usages: &reims_vgpu_core::pipeline::NoUsage,
                    streams: &streams,
                    accesses: &mut accesses,
                },
                repointed: Ok(REPLACEMENT),
            },
        )
        .expect_err("two bytes are not a command stream");
        assert_eq!(
            refusal.gap(),
            None,
            "a stream that does not frame is not an input this bridge is \
             missing; it is bytes that are not a stream"
        );
        let Blocked::Refused(Refused::Exec(walk)) = refusal else {
            panic!("an exec's stream must refuse under the walk's own vocabulary");
        };
        assert_eq!(
            refusal.slug(),
            walk.reason(),
            "carried whole from the layer that judged it, not re-worded here"
        );
    }

    /// A task definition crosses whole: the packet's own fields, decoded, with
    /// no access list because the operation names nothing for one to be about.
    ///
    /// The two crossing lifetime commands are asserted for their *content* and
    /// not only for the fact that they cross. A bridge that produced a
    /// well-formed envelope around a task id it had defaulted would pass the
    /// ledger test above and hand the model a definition for task zero.
    #[test]
    fn a_task_definition_crosses_carrying_its_own_decoded_fields() {
        use reims_vgpu_core::identity::{DirectoryFrame, TaskId};
        use reims_vgpu_core::lifecycle::LifecycleOp;

        let mut drained = drained(0x38);
        // `(task_id << 1) | is_kernel_task`, an address-space length this model
        // deliberately does not carry, then the directory page frame — at the
        // protocol's own offsets rather than at ones restated here.
        use reims_vgpu_protocol::fifo::{DEFINE_TASK_DIRECTORY_PFN, DEFINE_TASK_RAW_ID};
        drained.payload[DEFINE_TASK_RAW_ID..DEFINE_TASK_RAW_ID + 4]
            .copy_from_slice(&((7u32 << 1) | 1).to_le_bytes());
        drained.payload[DEFINE_TASK_DIRECTORY_PFN..DEFINE_TASK_DIRECTORY_PFN + 4]
            .copy_from_slice(&0x1234u32.to_le_bytes());

        let built = packet(
            Fifo::ROOT,
            SessionGeneration::FIRST,
            StampSlot(0),
            &drained,
            resolvers(&NOTHING_MAPPED),
            device_inputs(&mut NoAccesses),
        )
        .expect("a task definition needs no namespace");
        let Payload::ResourceLifecycle(payload) = &built.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        assert_eq!(
            payload.op(),
            &LifecycleOp::DefineTask {
                task: TaskId(7),
                kernel: true,
                directory: DirectoryFrame(0x1234),
            },
            "the fields are the packet's, not defaults around a well-formed \
             envelope"
        );
        assert!(
            payload.accesses().is_empty(),
            "a task definition names no resource, so there is nothing for an \
             access to be about"
        );
    }

    /// A channel-lifetime command with no room for its domain is refused, not
    /// defaulted. Opening domain 0 would name the root FIFO.
    #[test]
    fn a_channel_command_too_short_to_name_a_domain_is_refused() {
        let mut short = drained(0x30);
        short.payload.clear();
        let refusal = packet(
            Fifo::ROOT,
            SessionGeneration::FIRST,
            StampSlot(0),
            &short,
            resolvers(&NOTHING_MAPPED),
            device_inputs(&mut NoAccesses),
        )
        .expect_err("no domain, no transition");
        assert!(matches!(refusal, Blocked::Refused(_)));
        assert_eq!(
            refusal.gap(),
            None,
            "a short payload is not a missing input"
        );
        assert_eq!(
            control::ControlKind::of(Channel::Root, 0x30)
                .and_then(control::ControlKind::channel_transition),
            Some(control::ChannelTransition::Open),
            "the opcode above stopped being the channel-open command, so this test refuses \
             something else"
        );
    }

    /// The envelope: the slot is the FIFO's, the value is the packet's, and a
    /// wait's raw index is masked exactly once.
    #[test]
    fn the_envelope_carries_the_channels_slot_and_the_packets_values() {
        let raw_index = u32::MAX;
        let mut nop = drained(0x1e);
        nop.completion_stamp = 0xDEAD_BEEF;
        nop.stamp_waits = vec![drain::StampWait {
            index: raw_index,
            value: 7,
        }];
        let built = packet(
            fifo_for(Channel::Child),
            SessionGeneration::FIRST,
            StampSlot(3),
            &nop,
            resolvers(&NOTHING_MAPPED),
            device_inputs(&mut NoAccesses),
        )
        .expect("CmdNOP is control");
        assert_eq!(
            built.completion,
            Some(CompletionStamp {
                slot: StampSlot(3),
                value: StampValue(0xDEAD_BEEF),
            })
        );
        assert_eq!(
            built.stamp_waits,
            vec![StampWait {
                slot: StampSlot(stamp_slot_index(raw_index)),
                value: StampValue(7),
            }]
        );
        assert_ne!(
            stamp_slot_index(raw_index),
            raw_index,
            "the raw index used above is not masked by anything, so the assertion that it was \
             masked would hold for a bridge that carried it through untouched"
        );
    }

    /// A lifetime command's refs resolve, and what the packet is still short of
    /// depends on whether it named a resource — not on which opcode it is.
    ///
    /// The same opcode, twice, with two payloads. An invalidate naming nothing
    /// is an operation over no resources and crosses; an invalidate naming one
    /// owes that resource an access and does not. A bridge whose partition was
    /// per-opcode would give both answers to whichever list it was written
    /// against, and the wrong one is silent: an invalidate that crossed with an
    /// empty access list would move content authority with nothing ordered
    /// against it.
    #[test]
    fn a_lifetime_command_owes_an_access_for_each_resource_it_names() {
        use reims_vgpu_core::lifecycle::{LifecycleKind, LifecycleOp};
        use reims_vgpu_protocol::packets::LEDGER;

        let row = LEDGER
            .iter()
            .find(|row| {
                LifecycleKind::of(row.channel, row.opcode) == Some(LifecycleKind::Invalidate)
            })
            .expect("the ledger has an invalidate row");
        let fifo = fifo_for(row.channel);
        let put = |packet: &drain::Packet| packet_of(fifo, packet);

        // Zero records. The operation names no resource, so the empty access
        // list is its own statement rather than a shortfall.
        let empty = put(&drained(row.opcode)).expect("an invalidate over nothing is an operation");
        let Payload::ResourceLifecycle(payload) = &empty.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        assert!(matches!(
            payload.op(),
            LifecycleOp::Invalidate { resources, .. } if resources.is_empty()
        ));
        assert!(payload.accesses().is_empty());

        // One record. The ref resolves and the resource it names is keyed, so
        // the packet crosses with exactly one access — over that resource, at
        // the whole-backing rung, in a direction that orders against a reader.
        let named = put(&drained_with_list(row.channel, row.opcode))
            .expect("a named resource's access is answerable");
        let Payload::ResourceLifecycle(payload) = &named.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        let LifecycleOp::Invalidate { task, resources } = payload.op() else {
            panic!("the invalidate opcode built another operation");
        };
        assert_eq!(resources.len(), 1, "the list carried one record");
        assert_eq!(
            payload.accesses().len(),
            1,
            "one resource, one access — the envelope is what holds those together"
        );
        let access = payload.accesses()[0];
        assert_eq!(
            access.key,
            reims_vgpu_core::access::AccessKey::Whole(
                EveryStorage
                    .storage(*task, resources[0])
                    .expect("the fixture keys every resource")
            ),
            "the access is keyed on the storage the resolver gave for the resource \
             the operation named, not on a default or on another resource's"
        );
        assert_eq!(access.mode, reims_vgpu_core::access::AccessMode::Write);
        assert_eq!(access.domain, fifo.domain());

        // The same packet with nothing to key it on. The resource is still the
        // operation's, so the access is still there — at the coarse rung rather
        // than absent, because an access list one short of the operation's own
        // resources is a packet the envelope refuses outright.
        let unkeyed = packet(
            fifo,
            SessionGeneration::FIRST,
            StampSlot(0),
            &drained_with_list(row.channel, row.opcode),
            Resolvers {
                mappings: &EveryMapping,
                objects: &EveryObject,
                storage: &NoStorage,
                replies: &EveryReply,
            },
            device_inputs(&mut NoAccesses),
        )
        .expect("an unkeyable target is not a lost packet");
        let Payload::ResourceLifecycle(payload) = &unkeyed.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        assert_eq!(
            payload.accesses().iter().map(|a| a.key).collect::<Vec<_>>(),
            vec![reims_vgpu_core::access::AccessKey::DomainOnly],
            "with no storage term the ordering is the submission domain's, and the \
             access is still declared"
        );
    }

    /// An object namespace that names nothing, for the packets whose whole
    /// question is what an unnameable reference does.
    struct NoObjects;

    impl reims_vgpu_core::resolve::TaskNamespaces for NoObjects {
        fn resource(
            &self,
            _task: reims_vgpu_core::identity::TaskId,
            _object_ref: u32,
        ) -> Option<reims_vgpu_core::identity::ResourceId> {
            None
        }
    }

    /// A delete whose ref names nothing still crosses; a re-point's does not.
    ///
    /// **The two commands carry the same `{task, object}` record and must not
    /// get the same answer.** The guest clears its own object-list slot before
    /// it sends the delete, so an unnameable ref is the *ordinary* case for a
    /// delete — a driven macos-15 boot through this model refused 741 packets of
    /// guest work on it, each one a `CmdDeleteResource` whose device-side
    /// teardown is keyed by `(task, ref)` and needed no name at all. A re-point
    /// with no resource to re-point is not a re-point, and still refuses.
    ///
    /// The empty access list is the other half: a delete this model cannot name
    /// is ordered against nothing, because a resource with no name is one no
    /// other operation in the model can name either.
    #[test]
    fn an_unnamed_ref_still_deletes_and_still_refuses_a_repoint() {
        use reims_vgpu_core::lifecycle::{LifecycleKind, LifecycleOp};

        let row_for = |kind| {
            LEDGER
                .iter()
                .find(|row| LifecycleKind::of(row.channel, row.opcode) == Some(kind))
                .expect("the ledger has a row for every lifetime kind")
        };
        let unnamed = |row: &reims_vgpu_protocol::packets::Packet| {
            let fifo = fifo_for(row.channel);
            packet(
                fifo,
                SessionGeneration::FIRST,
                StampSlot(0),
                &drained(row.opcode),
                Resolvers {
                    mappings: &EveryMapping,
                    objects: &NoObjects,
                    storage: &EveryStorage,
                    replies: &EveryReply,
                },
                device_inputs(&mut NoAccesses),
            )
        };

        let deleted = unnamed(row_for(LifecycleKind::DeleteResource))
            .expect("a delete is not lost to a reference the guest already cleared");
        let Payload::ResourceLifecycle(payload) = &deleted.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        assert!(
            matches!(
                payload.op(),
                LifecycleOp::DeleteResource {
                    object_ref,
                    resource: None,
                    ..
                } if *object_ref == 0
            ),
            "the reference travels even when the name does not: {:?}",
            payload.op()
        );
        assert!(
            payload.accesses().is_empty(),
            "a delete this model cannot name is ordered against nothing"
        );

        assert!(
            matches!(
                unnamed(row_for(LifecycleKind::ReplacePhysical)),
                Err(Blocked::Refused { .. })
            ),
            "a re-point with no resource to re-point is not a re-point"
        );
    }

    /// A backing retirement whose mapping resolves to nothing still retires.
    ///
    /// The third command under the same rule as the two above, and the one
    /// whose id is a **mapping** rather than an object-list ref. A driven
    /// macos-15 boot refused one packet a boot on it. The operation carries the
    /// guest's mapping id either way, because that is the only thing the rail's
    /// own teardown can be keyed by.
    #[test]
    fn a_retirement_whose_mapping_resolves_to_nothing_still_retires() {
        use reims_vgpu_core::lifecycle::{LifecycleKind, LifecycleOp};

        let row = LEDGER
            .iter()
            .find(|row| {
                LifecycleKind::of(row.channel, row.opcode) == Some(LifecycleKind::DeleteBacking)
            })
            .expect("the ledger has a backing-retirement row");
        let built = packet(
            fifo_for(row.channel),
            SessionGeneration::FIRST,
            StampSlot(0),
            &drained(row.opcode),
            Resolvers {
                mappings: &NOTHING_MAPPED,
                objects: &EveryObject,
                storage: &EveryStorage,
                replies: &EveryReply,
            },
            device_inputs(&mut NoAccesses),
        )
        .expect("a retirement is not lost to an unheld mapping");
        let Payload::ResourceLifecycle(payload) = &built.payload else {
            panic!("a lifetime command must not build another class's payload");
        };
        assert!(
            matches!(
                payload.op(),
                LifecycleOp::DeleteBacking { backing: None, .. }
            ),
            "the mapping named no bytes this model holds: {:?}",
            payload.op()
        );
        assert!(payload.accesses().is_empty());
    }

    /// [`packet`] over the ledger sweep's namespaces, so a test naming one
    /// packet does not restate the four arguments the sweep already fixes.
    fn packet_of(fifo: Fifo, drained: &drain::Packet) -> Result<Packet, Blocked> {
        packet(
            fifo,
            SessionGeneration::FIRST,
            StampSlot(0),
            drained,
            resolvers(&EveryMapping),
            device_inputs(&mut NoAccesses),
        )
    }

    /// A packet's ordering is priced by mode, and "nobody knows" is priced
    /// apart from "read-modify-write".
    ///
    /// [`reims_vgpu_core::access::AccessMode::Unknown`] orders exactly as
    /// `ReadWrite` does, so the only thing the split can ever buy is a number:
    /// how much of a boot's ordering exists because a slot genuinely is
    /// read-modify-write, and how much exists because no reflection was
    /// published for the pipeline that bound it. Seam 6 asks for zero supported
    /// operations reaching that fallback, and until this census the variant
    /// carried the argument for counting without anything doing the counting —
    /// which makes the required zero unmeasured rather than met.
    ///
    /// Two lifetime commands with the same shape and different modes, so a
    /// census that bucketed by class rather than by mode would pass with both
    /// landing in one row: an invalidate declares a write, a discard declares
    /// that it does not know.
    #[test]
    fn an_unknown_mode_access_is_counted_apart_from_a_known_one() {
        use crate::runtime::drain::{note_access_modes, store_route_count};
        use reims_vgpu_core::lifecycle::LifecycleKind;
        use reims_vgpu_protocol::packets::LEDGER;

        let row_of = |want: LifecycleKind| {
            LEDGER
                .iter()
                .find(|row| LifecycleKind::of(row.channel, row.opcode) == Some(want))
                .expect("the ledger has a row for every settled lifetime kind")
        };
        // A device only so the census can ask what the model thinks of a
        // pipeline a draw left unanswered. This fixture's packets are lifetime
        // packets and bind none, so the answer is never reached — the argument
        // is what the counting door takes, not what this test is about.
        let device = crate::model::DeviceState::new(crate::model::DeviceId(1), 12);
        let counted = |want: LifecycleKind| {
            let row = row_of(want);
            let fifo = fifo_for(row.channel);
            let built = packet_of(fifo, &drained_with_list(row.channel, row.opcode))
                .expect("a named resource's access is answerable");
            assert_eq!(
                built.payload.accesses().len(),
                1,
                "the fixture's one record owes exactly one access"
            );
            let before = [
                store_route_count("access_mode_read"),
                store_route_count("access_mode_write"),
                store_route_count("access_mode_read_write"),
                store_route_count("access_mode_unknown"),
            ];
            note_access_modes(&device, &built);
            [
                store_route_count("access_mode_read") - before[0],
                store_route_count("access_mode_write") - before[1],
                store_route_count("access_mode_read_write") - before[2],
                store_route_count("access_mode_unknown") - before[3],
            ]
        };

        assert_eq!(
            counted(LifecycleKind::Invalidate),
            [0, 1, 0, 0],
            "an invalidate states the direction, so it costs the known column"
        );
        assert_eq!(
            counted(LifecycleKind::Discard),
            [0, 0, 0, 1],
            "a discard does not, and the whole point of the variant is that this \
             lands somewhere a reader can subtract"
        );
    }

    /// A present crosses carrying one read of the backing its target resolves
    /// to, and the frame's target is the packet's own word rather than a
    /// default.
    ///
    /// Two things a well-formed envelope would pass without: the access is over
    /// the backing *this* mapping resolved to, and it is a read. A present
    /// modelled as a writer would reserve a content version, beat the real
    /// writer's, and publish bytes nothing produced.
    #[test]
    fn a_present_crosses_carrying_one_read_of_its_targets_backing() {
        use reims_vgpu_core::access::{AccessKey, AccessMode, ResourceKey};
        use reims_vgpu_core::identity::MappingId;
        use reims_vgpu_core::present::PresentForm;
        use reims_vgpu_protocol::packets::LEDGER;

        let row = LEDGER
            .iter()
            .find(|row| classify(row.channel, row.opcode) == Some(PayloadClass::Present))
            .expect("the ledger has a present row");
        let form = PresentForm::of(row.channel, row.opcode).expect("a present row has a form");

        let mut shown = drained(row.opcode);
        let target = 9u32;
        let at = form.target_offset();
        shown.payload[at..at + 4].copy_from_slice(&target.to_le_bytes());

        let live = Mappings(vec![target]);
        let fifo = fifo_for(row.channel);
        let built = packet(
            fifo,
            SessionGeneration::FIRST,
            StampSlot(0),
            &shown,
            resolvers(&live),
            device_inputs(&mut NoAccesses),
        )
        .expect("a present resolves in the mapping namespace");
        let Payload::Present(payload) = &built.payload else {
            panic!("a present must not build another class's payload");
        };
        assert_eq!(
            payload.packet().mapping,
            MappingId(target),
            "the target is the packet's own word, not a default"
        );
        let backing = {
            use reims_vgpu_core::resolve::MappingResolver as _;
            live.backing(MappingId(target)).expect("live")
        };
        assert_eq!(
            payload.accesses(),
            &[reims_vgpu_core::access::AccessIntent {
                domain: fifo.domain(),
                key: AccessKey::Whole(ResourceKey {
                    backing,
                    heap: None,
                }),
                mode: AccessMode::Read,
                api_stages: 0,
                input_content_version: None,
                output_content_version: None,
            }],
            "one read of the whole of the backing the target resolved to"
        );

        // The same packet against a namespace the target is not live in. It
        // still crosses — a frame the device cannot resolve is still a frame the
        // guest is waiting on — and it says outright that it does not know what
        // it touches.
        let unresolved = packet(
            fifo,
            SessionGeneration::FIRST,
            StampSlot(0),
            &shown,
            resolvers(&NOTHING_MAPPED),
            device_inputs(&mut NoAccesses),
        )
        .expect("an unresolvable target is not a missing input");
        let Payload::Present(payload) = &unresolved.payload else {
            panic!("still a present");
        };
        assert_eq!(
            payload.accesses()[0].key,
            AccessKey::DomainOnly,
            "no backing to name, and the vocabulary for that is not a guess"
        );
        assert_eq!(
            payload.accesses()[0].key.rung(),
            3,
            "and it prices itself on the precision ladder as the coarsest rung"
        );
    }

    /// A present too short for its own trailer is refused, not shown.
    ///
    /// Clamping would present mapping zero and complete the packet in silence,
    /// which is a frame the guest believes it showed.
    #[test]
    fn a_present_too_short_for_its_trailer_is_refused_rather_than_shown() {
        use reims_vgpu_protocol::packets::LEDGER;

        let row = LEDGER
            .iter()
            .find(|row| classify(row.channel, row.opcode) == Some(PayloadClass::Present))
            .expect("the ledger has a present row");
        let mut short = drained(row.opcode);
        short.payload.clear();
        let refusal = packet(
            fifo_for(row.channel),
            SessionGeneration::FIRST,
            StampSlot(0),
            &short,
            resolvers(&NOTHING_MAPPED),
            device_inputs(&mut NoAccesses),
        )
        .expect_err("no trailer, no frame");
        assert_eq!(
            refusal.gap(),
            None,
            "a short payload is this guest's bytes, not this device's incompleteness"
        );
        assert!(matches!(
            refusal,
            Blocked::Refused(Refused::Present(
                reims_vgpu_core::present::ResolveRefusal::Payload(_)
            ))
        ));
    }

    /// The device's channel numbering, which is what makes the channel and the
    /// domain one value rather than two that can disagree.
    #[test]
    fn the_root_fifo_is_channel_zero_and_children_are_the_rest() {
        assert_eq!(Fifo::ROOT.domain(), ChannelId(0));
        assert_eq!(Fifo::ROOT.channel(), Channel::Root);
        assert_eq!(
            Fifo::child(0),
            None,
            "channel 0 is the root FIFO, not a child"
        );
        assert_eq!(
            Fifo::child(MAX_CHANNELS as u32),
            None,
            "one past the last channel this device has"
        );
        for id in 1..MAX_CHANNELS as u32 {
            let fifo = Fifo::child(id).expect("a child this device has");
            assert_eq!(fifo.domain(), ChannelId(id));
            assert_eq!(fifo.channel(), Channel::Child);
        }
    }
}

/// [`device_packet`] against a real [`crate::model::DeviceState`].
///
/// Separate from the module's own suite because the questions are different:
/// that one asks what [`packet`] builds from stated namespaces, and this asks
/// whether the device's namespaces can be *assembled* into the ones it asks
/// for. Those fail in different places — a wrong payload against a wrong door —
/// and a fixture holding both would answer neither cleanly.
#[cfg(test)]
mod device_tests {
    use super::*;
    use crate::model::{DeviceId, DeviceState, CHILD_OP_EXEC_INDIRECT2};
    use crate::runtime::exec::ExecSubmission;
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::access::{BackingId, ByteRange};
    use reims_vgpu_core::lifecycle::Storage;

    const TASK: u32 = 2;
    const TEXTURE: u32 = 11;
    const BACKING: BackingId = BackingId(0x9000);
    const SESSION: SessionGeneration = SessionGeneration::FIRST;
    const SLOT: StampSlot = StampSlot(1);

    /// One `generateMipmapsForTexture:` record, framed the way the guest's
    /// serializer frames one.
    ///
    /// Written here rather than reached for, because the writing side of the
    /// record and segment layouts lives in `reims_vgpu_core`'s own test module
    /// and no production path needs it. What is duplicated is an opcode, a
    /// length and a header — all three read from the wire crate's constants, so
    /// a layout change breaks this the same way it breaks the reader.
    fn one_blit_stream(texture: u32) -> Vec<u8> {
        use reims_vgpu_protocol::segment::SegmentKind;
        use reims_vgpu_wire::ops::segment::SEGMENT_HEADER_LEN;

        let payload = texture.to_le_bytes();
        let mut record = Vec::new();
        record
            .extend_from_slice(&reims_vgpu_wire::ops::blit::OPCODE_GENERATE_MIPMAPS.to_le_bytes());
        record.extend_from_slice(
            &((reims_vgpu_protocol::decode::OP_HEADER_LEN + payload.len()) as u32).to_le_bytes(),
        );
        record.extend_from_slice(&payload);

        let mut out = Vec::new();
        out.extend_from_slice(&((SEGMENT_HEADER_LEN + record.len()) as u32).to_le_bytes());
        out.push(SegmentKind::Blit.wire_type());
        // Self-contained: the segment opens its encoder and ends it.
        out.push(0);
        out.push(0);
        // The byte the serializer never writes.
        out.push(0xaa);
        out.extend_from_slice(&record);
        out
    }

    /// One `executeCommandsInBuffer:withRange:`, in a render segment.
    ///
    /// The ICB is the object whose create descriptor names no allocation and no
    /// window, so this device declares it `Storage::NoBytes` — which is why it
    /// is the record this test is about.
    fn one_icb_stream(icb: u32) -> Vec<u8> {
        use reims_vgpu_protocol::segment::SegmentKind;
        use reims_vgpu_wire::ops::segment::SEGMENT_HEADER_LEN;

        let mut payload = Vec::new();
        payload.extend_from_slice(&icb.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());
        payload.extend_from_slice(&4u64.to_le_bytes());
        let mut record = Vec::new();
        record.extend_from_slice(
            &reims_vgpu_wire::ops::render::OPCODE_EXECUTE_COMMANDS_RANGE.to_le_bytes(),
        );
        record.extend_from_slice(
            &((reims_vgpu_protocol::decode::OP_HEADER_LEN + payload.len()) as u32).to_le_bytes(),
        );
        record.extend_from_slice(&payload);

        let mut out = Vec::new();
        out.extend_from_slice(&((SEGMENT_HEADER_LEN + record.len()) as u32).to_le_bytes());
        out.push(SegmentKind::Render.wire_type());
        out.push(0);
        out.push(0);
        out.push(0xaa);
        out.extend_from_slice(&record);
        out
    }

    fn exec_packet(payload: Vec<u8>) -> drain::Packet {
        drain::Packet {
            opcode: CHILD_OP_EXEC_INDIRECT2,
            stamp_waits: Vec::new(),
            total_size: 0,
            completion_stamp: 9,
            payload,
            next_head: 0,
        }
    }

    fn device_with_texture() -> DeviceState {
        let mut state = DeviceState::new(DeviceId(1), 12);
        state.define_task(TASK, 1 << 20, 0x200);
        let _declared = state
            .declare_object(
                TASK,
                TEXTURE,
                Storage::Dedicated {
                    backing: BACKING,
                    extent: ByteRange {
                        offset: 0,
                        length: 0x4000,
                    },
                },
            )
            .expect("the task is open");
        state
    }

    /// The reads for a packet that is not a re-point, with `submission`
    /// stated by each test.
    fn reads(submission: Option<&ExecSubmission>) -> PacketReads<'_> {
        PacketReads {
            submission,
            repointed: Err(crate::runtime::objects::RepointStorageRefusal::Unnamed),
        }
    }

    /// The assembly the switch is: a real device's namespaces answer a real
    /// command stream, and the access the record claims comes out keyed on the
    /// storage the device declared.
    ///
    /// This is the shape that could not be written before the access door
    /// existed: the resolver and the access source both borrow the device, and
    /// the walk alternates between them once per record.
    #[test]
    fn a_command_stream_resolves_and_records_through_the_devices_own_doors() {
        let state = device_with_texture();
        let host = FakeHost::new();
        let submission = ExecSubmission::stated(TASK, vec![one_blit_stream(TEXTURE)]);
        let drained = exec_packet(Vec::new());

        let built = device_packet(
            &state,
            &host,
            Fifo::child(1).expect("a child channel"),
            SESSION,
            SLOT,
            &drained,
            reads(Some(&submission)),
        )
        .expect("a well-framed blit over a declared texture");

        let Payload::Exec(work) = built.payload else {
            panic!("an exec opcode is the exec class");
        };
        let named: Vec<u64> = work
            .accesses
            .iter()
            .filter_map(|a| match a.key {
                reims_vgpu_core::access::AccessKey::Range(r, _)
                | reims_vgpu_core::access::AccessKey::Subresource(r, _)
                | reims_vgpu_core::access::AccessKey::Whole(r) => Some(r.backing.0),
                reims_vgpu_core::access::AccessKey::DomainOnly
                | reims_vgpu_core::access::AccessKey::Heap(_) => None,
            })
            .collect();
        assert_eq!(
            named,
            vec![BACKING.0],
            "the access is keyed on the storage the device declared for the ref"
        );
        assert_eq!(built.domain, ChannelId(1));
    }

    /// A ref the task never declared refuses the whole packet, rather than
    /// yielding a transaction that is missing the edge the record asked for.
    #[test]
    fn an_undeclared_ref_refuses_the_whole_exec_packet() {
        let state = device_with_texture();
        let host = FakeHost::new();
        let submission = ExecSubmission::stated(TASK, vec![one_blit_stream(TEXTURE + 1)]);
        let drained = exec_packet(Vec::new());

        let blocked = device_packet(
            &state,
            &host,
            Fifo::child(1).expect("a child channel"),
            SESSION,
            SLOT,
            &drained,
            reads(Some(&submission)),
        )
        .expect_err("nothing names that ref and no guest list answers for it");
        assert!(
            matches!(blocked, Blocked::Refused(Refused::Exec(_))),
            "the walk's own refusal, not a gap: {blocked:?}"
        );
    }

    /// An exec packet whose command buffers this device did not read is a gap
    /// and not an empty transaction.
    ///
    /// The failure this rules out is silent: with no streams the walk has
    /// nothing to refuse, so the builder finishes an access list of length zero
    /// and the packet admits, orders against nothing, and executes nothing —
    /// a whole submission of guest draws reported as done.
    #[test]
    fn an_exec_packet_without_its_streams_is_a_gap() {
        let state = device_with_texture();
        let host = FakeHost::new();
        let drained = exec_packet(Vec::new());

        let blocked = device_packet(
            &state,
            &host,
            Fifo::child(1).expect("a child channel"),
            SESSION,
            SLOT,
            &drained,
            reads(None),
        )
        .expect_err("the caller read no command buffers for it");
        assert_eq!(blocked.gap(), Some(Gap::ExecStreamsUnread));
    }
    /// An indirect command buffer does not refuse the packet it is in.
    ///
    /// The ICB owns no bytes this device can address, so it is declared
    /// `Storage::NoBytes` — and `IcbOp::participations` claims a participation
    /// on it anyway, which is right: the operation names the object. Before the
    /// owner learned to answer a no-bytes participation, that claim came back
    /// as `NotDeclared` and refused the whole exec transaction, which is every
    /// draw in the packet.
    ///
    /// No rail driven so far sends one (`icb_ok = 0` on all five), so this is
    /// the only place the behaviour is stated.
    #[test]
    fn an_indirect_command_buffer_does_not_refuse_its_packet() {
        let state = device_with_texture();
        let icb = TEXTURE + 4;
        let _declared = state
            .declare_object(TASK, icb, Storage::NoBytes)
            .expect("the task is open");
        let host = FakeHost::new();
        let submission = ExecSubmission::stated(TASK, vec![one_icb_stream(icb)]);
        let drained = exec_packet(Vec::new());

        let built = device_packet(
            &state,
            &host,
            Fifo::child(1).expect("a child channel"),
            SESSION,
            SLOT,
            &drained,
            reads(Some(&submission)),
        )
        .expect("an ICB execute is work this device does");

        let Payload::Exec(work) = built.payload else {
            panic!("an exec opcode is the exec class");
        };
        assert!(
            work.accesses
                .iter()
                .any(|a| a.key == reims_vgpu_core::access::AccessKey::DomainOnly),
            "the ICB participates, at the rung a resource with no bytes has"
        );
    }
    /// A control packet, long enough for every control command's own layout.
    fn control_packet(opcode: u16, completion: u32) -> drain::Packet {
        drain::Packet {
            opcode,
            stamp_waits: Vec::new(),
            total_size: 0,
            completion_stamp: completion,
            payload: vec![0u8; 64],
            next_head: 0,
        }
    }

    /// One packet's whole path through the model: built, admitted, released to
    /// run, completed, published.
    ///
    /// **The spine of the cutover.** Each of the four doors already had its own
    /// test; what this asserts is that they compose on one real device — the
    /// ordinal `admit_packet` issued is the one `take_ready` hands back and the
    /// one `complete_transaction` publishes for, and the word the guest may
    /// then read is the word the packet carried.
    #[test]
    fn a_packet_is_built_admitted_released_completed_and_published() {
        let state = device_with_texture();
        let host = FakeHost::new();
        let drained = control_packet(crate::model::ROOT_OP_DEFINE_FIFO, 7);

        let built = device_packet(
            &state,
            &host,
            Fifo::ROOT,
            SESSION,
            StampSlot(0),
            &drained,
            reads(None),
        )
        .expect("a control packet on the root FIFO");
        let admission = state.admit_packet(&built).expect("the root domain is open");
        let ingress = admission.admitted.transaction.identity.ingress;

        assert!(
            admission.admitted.ready,
            "it waits on nothing, so it may run at once"
        );
        assert_eq!(
            state.take_ready(),
            vec![ingress],
            "and it leaves the model by the one door work leaves by"
        );

        let published = state
            .complete_transaction(admission.epoch, ingress)
            .expect("completed under the incarnation it was admitted into");
        assert_eq!(
            published
                .iter()
                .filter_map(|r| r.stamp)
                .map(|s| s.value.0)
                .collect::<Vec<_>>(),
            vec![7],
            "the guest may now read the word the packet carried"
        );
    }

    /// A channel publishes in its own order, whatever order its work finishes
    /// in.
    ///
    /// The second packet completing first publishes **nothing**: its word is
    /// held behind the first position's, and both arrive when the first
    /// completes. That is the property the drain's per-packet latch cannot
    /// express, and it is why publication moves to the model rather than
    /// staying a value this device writes when a handler returns.
    #[test]
    fn a_channels_words_arrive_in_the_channels_order_and_not_the_hosts() {
        let state = device_with_texture();
        let host = FakeHost::new();

        let admit = |completion: u32| {
            let drained = control_packet(crate::model::ROOT_OP_DEFINE_FIFO, completion);
            let built = device_packet(
                &state,
                &host,
                Fifo::ROOT,
                SESSION,
                StampSlot(0),
                &drained,
                reads(None),
            )
            .expect("a control packet on the root FIFO");
            state.admit_packet(&built).expect("the root domain is open")
        };
        let first = admit(11);
        let second = admit(22);

        let out_of_order = state
            .complete_transaction(second.epoch, second.admitted.transaction.identity.ingress)
            .expect("live incarnation");
        assert!(
            out_of_order.is_empty(),
            "the second position's word is behind the first's: {out_of_order:?}"
        );

        let in_order = state
            .complete_transaction(first.epoch, first.admitted.transaction.identity.ingress)
            .expect("live incarnation");
        assert_eq!(
            in_order
                .iter()
                .filter_map(|r| r.stamp)
                .map(|s| s.value.0)
                .collect::<Vec<_>>(),
            vec![11, 22],
            "and both arrive, in the channel's order"
        );
    }
}
