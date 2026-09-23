//! The EXEC transaction: everything one accepted command-stream packet means,
//! frozen.
//!
//! # Built once, then read-only
//!
//! The value of an immutable transaction is not tidiness. It is that a
//! dependency compiler, a scheduler and an executor all read the same thing and
//! none of them can be the reason it changed — so "what does this packet touch"
//! has one answer for the whole life of the work, and a late discovery is a
//! type error rather than a race.
//!
//! [`ExecBuilder`] is the only way to make one, it consumes itself to produce
//! the transaction, and the transaction has no mutating method. A resolver that
//! wanted to add an access after the fact would have to rebuild.
//!
//! # Records point into arenas, and the arenas are the transaction's
//!
//! A bind record carries a counted array; a pass descriptor is 592 bytes; a
//! resource barrier names a list. Storing those inline would make every record
//! the size of the largest one, in a `Vec` that is walked per draw. So the
//! variable-length parts live in per-kind arenas on the transaction and the
//! operations name windows of them — which also means the whole packet is three
//! or four allocations rather than one per record.
//!
//! # The vocabulary here is the vocabulary `operation` counts
//!
//! [`ResolvedOperation`] has one variant per non-empty operation class.
//! `InfoQuery` and `CompletionEffect` have no variant because they have no
//! judged operations, and that is not an omission this module has to be trusted
//! about: `operation::tests::every_class_has_a_payload_or_a_reason_to_be_empty`
//! fails the moment either count moves off zero.

use crate::access::{
    AccessIntent, AccessKey, AccessSource, BackingId, ContentVersion, Participation,
};
use crate::bind::{BufferBinding, ObjectBinding};
use crate::blit::BlitOp;
use crate::compute::{ComputeExtent, ComputeOp};
use crate::encoder::{ComputeEncoderState, RenderEncoderState};
use crate::icb::IcbOp;
use crate::identity::{
    ChannelId, ChannelSequence, IngressOrdinal, ResourceId, SessionGeneration, TransactionIdentity,
};
use crate::operation::OperationClass;
use crate::pass::PassDescriptor;
use crate::render::{RenderOp, ScissorRect, Viewport};
use crate::resource_state::ResourceStateOp;
use crate::stream::{
    SegmentBegin, SegmentEnd, SegmentKind, SegmentLifetime, SegmentOpening, StreamCursor,
    StreamPosition, StreamRefusal,
};
use crate::sync::{BarrierOp, EventOp, FenceOp};

/// One resolved operation, in the class the vocabulary put it in.
///
/// Eight variants for eleven [`OperationClass`]es, and the three missing ones
/// are missing because they are not records inside an encoder. A boundary *is*
/// the encoder — [`ExecBuilder::begin_encoder`] and [`ExecBuilder::end_segment`]
/// are how one enters a transaction, and [`ResolvedStream::begin`] is where its
/// opening is kept — so a boundary payload here would be a second
/// representation of the same fact, one that could be recorded at a position
/// *inside* the encoder it opens. An info query answers into a reply buffer
/// rather than an encoder, and the completion class has no record at all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResolvedOperation {
    Render(RenderOp),
    Compute(ComputeOp),
    Blit(BlitOp),
    Event(EventOp),
    Fence(FenceOp),
    Barrier(BarrierOp),
    ResourceState(ResourceStateOp),
    IndirectCommand(IcbOp),
}

impl ResolvedOperation {
    /// The vocabulary class this operation belongs to.
    ///
    /// An exhaustive match, and that is the point: rail and segment
    /// admissibility are both answered from the class, so a variant added
    /// without a class here does not compile rather than quietly inheriting
    /// whatever the last arm said.
    #[must_use]
    pub const fn class(&self) -> OperationClass {
        match self {
            Self::Render(_) => OperationClass::Render,
            Self::Compute(_) => OperationClass::Compute,
            Self::Blit(_) => OperationClass::Blit,
            Self::Event(_) => OperationClass::Event,
            Self::Fence(_) => OperationClass::Fence,
            Self::Barrier(_) => OperationClass::Barrier,
            Self::ResourceState(_) => OperationClass::ResourceState,
            Self::IndirectCommand(_) => OperationClass::IndirectCommand,
        }
    }

    /// The pipeline this record binds, if it binds one.
    ///
    /// **The link between "a record names a pipeline" and "a transaction is
    /// held for its compilation".** [`crate::session::SessionModel::admit`]
    /// checks every `pipeline_wait` against this transaction's leases, and
    /// [`ExecWork::pipeline_leases`] was filled by nothing but a test — so a
    /// transaction built from a guest's stream leased nothing, every wait
    /// naming a pipeline it plainly uses read as `UnleasedPipelineWait`, and a
    /// caller that dropped the waits instead would run a draw against a
    /// pipeline still being compiled.
    ///
    /// Exhaustive over the classes for the same reason [`Self::class`] is: a
    /// tenth operation that binds a pipeline must not inherit "no" from
    /// whichever arm was last. The two that bind one are the render and compute
    /// pipeline-state records; an indirect command buffer's slots name
    /// pipelines too, but not at the record that executes it, so the lease for
    /// those belongs to whoever built the buffer.
    #[must_use]
    pub const fn pipeline_lease(&self) -> Option<ResourceId> {
        match self {
            Self::Render(RenderOp::SetPipeline { pipeline })
            | Self::Compute(ComputeOp::SetPipeline { pipeline }) => Some(*pipeline),
            Self::Render(_)
            | Self::Compute(_)
            | Self::Blit(_)
            | Self::Event(_)
            | Self::Fence(_)
            | Self::Barrier(_)
            | Self::ResourceState(_)
            | Self::IndirectCommand(_) => None,
        }
    }

    /// Every participation this record declares by itself, appended to `out`.
    ///
    /// **The link between "a record resolves" and "a transaction has
    /// accesses".** Each payload module states what its own records name — a
    /// draw's index buffer, a copy's two ends, a synchronise's content, and the
    /// classes that name nothing say so — and this is the one place those
    /// answers are collected. Before it, every one of those methods was
    /// reachable only from its own tests, and no caller could ask a resolved
    /// stream what it touched without knowing the whole vocabulary itself.
    ///
    /// Exhaustive, like [`Self::class`], and for the same reason: a variant
    /// added without an arm here would silently contribute no accesses, and an
    /// operation missing from the access list is a hazard edge that does not
    /// get built — a race, not a slowdown.
    ///
    /// `arenas` is the transaction's own, and only one arm reads it: a render
    /// `WriteDescriptor` carries a [`crate::render::PassDescriptorSlot`],
    /// because the descriptor is 592 bytes and a record that carried it by
    /// value would make every eight-byte record that size. Its participations
    /// are the pass's cost *before any draw* — which is what makes a pass with
    /// no draws still a write — and this is the only scope that can reach it.
    /// A slot past the end of the arena contributes nothing rather than
    /// panicking: the arena and the slot are built together by
    /// [`ExecBuilder`], so a mismatch is a bug in this crate, and taking a
    /// stream down over it would lose the whole packet.
    ///
    /// Appended rather than returned so a caller walking a whole EXEC keeps one
    /// buffer. A `Vec` per record would be an allocation per record on the
    /// hottest path this crate has.
    pub fn participations(&self, arenas: &ExecArenas, out: &mut Vec<Participation>) {
        match self {
            Self::Render(op) => {
                out.extend(op.participations());
                if let RenderOp::WriteDescriptor { descriptor } = op {
                    if let Some(pass) = arenas.pass_descriptors.get(descriptor.0 as usize) {
                        pass.extend_participations(out);
                    }
                }
            }
            Self::Compute(op) => out.extend(op.participations()),
            Self::Blit(op) => out.extend(op.participations()),
            Self::Event(op) => out.extend(op.participations()),
            Self::Fence(op) => out.extend(op.participations()),
            Self::Barrier(op) => out.extend(op.participations()),
            Self::ResourceState(op) => out.extend(op.participations()),
            Self::IndirectCommand(op) => out.extend(op.participations()),
        }
    }
}

/// One record, and where it sits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamRecord {
    pub at: StreamPosition,
    pub op: ResolvedOperation,
}

/// One encoder's worth of resolved records.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedStream {
    pub begin: SegmentBegin,
    pub records: Vec<StreamRecord>,
}

impl ResolvedStream {
    #[must_use]
    pub const fn kind(&self) -> SegmentKind {
        self.begin.kind
    }
}

/// A content version this transaction makes current, and exactly the memory it
/// covers.
///
/// **Derived from the accesses, never stated beside them.** A version claim
/// *is* a write access's claim to produce the next content of the memory it
/// names, so the two cannot be separate lists without being able to disagree
/// about the region — and that disagreement is not hypothetical. A reservation
/// that named a whole backing while its access named a range let two writers of
/// disjoint ranges both claim to produce one backing's next version, with
/// nothing ordering them and no legal answer to which version the backing ended
/// at. Region coverage is the access's own key, and the same `may_alias` that
/// decides hazards decides whether two claims collide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VersionPublication {
    pub backing: BackingId,
    /// Exactly the memory the producing access named.
    pub region: AccessKey,
    pub to: ContentVersion,
}

/// The content versions an access list makes current.
///
/// One per access that declared an output version and names memory. A heap
/// declaration or a domain-only access produces none: neither names bytes, so
/// neither can claim to have produced any.
///
/// A function of the accesses and nothing else, which is why it is free rather
/// than a method on [`ExecWork`]: a lifecycle synchronise and a query reply are
/// writes a guest reads back, and the reference interpreter has to publish
/// their versions by the same rule it publishes a draw's.
pub fn published_versions(
    accesses: &[AccessIntent],
) -> impl Iterator<Item = VersionPublication> + '_ {
    accesses.iter().filter_map(|access| {
        let to = access.output_content_version?;
        let backing = match access.key {
            AccessKey::Range(r, _) | AccessKey::Subresource(r, _) | AccessKey::Whole(r) => {
                r.backing
            }
            AccessKey::Heap(_) | AccessKey::DomainOnly => return None,
        };
        Some(VersionPublication {
            backing,
            region: access.key,
            to,
        })
    })
}

/// Something this transaction's *records* must wait for that is not a hazard
/// edge.
///
/// Hazard edges are compiled from accesses and always point backwards in
/// ingress order. These do not: a guest may wait for an event value nothing has
/// produced yet, and that is ordinary rather than an error. The two are
/// separate types because they are separate questions, and [`crate::ready`]
/// tracks them apart for exactly that reason.
///
/// **Completion-stamp waits are not here.** They are the packet's, decoded from
/// its envelope at ingress before any side effect, and a control packet that
/// carries no records carries them too — see
/// [`crate::transaction::DeviceTransaction::stamp_waits`]. A `Stamp` arm here
/// would be that same wait stated a second time, on the one class of packet
/// that also has an envelope to state it in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prerequisite {
    /// An event value.
    Event { event: ResourceId, value: u64 },
    /// A fence another encoder updates.
    Fence { fence: ResourceId },
}

/// The variable-length parts of a packet's records.
///
/// One arena per entry shape, because the shapes have different sizes and a
/// single arena of a union would make the smallest entry as large as the
/// largest.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExecArenas {
    pub buffer_bindings: Vec<BufferBinding>,
    pub object_bindings: Vec<ObjectBinding>,
    /// Resources named by a barrier's counted list.
    pub resources: Vec<ResourceId>,
    pub pass_descriptors: Vec<PassDescriptor>,
    pub viewports: Vec<Viewport>,
    pub scissors: Vec<ScissorRect>,
}

/// One EXEC packet's resolved contents, frozen, and not yet anywhere in
/// particular.
///
/// **Deliberately carries no identity.** Resolving a packet's records is a
/// function of the packet's bytes and the namespaces they name; it does not
/// observe arrival, so it cannot know the packet's channel sequence or ingress
/// ordinal, and being unable to *say* is what keeps a resolver from stating one
/// that disagrees with the envelope's. [`Self::stamp`] pairs this with the
/// [`TransactionIdentity`] that admission assigned, and is the only way to get
/// an [`ExecTransaction`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExecWork {
    pub streams: Vec<ResolvedStream>,
    /// Everything this transaction touches, at the precision the records
    /// supplied.
    pub accesses: Vec<AccessIntent>,
    pub pipeline_leases: Vec<ResourceId>,
    pub prerequisites: Vec<Prerequisite>,
    pub arenas: ExecArenas,
    /// Why this transaction's bound slots stayed `Unknown`. See
    /// [`UsageCensus`]; carried on the work rather than reported from the walk
    /// so that the numbers describe exactly the walk whose accesses are beside
    /// them.
    pub usage_census: UsageCensus,
}

impl ExecWork {
    /// Every record, in execution order.
    pub fn records(&self) -> impl Iterator<Item = &StreamRecord> {
        self.streams.iter().flat_map(|s| s.records.iter())
    }

    /// The content versions this work makes current.
    pub fn published_versions(&self) -> impl Iterator<Item = VersionPublication> + '_ {
        published_versions(&self.accesses)
    }

    /// Pair this work with where the packet carrying it arrived.
    ///
    /// Borrows rather than consuming, because the work belongs to the
    /// [`crate::transaction::DeviceTransaction`] that carries it and an
    /// executor that owned a second copy would be a second answer to "what does
    /// this packet touch".
    #[must_use]
    pub const fn stamp(&self, identity: TransactionIdentity) -> ExecTransaction<'_> {
        ExecTransaction {
            identity,
            work: self,
        }
    }

    /// The pipelines this transaction's **render** records bind.
    ///
    /// Derived from the records rather than kept beside
    /// [`Self::pipeline_leases`], which is deliberately class-blind: the
    /// ordering plane holds a compute pipeline for exactly the reasons it holds
    /// a render one, so splitting *that* list would give the model two lists to
    /// wait on. A backend asking "which render pipelines must I have translated"
    /// is asking a different question, and this answers it from the same single
    /// source — a second stored list could disagree with the records, and this
    /// cannot.
    ///
    /// Deduplicated, like [`Self::pipeline_leases`]: a stream that binds one
    /// pipeline for forty draws names it once.
    pub fn render_pipeline_leases(&self) -> Vec<ResourceId> {
        let mut out: Vec<ResourceId> = Vec::new();
        for record in self.records() {
            if let ResolvedOperation::Render(crate::render::RenderOp::SetPipeline { pipeline }) =
                record.op
            {
                if !out.contains(&pipeline) {
                    out.push(pipeline);
                }
            }
        }
        out
    }

    /// The **compute** pipelines this transaction dispatches, each with the
    /// threadgroup size the dispatch stated.
    ///
    /// The compute half of [`Self::render_pipeline_leases`], and it cannot be
    /// the same shape: a kernel's translation is keyed by its threadgroup size
    /// as well as its ref, so the pre-scan needs a pair and a bare lease list
    /// does not spell one. That extra field is why this reads the dispatches
    /// rather than the leases --- the size is the *dispatch's* statement, and
    /// one pipeline dispatched at two sizes is two translations.
    ///
    /// Three things the byte scan this replaces could not say, each of which
    /// leaves a lease readied on an answer that never examined it:
    ///
    /// - A dispatch whose selector carries **no** threadgroup size
    ///   ([`crate::compute::DispatchOp::ThreadsIndirect`]) contributes nothing,
    ///   because there is nothing for it to contribute. The scan read a zeroed
    ///   field and rejected it as an out-of-range extent, which is the same
    ///   word it uses for "the guest asked for a grid of nothing".
    /// - The bound pipeline **survives an encoder continuation**. A stream is
    ///   one encoder however many segments carry it, so a dispatch in the
    ///   continuation runs under the pipeline the first segment bound; a scan
    ///   that resets at each framed segment loses exactly that one.
    /// - A pipeline **bound and never dispatched** is not here, and it is in
    ///   [`Self::pipeline_leases`]. It compiles nothing because nothing runs
    ///   it, and the ordering plane holds it regardless.
    ///
    /// Deduplicated on the pair: a loop dispatching one kernel at one size
    /// forty times names it once.
    pub fn compute_dispatch_translations(&self) -> Vec<(ResourceId, ComputeExtent)> {
        use crate::compute::DispatchOp;

        let mut out: Vec<(ResourceId, ComputeExtent)> = Vec::new();
        for stream in &self.streams {
            // Per stream, because a stream is one encoder: the bound pipeline
            // is encoder state and does not end where a segment does.
            let mut bound: Option<ResourceId> = None;
            for record in &stream.records {
                let ResolvedOperation::Compute(op) = record.op else {
                    continue;
                };
                let threads_per_group = match op {
                    ComputeOp::SetPipeline { pipeline } => {
                        bound = Some(pipeline);
                        continue;
                    }
                    ComputeOp::Dispatch(
                        DispatchOp::Threadgroups {
                            threads_per_group, ..
                        }
                        | DispatchOp::Threads {
                            threads_per_group, ..
                        }
                        | DispatchOp::ThreadgroupsIndirect {
                            threads_per_group, ..
                        },
                    ) => threads_per_group,
                    // The selector has no such field. Not a zero, and not a
                    // size this could guess.
                    ComputeOp::Dispatch(DispatchOp::ThreadsIndirect { .. }) => continue,
                    _ => continue,
                };
                let Some(pipeline) = bound else { continue };
                let item = (pipeline, threads_per_group);
                if !out.contains(&item) {
                    out.push(item);
                }
            }
        }
        out
    }

    /// How many records this packet carries.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.streams.iter().map(|s| s.records.len()).sum()
    }

    /// Whether this transaction writes anything at all.
    ///
    /// A transaction that writes nothing cannot be the producer half of a
    /// hazard, which is worth one question rather than a scan per candidate
    /// edge.
    #[must_use]
    pub fn writes_anything(&self) -> bool {
        self.accesses.iter().any(|a| a.mode.writes())
    }
}

/// One accepted EXEC packet as an executor sees it: where the packet sits, and
/// a borrow of what it resolved to.
///
/// **A view, not a container.** The work is the envelope's — see
/// [`crate::transaction::Payload::Exec`] — and the identity is the envelope's
/// too. Owning either would make an admitted EXEC representable twice, and two
/// representations of one packet are two answers to "what does it touch". This
/// is the derivation the plan writes as
/// `submission_domain = DeviceTransaction.channel`, spelled as a borrow so
/// there is nothing to keep in step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecTransaction<'a> {
    /// Assigned by admission, which is the only service that observes arrival.
    pub identity: TransactionIdentity,
    pub work: &'a ExecWork,
}

impl ExecTransaction<'_> {
    /// The semantic lifetime this was accepted in.
    #[must_use]
    pub const fn session(&self) -> SessionGeneration {
        self.identity.session
    }

    /// The submission ordering domain.
    #[must_use]
    pub const fn domain(&self) -> ChannelId {
        self.identity.domain
    }

    /// Position within that domain.
    #[must_use]
    pub const fn domain_sequence(&self) -> ChannelSequence {
        self.identity.domain_sequence
    }

    /// Position in the device's single arrival order.
    #[must_use]
    pub const fn ingress(&self) -> IngressOrdinal {
        self.identity.ingress
    }

    /// Everything this transaction touches.
    #[must_use]
    pub fn accesses(&self) -> &[AccessIntent] {
        &self.work.accesses
    }

    /// Waits that are not hazard edges.
    #[must_use]
    pub fn prerequisites(&self) -> &[Prerequisite] {
        &self.work.prerequisites
    }
    /// Every record, in execution order.
    pub fn records(&self) -> impl Iterator<Item = &StreamRecord> {
        self.work.records()
    }

    /// The content versions this transaction makes current.
    pub fn published_versions(&self) -> impl Iterator<Item = VersionPublication> + '_ {
        self.work.published_versions()
    }

    /// How many records this packet carries.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.work.record_count()
    }

    /// Whether this transaction writes anything at all.
    #[must_use]
    pub fn writes_anything(&self) -> bool {
        self.work.writes_anything()
    }
}

/// Builds one [`ExecTransaction`], and refuses the shapes that cannot execute.
///
/// It owns a [`StreamCursor`], so the segment/encoder rules are enforced here
/// rather than restated: a record with no open encoder, a rail that disagrees
/// with its segment, an encoder that never ended — each is the cursor's refusal
/// and reaches the caller unchanged.
#[derive(Debug)]
pub struct ExecBuilder<'u> {
    cursor: StreamCursor,
    /// What an executor has published about the pipelines this packet binds.
    ///
    /// A field of the builder rather than an argument to [`Self::record`],
    /// because it is used for exactly one thing — narrowing the footprint the
    /// builder's own encoder tables answer — and a per-record argument would
    /// let two records of one packet be resolved against two different answers.
    /// It is the same shape as the tables themselves: state of the walk, not of
    /// the record.
    ///
    /// [`Self::new`] supplies [`crate::pipeline::NoUsage`], which is the
    /// conservative answer and not an absent one; see that type.
    usages: &'u dyn crate::pipeline::UsageSource,
    streams: Vec<ResolvedStream>,
    open: Option<ResolvedStream>,
    accesses: Vec<AccessIntent>,
    pipeline_leases: Vec<ResourceId>,
    prerequisites: Vec<Prerequisite>,
    arenas: ExecArenas,
    /// The open encoder's binding tables. See [`EncoderBindings`].
    bindings: EncoderBindings,
    /// Scratch [`Self::record`] gathers each operation's participations into,
    /// so the walk costs no allocation after the first record.
    participation_scratch: Vec<Participation>,
    /// Why this walk's slots stayed `Unknown`. See [`UsageCensus`].
    usage_census: UsageCensus,
}

/// The binding state of whichever encoder is open.
///
/// **What makes a draw's footprint the draw's, and not just its index
/// buffer's.** A bind record touches no memory and a draw's own fields name
/// only what they carry, so the memory a draw reads through its bound slots is
/// a fact of the *encoder* — accumulated across the records before it and read
/// back at the draw. Without this the tables in [`crate::encoder`] had no
/// writer and no reader, and a transaction that wrote a buffer and then drew
/// with it bound compiled no edge between the two.
///
/// One encoder's worth, not one per stream: the cursor keeps exactly one
/// encoder open, a continuation keeps writing into it, and a new encoder starts
/// with everything unbound — which is Metal's rule and not a simplification.
/// The blit and event encoders bind nothing, so they hold no tables rather than
/// empty ones.
#[derive(Debug)]
enum EncoderBindings {
    Render(Box<RenderEncoderState>),
    Compute(Box<ComputeEncoderState>),
    /// An encoder with no binding tables, or no encoder at all.
    None,
}

impl EncoderBindings {
    /// Start a freshly opened encoder of this kind with nothing bound.
    ///
    /// Reuses the tables in place where the kind is the one already held,
    /// which is the common case — a packet is many encoders of a few kinds —
    /// so the argument-table capacity is reserved once per builder rather than
    /// once per encoder.
    fn open(&mut self, kind: SegmentKind) {
        match (kind, &mut *self) {
            (SegmentKind::Render, Self::Render(state)) => state.clear(),
            (SegmentKind::Compute, Self::Compute(state)) => state.clear(),
            (SegmentKind::Render, _) => *self = Self::Render(Box::default()),
            (SegmentKind::Compute, _) => *self = Self::Compute(Box::default()),
            (SegmentKind::Blit | SegmentKind::Event | SegmentKind::Info, _) => *self = Self::None,
        }
    }

    /// What this record reads through the bound slots, appended to `out`.
    ///
    /// Only a draw and a dispatch read the tables; every other record either
    /// writes a slot or carries state.
    ///
    /// Every bound slot participates as [`crate::access::AccessMode::Unknown`]
    /// until a pipeline publishes what its shader does with each one. That is
    /// [`crate::pipeline::BindingUsage`], and the layer that can produce it is
    /// the executor that compiled the shader — so the narrowing arrives with
    /// the pipeline, not from here.
    ///
    /// **Changes nothing.** Taking a footprint is not keeping it — see
    /// [`Self::keep`].
    fn footprint_into(
        &self,
        op: &ResolvedOperation,
        usages: &dyn crate::pipeline::UsageSource,
        out: &mut Vec<Participation>,
        census: &mut UsageCensus,
    ) {
        use reims_vgpu_protocol::render::ShaderStage;
        match (self, op) {
            (Self::Render(state), ResolvedOperation::Render(RenderOp::Draw(_))) => {
                // The bound pipeline is the encoder's own state, so the
                // question asked is about the pipeline this draw will actually
                // use. A draw with nothing bound asks nothing and every slot
                // stays `Unknown`, which is what a draw the device cannot name
                // a shader for deserves.
                let bound = state.pipeline;
                let vertex =
                    bound.and_then(|id| usages.binding_usage(id, Some(ShaderStage::Vertex)));
                let fragment =
                    bound.and_then(|id| usages.binding_usage(id, Some(ShaderStage::Fragment)));
                census.note_draw(bound, vertex.is_some(), fragment.is_some());
                let before = out.len();
                let had_a_table = vertex.is_some() || fragment.is_some();
                state.footprint_into(vertex, fragment, out);
                census.note_unknowns(&out[before..], had_a_table);
            }
            (Self::Compute(state), ResolvedOperation::Compute(ComputeOp::Dispatch(_))) => {
                let bound = state.pipeline;
                let kernel = bound.and_then(|id| usages.binding_usage(id, None));
                census.note_dispatch(bound, kernel.is_some());
                let before = out.len();
                let had_a_table = kernel.is_some();
                state.footprint_into(kernel, out);
                census.note_unknowns(&out[before..], had_a_table);
            }
            _ => {}
        }
    }

    /// Take a record into the encoder's tables: what its footprint answered is
    /// now declared, and what it binds is now bound.
    ///
    /// **The only door out of [`ExecBuilder::record`]'s success path, and the
    /// reason it is one method.** [`Self::footprint_into`] runs before the
    /// record is placed, because the accesses have to be built before anything
    /// can refuse them — so a record that is then refused has had its footprint
    /// taken and must not have had it *declared*. Marking inside the gather is
    /// how the declaration used to be lost: the refused record took it, and the
    /// next draw named nothing, which is a hazard edge that does not get built.
    ///
    /// Declaring before applying is the order, and not an arbitrary one: the
    /// footprint that was answered is the one the tables held when it was
    /// gathered, and a bind in the same record would move slots out from under
    /// it. No record is both, so the two never actually meet — which is why the
    /// order can be stated rather than defended.
    fn keep(&mut self, op: &ResolvedOperation, arenas: &ExecArenas) {
        match (&mut *self, op) {
            (Self::Render(state), ResolvedOperation::Render(RenderOp::Draw(_))) => state.declare(),
            (Self::Compute(state), ResolvedOperation::Compute(ComputeOp::Dispatch(_))) => {
                state.declare();
            }
            _ => {}
        }
        self.apply(op, arenas);
    }

    /// Apply one accepted record to the open encoder's tables.
    ///
    /// A record the cursor refused never reaches here, so a refusal leaves no
    /// binding behind claiming it ran — the same rule [`ExecBuilder::record`]
    /// applies to the accesses.
    ///
    /// A record whose rail disagrees with the tables it stands in is not
    /// applied. That pairing is the cursor's refusal to make, and by the time
    /// this runs it has already made it.
    fn apply(&mut self, op: &ResolvedOperation, arenas: &ExecArenas) {
        match (self, op) {
            (Self::Render(state), ResolvedOperation::Render(op)) => {
                state.apply(op, &arenas.buffer_bindings, &arenas.object_bindings);
            }
            (Self::Compute(state), ResolvedOperation::Compute(op)) => {
                state.apply(op, &arenas.buffer_bindings, &arenas.object_bindings);
            }
            _ => {}
        }
    }
}

impl<'u> Default for ExecBuilder<'u> {
    fn default() -> Self {
        Self::new()
    }
}

/// The answer [`ExecBuilder::new`] uses: nothing published, every bound slot
/// `Unknown`. A `static` because the builder holds a reference and this one has
/// no state to be per-builder.
static NO_USAGE: crate::pipeline::NoUsage = crate::pipeline::NoUsage;

impl<'u> ExecBuilder<'u> {
    /// A builder for a packet whose contents are not yet known and whose
    /// position in the arrival order is not this layer's to know. See
    /// [`ExecWork`].
    ///
    /// Nothing is published to it, so every bound slot of every draw and
    /// dispatch participates as [`crate::access::AccessMode::Unknown`]. A
    /// caller that can do better uses [`ExecBuilder::with_usage`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            cursor: StreamCursor::new(),
            usages: &NO_USAGE,
            streams: Vec::new(),
            open: None,
            accesses: Vec::new(),
            pipeline_leases: Vec::new(),
            prerequisites: Vec::new(),
            arenas: ExecArenas::default(),
            bindings: EncoderBindings::None,
            participation_scratch: Vec::new(),
            usage_census: UsageCensus::default(),
        }
    }

    /// The same builder, resolving its footprints against what an executor has
    /// published.
    ///
    /// **The narrowing arrives here and nowhere else.** Set once for the whole
    /// packet: a packet's records are one transaction and one access list, so
    /// two records resolved against two different answers would be one
    /// transaction describing two different pipelines' shaders.
    #[must_use]
    pub fn with_usage(usages: &'u dyn crate::pipeline::UsageSource) -> Self {
        Self {
            usages,
            ..Self::new()
        }
    }

    /// The arenas, so a resolver can file variable-length entries and name the
    /// window it filed them at.
    pub fn arenas_mut(&mut self) -> &mut ExecArenas {
        &mut self.arenas
    }

    /// A protection envelope armed the next segment.
    pub fn protection_envelope(
        &mut self,
        options: crate::stream::ProtectionOptions,
    ) -> Result<(), StreamRefusal> {
        self.cursor.protection_envelope(options)
    }

    /// Open an encoder, named by the type byte in its segment header.
    pub fn begin_segment(
        &mut self,
        wire_type: u8,
        lifetime: SegmentLifetime,
    ) -> Result<(), StreamRefusal> {
        let opening = self.cursor.begin(wire_type, lifetime)?;
        self.opened(opening);
        Ok(())
    }

    /// Open an encoder whose kind is already established.
    ///
    /// A stream walker cannot cut a segment's window without knowing which
    /// family the segment is, so by the time it gets here the type byte has
    /// been parsed. Handing back the byte for this to parse again is the
    /// "resolve twice" shape the replacement exists to remove.
    pub fn begin_encoder(
        &mut self,
        kind: SegmentKind,
        lifetime: SegmentLifetime,
    ) -> Result<(), StreamRefusal> {
        let opening = self.cursor.begin_kind(kind, lifetime)?;
        self.opened(opening);
        Ok(())
    }

    /// A segment opened. Only a *new* encoder makes a new [`ResolvedStream`];
    /// a continuation keeps writing into the one already open, which is what
    /// makes an encoder spanning segments one encoder here rather than two.
    fn opened(&mut self, opening: SegmentOpening) {
        if let SegmentOpening::Opened(_, begin) = opening {
            // A new encoder starts with everything unbound; a continuation
            // keeps the tables it has been filling.
            self.bindings.open(begin.kind);
            self.open = Some(ResolvedStream {
                begin,
                records: Vec::new(),
            });
        }
    }

    /// Record one resolved operation inside the open encoder, and declare
    /// every access it names.
    ///
    /// The rail is taken from the operation's own class rather than passed
    /// alongside it, so a caller cannot hand a compute payload in under a
    /// render rail and have it accepted by the segment it is standing in.
    ///
    /// The accesses are derived here rather than declared beside the record,
    /// and that is the point. A transaction whose `accesses` disagree with its
    /// `streams` is representable the moment the two are supplied separately —
    /// and the disagreement is silent, because nothing downstream reads the
    /// records to check. Here the record *is* the declaration: the operation
    /// states its own participation, `source` places it, and neither can be
    /// supplied without the other.
    ///
    /// # Errors
    ///
    /// As before, plus [`StreamRefusal::Access`] where a participation cannot
    /// be placed. The whole transaction refuses; see that variant.
    pub fn record(
        &mut self,
        op: ResolvedOperation,
        source: &mut (impl AccessSource + ?Sized),
    ) -> Result<StreamPosition, StreamRefusal> {
        // Taken out and put back, so the walk costs no allocation after the
        // first record of the first transaction. It also releases the borrow
        // of `self` that `participations` takes on the arenas.
        let mut parts = core::mem::take(&mut self.participation_scratch);
        parts.clear();
        op.participations(&self.arenas, &mut parts);
        // What the record reads through the encoder's bound slots. Derived
        // here, beside the record's own participations, for the same reason
        // those are derived rather than declared: a footprint supplied
        // separately can disagree with the records that produced it.
        //
        // Gathering only. Whether the encoder has *declared* what it just
        // answered is settled on the success path below, because everything
        // between here and there can still refuse the record.
        self.bindings
            .footprint_into(&op, self.usages, &mut parts, &mut self.usage_census);
        // Pushed straight onto the transaction's list and rolled back on any
        // refusal, rather than gathered into a second vector: a record's
        // accesses are at most a handful and a `Vec` per record would be an
        // allocation per record on the hottest path this crate has.
        let before = self.accesses.len();
        let mut refused = None;
        for participation in &parts {
            match source.access(participation) {
                Ok(access) => self.accesses.push(access),
                Err(refusal) => {
                    refused = Some(StreamRefusal::Access(refusal));
                    break;
                }
            }
        }
        self.participation_scratch = parts;
        // Derived from the record, exactly as the accesses above are, and after
        // placement so a record the cursor refuses leaves no lease behind
        // claiming it ran.
        let lease = op.pipeline_lease();
        match refused.map_or_else(|| self.place(op), Err) {
            Ok(at) => {
                if let Some(pipeline) = lease {
                    self.lease_pipeline(pipeline);
                }
                self.bindings.keep(&op, &self.arenas);
                Ok(at)
            }
            Err(refusal) => {
                // Nothing half-applied: a record the cursor would not take
                // leaves no accesses behind claiming it ran.
                self.accesses.truncate(before);
                Err(refusal)
            }
        }
    }

    /// The rail and cursor half of [`Self::record`].
    fn place(&mut self, op: ResolvedOperation) -> Result<StreamPosition, StreamRefusal> {
        let rail = match rail_of(&op) {
            Some(rail) => rail,
            None => {
                // A record whose class exists on more than one rail is admitted
                // by whichever encoder is open — but only by an encoder that
                // carries the class at all, which is the check the single-rail
                // records get for free from their own rail.
                let Some(open) = self.cursor.open_encoder() else {
                    return Err(StreamRefusal::RecordOutsideEncoder);
                };
                if !admissible_on(&op, open) {
                    return Err(StreamRefusal::RailMismatch {
                        segment: open,
                        record: open.rail(),
                    });
                }
                open.rail()
            }
        };
        let at = self.cursor.record(rail)?;
        self.open
            .as_mut()
            .expect("the cursor accepted a record, so an encoder is open")
            .records
            .push(StreamRecord { at, op });
        Ok(at)
    }

    /// End the open segment, and the encoder inside it if it ends here.
    ///
    /// A held encoder's [`ResolvedStream`] is not filed: it is still being
    /// written to, and filing it would put a half-recorded encoder into the
    /// transaction with its remaining records landing nowhere.
    pub fn end_segment(&mut self) -> Result<SegmentEnd, StreamRefusal> {
        let end = self.cursor.end()?;
        if matches!(end, SegmentEnd::EncoderEnded { .. }) {
            self.streams
                .push(self.open.take().expect("an encoder was open"));
        }
        Ok(end)
    }

    /// State an access beside the records rather than deriving it from them.
    ///
    /// **Test-only, and the gate is the invariant.** A transaction whose
    /// `accesses` disagree with its `streams` is representable the moment the
    /// two can be supplied separately, and the disagreement is silent because
    /// nothing downstream reads the records to check — which is why
    /// [`Self::record`] derives them. What is left is a test that wants an
    /// access shape the registry would not produce, and the `cfg` is what says
    /// no production path may want the same thing.
    #[cfg(test)]
    pub(crate) fn declare_access(&mut self, access: AccessIntent) {
        self.accesses.push(access);
    }

    /// Hold this transaction for one pipeline's compilation.
    ///
    /// Private, and the privacy is the invariant: a lease list that could be
    /// added to beside [`Self::record`] is one that could name a pipeline no
    /// record binds, or omit one every draw uses, and nothing downstream reads
    /// the records to notice. The one door is the record, which is the same
    /// rule the accesses are under — see [`Self::declare_access`].
    ///
    /// Deduplicated, because the guest re-binds the same pipeline on every
    /// draw: without this the list is one entry per draw and the admission
    /// check one lookup per entry.
    fn lease_pipeline(&mut self, pipeline: ResourceId) {
        if !self.pipeline_leases.contains(&pipeline) {
            self.pipeline_leases.push(pipeline);
        }
    }

    pub fn require(&mut self, prerequisite: Prerequisite) {
        self.prerequisites.push(prerequisite);
    }

    /// Freeze the transaction.
    ///
    /// Consumes the builder, so the value that comes out cannot be the one that
    /// was still being written to.
    pub fn finish(mut self) -> Result<ExecWork, StreamRefusal> {
        // `finish` on the cursor is what refuses an encoder that never ended,
        // and an envelope that armed nothing.
        self.cursor.finish()?;
        debug_assert!(self.open.is_none(), "the cursor would have refused");
        self.streams.shrink_to_fit();
        self.accesses.shrink_to_fit();
        Ok(ExecWork {
            streams: core::mem::take(&mut self.streams),
            accesses: core::mem::take(&mut self.accesses),
            pipeline_leases: core::mem::take(&mut self.pipeline_leases),
            prerequisites: core::mem::take(&mut self.prerequisites),
            arenas: core::mem::take(&mut self.arenas),
            usage_census: self.usage_census,
        })
    }
}

/// Why a bound slot stayed [`crate::access::AccessMode::Unknown`], counted as
/// the walk decides it.
///
/// # The reading this exists to make possible
///
/// `Unknown` conflicts with everything, so it is the number the ordering plane
/// is judged on — and a total tells you nothing about what to fix. A draw whose
/// pipeline the walk never saw bound and a draw whose shader the executor could
/// not reflect both contribute every slot they bind, and they have nothing in
/// common: the first is a hole in the walk's own encoder state, the second is a
/// contract term the rail has not recovered. Counting them apart is what says
/// which one the next move is about.
///
/// Counted at the one place the answer is decided, so it cannot describe a
/// different walk than the one that ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UsageCensus {
    /// Draws whose encoder had no pipeline bound at all. Every slot they bind
    /// is `Unknown` and no executor could have helped.
    pub draws_without_a_pipeline: u32,
    /// Draws whose pipeline was bound and published neither stage.
    pub draws_with_no_stage_published: u32,
    /// Draws whose pipeline published one stage and refused the other.
    pub draws_with_one_stage_published: u32,
    /// Draws answered by both stages, which is the whole point.
    pub draws_with_both_stages_published: u32,
    /// Dispatches whose encoder had no pipeline bound.
    pub dispatches_without_a_pipeline: u32,
    /// Dispatches whose kernel published nothing.
    pub dispatches_with_no_stage_published: u32,
    /// Dispatches answered by their kernel.
    pub dispatches_published: u32,
    /// `Unknown` participations a *published* table produced.
    ///
    /// The other half of the row, and the one a count of draws cannot give:
    /// one conceded slot in a pipeline that draws ten thousand times is ten
    /// thousand conflicting participations, and one wholly unpublished pipeline
    /// that draws twice is four. Which of those the residue is decides whether
    /// the next move is about the publisher or about the reflection it
    /// publishes, so they are weighted the way the ordering plane weighs them.
    pub unknown_participations_from_a_table: u32,
    /// `Unknown` participations emitted because there was no table to ask.
    pub unknown_participations_with_no_table: u32,
    /// The first pipeline a draw or dispatch found unanswered.
    ///
    /// A count says how much was lost and never which pipeline lost it, and the
    /// two candidate causes — a pipeline whose executor has not published yet,
    /// and one whose usage a withdrawal or retirement threw away — are told
    /// apart only by asking the table what state that pipeline is in. The name
    /// is what lets the device ask. First rather than last, and one rather than
    /// a list, because this is a diagnosis and not a population: the walk must
    /// not allocate for it.
    pub first_unanswered_pipeline: Option<ResourceId>,
}

impl UsageCensus {
    fn note_draw(&mut self, bound: Option<ResourceId>, vertex: bool, fragment: bool) {
        let stages = usize::from(vertex) + usize::from(fragment);
        let counter = match (bound, stages) {
            (None, _) => &mut self.draws_without_a_pipeline,
            (Some(_), 0) => &mut self.draws_with_no_stage_published,
            (Some(_), 1) => &mut self.draws_with_one_stage_published,
            (Some(_), _) => &mut self.draws_with_both_stages_published,
        };
        *counter = counter.saturating_add(1);
        self.note_unanswered(bound, stages == 0);
    }

    fn note_dispatch(&mut self, bound: Option<ResourceId>, kernel: bool) {
        let counter = match (bound, kernel) {
            (None, _) => &mut self.dispatches_without_a_pipeline,
            (Some(_), false) => &mut self.dispatches_with_no_stage_published,
            (Some(_), true) => &mut self.dispatches_published,
        };
        *counter = counter.saturating_add(1);
        self.note_unanswered(bound, !kernel);
    }

    /// Split the `Unknown`s a footprint just produced by whether a table
    /// answered them.
    fn note_unknowns(&mut self, produced: &[Participation], had_a_table: bool) {
        let unknown = produced
            .iter()
            .filter(|p| p.mode == crate::access::AccessMode::Unknown)
            .count();
        let Ok(unknown) = u32::try_from(unknown) else {
            return;
        };
        let counter = if had_a_table {
            &mut self.unknown_participations_from_a_table
        } else {
            &mut self.unknown_participations_with_no_table
        };
        *counter = counter.saturating_add(unknown);
    }

    /// Keep the first pipeline that answered nothing, and only that one.
    fn note_unanswered(&mut self, bound: Option<ResourceId>, nothing_published: bool) {
        if let (Some(id), true, None) = (bound, nothing_published, self.first_unanswered_pipeline) {
            self.first_unanswered_pipeline = Some(id);
        }
    }
}

#[cfg(test)]
mod usage_census_tests {
    use super::*;

    /// The four draw outcomes are counted apart, because they name four
    /// different next moves.
    #[test]
    fn the_reason_a_draws_slots_stayed_unknown_is_counted_and_not_summed() {
        let id = ResourceId {
            slot: crate::identity::ObjectListRef(9),
            generation: crate::identity::SlotGeneration::default(),
        };
        let mut c = UsageCensus::default();
        c.note_draw(None, false, false);
        c.note_draw(Some(id), false, false);
        c.note_draw(Some(id), true, false);
        c.note_draw(Some(id), false, true);
        c.note_draw(Some(id), true, true);
        assert_eq!(
            c,
            UsageCensus {
                draws_without_a_pipeline: 1,
                draws_with_no_stage_published: 1,
                draws_with_one_stage_published: 2,
                draws_with_both_stages_published: 1,
                first_unanswered_pipeline: Some(id),
                ..UsageCensus::default()
            },
            "one stage published is one row whichever stage it was: the pair \
             says how much of the draw was answered, and which half is the \
             rail's own refusal census"
        );
    }

    /// The `Unknown`s are split by whether a table produced them, and they are
    /// weighted the way the ordering plane weighs them.
    ///
    /// A count of *draws* cannot answer this row. One conceded slot in a
    /// pipeline that draws ten thousand times is ten thousand conflicting
    /// participations; one wholly unpublished pipeline that draws twice is
    /// four. Which of those the residue is decides whether the next move is
    /// about the publisher or about the reflection it publishes.
    #[test]
    fn an_unknown_is_charged_to_the_table_that_conceded_it_or_to_the_absence_of_one() {
        use crate::access::AccessMode;
        use crate::identity::{ObjectListRef, SlotGeneration};

        let part = |mode| Participation {
            resource: ResourceId {
                slot: ObjectListRef(1),
                generation: SlotGeneration::default(),
            },
            extent: crate::access::ParticipationExtent::Whole,
            mode,
            api_stages: 0,
        };
        let mut c = UsageCensus::default();
        c.note_unknowns(
            &[
                part(AccessMode::Unknown),
                part(AccessMode::Read),
                part(AccessMode::Unknown),
            ],
            true,
        );
        c.note_unknowns(&[part(AccessMode::Unknown)], false);
        assert_eq!(c.unknown_participations_from_a_table, 2);
        assert_eq!(c.unknown_participations_with_no_table, 1);
        assert_eq!(
            c.draws_without_a_pipeline, 0,
            "counting the participations is not counting the draws, which is \
             the whole reason both rows exist"
        );
    }

    /// A dispatch has one stage, so "published" and "no stage" are the whole
    /// answer, and a kernel can never be half-answered.
    #[test]
    fn a_dispatch_is_answered_or_not_and_never_half() {
        let id = ResourceId {
            slot: crate::identity::ObjectListRef(4),
            generation: crate::identity::SlotGeneration::default(),
        };
        let mut c = UsageCensus::default();
        c.note_dispatch(None, false);
        c.note_dispatch(Some(id), false);
        c.note_dispatch(Some(id), true);
        assert_eq!(
            c,
            UsageCensus {
                dispatches_without_a_pipeline: 1,
                dispatches_with_no_stage_published: 1,
                dispatches_published: 1,
                first_unanswered_pipeline: Some(id),
                ..UsageCensus::default()
            }
        );
        assert_eq!(
            c.draws_without_a_pipeline, 0,
            "a dispatch never lands in a draw's row; the two rails are read \
             against different denominators"
        );
    }
}

/// Which rail a resolved operation's records are read on, when its class
/// belongs to exactly one.
///
/// `None` for the four classes that exist on more than one encoder --- fence,
/// barrier, resource state and indirect command. Those are admitted by
/// whichever encoder is open, and [`admissible_on`] is what keeps that from
/// meaning "any encoder at all". The `None` arm also carries the three classes
/// with no [`ResolvedOperation`] variant at all, which reach neither this
/// function nor that one.
fn rail_of(op: &ResolvedOperation) -> Option<reims_vgpu_protocol::closure::Rail> {
    use reims_vgpu_protocol::closure::Rail;
    Some(match op.class() {
        OperationClass::Render => Rail::Render,
        OperationClass::Compute => Rail::Compute,
        OperationClass::Blit => Rail::Blit,
        OperationClass::Event => Rail::Event,
        // No payload today, so unreachable; answered rather than left to the
        // `None` arm because the wire has an [`SegmentKind::Info`] segment and
        // a record of this class would be inside it, not on whichever encoder
        // happened to be open. Falling through would have made a future info
        // payload multi-encoder by default.
        OperationClass::InfoQuery => Rail::Info,
        // The four multi-encoder classes, and the three with no payload: a
        // boundary is the segment rather than a record inside one, an info
        // query answers into a reply buffer, and the completion class has no
        // record at all. Only the first four can be reached.
        OperationClass::EncoderBoundary
        | OperationClass::Fence
        | OperationClass::Barrier
        | OperationClass::ResourceState
        | OperationClass::IndirectCommand
        | OperationClass::CompletionEffect => return None,
    })
}

/// Whether a multi-rail record may appear inside a `kind` segment.
///
/// The sets are narrower than "more than one" and the narrowness is the point:
/// a fence exists on the render and blit encoders and **not** on the compute
/// one, because the compute pair is unresolved; a barrier exists on render and
/// compute and not on blit. Admitting a class on every encoder that is not its
/// own would let a compute fence through the one door the ledger closed.
///
/// Hard-coded and then checked against the ledger, which is the same
/// arrangement the payload vocabularies use: the table is what runs, and the
/// test is what says the table still describes the contract.
fn admissible_on(op: &ResolvedOperation, kind: SegmentKind) -> bool {
    class_admissible_on(op.class(), kind)
}

/// Whether a class of record may appear inside a `kind` segment.
///
/// Keyed on the class rather than on the payload variant, because that is what
/// the ledger is keyed on. A payload added to an existing class inherits its
/// class's answer instead of needing one written for it, and a class added
/// without an answer does not compile.
///
/// The sets are narrower than "more than one" and the narrowness is the point:
/// a fence exists on the render and blit encoders and **not** on the compute
/// one, because the compute pair is unresolved; a barrier exists on render and
/// compute and not on blit; residency's rows are all unresolved, so the
/// resource-state class reaches only the encoders whose *content* records are
/// judged. Admitting a class on every encoder that is not its own would let a
/// compute fence through the one door the ledger closed.
///
/// Hard-coded and then checked against the ledger, which is the same
/// arrangement the payload vocabularies use: the table is what runs, and the
/// test is what says the table still describes the contract.
const fn class_admissible_on(class: OperationClass, kind: SegmentKind) -> bool {
    match class {
        OperationClass::Fence => matches!(kind, SegmentKind::Render | SegmentKind::Blit),
        OperationClass::Barrier => matches!(kind, SegmentKind::Render | SegmentKind::Compute),
        OperationClass::ResourceState => matches!(kind, SegmentKind::Blit | SegmentKind::Compute),
        OperationClass::IndirectCommand => matches!(kind, SegmentKind::Render | SegmentKind::Blit),
        // The single-rail classes never reach here, and the three classes with
        // no stream records at all reach nothing. A boundary is the segment
        // rather than a record inside one: which encoders it may open is
        // [`StreamCursor`]'s answer, not this table's.
        OperationClass::EncoderBoundary
        | OperationClass::Render
        | OperationClass::Compute
        | OperationClass::Blit
        | OperationClass::Event
        | OperationClass::InfoQuery
        | OperationClass::CompletionEffect => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessKey, AccessMode, ResourceKey, StubRegistry};
    use crate::bind::BindSpan;
    use crate::identity::{ObjectListRef, SlotGeneration};
    use crate::stream::ProtectionOptions;
    use crate::sync::{BarrierOp, BarrierTarget, ResourceSpan};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn builder() -> ExecBuilder<'static> {
        ExecBuilder::new()
    }

    fn a_blit() -> ResolvedOperation {
        ResolvedOperation::Blit(BlitOp::GenerateMipmaps { texture: res(1) })
    }

    /// A draw that reads one index buffer of its own, so a test can tell the
    /// record's own footprint from the encoder's.
    fn a_draw() -> ResolvedOperation {
        ResolvedOperation::Render(RenderOp::Draw(crate::render::DrawOp::Indexed {
            primitive: crate::render::PrimitiveType(0),
            index: crate::render::IndexSource {
                buffer: res(5),
                offset: 0,
                index_type: crate::render::IndexType::Uint16,
            },
            index_count: 3,
            instances: crate::render::Instancing::default(),
            base_vertex: 0,
        }))
    }

    /// Which backing an access names, for the tests that care only about that.
    fn backing(key: AccessKey) -> Option<BackingId> {
        match key {
            AccessKey::Range(r, _) | AccessKey::Subresource(r, _) | AccessKey::Whole(r) => {
                Some(r.backing)
            }
            AccessKey::Heap(_) | AccessKey::DomainOnly => None,
        }
    }

    /// File buffer bindings in the transaction's arena and name the window,
    /// the way a resolver would.
    fn bind_arena(b: &mut ExecBuilder, buffers: &[ResourceId]) -> BindSpan {
        let start = b.arenas.buffer_bindings.len() as u32;
        for &buffer in buffers {
            b.arenas.buffer_bindings.push(BufferBinding {
                buffer: Some(buffer),
                offset: 0,
                stride: None,
            });
        }
        BindSpan {
            start,
            len: buffers.len() as u32,
        }
    }

    /// A draw declares what it reads through the encoder's bound slots, not
    /// only what its own fields name.
    ///
    /// The binding tables had a writer and a reader in [`crate::encoder`] and
    /// neither was reachable, so a buffer bound at slot 0 and read by the next
    /// draw produced no access — and a transaction that wrote it earlier
    /// compiled no edge to the draw that reads it.
    #[test]
    fn a_draw_reads_the_slots_the_binds_before_it_filled() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let entries = bind_arena(&mut b, &[res(7), res(8)]);
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                first: 0,
                entries,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");
        assert!(
            b.accesses.is_empty(),
            "a bind writes a slot and touches no memory"
        );

        b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
            .expect("draw");
        let read: Vec<_> = b.accesses.iter().filter_map(|a| backing(a.key)).collect();
        assert!(
            read.contains(&BackingId(7)) && read.contains(&BackingId(8)),
            "a draw reads both bound buffers, got {read:?}"
        );
        // Unknown until a pipeline says otherwise, and unknown conflicts with
        // a reader — which is the point of declaring it at all.
        assert!(b
            .accesses
            .iter()
            .filter(|a| backing(a.key) != Some(BackingId(5)))
            .all(|a| a.mode == AccessMode::Unknown));
    }

    /// A published reflection narrows the draw's footprint, and an unpublished
    /// stage stays `Unknown`.
    ///
    /// **The gate on the one door the model's conservatism is bought back
    /// through.** Before anything published, the first driven boot that counted
    /// found `access_mode_unknown` at 81 % of every access the dependency graph
    /// ordered against — every bound slot of every draw, because
    /// [`crate::encoder::slot_mode`] has no other answer without a
    /// [`crate::pipeline::BindingUsage`].
    ///
    /// Three slots and three answers, because the three are the three things a
    /// reflection can say and each is wrong in a different way if collapsed:
    /// slot 0 is read (narrowed, and no longer conflicts with another reader),
    /// slot 1 is written (narrowed, and still does), slot 2 is *not referenced*
    /// and contributes no participation at all. The fragment stage publishes
    /// nothing, so its own bound slot keeps the conservative answer — which is
    /// what says an unpublished stage is not silently read as "references
    /// nothing".
    #[test]
    fn a_published_reflection_narrows_the_slots_it_names_and_no_others() {
        use crate::pipeline::{BindingUsage, PipelineState, PipelineTable, PublishedUsage};
        use reims_vgpu_protocol::render::ShaderStage;

        let pipeline = res(9);
        let mut table = PipelineTable::new();
        assert!(table.declare(pipeline, crate::identity::SessionGeneration::FIRST));
        assert!(table.advance(pipeline, PipelineState::Translating));
        assert!(table.advance(pipeline, PipelineState::Compiling));
        assert!(table.advance(pipeline, PipelineState::Ready));
        // Vertex reflected, fragment not. Both are real positions for an
        // executor and the two answers differ.
        assert!(table.publish_usage(
            pipeline,
            PublishedUsage::render(
                Some(BindingUsage::new(
                    vec![Some(AccessMode::Read), Some(AccessMode::Write), None],
                    Vec::new(),
                )),
                None,
            )
        ));

        let mut b = ExecBuilder::with_usage(&table);
        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let entries = bind_arena(&mut b, &[res(7), res(8), res(6)]);
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: ShaderStage::Vertex,
                first: 0,
                entries,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");
        let fragment = bind_arena(&mut b, &[res(4)]);
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: ShaderStage::Fragment,
                first: 0,
                entries: fragment,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");
        b.record(
            ResolvedOperation::Render(RenderOp::SetPipeline { pipeline }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("set pipeline");
        b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
            .expect("draw");

        let mode_of = |id: u64| {
            b.accesses
                .iter()
                .find(|a| backing(a.key) == Some(BackingId(id)))
                .map(|a| a.mode)
        };
        assert_eq!(
            mode_of(7),
            Some(AccessMode::Read),
            "the reflection said read"
        );
        assert_eq!(
            mode_of(8),
            Some(AccessMode::Write),
            "and said write for the next slot"
        );
        assert_eq!(
            mode_of(6),
            None,
            "a slot the shader does not reference contributes nothing at all"
        );
        assert_eq!(
            mode_of(4),
            Some(AccessMode::Unknown),
            "the fragment stage published nothing, and nothing published is not \
             'references nothing'"
        );
    }

    /// A pipeline's reflection does not outlive the build it described.
    ///
    /// Every door that un-builds or unnames a pipeline drops it, and the reason
    /// is the same at each: an id the guest can reuse would otherwise answer
    /// with another object's shader, and a narrower footprint derived from the
    /// wrong shader is a missing hazard edge rather than a wrong log line.
    #[test]
    fn a_reflection_dies_with_the_build_it_described() {
        use crate::pipeline::{BindingUsage, PipelineState, PipelineTable, PublishedUsage};
        use reims_vgpu_protocol::render::ShaderStage;

        let id = res(9);
        let usage = || {
            PublishedUsage::render(
                Some(BindingUsage::new(vec![Some(AccessMode::Read)], Vec::new())),
                None,
            )
        };
        let ready = || {
            let mut t = PipelineTable::new();
            assert!(t.declare(id, crate::identity::SessionGeneration::FIRST));
            assert!(t.advance(id, PipelineState::Translating));
            assert!(t.advance(id, PipelineState::Compiling));
            assert!(t.advance(id, PipelineState::Ready));
            assert!(t.publish_usage(id, usage()));
            t
        };
        let published = |t: &PipelineTable| {
            crate::pipeline::UsageSource::binding_usage(t, id, Some(ShaderStage::Vertex)).is_some()
        };

        let mut t = ready();
        assert!(published(&t), "the fixture publishes");
        assert!(t.withdraw(id));
        assert!(
            !published(&t),
            "the executor said it no longer holds the translation this described"
        );

        let mut t = ready();
        assert!(t.retire(id));
        assert!(!published(&t), "the guest deleted the name");

        let mut t = ready();
        assert!(!t.device_lost().is_empty());
        assert!(!published(&t), "the incarnation that built it is gone");

        let mut t = ready();
        let closed = t.generation_closed(crate::identity::SessionGeneration::FIRST);
        assert!(closed.removed.contains(&id));
        assert!(!published(&t), "the generation that named it closed");

        // And a reflection landing after the name is gone does not resurrect
        // it.
        let mut t = ready();
        assert!(t.retire(id));
        assert!(
            !t.publish_usage(id, usage()),
            "a compile finishing after the delete describes a build nothing may bind"
        );
        assert!(!published(&t));
    }

    /// Three draws over one binding table declare its slots once.
    ///
    /// Every participation is a namespace resolution, a residency lookup and a
    /// reserved content version — an unreflected slot is an `Unknown`, which
    /// writes — so a draw loop that re-declared would reserve one version of
    /// each bound buffer per draw and keep the last.
    #[test]
    fn a_draw_loop_declares_its_bindings_once() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let entries = bind_arena(&mut b, &[res(7), res(8)]);
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                first: 0,
                entries,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");
        for _ in 0..3 {
            b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
                .expect("draw");
        }
        let mut named: Vec<_> = b.accesses.iter().filter_map(|a| backing(a.key)).collect();
        named.sort_unstable();
        assert_eq!(
            named,
            vec![
                BackingId(5),
                BackingId(5),
                BackingId(5),
                BackingId(7),
                BackingId(8)
            ],
            "each draw names its own index buffer; the bindings are named once"
        );
    }

    /// A pipeline bound between two draws re-declares the table: what a bound
    /// slot contributes is the pipeline's answer.
    #[test]
    fn a_pipeline_between_two_draws_re_declares_the_bindings() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let entries = bind_arena(&mut b, &[res(7)]);
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                first: 0,
                entries,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");
        b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
            .expect("draw");
        b.record(
            ResolvedOperation::Render(RenderOp::SetPipeline { pipeline: res(9) }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("pipeline");
        b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
            .expect("draw");
        let declarations = b
            .accesses
            .iter()
            .filter_map(|a| backing(a.key))
            .filter(|b| *b == BackingId(7))
            .count();
        assert_eq!(declarations, 2, "once under each pipeline");
    }

    /// A record that binds nothing leaves the tables alone, so a draw with no
    /// binds before it declares only its own fields.
    #[test]
    fn a_draw_with_nothing_bound_reads_only_what_it_names() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
            .expect("draw");
        let read: Vec<_> = b.accesses.iter().filter_map(|a| backing(a.key)).collect();
        assert_eq!(
            read,
            vec![BackingId(5)],
            "the index buffer, and nothing else"
        );
    }

    /// A new encoder starts with everything unbound. Metal's rule, and a table
    /// that survived would make the next draw name a resource nothing bound.
    #[test]
    fn a_new_encoder_inherits_no_bindings_from_the_one_before_it() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let entries = bind_arena(&mut b, &[res(7)]);
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                first: 0,
                entries,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");
        b.end_segment().expect("end");

        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
            .expect("draw");
        let read: Vec<_> = b.accesses.iter().filter_map(|a| backing(a.key)).collect();
        assert_eq!(read, vec![BackingId(5)]);
    }

    /// An encoder that spans two segments keeps its tables. The bind and the
    /// draw are in different segments and the same encoder, which is exactly
    /// the case a per-segment table would lose.
    #[test]
    fn a_continued_encoder_keeps_the_slots_the_first_segment_bound() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime {
                continues_previous: false,
                continues_into_next: true,
            },
        )
        .expect("open");
        let entries = bind_arena(&mut b, &[res(7)]);
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                first: 0,
                entries,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");
        b.end_segment().expect("held");

        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime {
                continues_previous: true,
                continues_into_next: false,
            },
        )
        .expect("continue");
        b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
            .expect("draw");
        let read: Vec<_> = b.accesses.iter().filter_map(|a| backing(a.key)).collect();
        assert!(read.contains(&BackingId(7)), "got {read:?}");
    }

    /// A dispatch reads the compute encoder's tables the same way, and a
    /// compute bind never reaches a render table.
    #[test]
    fn a_dispatch_reads_the_compute_encoders_slots() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Compute.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let entries = bind_arena(&mut b, &[res(9)]);
        b.record(
            ResolvedOperation::Compute(ComputeOp::BindBuffers { first: 0, entries }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");
        b.record(
            ResolvedOperation::Compute(ComputeOp::Dispatch(
                crate::compute::DispatchOp::Threadgroups {
                    groups: crate::compute::ComputeExtent {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    threads_per_group: crate::compute::ComputeExtent {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                },
            )),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("dispatch");
        let read: Vec<_> = b.accesses.iter().filter_map(|a| backing(a.key)).collect();
        assert_eq!(read, vec![BackingId(9)]);
    }

    /// Unbinding a slot removes it from the next draw's footprint. The guest
    /// unbinds by naming no object, and a slot that kept the old resource would
    /// order every later draw against memory nothing reads.
    #[test]
    fn an_unbound_slot_leaves_the_next_draws_footprint() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let entries = bind_arena(&mut b, &[res(7)]);
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                first: 0,
                entries,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");
        let cleared = {
            let start = b.arenas.buffer_bindings.len() as u32;
            b.arenas.buffer_bindings.push(BufferBinding {
                buffer: None,
                offset: 0,
                stride: None,
            });
            BindSpan { start, len: 1 }
        };
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                first: 0,
                entries: cleared,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("unbind");
        b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
            .expect("draw");
        let read: Vec<_> = b.accesses.iter().filter_map(|a| backing(a.key)).collect();
        assert_eq!(read, vec![BackingId(5)]);
    }

    fn a_barrier() -> ResolvedOperation {
        ResolvedOperation::Barrier(BarrierOp {
            target: BarrierTarget::Resources(ResourceSpan { start: 0, len: 1 }),
            after_stages: None,
            before_stages: None,
        })
    }

    #[test]
    fn a_finished_transaction_carries_its_records_in_order() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Blit.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let first = b
            .record(a_blit(), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        let second = b
            .record(a_blit(), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        b.end_segment().expect("end");
        b.begin_segment(
            SegmentKind::Blit.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        let third = b
            .record(a_blit(), &mut StubRegistry(ChannelId(1)))
            .expect("record");
        b.end_segment().expect("end");

        let tx = b.finish().expect("frozen");
        assert_eq!(tx.streams.len(), 2);
        assert_eq!(tx.record_count(), 3);
        let positions: Vec<_> = tx.records().map(|r| r.at).collect();
        assert_eq!(positions, vec![first, second, third]);
        assert!(positions.windows(2).all(|w| w[0] < w[1]));
    }

    /// The builder cannot say where its packet arrived, so the only way a
    /// transaction gets an identity is by being stamped with one — and the
    /// stamped value is exactly the one it was handed.
    #[test]
    fn work_carries_no_identity_until_it_is_stamped_with_one() {
        let identity = TransactionIdentity {
            session: SessionGeneration::FIRST,
            domain: ChannelId(1),
            domain_sequence: ChannelSequence(7),
            ingress: IngressOrdinal(42),
        };
        let work = builder().finish().expect("frozen");
        let tx = work.stamp(identity);
        assert_eq!(tx.identity, identity);
        assert_eq!(tx.ingress(), IngressOrdinal(42));
        assert_eq!(tx.domain(), ChannelId(1));
        assert_eq!(tx.domain_sequence(), ChannelSequence(7));
        assert_eq!(tx.session(), SessionGeneration::FIRST);
        assert_eq!(
            *tx.work, work,
            "stamping adds a position and changes nothing about the work"
        );
    }

    /// The builder does not restate the stream rules; it is the cursor that
    /// refuses, and the refusal reaches the caller unchanged.
    #[test]
    fn the_stream_rules_are_the_cursors_and_are_not_restated() {
        let mut b = builder();
        assert_eq!(
            b.record(a_blit(), &mut StubRegistry(ChannelId(1))),
            Err(StreamRefusal::RecordOutsideEncoder)
        );

        let mut b = builder();
        b.begin_segment(
            SegmentKind::Blit.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        assert_eq!(
            b.record(
                ResolvedOperation::Render(RenderOp::SetPipeline { pipeline: res(1) }),
                &mut StubRegistry(ChannelId(1))
            ),
            Err(StreamRefusal::RailMismatch {
                segment: SegmentKind::Blit,
                record: reims_vgpu_protocol::closure::Rail::Render,
            })
        );

        let mut b = builder();
        b.begin_segment(
            SegmentKind::Compute.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        assert_eq!(
            b.finish().err(),
            Some(StreamRefusal::EncoderNeverEnded(SegmentKind::Compute))
        );
    }

    /// A record that exists on more than one rail is admitted by whichever
    /// encoder is open — and refused by an encoder its class does not appear
    /// on. A barrier on the blit encoder is the pointed case: there is no such
    /// record, and admitting it would be inventing one.
    #[test]
    fn a_multi_rail_record_is_admitted_only_by_an_encoder_that_carries_it() {
        for kind in [SegmentKind::Render, SegmentKind::Compute] {
            let mut b = builder();
            b.begin_segment(kind.wire_type(), SegmentLifetime::SELF_CONTAINED)
                .expect("open");
            b.record(a_barrier(), &mut StubRegistry(ChannelId(1)))
                .expect("a barrier exists on both");
            b.end_segment().expect("end");
            assert_eq!(b.finish().expect("frozen").record_count(), 1);
        }
        for kind in [SegmentKind::Blit, SegmentKind::Event, SegmentKind::Info] {
            let mut b = builder();
            b.begin_segment(kind.wire_type(), SegmentLifetime::SELF_CONTAINED)
                .expect("open");
            assert_eq!(
                b.record(a_barrier(), &mut StubRegistry(ChannelId(1))),
                Err(StreamRefusal::RailMismatch {
                    segment: kind,
                    record: kind.rail(),
                }),
                "{kind:?} carries no barrier record"
            );
        }
    }

    /// The admissibility table is the ledger's, and this is what says so.
    ///
    /// Driven over **every** class rather than over a written list of probes.
    /// For each one, the segments it is admitted on are exactly the rails the
    /// ledger has judged an operation of that class on. A class whose rows are
    /// all unresolved is admitted nowhere, which is residency's case today and
    /// the compute fence pair's — the selector exists and the door is closed.
    ///
    /// The probe list this replaced could not see a class it did not name, and
    /// a payload added under an existing class inherits that class's answer
    /// now rather than needing an entry nobody remembers to add.
    #[test]
    fn the_admissibility_table_matches_the_ledger() {
        use crate::operation::{classify, OperationHome};
        use reims_vgpu_protocol::closure::LEDGER;

        for &class in OperationClass::ALL {
            for &kind in SegmentKind::ALL {
                let ledger_has_one = LEDGER.iter().any(|o| {
                    o.rail == kind.rail()
                        && classify(o) == Some(OperationHome::Stream(class))
                        && !matches!(
                            o.closure,
                            reims_vgpu_protocol::closure::Closure::Refused { .. }
                        )
                });
                // A single-rail class is admitted by its own rail rather than
                // by this table, and the boundary is the segment itself: the
                // ledger has boundary rows on every rail and none of them is a
                // record this table can admit.
                let single_rail = matches!(
                    class,
                    OperationClass::Render
                        | OperationClass::Compute
                        | OperationClass::Blit
                        | OperationClass::Event
                        | OperationClass::InfoQuery
                );
                if single_rail || matches!(class, OperationClass::EncoderBoundary) {
                    continue;
                }
                assert_eq!(
                    class_admissible_on(class, kind),
                    ledger_has_one,
                    "{class:?} on {kind:?}"
                );
            }
        }
    }

    /// Every payload variant reports the class its records are judged under,
    /// and a class this table admits somewhere has a payload that can reach it.
    #[test]
    fn every_multi_rail_class_that_is_admitted_somewhere_has_a_payload() {
        let samples = [
            ResolvedOperation::Fence(FenceOp {
                kind: crate::sync::FenceKind::Update,
                fence: res(1),
                stages: None,
            }),
            a_barrier(),
            ResolvedOperation::ResourceState(ResourceStateOp {
                directive: crate::resource_state::ContentDirective::Synchronize,
                target: crate::resource_state::ResourceStateTarget::Encoder,
            }),
            ResolvedOperation::IndirectCommand(IcbOp::ExecuteRange {
                icb: res(1),
                commands: crate::icb::CommandRange::default(),
            }),
        ];
        for &class in OperationClass::ALL {
            let admitted = SegmentKind::ALL
                .iter()
                .any(|&kind| class_admissible_on(class, kind));
            if !admitted {
                continue;
            }
            assert!(
                samples.iter().any(|op| op.class() == class),
                "{class:?} is admitted and has no payload to admit"
            );
        }
    }

    /// An envelope that armed nothing fails at freeze, not silently.
    #[test]
    fn an_unclaimed_protection_envelope_refuses_the_whole_packet() {
        let mut b = builder();
        b.protection_envelope(ProtectionOptions(0x44))
            .expect("armed");
        assert_eq!(
            b.finish().err(),
            Some(StreamRefusal::ProtectionEnvelopeUnclaimed)
        );
    }

    /// Binding a pipeline leases it, on both rails, and binding it again does
    /// not lease it twice.
    ///
    /// Through `record`, because that is the only door: the lease list used to
    /// be fillable beside the records and was filled by nothing but a test, so
    /// a transaction built from a guest's stream leased nothing at all and
    /// every wait naming a pipeline it plainly uses read as unleased.
    ///
    /// The guest re-binds the same pipeline on every draw, so without the
    /// dedup the list is one entry per draw and the admission check one lookup
    /// per entry.
    #[test]
    fn binding_a_pipeline_leases_it_once_however_often_it_is_bound() {
        for (kind, bind) in [
            (SegmentKind::Render, |p| {
                ResolvedOperation::Render(RenderOp::SetPipeline { pipeline: p })
            }),
            (SegmentKind::Compute, |p| {
                ResolvedOperation::Compute(crate::compute::ComputeOp::SetPipeline { pipeline: p })
            }),
        ] as [(SegmentKind, fn(ResourceId) -> ResolvedOperation); 2]
        {
            let mut b = builder();
            b.begin_segment(kind.wire_type(), SegmentLifetime::SELF_CONTAINED)
                .expect("open");
            for pipeline in [res(3), res(3), res(4)] {
                b.record(bind(pipeline), &mut StubRegistry(ChannelId(1)))
                    .expect("its own rail");
            }
            b.end_segment().expect("end");
            let work = b.finish().expect("frozen");
            assert_eq!(work.pipeline_leases, vec![res(3), res(4)], "{kind:?}");
            assert_eq!(work.record_count(), 3, "{kind:?}: every bind is a record");
        }
    }

    /// The render arm of the lease list holds the render binds and nothing
    /// else, once each.
    ///
    /// This is what the render pre-scan is handed to decide which pipelines it
    /// must have translated before a draw can run. Two things would break it,
    /// and only a packet carrying both classes can tell them apart: a compute
    /// lease reaching the render arm sends a compute pipeline down the
    /// translator's render path, where its inputs are the wrong shape and it
    /// refuses; and a duplicate turns a forty-draw stream into forty visits to
    /// the same memo, which is the cost the class-blind
    /// [`ExecWork::pipeline_leases`] was already deduplicated to avoid.
    #[test]
    fn the_render_arm_of_the_leases_is_the_render_binds_deduplicated() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Render.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open render");
        for pipeline in [res(3), res(3), res(4)] {
            b.record(
                ResolvedOperation::Render(RenderOp::SetPipeline { pipeline }),
                &mut StubRegistry(ChannelId(1)),
            )
            .expect("render bind");
        }
        b.end_segment().expect("end render");

        b.begin_segment(
            SegmentKind::Compute.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open compute");
        b.record(
            ResolvedOperation::Compute(crate::compute::ComputeOp::SetPipeline { pipeline: res(9) }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("compute bind");
        b.end_segment().expect("end compute");

        let work = b.finish().expect("frozen");
        assert_eq!(
            work.pipeline_leases,
            vec![res(3), res(4), res(9)],
            "the ordering plane waits on all three"
        );
        assert_eq!(
            work.render_pipeline_leases(),
            vec![res(3), res(4)],
            "the render pre-scan is asked about two"
        );
    }

    /// The compute arm names the pipeline a dispatch ran under, at the size
    /// that dispatch stated, and keeps the pipeline across a continuation.
    ///
    /// Each assertion below is one thing the byte scan this replaced could not
    /// say. The continuation is the one that costs a frame: a scan resetting
    /// its bound ref at every framed segment sees a dispatch with no pipeline
    /// and skips it, while the ordering plane still holds that pipeline's lease
    /// and readies it on the verdict — a transaction released against a kernel
    /// that is still translating.
    #[test]
    fn the_compute_arm_pairs_each_dispatch_with_the_pipeline_and_size_it_ran_under() {
        use crate::compute::{ComputeExtent, ComputeOp, DispatchOp};

        const fn extent(width: u64) -> ComputeExtent {
            ComputeExtent {
                width,
                height: 1,
                depth: 1,
            }
        }
        fn dispatch(threads_per_group: ComputeExtent) -> ResolvedOperation {
            ResolvedOperation::Compute(ComputeOp::Dispatch(DispatchOp::Threadgroups {
                groups: extent(1),
                threads_per_group,
            }))
        }
        fn set(pipeline: ResourceId) -> ResolvedOperation {
            ResolvedOperation::Compute(ComputeOp::SetPipeline { pipeline })
        }

        let mut b = builder();
        b.begin_segment(
            SegmentKind::Compute.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        for op in [
            set(res(3)),
            // Twice at one size is one translation.
            dispatch(extent(64)),
            dispatch(extent(64)),
            // The same kernel at another size is another.
            dispatch(extent(32)),
            // A grid read from a buffer states its threadgroup size.
            ResolvedOperation::Compute(ComputeOp::Dispatch(DispatchOp::ThreadsIndirect {
                source: crate::compute::IndirectSource {
                    buffer: res(5),
                    offset: 0,
                },
            })),
            // Bound and never dispatched: the ordering plane holds it, the
            // translator is never asked about it.
            set(res(4)),
        ] {
            b.record(op, &mut StubRegistry(ChannelId(1)))
                .expect("record");
        }
        b.end_segment().expect("end");
        let work = b.finish().expect("frozen");

        assert_eq!(
            work.compute_dispatch_translations(),
            vec![(res(3), extent(64)), (res(3), extent(32))],
            "one pair per distinct (pipeline, size) a dispatch stated"
        );
        assert_eq!(
            work.pipeline_leases,
            vec![res(3), res(4)],
            "the ordering plane holds the pipeline nothing dispatched too"
        );
        assert!(
            work.render_pipeline_leases().is_empty(),
            "no render bind reaches the render arm"
        );
    }

    /// A record the cursor refuses leaves no lease behind claiming it ran.
    ///
    /// The same rollback the accesses get, and it has to be the same one: a
    /// transaction holding a lease for a pipeline no record of its own binds
    /// waits for a compilation it has no interest in, which the guest
    /// experiences as a frame that never arrives.
    /// **A record the builder does not keep must not take the encoder's
    /// declaration with it.**
    ///
    /// The footprint has to be gathered before anything can refuse it — the
    /// accesses are built from it — so a refused record has had its footprint
    /// *taken*. Marking the slots declared as part of that gather meant the
    /// refused record consumed the declaration and the next draw named
    /// nothing: no access, no hazard edge, and no log line saying so. A
    /// resolver that refuses one record and keeps going is a wrong frame.
    #[test]
    fn a_refused_draw_does_not_consume_the_encoders_declaration() {
        let mut b = builder();
        b.begin_encoder(SegmentKind::Render, SegmentLifetime::SELF_CONTAINED)
            .expect("open");
        let entries = bind_arena(&mut b, &[res(7)]);
        b.record(
            ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                first: 0,
                entries,
            }),
            &mut StubRegistry(ChannelId(1)),
        )
        .expect("bind");

        // A source that refuses everything, so this draw is not kept.
        let mut refusing = |p: &Participation| {
            Err(crate::access::AccessRefusal {
                resource: p.resource,
                reason: "the test refused it",
            })
        };
        assert!(matches!(
            b.record(a_draw(), &mut refusing),
            Err(StreamRefusal::Access(_))
        ));
        assert!(b.accesses.is_empty(), "and it left no accesses behind");

        // The next draw is the first one the transaction keeps, so it is the
        // one that owes the binding.
        b.record(a_draw(), &mut StubRegistry(ChannelId(1)))
            .expect("draw");
        let mut named: Vec<_> = b.accesses.iter().filter_map(|a| backing(a.key)).collect();
        named.sort_unstable();
        assert_eq!(
            named,
            vec![BackingId(5), BackingId(7)],
            "the kept draw names its index buffer and the slot the bind filled"
        );
    }

    #[test]
    fn a_refused_bind_leases_nothing() {
        let mut b = builder();
        b.begin_segment(
            SegmentKind::Blit.wire_type(),
            SegmentLifetime::SELF_CONTAINED,
        )
        .expect("open");
        assert!(b
            .record(
                ResolvedOperation::Render(RenderOp::SetPipeline { pipeline: res(3) }),
                &mut StubRegistry(ChannelId(1)),
            )
            .is_err());
        b.end_segment().expect("end");
        assert!(b.finish().expect("frozen").pipeline_leases.is_empty());
    }

    /// Only the two pipeline-state records lease anything.
    ///
    /// A census rather than a spot check: the answer is exhaustive over
    /// `ResolvedOperation`, and an operation that grew a pipeline reference
    /// without being added to it would silently lease nothing.
    #[test]
    fn no_other_record_class_leases_a_pipeline() {
        for op in [
            a_blit(),
            a_barrier(),
            ResolvedOperation::Render(RenderOp::SetDepthStencilState { state: res(3) }),
            ResolvedOperation::IndirectCommand(crate::icb::IcbOp::ExecuteRange {
                icb: res(3),
                commands: crate::icb::CommandRange {
                    location: 0,
                    length: 1,
                },
            }),
        ] {
            assert_eq!(op.pipeline_lease(), None, "{:?}", op.class());
        }
        assert_eq!(
            ResolvedOperation::Render(RenderOp::SetPipeline { pipeline: res(3) }).pipeline_lease(),
            Some(res(3))
        );
        assert_eq!(
            ResolvedOperation::Compute(crate::compute::ComputeOp::SetPipeline { pipeline: res(4) })
                .pipeline_lease(),
            Some(res(4))
        );
    }

    /// Prerequisites are kept apart from accesses, because one of them may name
    /// work that has not arrived.
    #[test]
    fn prerequisites_and_accesses_are_separate_lists() {
        let mut b = builder();
        b.require(Prerequisite::Event {
            event: res(5),
            value: 3,
        });
        b.declare_access(AccessIntent {
            domain: ChannelId(1),
            key: AccessKey::Whole(ResourceKey {
                backing: BackingId(1),
                heap: None,
            }),
            mode: AccessMode::Write,
            api_stages: 0,
            input_content_version: None,
            output_content_version: Some(ContentVersion(2)),
        });
        let tx = b.finish().expect("frozen");
        assert_eq!(tx.prerequisites.len(), 1);
        assert_eq!(tx.accesses.len(), 1);
        assert!(tx.writes_anything());
    }

    /// A transaction that touches nothing says so, and is not mistaken for one
    /// whose participation was never worked out.
    #[test]
    fn a_transaction_with_no_accesses_writes_nothing() {
        let tx = builder().finish().expect("frozen");
        assert!(!tx.writes_anything());
        assert_eq!(tx.record_count(), 0);
        assert!(tx.accesses.is_empty());
    }

    /// A version claim is the write access's claim, so it is read off the
    /// access and cannot name different memory than the write did.
    #[test]
    fn a_published_version_covers_exactly_the_region_its_access_named() {
        let region = AccessKey::Range(
            crate::access::ResourceKey {
                backing: BackingId(9),
                heap: None,
            },
            crate::access::ByteRange {
                offset: 64,
                length: 128,
            },
        );
        let mut b = builder();
        b.declare_access(AccessIntent {
            domain: ChannelId(1),
            key: region,
            mode: crate::access::AccessMode::Write,
            api_stages: 0,
            input_content_version: None,
            output_content_version: Some(ContentVersion(2)),
        });
        let tx = b.finish().expect("frozen");
        assert_eq!(
            tx.published_versions().collect::<Vec<_>>(),
            vec![VersionPublication {
                backing: BackingId(9),
                region,
                to: ContentVersion(2),
            }]
        );
    }

    /// A heap declaration and a domain-only access name no bytes, so neither
    /// can claim to have produced any.
    #[test]
    fn an_access_that_names_no_memory_publishes_no_version() {
        let mut b = builder();
        for key in [
            AccessKey::Heap(crate::access::HeapId {
                id: 1,
                membership_generation: 0,
            }),
            AccessKey::DomainOnly,
        ] {
            b.declare_access(AccessIntent {
                domain: ChannelId(1),
                key,
                mode: crate::access::AccessMode::Write,
                api_stages: 0,
                input_content_version: None,
                output_content_version: Some(ContentVersion(2)),
            });
        }
        let tx = b.finish().expect("frozen");
        assert_eq!(tx.published_versions().count(), 0);
    }

    /// Every operation class answers the participation question, and the two
    /// that route through something other than their own fields answer it
    /// correctly.
    ///
    /// The aggregation is exhaustive by construction — the match in
    /// `participations` has no wildcard — so what a test can still catch is an
    /// arm wired to the wrong source. Two are:
    ///
    /// * `WriteDescriptor` is the only arm that reads the arena, and it is the
    ///   only participation a *pass* contributes. Wiring it to the record's own
    ///   (empty) answer would lose every attachment of every pass, and a pass
    ///   with no draws would become a transaction that touches nothing.
    /// * A barrier carries a resource list and declares no participation on it.
    ///   Reading that list as accesses would order every barrier against
    ///   everything it named.
    #[test]
    fn every_class_answers_what_it_touches_and_only_the_pass_reads_the_arena() {
        let mut arenas = ExecArenas::default();
        let mut pass = crate::pass::PassDescriptor::empty();
        pass.visibility_result_buffer =
            Some(crate::pass::VisibilityResultBuffer { buffer: res(11) });
        arenas.pass_descriptors.push(pass);
        arenas.resources.push(res(1));

        let ask = |op: ResolvedOperation, arenas: &ExecArenas| -> Vec<Participation> {
            let mut out = Vec::new();
            op.participations(arenas, &mut out);
            out
        };

        // The pass's own footprint, reached only through the arena.
        let write_descriptor = ResolvedOperation::Render(RenderOp::WriteDescriptor {
            descriptor: crate::render::PassDescriptorSlot(0),
        });
        let parts = ask(write_descriptor, &arenas);
        assert_eq!(parts.len(), 1, "the visibility buffer is the pass's write");
        assert_eq!(parts[0].resource, res(11));
        assert_eq!(parts[0].mode, AccessMode::Write);
        // And it really is the arena that supplied it: the same record against
        // an arena that does not hold the slot contributes nothing rather than
        // panicking.
        assert!(ask(write_descriptor, &ExecArenas::default()).is_empty());

        // A barrier names a resource list and participates in none of it.
        assert!(ask(a_barrier(), &arenas).is_empty());
        // A fence and an event name their own object and no memory.
        assert!(ask(
            ResolvedOperation::Fence(crate::sync::FenceOp {
                kind: crate::sync::FenceKind::Update,
                fence: res(2),
                stages: None,
            }),
            &arenas
        )
        .is_empty());
        assert!(ask(
            ResolvedOperation::Event(crate::sync::EventOp {
                kind: crate::sync::EventKind::Signal,
                event: res(3),
                value: 9,
            }),
            &arenas
        )
        .is_empty());
        // A transfer names its operand.
        let blit = ask(a_blit(), &arenas);
        assert_eq!(blit.len(), 1);
        assert_eq!(blit[0].resource, res(1));

        // A synchronise reads the content it publishes; the four directives
        // with no modelled effect name nothing.
        use crate::resource_state::{ContentDirective, ResourceStateOp, ResourceStateTarget};
        let target = ResourceStateTarget::Resource {
            resource: res(4),
            subresource: None,
        };
        let sync = ask(
            ResolvedOperation::ResourceState(ResourceStateOp {
                directive: ContentDirective::Synchronize,
                target,
            }),
            &arenas,
        );
        assert_eq!(sync.len(), 1);
        assert_eq!(sync[0].resource, res(4));
        assert_eq!(sync[0].mode, AccessMode::Read);
        for directive in [
            ContentDirective::OptimizeForCpu,
            ContentDirective::OptimizeForGpu,
            ContentDirective::InvalidateCompressed,
            ContentDirective::FlushCompressedReinterpretation,
        ] {
            assert!(
                ask(
                    ResolvedOperation::ResourceState(ResourceStateOp { directive, target }),
                    &arenas
                )
                .is_empty(),
                "{directive:?}"
            );
        }
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

    /// A source that answers like [`StubRegistry`] and refuses on a schedule.
    ///
    /// The refusal is what drives the rollback. It counts every call, so the
    /// sweep can compare what the builder kept against what the source was
    /// actually asked for — which is the only way to tell "the record declared
    /// nothing" from "the record's accesses were dropped".
    struct Flaky {
        domain: ChannelId,
        /// Calls before the next refusal, or `None` to answer everything.
        refuse_at: Option<u32>,
        calls: u32,
        answered: u32,
    }

    impl AccessSource for Flaky {
        fn access(
            &mut self,
            participation: &Participation,
        ) -> Result<AccessIntent, crate::access::AccessRefusal> {
            self.calls += 1;
            if self.refuse_at == Some(self.calls) {
                return Err(crate::access::AccessRefusal {
                    resource: participation.resource,
                    reason: "sweep_refused_this_access",
                });
            }
            self.answered += 1;
            Ok(participation.resolve(
                self.domain,
                ResourceKey {
                    backing: BackingId(u64::from(participation.resource.slot.0)),
                    heap: None,
                },
                None,
                None,
            ))
        }
    }

    /// The classes an encoder of each kind carries, so the driver spends most
    /// of its steps inside the rules rather than bouncing off them.
    ///
    /// Read off [`class_admissible_on`] and the single-rail mapping — driving,
    /// not checking: what the builder does with a record it should not have
    /// taken is asserted below whatever this picks.
    fn admissible_shapes(kind: SegmentKind) -> &'static [u64] {
        match kind {
            SegmentKind::Render => &[0, 0, 1, 5, 6, 8, 10, 12, 12],
            SegmentKind::Compute => &[2, 3, 5, 9],
            SegmentKind::Blit => &[4, 6, 8, 9, 11],
            SegmentKind::Event => &[7],
            SegmentKind::Info => &[],
        }
    }

    /// Every operation shape the sweep drives, each in its own class.
    fn some_op(rng: &mut Rng, open: Option<SegmentKind>, entries: BindSpan) -> ResolvedOperation {
        // Mostly a shape the open encoder carries, so the sweep reaches the
        // paths past admission; the rest is whatever, so it keeps reaching the
        // refusals too.
        let which = match open.map(admissible_shapes) {
            Some(shapes) if !shapes.is_empty() && rng.below(4) != 0 => {
                shapes[rng.below(shapes.len() as u64) as usize]
            }
            _ => rng.below(13),
        };
        let r = res(rng_slot(rng));
        let r2 = res(rng_slot(rng));
        match which {
            0 => a_draw(),
            1 => ResolvedOperation::Render(RenderOp::SetPipeline { pipeline: r }),
            2 => ResolvedOperation::Compute(ComputeOp::SetPipeline { pipeline: r }),
            3 => ResolvedOperation::Compute(ComputeOp::Dispatch(
                crate::compute::DispatchOp::Threadgroups {
                    groups: crate::compute::ComputeExtent {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    threads_per_group: crate::compute::ComputeExtent {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                },
            )),
            4 => a_blit(),
            5 => a_barrier(),
            6 => ResolvedOperation::Fence(crate::sync::FenceOp {
                kind: reims_vgpu_protocol::sync::FenceKind::Update,
                fence: r,
                stages: None,
            }),
            7 => ResolvedOperation::Event(crate::sync::EventOp {
                kind: reims_vgpu_protocol::sync::EventKind::Signal,
                event: r,
                value: 1,
            }),
            8 => ResolvedOperation::IndirectCommand(crate::icb::IcbOp::ExecuteRange {
                icb: r,
                commands: crate::icb::CommandRange {
                    location: 0,
                    length: 4,
                },
            }),
            9 => ResolvedOperation::ResourceState(ResourceStateOp {
                directive: crate::resource_state::ContentDirective::Synchronize,
                target: crate::resource_state::ResourceStateTarget::Resource {
                    resource: r,
                    subresource: None,
                },
            }),
            // A record naming two resources, so a refusal can land between
            // them and leave the first already answered.
            10 => {
                ResolvedOperation::Render(RenderOp::Draw(crate::render::DrawOp::IndexedIndirect {
                    primitive: crate::render::PrimitiveType(0),
                    index: crate::render::IndexSource {
                        buffer: r,
                        offset: 0,
                        index_type: crate::render::IndexType::Uint16,
                    },
                    arguments: crate::bind::IndirectSource {
                        buffer: r2,
                        offset: 0,
                    },
                }))
            }
            11 => ResolvedOperation::Blit(BlitOp::BufferToBuffer {
                source: r,
                source_offset: 0,
                dest: r2,
                dest_offset: 0,
                size: 16,
            }),
            // The record that fills the encoder's tables. Its own footprint is
            // empty — a bind writes a slot and touches no memory — and what it
            // buys is the *next* draw's.
            _ => ResolvedOperation::Render(RenderOp::BindBuffers {
                stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                first: 0,
                entries,
            }),
        }
    }

    fn rng_slot(rng: &mut Rng) -> u32 {
        rng.below(4) as u32 + 1
    }

    /// What the builder held, as far as a caller can see it.
    #[derive(Clone, Debug, PartialEq)]
    struct Held {
        accesses: usize,
        leases: Vec<ResourceId>,
        streams: usize,
        open_records: Option<usize>,
    }

    fn held(b: &ExecBuilder) -> Held {
        Held {
            accesses: b.accesses.len(),
            leases: b.pipeline_leases.clone(),
            streams: b.streams.len(),
            open_records: b.open.as_ref().map(|s| s.records.len()),
        }
    }

    #[derive(Default)]
    struct Census {
        finished: u32,
        refused_finish: u32,
        records_kept: u32,
        records_refused: u32,
        /// Records refused by the source rather than by the cursor, which are
        /// the ones that had already declared accesses.
        refused_by_the_source: u32,
        /// Refusals that arrived after the source had already answered part of
        /// the record's list, which is the only shape a partial roll-back can
        /// be seen in.
        refused_mid_record: u32,
        leases: u32,
        rebound_pipelines: u32,
        binds: u32,
        /// Bind entries that changed the slot, and entries that did not.
        rearming_binds: u32,
        unchanged_binds: u32,
        /// Draws that read slots a bind before them had filled, which is the
        /// case the encoder tables exist for.
        draws_over_bindings: u32,
    }

    /// **A record either takes effect entirely or leaves nothing behind, and a
    /// finished transaction's derived lists are functions of the records it
    /// kept.**
    ///
    /// Three shadows, none of which knows how the builder works.
    ///
    /// The first is [`Held`]: everything a refusal must not change, snapshotted
    /// before the call and compared after. A record the cursor declines after
    /// the source has already answered two of its participations is the shape
    /// that costs a transaction its meaning — two accesses claiming work that
    /// is not in any stream — and it is driven here rather than reasoned about.
    ///
    /// The second is the lease list, recomputed at the end by scanning the
    /// finished transaction's records for the ones that bind a pipeline. The
    /// builder accumulates it incrementally and deduplicates as it goes; the
    /// shadow scans the whole thing once and deduplicates at the end, so a
    /// lease taken for a record that was refused, or dropped because an equal
    /// one came earlier under a different generation, disagrees.
    ///
    /// The third is the source's own call count: the builder must keep exactly
    /// the accesses the source answered for records it kept, and no others.
    #[test]
    fn a_record_is_kept_whole_or_not_at_all() {
        let mut census = Census::default();
        for seed in 0..500u64 {
            let mut rng = Rng::new(seed + 1);
            let mut b = builder();
            let mut source = Flaky {
                domain: ChannelId(1),
                refuse_at: None,
                calls: 0,
                answered: 0,
            };
            // Accesses the builder is entitled to hold: what the source
            // answered for records that were kept.
            let mut owed = 0usize;
            // Which encoder the driver believes is open. Followed from the
            // builder's answers rather than predicted — it steers the driver
            // and asserts nothing.
            let mut open: Option<SegmentKind> = None;
            // The shadow's model of the open render encoder's vertex buffer
            // table: which slots hold what, which of those a draw has already
            // declared, and which pipeline is bound. A map and a set rather
            // than two parallel vectors, so the shadow cannot make the real
            // table's mistakes about growth, gaps or the slot past the end.
            let mut slots: std::collections::HashMap<u32, ResourceId> =
                std::collections::HashMap::new();
            let mut declared: std::collections::HashSet<u32> = std::collections::HashSet::new();
            let mut pipeline: Option<ResourceId> = None;

            for _ in 0..24 {
                let before = held(&b);
                // A tenth of the steps are deliberately out of order — a
                // record with nothing open, a second begin, an end with
                // nothing to end — so the refusal paths stay driven.
                let step = if rng.below(10) == 0 {
                    rng.below(3)
                } else if open.is_none() {
                    0
                } else if rng.below(6) == 0 {
                    1
                } else {
                    2
                };
                match step {
                    0 => {
                        let kind = match rng.below(5) {
                            0 => SegmentKind::Render,
                            1 => SegmentKind::Compute,
                            2 => SegmentKind::Blit,
                            3 => SegmentKind::Event,
                            _ => SegmentKind::Info,
                        };
                        let lifetime = SegmentLifetime {
                            continues_previous: false,
                            continues_into_next: false,
                        };
                        if b.begin_encoder(kind, lifetime).is_err() {
                            assert_eq!(held(&b), before, "seed {seed}: a refused begin");
                        } else {
                            open = Some(kind);
                            // A new encoder starts with everything unbound and
                            // no pipeline, which is Metal's rule rather than a
                            // simplification.
                            slots.clear();
                            declared.clear();
                            pipeline = None;
                        }
                    }
                    1 => {
                        if b.end_segment().is_err() {
                            assert_eq!(held(&b), before, "seed {seed}: a refused end");
                        } else {
                            open = None;
                        }
                    }
                    _ => {
                        // Refuse somewhere inside this record's own list, so
                        // both "refused on its first participation" and
                        // "refused with some already answered" occur.
                        let calls_before = source.calls;
                        let answered_before = source.answered;
                        source.refuse_at =
                            (rng.below(4) == 0).then(|| calls_before + rng.below(3) as u32 + 1);
                        // Two slots' worth of bindings filed the way a
                        // resolver would file them, whether or not the shape
                        // drawn below is the one that names them.
                        let width = rng.below(3) as usize + 1;
                        // Varied, so a rebind is sometimes the same resource
                        // and sometimes a different one. A guest re-binds its
                        // whole table between draws, and only the entries that
                        // actually changed may make a draw declare again.
                        let filed: Vec<ResourceId> =
                            (0..width).map(|_| res(rng.below(2) as u32 + 10)).collect();
                        let entries = bind_arena(&mut b, &filed);
                        let op = some_op(&mut rng, open, entries);
                        match b.record(op, &mut source) {
                            Ok(_) => {
                                census.records_kept += 1;
                                let declared_here = (source.answered - answered_before) as usize;
                                owed += declared_here;
                                // The claim this module exists for: a draw
                                // declares what its own fields name *and* what
                                // the encoder had bound before it. The shadow
                                // is a count, cleared when an encoder opens.
                                match op {
                                    ResolvedOperation::Render(RenderOp::BindBuffers {
                                        first: 0,
                                        ..
                                    }) => {
                                        assert_eq!(
                                            declared_here, 0,
                                            "seed {seed}: a bind touches no memory"
                                        );
                                        // A bind of what the slot already holds
                                        // changes nothing, including whether a
                                        // draw has to declare it again. A guest
                                        // that re-binds its whole table between
                                        // draws must not make every draw
                                        // declare everything.
                                        for (i, r) in filed.iter().enumerate() {
                                            let slot = i as u32;
                                            if slots.get(&slot) != Some(r) {
                                                slots.insert(slot, *r);
                                                declared.remove(&slot);
                                                census.rearming_binds += 1;
                                            } else {
                                                census.unchanged_binds += 1;
                                            }
                                        }
                                        census.binds += 1;
                                    }
                                    // What a bound slot contributes is the
                                    // pipeline's answer, so a *different*
                                    // pipeline makes every slot a fresh
                                    // question and the same one changes
                                    // nothing.
                                    ResolvedOperation::Render(RenderOp::SetPipeline {
                                        pipeline: bound_now,
                                    }) => {
                                        if pipeline != Some(bound_now) {
                                            pipeline = Some(bound_now);
                                            declared.clear();
                                        }
                                    }
                                    ResolvedOperation::Render(RenderOp::Draw(draw)) => {
                                        let own = match draw {
                                            crate::render::DrawOp::Indexed { .. } => 1,
                                            crate::render::DrawOp::IndexedIndirect { .. } => 2,
                                            _ => unreachable!("the sweep drives two draw shapes"),
                                        };
                                        let owed_here =
                                            slots.keys().filter(|s| !declared.contains(s)).count();
                                        assert_eq!(
                                            declared_here,
                                            own + owed_here,
                                            "seed {seed}: a draw names its own buffers and \
                                             every bound slot no draw has declared yet"
                                        );
                                        if owed_here > 0 {
                                            census.draws_over_bindings += 1;
                                        }
                                        // Declared once per encoder per
                                        // pipeline: a draw loop pays for its
                                        // bindings on its first iteration.
                                        declared.extend(slots.keys().copied());
                                    }
                                    _ => {}
                                }
                                if let Some(pipeline) = op.pipeline_lease() {
                                    if before.leases.contains(&pipeline) {
                                        census.rebound_pipelines += 1;
                                    } else {
                                        census.leases += 1;
                                    }
                                }
                            }
                            Err(refusal) => {
                                census.records_refused += 1;
                                if matches!(refusal, StreamRefusal::Access(_)) {
                                    census.refused_by_the_source += 1;
                                    if source.answered > answered_before {
                                        census.refused_mid_record += 1;
                                    }
                                }
                                assert_eq!(
                                    held(&b),
                                    before,
                                    "seed {seed}: a refused record left something behind"
                                );
                            }
                        }
                        source.refuse_at = None;
                    }
                }
                assert_eq!(
                    b.accesses.len(),
                    owed,
                    "seed {seed}: the builder holds accesses no kept record answered for"
                );
            }

            // Mostly close what is open, so a transaction that had a legal
            // shape gets to finish; sometimes not, because an encoder that
            // never ended is a refusal the cursor owes and the sweep wants it.
            if open.is_some() && rng.below(4) != 0 {
                let _ = b.end_segment();
            }

            match b.finish() {
                Err(_) => census.refused_finish += 1,
                Ok(work) => {
                    census.finished += 1;
                    assert_eq!(work.accesses.len(), owed, "seed {seed}: accesses at finish");

                    // Every access carries the source's domain, so a
                    // transaction cannot have been assembled from two.
                    for access in &work.accesses {
                        assert_eq!(access.domain, ChannelId(1), "seed {seed}");
                    }

                    // The lease list, recomputed by scanning the records the
                    // transaction actually kept.
                    let mut want: Vec<ResourceId> = Vec::new();
                    for record in work.records() {
                        if let Some(pipeline) = record.op.pipeline_lease() {
                            if !want.contains(&pipeline) {
                                want.push(pipeline);
                            }
                        }
                    }
                    assert_eq!(work.pipeline_leases, want, "seed {seed}: leases");

                    // Positions never go backwards, and the record count is the
                    // records.
                    let mut last: Option<StreamPosition> = None;
                    let mut counted = 0usize;
                    for record in work.records() {
                        if let Some(previous) = last {
                            assert!(
                                record.at > previous,
                                "seed {seed}: {:?} does not follow {previous:?}",
                                record.at
                            );
                        }
                        last = Some(record.at);
                        counted += 1;
                    }
                    assert_eq!(counted, work.record_count(), "seed {seed}: record_count");

                    assert_eq!(
                        work.writes_anything(),
                        work.accesses.iter().any(|a| a.mode.writes()),
                        "seed {seed}: writes_anything"
                    );
                }
            }
        }

        assert!(census.finished > 250, "{}", census.finished);
        assert!(census.refused_finish > 50, "{}", census.refused_finish);
        assert!(census.records_kept > 3000, "{}", census.records_kept);
        assert!(census.records_refused > 1500, "{}", census.records_refused);
        assert!(
            census.refused_by_the_source > 200,
            "the source never refused: {}",
            census.refused_by_the_source
        );
        assert!(
            census.refused_mid_record > 50,
            "no record was refused with part of its list already answered: {}",
            census.refused_mid_record
        );
        assert!(census.leases > 250, "{}", census.leases);
        assert!(
            census.rebound_pipelines > 50,
            "no pipeline was ever rebound: {}",
            census.rebound_pipelines
        );
        assert!(census.binds > 200, "{}", census.binds);
        assert!(census.rearming_binds > 300, "{}", census.rearming_binds);
        assert!(
            census.unchanged_binds > 50,
            "no slot was ever re-bound to what it already held: {}",
            census.unchanged_binds
        );
        assert!(
            census.draws_over_bindings > 50,
            "no draw ever read a bound slot: {}",
            census.draws_over_bindings
        );
    }
}
