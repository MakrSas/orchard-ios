//! Turning a decoded record into a resolved operation.
//!
//! # This is the second of decode's two questions, and it is answered once
//!
//! `reims_vgpu_protocol::decode` answers *what did the guest write* and hands
//! back records with the guest's own `u32` refs still on them. This module
//! answers *which object does each ref name*, which is a question about device
//! state, and it is the only place that answers it. A decoder that resolved
//! refs would have to borrow live device state to read a record; a model that
//! resolved them a second time downstream could get a different answer than the
//! one the operation was planned against. The whole replacement exists partly
//! to delete that second shape.
//!
//! # A ref that names nothing is a refusal, not a zero
//!
//! [`ResourceId`] carries a generation, so a guest that deletes an object and
//! creates another in the same slot produces an id the old work no longer
//! matches. Resolution is where that check happens: a ref with no live object
//! refuses by name and the operation never enters the model. Substituting a
//! null id would put an operation into the stream that names something that
//! does not exist, and the dependency graph would then order real work against
//! it.
//!
//! # Counted lists go to the arena, not into the operation
//!
//! A barrier's resource list is guest-sized. The operation names a window of
//! the transaction's arena — two `u32` — so one arena serves every class that
//! carries a list and no operation is as large as the largest list anyone sent.
//! Appending is the only copy in the path, and it copies resolved ids rather
//! than guest bytes.

use crate::bind::{BindSpan, BufferBinding, IndirectSource, LodClamp, ObjectBinding};
use crate::blit::{
    BlitOp, BlitOptions, BufferSpan, FillPattern, ImagePitch, Origin3, Size3, SpanOrigin,
    TexturePoint,
};
use crate::compute::{ComputeExtent, ComputeOp, ComputeOrigin, DispatchOp};
use crate::exec::{ExecArenas, ResolvedOperation};
use crate::icb::{CommandRange, IcbOp};
use crate::identity::ResourceId;
use crate::operation::{classify, OperationClass, OperationHome};
use crate::pass::{
    Attachment, AttachmentSlot, DepthResolveFilter, LoadAction, PassDescriptor, RenderTargetExtent,
    StencilResolveFilter, StoreAction, StoreActionOptions, VisibilityResultBuffer,
};
use crate::render::{
    DrawOp, FloatBits, IndexSource, Instancing, PassDescriptorSlot, PrimitiveType, RenderOp,
    ScissorRect, StateSpan, Viewport,
};
use crate::resource_state::{ResourceStateOp, ResourceStateTarget, SliceLevel};
use crate::sync::{BarrierOp, BarrierTarget, EventOp, FenceOp, ResourceSpan};
use reims_vgpu_protocol::closure::{self, Rail};
use reims_vgpu_protocol::decode::blit as record;
use reims_vgpu_protocol::decode::blit::{
    BlitRecord, FillPattern as RecordFill, Origin as RecordOrigin, Size as RecordSize,
    TextureEndpoint,
};
use reims_vgpu_protocol::decode::compute as protocol_compute;
use reims_vgpu_protocol::decode::compute::{
    ComputeRecord, DispatchRecord, Extent as RecordExtent, IndirectRef as ComputeIndirect,
};
use reims_vgpu_protocol::decode::icb::IcbRecord;
use reims_vgpu_protocol::decode::render as protocol_render;
use reims_vgpu_protocol::decode::render::{
    DrawRecord, IndexRef as RecordIndexRef, IndirectRef as RenderIndirect,
    Instancing as RecordInstancing, RenderRecord,
};
use reims_vgpu_protocol::decode::resource_state::{RecordTarget, ResourceStateRecord};
use reims_vgpu_protocol::decode::sync::{BarrierRecord, EventRecord, FenceRecord, SyncRecord};
use reims_vgpu_protocol::decode::{self, Op};
use reims_vgpu_protocol::decode::{
    AttachmentPrefix, BufferBind, BufferStrideBind, DecodeRefusal, RefBind, RenderPassBody,
    SamplerLodBind, ScissorRect as WireScissorRect, Viewport as WireViewport,
};

/// What a guest ref names right now.
///
/// One method, because that is the whole of what this layer may ask device
/// state. Anything more — a resource's size, its format, where its content is —
/// is a question the model answers from its own records, and a resolver that
/// could answer it would make resolution a second place those facts live.
pub trait RefResolver {
    /// The live resource a ref names, or `None` when the slot holds nothing.
    fn resource(&self, object_ref: u32) -> Option<ResourceId>;
}

/// Every task's object namespace, asked for by the task that owns it.
///
/// # Why this exists beside [`RefResolver`] rather than replacing it
///
/// A `RefResolver` **is** one namespace. An interpreter holding one is already
/// inside the task whose refs it is resolving, so "which task" is not a question
/// it can get wrong, and giving it a task parameter would be a field nothing
/// could disagree about.
///
/// A byte-to-operation join is not in that position. It is handed a payload; the
/// task id is *inside* that payload; and it learns which task only by decoding.
/// A join taking a `RefResolver` therefore takes the **caller's guess** about
/// which namespace the packet meant — on exactly the packets whose own contract
/// is that the task is the header's and not a guess. Object-list refs are
/// per-task slot numbers, so the wrong namespace does not refuse: it resolves,
/// to another task's resource, and the operation retires or invalidates storage
/// the packet never named.
///
/// So [`crate::lifecycle`]'s joins take this instead and bind the task from the
/// bytes they just decoded, with [`InTask`]. A caller cannot pass the wrong
/// namespace because it no longer picks one.
///
/// # It is not a second lifetime model
///
/// One method, and it answers exactly what `RefResolver` answers, for a stated
/// task. Nothing here mints, leases, retires or generates: the generation in the
/// returned [`ResourceId`] is the namespace's, and this trait is the routing to
/// the namespace and nothing else.
pub trait TaskNamespaces {
    /// The live resource a ref names in one task's object namespace, or `None`
    /// when that task has no namespace or the slot holds nothing.
    fn resource(&self, task: crate::identity::TaskId, object_ref: u32) -> Option<ResourceId>;
}

/// One task's namespace out of a source of many, in the shape [`RefResolver`]
/// wants.
///
/// The binder, and the only way a [`TaskNamespaces`] becomes a `RefResolver`.
/// Constructed where a task id has just been decoded from a packet, so the
/// namespace and the packet's own task cannot come apart.
#[derive(Clone, Copy)]
pub struct InTask<'a, S: ?Sized> {
    source: &'a S,
    task: crate::identity::TaskId,
}

impl<'a, S: TaskNamespaces + ?Sized> InTask<'a, S> {
    /// Bind a namespace source to one task.
    pub const fn new(source: &'a S, task: crate::identity::TaskId) -> Self {
        Self { source, task }
    }

    /// The task this resolves in.
    #[must_use]
    pub const fn task(&self) -> crate::identity::TaskId {
        self.task
    }
}

impl<S: TaskNamespaces + ?Sized> RefResolver for InTask<'_, S> {
    fn resource(&self, object_ref: u32) -> Option<ResourceId> {
        self.source.resource(self.task, object_ref)
    }
}

/// A borrow of a resolver resolves the same as the resolver.
///
/// Stated so a caller holding `&Namespace` can hand it to a wrapper that takes
/// one by value without an owned copy of a namespace being created to satisfy a
/// type.
impl<R: RefResolver + ?Sized> RefResolver for &R {
    fn resource(&self, object_ref: u32) -> Option<ResourceId> {
        (**self).resource(object_ref)
    }
}

/// One namespace standing in for every task's.
///
/// **For a holder of exactly one namespace, and it says so in its name.** A
/// bench, a fixture, or a model whose packets all name one task has one
/// namespace and no routing question to get wrong; making it satisfy
/// [`TaskNamespaces`] should not cost it a map keyed by a task id it does not
/// have.
///
/// It is the shape [`TaskNamespaces`] exists to stop a *device* from taking. A
/// device with more than one task that wrapped one namespace this way would
/// resolve every packet's refs in whichever namespace it wrapped, which is the
/// wrong-namespace resolution described above — so this is deliberately not
/// something the device can arrive at by accident: it has to be written, by
/// name, at the call site.
#[derive(Clone, Copy, Debug)]
pub struct SameForEveryTask<R>(pub R);

impl<R: RefResolver> TaskNamespaces for SameForEveryTask<R> {
    fn resource(&self, _task: crate::identity::TaskId, object_ref: u32) -> Option<ResourceId> {
        self.0.resource(object_ref)
    }
}

/// The storage a task's live resource name answers for, as an access is keyed
/// on it.
///
/// # Why this is not a fourth method on [`TaskNamespaces`]
///
/// [`TaskNamespaces`] answers what a *ref* names, and its whole claim is that it
/// mints nothing: the generation it hands back is the namespace's. This answers
/// something else — where a named resource's bytes are — and the two have
/// different holders on a real device. Folded together, the one implementor that
/// can answer refs but not storage would have to return `None` from a method it
/// has no term for, and a caller could not tell that from "this slot holds
/// nothing".
///
/// It is also what stops [`SameForEveryTask`] from having to answer: a fixture
/// wrapping one [`RefResolver`] has no storage term at all, and a trait it does
/// not implement is a compile error rather than a `None` that reads as an empty
/// slot.
///
/// # `None` is "no key", never "no ordering"
///
/// A caller that gets `None` still has an access to build. What it has lost is
/// the *key*, so the honest access is [`crate::access::AccessKey::DomainOnly`] —
/// ordering from the submission domain alone — and not a dropped access, which
/// would order the operation against nothing at all.
pub trait ResourceStorage {
    /// The key an access to this resource is built on, or `None` when the task
    /// has no namespace, the name is not live in it, or this holder has no
    /// storage term for it.
    fn storage(
        &self,
        task: crate::identity::TaskId,
        resource: ResourceId,
    ) -> Option<crate::access::ResourceKey>;
}

/// Which backing a guest mapping's surface currently occupies.
///
/// **A separate trait from [`RefResolver`] because it answers about a separate
/// namespace.** A mapping id and an object-list ref arrive as `u32`s that
/// overlap numerically and name unrelated things — see
/// [`crate::identity::MappingId`] — so a resolver that answered both from one
/// method would let a caller ask the wrong question and get a plausible answer.
/// The mapper owns this one; the object list owns the other.
pub trait MappingResolver {
    /// The backing a mapping's surface is resolved to, or `None` when the
    /// mapping names no live surface.
    fn backing(&self, mapping: crate::identity::MappingId) -> Option<crate::access::BackingId>;
}

/// Why a record could not become an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveRefusal {
    /// The bytes never became a record.
    Decode(DecodeRefusal),
    /// A ref names no live object.
    ///
    /// Carries the guest's number, because the number is what a log line has to
    /// contain for anyone to find which object the guest thought it had.
    UnknownRef { object_ref: u32 },
    /// A field's value names nothing the API defines.
    ///
    /// Distinct from the decoder's refusal of the same shape: this one is
    /// raised by resolution, on a field the decoder carried verbatim because
    /// what the value *means* is a model question. The pass descriptor's store
    /// action is the case — the record shape represents four of them, and a
    /// fifth folded onto its nearest neighbour is either a discarded frame or a
    /// resolve that never happens.
    UndefinedOrdinal { field: &'static str, value: u32 },
    /// A counted list is longer than an arena window can name.
    ///
    /// A window is two `u32`, so a transaction cannot hold more than `u32::MAX`
    /// entries across all its lists. Refusing is the honest answer: a truncated
    /// list is a barrier that orders less than the guest asked for, which is
    /// worse than one that did not happen.
    ArenaOverflow { wanted: usize },
}

impl ResolveRefusal {
    /// The stable reason string for the failure channel.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Decode(inner) => inner.reason(),
            Self::UnknownRef { .. } => "resolve_ref_names_no_object",
            Self::UndefinedOrdinal { .. } => "resolve_field_ordinal_undefined",
            Self::ArenaOverflow { .. } => "resolve_list_exceeds_arena_window",
        }
    }
}

impl From<DecodeRefusal> for ResolveRefusal {
    fn from(inner: DecodeRefusal) -> Self {
        Self::Decode(inner)
    }
}

/// Resolve one ref, refusing by name when it holds nothing.
fn one(resolver: &impl RefResolver, object_ref: u32) -> Result<ResourceId, ResolveRefusal> {
    resolver
        .resource(object_ref)
        .ok_or(ResolveRefusal::UnknownRef { object_ref })
}

/// Append a resolved list to the arena and name the window.
///
/// The whole list resolves before anything is appended. A list with a dead ref
/// in the middle would otherwise leave half of itself in the arena behind a
/// window nobody names — harmless today and exactly the kind of thing that
/// stops being harmless when an arena is reused.
///
/// Entries resolve through [`one`] rather than [`bound`], even though they
/// arrive in the same `RefBind` shape a texture bind uses. A residency list has
/// no slot to clear: an entry is a resource the following work will touch, so
/// there is nothing for a nil entry to mean and reading one as an unbind would
/// silently shorten the list a barrier orders over.
fn append_refs(
    arenas: &mut ExecArenas,
    resolver: &impl RefResolver,
    refs: &[RefBind],
) -> Result<ResourceSpan, ResolveRefusal> {
    let (start, len) = window(arenas.resources.len(), refs.len())?;
    arenas.resources.reserve(refs.len());
    let mark = arenas.resources.len();
    for entry in refs {
        match one(resolver, entry.object_ref.get()) {
            Ok(id) => arenas.resources.push(id),
            Err(refusal) => {
                arenas.resources.truncate(mark);
                return Err(refusal);
            }
        }
    }
    Ok(ResourceSpan { start, len })
}

fn origin(o: RecordOrigin) -> Origin3 {
    Origin3 {
        x: o.x,
        y: o.y,
        z: o.z,
    }
}

fn size(s: RecordSize) -> Size3 {
    Size3 {
        width: s.width,
        height: s.height,
        depth: s.depth,
    }
}

fn endpoint(
    resolver: &impl RefResolver,
    e: TextureEndpoint,
) -> Result<TexturePoint, ResolveRefusal> {
    Ok(TexturePoint {
        texture: one(resolver, e.texture_ref)?,
        slice: e.slice,
        level: e.level,
        origin: origin(e.origin),
    })
}

/// Resolve a transfer record.
pub fn blit(record: &BlitRecord, resolver: &impl RefResolver) -> Result<BlitOp, ResolveRefusal> {
    Ok(match *record {
        BlitRecord::BufferToBuffer(record::BufferToBuffer {
            source_ref,
            source_offset,
            dest_ref,
            dest_offset,
            size: bytes,
        }) => BlitOp::BufferToBuffer {
            source: one(resolver, source_ref)?,
            source_offset,
            dest: one(resolver, dest_ref)?,
            dest_offset,
            size: bytes,
        },
        BlitRecord::BufferToTexture(record::BufferToTexture {
            source_ref,
            source_offset,
            bytes_per_row,
            bytes_per_image,
            size: extent,
            dest,
            options,
        }) => BlitOp::BufferToTexture {
            source: one(resolver, source_ref)?,
            source_offset,
            source_pitch: ImagePitch {
                bytes_per_row,
                bytes_per_image,
            },
            size: size(extent),
            dest: endpoint(resolver, dest)?,
            options: BlitOptions(options),
        },
        BlitRecord::TextureToBuffer(record::TextureToBuffer {
            source,
            size: extent,
            dest_ref,
            dest_offset,
            bytes_per_row,
            bytes_per_image,
            options,
        }) => BlitOp::TextureToBuffer {
            source: endpoint(resolver, source)?,
            size: size(extent),
            dest: one(resolver, dest_ref)?,
            dest_offset,
            dest_pitch: ImagePitch {
                bytes_per_row,
                bytes_per_image,
            },
            options: BlitOptions(u32::from(options)),
        },
        BlitRecord::TextureRegion(record::TextureRegion {
            source,
            dest,
            size: extent,
            options,
        }) => BlitOp::TextureRegion {
            source: endpoint(resolver, source)?,
            dest: endpoint(resolver, dest)?,
            size: size(extent),
            options: BlitOptions(options),
        },
        BlitRecord::TextureSlices(record::TextureSlices {
            source_ref,
            source_slice,
            source_level,
            dest_ref,
            dest_slice,
            dest_level,
            slice_count,
            level_count,
        }) => BlitOp::TextureSlices {
            source: SpanOrigin {
                texture: one(resolver, source_ref)?,
                base_slice: source_slice,
                base_level: source_level,
            },
            dest: SpanOrigin {
                texture: one(resolver, dest_ref)?,
                base_slice: dest_slice,
                base_level: dest_level,
            },
            // The record's own single count of each, reaching both ends
            // because there is one of it. See [`SpanOrigin`].
            slice_count,
            level_count,
        },
        BlitRecord::FillBuffer(record::FillBuffer {
            buffer_ref,
            location,
            length,
            pattern,
        }) => BlitOp::FillBuffer {
            dest: BufferSpan {
                buffer: one(resolver, buffer_ref)?,
                offset: location,
                length,
            },
            pattern: match pattern {
                RecordFill::Byte(b) => FillPattern::Byte(b),
                RecordFill::Pattern4(w) => FillPattern::Pattern4(w),
            },
        },
        BlitRecord::GenerateMipmaps(record::GenerateMipmaps { texture_ref }) => {
            BlitOp::GenerateMipmaps {
                texture: one(resolver, texture_ref)?,
            }
        }
    })
}

/// Resolve a content-representation record.
pub fn resource_state(
    record: &ResourceStateRecord,
    resolver: &impl RefResolver,
) -> Result<ResourceStateOp, ResolveRefusal> {
    let target = match record.target {
        RecordTarget::WholeResource { object_ref } => ResourceStateTarget::Resource {
            resource: one(resolver, object_ref)?,
            subresource: None,
        },
        RecordTarget::SliceLevel {
            texture_ref,
            slice,
            level,
        } => ResourceStateTarget::Resource {
            resource: one(resolver, texture_ref)?,
            subresource: Some(SliceLevel { slice, level }),
        },
        RecordTarget::Encoder => ResourceStateTarget::Encoder,
    };
    Ok(ResourceStateOp {
        directive: record.directive,
        target,
    })
}

/// Resolve an indirect-command record.
pub fn icb(record: &IcbRecord, resolver: &impl RefResolver) -> Result<IcbOp, ResolveRefusal> {
    Ok(match *record {
        IcbRecord::ExecuteRange { icb_ref, commands } => IcbOp::ExecuteRange {
            icb: one(resolver, icb_ref)?,
            commands: CommandRange {
                location: commands.location,
                length: commands.length,
            },
        },
        IcbRecord::ExecuteIndirect { icb_ref, arguments } => IcbOp::ExecuteIndirect {
            icb: one(resolver, icb_ref)?,
            arguments: crate::bind::IndirectSource {
                buffer: one(resolver, arguments.buffer_ref)?,
                offset: arguments.offset,
            },
        },
        IcbRecord::Optimize { icb_ref, commands } => IcbOp::Optimize {
            icb: one(resolver, icb_ref)?,
            commands: CommandRange {
                location: commands.location,
                length: commands.length,
            },
        },
    })
}

/// Resolve a fence record.
pub fn fence(record: &FenceRecord, resolver: &impl RefResolver) -> Result<FenceOp, ResolveRefusal> {
    Ok(FenceOp {
        kind: record.kind,
        fence: one(resolver, record.fence_ref)?,
        stages: record.stages,
    })
}

/// Resolve an event record.
pub fn event(record: &EventRecord, resolver: &impl RefResolver) -> Result<EventOp, ResolveRefusal> {
    Ok(EventOp {
        kind: record.kind,
        event: one(resolver, record.event_ref)?,
        value: record.value,
    })
}

/// Resolve a barrier record, appending any list it names to the arena.
pub fn barrier(
    record: &BarrierRecord<'_>,
    resolver: &impl RefResolver,
    arenas: &mut ExecArenas,
) -> Result<BarrierOp, ResolveRefusal> {
    Ok(match *record {
        BarrierRecord::Resources {
            refs,
            after_stages,
            before_stages,
        } => BarrierOp {
            target: BarrierTarget::Resources(append_refs(arenas, resolver, refs)?),
            after_stages,
            before_stages,
        },
        BarrierRecord::Scope {
            scope,
            after_stages,
            before_stages,
            ..
        } => BarrierOp {
            target: BarrierTarget::Scope(scope),
            after_stages,
            before_stages,
        },
        BarrierRecord::Texture => BarrierOp {
            target: BarrierTarget::Texture,
            after_stages: None,
            before_stages: None,
        },
    })
}

/// Resolve any ordering record.
pub fn sync(
    record: &SyncRecord<'_>,
    resolver: &impl RefResolver,
    arenas: &mut ExecArenas,
) -> Result<SyncResolved, ResolveRefusal> {
    Ok(match record {
        SyncRecord::Fence(r) => SyncResolved::Fence(fence(r, resolver)?),
        SyncRecord::Event(r) => SyncResolved::Event(event(r, resolver)?),
        SyncRecord::Barrier(r) => SyncResolved::Barrier(barrier(r, resolver, arenas)?),
    })
}

/// The three ordering operations, which are three classes rather than one.
///
/// Returned as a small enum rather than folded into one class: a fence, an
/// event and a barrier order different things over different scopes, and the
/// census counts them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncResolved {
    Fence(FenceOp),
    Event(EventOp),
    Barrier(BarrierOp),
}

/// The ref a bind entry uses to mean "nothing".
///
/// A guest unbinds by naming no object, and the serializer writes zero for a
/// nil argument — the pass descriptor's `visibilityResultBuffer` is the record
/// that shows it, written as zero when the property is unset. So zero in a bind
/// entry is an unbind, and every other value is a name that must resolve.
///
/// This is the one place the two are told apart. Resolving zero would either
/// refuse a legal unbind or, if slot zero ever held an object, bind an
/// unrelated one — and `BufferBinding::buffer` is an `Option` precisely so the
/// difference survives into the model.
const NIL_REF: u32 = 0;

/// Resolve a bind entry's ref, where zero means the guest unbound the slot.
fn bound(
    resolver: &impl RefResolver,
    object_ref: u32,
) -> Result<Option<ResourceId>, ResolveRefusal> {
    if object_ref == NIL_REF {
        return Ok(None);
    }
    one(resolver, object_ref).map(Some)
}

/// The window `count` entries appended at `at` would occupy.
fn window(at: usize, count: usize) -> Result<(u32, u32), ResolveRefusal> {
    let overflow = || ResolveRefusal::ArenaOverflow { wanted: count };
    let start = u32::try_from(at).map_err(|_| overflow())?;
    let len = u32::try_from(count).map_err(|_| overflow())?;
    start.checked_add(len).ok_or_else(overflow)?;
    Ok((start, len))
}

/// Append resolved buffer bindings and name the window.
///
/// `stride` is the operation's, not the entry's: the record's opcode decides
/// the entry shape for every entry it carries, so a strided record's entries
/// all carry a stride and a plain record's carry none.
fn append_buffer_binds(
    arenas: &mut ExecArenas,
    resolver: &impl RefResolver,
    entries: &[BufferBind],
) -> Result<BindSpan, ResolveRefusal> {
    let (start, len) = window(arenas.buffer_bindings.len(), entries.len())?;
    let mark = arenas.buffer_bindings.len();
    arenas.buffer_bindings.reserve(entries.len());
    for entry in entries {
        match bound(resolver, entry.buffer_ref.get()) {
            Ok(buffer) => arenas.buffer_bindings.push(BufferBinding {
                buffer,
                offset: entry.offset.get(),
                stride: None,
            }),
            Err(refusal) => {
                arenas.buffer_bindings.truncate(mark);
                return Err(refusal);
            }
        }
    }
    Ok(BindSpan { start, len })
}

/// The same for the entries that carry an attribute stride.
fn append_stride_binds(
    arenas: &mut ExecArenas,
    resolver: &impl RefResolver,
    entries: &[BufferStrideBind],
) -> Result<BindSpan, ResolveRefusal> {
    let (start, len) = window(arenas.buffer_bindings.len(), entries.len())?;
    let mark = arenas.buffer_bindings.len();
    arenas.buffer_bindings.reserve(entries.len());
    for entry in entries {
        match bound(resolver, entry.buffer_ref.get()) {
            Ok(buffer) => arenas.buffer_bindings.push(BufferBinding {
                buffer,
                offset: entry.offset.get(),
                stride: Some(entry.attribute_stride.get()),
            }),
            Err(refusal) => {
                arenas.buffer_bindings.truncate(mark);
                return Err(refusal);
            }
        }
    }
    Ok(BindSpan { start, len })
}

/// Append resolved texture or sampler bindings and name the window.
fn append_object_binds(
    arenas: &mut ExecArenas,
    resolver: &impl RefResolver,
    entries: &[RefBind],
) -> Result<BindSpan, ResolveRefusal> {
    let (start, len) = window(arenas.object_bindings.len(), entries.len())?;
    let mark = arenas.object_bindings.len();
    arenas.object_bindings.reserve(entries.len());
    for entry in entries {
        match bound(resolver, entry.object_ref.get()) {
            Ok(object) => arenas.object_bindings.push(ObjectBinding {
                object,
                lod_clamps: None,
            }),
            Err(refusal) => {
                arenas.object_bindings.truncate(mark);
                return Err(refusal);
            }
        }
    }
    Ok(BindSpan { start, len })
}

/// The same for sampler entries that carry their own clamps.
///
/// The clamps are per entry rather than per record — which is what the plural
/// fixtures established — so they are read off each entry here and not off the
/// first one.
fn append_sampler_lod_binds(
    arenas: &mut ExecArenas,
    resolver: &impl RefResolver,
    entries: &[SamplerLodBind],
) -> Result<BindSpan, ResolveRefusal> {
    let (start, len) = window(arenas.object_bindings.len(), entries.len())?;
    let mark = arenas.object_bindings.len();
    arenas.object_bindings.reserve(entries.len());
    for entry in entries {
        match bound(resolver, entry.sampler_ref.get()) {
            Ok(object) => arenas.object_bindings.push(ObjectBinding {
                object,
                lod_clamps: Some((
                    LodClamp::from_f32(entry.lod_min_clamp.get()),
                    LodClamp::from_f32(entry.lod_max_clamp.get()),
                )),
            }),
            Err(refusal) => {
                arenas.object_bindings.truncate(mark);
                return Err(refusal);
            }
        }
    }
    Ok(BindSpan { start, len })
}

fn compute_extent(e: RecordExtent) -> ComputeExtent {
    ComputeExtent {
        width: e.width,
        height: e.height,
        depth: e.depth,
    }
}

fn indirect(
    resolver: &impl RefResolver,
    source: ComputeIndirect,
) -> Result<IndirectSource, ResolveRefusal> {
    Ok(IndirectSource {
        buffer: one(resolver, source.buffer_ref)?,
        offset: source.offset,
    })
}

/// Resolve a compute record.
pub fn compute(
    record: &ComputeRecord<'_>,
    resolver: &impl RefResolver,
    arenas: &mut ExecArenas,
) -> Result<ComputeOp, ResolveRefusal> {
    Ok(match *record {
        ComputeRecord::BindBuffers(protocol_compute::BindBuffers { first, entries }) => {
            ComputeOp::BindBuffers {
                first,
                entries: append_buffer_binds(arenas, resolver, entries)?,
            }
        }
        ComputeRecord::BindBuffersWithStride(protocol_compute::BindBuffersWithStride {
            first,
            entries,
        }) => ComputeOp::BindBuffersWithStride {
            first,
            entries: append_stride_binds(arenas, resolver, entries)?,
        },
        ComputeRecord::BindTextures(protocol_compute::BindTextures { first, entries }) => {
            ComputeOp::BindTextures {
                first,
                entries: append_object_binds(arenas, resolver, entries)?,
            }
        }
        ComputeRecord::BindSamplers(protocol_compute::BindSamplers { first, entries }) => {
            ComputeOp::BindSamplers {
                first,
                entries: append_object_binds(arenas, resolver, entries)?,
            }
        }
        ComputeRecord::BindSamplersWithLod(protocol_compute::BindSamplersWithLod {
            first,
            entries,
        }) => ComputeOp::BindSamplersWithLod {
            first,
            entries: append_sampler_lod_binds(arenas, resolver, entries)?,
        },
        ComputeRecord::RebindBufferOffset(protocol_compute::RebindBufferOffset {
            index,
            offset,
            stride,
        }) => ComputeOp::RebindBufferOffset {
            index,
            offset,
            stride,
        },
        ComputeRecord::SetPipeline(protocol_compute::SetPipeline { pipeline_ref }) => {
            ComputeOp::SetPipeline {
                pipeline: one(resolver, pipeline_ref)?,
            }
        }
        ComputeRecord::SetStageInRegion(protocol_compute::SetStageInRegion { origin, size }) => {
            ComputeOp::SetStageInRegion {
                origin: ComputeOrigin {
                    x: origin.x,
                    y: origin.y,
                    z: origin.z,
                },
                size: compute_extent(size),
            }
        }
        ComputeRecord::SetStageInRegionIndirect(protocol_compute::SetStageInRegionIndirect {
            source,
        }) => ComputeOp::SetStageInRegionIndirect {
            source: indirect(resolver, source)?,
        },
        ComputeRecord::SetThreadgroupMemory(protocol_compute::SetThreadgroupMemory {
            index,
            length,
        }) => ComputeOp::SetThreadgroupMemory { index, length },
        ComputeRecord::SetImageblockSize(protocol_compute::SetImageblockSize { width, height }) => {
            ComputeOp::SetImageblockSize { width, height }
        }
        ComputeRecord::WriteDescriptor(protocol_compute::WriteDescriptor { dispatch_type }) => {
            ComputeOp::WriteDescriptor { dispatch_type }
        }
        ComputeRecord::Dispatch(d) => ComputeOp::Dispatch(match d {
            DispatchRecord::Threadgroups(protocol_compute::Threadgroups {
                groups,
                threads_per_group,
            }) => DispatchOp::Threadgroups {
                groups: compute_extent(groups),
                threads_per_group: compute_extent(threads_per_group),
            },
            DispatchRecord::Threads(protocol_compute::Threads {
                threads,
                threads_per_group,
            }) => DispatchOp::Threads {
                threads: compute_extent(threads),
                threads_per_group: compute_extent(threads_per_group),
            },
            DispatchRecord::ThreadgroupsIndirect(protocol_compute::ThreadgroupsIndirect {
                source,
                threads_per_group,
            }) => DispatchOp::ThreadgroupsIndirect {
                source: indirect(resolver, source)?,
                threads_per_group: compute_extent(threads_per_group),
            },
            DispatchRecord::ThreadsIndirect(protocol_compute::ThreadsIndirect { source }) => {
                DispatchOp::ThreadsIndirect {
                    source: indirect(resolver, source)?,
                }
            }
        }),
    })
}

/// Resolve a viewport arena window.
///
/// The wire's viewport is six `f64` and the model's is six bit patterns. The
/// conversion is a `to_bits`, and it is here rather than in the model's type
/// because the model's reason for holding bits is comparison: a state table has
/// to answer "is this the viewport that is already set", and a NaN depth bound
/// must stay equal to itself.
fn append_viewports(
    arenas: &mut ExecArenas,
    ports: &[WireViewport],
) -> Result<StateSpan, ResolveRefusal> {
    let (start, len) = window(arenas.viewports.len(), ports.len())?;
    arenas.viewports.reserve(ports.len());
    for port in ports {
        arenas.viewports.push(Viewport {
            origin_x_bits: port.origin_x.get().to_bits(),
            origin_y_bits: port.origin_y.get().to_bits(),
            width_bits: port.width.get().to_bits(),
            height_bits: port.height.get().to_bits(),
            z_near_bits: port.znear.get().to_bits(),
            z_far_bits: port.zfar.get().to_bits(),
        });
    }
    Ok(StateSpan { start, len })
}

/// Resolve a scissor arena window. Integers on both sides; nothing to convert
/// but the endianness the view already handled.
fn append_scissors(
    arenas: &mut ExecArenas,
    rects: &[WireScissorRect],
) -> Result<StateSpan, ResolveRefusal> {
    let (start, len) = window(arenas.scissors.len(), rects.len())?;
    arenas.scissors.reserve(rects.len());
    for rect in rects {
        arenas.scissors.push(ScissorRect {
            x: rect.x.get(),
            y: rect.y.get(),
            width: rect.width.get(),
            height: rect.height.get(),
        });
    }
    Ok(StateSpan { start, len })
}

/// Resolve one attachment slot out of its wire prefix.
///
/// An unattached slot is one whose texture ref is zero — the record is a fixed
/// shape and carries all eight colour slots whether the guest filled them or
/// not, so "absent" is spelled the same way an unbound slot is. A store action
/// outside the four the record shape represents refuses rather than folding
/// onto a neighbour: guessed wrong it is either a discarded frame or a resolve
/// that never happens.
fn attachment(
    resolver: &impl RefResolver,
    slot: AttachmentSlot,
    prefix: &AttachmentPrefix,
    clear_bits: [u64; 4],
) -> Result<Attachment, ResolveRefusal> {
    // Refused rather than folded, for the reason the two beside it give. The
    // fold `LoadAction::from_declared` performs is written for a caller with no
    // refusal channel; this one has one, and a value outside the closed set
    // here is a corrupt record or a wrong offset. Folding it to `DontCare`
    // silently composites onto whatever was in the attachment.
    let raw_load = prefix.load_action.get();
    let load = LoadAction::parse(raw_load).ok_or(ResolveRefusal::UndefinedOrdinal {
        field: "load_action",
        value: u32::from(raw_load),
    })?;
    let raw_store = prefix.store_action.get();
    let store = StoreAction::parse(raw_store).ok_or(ResolveRefusal::UndefinedOrdinal {
        field: "store_action",
        value: u32::from(raw_store),
    })?;
    // The word beside the store action, which nothing read until now. An
    // undeclared value refuses for the reason the store action's does: the set
    // has one flag, so a value outside it is a corrupt record or a wrong
    // offset, and folding it onto the flag would claim the guest asked for
    // programmable sample positions it never asked for.
    let raw_options = prefix.store_action_options.get();
    let store_options = StoreActionOptions::parse(u64::from(raw_options)).ok_or(
        ResolveRefusal::UndefinedOrdinal {
            field: "store_action_options",
            value: u32::from(raw_options),
        },
    )?;
    Ok(Attachment {
        slot,
        texture: bound(resolver, prefix.texture_ref.get())?,
        level: prefix.level.get(),
        slice: prefix.slice.get(),
        depth_plane: prefix.depth_plane.get(),
        resolve_texture: bound(resolver, prefix.resolve_texture_ref.get())?,
        resolve_level: prefix.resolve_level.get(),
        resolve_slice: prefix.resolve_slice.get(),
        resolve_depth_plane: prefix.resolve_depth_plane.get(),
        load,
        store,
        store_options,
        clear_bits,
    })
}

/// Resolve a render-pass descriptor.
///
/// Every slot is resolved, attached or not. The record carries eight colour
/// slots as a fixed shape, and a model that skipped the empty ones would have
/// to invent them again for anything that iterates slots by index.
pub fn pass_descriptor(
    body: &RenderPassBody,
    resolver: &impl RefResolver,
) -> Result<PassDescriptor, ResolveRefusal> {
    let mut descriptor = PassDescriptor::empty();
    for (index, wire) in body.color.iter().enumerate() {
        descriptor.color[index] = attachment(
            resolver,
            AttachmentSlot::Color(index as u8),
            &wire.prefix,
            wire.clear_color_bits.map(|bits| bits.get()),
        )?;
    }
    descriptor.depth = attachment(
        resolver,
        AttachmentSlot::Depth,
        &body.depth.prefix,
        [body.depth.clear_depth_bits.get(), 0, 0, 0],
    )?;
    descriptor.stencil = attachment(
        resolver,
        AttachmentSlot::Stencil,
        &body.stencil.prefix,
        [u64::from(body.stencil.clear_stencil.get()), 0, 0, 0],
    )?;
    // The two words beside the depth and stencil clear values, which nothing
    // read until now. An undeclared ordinal refuses for the reason the store
    // action's does: both sets are closed, so a value outside one is a corrupt
    // record or a wrong offset, and folding it onto a neighbour resolves at a
    // filter the guest did not ask for.
    let raw_depth_filter = body.depth.resolve_filter.get();
    descriptor.depth_resolve_filter =
        DepthResolveFilter::parse(raw_depth_filter).ok_or(ResolveRefusal::UndefinedOrdinal {
            field: "depth_resolve_filter",
            value: u32::from(raw_depth_filter),
        })?;
    let raw_stencil_filter = body.stencil.resolve_filter.get();
    descriptor.stencil_resolve_filter = StencilResolveFilter::parse(raw_stencil_filter).ok_or(
        ResolveRefusal::UndefinedOrdinal {
            field: "stencil_resolve_filter",
            value: u32::from(raw_stencil_filter),
        },
    )?;
    descriptor.visibility_result_buffer = bound(resolver, body.visibility_result_buffer_ref.get())?
        .map(|buffer| VisibilityResultBuffer { buffer });
    descriptor.extent = RenderTargetExtent {
        width: body.render_target_width.get(),
        height: body.render_target_height.get(),
        array_length: body.render_target_array_length.get(),
    };
    Ok(descriptor)
}

/// Resolve a render record.
pub fn render(
    record: &RenderRecord<'_>,
    resolver: &impl RefResolver,
    arenas: &mut ExecArenas,
) -> Result<RenderOp, ResolveRefusal> {
    Ok(match *record {
        RenderRecord::Draw(d) => RenderOp::Draw(draw(&d, resolver)?),
        RenderRecord::BindBuffers(protocol_render::BindBuffers {
            stage,
            first,
            entries,
        }) => RenderOp::BindBuffers {
            stage,
            first,
            entries: append_buffer_binds(arenas, resolver, entries)?,
        },
        RenderRecord::BindBuffersWithStride(protocol_render::BindBuffersWithStride {
            first,
            entries,
        }) => RenderOp::BindBuffersWithStride {
            first,
            entries: append_stride_binds(arenas, resolver, entries)?,
        },
        RenderRecord::BindTextures(protocol_render::BindTextures {
            stage,
            first,
            entries,
        }) => RenderOp::BindTextures {
            stage,
            first,
            entries: append_object_binds(arenas, resolver, entries)?,
        },
        RenderRecord::BindSamplers(protocol_render::BindSamplers {
            stage,
            first,
            entries,
        }) => RenderOp::BindSamplers {
            stage,
            first,
            entries: append_object_binds(arenas, resolver, entries)?,
        },
        RenderRecord::BindSamplersWithLod(protocol_render::BindSamplersWithLod {
            stage,
            first,
            entries,
        }) => RenderOp::BindSamplersWithLod {
            stage,
            first,
            entries: append_sampler_lod_binds(arenas, resolver, entries)?,
        },
        RenderRecord::RebindBufferOffset(protocol_render::RebindBufferOffset {
            stage,
            index,
            offset,
            stride,
        }) => RenderOp::RebindBufferOffset {
            stage,
            index,
            offset,
            stride,
        },
        RenderRecord::SetPipeline(protocol_render::SetPipeline { pipeline_ref }) => {
            RenderOp::SetPipeline {
                pipeline: one(resolver, pipeline_ref)?,
            }
        }
        RenderRecord::SetDepthStencilState(protocol_render::SetDepthStencilState { state_ref }) => {
            RenderOp::SetDepthStencilState {
                state: one(resolver, state_ref)?,
            }
        }
        RenderRecord::WriteDescriptor(protocol_render::WriteDescriptor { descriptor }) => {
            let resolved = pass_descriptor(descriptor, resolver)?;
            let slot = u32::try_from(arenas.pass_descriptors.len())
                .map_err(|_| ResolveRefusal::ArenaOverflow { wanted: 1 })?;
            arenas.pass_descriptors.push(resolved);
            RenderOp::WriteDescriptor {
                descriptor: PassDescriptorSlot(slot),
            }
        }
        RenderRecord::SetViewports(ports) => {
            RenderOp::SetViewports(append_viewports(arenas, ports)?)
        }
        RenderRecord::SetScissorRects(rects) => {
            RenderOp::SetScissorRects(append_scissors(arenas, rects)?)
        }
        RenderRecord::SetCullMode(r) => RenderOp::SetCullMode(r.mode),
        RenderRecord::SetFrontFacingWinding(r) => RenderOp::SetFrontFacingWinding(r.winding),
        RenderRecord::SetDepthClipMode(r) => RenderOp::SetDepthClipMode(r.mode),
        RenderRecord::SetTriangleFillMode(r) => RenderOp::SetTriangleFillMode(r.mode),
        RenderRecord::SetDepthBias(protocol_render::SetDepthBias {
            bias_bits,
            slope_scale_bits,
            clamp_bits,
        }) => RenderOp::SetDepthBias {
            bias: FloatBits(bias_bits),
            slope_scale: FloatBits(slope_scale_bits),
            clamp: FloatBits(clamp_bits),
        },
        RenderRecord::SetLineWidth(protocol_render::SetLineWidth { width_bits }) => {
            RenderOp::SetLineWidth(FloatBits(width_bits))
        }
        RenderRecord::SetBlendColor(protocol_render::SetBlendColor {
            red_bits,
            green_bits,
            blue_bits,
            alpha_bits,
        }) => RenderOp::SetBlendColor {
            red: FloatBits(red_bits),
            green: FloatBits(green_bits),
            blue: FloatBits(blue_bits),
            alpha: FloatBits(alpha_bits),
        },
        RenderRecord::SetStencilReference(protocol_render::SetStencilReference { front, back }) => {
            RenderOp::SetStencilReference { front, back }
        }
        RenderRecord::SetStoreAction(protocol_render::SetStoreAction { target, action }) => {
            RenderOp::SetStoreAction { target, action }
        }
        RenderRecord::SetVisibilityResultMode(protocol_render::SetVisibilityResultMode {
            mode,
            offset,
        }) => RenderOp::SetVisibilityResultMode { mode, offset },
    })
}

fn index_source(
    resolver: &impl RefResolver,
    index: RecordIndexRef,
) -> Result<IndexSource, ResolveRefusal> {
    Ok(IndexSource {
        buffer: one(resolver, index.buffer_ref)?,
        offset: index.offset,
        index_type: index.index_type,
    })
}

fn draw_arguments(
    resolver: &impl RefResolver,
    arguments: RenderIndirect,
) -> Result<IndirectSource, ResolveRefusal> {
    Ok(IndirectSource {
        buffer: one(resolver, arguments.buffer_ref)?,
        offset: arguments.offset,
    })
}

fn instancing(i: RecordInstancing) -> Instancing {
    Instancing {
        count: i.count,
        base: i.base,
    }
}

/// Resolve a draw record.
pub fn draw(record: &DrawRecord, resolver: &impl RefResolver) -> Result<DrawOp, ResolveRefusal> {
    Ok(match *record {
        DrawRecord::Primitives(protocol_render::Primitives {
            primitive,
            vertex_start,
            vertex_count,
            instances,
        }) => DrawOp::Primitives {
            primitive: PrimitiveType(primitive),
            vertex_start,
            vertex_count,
            instances: instancing(instances),
        },
        DrawRecord::Indexed(protocol_render::Indexed {
            primitive,
            index,
            index_count,
            instances,
            base_vertex,
        }) => DrawOp::Indexed {
            primitive: PrimitiveType(primitive),
            index: index_source(resolver, index)?,
            index_count,
            instances: instancing(instances),
            base_vertex,
        },
        DrawRecord::PrimitivesIndirect(protocol_render::PrimitivesIndirect {
            primitive,
            arguments,
        }) => DrawOp::PrimitivesIndirect {
            primitive: PrimitiveType(primitive),
            arguments: draw_arguments(resolver, arguments)?,
        },
        DrawRecord::IndexedIndirect(protocol_render::IndexedIndirect {
            primitive,
            index,
            arguments,
        }) => DrawOp::IndexedIndirect {
            primitive: PrimitiveType(primitive),
            index: index_source(resolver, index)?,
            arguments: draw_arguments(resolver, arguments)?,
        },
    })
}

/// Decode and resolve one record, choosing the decoder from the ledger.
///
/// # The ledger picks the decoder, so an unjudged opcode never reaches one
///
/// The class an opcode belongs to is the ledger's answer, and asking it first
/// means the dispatch and the admission are the same step. An opcode with no
/// row, a row the ledger has not settled, and a row it settled as a refusal all
/// stop here with their own reason — none of them reaches a decoder that might
/// have lifted a record the model has no use for.
///
/// That is also why there is no `rail`-free form: three encoders number their
/// records independently, and the same opcode is a different record on each.
///
/// # An encoder boundary is not a record
///
/// Segment framing is the stream's own vocabulary and it reaches
/// [`crate::exec::ExecBuilder::begin_segment`] rather than this. A boundary
/// arriving here is a caller mistake rather than a guest one, and it refuses
/// with a reason that says so instead of being decoded as whatever record its
/// opcode field happens to hold.
pub fn operation(
    rail: Rail,
    op: &Op<'_>,
    resolver: &impl RefResolver,
    arenas: &mut ExecArenas,
) -> Result<ResolvedOperation, ResolveRefusal> {
    let opcode = op.opcode();
    let unjudged = || ResolveRefusal::Decode(decode::no_record(rail, opcode));
    let Some(row) = closure::find(rail, opcode) else {
        return Err(unjudged());
    };
    let Some(OperationHome::Stream(class)) = classify(row) else {
        return Err(unjudged());
    };
    Ok(match class {
        OperationClass::Render => {
            ResolvedOperation::Render(render(&decode::render::decode(op)?, resolver, arenas)?)
        }
        OperationClass::Compute => {
            ResolvedOperation::Compute(compute(&decode::compute::decode(op)?, resolver, arenas)?)
        }
        OperationClass::Blit => {
            ResolvedOperation::Blit(blit(&decode::blit::decode(op)?, resolver)?)
        }
        OperationClass::Event | OperationClass::Fence | OperationClass::Barrier => {
            match sync(&decode::sync::decode(rail, op)?, resolver, arenas)? {
                SyncResolved::Fence(o) => ResolvedOperation::Fence(o),
                SyncResolved::Event(o) => ResolvedOperation::Event(o),
                SyncResolved::Barrier(o) => ResolvedOperation::Barrier(o),
            }
        }
        OperationClass::ResourceState => ResolvedOperation::ResourceState(resource_state(
            &decode::resource_state::decode(rail, op)?,
            resolver,
        )?),
        OperationClass::IndirectCommand => {
            ResolvedOperation::IndirectCommand(icb(&decode::icb::decode(rail, op)?, resolver)?)
        }
        // A boundary is the segment, an info query has a reply destination
        // rather than an encoder, and the completion class has no record at all.
        // None of the three is a stream record to decode here.
        OperationClass::EncoderBoundary
        | OperationClass::InfoQuery
        | OperationClass::CompletionEffect => return Err(unjudged()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};
    use reims_vgpu_protocol::closure::Rail;
    use reims_vgpu_protocol::residency::RenderStages;
    use reims_vgpu_protocol::sync::{BarrierScope, EventKind, FenceKind};

    use reims_vgpu_protocol::decode::{U16le, U32le, U64le};
    use reims_vgpu_protocol::render::{IndexType, ShaderStage};

    fn u16le(v: u16) -> U16le {
        U16le::new(v)
    }

    fn u32le(v: u32) -> U32le {
        U32le::new(v)
    }

    fn u64le(v: u64) -> U64le {
        U64le::new(v)
    }

    fn wire_viewport(
        origin_x: f64,
        origin_y: f64,
        width: f64,
        height: f64,
        znear: f64,
        zfar: f64,
    ) -> WireViewport {
        WireViewport {
            origin_x: reims_vgpu_protocol::decode::F64le::new(origin_x),
            origin_y: reims_vgpu_protocol::decode::F64le::new(origin_y),
            width: reims_vgpu_protocol::decode::F64le::new(width),
            height: reims_vgpu_protocol::decode::F64le::new(height),
            znear: reims_vgpu_protocol::decode::F64le::new(znear),
            zfar: reims_vgpu_protocol::decode::F64le::new(zfar),
        }
    }

    /// A pass descriptor body with nothing attached, which is what the guest
    /// sends before it fills any slot in.
    fn pass_body() -> RenderPassBody {
        fn prefix() -> AttachmentPrefix {
            AttachmentPrefix {
                texture_ref: u32le(0),
                resolve_texture_ref: u32le(0),
                level: u16le(0),
                slice: u16le(0),
                depth_plane: u16le(0),
                resolve_level: u16le(0),
                resolve_slice: u16le(0),
                resolve_depth_plane: u16le(0),
                load_action: u16le(0),
                store_action: u16le(0),
                store_action_options: u16le(0),
                unwritten_above_store_action_options: [0; 2],
            }
        }
        RenderPassBody {
            depth: reims_vgpu_protocol::decode::DepthAttachmentBody {
                prefix: prefix(),
                clear_depth_bits: u64le(0),
                resolve_filter: u16le(0),
                unwritten_above_resolve_filter: [0; 2],
            },
            stencil: reims_vgpu_protocol::decode::StencilAttachmentBody {
                prefix: prefix(),
                clear_stencil: u32le(0),
                resolve_filter: u16le(0),
                unwritten_above_resolve_filter: [0; 2],
            },
            color: core::array::from_fn(|_| reims_vgpu_protocol::decode::ColorAttachmentBody {
                prefix: prefix(),
                clear_color_bits: [u64le(0); 4],
            }),
            visibility_result_buffer_ref: u32le(0),
            render_target_array_length: u64le(0),
            render_target_width: u64le(0),
            render_target_height: u64le(0),
        }
    }

    /// A resolver over a fixed set of live refs.
    struct Live(Vec<u32>);

    impl RefResolver for Live {
        fn resource(&self, object_ref: u32) -> Option<ResourceId> {
            self.0.contains(&object_ref).then_some(ResourceId {
                slot: ObjectListRef(object_ref),
                generation: SlotGeneration(7),
            })
        }
    }

    fn id(object_ref: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(object_ref),
            generation: SlotGeneration(7),
        }
    }

    /// A copy's two endpoints resolve independently, and the operation carries
    /// ids rather than the numbers the guest wrote.
    #[test]
    fn a_transfer_resolves_both_of_its_endpoints() {
        let live = Live(vec![5151, 5252]);
        let record = BlitRecord::BufferToBuffer(record::BufferToBuffer {
            source_ref: 5151,
            source_offset: 0x10,
            dest_ref: 5252,
            dest_offset: 0x20,
            size: 0x30,
        });
        assert_eq!(
            blit(&record, &live),
            Ok(BlitOp::BufferToBuffer {
                source: id(5151),
                source_offset: 0x10,
                dest: id(5252),
                dest_offset: 0x20,
                size: 0x30,
            })
        );
    }

    /// A ref naming nothing refuses with the guest's own number. It does not
    /// become a null id: an operation naming an object that does not exist
    /// would still be ordered against by everything downstream.
    #[test]
    fn a_dead_ref_refuses_by_name_rather_than_resolving_to_nothing() {
        let live = Live(vec![5151]);
        let record = BlitRecord::BufferToBuffer(record::BufferToBuffer {
            source_ref: 5151,
            source_offset: 0,
            dest_ref: 4242,
            dest_offset: 0,
            size: 1,
        });
        assert_eq!(
            blit(&record, &live),
            Err(ResolveRefusal::UnknownRef { object_ref: 4242 })
        );
    }

    /// A whole-resource content record resolves to no subresource, and the
    /// `slice:level:` form to exactly one. `None` is not level zero.
    #[test]
    fn the_two_content_targets_stay_apart_through_resolution() {
        let live = Live(vec![7]);
        let whole = ResourceStateRecord {
            directive: reims_vgpu_protocol::resource_state::ContentDirective::Synchronize,
            target: RecordTarget::WholeResource { object_ref: 7 },
        };
        let sliced = ResourceStateRecord {
            directive: reims_vgpu_protocol::resource_state::ContentDirective::Synchronize,
            target: RecordTarget::SliceLevel {
                texture_ref: 7,
                slice: 0,
                level: 0,
            },
        };
        let whole = resource_state(&whole, &live).expect("resolved");
        let sliced = resource_state(&sliced, &live).expect("resolved");
        assert_eq!(
            whole.target,
            ResourceStateTarget::Resource {
                resource: id(7),
                subresource: None
            }
        );
        assert_ne!(whole.target, sliced.target);
    }

    /// A barrier's list lands in the arena and the operation names a window of
    /// it. The operation is two `u32` whatever the guest sent.
    #[test]
    fn a_barrier_list_goes_to_the_arena_and_the_operation_names_a_window() {
        let live = Live(vec![1, 2, 3]);
        let mut arenas = ExecArenas::default();
        let refs = [1u32, 2, 3].map(|v| RefBind {
            object_ref: reims_vgpu_protocol::decode::U32le::new(v),
        });
        let record = BarrierRecord::Resources {
            refs: &refs,
            after_stages: Some(RenderStages(1)),
            before_stages: Some(RenderStages(2)),
        };
        let op = barrier(&record, &live, &mut arenas).expect("resolved");
        let BarrierTarget::Resources(span) = op.target else {
            panic!("not a resource barrier");
        };
        assert_eq!(span, ResourceSpan { start: 0, len: 3 });
        assert_eq!(&arenas.resources[span.range()], &[id(1), id(2), id(3)]);

        // A second list starts where the first ended, so one arena serves the
        // whole packet.
        let second = barrier(&record, &live, &mut arenas).expect("resolved");
        let BarrierTarget::Resources(span) = second.target else {
            panic!("not a resource barrier");
        };
        assert_eq!(span, ResourceSpan { start: 3, len: 3 });
    }

    /// A list with a dead ref in it leaves nothing behind. Half a list in the
    /// arena behind a window nobody names is harmless right up until the arena
    /// is reused.
    #[test]
    fn a_refused_list_leaves_the_arena_as_it_found_it() {
        let live = Live(vec![1, 3]);
        let mut arenas = ExecArenas::default();
        let refs = [1u32, 2, 3].map(|v| RefBind {
            object_ref: reims_vgpu_protocol::decode::U32le::new(v),
        });
        let record = BarrierRecord::Resources {
            refs: &refs,
            after_stages: None,
            before_stages: None,
        };
        assert_eq!(
            barrier(&record, &live, &mut arenas),
            Err(ResolveRefusal::UnknownRef { object_ref: 2 })
        );
        assert!(arenas.resources.is_empty());
    }

    /// A scope barrier names no list, so it appends nothing.
    #[test]
    fn a_scope_barrier_touches_no_arena() {
        let live = Live(Vec::new());
        let mut arenas = ExecArenas::default();
        let record = BarrierRecord::Scope {
            scope: BarrierScope(BarrierScope::BUFFERS),
            after_stages: None,
            before_stages: None,
            unidentified_u8: None,
        };
        let op = barrier(&record, &live, &mut arenas).expect("resolved");
        assert_eq!(
            op.target,
            BarrierTarget::Scope(BarrierScope(BarrierScope::BUFFERS))
        );
        assert!(arenas.resources.is_empty());
    }

    /// The three ordering records reach three different classes.
    #[test]
    fn the_three_ordering_records_stay_three_classes() {
        let live = Live(vec![9]);
        let mut arenas = ExecArenas::default();
        let fence_record = SyncRecord::Fence(FenceRecord {
            kind: FenceKind::Update,
            fence_ref: 9,
            stages: None,
        });
        let event_record = SyncRecord::Event(EventRecord {
            kind: EventKind::Signal,
            event_ref: 9,
            value: 5,
        });
        let barrier_record = SyncRecord::Barrier(BarrierRecord::Texture);
        assert!(matches!(
            sync(&fence_record, &live, &mut arenas),
            Ok(SyncResolved::Fence(_))
        ));
        assert!(matches!(
            sync(&event_record, &live, &mut arenas),
            Ok(SyncResolved::Event(_))
        ));
        assert!(matches!(
            sync(&barrier_record, &live, &mut arenas),
            Ok(SyncResolved::Barrier(_))
        ));
        // The rail is not consulted here: a record has already been recognised
        // on its own rail by the time it reaches resolution.
        let _ = Rail::Render;
    }

    fn buffer_entry(object_ref: u32, offset: u64) -> BufferBind {
        BufferBind {
            buffer_ref: reims_vgpu_protocol::decode::U32le::new(object_ref),
            offset: reims_vgpu_protocol::decode::U64le::new(offset),
        }
    }

    /// A bind entry naming ref zero is an unbind, not a resource. The two have
    /// to stay apart: a slot holding a resource orders against it, and a slot
    /// the guest cleared orders against nothing.
    #[test]
    fn a_bind_entry_of_zero_unbinds_the_slot_rather_than_naming_an_object() {
        let live = Live(vec![5151]);
        let mut arenas = ExecArenas::default();
        let entries = [buffer_entry(5151, 0x10), buffer_entry(0, 0)];
        let record = ComputeRecord::BindBuffers(protocol_compute::BindBuffers {
            first: 2,
            entries: &entries,
        });
        let ComputeOp::BindBuffers { first, entries } =
            compute(&record, &live, &mut arenas).expect("resolved")
        else {
            panic!("not a buffer bind");
        };
        assert_eq!(first, 2);
        assert_eq!(
            &arenas.buffer_bindings[entries.range()],
            &[
                BufferBinding {
                    buffer: Some(id(5151)),
                    offset: 0x10,
                    stride: None,
                },
                BufferBinding {
                    buffer: None,
                    offset: 0,
                    stride: None,
                },
            ]
        );
    }

    /// A nonzero ref that names nothing is still a refusal. Only zero is an
    /// unbind, so a stale ref cannot quietly become one.
    #[test]
    fn a_stale_bind_ref_is_refused_rather_than_read_as_an_unbind() {
        let live = Live(Vec::new());
        let mut arenas = ExecArenas::default();
        let entries = [buffer_entry(5151, 0)];
        let record = ComputeRecord::BindBuffers(protocol_compute::BindBuffers {
            first: 0,
            entries: &entries,
        });
        assert_eq!(
            compute(&record, &live, &mut arenas),
            Err(ResolveRefusal::UnknownRef { object_ref: 5151 })
        );
        assert!(arenas.buffer_bindings.is_empty());
    }

    /// The strided bind puts a stride on every entry and the plain one on none.
    /// The shape is the record's, so an arena element cannot end up holding
    /// both.
    #[test]
    fn the_stride_is_the_records_and_reaches_every_entry() {
        let live = Live(vec![1, 2]);
        let mut arenas = ExecArenas::default();
        let entries = [1u32, 2].map(|r| BufferStrideBind {
            buffer_ref: reims_vgpu_protocol::decode::U32le::new(r),
            offset: reims_vgpu_protocol::decode::U64le::new(0x10),
            attribute_stride: reims_vgpu_protocol::decode::U64le::new(0x20),
        });
        let record =
            ComputeRecord::BindBuffersWithStride(protocol_compute::BindBuffersWithStride {
                first: 0,
                entries: &entries,
            });
        let ComputeOp::BindBuffersWithStride { entries, .. } =
            compute(&record, &live, &mut arenas).expect("resolved")
        else {
            panic!("not a strided bind");
        };
        assert!(arenas.buffer_bindings[entries.range()]
            .iter()
            .all(|b| b.stride == Some(0x20)));
    }

    /// Sampler clamps are per entry. Two entries with different clamps have to
    /// keep them, which is what the plural fixture established on the wire.
    #[test]
    fn sampler_clamps_are_read_off_each_entry() {
        let live = Live(vec![6363, 6464]);
        let mut arenas = ExecArenas::default();
        let entries =
            [(6363u32, 0.25f32, 0.75f32), (6464, 0.125, 0.875)].map(|(r, lo, hi)| SamplerLodBind {
                sampler_ref: reims_vgpu_protocol::decode::U32le::new(r),
                lod_min_clamp: reims_vgpu_protocol::decode::F32le::new(lo),
                lod_max_clamp: reims_vgpu_protocol::decode::F32le::new(hi),
            });
        let record = ComputeRecord::BindSamplersWithLod(protocol_compute::BindSamplersWithLod {
            first: 0,
            entries: &entries,
        });
        let ComputeOp::BindSamplersWithLod { entries, .. } =
            compute(&record, &live, &mut arenas).expect("resolved")
        else {
            panic!("not a sampler bind");
        };
        let bound = &arenas.object_bindings[entries.range()];
        assert_eq!(
            bound[0].lod_clamps,
            Some((LodClamp::from_f32(0.25), LodClamp::from_f32(0.75)))
        );
        assert_eq!(
            bound[1].lod_clamps,
            Some((LodClamp::from_f32(0.125), LodClamp::from_f32(0.875)))
        );
    }

    /// Buffers and objects go to different arenas, so one record's window
    /// cannot be read against the other's.
    #[test]
    fn buffer_and_object_bindings_land_in_their_own_arenas() {
        let live = Live(vec![1]);
        let mut arenas = ExecArenas::default();
        let buffers = [buffer_entry(1, 0)];
        let objects = [RefBind {
            object_ref: reims_vgpu_protocol::decode::U32le::new(1),
        }];
        let a = compute(
            &ComputeRecord::BindBuffers(protocol_compute::BindBuffers {
                first: 0,
                entries: &buffers,
            }),
            &live,
            &mut arenas,
        )
        .expect("resolved");
        let b = compute(
            &ComputeRecord::BindTextures(protocol_compute::BindTextures {
                first: 0,
                entries: &objects,
            }),
            &live,
            &mut arenas,
        )
        .expect("resolved");
        let (ComputeOp::BindBuffers { entries: x, .. }, ComputeOp::BindTextures { entries: y, .. }) =
            (a, b)
        else {
            panic!("wrong ops");
        };
        // Both windows start at zero, and they mean different arenas.
        assert_eq!(x.start, 0);
        assert_eq!(y.start, 0);
        assert_eq!(arenas.buffer_bindings.len(), 1);
        assert_eq!(arenas.object_bindings.len(), 1);
    }

    /// A rebind names no buffer: the slot keeps whatever it holds, so nothing
    /// is resolved and nothing needs to be live.
    #[test]
    fn a_rebind_resolves_nothing_because_it_names_nothing() {
        let live = Live(Vec::new());
        let mut arenas = ExecArenas::default();
        let record = ComputeRecord::RebindBufferOffset(protocol_compute::RebindBufferOffset {
            index: 6,
            offset: 0x5678,
            stride: Some(0x20),
        });
        assert_eq!(
            compute(&record, &live, &mut arenas),
            Ok(ComputeOp::RebindBufferOffset {
                index: 6,
                offset: 0x5678,
                stride: Some(0x20),
            })
        );
    }

    /// An indirect dispatch resolves the buffer its grid comes from, and a
    /// direct one resolves nothing.
    #[test]
    fn only_an_indirect_dispatch_resolves_a_buffer() {
        let live = Live(vec![5151]);
        let mut arenas = ExecArenas::default();
        let direct = ComputeRecord::Dispatch(DispatchRecord::Threadgroups(
            protocol_compute::Threadgroups {
                groups: RecordExtent {
                    width: 1,
                    height: 2,
                    depth: 3,
                },
                threads_per_group: RecordExtent {
                    width: 4,
                    height: 5,
                    depth: 6,
                },
            },
        ));
        let ComputeOp::Dispatch(op) = compute(&direct, &live, &mut arenas).expect("resolved")
        else {
            panic!("not a dispatch");
        };
        assert_eq!(op.indirect_read(), None);

        let indirect_record = ComputeRecord::Dispatch(DispatchRecord::ThreadsIndirect(
            protocol_compute::ThreadsIndirect {
                source: ComputeIndirect {
                    buffer_ref: 5151,
                    offset: 0x1111,
                },
            },
        ));
        let ComputeOp::Dispatch(op) =
            compute(&indirect_record, &live, &mut arenas).expect("resolved")
        else {
            panic!("not a dispatch");
        };
        let (source, extent) = op.indirect_read().expect("indirect");
        assert_eq!(source.buffer, id(5151));
        assert_eq!(source.offset, 0x1111);
        // The SPI form's argument encoding is not established, so the read has
        // no extent and the caller widens rather than guessing one.
        assert_eq!(extent, None);
    }

    /// A pass descriptor resolves every slot, attached or not. The record is a
    /// fixed shape carrying eight colour slots, and a model that skipped the
    /// empty ones would have to invent them again for anything iterating slots
    /// by index.
    #[test]
    fn a_pass_descriptor_resolves_all_ten_slots_and_an_unfilled_one_is_unattached() {
        let live = Live(vec![4242, 4343, 5151]);
        let mut body = pass_body();
        body.color[0].prefix.texture_ref = u32le(4242);
        body.color[0].prefix.store_action = u16le(1);
        body.color[0].clear_color_bits = [0.25f64, 0.5, 0.75, 1.0].map(|v| u64le(v.to_bits()));
        body.depth.prefix.texture_ref = u32le(4343);
        body.visibility_result_buffer_ref = u32le(5151);
        body.render_target_width = u64le(0x1234);
        body.render_target_height = u64le(0x5678);

        let descriptor = pass_descriptor(&body, &live).expect("resolved");
        assert_eq!(descriptor.color[0].texture, Some(id(4242)));
        assert_eq!(
            descriptor.color[0].clear_color(),
            Some([0.25, 0.5, 0.75, 1.0])
        );
        assert_eq!(descriptor.color[0].store, StoreAction::Store);
        assert_eq!(descriptor.depth.texture, Some(id(4343)));
        assert_eq!(descriptor.color[1].texture, None);
        assert_eq!(descriptor.attachments().count(), 10);
        assert_eq!(descriptor.attached().count(), 2);
        assert_eq!(
            descriptor.visibility_result_buffer,
            Some(VisibilityResultBuffer { buffer: id(5151) })
        );
        assert_eq!(descriptor.extent.width, 0x1234);
        assert_eq!(descriptor.extent.height, 0x5678);
    }

    /// A store action outside the four the record shape represents refuses.
    /// Folded onto a neighbour it is either a discarded frame or a resolve that
    /// never happens, and neither is a thing to guess at.
    #[test]
    fn an_undefined_store_action_refuses_rather_than_folding() {
        let live = Live(vec![4242]);
        let mut body = pass_body();
        body.color[0].prefix.texture_ref = u32le(4242);
        body.color[0].prefix.store_action = u16le(9);
        assert_eq!(
            pass_descriptor(&body, &live),
            Err(ResolveRefusal::UndefinedOrdinal {
                field: "store_action",
                value: 9,
            })
        );
    }

    /// And the load action beside them, which was the one field of the three
    /// that folded.
    ///
    /// `LoadAction::from_declared` is total on purpose and its doc leaves the
    /// *reporting* of an out-of-contract ordinal to the caller, because its
    /// callers are the ones with no packet to refuse. This resolver has one.
    /// Folded to `DontCare` a corrupt ordinal means "composite onto whatever
    /// was there", with nothing on the failure channel to say the record was
    /// not read as written.
    #[test]
    fn an_undefined_load_action_refuses_like_the_store_action_beside_it() {
        let live = Live(vec![4242]);
        let mut body = pass_body();
        body.color[0].prefix.texture_ref = u32le(4242);
        body.color[0].prefix.load_action = u16le(7);
        assert_eq!(
            pass_descriptor(&body, &live),
            Err(ResolveRefusal::UndefinedOrdinal {
                field: "load_action",
                value: 7,
            })
        );
    }

    /// Each slot's store-action options are its own, and an undeclared value
    /// refuses.
    ///
    /// The word sat in the wire prefix from the day the layout was measured and
    /// nothing read it, so a pass asking for a resolve at programmable sample
    /// positions was indistinguishable from one asking for the ordinary
    /// resolve. Driven per slot because the guest sets it per slot — a resolver
    /// that read one slot's word for every slot would pass a test that only set
    /// one.
    #[test]
    fn each_slots_store_action_options_are_read_from_its_own_prefix() {
        let live = Live(vec![4242]);
        let mut body = pass_body();
        body.color[0].prefix.texture_ref = u32le(4242);
        body.color[1].prefix.texture_ref = u32le(4242);
        body.color[0].prefix.store_action_options = u16le(1);
        body.depth.prefix.store_action_options = u16le(1);
        let descriptor = pass_descriptor(&body, &live).expect("resolved");
        assert_eq!(
            descriptor.color[0].store_options,
            StoreActionOptions::CustomSamplePositions
        );
        assert_eq!(descriptor.color[1].store_options, StoreActionOptions::None);
        assert_eq!(
            descriptor.depth.store_options,
            StoreActionOptions::CustomSamplePositions
        );
        assert_eq!(descriptor.stencil.store_options, StoreActionOptions::None);

        // The set has one declared flag, so a value outside it is a corrupt
        // record or a wrong offset. Folding it onto the flag would claim the
        // guest asked for programmable sample positions it never asked for.
        for raw in [2u16, 3, 0x1111] {
            let mut body = pass_body();
            body.stencil.prefix.store_action_options = u16le(raw);
            assert_eq!(
                pass_descriptor(&body, &live),
                Err(ResolveRefusal::UndefinedOrdinal {
                    field: "store_action_options",
                    value: u32::from(raw),
                }),
                "{raw:#x}"
            );
        }
    }

    /// The depth and stencil slots each carry a resolve filter, and they are
    /// two ordinal spaces.
    ///
    /// The words sat on the two attachment bodies from the day the layout was
    /// measured and nothing read them, so a pass asking to resolve depth at the
    /// furthest sample was indistinguishable from one asking for sample zero —
    /// and the guest reads a resolved depth buffer back as geometry, so that
    /// difference surfaces as wrong occlusion somewhere later rather than as a
    /// wrong frame now.
    ///
    /// Both slots are driven with the *same* ordinal, because that is the
    /// arrangement a shared filter type would pass: `1` is `Min` on the depth
    /// slot and `DepthResolvedSample` on the stencil one.
    #[test]
    fn each_slots_resolve_filter_is_read_in_its_own_ordinal_space() {
        let live = Live(Vec::new());
        let mut body = pass_body();
        body.depth.resolve_filter = u16le(1);
        body.stencil.resolve_filter = u16le(1);
        let descriptor = pass_descriptor(&body, &live).expect("resolved");
        assert_eq!(descriptor.depth_resolve_filter, DepthResolveFilter::Min);
        assert_eq!(
            descriptor.stencil_resolve_filter,
            StencilResolveFilter::DepthResolvedSample
        );

        // And the value the capture carries on each, which is not the same
        // ordinal on the two slots.
        let mut body = pass_body();
        body.depth.resolve_filter = u16le(2);
        let descriptor = pass_descriptor(&body, &live).expect("resolved");
        assert_eq!(descriptor.depth_resolve_filter, DepthResolveFilter::Max);
        assert_eq!(
            descriptor.stencil_resolve_filter,
            StencilResolveFilter::Sample0,
            "a slot whose word is zero carries what the API starts at"
        );
    }

    /// Both sets are closed, so a value outside one is a corrupt record or a
    /// wrong offset rather than a filter with no contract yet. `2` is declared
    /// on the depth slot and not on the stencil one, which is what makes the
    /// pair worth driving separately.
    #[test]
    fn an_undeclared_resolve_filter_refuses_by_name() {
        let live = Live(Vec::new());
        for raw in [3u16, 4, 0xffff] {
            let mut body = pass_body();
            body.depth.resolve_filter = u16le(raw);
            assert_eq!(
                pass_descriptor(&body, &live),
                Err(ResolveRefusal::UndefinedOrdinal {
                    field: "depth_resolve_filter",
                    value: u32::from(raw),
                }),
                "{raw:#x}"
            );
        }
        for raw in [2u16, 3, 0xffff] {
            let mut body = pass_body();
            body.stencil.resolve_filter = u16le(raw);
            assert_eq!(
                pass_descriptor(&body, &live),
                Err(ResolveRefusal::UndefinedOrdinal {
                    field: "stencil_resolve_filter",
                    value: u32::from(raw),
                }),
                "{raw:#x}"
            );
        }
    }

    /// The visibility buffer lives on the pass and nowhere else, so a pass
    /// without one resolves to `None` rather than to a resource named zero.
    #[test]
    fn a_pass_without_a_visibility_buffer_names_no_resource() {
        let live = Live(Vec::new());
        let body = pass_body();
        let descriptor = pass_descriptor(&body, &live).expect("resolved");
        assert_eq!(descriptor.visibility_result_buffer, None);
    }

    /// A viewport's doubles become bit patterns, and a NaN depth bound survives
    /// as itself. That is what a state table needs to answer "already set".
    #[test]
    fn a_viewport_becomes_bits_and_a_nan_bound_stays_equal_to_itself() {
        let live = Live(Vec::new());
        let mut arenas = ExecArenas::default();
        let ports = [wire_viewport(1.0, 2.0, 3.0, 4.0, f64::NAN, 1.0)];
        let record = RenderRecord::SetViewports(&ports);
        let RenderOp::SetViewports(span) = render(&record, &live, &mut arenas).expect("resolved")
        else {
            panic!("not viewports");
        };
        let port = arenas.viewports[span.range()][0];
        assert_eq!(port.origin_x_bits, 1.0f64.to_bits());
        assert_eq!(port.z_near_bits, f64::NAN.to_bits());
        assert_eq!(port, port);
    }

    /// An indexed draw resolves its index buffer and an indirect one its
    /// argument buffer; a plain draw resolves neither and needs nothing live.
    #[test]
    fn a_draw_resolves_exactly_the_buffers_its_record_names() {
        let live = Live(vec![5151, 5252]);
        let plain = DrawRecord::Primitives(protocol_render::Primitives {
            primitive: 3,
            vertex_start: 0,
            vertex_count: 3,
            instances: RecordInstancing::default(),
        });
        let resolved = draw(&plain, &Live(Vec::new())).expect("resolved");
        assert_eq!(resolved.index_read(), None);
        assert_eq!(resolved.indirect_read(), None);

        let indexed = DrawRecord::IndexedIndirect(protocol_render::IndexedIndirect {
            primitive: 3,
            index: RecordIndexRef {
                buffer_ref: 5151,
                offset: 0x100,
                index_type: IndexType::Uint32,
            },
            arguments: RenderIndirect {
                buffer_ref: 5252,
                offset: 0x200,
            },
        });
        let resolved = draw(&indexed, &live).expect("resolved");
        let (source, range) = resolved.index_read().expect("indexed");
        assert_eq!(source.buffer, id(5151));
        // The count is in the argument buffer, so the range is not established
        // and the caller widens rather than inventing one.
        assert_eq!(range, None);
        let (arguments, bytes) = resolved.indirect_read().expect("indirect");
        assert_eq!(arguments.buffer, id(5252));
        assert_eq!(bytes, crate::render::DRAW_INDEXED_INDIRECT_ARGS_BYTES);
    }

    /// Both stages' binds land in one arena and keep the stage they were sent
    /// on. The stage is the operation's, so two stages' windows can interleave
    /// without either losing which table it fills.
    #[test]
    fn the_two_stages_share_an_arena_and_keep_their_stages() {
        let live = Live(vec![4242]);
        let mut arenas = ExecArenas::default();
        let entries = [RefBind {
            object_ref: u32le(4242),
        }];
        let mut spans = Vec::new();
        for stage in [ShaderStage::Vertex, ShaderStage::Fragment] {
            let record = RenderRecord::BindTextures(protocol_render::BindTextures {
                stage,
                first: 0,
                entries: &entries,
            });
            let RenderOp::BindTextures {
                stage: got,
                entries: span,
                ..
            } = render(&record, &live, &mut arenas).expect("resolved")
            else {
                panic!("not a texture bind");
            };
            assert_eq!(got, stage);
            spans.push(span);
        }
        assert_ne!(spans[0].start, spans[1].start);
        assert_eq!(arenas.object_bindings.len(), 2);
    }

    /// A pass descriptor goes to the descriptor arena and the operation names a
    /// slot. The descriptor is 592 bytes on the wire, and a record carrying it
    /// by value would make every eight-byte operation that size.
    #[test]
    fn a_pass_descriptor_goes_to_the_arena_and_the_operation_names_a_slot() {
        let live = Live(Vec::new());
        let mut arenas = ExecArenas::default();
        let body = pass_body();
        let record =
            RenderRecord::WriteDescriptor(protocol_render::WriteDescriptor { descriptor: &body });
        let RenderOp::WriteDescriptor { descriptor } =
            render(&record, &live, &mut arenas).expect("resolved")
        else {
            panic!("not a descriptor");
        };
        assert_eq!(descriptor, PassDescriptorSlot(0));
        assert_eq!(arenas.pass_descriptors.len(), 1);

        let second = render(&record, &live, &mut arenas).expect("resolved");
        assert_eq!(
            second,
            RenderOp::WriteDescriptor {
                descriptor: PassDescriptorSlot(1)
            }
        );
    }

    /// Frame a record the way the serializer frames one.
    fn framed(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let total = (reims_vgpu_protocol::decode::OP_HEADER_LEN + payload.len()) as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&total.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// A resolver that answers every ref, so a record's failure to resolve is
    /// never about a missing object.
    struct Everything;

    impl RefResolver for Everything {
        fn resource(&self, object_ref: u32) -> Option<ResourceId> {
            Some(ResourceId {
                slot: ObjectListRef(object_ref),
                generation: SlotGeneration(7),
            })
        }
    }

    /// **Every judged stream operation in the ledger resolves**, and lands in
    /// the class the ledger put it in.
    ///
    /// This is the property the whole decode-and-resolve path exists to have:
    /// the model can represent everything the ledger says the device does.
    /// Driven off the ledger rather than a written list, so an operation judged
    /// without a path here fails on the row that judged it.
    ///
    /// The payload is zero-filled and long enough for the widest body. That
    /// makes every count zero and every ordinal the API's first value, which is
    /// exactly what this test wants: it is asking whether a path exists, not
    /// what a particular guest sent. The per-rail tests are where the fields
    /// are pinned.
    #[test]
    fn every_judged_stream_operation_resolves_into_its_own_class() {
        use reims_vgpu_protocol::closure::LEDGER;

        let payload = [0u8; 1024];
        let mut seen = 0usize;
        let mut refused = 0usize;
        for row in LEDGER {
            let Some(opcode) = row.opcode else {
                continue;
            };
            let Some(OperationHome::Stream(class)) = classify(row) else {
                continue;
            };
            // The three classes with no stream record to decode. Their absence
            // is asserted below rather than skipped silently.
            if matches!(
                class,
                OperationClass::EncoderBoundary
                    | OperationClass::InfoQuery
                    | OperationClass::CompletionEffect
            ) {
                continue;
            }
            // One record's length is exact rather than a minimum: the
            // compressed-reinterpretation flush's selector takes no argument,
            // so a payload means the bytes are not that record. Every other
            // body is a fixed shape a longer buffer merely contains.
            let body: &[u8] = if opcode
                == reims_vgpu_protocol::resource_state::OPCODE_COMPRESSED_REINTERPRETATION_FLUSH
                && row.rail == Rail::Compute
            {
                &[]
            } else {
                &payload
            };
            let bytes = framed(opcode, body);
            let view = reims_vgpu_protocol::decode::op(&bytes, 0).expect("framed");
            let mut arenas = ExecArenas::default();
            // A row the ledger settled as a refusal is judged and still must
            // not become an operation. It refuses by that name, which is what
            // keeps a closed decision from reading as an open one.
            if matches!(
                row.closure,
                reims_vgpu_protocol::closure::Closure::Refused { .. }
            ) {
                assert_eq!(
                    operation(row.rail, &view, &Everything, &mut arenas),
                    Err(ResolveRefusal::Decode(DecodeRefusal::RefusedByContract {
                        rail: row.rail,
                        opcode,
                    })),
                    "{:?} {opcode:#x}",
                    row.rail
                );
                refused += 1;
                continue;
            }
            seen += 1;
            let resolved = operation(row.rail, &view, &Everything, &mut arenas)
                .unwrap_or_else(|e| panic!("{:?} {opcode:#x} did not resolve: {e:?}", row.rail));
            assert_eq!(
                resolved.class(),
                class,
                "{:?} {opcode:#x} landed in the wrong class",
                row.rail
            );
        }
        // The census the vocabulary prints is 105 stream operations. Six of
        // them carry no opcode at all — the four `beginSegment:` boundaries,
        // and the blit `withCommand:` selectors that write their command
        // argument into the record's opcode field, so they *are* whichever
        // opcode they emitted. Neither shape is dispatched by opcode, so
        // neither reaches this path, and 99 is what remains.
        //
        // This total grows as the ledger closes rows: an unresolved row is not
        // a judged operation and reaches neither counter, so a row settling
        // either way moves it. That is the intent — it is the number of
        // operations the model claims a path for.
        //
        // Five of the 99 are settled refusals, counted apart rather than folded
        // into either total: a refusal that reads as an unwritten path would
        // send someone to write it.
        assert_eq!(seen + refused, 99);
        assert_eq!(refused, 5);
    }

    /// An opcode the ledger has never heard of stops at the dispatcher, and an
    /// unresolved one stops there too — neither reaches a decoder that might
    /// have lifted a record the model has no use for.
    #[test]
    fn an_unjudged_opcode_never_reaches_a_decoder() {
        let bytes = framed(0x7fff, &[0u8; 32]);
        let view = reims_vgpu_protocol::decode::op(&bytes, 0).expect("framed");
        let mut arenas = ExecArenas::default();
        assert_eq!(
            operation(Rail::Render, &view, &Everything, &mut arenas),
            Err(ResolveRefusal::Decode(DecodeRefusal::UnknownOpcode {
                rail: Rail::Render,
                opcode: 0x7fff,
            }))
        );

        // A residency opcode: the layout is derived and the row is not settled.
        let bytes = framed(0x86, &[0u8; 32]);
        let view = reims_vgpu_protocol::decode::op(&bytes, 0).expect("framed");
        assert_eq!(
            operation(Rail::Render, &view, &Everything, &mut arenas),
            Err(ResolveRefusal::Decode(DecodeRefusal::Unjudged {
                rail: Rail::Render,
                opcode: 0x86,
            }))
        );
    }

    /// A ledger row settled as a refusal keeps its own reason through the
    /// dispatcher. It is not an open question and must not be reported as one.
    #[test]
    fn a_refused_row_keeps_its_reason_through_the_dispatcher() {
        let mut payload = 7u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&5u64.to_le_bytes());
        payload.extend_from_slice(&42u32.to_le_bytes());
        let opcode = reims_vgpu_protocol::sync::OPCODE_WAIT_EVENT_TIMEOUT;
        let bytes = framed(opcode, &payload);
        let view = reims_vgpu_protocol::decode::op(&bytes, 0).expect("framed");
        let mut arenas = ExecArenas::default();
        assert_eq!(
            operation(Rail::Event, &view, &Everything, &mut arenas),
            Err(ResolveRefusal::Decode(DecodeRefusal::RefusedByContract {
                rail: Rail::Event,
                opcode,
            }))
        );
    }

    /// Every refusal reason is distinct, and a decode refusal keeps its own.
    #[test]
    fn refusal_reasons_are_distinct_and_a_decode_refusal_keeps_its_own() {
        let decode = DecodeRefusal::UnknownOpcode {
            rail: Rail::Blit,
            opcode: 1,
        };
        assert_eq!(
            ResolveRefusal::from(decode).reason(),
            decode.reason(),
            "a decode refusal must not be renamed on its way through"
        );
        let mine = [
            ResolveRefusal::UnknownRef { object_ref: 1 },
            ResolveRefusal::UndefinedOrdinal {
                field: "store_action",
                value: 9,
            },
            ResolveRefusal::ArenaOverflow { wanted: 2 },
        ];
        // The list above must name every variant this module raises. Spelled
        // out here without a wildcard so that adding a variant stops the build
        // on this arm rather than quietly leaving its reason unchecked.
        for refusal in mine {
            match refusal {
                ResolveRefusal::Decode(_)
                | ResolveRefusal::UnknownRef { .. }
                | ResolveRefusal::UndefinedOrdinal { .. }
                | ResolveRefusal::ArenaOverflow { .. } => {}
            }
        }
        let mut seen: Vec<&str> = mine.iter().map(|r| r.reason()).collect();
        seen.push(ResolveRefusal::from(decode).reason());
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before);
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

    /// Deliberately tiny, so the nil ref, a live ref and a dead ref all appear
    /// in the same list constantly.
    const REF_POOL: u32 = 5;

    fn some_ref(rng: &mut Rng) -> u32 {
        rng.below(u64::from(REF_POOL)) as u32
    }

    fn some_refs(rng: &mut Rng) -> Vec<u32> {
        (0..rng.below(4)).map(|_| some_ref(rng)).collect()
    }

    /// What the shadow says a record should have appended.
    ///
    /// A plain list per arena: no window arithmetic, no reservation, no mark,
    /// no rollback. That is the whole point — the shadow resolves a record's
    /// refs to completion *before* it appends anything, which is the opposite
    /// shape from resolution's append-and-unwind, so a rollback that misses an
    /// entry disagrees.
    #[derive(Debug)]
    enum Appended {
        Nothing,
        Resources(Vec<ResourceId>),
        Buffers(Vec<BufferBinding>),
        Objects(Vec<ObjectBinding>),
        Viewports(Vec<Viewport>),
        Scissors(Vec<ScissorRect>),
        /// The descriptor arena, named by the two refs the sweep varies.
        Descriptor {
            color0: Option<ResourceId>,
            depth: Option<ResourceId>,
        },
    }

    /// The nil rule, stated here rather than read out of [`bound`].
    ///
    /// A bind entry naming zero unbinds the slot, whether or not slot zero
    /// happens to hold a live object.
    fn shadow_bound(live: &Live, object_ref: u32) -> Result<Option<ResourceId>, u32> {
        if object_ref == 0 {
            return Ok(None);
        }
        live.resource(object_ref).map(Some).ok_or(object_ref)
    }

    /// A ref in a position that has no way to spell "nothing".
    fn shadow_one(live: &Live, object_ref: u32) -> Result<ResourceId, u32> {
        live.resource(object_ref).ok_or(object_ref)
    }

    #[derive(Default)]
    struct Census {
        resolved: u32,
        refused: u32,
        /// Records whose first dead ref was not their first ref, which is the
        /// only shape a partial append can be seen in.
        refused_past_the_first_ref: u32,
        /// Bind entries that named zero.
        unbound_slots: u32,
        /// Bind entries that named zero while slot zero held a live object.
        unbound_over_a_live_zero: u32,
        families: [u32; 11],
    }

    /// **A refused record leaves every arena exactly as it found it, and an
    /// accepted one names a window holding exactly the guest's list in the
    /// guest's order.**
    ///
    /// The shadow is six plain `Vec`s and a resolution that scans a record's
    /// refs to completion before appending anything. It has no `mark`, no
    /// `truncate` and no `reserve`, so it cannot make resolution's mistake of
    /// leaving a partial list behind — the one [`append_refs`]'s doc names.
    /// It states the nil rule for itself too: resolution splits ref zero two
    /// ways, an unbind in a bind entry and a name everywhere else, and a sweep
    /// that asked [`bound`] which position it was in would agree with any split
    /// at all.
    #[test]
    fn a_record_appends_its_whole_list_or_none_of_it() {
        let mut census = Census::default();
        for seed in 0..600u64 {
            let mut rng = Rng::new(seed + 1);
            // A fresh live set per seed, including a slot zero that is
            // sometimes occupied.
            let live = Live((0..REF_POOL).filter(|_| rng.below(2) == 0).collect());
            let zero_is_live = live.resource(0).is_some();
            let mut arenas = ExecArenas::default();
            let mut shadow = ExecArenas::default();

            for _ in 0..12 {
                let before = arenas.clone();
                let family = rng.below(11) as usize;
                census.families[family] += 1;
                // Every ref the record names, in the order resolution reads
                // them, and whether they sit in bind positions.
                let named: Vec<u32>;
                let binds;

                let (got, want): (Result<Option<ResourceSpan>, ResolveRefusal>, _) = match family {
                    // A barrier's residency list. Its entries are `RefBind`,
                    // the same shape a texture bind uses, and they are *not*
                    // bind entries: a residency list has no slot to clear, so
                    // zero names nothing and refuses.
                    0 => {
                        named = some_refs(&mut rng);
                        binds = false;
                        let refs: Vec<RefBind> = named
                            .iter()
                            .map(|&v| RefBind {
                                object_ref: u32le(v),
                            })
                            .collect();
                        let want = named
                            .iter()
                            .map(|&v| shadow_one(&live, v))
                            .collect::<Result<Vec<_>, _>>()
                            .map(Appended::Resources);
                        let record = BarrierRecord::Resources {
                            refs: &refs,
                            after_stages: None,
                            before_stages: None,
                        };
                        let got = barrier(&record, &live, &mut arenas).map(|op| match op.target {
                            BarrierTarget::Resources(span) => Some(span),
                            other => panic!("a resource barrier resolved to {other:?}"),
                        });
                        (got, want)
                    }
                    1 => {
                        named = Vec::new();
                        binds = false;
                        let record = BarrierRecord::Scope {
                            scope: BarrierScope(BarrierScope::BUFFERS),
                            after_stages: None,
                            before_stages: None,
                            unidentified_u8: None,
                        };
                        let got = barrier(&record, &live, &mut arenas).map(|_| None);
                        (got, Ok(Appended::Nothing))
                    }
                    2 => {
                        named = some_refs(&mut rng);
                        binds = true;
                        let entries: Vec<BufferBind> =
                            named.iter().map(|&v| buffer_entry(v, 0x10)).collect();
                        let want = entries
                            .iter()
                            .map(|e| {
                                shadow_bound(&live, e.buffer_ref.get()).map(|buffer| {
                                    BufferBinding {
                                        buffer,
                                        offset: e.offset.get(),
                                        stride: None,
                                    }
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(Appended::Buffers);
                        let record = ComputeRecord::BindBuffers(protocol_compute::BindBuffers {
                            first: 0,
                            entries: &entries,
                        });
                        let got = compute(&record, &live, &mut arenas).map(|op| match op {
                            ComputeOp::BindBuffers { entries, .. } => Some(entries),
                            other => panic!("a buffer bind resolved to {other:?}"),
                        });
                        (got, want)
                    }
                    3 => {
                        named = some_refs(&mut rng);
                        binds = true;
                        let entries: Vec<BufferStrideBind> = named
                            .iter()
                            .map(|&v| BufferStrideBind {
                                buffer_ref: u32le(v),
                                offset: u64le(0x20),
                                attribute_stride: u64le(0x30),
                            })
                            .collect();
                        let want = entries
                            .iter()
                            .map(|e| {
                                shadow_bound(&live, e.buffer_ref.get()).map(|buffer| {
                                    BufferBinding {
                                        buffer,
                                        offset: e.offset.get(),
                                        stride: Some(e.attribute_stride.get()),
                                    }
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(Appended::Buffers);
                        let record = ComputeRecord::BindBuffersWithStride(
                            protocol_compute::BindBuffersWithStride {
                                first: 0,
                                entries: &entries,
                            },
                        );
                        let got = compute(&record, &live, &mut arenas).map(|op| match op {
                            ComputeOp::BindBuffersWithStride { entries, .. } => Some(entries),
                            other => panic!("a strided bind resolved to {other:?}"),
                        });
                        (got, want)
                    }
                    4 => {
                        named = some_refs(&mut rng);
                        binds = true;
                        let entries: Vec<RefBind> = named
                            .iter()
                            .map(|&v| RefBind {
                                object_ref: u32le(v),
                            })
                            .collect();
                        let want = entries
                            .iter()
                            .map(|e| {
                                shadow_bound(&live, e.object_ref.get()).map(|object| {
                                    ObjectBinding {
                                        object,
                                        lod_clamps: None,
                                    }
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(Appended::Objects);
                        let record = ComputeRecord::BindTextures(protocol_compute::BindTextures {
                            first: 0,
                            entries: &entries,
                        });
                        let got = compute(&record, &live, &mut arenas).map(|op| match op {
                            ComputeOp::BindTextures { entries, .. } => Some(entries),
                            other => panic!("a texture bind resolved to {other:?}"),
                        });
                        (got, want)
                    }
                    5 => {
                        named = some_refs(&mut rng);
                        binds = true;
                        let entries: Vec<SamplerLodBind> = named
                            .iter()
                            .map(|&v| SamplerLodBind {
                                sampler_ref: u32le(v),
                                lod_min_clamp: reims_vgpu_protocol::decode::F32le::new(0.25),
                                lod_max_clamp: reims_vgpu_protocol::decode::F32le::new(8.0),
                            })
                            .collect();
                        let want = entries
                            .iter()
                            .map(|e| {
                                shadow_bound(&live, e.sampler_ref.get()).map(|object| {
                                    ObjectBinding {
                                        object,
                                        lod_clamps: Some((
                                            LodClamp::from_f32(e.lod_min_clamp.get()),
                                            LodClamp::from_f32(e.lod_max_clamp.get()),
                                        )),
                                    }
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(Appended::Objects);
                        let record = ComputeRecord::BindSamplersWithLod(
                            protocol_compute::BindSamplersWithLod {
                                first: 0,
                                entries: &entries,
                            },
                        );
                        let got = compute(&record, &live, &mut arenas).map(|op| match op {
                            ComputeOp::BindSamplersWithLod { entries, .. } => Some(entries),
                            other => panic!("a sampler bind resolved to {other:?}"),
                        });
                        (got, want)
                    }
                    // The two state arenas, which name no ref at all and so
                    // can never refuse — but still share the arena discipline.
                    6 => {
                        named = Vec::new();
                        binds = false;
                        let ports: Vec<WireViewport> = (0..rng.below(4))
                            .map(|i| wire_viewport(i as f64, 2.0, 3.0, 4.0, 0.0, 1.0))
                            .collect();
                        let want = Ok(Appended::Viewports(
                            ports
                                .iter()
                                .map(|p| Viewport {
                                    origin_x_bits: p.origin_x.get().to_bits(),
                                    origin_y_bits: p.origin_y.get().to_bits(),
                                    width_bits: p.width.get().to_bits(),
                                    height_bits: p.height.get().to_bits(),
                                    z_near_bits: p.znear.get().to_bits(),
                                    z_far_bits: p.zfar.get().to_bits(),
                                })
                                .collect(),
                        ));
                        let record = RenderRecord::SetViewports(&ports);
                        let got = render(&record, &live, &mut arenas).map(|op| match op {
                            RenderOp::SetViewports(span) => Some(span),
                            other => panic!("viewports resolved to {other:?}"),
                        });
                        (got, want)
                    }
                    7 => {
                        named = Vec::new();
                        binds = false;
                        let rects: Vec<WireScissorRect> = (0..rng.below(4))
                            .map(|i| WireScissorRect {
                                x: u64le(i),
                                y: u64le(i + 1),
                                width: u64le(16),
                                height: u64le(32),
                            })
                            .collect();
                        let want = Ok(Appended::Scissors(
                            rects
                                .iter()
                                .map(|s| ScissorRect {
                                    x: s.x.get(),
                                    y: s.y.get(),
                                    width: s.width.get(),
                                    height: s.height.get(),
                                })
                                .collect(),
                        ));
                        let record = RenderRecord::SetScissorRects(&rects);
                        let got = render(&record, &live, &mut arenas).map(|op| match op {
                            RenderOp::SetScissorRects(span) => Some(span),
                            other => panic!("scissors resolved to {other:?}"),
                        });
                        (got, want)
                    }
                    // A single ref in a position with no unbind spelling.
                    8 => {
                        named = vec![some_ref(&mut rng)];
                        binds = false;
                        let want = shadow_one(&live, named[0]).map(|_| Appended::Nothing);
                        let record = RenderRecord::SetPipeline(protocol_render::SetPipeline {
                            pipeline_ref: named[0],
                        });
                        let got = render(&record, &live, &mut arenas).map(|_| None);
                        (got, want)
                    }
                    // Two refs, so a record can refuse on its second with its
                    // first already resolved.
                    9 => {
                        named = vec![some_ref(&mut rng), some_ref(&mut rng)];
                        binds = false;
                        let want = shadow_one(&live, named[0])
                            .and_then(|_| shadow_one(&live, named[1]))
                            .map(|_| Appended::Nothing);
                        let record = BlitRecord::BufferToBuffer(record::BufferToBuffer {
                            source_ref: named[0],
                            source_offset: 0,
                            dest_ref: named[1],
                            dest_offset: 0,
                            size: 64,
                        });
                        let got = blit(&record, &live).map(|_| None);
                        (got, want)
                    }
                    // A pass descriptor: ten bind-shaped refs resolved into one
                    // value, appended to an arena of its own.
                    _ => {
                        named = vec![some_ref(&mut rng), some_ref(&mut rng)];
                        binds = true;
                        let mut body = pass_body();
                        body.color[0].prefix.texture_ref = u32le(named[0]);
                        body.depth.prefix.texture_ref = u32le(named[1]);
                        let want = shadow_bound(&live, named[0]).and_then(|color0| {
                            shadow_bound(&live, named[1])
                                .map(|depth| Appended::Descriptor { color0, depth })
                        });
                        let record =
                            RenderRecord::WriteDescriptor(protocol_render::WriteDescriptor {
                                descriptor: &body,
                            });
                        let got = render(&record, &live, &mut arenas).map(|op| match op {
                            RenderOp::WriteDescriptor { descriptor } => Some(ResourceSpan {
                                start: descriptor.0,
                                len: 1,
                            }),
                            other => panic!("a descriptor resolved to {other:?}"),
                        });
                        (got, want)
                    }
                };

                match (got, want) {
                    (Err(refusal), Err(object_ref)) => {
                        census.refused += 1;
                        assert_eq!(
                            refusal,
                            ResolveRefusal::UnknownRef { object_ref },
                            "the refusal must name the guest's first dead ref"
                        );
                        // Every ref before the dead one resolved, so a record
                        // refusing past its first ref had already done work.
                        let at = named
                            .iter()
                            .position(|&v| v == object_ref)
                            .expect("the refusal names a ref the record carried");
                        if at > 0 {
                            census.refused_past_the_first_ref += 1;
                        }
                        assert_eq!(
                            arenas, before,
                            "a refused record left something in an arena"
                        );
                    }
                    (Ok(span), Ok(appended)) => {
                        census.resolved += 1;
                        if binds {
                            let zeros = named.iter().filter(|&&v| v == 0).count() as u32;
                            census.unbound_slots += zeros;
                            if zero_is_live {
                                census.unbound_over_a_live_zero += zeros;
                            }
                        }
                        apply(&mut shadow, &arenas, span, appended);
                        assert_eq!(
                            arenas, shadow,
                            "the arenas and the shadow disagree after a resolve"
                        );
                    }
                    (got, want) => panic!("resolution said {got:?}, the shadow said {want:?}"),
                }
            }
        }

        for (family, count) in census.families.iter().enumerate() {
            assert!(*count > 400, "family {family} drove {count} records");
        }
        assert!(census.resolved > 2000, "{}", census.resolved);
        assert!(census.refused > 1000, "{}", census.refused);
        assert!(
            census.refused_past_the_first_ref > 200,
            "nothing refused past its first ref: {}",
            census.refused_past_the_first_ref
        );
        assert!(census.unbound_slots > 200, "{}", census.unbound_slots);
        assert!(
            census.unbound_over_a_live_zero > 50,
            "slot zero was never live while a bind named it: {}",
            census.unbound_over_a_live_zero
        );
    }

    /// Push what the shadow expected, checking that the window resolution
    /// returned names exactly those entries in the real arena.
    fn apply(
        shadow: &mut ExecArenas,
        arenas: &ExecArenas,
        span: Option<ResourceSpan>,
        appended: Appended,
    ) {
        /// The window `count` entries would occupy, given what is already
        /// there. The shadow's whole arithmetic.
        fn expected(at: usize, count: usize) -> ResourceSpan {
            ResourceSpan {
                start: u32::try_from(at).expect("arena fits"),
                len: u32::try_from(count).expect("list fits"),
            }
        }
        match appended {
            Appended::Nothing => assert!(span.is_none(), "a record with no list named {span:?}"),
            Appended::Resources(entries) => {
                let want = expected(shadow.resources.len(), entries.len());
                assert_eq!(span, Some(want));
                assert_eq!(&arenas.resources[want.range()], &entries[..]);
                shadow.resources.extend(entries);
            }
            Appended::Buffers(entries) => {
                let want = expected(shadow.buffer_bindings.len(), entries.len());
                assert_eq!(span, Some(want));
                assert_eq!(&arenas.buffer_bindings[want.range()], &entries[..]);
                shadow.buffer_bindings.extend(entries);
            }
            Appended::Objects(entries) => {
                let want = expected(shadow.object_bindings.len(), entries.len());
                assert_eq!(span, Some(want));
                assert_eq!(&arenas.object_bindings[want.range()], &entries[..]);
                shadow.object_bindings.extend(entries);
            }
            Appended::Viewports(entries) => {
                let want = expected(shadow.viewports.len(), entries.len());
                assert_eq!(span, Some(want));
                assert_eq!(&arenas.viewports[want.range()], &entries[..]);
                shadow.viewports.extend(entries);
            }
            Appended::Scissors(entries) => {
                let want = expected(shadow.scissors.len(), entries.len());
                assert_eq!(span, Some(want));
                assert_eq!(&arenas.scissors[want.range()], &entries[..]);
                shadow.scissors.extend(entries);
            }
            Appended::Descriptor { color0, depth } => {
                let want = expected(shadow.pass_descriptors.len(), 1);
                assert_eq!(span, Some(want));
                let landed = arenas.pass_descriptors[want.start as usize];
                // The two refs the sweep varies. The rest of the descriptor is
                // the fixed shape `pass_body` builds and the named tests pin,
                // so it is carried across rather than restated.
                assert_eq!(landed.color[0].texture, color0);
                assert_eq!(landed.depth.texture, depth);
                shadow.pass_descriptors.push(landed);
            }
        }
    }
}
