//! The binding tables, and how a draw or a dispatch turns them into a
//! footprint.
//!
//! # A bind record writes a slot; this is the slots
//!
//! [`crate::compute`] and [`crate::render`] both say the same thing about
//! binds: they touch no memory. This is where the memory they *will* touch
//! accumulates, and where a dispatch or a draw reads it back out.
//!
//! # A table is not capped
//!
//! Apple's serializer truncates a plural bind at the stage's argument-table
//! size — 128 textures, 31 buffers, 16 samplers, measured — and those numbers
//! are the right *capacity hint* and the wrong *bound*. `reims-vgpu` has twice
//! shipped a cap standing in for them and twice dropped whole binds, so the
//! table here grows to whatever slot the guest names. The limits reserve the
//! usual size once; nothing refuses a slot above them.
//!
//! # What a bound slot contributes is the pipeline's answer, not the model's
//!
//! A buffer bound at slot 3 might be read, written, or never referenced. The
//! record does not say, the residency declarations that would say are entirely
//! unresolved in the ledger, and the thing that actually knows is the compiled
//! shader. So the footprint takes a [`crate::pipeline::BindingUsage`] — an
//! immutable fact an executor publishes when it finishes compiling — and
//! without one every bound slot participates as
//! [`AccessMode::Unknown`] over the whole resource.
//!
//! That fallback is correct and expensive, and both halves matter. It is
//! correct because `Unknown` conflicts with everything, so no edge is missed.
//! It is expensive because it conflicts with everything, which is why
//! `AccessMode::Unknown` is a distinct variant from `ReadWrite`: the census can
//! count how much ordering is being bought by not knowing, and a number that
//! can be counted is a number that can be driven down.

use crate::access::{AccessMode, Participation, ParticipationExtent};
use crate::bind::{BindSpan, BufferBinding, ObjectBinding};
use crate::compute::ComputeOp;
use crate::identity::ResourceId;
use crate::pipeline::BindingUsage;
use crate::render::RenderOp;
use reims_vgpu_protocol::render::ShaderStage;

/// The argument-table sizes Apple's serializer truncates a plural bind at.
///
/// Capacity, not capacity limits. See the module documentation.
mod table_hint {
    pub use reims_vgpu_protocol::bind::{
        BUFFER_TABLE_HINT as BUFFER, SAMPLER_TABLE_HINT as SAMPLER, TEXTURE_TABLE_HINT as TEXTURE,
    };
}

/// One resource class's slots for one stage.
///
/// Sparse in content and sparse in storage: the bound slots in slot order,
/// and nothing at all for the gaps between them.
///
/// # The slot number is the guest's and the storage is not
///
/// The module doc's law is that nothing refuses a slot above the argument
/// table's size, and it is about *meaning*: a high slot is a slot, not an
/// error. Storing the table as a `Vec<Option<T>>` indexed by slot read that as
/// a claim about layout too, and the two are not the same claim. One plural
/// bind record naming `first = 0xF000_0000` made the table materialise four
/// billion empty slots — a `handle_alloc_error` abort on guest data, which is
/// the one failure this model may not have, and below the abort a guest could
/// buy hundreds of megabytes per record and keep them.
///
/// So the law stands and the storage does not follow the slot number: an entry
/// costs what a binding costs whatever its slot is, and the gap between two
/// bound slots costs nothing. A lookup is a binary search over the bound slots
/// rather than an index into the reached ones, which for a table the guest
/// actually binds — tens of entries — is the same handful of cache lines, and
/// every walk over the table is now over what is bound rather than over how
/// far the guest reached.
///
/// # A slot remembers whether the footprint already has it
///
/// A draw's footprint is a function of the binding and the pipeline's usage of
/// it, and neither changes between two draws with no record in between — so the
/// second draw's participation would be the first one's, again. That is not
/// free: every participation is a namespace resolution, a residency lookup and,
/// for a write, a *reserved content version*. A thousand draws over twenty
/// bound slots would reserve twenty thousand versions of the same memory, of
/// which one survives.
///
/// So a slot carries whether the open encoder has already declared it, and only
/// the ones that have not are built. A transaction's footprint is a set; the
/// declaration order inside it orders nothing, which is why declaring once is
/// the same answer and not an approximation of it.
///
/// **Declaration is an epoch and not a flag**, for the reason
/// [`SlotTable::undeclare`] gives: the guest binds a pipeline per draw and a
/// pipeline change takes back every declaration, so that has to be one write
/// rather than a walk.
#[derive(Clone, Debug)]
pub struct SlotTable<T> {
    /// Bound slots, ascending and unique. Nothing is stored for an unbound
    /// slot, so this is as long as the guest has bound and not as far as it
    /// has reached.
    bound: Vec<Slot<T>>,
    /// The declaration epoch. A slot is declared exactly when it carries this
    /// value, so [`SlotTable::undeclare`] is one increment and never a walk.
    /// Starts at one, because zero is what a freshly bound slot carries and
    /// means "declared under no epoch at all".
    epoch: u64,
}

/// One bound slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot<T> {
    slot: u32,
    value: T,
    /// The epoch this slot was declared under. See [`SlotTable::epoch`].
    declared: u64,
}

impl<T: Copy + PartialEq> SlotTable<T> {
    #[must_use]
    pub fn with_hint(hint: u32) -> Self {
        Self {
            bound: Vec::with_capacity(hint as usize),
            epoch: 1,
        }
    }

    /// Where `slot` is, or where it would go.
    fn find(&self, slot: u32) -> Result<usize, usize> {
        self.bound.binary_search_by_key(&slot, |s| s.slot)
    }

    /// Bind `value` at `slot`, or unbind it with `None`.
    ///
    /// A bind of what the slot already holds changes nothing, including whether
    /// the footprint has it. That is what the comparability of the binding
    /// types is for: a guest that re-binds its whole table between draws — and
    /// they do — must not make the next draw declare everything again.
    ///
    /// Unbinding a slot nothing bound is not a write: the gap was already the
    /// absence of an entry, and storing one for it would be the layout the
    /// type doc refuses.
    pub fn set(&mut self, slot: u32, value: Option<T>) {
        match (self.find(slot), value) {
            (Ok(at), Some(value)) => {
                if self.bound[at].value == value {
                    return;
                }
                self.bound[at].value = value;
                // A new binding is a new question for the footprint.
                self.bound[at].declared = 0;
            }
            (Ok(at), None) => {
                self.bound.remove(at);
            }
            (Err(at), Some(value)) => self.bound.insert(
                at,
                Slot {
                    slot,
                    value,
                    declared: 0,
                },
            ),
            (Err(_), None) => {}
        }
    }

    /// Every bound slot the open encoder has not declared yet.
    pub fn undeclared(&self) -> impl Iterator<Item = (u32, T)> + '_ {
        self.bound
            .iter()
            .filter(move |s| s.declared != self.epoch)
            .map(|s| (s.slot, s.value))
    }

    /// Record that the footprint now has every bound slot.
    ///
    /// Marks the slots a reflection said nothing about too: under this pipeline
    /// they contribute nothing, and that is as much an answer as a
    /// participation. A pipeline change is what takes it back.
    pub fn mark_declared(&mut self) {
        let epoch = self.epoch;
        for slot in &mut self.bound {
            slot.declared = epoch;
        }
    }

    /// Forget what the footprint has, keeping the bindings.
    ///
    /// The pipeline decides what a bound slot contributes, so a new one makes
    /// every slot's contribution a fresh question — and the guest binds a
    /// pipeline per draw, so this is one increment and touches no slot.
    pub fn undeclare(&mut self) {
        self.epoch += 1;
    }

    #[must_use]
    pub fn get(&self, slot: u32) -> Option<T> {
        self.find(slot).ok().map(|at| self.bound[at].value)
    }

    /// Every bound slot, with its index.
    pub fn bound(&self) -> impl Iterator<Item = (u32, T)> + '_ {
        self.bound.iter().map(|s| (s.slot, s.value))
    }

    /// How far the guest has reached, which is not how many are bound.
    #[must_use]
    pub fn extent(&self) -> usize {
        self.bound.last().map_or(0, |s| s.slot as usize + 1)
    }

    pub fn clear(&mut self) {
        self.bound.clear();
    }
}

/// Two tables are equal when the same slots hold the same bindings and the
/// same ones are declared.
///
/// Written out rather than derived: the epoch is how declaration is *stored*
/// and not what it *is*, so two tables that have declared the same slots after
/// different numbers of pipeline changes are the same table.
impl<T: PartialEq> PartialEq for SlotTable<T> {
    fn eq(&self, other: &Self) -> bool {
        self.bound.len() == other.bound.len()
            && self.bound.iter().zip(&other.bound).all(|(a, b)| {
                a.slot == b.slot
                    && a.value == b.value
                    && (a.declared == self.epoch) == (b.declared == other.epoch)
            })
    }
}

impl<T: Eq> Eq for SlotTable<T> {}

impl<T: Copy> Default for SlotTable<T> {
    fn default() -> Self {
        Self {
            bound: Vec::new(),
            epoch: 1,
        }
    }
}

/// One stage's three tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageTables {
    pub buffers: SlotTable<BufferBinding>,
    pub textures: SlotTable<ObjectBinding>,
    /// Samplers bind no memory, so they never appear in a footprint. They are
    /// held because the encoder's state is the encoder's state — an executor
    /// building a descriptor set needs them — and not because they order
    /// anything.
    pub samplers: SlotTable<ObjectBinding>,
}

impl Default for StageTables {
    fn default() -> Self {
        Self {
            buffers: SlotTable::with_hint(table_hint::BUFFER),
            textures: SlotTable::with_hint(table_hint::TEXTURE),
            samplers: SlotTable::with_hint(table_hint::SAMPLER),
        }
    }
}

impl StageTables {
    /// Unbind everything, keeping the storage.
    ///
    /// A new encoder starts with nothing bound, and it starts often — so this
    /// is a clear rather than a fresh set of tables: the argument-table
    /// capacity is reserved once per builder and reused by every encoder after
    /// the first.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.samplers.clear();
    }

    /// Append what this stage's bound slots contribute.
    ///
    /// Written into a caller's buffer rather than returned, because this runs
    /// once per draw and a fresh `Vec` per draw is an allocation the frame does
    /// not need.
    ///
    /// **Reads and does not mark.** What the footprint *has* is
    /// [`Self::declare`], and the two are separate because a caller may gather
    /// a record's footprint and then not keep the record — a resolution that
    /// refuses, a cursor that declines the record's rail. Marked here, that
    /// discarded footprint would take the declaration with it and the next
    /// draw would name nothing, which is a missing hazard edge and not a
    /// missing log line.
    pub fn footprint_into(&self, usage: Option<&BindingUsage>, out: &mut Vec<Participation>) {
        for (slot, binding) in self.buffers.undeclared() {
            let Some(buffer) = binding.buffer else {
                continue;
            };
            let Some(mode) = slot_mode(usage, |u| u.buffer(slot)) else {
                continue;
            };
            out.push(Participation {
                resource: buffer,
                // The bind names an offset and no length: what the shader reads
                // from there is the shader's business and the buffer's size is
                // the resource's. Whole is the honest extent, and it is where
                // reflection could narrow later.
                extent: ParticipationExtent::Whole,
                mode,
                api_stages: NO_STAGES,
            });
        }
        for (slot, binding) in self.textures.undeclared() {
            let Some(texture) = binding.object else {
                continue;
            };
            let Some(mode) = slot_mode(usage, |u| u.texture(slot)) else {
                continue;
            };
            out.push(Participation {
                resource: texture,
                extent: ParticipationExtent::Whole,
                mode,
                api_stages: NO_STAGES,
            });
        }
    }

    /// Record that a footprint this stage answered is now in a transaction.
    ///
    /// Both tables: a slot that contributed nothing under this pipeline is as
    /// answered as one that contributed a participation, and only a new
    /// pipeline makes either a fresh question. Samplers bind no memory and are
    /// never in the footprint, so their slots are left alone.
    pub fn declare(&mut self) {
        self.buffers.mark_declared();
        self.textures.mark_declared();
    }

    /// Forget what the footprint has, keeping the bindings. See
    /// [`SlotTable::undeclare`].
    fn undeclare(&mut self) {
        self.buffers.undeclare();
        self.textures.undeclare();
    }
}

/// The mode a bound slot contributes, or `None` when the pipeline does not
/// reference it.
///
/// Without a reflection the answer is [`AccessMode::Unknown`] for every bound
/// slot: a slot the model cannot ask about might be written, and a footprint
/// that guessed read would miss the edge that matters.
fn slot_mode(
    usage: Option<&BindingUsage>,
    ask: impl FnOnce(&BindingUsage) -> Option<AccessMode>,
) -> Option<AccessMode> {
    match usage {
        Some(usage) => ask(usage),
        None => Some(AccessMode::Unknown),
    }
}

/// A bound slot declares no stage of its own; the stages are the pipeline's.
const NO_STAGES: u32 = 0;

/// Bind `entries` at consecutive slots from `first`.
///
/// The record's count is the guest's and the slot numbers follow from it, so a
/// plural bind is one loop here rather than a slot number recomputed at every
/// call site. A count that would reach past `u32` cannot exist: the entries
/// came out of a transaction arena whose window is already a `u32` pair.
fn bind_span<T: Copy + PartialEq>(table: &mut SlotTable<T>, first: u32, entries: &[T]) {
    for (offset, entry) in entries.iter().enumerate() {
        let Ok(offset) = u32::try_from(offset) else {
            return;
        };
        let Some(slot) = first.checked_add(offset) else {
            return;
        };
        table.set(slot, Some(*entry));
    }
}

/// Move an already-bound buffer's offset, and optionally its stride.
///
/// A slot holding nothing stays holding nothing. The record names no buffer —
/// it moves whatever the slot holds — so an unbound slot has nothing to move,
/// and inventing a binding here would make a resource out of a record that
/// named none.
fn rebind_offset(
    table: &mut SlotTable<BufferBinding>,
    index: u32,
    offset: u64,
    stride: Option<u64>,
) {
    let Some(mut binding) = table.get(index) else {
        return;
    };
    binding.offset = offset;
    if stride.is_some() {
        binding.stride = stride;
    }
    table.set(index, Some(binding));
}

/// The compute encoder's state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputeEncoderState {
    pub tables: StageTables,
    pub pipeline: Option<ResourceId>,
}

impl ComputeEncoderState {
    /// Return to what a freshly opened encoder holds, keeping the storage.
    pub fn clear(&mut self) {
        self.tables.clear();
        self.pipeline = None;
    }

    /// What a dispatch reads through the bound slots.
    pub fn footprint_into(&self, usage: Option<&BindingUsage>, out: &mut Vec<Participation>) {
        self.tables.footprint_into(usage, out);
    }

    /// Record that the footprint this encoder answered is now in a
    /// transaction. See [`StageTables::footprint_into`] for why it is a second
    /// call.
    pub fn declare(&mut self) {
        self.tables.declare();
    }

    /// Apply one record to the encoder's state.
    ///
    /// The entries are the transaction's arenas, because a bind record names a
    /// window of one rather than carrying its entries. Exhaustive over
    /// [`ComputeOp`] for the reason every other match on the vocabulary is: a
    /// record that changes what a slot holds and is not applied here makes the
    /// next dispatch's footprint name the *previous* resource, which is a
    /// missing hazard edge rather than a wrong log line.
    pub fn apply(&mut self, op: &ComputeOp, buffers: &[BufferBinding], objects: &[ObjectBinding]) {
        match *op {
            ComputeOp::BindBuffers { first, entries }
            | ComputeOp::BindBuffersWithStride { first, entries } => {
                bind_span(&mut self.tables.buffers, first, span(buffers, entries));
            }
            ComputeOp::BindTextures { first, entries } => {
                bind_span(&mut self.tables.textures, first, span(objects, entries));
            }
            ComputeOp::BindSamplers { first, entries }
            | ComputeOp::BindSamplersWithLod { first, entries } => {
                bind_span(&mut self.tables.samplers, first, span(objects, entries));
            }
            ComputeOp::RebindBufferOffset {
                index,
                offset,
                stride,
            } => rebind_offset(&mut self.tables.buffers, index, offset, stride),
            ComputeOp::SetPipeline { pipeline } => {
                // What a bound slot contributes is the pipeline's answer, so a
                // different pipeline makes every slot's contribution a fresh
                // question. The same one changes nothing.
                if self.pipeline != Some(pipeline) {
                    self.pipeline = Some(pipeline);
                    self.tables.undeclare();
                }
            }
            // State a dispatch carries rather than state a slot holds, and a
            // dispatch reads the tables without changing them.
            ComputeOp::SetStageInRegion { .. }
            | ComputeOp::SetStageInRegionIndirect { .. }
            | ComputeOp::SetThreadgroupMemory { .. }
            | ComputeOp::SetImageblockSize { .. }
            | ComputeOp::WriteDescriptor { .. }
            | ComputeOp::Dispatch(_) => {}
        }
    }
}

/// The entries a bind record's window names.
///
/// Out of range is empty rather than a panic: the arena and the window are
/// built together by [`crate::exec::ExecBuilder`], so a mismatch is a bug in
/// this crate, and taking a whole packet down over it would lose work the guest
/// is waiting on. An empty span binds nothing, which is the same answer the
/// footprint would give for slots that were never written.
fn span<T>(arena: &[T], window: BindSpan) -> &[T] {
    let start = window.start as usize;
    let end = start.saturating_add(window.len as usize);
    arena.get(start..end).unwrap_or(&[])
}

/// The render encoder's state.
///
/// Two stages, and they are separate tables rather than one with a stage tag:
/// slot 3 of the vertex stage and slot 3 of the fragment stage are different
/// slots, and a single table keyed by index alone would have them overwrite each
/// other.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderEncoderState {
    pub vertex: StageTables,
    pub fragment: StageTables,
    pub pipeline: Option<ResourceId>,
    pub depth_stencil: Option<ResourceId>,
    /// The occlusion-query mode and its offset into the pass's visibility
    /// buffer. The buffer itself is the pass descriptor's.
    pub visibility: Option<VisibilityState>,
}

/// The occlusion-query state a draw inherits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VisibilityState {
    pub mode: u64,
    pub offset: u64,
}

impl RenderEncoderState {
    /// Return to what a freshly opened encoder holds, keeping the storage.
    pub fn clear(&mut self) {
        self.vertex.clear();
        self.fragment.clear();
        self.pipeline = None;
        self.depth_stencil = None;
        self.visibility = None;
    }

    #[must_use]
    pub fn stage(&self, stage: ShaderStage) -> &StageTables {
        match stage {
            ShaderStage::Vertex => &self.vertex,
            ShaderStage::Fragment => &self.fragment,
        }
    }

    pub fn stage_mut(&mut self, stage: ShaderStage) -> &mut StageTables {
        match stage {
            ShaderStage::Vertex => &mut self.vertex,
            ShaderStage::Fragment => &mut self.fragment,
        }
    }

    /// What a draw reads through the bound slots of both stages.
    ///
    /// The two usages are separate because the two stages are: a buffer the
    /// vertex shader reads and the fragment shader writes is two participations
    /// on one resource, and collapsing them would lose the write.
    pub fn footprint_into(
        &self,
        vertex_usage: Option<&BindingUsage>,
        fragment_usage: Option<&BindingUsage>,
        out: &mut Vec<Participation>,
    ) {
        self.vertex.footprint_into(vertex_usage, out);
        self.fragment.footprint_into(fragment_usage, out);
    }

    /// Record that the footprint this encoder answered is now in a
    /// transaction. Both stages, because the footprint was both stages'.
    pub fn declare(&mut self) {
        self.vertex.declare();
        self.fragment.declare();
    }

    /// Apply one record to the encoder's state.
    ///
    /// Exhaustive over [`RenderOp`], for the reason
    /// [`ComputeEncoderState::apply`] gives. The stage is the record's own:
    /// `BindBuffersWithStride` has no fragment form because the API has no
    /// fragment attribute stride, and reading a stage off anything but the
    /// record would put a vertex bind in the fragment table.
    pub fn apply(&mut self, op: &RenderOp, buffers: &[BufferBinding], objects: &[ObjectBinding]) {
        match *op {
            RenderOp::BindBuffers {
                stage,
                first,
                entries,
            } => bind_span(
                &mut self.stage_mut(stage).buffers,
                first,
                span(buffers, entries),
            ),
            RenderOp::BindBuffersWithStride { first, entries } => bind_span(
                &mut self.stage_mut(ShaderStage::Vertex).buffers,
                first,
                span(buffers, entries),
            ),
            RenderOp::BindTextures {
                stage,
                first,
                entries,
            } => bind_span(
                &mut self.stage_mut(stage).textures,
                first,
                span(objects, entries),
            ),
            RenderOp::BindSamplers {
                stage,
                first,
                entries,
            }
            | RenderOp::BindSamplersWithLod {
                stage,
                first,
                entries,
            } => bind_span(
                &mut self.stage_mut(stage).samplers,
                first,
                span(objects, entries),
            ),
            RenderOp::RebindBufferOffset {
                stage,
                index,
                offset,
                stride,
            } => rebind_offset(&mut self.stage_mut(stage).buffers, index, offset, stride),
            RenderOp::SetPipeline { pipeline } => {
                // See [`ComputeEncoderState::apply`]: the pipeline decides what
                // a bound slot contributes, on both stages.
                if self.pipeline != Some(pipeline) {
                    self.pipeline = Some(pipeline);
                    self.vertex.undeclare();
                    self.fragment.undeclare();
                }
            }
            RenderOp::SetDepthStencilState { state } => self.depth_stencil = Some(state),
            RenderOp::SetVisibilityResultMode { mode, offset } => {
                self.visibility = Some(VisibilityState { mode, offset });
            }
            // Fixed-function state and the pass descriptor. None of it changes
            // what a slot holds, and a draw reads the tables without changing
            // them.
            RenderOp::Draw(_)
            | RenderOp::WriteDescriptor { .. }
            | RenderOp::SetViewports(_)
            | RenderOp::SetScissorRects(_)
            | RenderOp::SetCullMode(_)
            | RenderOp::SetFrontFacingWinding(_)
            | RenderOp::SetDepthClipMode(_)
            | RenderOp::SetTriangleFillMode(_)
            | RenderOp::SetDepthBias { .. }
            | RenderOp::SetLineWidth(_)
            | RenderOp::SetBlendColor { .. }
            | RenderOp::SetStencilReference { .. }
            | RenderOp::SetStoreAction { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn buffer(id: u32) -> BufferBinding {
        BufferBinding {
            buffer: Some(res(id)),
            offset: 0x40,
            stride: None,
        }
    }

    fn texture(id: u32) -> ObjectBinding {
        ObjectBinding {
            object: Some(res(id)),
            lod_clamps: None,
        }
    }

    /// The table grows past every measured argument-table size rather than
    /// refusing there. Both of this project's dropped-bind regressions were a
    /// cap standing in for a capacity.
    #[test]
    fn a_table_grows_past_the_serializers_own_truncation_points() {
        let mut table = SlotTable::<BufferBinding>::with_hint(table_hint::BUFFER);
        let far = table_hint::TEXTURE + 500;
        table.set(far, Some(buffer(1)));
        assert_eq!(table.get(far), Some(buffer(1)));
        assert_eq!(table.bound().count(), 1);
        assert_eq!(table.extent(), far as usize + 1);
    }

    /// Growing does not bind the slots it grew past.
    #[test]
    fn growing_leaves_the_gap_unbound() {
        let mut table = SlotTable::<BufferBinding>::default();
        table.set(0, Some(buffer(1)));
        table.set(10, Some(buffer(2)));
        assert_eq!(table.get(5), None);
        let bound: Vec<_> = table.bound().map(|(slot, _)| slot).collect();
        assert_eq!(bound, vec![0, 10]);
    }

    /// Unbinding a slot leaves it holding nothing, which is not a resource.
    #[test]
    fn unbinding_leaves_nothing_rather_than_a_stale_resource() {
        let mut table = SlotTable::<BufferBinding>::default();
        table.set(2, Some(buffer(1)));
        table.set(2, None);
        assert_eq!(table.get(2), None);
        assert_eq!(table.bound().count(), 0);
    }

    /// Without a reflection every bound slot is unknown, and unknown conflicts
    /// with everything.
    #[test]
    fn an_unreflected_pipeline_makes_every_bound_slot_unknown() {
        let mut state = ComputeEncoderState::default();
        state.tables.buffers.set(0, Some(buffer(1)));
        state.tables.textures.set(4, Some(texture(2)));
        state.tables.samplers.set(0, Some(texture(3)));

        let mut out = Vec::new();
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 2, "samplers bind no memory");
        for part in &out {
            assert_eq!(part.mode, AccessMode::Unknown);
            assert_eq!(part.extent, ParticipationExtent::Whole);
            assert!(part.mode.writes(), "unknown must conflict with a reader");
        }
    }

    /// With one, a slot the pipeline does not reference contributes nothing and
    /// the rest contribute what the shader does.
    #[test]
    fn a_reflection_narrows_the_footprint_to_what_the_shader_touches() {
        let mut state = ComputeEncoderState::default();
        state.tables.buffers.set(0, Some(buffer(1)));
        state.tables.buffers.set(1, Some(buffer(2)));
        state.tables.buffers.set(2, Some(buffer(3)));

        let usage = BindingUsage::new(
            vec![Some(AccessMode::Read), None, Some(AccessMode::Write)],
            Vec::new(),
        );
        let mut out = Vec::new();
        state.footprint_into(Some(&usage), &mut out);
        state.declare();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].resource, res(1));
        assert_eq!(out[0].mode, AccessMode::Read);
        assert_eq!(out[1].resource, res(3));
        assert_eq!(out[1].mode, AccessMode::Write);
    }

    /// A slot past the reflection's own length is not referenced by the
    /// pipeline, so it contributes nothing — rather than falling back to
    /// unknown, which would make a long-tailed bind table expensive forever.
    #[test]
    fn a_slot_beyond_the_reflection_is_unreferenced() {
        let mut state = ComputeEncoderState::default();
        state.tables.buffers.set(9, Some(buffer(1)));
        let usage = BindingUsage::new(vec![Some(AccessMode::Read)], Vec::new());
        let mut out = Vec::new();
        state.footprint_into(Some(&usage), &mut out);
        state.declare();
        assert!(out.is_empty());
    }

    /// The two render stages are separate slots with the same index, and a
    /// resource read by one and written by the other keeps both.
    #[test]
    fn the_two_render_stages_do_not_share_slot_numbers() {
        let mut state = RenderEncoderState::default();
        state
            .stage_mut(ShaderStage::Vertex)
            .buffers
            .set(3, Some(buffer(1)));
        state
            .stage_mut(ShaderStage::Fragment)
            .buffers
            .set(3, Some(buffer(1)));
        assert_eq!(
            state.stage(ShaderStage::Vertex).buffers.get(3),
            Some(buffer(1))
        );

        let read = BindingUsage::new(vec![None, None, None, Some(AccessMode::Read)], Vec::new());
        let write = BindingUsage::new(vec![None, None, None, Some(AccessMode::Write)], Vec::new());
        let mut out = Vec::new();
        state.footprint_into(Some(&read), Some(&write), &mut out);
        state.declare();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].mode, AccessMode::Read);
        assert_eq!(out[1].mode, AccessMode::Write);
        assert_eq!(out[0].resource, out[1].resource);
    }

    /// A plural bind fills consecutive slots from `first`, which is the only
    /// thing the record says about where its entries land.
    #[test]
    fn a_plural_bind_fills_consecutive_slots_from_first() {
        let mut state = ComputeEncoderState::default();
        let arena = [buffer(1), buffer(2), buffer(3)];
        state.apply(
            &ComputeOp::BindBuffers {
                first: 4,
                entries: BindSpan { start: 0, len: 3 },
            },
            &arena,
            &[],
        );
        assert_eq!(state.tables.buffers.get(4), Some(buffer(1)));
        assert_eq!(state.tables.buffers.get(6), Some(buffer(3)));
        assert_eq!(state.tables.buffers.get(3), None);
    }

    /// A window naming entries the arena does not have binds nothing. The two
    /// are built together, so a mismatch is a bug in this crate — and losing a
    /// whole packet over it would lose work the guest is waiting on.
    #[test]
    fn a_window_past_the_arena_binds_nothing() {
        let mut state = ComputeEncoderState::default();
        state.apply(
            &ComputeOp::BindBuffers {
                first: 0,
                entries: BindSpan { start: 2, len: 3 },
            },
            &[buffer(1)],
            &[],
        );
        assert_eq!(state.tables.buffers.bound().count(), 0);
    }

    /// A rebind moves the offset of whatever the slot holds. It names no
    /// buffer, so the slot keeps the one bound before it.
    #[test]
    fn a_rebind_moves_the_offset_and_keeps_the_buffer() {
        let mut state = ComputeEncoderState::default();
        state.apply(
            &ComputeOp::BindBuffers {
                first: 2,
                entries: BindSpan { start: 0, len: 1 },
            },
            &[buffer(1)],
            &[],
        );
        state.apply(
            &ComputeOp::RebindBufferOffset {
                index: 2,
                offset: 0x900,
                stride: None,
            },
            &[],
            &[],
        );
        let bound = state.tables.buffers.get(2).expect("still bound");
        assert_eq!(bound.buffer, Some(res(1)));
        assert_eq!(bound.offset, 0x900);
        assert_eq!(
            bound.stride, None,
            "the plain form carries no stride and clears none"
        );
    }

    /// The strided form sets one; the plain form after it leaves it alone.
    /// `setBufferOffset:atIndex:` moves the offset and says nothing about the
    /// attribute stride, so a slot that lost its stride there would feed the
    /// next draw a vertex layout the guest never changed.
    #[test]
    fn a_plain_rebind_does_not_clear_a_stride_the_strided_form_set() {
        let mut state = ComputeEncoderState::default();
        state.apply(
            &ComputeOp::BindBuffers {
                first: 0,
                entries: BindSpan { start: 0, len: 1 },
            },
            &[buffer(1)],
            &[],
        );
        state.apply(
            &ComputeOp::RebindBufferOffset {
                index: 0,
                offset: 0x10,
                stride: Some(32),
            },
            &[],
            &[],
        );
        state.apply(
            &ComputeOp::RebindBufferOffset {
                index: 0,
                offset: 0x20,
                stride: None,
            },
            &[],
            &[],
        );
        let bound = state.tables.buffers.get(0).expect("bound");
        assert_eq!((bound.offset, bound.stride), (0x20, Some(32)));
    }

    /// A rebind of a slot holding nothing binds nothing. The record names no
    /// buffer, so there is none to invent.
    #[test]
    fn a_rebind_of_an_unbound_slot_binds_nothing() {
        let mut state = ComputeEncoderState::default();
        state.apply(
            &ComputeOp::RebindBufferOffset {
                index: 0,
                offset: 0x40,
                stride: None,
            },
            &[],
            &[],
        );
        assert_eq!(state.tables.buffers.get(0), None);
    }

    /// The vertex-stride bind has no fragment form, so it lands in the vertex
    /// table however the record reached the encoder.
    #[test]
    fn the_stride_bind_is_the_vertex_stages() {
        let mut state = RenderEncoderState::default();
        state.apply(
            &RenderOp::BindBuffersWithStride {
                first: 0,
                entries: BindSpan { start: 0, len: 1 },
            },
            &[buffer(1)],
            &[],
        );
        assert_eq!(
            state.stage(ShaderStage::Vertex).buffers.get(0),
            Some(buffer(1))
        );
        assert_eq!(state.stage(ShaderStage::Fragment).buffers.get(0), None);
    }

    /// A sampler bind lands in the sampler table, never the texture one: the
    /// two carry the same entry layout and only the record says which is which.
    #[test]
    fn a_sampler_bind_never_reaches_the_texture_table() {
        let mut state = RenderEncoderState::default();
        state.apply(
            &RenderOp::BindSamplers {
                stage: ShaderStage::Fragment,
                first: 0,
                entries: BindSpan { start: 0, len: 1 },
            },
            &[],
            &[texture(3)],
        );
        assert_eq!(
            state.stage(ShaderStage::Fragment).samplers.get(0),
            Some(texture(3))
        );
        assert_eq!(state.stage(ShaderStage::Fragment).textures.get(0), None);
        let mut out = Vec::new();
        state.footprint_into(None, None, &mut out);
        state.declare();
        assert!(out.is_empty(), "samplers bind no memory");
    }

    /// The footprint is written into the caller's buffer rather than replacing
    /// it, and a second draw over an unchanged table adds nothing to it.
    ///
    /// Both halves matter to the same loop. A draw declares its bindings, and
    /// every participation it declares is a namespace resolution, a residency
    /// lookup and — for a write, which an unreflected slot is — a reserved
    /// content version. A thousand draws with no bind between them would
    /// reserve a thousand versions of one buffer, of which one survives.
    #[test]
    fn a_second_draw_over_an_unchanged_table_declares_nothing() {
        let mut state = ComputeEncoderState::default();
        state.tables.buffers.set(0, Some(buffer(1)));
        let mut out = vec![];
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 1);
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 1, "the second draw's bindings are already in it");

        // A bind that changes nothing changes nothing here either: guests
        // re-bind whole tables between draws.
        state.tables.buffers.set(0, Some(buffer(1)));
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 1);

        // A bind of something else is a new participation, and only that slot's.
        state.tables.buffers.set(3, Some(buffer(2)));
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].resource, res(2));
    }

    /// A new pipeline makes every bound slot a fresh question: what a slot
    /// contributes is the pipeline's answer, and a table that stayed declared
    /// across a pipeline change would hold the previous shader's answer.
    #[test]
    fn a_new_pipeline_re_declares_every_bound_slot() {
        let mut state = ComputeEncoderState::default();
        state.apply(
            &ComputeOp::BindBuffers {
                first: 0,
                entries: BindSpan { start: 0, len: 1 },
            },
            &[buffer(1)],
            &[],
        );
        let mut out = vec![];
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 1);

        // The same pipeline is not a change.
        state.apply(&ComputeOp::SetPipeline { pipeline: res(9) }, &[], &[]);
        state.apply(&ComputeOp::SetPipeline { pipeline: res(9) }, &[], &[]);
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 2, "the first bind of it is a change");

        state.apply(&ComputeOp::SetPipeline { pipeline: res(8) }, &[], &[]);
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 3);
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 3);
    }

    /// A render pipeline re-declares both stages. One stage kept would feed the
    /// next draw the previous shader's answer about half its bindings.
    #[test]
    fn a_new_render_pipeline_re_declares_both_stages() {
        let mut state = RenderEncoderState::default();
        state
            .stage_mut(ShaderStage::Vertex)
            .buffers
            .set(0, Some(buffer(1)));
        state
            .stage_mut(ShaderStage::Fragment)
            .buffers
            .set(0, Some(buffer(2)));
        let mut out = vec![];
        state.footprint_into(None, None, &mut out);
        state.declare();
        assert_eq!(out.len(), 2);
        state.footprint_into(None, None, &mut out);
        state.declare();
        assert_eq!(out.len(), 2);

        state.apply(&RenderOp::SetPipeline { pipeline: res(9) }, &[], &[]);
        state.footprint_into(None, None, &mut out);
        state.declare();
        assert_eq!(out.len(), 4);
    }

    /// **A slot re-bound after a pipeline change is an ordinary frame, and it
    /// used to be a panic.**
    ///
    /// `declared` is allowed to be shorter than `slots` — that is how a
    /// pipeline change takes back every declaration without walking the table
    /// — and `set` indexed it anyway, guarded only by `slots`'s length. So
    /// bind, draw, bind a different pipeline, re-bind the slot: the pipeline
    /// change emptied `declared`, the slot was still inside `slots`, and the
    /// write went past the end of a zero-length vector.
    ///
    /// The guest does exactly this. It binds a pipeline per draw and re-binds
    /// its buffer table between them, so the sequence is not a corner — it is
    /// what a second material in one encoder looks like.
    #[test]
    fn a_slot_rebound_after_a_pipeline_change_is_not_a_panic() {
        let mut state = RenderEncoderState::default();
        state
            .stage_mut(ShaderStage::Vertex)
            .buffers
            .set(0, Some(buffer(1)));
        let mut out = vec![];
        state.footprint_into(None, None, &mut out);
        state.declare();
        assert_eq!(out.len(), 1, "the draw declares the bound slot");

        // A different pipeline: every slot's contribution is a fresh question.
        state.apply(&RenderOp::SetPipeline { pipeline: res(9) }, &[], &[]);
        // And the guest re-binds the table under it.
        state
            .stage_mut(ShaderStage::Vertex)
            .buffers
            .set(0, Some(buffer(2)));
        state.footprint_into(None, None, &mut out);
        state.declare();
        assert_eq!(
            out.len(),
            2,
            "the new binding is declared under the new pipeline"
        );
        state.footprint_into(None, None, &mut out);
        state.declare();
        assert_eq!(out.len(), 2, "and only once");
    }

    /// Re-binding what the slot already holds, after a pipeline change, still
    /// takes the declaration the pipeline change already took.
    ///
    /// The other half of the same encoding: `set` returns early when the value
    /// is unchanged, so the slot's declared flag is whatever the pipeline
    /// change left it — which is undeclared, and must stay so.
    #[test]
    fn an_unchanged_rebind_after_a_pipeline_change_still_declares() {
        let mut state = RenderEncoderState::default();
        state
            .stage_mut(ShaderStage::Vertex)
            .buffers
            .set(0, Some(buffer(1)));
        let mut out = vec![];
        state.footprint_into(None, None, &mut out);
        state.declare();
        assert_eq!(out.len(), 1);
        state.apply(&RenderOp::SetPipeline { pipeline: res(9) }, &[], &[]);
        state
            .stage_mut(ShaderStage::Vertex)
            .buffers
            .set(0, Some(buffer(1)));
        state.footprint_into(None, None, &mut out);
        state.declare();
        assert_eq!(
            out.len(),
            2,
            "the pipeline changed, so the slot is a fresh question"
        );
    }

    /// Unbinding a declared slot and binding it again declares it again: the
    /// resource changed, whatever the footprint used to hold.
    #[test]
    fn a_slot_rebound_after_being_declared_is_declared_again() {
        let mut state = ComputeEncoderState::default();
        state.tables.buffers.set(0, Some(buffer(1)));
        let mut out = vec![];
        state.footprint_into(None, &mut out);
        state.declare();
        state.tables.buffers.set(0, None);
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 1, "an unbound slot contributes nothing");
        state.tables.buffers.set(0, Some(buffer(1)));
        state.footprint_into(None, &mut out);
        state.declare();
        assert_eq!(out.len(), 2);
    }
}
