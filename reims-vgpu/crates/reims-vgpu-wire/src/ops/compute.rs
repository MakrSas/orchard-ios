//! Compute encoder records.
//!
//! The records a `PGSerializerComputeCommandEncoder` writes, each the shared
//! 8-byte [`crate::op::OpHeader`] followed by a per-opcode payload. Derived by
//! calling the Metal method with distinctive arguments and reading the bytes
//! back; the fixture that pins each is named in its doc. See `oracle/oracle.m`'s
//! `computeCases`.
//!
//! # A third opcode space
//!
//! Compute opcodes run `0xc8`–`0xe5`, as far from the render encoder's
//! `0x00`–`0x8a` as the blit encoder's `0x12c`–`0x13e` is from both. Each
//! encoder class numbers its own records and the segment header says which
//! class wrote a segment ([`crate::ops::segment`]), so the three spaces never
//! have to be told apart by content.
//!
//! # The bind records are the render encoder's, byte for byte
//!
//! `setBuffer:offset:atIndex:`, `setTexture:atIndex:` and `setSamplerState:`
//! write the same `[u32 first][u32 count]` head and the same entry widths as
//! their render counterparts, at compute opcodes. Rather than restate the
//! layout, this module re-exports the declarations — one declaration, so drift
//! between the two is impossible rather than merely detectable.
//!
//! # Reversals to watch
//!
//! Two records put their arguments in the opposite order to the selector, and
//! both are the kind of thing a reader reconstructs wrongly from the API:
//! [`DispatchIndirect`] writes `threadsPerThreadgroup` *before* the indirect
//! buffer, and [`StageInRegion`] writes an `MTLRegion`'s **size before its
//! origin** though the struct declares origin first.
//!
//! # Relationship to `reims_vgpu::runtime::decode::compute`
//!
//! Recorded per record where the two disagree. Nothing here changes that
//! module; the divergences are findings.

use crate::le::{U16le, U32le, U64le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

pub use crate::ops::render::{
    BindHeader, BufferBind, BufferOffset, BufferOffsetStride, BufferStrideBind, RefBind,
    SamplerLodBind,
};

// --- 0xc8 / 0xca the direct dispatches -------------------------------------

pub const OPCODE_DISPATCH_THREADGROUPS: u32 = 0xc8;
pub const OPCODE_DISPATCH_THREADS: u32 = 0xca;
pub const DISPATCH_TOTAL_LEN: u32 = 56;

/// Six `u64`: the grid, then the threadgroup.
///
/// `dispatchThreadgroups:` and `dispatchThreads:` write the identical record and
/// differ only in opcode — which is the whole difference between them, because
/// the first count is threadgroups in one and threads in the other. Fixtures
/// `compute_dispatch_threadgroups` and `compute_dispatch_threads`, both with six
/// distinct values `0x11`–`0x66` so no two of the six fields can be confused.
///
/// Nothing is narrowed here, unlike almost every other record in this protocol.
#[repr(C)]
#[derive(Debug)]
pub struct Dispatch {
    pub groups_width: U64le,
    pub groups_height: U64le,
    pub groups_depth: U64le,
    pub threads_width: U64le,
    pub threads_height: U64le,
    pub threads_depth: U64le,
}

// SAFETY: six align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for Dispatch {}

#[inline]
pub fn is_dispatch(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_DISPATCH_THREADGROUPS | OPCODE_DISPATCH_THREADS
    )
}

pub fn dispatch<'a>(op: &Op<'a>) -> Result<&'a Dispatch, WireError> {
    debug_assert!(is_dispatch(op.opcode()));
    view::<Dispatch>(op.payload)
}

// --- 0xc9 dispatchThreadgroupsWithIndirectBuffer:… -------------------------

pub const OPCODE_DISPATCH_THREADGROUPS_INDIRECT: u32 = 0xc9;
pub const DISPATCH_THREADGROUPS_INDIRECT_TOTAL_LEN: u32 = 44;

/// A dispatch whose grid comes from a buffer.
///
/// **The record reverses the selector.** `threadsPerThreadgroup` is the last
/// argument and it is written first; the buffer, the selector's first argument,
/// is written last. Fixture `compute_dispatch_threadgroups_indirect` (threads
/// `0x44`/`0x55`/`0x66`, buffer 5151 at `0x1111`).
///
/// Its sibling `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:` is
/// [`DispatchThreadsIndirect`] below, at its own opcode and its own layout.
#[repr(C)]
#[derive(Debug)]
pub struct DispatchIndirect {
    pub threads_width: U64le,
    pub threads_height: U64le,
    pub threads_depth: U64le,
    pub indirect_buffer_offset: U64le,
    pub indirect_buffer_ref: U32le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DispatchIndirect {}

pub fn dispatch_indirect<'a>(op: &Op<'a>) -> Result<&'a DispatchIndirect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DISPATCH_THREADGROUPS_INDIRECT);
    view::<DispatchIndirect>(op.payload)
}

// --- 0xe6 dispatchThreadsWithIndirectBuffer:indirectBufferOffset: ----------

pub const OPCODE_DISPATCH_THREADS_INDIRECT: u32 = 0xe6;
pub const DISPATCH_THREADS_INDIRECT_TOTAL_LEN: u32 = 20;

/// A non-uniform-threadgroup dispatch whose entire grid comes from a buffer.
///
/// The threadgroup-granular sibling [`DispatchIndirect`] carries
/// `threadsPerThreadgroup` because its selector takes it; this one does not,
/// because its selector does not. Twenty bytes: header, `u64` offset, `u32`
/// ref. Fixture `compute_dispatch_threads_indirect` (buffer 5151 at `0x2222`).
///
/// # It reverses the other indirect dispatch's field order
///
/// [`DispatchIndirect`] writes the buffer ref **last**, after the size and the
/// offset. This record has no size, and it writes the offset first and the ref
/// second — the same relative order, so a reader that strips the size from the
/// threadgroup form lands on this one correctly. `setStageInRegionWithIndirectBuffer:`
/// is the counterexample worth knowing: it writes ref *then* offset.
///
/// # It emits only under `-setSupportsDispatchThreadsIndirect:`
///
/// With the flag off Apple's serializer **fails an assertion** rather than
/// returning silently, so this selector's manifest row read
/// `REFUSED_BY_SERIALIZER` — a claim about Apple, and wrong. The capability
/// sweep that exists to catch that could not see it: the sweep diffs the two
/// passes' `silent` lists and an assertion lands on `unsupported` instead. What
/// found it is `capability_content_deltas`, which diffs the records themselves.
#[repr(C)]
#[derive(Debug)]
pub struct DispatchThreadsIndirect {
    pub indirect_buffer_offset: U64le,
    pub indirect_buffer_ref: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DispatchThreadsIndirect {}

pub fn dispatch_threads_indirect<'a>(
    op: &Op<'a>,
) -> Result<&'a DispatchThreadsIndirect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DISPATCH_THREADS_INDIRECT);
    view::<DispatchThreadsIndirect>(op.payload)
}

// --- 0xe3 insertCompressedTextureReinterpretationFlush ---------------------

pub const OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH: u32 = 0xe3;
pub const INSERT_COMPRESSED_TEXTURE_FLUSH_TOTAL_LEN: u32 = 8;

/// Mark compressed-texture reinterpretation state stale, with no operands.
///
/// The header alone — eight bytes, and the opcode is the entire content, the
/// same shape as the four control-flow markers. No body type, for the reason
/// [`is_control_flow_marker`] gives. Fixture
/// `compute_insert_compressed_texture_reinterpretation_flush`.
///
/// # It emits only under `-setSupportsInsertCompressedTextureReinterpretationFlush:`
///
/// Same shape as [`DispatchThreadsIndirect`]: an assertion at the default state,
/// a record under the flag, and a manifest row that said Apple refuses it. Note
/// that its blit-encoder siblings `invalidateCompressedTexture*` are gated on
/// `BlitEncoderSPI` instead — one feature, two flags, so "the family's flag" is
/// not a thing to reason from.
#[inline]
pub fn is_insert_compressed_texture_flush(opcode: u32) -> bool {
    opcode == OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH
}

// --- 0xcb / 0xcc / 0xce the bind records -----------------------------------

pub const OPCODE_SET_BUFFER: u32 = 0xcb;
pub const OPCODE_SET_SAMPLER: u32 = 0xcc;
pub const OPCODE_SET_TEXTURE: u32 = 0xce;

/// The buffer bind, at the compute encoder's opcode.
///
/// `setBytes:length:atIndex:` shares it: the serializer stages the bytes through
/// the command stream's allocator and records the *staging buffer's* ref and
/// offset, exactly as the render encoder does. Fixtures `compute_set_buffer`,
/// `compute_set_buffers_range` (two slots, two offsets) and `compute_set_bytes`
/// (staging buffer 8181 at `0x9999`).
pub fn buffer_binds<'a>(op: &Op<'a>) -> Result<(&'a BindHeader, &'a [BufferBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_BUFFER);
    bind_entries::<BufferBind>(op.payload)
}

/// Split a bind record into its head and `count` entries.
///
/// The types are the render encoder's; the *walk* is repeated rather than
/// delegated, because `crate::ops::render`'s entry point asserts a render
/// opcode and a compute record does not have one. Sharing the declaration and
/// not the dispatch is the point.
fn bind_entries<T: Wire>(payload: &[u8]) -> Result<(&BindHeader, &[T]), WireError> {
    let (head, rest) = crate::view::split::<BindHeader>(payload)?;
    let entries = crate::view::view_slice::<T>(rest, head.count.get() as usize)?;
    Ok((head, entries))
}

/// The texture or sampler bind: a ref per slot, four bytes each.
///
/// Fixtures `compute_set_texture`, `compute_set_textures_range` (two distinct
/// texture refs, so the trailing array is shown to be an array),
/// `compute_set_sampler_state`.
#[inline]
pub fn is_ref_bind(opcode: u32) -> bool {
    matches!(opcode, OPCODE_SET_SAMPLER | OPCODE_SET_TEXTURE)
}

pub fn ref_binds<'a>(op: &Op<'a>) -> Result<(&'a BindHeader, &'a [RefBind]), WireError> {
    debug_assert!(is_ref_bind(op.opcode()));
    bind_entries::<RefBind>(op.payload)
}

// --- 0xcd setSamplerState:lodMinClamp:lodMaxClamp:atIndex: -----------------

pub const OPCODE_SET_SAMPLER_LOD: u32 = 0xcd;

/// The compute stage's sampler binds with level-of-detail clamps.
///
/// The entry is the render stage's [`crate::ops::render::SamplerLodBind`],
/// re-exported at the top of this module with the other bind entries rather
/// than declared again: same three fields, same twelve-byte stride, and two
/// identical declarations are two things that can drift.
///
/// **The clamps are per entry, not per record.** With `count == 1` a pair of
/// floats after the ref could be either; `compute_set_sampler_states_lod` binds
/// two slots with four distinct clamps (0.25/0.75 and 0.125/0.875) in a 40-byte
/// record, which is the head plus two twelve-byte entries. That is the case the
/// singular form cannot supply.
///
/// The clamps are 32-bit floats where every viewport depth bound in this
/// protocol is 64-bit. Both widths are carried and they are not interchangeable.
pub fn sampler_lod_binds<'a>(
    op: &Op<'a>,
) -> Result<(&'a BindHeader, &'a [SamplerLodBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_SAMPLER_LOD);
    bind_entries::<SamplerLodBind>(op.payload)
}

// --- 0xcf setBufferOffset:atIndex: -----------------------------------------

// --- 0xd9 / 0xda the binds that carry an attribute stride ------------------

pub const OPCODE_SET_BUFFER_STRIDE: u32 = 0xd9;
pub const OPCODE_SET_BUFFER_OFFSET_STRIDE: u32 = 0xda;
pub const SET_BUFFER_OFFSET_STRIDE_TOTAL_LEN: u32 = 28;

/// The compute stage's attribute-stride binds, gated on
/// `-supportsDynamicAttributeStride`.
///
/// Three selectors and two opcodes: the singular and plural buffer binds both
/// write `0xd9`, and `setBufferOffset:attributeStride:atIndex:` writes `0xda`.
/// Layouts are the render stage's [`BufferStrideBind`] and
/// [`BufferOffsetStride`] exactly — twenty-byte entries behind a
/// [`BindHeader`], and an index with two `u64` — so they are re-exported rather
/// than restated.
///
/// These three had been captured at the *default* capability state, come back
/// silent, and been recorded as records Apple does not emit. The render
/// encoder's four forms were driven when `DynamicAttributeStride` was chased
/// and these were missed, which is the difference between driving a family and
/// driving a flag. Fixtures `compute_set_buffer_stride`,
/// `compute_set_buffers_stride_range` (two entries at two offsets *and* two
/// strides, which is what shows both are per entry) and
/// `compute_set_buffer_offset_stride`.
///
/// Unlike the render pair, `reims_vgpu::runtime::decode::compute` had both
/// opcodes and the twenty-byte entry already, and the capture agrees field for
/// field. `0xa5`/`0xa6` on the render encoder were wrong and cost every strided
/// vertex bind; these were right and untested.
#[inline]
pub fn is_buffer_stride_bind(opcode: u32) -> bool {
    opcode == OPCODE_SET_BUFFER_STRIDE
}

pub fn buffer_stride_binds<'a>(
    op: &Op<'a>,
) -> Result<(&'a BindHeader, &'a [BufferStrideBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_BUFFER_STRIDE);
    let (head, rest) = crate::view::split::<BindHeader>(op.payload)?;
    let entries = crate::view::view_slice::<BufferStrideBind>(rest, head.count.get() as usize)?;
    Ok((head, entries))
}

pub fn buffer_offset_stride<'a>(op: &Op<'a>) -> Result<&'a BufferOffsetStride, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_BUFFER_OFFSET_STRIDE);
    view::<BufferOffsetStride>(op.payload)
}

pub const OPCODE_SET_BUFFER_OFFSET: u32 = 0xcf;
pub const SET_BUFFER_OFFSET_TOTAL_LEN: u32 = 20;

/// Rebind an already-bound buffer's offset. The render encoder's
/// [`BufferOffset`] record at a compute opcode; fixture
/// `compute_set_buffer_offset` (index 6, offset `0x5678`).
pub fn set_buffer_offset<'a>(op: &Op<'a>) -> Result<&'a BufferOffset, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_BUFFER_OFFSET);
    view::<BufferOffset>(op.payload)
}

// --- 0xd0 setComputePipelineState: -----------------------------------------

pub const OPCODE_SET_PIPELINE_STATE: u32 = 0xd0;
pub const SET_PIPELINE_STATE_TOTAL_LEN: u32 = 12;

/// A single object ref, and the whole payload. Fixture
/// `compute_set_pipeline_state` (6161).
///
/// [`OPCODE_UPDATE_FENCE`] and [`OPCODE_WAIT_FOR_FENCE`] write the same four
/// bytes — see [`Ref`], which is what this returns.
pub fn set_pipeline_state<'a>(op: &Op<'a>) -> Result<&'a Ref, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_PIPELINE_STATE);
    view::<Ref>(op.payload)
}

/// One object ref: the whole payload of the pipeline-state and fence records.
#[repr(C)]
#[derive(Debug)]
pub struct Ref {
    pub object_ref: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for Ref {}

// --- 0xd1 setStageInRegion: ------------------------------------------------

pub const OPCODE_SET_STAGE_IN_REGION: u32 = 0xd1;
pub const SET_STAGE_IN_REGION_TOTAL_LEN: u32 = 56;

/// An `MTLRegion`, **size first**.
///
/// `MTLRegion` declares `origin` then `size`, and this record writes them the
/// other way round. Fixture `compute_set_stage_in_region` uses origin
/// `0x11`/`0x22`/`0x33` and size `0x44`/`0x55`/`0x66`, six distinct values, and
/// the `0x44` lands at `+0`. Every blit record that carries an origin and a size
/// writes them in the declared order, so this is not a protocol-wide
/// convention — it is this record's.
#[repr(C)]
#[derive(Debug)]
pub struct StageInRegion {
    pub size_width: U64le,
    pub size_height: U64le,
    pub size_depth: U64le,
    pub origin_x: U64le,
    pub origin_y: U64le,
    pub origin_z: U64le,
}

// SAFETY: six align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for StageInRegion {}

pub fn set_stage_in_region<'a>(op: &Op<'a>) -> Result<&'a StageInRegion, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_STAGE_IN_REGION);
    view::<StageInRegion>(op.payload)
}

pub const OPCODE_SET_STAGE_IN_REGION_INDIRECT: u32 = 0xd2;
pub const SET_STAGE_IN_REGION_INDIRECT_TOTAL_LEN: u32 = 20;

/// The stage-in region, read from a buffer instead of the record.
///
/// **The ref leads and the offset trails here**, which is the opposite of
/// [`DispatchIndirect`] two opcodes earlier. Two indirect records on one
/// encoder with opposite field orders, so neither can be inferred from the
/// other. Fixture `compute_set_stage_in_region_indirect` (buffer 5151 at
/// `0x1111`).
#[repr(C)]
#[derive(Debug)]
pub struct StageInRegionIndirect {
    pub indirect_buffer_ref: U32le,
    pub indirect_buffer_offset: U64le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for StageInRegionIndirect {}

pub fn set_stage_in_region_indirect<'a>(
    op: &Op<'a>,
) -> Result<&'a StageInRegionIndirect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_STAGE_IN_REGION_INDIRECT);
    view::<StageInRegionIndirect>(op.payload)
}

// --- 0xd3 setThreadgroupMemoryLength:atIndex: ------------------------------

pub const OPCODE_SET_THREADGROUP_MEMORY_LENGTH: u32 = 0xd3;
pub const SET_THREADGROUP_MEMORY_LENGTH_TOTAL_LEN: u32 = 20;

/// How much threadgroup memory a slot gets. Fixture
/// `compute_set_threadgroup_memory_length` (length `0x1100` at index 3).
///
/// The length leads as a `u64` and the index trails as a `u32`, which is the
/// opposite arrangement to [`BufferOffset`] — where the index leads and the
/// value trails — even though both records are "a number and a slot".
#[repr(C)]
#[derive(Debug)]
pub struct ThreadgroupMemoryLength {
    pub length: U64le,
    pub index: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ThreadgroupMemoryLength {}

pub fn set_threadgroup_memory_length<'a>(
    op: &Op<'a>,
) -> Result<&'a ThreadgroupMemoryLength, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_THREADGROUP_MEMORY_LENGTH);
    view::<ThreadgroupMemoryLength>(op.payload)
}

// --- 0xd4 / 0xd5 fences ----------------------------------------------------

pub const OPCODE_UPDATE_FENCE: u32 = 0xd4;
pub const OPCODE_WAIT_FOR_FENCE: u32 = 0xd5;
pub const FENCE_TOTAL_LEN: u32 = 12;

/// A compute fence record is **the ref alone**, four bytes.
///
/// The render encoder's fence record is 16 bytes and carries a `stages` mask
/// beside the ref ([`crate::ops::render::Fence`]); this one has no such field
/// and no room for one. Same selector name, same object, different record —
/// which is exactly the shape of mistake a decoder makes when it shares a
/// layout across encoders. Fixtures `compute_update_fence` and
/// `compute_wait_for_fence` (6464).
#[inline]
pub fn is_fence(opcode: u32) -> bool {
    matches!(opcode, OPCODE_UPDATE_FENCE | OPCODE_WAIT_FOR_FENCE)
}

pub fn fence<'a>(op: &Op<'a>) -> Result<&'a Ref, WireError> {
    debug_assert!(is_fence(op.opcode()));
    view::<Ref>(op.payload)
}

// --- 0xd6 / 0xd7 barriers --------------------------------------------------

pub const OPCODE_MEMORY_BARRIER_RESOURCES: u32 = 0xd6;

/// A barrier over a named list of resources — count then refs, and **no stage
/// masks**.
///
/// The render encoder's version carries `afterStages` and `beforeStages`
/// because its selector takes them; the compute selector does not, and the
/// record is four bytes shorter rather than four bytes of zero. Fixture
/// `compute_memory_barrier_resources` (count 2, a buffer at 5151 and a texture
/// at 4343, so the array is shown to hold resources of any kind).
#[repr(C)]
#[derive(Debug)]
pub struct MemoryBarrierResources {
    pub count: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for MemoryBarrierResources {}

pub fn memory_barrier_resources<'a>(
    op: &Op<'a>,
) -> Result<(&'a MemoryBarrierResources, &'a [RefBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_MEMORY_BARRIER_RESOURCES);
    let (head, rest) = crate::view::split::<MemoryBarrierResources>(op.payload)?;
    let refs = crate::view::view_slice::<RefBind>(rest, head.count.get() as usize)?;
    Ok((head, refs))
}

pub const OPCODE_MEMORY_BARRIER_SCOPE: u32 = 0xd7;
pub const MEMORY_BARRIER_SCOPE_TOTAL_LEN: u32 = 12;

/// A barrier over a scope: two bytes of a four-byte payload.
///
/// The other two are **never written** — `compute_memory_barrier_scope` came
/// back `04 00 AA AA` — so they hold whatever the guest's ring last contained
/// and belong to no field. The render encoder's scope barrier fills all four,
/// because it has two stage masks to put there.
///
/// `scope` is read as two bytes because two were written. It cannot be shown to
/// be two rather than one beside a zero: `MTLBarrierScope` defines no value
/// above 4. The render encoder's [`crate::ops::render::MemoryBarrierScope`]
/// makes the opposite choice for the same reason, and says so — there the byte
/// at `+1` is followed by two fields, so folding it into `scope` would risk
/// swallowing one.
///
/// # A second selector writes this record, and only sometimes
///
/// `maybeEmitSerialBarrier` — gated on
/// `-setSupportsComputePassDescriptorDispatchType:` — emits this opcode with
/// `scope == 3`, which is `MTLBarrierScopeBuffers | MTLBarrierScopeTextures`.
/// Its name says it emits conditionally and the condition is measured: with the
/// pass's dispatch type left where the encoder starts it,
/// `compute_maybe_emit_serial_barrier` produces the record;
/// `compute_maybe_emit_serial_barrier_concurrent` sets the type to 1 through
/// `setCurrentDispatchType:` first and produces **nothing at all**. So the
/// selector both emits and does not, and its manifest row is `Covered` while a
/// `silent` entry for it also exists — the two are the two cases, not a
/// contradiction.
///
/// One case would have been enough to record an opcode and would have got the
/// contract wrong: it would say a guest calling this always gets a barrier.
///
/// # The guest does not have to call it — a serial pass barriers after **every**
/// dispatch
///
/// With that same flag on, each of the six dispatch and ICB-execute selectors
/// emits two records: its own, then this one at `scope == 3`. Fixtures
/// `compute_dispatch_threadgroups_serial` and
/// `compute_execute_commands_range_serial` pin both halves; the ICB one is there
/// because "every dispatch barriers" and "every one of those selectors barriers"
/// are different claims and an ICB execution is not a dispatch.
///
/// It is the **pass's dispatch type** that decides, not the selector:
/// `compute_dispatch_threadgroups_concurrent` drives the same selector under the
/// same flag after `setCurrentDispatchType:1` and emits one record.
///
/// And the flag has to be on when the **encoder is created**. Forcing it after
/// `initWithCommandBuffer:descriptor:serializer:` produces one record, so the
/// encoder reads the capability at init and a capture that flips it mid-case
/// measures nothing. That cost a probe to learn and is the kind of thing no type
/// encoding shows.
///
/// The consequence for a reader is not a layout: it is that **the number of
/// records a dispatch selector produces is capability- and state-dependent**. A
/// decoder is fine — it sees a stream of records — but anything counting records
/// per guest call, or assuming a compute pass orders its own dispatches, is not.
#[repr(C)]
#[derive(Debug)]
pub struct MemoryBarrierScope {
    pub scope: U16le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for MemoryBarrierScope {}

pub fn memory_barrier_scope<'a>(op: &Op<'a>) -> Result<&'a MemoryBarrierScope, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_MEMORY_BARRIER_SCOPE);
    view::<MemoryBarrierScope>(op.payload)
}

// --- 0xe4 / 0xe5 indirect command buffer execution -------------------------

pub const OPCODE_EXECUTE_COMMANDS_RANGE: u32 = 0xe4;
pub const EXECUTE_COMMANDS_RANGE_TOTAL_LEN: u32 = 28;
pub const OPCODE_EXECUTE_COMMANDS_INDIRECT: u32 = 0xe5;
pub const EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN: u32 = 24;

/// Execute a literal range of an indirect command buffer.
///
/// Byte for byte the render encoder's
/// [`crate::ops::render::ExecuteCommandsRange`] and the blit encoder's
/// [`crate::ops::blit::IcbRange`], at a third opcode. Fixture
/// `compute_execute_commands_range` (ICB 7171, range `0x1100`/`0x2200`).
pub fn execute_commands_range<'a>(
    op: &Op<'a>,
) -> Result<&'a crate::ops::render::ExecuteCommandsRange, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_EXECUTE_COMMANDS_RANGE);
    view::<crate::ops::render::ExecuteCommandsRange>(op.payload)
}

/// Execute an indirect command buffer with the range coming from a buffer.
/// Fixture `compute_execute_commands_indirect` (ICB 7171, buffer 5151 at
/// `0x1111`).
pub fn execute_commands_indirect<'a>(
    op: &Op<'a>,
) -> Result<&'a crate::ops::render::ExecuteCommandsIndirect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_EXECUTE_COMMANDS_INDIRECT);
    view::<crate::ops::render::ExecuteCommandsIndirect>(op.payload)
}

// --- 0xd8 setImageblockWidth:height: ---------------------------------------

pub const OPCODE_SET_IMAGEBLOCK_SIZE: u32 = 0xd8;
pub const SET_IMAGEBLOCK_SIZE_TOTAL_LEN: u32 = 16;

/// The imageblock's dimensions in pixels.
///
/// Both arguments are declared `Q` and both reach the wire as `u32`. Fixtures
/// `compute_set_imageblock_size` (`0x11`, `0x22`) and `..._alt` (`0x3333`,
/// `0x4444`) — driven twice because a record whose two arguments have the same
/// type cannot show which slot is which from one observation.
///
/// Gated on `-setSupportsImageBlocks:`, so it was captured writing nothing and
/// carried `EMITS_NO_OPERATION` until the flag was forced.
#[repr(C)]
#[derive(Debug)]
pub struct ImageblockSize {
    pub width: U32le,
    pub height: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ImageblockSize {}

pub fn set_imageblock_size<'a>(op: &Op<'a>) -> Result<&'a ImageblockSize, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_IMAGEBLOCK_SIZE);
    view::<ImageblockSize>(op.payload)
}

// --- 0xdb writeDescriptor --------------------------------------------------

pub const OPCODE_WRITE_DESCRIPTOR: u32 = 0xdb;
pub const WRITE_DESCRIPTOR_TOTAL_LEN: u32 = 12;

/// The compute pass descriptor, which is one field: the dispatch type.
///
/// # The dispatch type is not sent when it is set
///
/// `setCurrentDispatchType:` writes **nothing** — it is silent under every
/// capability this serializer has, which is measured every capture. The value
/// reaches the wire only here, when the guest asks for the descriptor to be
/// written. So a reader that expected a record per `setCurrentDispatchType:`
/// would see none and conclude the guest never set one.
///
/// Derived rather than assumed: `compute_write_descriptor` runs at the
/// encoder's starting type and reads 0, `compute_write_descriptor_concurrent`
/// sets the type to 1 first and reads 1. Without the second case this word is
/// a zero that could be anything.
#[repr(C)]
#[derive(Debug)]
pub struct PassDescriptor {
    pub dispatch_type: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for PassDescriptor {}

pub fn write_descriptor<'a>(op: &Op<'a>) -> Result<&'a PassDescriptor, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_WRITE_DESCRIPTOR);
    view::<PassDescriptor>(op.payload)
}

// --- 0xdc–0xe2 GPU control flow --------------------------------------------

pub const OPCODE_START_DO_WHILE: u32 = 0xdc;
pub const OPCODE_END_DO_WHILE: u32 = 0xdd;
pub const OPCODE_START_WHILE: u32 = 0xde;
pub const OPCODE_END_WHILE: u32 = 0xdf;
pub const OPCODE_START_IF: u32 = 0xe0;
pub const OPCODE_START_ELSE: u32 = 0xe1;
pub const OPCODE_END_IF: u32 = 0xe2;

pub const CONTROL_FLOW_MARKER_TOTAL_LEN: u32 = 8;
pub const CONTROL_FLOW_PREDICATE_TOTAL_LEN: u32 = 28;

/// The condition three of the seven control-flow records carry.
///
/// `encodeStartIf:offset:comparison:referenceValue:`,
/// `encodeStartWhile:…` and `encodeEndDoWhile:…` declare identical arguments and
/// write identical bodies; only the opcode differs. So the GPU reads a `u32` at
/// `offset` in `buffer_ref`, compares it against `reference_value` under
/// `comparison`, and branches — which makes this the one place in the protocol
/// where the *command stream itself* is data-dependent.
///
/// # Two of these fields are narrower than their selector declares
///
/// `offset` and `comparison` are both `Q` in the type encoding. `offset` reaches
/// the wire as a `u64` and `comparison` as a `u32`, and the record's own length
/// settles it: 8 header + 4 ref + 8 offset + 4 + 4 is exactly 28, so there is no
/// room for a second `u64`. Do not widen `comparison` to match its selector.
///
/// # The comparison ordinal is carried verbatim
///
/// `compute_encode_start_if` passed `0x22` and the record reads `0x22`;
/// `compute_encode_start_while` passed `0x44` and reads `0x44`. Both are outside
/// `MTLCompareFunction`'s defined range of 0–7, so the serializer neither
/// validates nor remaps the value — it is the guest's, and a reader must treat
/// an out-of-range ordinal as guest data rather than as impossible.
///
/// Fixtures `compute_encode_start_if` / `_alt`, `compute_encode_start_while` /
/// `_alt`, `compute_encode_end_do_while` / `_alt`. Each pair moves all four
/// fields, and the two use different buffers so a record that wrote one arm's
/// ref into another's could not read back correct.
///
/// # It emits only under `-setSupportsCommandBufferJump:`
///
/// With the flag off all seven selectors return having written nothing, and
/// this crate recorded that silence as `EMITS_NO_OPERATION` for each. That was
/// a claim about Apple and it was wrong seven times. The capture's per-flag
/// attribution passes name the flag; it is not guessed.
#[repr(C)]
#[derive(Debug)]
pub struct ControlFlowPredicate {
    pub buffer_ref: U32le,
    pub offset: U64le,
    pub comparison: U32le,
    pub reference_value: U32le,
}

// SAFETY: four align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ControlFlowPredicate {}

#[inline]
pub fn is_control_flow_predicate(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_START_IF | OPCODE_START_WHILE | OPCODE_END_DO_WHILE
    )
}

/// The four control-flow records that are **the header alone**.
///
/// `encodeStartElse`, `encodeEndIf`, `encodeEndWhile` and `encodeStartDoWhile`
/// take no arguments and write no payload: eight bytes, opcode and length, and
/// the opcode is the entire content. There is no body type for these, because a
/// zero-field struct would invite a reader to look for fields that are not
/// there.
///
/// Fixtures `compute_encode_start_else`, `compute_encode_end_if`,
/// `compute_encode_end_while`, `compute_encode_start_do_while`.
#[inline]
pub fn is_control_flow_marker(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_START_ELSE | OPCODE_END_IF | OPCODE_END_WHILE | OPCODE_START_DO_WHILE
    )
}

pub fn control_flow_predicate<'a>(op: &Op<'a>) -> Result<&'a ControlFlowPredicate, WireError> {
    debug_assert!(is_control_flow_predicate(op.opcode()));
    view::<ControlFlowPredicate>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OP_HEADER_LEN;
    use core::mem::size_of;

    #[test]
    fn each_fixed_length_record_is_its_body_plus_the_header() {
        for (name, body, total) in [
            ("Dispatch", size_of::<Dispatch>(), DISPATCH_TOTAL_LEN),
            (
                "DispatchIndirect",
                size_of::<DispatchIndirect>(),
                DISPATCH_THREADGROUPS_INDIRECT_TOTAL_LEN,
            ),
            (
                "BufferOffset",
                size_of::<BufferOffset>(),
                SET_BUFFER_OFFSET_TOTAL_LEN,
            ),
            ("Ref", size_of::<Ref>(), SET_PIPELINE_STATE_TOTAL_LEN),
            ("Ref (fence)", size_of::<Ref>(), FENCE_TOTAL_LEN),
            (
                "StageInRegion",
                size_of::<StageInRegion>(),
                SET_STAGE_IN_REGION_TOTAL_LEN,
            ),
            (
                "StageInRegionIndirect",
                size_of::<StageInRegionIndirect>(),
                SET_STAGE_IN_REGION_INDIRECT_TOTAL_LEN,
            ),
            (
                "ThreadgroupMemoryLength",
                size_of::<ThreadgroupMemoryLength>(),
                SET_THREADGROUP_MEMORY_LENGTH_TOTAL_LEN,
            ),
            (
                "ExecuteCommandsRange",
                size_of::<crate::ops::render::ExecuteCommandsRange>(),
                EXECUTE_COMMANDS_RANGE_TOTAL_LEN,
            ),
            (
                "ExecuteCommandsIndirect",
                size_of::<crate::ops::render::ExecuteCommandsIndirect>(),
                EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN,
            ),
            (
                "ControlFlowPredicate",
                size_of::<ControlFlowPredicate>(),
                CONTROL_FLOW_PREDICATE_TOTAL_LEN,
            ),
            (
                "ImageblockSize",
                size_of::<ImageblockSize>(),
                SET_IMAGEBLOCK_SIZE_TOTAL_LEN,
            ),
            (
                "PassDescriptor",
                size_of::<PassDescriptor>(),
                WRITE_DESCRIPTOR_TOTAL_LEN,
            ),
            (
                "DispatchThreadsIndirect",
                size_of::<DispatchThreadsIndirect>(),
                DISPATCH_THREADS_INDIRECT_TOTAL_LEN,
            ),
        ] {
            assert_eq!(
                body + OP_HEADER_LEN,
                total as usize,
                "{name}: body {body} + header does not make {total}"
            );
        }
    }

    /// The scope barrier's body stops two bytes short of its record, and those
    /// two are never written. Widening `scope` to fill them would make the view
    /// read a guest's stale ring bytes.
    #[test]
    fn the_scope_barrier_leaves_two_bytes_it_never_wrote() {
        assert_eq!(
            size_of::<MemoryBarrierScope>() + OP_HEADER_LEN + 2,
            MEMORY_BARRIER_SCOPE_TOTAL_LEN as usize
        );
    }

    /// The compute fence record is *not* the render one, and the difference is
    /// a whole field.
    ///
    /// Sharing a fence layout across encoders is the exact mistake this asserts
    /// against: reading the render `Fence` off a compute record would take four
    /// bytes past the payload as a `stages` mask.
    #[test]
    fn the_compute_fence_is_four_bytes_shorter_than_the_render_one() {
        assert_eq!(size_of::<Ref>() + OP_HEADER_LEN, FENCE_TOTAL_LEN as usize);
        assert_eq!(
            size_of::<crate::ops::render::Fence>(),
            size_of::<Ref>() + 4,
            "the render fence still carries a field the compute one does not"
        );
        assert_eq!(
            FENCE_TOTAL_LEN + 4,
            crate::ops::render::FENCE_TOTAL_LEN,
            "the two fence records are the same size again; one of them is wrong"
        );
    }

    /// The lod clamps live in the entry, so the record grows twelve bytes per
    /// slot. A layout that put them in the head would grow eight once.
    #[test]
    fn a_sampler_lod_entry_is_a_ref_and_two_floats() {
        assert_eq!(size_of::<SamplerLodBind>(), 12);
        assert_eq!(
            OP_HEADER_LEN + size_of::<BindHeader>() + 2 * size_of::<SamplerLodBind>(),
            40,
            "compute_set_sampler_states_lod is 40 bytes for two slots"
        );
    }

    /// No opcode belongs to two shape predicates.
    #[test]
    fn no_compute_opcode_answers_two_shape_predicates() {
        for opcode in 0xc0u32..=0xf0 {
            let hits = [
                is_dispatch(opcode),
                is_ref_bind(opcode),
                is_fence(opcode),
                opcode == OPCODE_DISPATCH_THREADGROUPS_INDIRECT,
                opcode == OPCODE_SET_BUFFER,
                opcode == OPCODE_SET_SAMPLER_LOD,
                opcode == OPCODE_SET_BUFFER_OFFSET,
                opcode == OPCODE_SET_PIPELINE_STATE,
                opcode == OPCODE_SET_STAGE_IN_REGION,
                opcode == OPCODE_SET_STAGE_IN_REGION_INDIRECT,
                opcode == OPCODE_SET_THREADGROUP_MEMORY_LENGTH,
                opcode == OPCODE_MEMORY_BARRIER_RESOURCES,
                opcode == OPCODE_MEMORY_BARRIER_SCOPE,
                is_control_flow_predicate(opcode),
                is_control_flow_marker(opcode),
                opcode == OPCODE_SET_IMAGEBLOCK_SIZE,
                opcode == OPCODE_WRITE_DESCRIPTOR,
                opcode == OPCODE_EXECUTE_COMMANDS_RANGE,
                opcode == OPCODE_EXECUTE_COMMANDS_INDIRECT,
            ]
            .into_iter()
            .filter(|hit| *hit)
            .count();
            assert!(
                hits <= 1,
                "opcode {opcode:#x} answers {hits} shape predicates"
            );
        }
    }

    /// Compute opcodes must not collide with the render encoder's, because a
    /// reader who lost the segment type would have no other way to tell them
    /// apart.
    ///
    /// They are separated by construction today, and this says so rather than
    /// leaving it to be noticed.
    #[test]
    fn the_compute_opcode_space_is_disjoint_from_the_render_one() {
        for opcode in [
            OPCODE_DISPATCH_THREADGROUPS,
            OPCODE_DISPATCH_THREADS,
            OPCODE_DISPATCH_THREADGROUPS_INDIRECT,
            OPCODE_SET_BUFFER,
            OPCODE_SET_SAMPLER,
            OPCODE_SET_TEXTURE,
            OPCODE_SET_SAMPLER_LOD,
            OPCODE_SET_BUFFER_OFFSET,
            OPCODE_SET_PIPELINE_STATE,
            OPCODE_SET_STAGE_IN_REGION,
            OPCODE_SET_STAGE_IN_REGION_INDIRECT,
            OPCODE_SET_THREADGROUP_MEMORY_LENGTH,
            OPCODE_UPDATE_FENCE,
            OPCODE_WAIT_FOR_FENCE,
            OPCODE_MEMORY_BARRIER_RESOURCES,
            OPCODE_MEMORY_BARRIER_SCOPE,
            OPCODE_EXECUTE_COMMANDS_RANGE,
            OPCODE_EXECUTE_COMMANDS_INDIRECT,
        ] {
            assert!(
                opcode > 0x8a,
                "compute opcode {opcode:#x} is inside the render encoder's range"
            );
        }
    }

    /// A guest-controlled count must never produce a slice past the record.
    #[test]
    fn a_count_past_the_record_is_refused() {
        fn record(buf: &mut [u8; 64], opcode: u32, head: &[u8], tail_len: usize) -> usize {
            let total = OP_HEADER_LEN + head.len() + tail_len;
            assert!(total <= buf.len());
            *buf = [0u8; 64];
            buf[..4].copy_from_slice(&opcode.to_le_bytes());
            buf[4..8].copy_from_slice(&(total as u32).to_le_bytes());
            buf[OP_HEADER_LEN..OP_HEADER_LEN + head.len()].copy_from_slice(head);
            total
        }
        let mut buf = [0u8; 64];

        // A sampler-lod bind claiming two entries with one entry's bytes.
        let mut head = [0u8; 8];
        head[4..8].copy_from_slice(&2u32.to_le_bytes());
        let n = record(
            &mut buf,
            OPCODE_SET_SAMPLER_LOD,
            &head,
            size_of::<SamplerLodBind>(),
        );
        let o = crate::op::op(&buf[..n], 0).expect("header fits");
        assert!(matches!(
            sampler_lod_binds(&o),
            Err(WireError::Short { .. })
        ));

        // A resource barrier claiming one ref and supplying none.
        let n = record(
            &mut buf,
            OPCODE_MEMORY_BARRIER_RESOURCES,
            &1u32.to_le_bytes(),
            0,
        );
        let o = crate::op::op(&buf[..n], 0).expect("header fits");
        assert!(matches!(
            memory_barrier_resources(&o),
            Err(WireError::Short { .. })
        ));
    }
}
