//! The guest's indirect command buffer: its serializer-object create body, its command
//! layout, and the encode/decode of one command slot's bytes.
//!
//! Rail-neutral, all of it — this is the guest's own serialization and it reads
//! the same whichever rail executes it. The host `MTLIndirectCommandBuffer` the
//! decode describes lives in [`metal`], which is the only half that needs a
//! Metal device.
//!
//! ## Create (wire)
//!
//! Guest create is serialized by
//! `PGSerializer newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:options:allocator:`
//! into an 88-byte serializer-object body (tag `0x36`) including a 52-byte command
//! **layout** at `+0x1c`. Product materializes a host ICB and caches it per
//! `(task_id, icb_ref)`.
//!
//! ## Command fills — buffer-backed, not stream opcodes
//!
//! There is **no** Reims VGPU compute-stream opcode for
//! `indirectComputeCommandAtIndex` fills. Guest CPU
//! writes into an ICB backing buffer via
//! `PGSerializerIndirectComputeCommand` (setPSO / setKernelBuffer /
//! concurrentDispatch*). Command slots use the layout from create:
//! - type `0x20` = concurrentDispatchThreadgroups, `0x40` = …Threads
//! - pipeline object-list ref at `pipelineStateOffset`
//! - kernel binds at `kernelBufferBindOffset` (0x14 B: ref@0, va@4, gpuva@0xc)
//! - dispatch args at `commandArgumentsOffset` (3×u64 grid + 3×u64 tptg)
//!
//! Product decode of that buffer → [`metal::fill_compute_command`]. Execute
//! (`0xe4`/`0xe5`) re-fills from registered command memory when present. Host
//! Metal fill API remains for tests without guest backing.

use crate::model::DeviceState;
use crate::protocol::endian::{ld32, ld64}; // ld64: 0x1d1 gpu_address + dispatch args
use crate::runtime::decode::resource::{
    decode_serializer_object_descriptor, icb_layout_attribute_stride_slot_count,
    icb_layout_kernel_tg_slot_count, icb_layout_table_len, Descriptor as ResourceDescriptor,
    IcbCommandLayout, IndirectCommandBufferDescriptor, ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE,
    ICB_BUFFER_BIND_STRIDE, ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS,
    ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS, ICB_CMD_TYPE_DRAW, ICB_CMD_TYPE_DRAW_INDEXED,
    ICB_CMD_TYPE_DRAW_INDEXED_PATCHES, ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS,
    ICB_CMD_TYPE_DRAW_MESH_THREADS, ICB_CMD_TYPE_DRAW_PATCHES, ICB_CONCURRENT_DISPATCH_ARGS_LEN,
    ICB_DRAW_INDEXED_PATCHES_ARGS_LEN, ICB_DRAW_MESH_ARGS_LEN, ICB_DRAW_PATCHES_ARGS_LEN,
    ICB_TESSELLATION_FACTOR_LEN, ICB_TG_MEMORY_STRIDE, OBJECT_TYPE_SERIALIZER_OBJECT,
}; // ICB_TG_MEMORY_STRIDE: object + kernel TG length tables
#[cfg(test)]
use crate::runtime::decode::resource::{
    MTL_INDIRECT_CMD_DRAW, MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES, MTL_INDIRECT_CMD_DRAW_PATCHES,
}; // slot-encoder fixtures only
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::objects;
use std::collections::HashMap;
use std::sync::OnceLock;

/// The host `MTLIndirectCommandBuffer` this module's decode describes.
///
/// Gated on the build, which is the lawful question — whether this binary
/// carries a Metal rail at all is a fact about the build, and the module names
/// `MTL*` types that do not exist without it. *Whether the Metal rail is the
/// one running* is never asked here: nothing in this module calls into it, and
/// its own callers are already inside that rail.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub mod metal;

/// A refusal on the indirect-command-buffer rail.
///
/// Every variant carries the registered slug naming **which** check refused.
/// Before that payload existed the five variants spoke for 153 checks — `Args`
/// alone for 84 — so a guest whose ICB never executed produced a log line
/// indistinguishable from thirty other bugs. The variant is the class; the slug
/// is the check.
///
/// There is deliberately no `Ok`: every function on this rail returns
/// `Result<_, IcbStatus>`, so success is `Ok(..)` and this type is *always* a
/// refusal. The old `Ok` variant was never constructed anywhere in the crate —
/// it survived only as an unreachable `Err(IcbStatus::Ok)` match arm — and
/// keeping it would have forced [`crate::observe::Refusal`] where
/// [`crate::observe::Decline`] is the honest shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcbStatus {
    /// A guest object, a cache entry, or a required buffer is not there.
    Missing(&'static str),
    /// A descriptor was found but is the wrong type or does not decode.
    BadDescriptor(&'static str),
    /// A host Metal call failed.
    MetalFailed(&'static str),
    /// No Metal device on this pathway — the Vulkan build's stubs, and the
    /// `system_device()` miss.
    NoMetal(&'static str),
    /// The decoded arguments do not satisfy the contract: a span past the end,
    /// a zero count, an unknown wire tag.
    Args(&'static str),
    /// The record decoded and is well-formed, and this device does not
    /// implement what it asks for on any pathway. Distinct from
    /// [`Self::NoMetal`], which is one pathway's stub, and from [`Self::Args`],
    /// which says the guest's bytes were the problem — here the guest is
    /// blameless and the answer is simply not built.
    Unsupported(&'static str),
}

impl crate::observe::Decline for IcbStatus {
    fn slug(&self) -> &'static str {
        match self {
            Self::Missing(s)
            | Self::BadDescriptor(s)
            | Self::MetalFailed(s)
            | Self::NoMetal(s)
            | Self::Args(s)
            | Self::Unsupported(s) => s,
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![(
            "class",
            match self {
                Self::Missing(_) => "missing",
                Self::BadDescriptor(_) => "bad_descriptor",
                Self::MetalFailed(_) => "metal_failed",
                Self::NoMetal(_) => "no_metal",
                Self::Args(_) => "args",
                Self::Unsupported(_) => "unsupported",
            }
            .to_string(),
        )]
    }
}

/// Carry an ICB refusal onto the compute rail without losing its name.
///
/// The compute rail is where an ICB refusal actually reaches the sink: the
/// session hands a [`crate::runtime::compute_exec::ComputeStatus`] back to
/// `exec`, which logs it as `compute_record reason=<slug>`. Before this
/// conversion existed each boundary invented its own coarse literal —
/// `icb_resolve_bad_descriptor_or_args` spoke for `BadDescriptor` *and* `Args`,
/// i.e. for 93 of this file's checks at once — so the reason died one frame
/// short of the log. Forwarding `self.slug()` means the check that refused is
/// the reason that prints.
impl From<IcbStatus> for crate::runtime::compute_exec::ComputeStatus {
    fn from(e: IcbStatus) -> Self {
        use crate::observe::Decline;
        let slug = e.slug();
        match e {
            IcbStatus::Missing(_) => Self::MissingBuffer(slug),
            IcbStatus::BadDescriptor(_) | IcbStatus::Args(_) | IcbStatus::Unsupported(_) => {
                Self::Unsupported(slug)
            }
            IcbStatus::MetalFailed(_) => Self::MetalFailed(slug),
            IcbStatus::NoMetal(_) => Self::NoMetal(slug),
        }
    }
}

/// One kernel-buffer bind for an ICB compute command fill (Metal setKernelBuffer).
#[derive(Clone, Debug, Default)]
pub struct IcbKernelBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    /// Byte offset into the buffer (host fill API, or resolved from [`Self::wire_va`]).
    pub offset: u64,
    /// Absolute guest VA from bind record `va@+4` (PGSerializer: base+offset).
    /// `0` means host-only fill / ref-at-base. Resolved to [`Self::offset`] before stage.
    pub wire_va: u64,
    /// Dynamic attribute stride (`setKernelBuffer:offset:attributeStride:atIndex:`).
    /// Wire: u64 at `attributeStrideOffset + index*8`; 0 = no stride API / default.
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
}

/// Dispatch form recorded into an ICB compute command (Metal concurrent* only).
#[derive(Clone, Copy, Debug)]
pub enum IcbFillDispatch {
    /// `concurrentDispatchThreadgroups:threadsPerThreadgroup:`
    ConcurrentThreadgroups {
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
    },
    /// `concurrentDispatchThreads:threadsPerThreadgroup:`
    ConcurrentThreads {
        threads_x: u32,
        threads_y: u32,
        threads_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
    },
}

/// One kernel-threadgroup-memory length for an ICB compute command fill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IcbThreadgroupMemory {
    pub index: u32,
    /// Byte length (`setThreadgroupMemoryLength:atIndex:`); 0 clears the slot.
    pub length: u64,
}

/// Arguments for product-path fill of one compute command slot.
#[derive(Clone, Debug)]
pub struct IcbComputeFill {
    pub command_index: u32,
    /// Serializer-object compute pipeline object-list ref (kernel function + optional stage-in).
    pub pipeline_ref: u32,
    pub buffers: Vec<IcbKernelBufferBind>,
    /// `setThreadgroupMemoryLength:atIndex:` entries (wire: u64 lengths table).
    pub threadgroup_memory: Vec<IcbThreadgroupMemory>,
    /// `setBarrier` when true, `clearBarrier` when false (wire: u32 at barrierOffset).
    pub barrier: bool,
    pub dispatch: IcbFillDispatch,
}

/// Stage for a render ICB buffer bind (layout table + Metal set*Buffer API).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IcbRenderBindStage {
    #[default]
    Vertex,
    Fragment,
    /// Object-shader stage (`setObjectBuffer`); wire at `objectBufferBindOffset`.
    Object,
    /// Mesh-shader stage (`setMeshBuffer`); wire at `meshBufferBindOffset`.
    Mesh,
}

#[cfg_attr(
    not(all(feature = "backend-metal", target_os = "macos")),
    allow(dead_code)
)]
impl IcbRenderBindStage {
    /// The bind count the create descriptor declared for this stage.
    ///
    /// Each stage is a separate Metal argument table with its own maximum, taken
    /// at ICB create from the four sibling fields the serializer-object body carries and
    /// pushed straight into the `MTLIndirectCommandBufferDescriptor` (see
    /// [`metal::materialize_metal_icb`]). They are decoded per stage, so they
    /// are compared per stage: a guest that overruns the vertex table and one
    /// that overruns the mesh table have made different mistakes.
    fn declared_bind_count(self, desc: &IndirectCommandBufferDescriptor) -> u16 {
        match self {
            Self::Vertex => desc.max_vertex_buffer_bind_count,
            Self::Fragment => desc.max_fragment_buffer_bind_count,
            Self::Object => desc.max_object_buffer_bind_count,
            Self::Mesh => desc.max_mesh_buffer_bind_count,
        }
    }

    /// The refusal slug for a bind past [`Self::declared_bind_count`].
    fn bind_past_max_slug(self) -> &'static str {
        match self {
            Self::Vertex => "icb_frc_vertex_bind_index_past_max",
            Self::Fragment => "icb_frc_fragment_bind_index_past_max",
            Self::Object => "icb_frc_object_bind_index_past_max",
            Self::Mesh => "icb_frc_mesh_bind_index_past_max",
        }
    }
}

/// Refuse a render ICB bind whose index is past what the create descriptor
/// declared for its stage.
///
/// The compute fill path has held this rule since it was written — see
/// `icb_fcc_bind_index_past_max` in [`metal::fill_compute_command`] — and the
/// render path did not, although it decodes all four sibling maxima and hands
/// every one of them to Metal at create. `MTLIndirectRenderCommand`'s `set*Buffer:` family
/// answers an index past the declared count the way every other out-of-range
/// Metal index does: an exception that aborts the process rather than a status
/// this device can decline. That is the same hazard
/// [`crate::backend::metal::constants`] documents for the direct bind paths.
///
/// Pure, and separated from the fill body on purpose: the fill needs a Metal
/// device and so cannot run on a Vulkan host, while the rule it applies is
/// arithmetic over decoded guest fields and is tested on every arm.
///
/// Its only production caller is therefore `backend-metal`-gated and this is
/// dead code on a Vulkan build — deliberately, because gating the rule to match
/// would take its tests off the one host that runs them. That the gated caller
/// still calls it was held by a source scan that could see across the `cfg`;
/// nothing does now, so a Metal-arm change that drops the call would leave this
/// function green and the bind unbounded.
#[cfg_attr(
    not(all(feature = "backend-metal", target_os = "macos")),
    allow(dead_code)
)]
pub(crate) fn refuse_render_bind_past_declared_max(
    stage: IcbRenderBindStage,
    index: u32,
    desc: &IndirectCommandBufferDescriptor,
) -> Result<(), IcbStatus> {
    if u64::from(index) >= u64::from(stage.declared_bind_count(desc)) {
        return Err(IcbStatus::Args(stage.bind_past_max_slug()));
    }
    Ok(())
}

/// One buffer bind for a render ICB command fill.
#[derive(Clone, Debug, Default)]
pub struct IcbRenderBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    /// Byte offset into the buffer (host fill API, or resolved from [`Self::wire_va`]).
    pub offset: u64,
    /// Absolute guest VA from bind record `va@+4` (PGSerializer: base+offset).
    /// `0` means host-only fill / ref-at-base. Resolved to [`Self::offset`] before stage.
    pub wire_va: u64,
    /// Dynamic attribute stride for vertex binds
    /// (`setVertexBuffer:offset:attributeStride:atIndex:`). Wire: u64 at
    /// `attributeStrideOffset + index*8`. Non-vertex stages ignore this field.
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
    /// Legacy convenience: `true` means [`IcbRenderBindStage::Fragment`].
    /// Prefer [`Self::stage`]; when `stage` is default Vertex and this is true,
    /// treat as Fragment (older call sites).
    pub is_fragment: bool,
    /// Bind stage (vertex / fragment / object / mesh). When `Object` or `Mesh`,
    /// overrides `is_fragment`.
    pub stage: IcbRenderBindStage,
}

impl IcbRenderBufferBind {
    /// Effective stage after reconciling `stage` and legacy `is_fragment`.
    pub fn effective_stage(&self) -> IcbRenderBindStage {
        match self.stage {
            IcbRenderBindStage::Object | IcbRenderBindStage::Mesh => self.stage,
            IcbRenderBindStage::Fragment => IcbRenderBindStage::Fragment,
            IcbRenderBindStage::Vertex if self.is_fragment => IcbRenderBindStage::Fragment,
            IcbRenderBindStage::Vertex => IcbRenderBindStage::Vertex,
        }
    }
}

/// Tessellation-factor buffer recorded at layout `tessellationFactorOffset`.
/// u32 ref@0 · u64 va@4 · u64 gpuva@0xc · u64 instanceStride@0x14.
#[derive(Clone, Copy, Debug, Default)]
pub struct IcbTessellationFactor {
    pub buffer_ref: u32,
    pub offset: u64,
    pub wire_va: u64,
    pub instance_stride: u64,
}

/// Draw form recorded into a render ICB command.
#[derive(Clone, Copy, Debug)]
pub enum IcbRenderDraw {
    /// command type `0x1` — drawPrimitives
    Primitives {
        primitive_type: u16,
        vertex_start: u64,
        vertex_count: u64,
        instance_count: u64,
        base_instance: u64,
    },
    /// command type `0x2` — drawIndexedPrimitives (PGSerializer layout).
    Indexed {
        primitive_type: u16,
        /// MTLIndexType (UInt16=0, UInt32=1).
        index_type: u16,
        index_buffer_ref: u32,
        index_count: u64,
        /// Byte offset into the index buffer (host fill or resolved from wire VA).
        index_buffer_offset: u64,
        /// Absolute guest VA of the index range (`va@+0x10` in DrawIndexed args); `0` = base.
        index_wire_va: u64,
        instance_count: u64,
        base_vertex: i64,
        base_instance: u64,
    },
    /// command type `0x4` — drawPatches (host RE PGSerializerIndirectRenderCommand).
    Patches {
        number_of_patch_control_points: u16,
        patch_start: u64,
        patch_count: u64,
        /// Optional patch-index buffer object-list ref (`0` = none / null Metal buffer).
        patch_index_buffer_ref: u32,
        patch_index_buffer_offset: u64,
        patch_index_wire_va: u64,
        instance_count: u64,
        base_instance: u64,
        tessellation_factor: IcbTessellationFactor,
    },
    /// command type `0x8` — drawIndexedPatches.
    IndexedPatches {
        number_of_patch_control_points: u16,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer_ref: u32,
        patch_index_buffer_offset: u64,
        patch_index_wire_va: u64,
        control_point_index_buffer_ref: u32,
        control_point_index_buffer_offset: u64,
        control_point_index_wire_va: u64,
        instance_count: u64,
        base_instance: u64,
        tessellation_factor: IcbTessellationFactor,
    },
    /// command type `0x100` — drawMeshThreads. `grid` is threadsPerGrid.
    MeshThreads(IcbMeshDraw),
    /// command type `0x80` — drawMeshThreadgroups. `grid` is
    /// threadgroupsPerGrid.
    MeshThreadgroups(IcbMeshDraw),
}

/// The record both mesh draw commands serialize.
///
/// Wire: three MTLSize as 3×u64 each at `commandArgumentsOffset`, total
/// [`ICB_DRAW_MESH_ARGS_LEN`] (`0x48` from host `setupCommandLayout`). Field
/// order matches Metal SPI `MTLIndirectDrawMesh*Arguments` — grid @0, object TG
/// @0x18, mesh TG @0x30. Fill IMPs are stubs; the layout follows
/// `setupCommandLayout` + concurrent-dispatch packing + SPI field order.
///
/// `drawMeshThreads` (`0x100`) and `drawMeshThreadgroups` (`0x80`) write byte-
/// identical records; the only difference is what the first MTLSize counts,
/// which the two [`IcbRenderDraw`] variants carry. One record, two meanings.
#[derive(Clone, Copy, Debug)]
pub struct IcbMeshDraw {
    /// threadsPerGrid or threadgroupsPerGrid, per the owning variant.
    pub grid: [u32; 3],
    pub object_tg: [u32; 3],
    pub mesh_tg: [u32; 3],
}

impl IcbMeshDraw {
    /// Read the nine u64 dimensions at `args` within `slot`. The caller has
    /// already proved `args + ICB_DRAW_MESH_ARGS_LEN <= slot.len()`.
    fn decode(slot: &[u8], args: usize) -> Self {
        let at = |off: usize| ld64(&slot[args + off..]) as u32;
        Self {
            grid: [at(0), at(8), at(0x10)],
            object_tg: [at(0x18), at(0x20), at(0x28)],
            mesh_tg: [at(0x30), at(0x38), at(0x40)],
        }
    }

    /// Write the nine dimensions at `args` within `slot`. The caller has
    /// already proved `args + ICB_DRAW_MESH_ARGS_LEN <= size`. Test-only, like
    /// its only caller [`encode_render_command_slot`].
    #[cfg(test)]
    fn encode(&self, slot: &mut [u8], args: usize) {
        use crate::protocol::endian::st64;
        for (i, v) in self
            .grid
            .iter()
            .chain(&self.object_tg)
            .chain(&self.mesh_tg)
            .enumerate()
        {
            st64(&mut slot[args + i * 8..], u64::from(*v));
        }
    }
}

/// Arguments for product-path fill of one render command slot.
#[derive(Clone, Debug)]
pub struct IcbRenderFill {
    pub command_index: u32,
    pub pipeline_ref: u32,
    pub buffers: Vec<IcbRenderBufferBind>,
    /// Object-stage TG memory lengths (`setObjectThreadgroupMemoryLength:atIndex:`).
    /// Wire: u64 table at `objectThreadgroupMemoryLengthOffset + index*8`.
    pub object_threadgroup_memory: Vec<IcbThreadgroupMemory>,
    pub draw: IcbRenderDraw,
}

/// Guest ICB command-memory association (backing buffer for CPU fills).
#[derive(Clone, Copy, Debug)]
pub struct IcbCommandMemory {
    pub gva: u64,
    pub byte_len: u64,
}

/// Decode one filled compute command slot from ICB backing bytes.
///
/// Returns `None` if the slot is empty/reset (command type 0).
pub fn decode_compute_command_slot(
    layout: &IcbCommandLayout,
    slot: &[u8],
    max_kernel_binds: u16,
) -> Result<Option<IcbComputeFill>, IcbStatus> {
    let cmd_size = layout.command_size as usize;
    if cmd_size == 0 || slot.len() < cmd_size {
        return Err(IcbStatus::Args("icb_dcs_slot_short"));
    }
    let type_off = layout.command_type_offset as usize;
    if type_off + 4 > slot.len() {
        return Err(IcbStatus::Args("icb_dcs_type_offset_oob"));
    }
    let cmd_type = ld32(&slot[type_off..]);
    if cmd_type == 0 {
        return Ok(None);
    }
    let dispatch = match cmd_type {
        ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS
        | ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS => {
            // Both commands serialize the same two MTLSize (grid, threadgroup)
            // as 6xu64; only what the first counts differs.
            let threadgroups = cmd_type == ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS;
            let args = layout.command_arguments_offset as usize;
            if args + ICB_CONCURRENT_DISPATCH_ARGS_LEN > slot.len() {
                return Err(IcbStatus::Args(if threadgroups {
                    "icb_dcs_tg_args_oob"
                } else {
                    "icb_dcs_threads_args_oob"
                }));
            }
            let d = |off: usize| ld64(&slot[args + off..]) as u32;
            let (x, y, z, tg_x, tg_y, tg_z) = (d(0), d(8), d(16), d(24), d(32), d(40));
            if threadgroups {
                IcbFillDispatch::ConcurrentThreadgroups {
                    grid_x: x,
                    grid_y: y,
                    grid_z: z,
                    tg_x,
                    tg_y,
                    tg_z,
                }
            } else {
                IcbFillDispatch::ConcurrentThreads {
                    threads_x: x,
                    threads_y: y,
                    threads_z: z,
                    tg_x,
                    tg_y,
                    tg_z,
                }
            }
        }
        _ => return Err(IcbStatus::Args("icb_dcs_unknown_command_type")),
    };

    // Pipeline ref may be 0 when the ICB was created with inheritPipelineState —
    // PSO then comes from the parent compute encoder at execute (Metal contract).
    let mut pipeline_ref = 0u32;
    if layout.pipeline_state_offset != 0 {
        let off = layout.pipeline_state_offset as usize;
        if off + 4 > slot.len() {
            return Err(IcbStatus::Args("icb_dcs_pipeline_offset_oob"));
        }
        pipeline_ref = ld32(&slot[off..]);
    }

    let mut buffers = Vec::new();
    if layout.kernel_buffer_bind_offset != 0 && max_kernel_binds > 0 {
        let base = layout.kernel_buffer_bind_offset as usize;
        for i in 0..max_kernel_binds as usize {
            let off = base + i * ICB_BUFFER_BIND_STRIDE;
            if off + 4 > slot.len() {
                break;
            }
            let buffer_ref = ld32(&slot[off..]);
            if buffer_ref == 0 {
                continue;
            }
            // Bind record 0x14 B: ref@0, va@4, gpuva@0xc. Offset is not a
            // separate field — guest writes base+offset into va (resolved later).
            let wire_va = if off + 0xc <= slot.len() {
                ld64(&slot[off + 4..])
            } else {
                0
            };
            // Attribute stride table (separate from bind record).
            let (attribute_stride, has_attribute_stride) =
                read_attribute_stride(layout, slot, i as u32);
            buffers.push(IcbKernelBufferBind {
                index: i as u32,
                buffer_ref,
                offset: 0,
                wire_va,
                attribute_stride,
                has_attribute_stride,
            });
        }
    }

    // Barrier: u32 at barrierOffset (setBarrier writes 1, clearBarrier writes 0).
    // PGSerializerIndirectComputeCommand setBarrier/clearBarrier.
    let mut barrier = false;
    if layout.barrier_offset != 0 {
        let bo = layout.barrier_offset as usize;
        if bo + 4 <= slot.len() {
            barrier = ld32(&slot[bo..]) != 0;
        }
    }

    // Threadgroup memory: u64 length table at threadgroupMemoryLengthOffset,
    // entry i at + i*8 (setThreadgroupMemoryLength:atIndex:).
    let mut threadgroup_memory = Vec::new();
    let tg_slots = icb_layout_kernel_tg_slot_count(layout);
    if tg_slots > 0 && layout.threadgroup_memory_length_offset != 0 {
        let base = layout.threadgroup_memory_length_offset as usize;
        for i in 0..tg_slots as usize {
            let off = base + i * ICB_TG_MEMORY_STRIDE;
            if off + 8 > slot.len() {
                break;
            }
            let length = ld64(&slot[off..]);
            if length != 0 {
                threadgroup_memory.push(IcbThreadgroupMemory {
                    index: i as u32,
                    length,
                });
            }
        }
    }

    Ok(Some(IcbComputeFill {
        command_index: 0, // caller sets index
        pipeline_ref,
        buffers,
        threadgroup_memory,
        barrier,
        dispatch,
    }))
}

/// Decode one filled **render** command slot (Draw / DrawIndexed) from ICB backing.
pub fn decode_render_command_slot(
    layout: &IcbCommandLayout,
    slot: &[u8],
    max_vertex_binds: u16,
    max_fragment_binds: u16,
) -> Result<Option<IcbRenderFill>, IcbStatus> {
    let cmd_size = layout.command_size as usize;
    if cmd_size == 0 || slot.len() < cmd_size {
        return Err(IcbStatus::Args("icb_drs_slot_short"));
    }
    let type_off = layout.command_type_offset as usize;
    if type_off + 4 > slot.len() {
        return Err(IcbStatus::Args("icb_drs_type_offset_oob"));
    }
    let cmd_type = ld32(&slot[type_off..]);
    if cmd_type == 0 {
        return Ok(None);
    }

    let mut pipeline_ref = 0u32;
    if layout.pipeline_state_offset != 0 {
        let off = layout.pipeline_state_offset as usize;
        if off + 4 > slot.len() {
            return Err(IcbStatus::Args("icb_drs_pipeline_offset_oob"));
        }
        pipeline_ref = ld32(&slot[off..]);
    }
    if pipeline_ref == 0 {
        return Err(IcbStatus::Missing("icb_drs_pipeline_ref_zero"));
    }

    let tessellation_factor = read_tessellation_factor(layout, slot);

    let mut buffers = Vec::new();
    let push_binds = |buffers: &mut Vec<IcbRenderBufferBind>,
                      base_off: u32,
                      count: u32,
                      stage: IcbRenderBindStage| {
        if base_off == 0 || count == 0 {
            return;
        }
        let base = base_off as usize;
        for i in 0..count as usize {
            let off = base + i * ICB_BUFFER_BIND_STRIDE;
            if off + 4 > slot.len() {
                break;
            }
            let buffer_ref = ld32(&slot[off..]);
            if buffer_ref == 0 {
                continue;
            }
            let wire_va = if off + 0xc <= slot.len() {
                ld64(&slot[off + 4..])
            } else {
                0
            };
            let (attribute_stride, has_attribute_stride) = if stage == IcbRenderBindStage::Vertex {
                read_attribute_stride(layout, slot, i as u32)
            } else {
                (0, false)
            };
            buffers.push(IcbRenderBufferBind {
                index: i as u32,
                buffer_ref,
                offset: 0,
                wire_va,
                attribute_stride,
                has_attribute_stride,
                is_fragment: stage == IcbRenderBindStage::Fragment,
                stage,
            });
        }
    };
    push_binds(
        &mut buffers,
        layout.vertex_buffer_bind_offset,
        u32::from(max_vertex_binds),
        IcbRenderBindStage::Vertex,
    );
    push_binds(
        &mut buffers,
        layout.fragment_buffer_bind_offset,
        u32::from(max_fragment_binds),
        IcbRenderBindStage::Fragment,
    );
    // Object/mesh bind table sizes from layout offsets (setupCommandLayout order).
    let max_object = icb_layout_stage_bind_count(
        layout.object_buffer_bind_offset,
        layout.mesh_buffer_bind_offset,
    );
    let max_mesh = icb_layout_stage_bind_count(
        layout.mesh_buffer_bind_offset,
        layout.kernel_buffer_bind_offset,
    );
    push_binds(
        &mut buffers,
        layout.object_buffer_bind_offset,
        max_object,
        IcbRenderBindStage::Object,
    );
    push_binds(
        &mut buffers,
        layout.mesh_buffer_bind_offset,
        max_mesh,
        IcbRenderBindStage::Mesh,
    );

    let args = layout.command_arguments_offset as usize;
    let draw = match cmd_type {
        ICB_CMD_TYPE_DRAW => {
            // Pack: u16 prim @0, u64 vertexStart @2, u64 vertexCount @0xa,
            // u64 instanceCount @0x12, u64 baseInstance @0x1a.
            if args + 0x22 > slot.len() {
                return Err(IcbStatus::Args("icb_drs_draw_args_oob"));
            }
            let prim = u16::from_le_bytes([slot[args], slot[args + 1]]);
            IcbRenderDraw::Primitives {
                primitive_type: prim,
                vertex_start: ld64(&slot[args + 2..]),
                vertex_count: ld64(&slot[args + 0xa..]),
                instance_count: ld64(&slot[args + 0x12..]),
                base_instance: ld64(&slot[args + 0x1a..]),
            }
        }
        ICB_CMD_TYPE_DRAW_INDEXED => {
            // DrawIndexed (PGSerializer): u16 prim @0, u16 indexType @2,
            // u32 indexBufferRef @4, u64 indexCount @8, u64 va @0x10, u64 gpuva @0x18,
            // u64 instanceCount @0x20, u64 baseVertex @0x28 (signed bit pattern),
            // u64 baseInstance @0x30.
            if args + 0x38 > slot.len() {
                return Err(IcbStatus::Args("icb_drs_indexed_args_oob"));
            }
            let prim = u16::from_le_bytes([slot[args], slot[args + 1]]);
            let index_type = u16::from_le_bytes([slot[args + 2], slot[args + 3]]);
            let index_buffer_ref = ld32(&slot[args + 4..]);
            if index_buffer_ref == 0 {
                return Err(IcbStatus::Missing("icb_drs_index_buffer_ref_zero"));
            }
            let index_wire_va = ld64(&slot[args + 0x10..]);
            IcbRenderDraw::Indexed {
                primitive_type: prim,
                index_type,
                index_buffer_ref,
                index_count: ld64(&slot[args + 8..]),
                index_buffer_offset: 0, // resolved from index_wire_va when non-zero
                index_wire_va,
                instance_count: ld64(&slot[args + 0x20..]),
                base_vertex: ld64(&slot[args + 0x28..]) as i64,
                base_instance: ld64(&slot[args + 0x30..]),
            }
        }
        ICB_CMD_TYPE_DRAW_PATCHES => {
            // u16 controlPoints@0, u64 patchStart@2, u64 patchCount@0xa,
            // u32 patchIndexRef@0x12, u64 va@0x16, u64 gpuva@0x1e,
            // u64 instanceCount@0x26, u64 baseInstance@0x2e.
            if args + ICB_DRAW_PATCHES_ARGS_LEN as usize > slot.len() {
                return Err(IcbStatus::Args("icb_drs_patches_args_oob"));
            }
            let cps = u16::from_le_bytes([slot[args], slot[args + 1]]);
            let patch_index_buffer_ref = ld32(&slot[args + 0x12..]);
            let patch_index_wire_va = ld64(&slot[args + 0x16..]);
            IcbRenderDraw::Patches {
                number_of_patch_control_points: cps,
                patch_start: ld64(&slot[args + 2..]),
                patch_count: ld64(&slot[args + 0xa..]),
                patch_index_buffer_ref,
                patch_index_buffer_offset: 0,
                patch_index_wire_va,
                instance_count: ld64(&slot[args + 0x26..]),
                base_instance: ld64(&slot[args + 0x2e..]),
                tessellation_factor,
            }
        }
        ICB_CMD_TYPE_DRAW_INDEXED_PATCHES => {
            // like DrawPatches through patchIndex, then
            // u32 controlPointIndexRef@0x26, u64 va@0x2a, u64 gpuva@0x32,
            // u64 instanceCount@0x3a, u64 baseInstance@0x42.
            if args + ICB_DRAW_INDEXED_PATCHES_ARGS_LEN as usize > slot.len() {
                return Err(IcbStatus::Args("icb_drs_indexed_patches_args_oob"));
            }
            let cps = u16::from_le_bytes([slot[args], slot[args + 1]]);
            let patch_index_buffer_ref = ld32(&slot[args + 0x12..]);
            let patch_index_wire_va = ld64(&slot[args + 0x16..]);
            let control_point_index_buffer_ref = ld32(&slot[args + 0x26..]);
            if control_point_index_buffer_ref == 0 {
                return Err(IcbStatus::Missing("icb_drs_control_point_ref_zero"));
            }
            let control_point_index_wire_va = ld64(&slot[args + 0x2a..]);
            IcbRenderDraw::IndexedPatches {
                number_of_patch_control_points: cps,
                patch_start: ld64(&slot[args + 2..]),
                patch_count: ld64(&slot[args + 0xa..]),
                patch_index_buffer_ref,
                patch_index_buffer_offset: 0,
                patch_index_wire_va,
                control_point_index_buffer_ref,
                control_point_index_buffer_offset: 0,
                control_point_index_wire_va,
                instance_count: ld64(&slot[args + 0x3a..]),
                base_instance: ld64(&slot[args + 0x42..]),
                tessellation_factor,
            }
        }
        ICB_CMD_TYPE_DRAW_MESH_THREADS | ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS => {
            let threads = cmd_type == ICB_CMD_TYPE_DRAW_MESH_THREADS;
            if args + ICB_DRAW_MESH_ARGS_LEN as usize > slot.len() {
                return Err(IcbStatus::Args(if threads {
                    "icb_drs_mesh_threads_args_oob"
                } else {
                    "icb_drs_mesh_threadgroups_args_oob"
                }));
            }
            let mesh = IcbMeshDraw::decode(slot, args);
            if threads {
                IcbRenderDraw::MeshThreads(mesh)
            } else {
                IcbRenderDraw::MeshThreadgroups(mesh)
            }
        }
        _ => return Err(IcbStatus::Args("icb_drs_unknown_command_type")),
    };

    // Object TG memory length table (setupCommandLayout: before kernel TG).
    let mut object_threadgroup_memory = Vec::new();
    let obj_tg_slots = icb_layout_object_tg_slot_count(layout);
    if obj_tg_slots > 0 && layout.object_threadgroup_memory_length_offset != 0 {
        let base = layout.object_threadgroup_memory_length_offset as usize;
        for i in 0..obj_tg_slots as usize {
            let off = base + i * ICB_TG_MEMORY_STRIDE;
            if off + 8 > slot.len() {
                break;
            }
            let length = ld64(&slot[off..]);
            if length != 0 {
                object_threadgroup_memory.push(IcbThreadgroupMemory {
                    index: i as u32,
                    length,
                });
            }
        }
    }

    Ok(Some(IcbRenderFill {
        command_index: 0,
        pipeline_ref,
        buffers,
        object_threadgroup_memory,
        draw,
    }))
}

/// Object-TG length table slot count between layout offsets.
fn icb_layout_object_tg_slot_count(layout: &IcbCommandLayout) -> u32 {
    icb_layout_table_len(
        layout.object_threadgroup_memory_length_offset,
        layout.threadgroup_memory_length_offset,
        ICB_TG_MEMORY_STRIDE,
    )
}

/// Read tessellation-factor table at `tessellationFactorOffset` (host RE).
fn read_tessellation_factor(layout: &IcbCommandLayout, slot: &[u8]) -> IcbTessellationFactor {
    if layout.tessellation_factor_offset == 0 {
        return IcbTessellationFactor::default();
    }
    let off = layout.tessellation_factor_offset as usize;
    if off + ICB_TESSELLATION_FACTOR_LEN > slot.len() {
        return IcbTessellationFactor::default();
    }
    IcbTessellationFactor {
        buffer_ref: ld32(&slot[off..]),
        wire_va: ld64(&slot[off + 4..]),
        offset: 0,
        instance_stride: ld64(&slot[off + 0x14..]),
    }
}

#[cfg(test)]
fn write_tessellation_factor(
    layout: &IcbCommandLayout,
    slot: &mut [u8],
    tf: &IcbTessellationFactor,
) -> Result<(), IcbStatus> {
    if layout.tessellation_factor_offset == 0 {
        return Ok(());
    }
    let off = layout.tessellation_factor_offset as usize;
    if off + ICB_TESSELLATION_FACTOR_LEN > slot.len() {
        return Err(IcbStatus::Args("icb_write_tess_factor_oob"));
    }
    use crate::protocol::endian::{st32, st64};
    st32(&mut slot[off..], tf.buffer_ref);
    let va = if tf.wire_va != 0 { tf.wire_va } else { 0 };
    st64(&mut slot[off + 4..], va);
    st64(&mut slot[off + 0xc..], va);
    st64(&mut slot[off + 0x14..], tf.instance_stride);
    Ok(())
}

/// Bind-table slot count between two layout offsets (`count × 0x14`).
fn icb_layout_stage_bind_count(start: u32, end: u32) -> u32 {
    icb_layout_table_len(start, end, ICB_BUFFER_BIND_STRIDE)
}

/// Encode one render Draw / DrawIndexed command slot (tests / fixtures).
#[cfg(test)]
pub fn encode_render_command_slot(
    layout: &IcbCommandLayout,
    fill: &IcbRenderFill,
) -> Result<Vec<u8>, IcbStatus> {
    use crate::protocol::endian::{st16, st32, st64};
    let size = layout.command_size as usize;
    if size == 0 {
        return Err(IcbStatus::Args("icb_ers_zero_command_size"));
    }
    let mut slot = vec![0u8; size];
    let type_off = layout.command_type_offset as usize;
    if layout.pipeline_state_offset != 0 {
        st32(
            &mut slot[layout.pipeline_state_offset as usize..],
            fill.pipeline_ref,
        );
    }
    for b in &fill.buffers {
        let base = match b.effective_stage() {
            IcbRenderBindStage::Vertex => layout.vertex_buffer_bind_offset,
            IcbRenderBindStage::Fragment => layout.fragment_buffer_bind_offset,
            IcbRenderBindStage::Object => layout.object_buffer_bind_offset,
            IcbRenderBindStage::Mesh => layout.mesh_buffer_bind_offset,
        } as usize;
        let off = base + (b.index as usize) * ICB_BUFFER_BIND_STRIDE;
        if off + ICB_BUFFER_BIND_STRIDE > size {
            return Err(IcbStatus::Args("icb_ers_bind_offset_oob"));
        }
        st32(&mut slot[off..], b.buffer_ref);
        // Wire VA = absolute GVA (base+offset). Prefer explicit wire_va; else 0
        // (host fill uses offset without a wire VA). Same 0x14 packing as
        // setVertexBuffer / setFragmentBuffer (ref@0 · va@4 · gpuva@0xc).
        let va = if b.wire_va != 0 { b.wire_va } else { 0 };
        st64(&mut slot[off + 4..], va);
        st64(&mut slot[off + 0xc..], va); // gpuva same as va for fixtures
        if b.effective_stage() == IcbRenderBindStage::Vertex && b.has_attribute_stride {
            write_attribute_stride(layout, &mut slot, b.index, b.attribute_stride)?;
        }
    }
    // Object TG memory lengths (u64 at objectThreadgroupMemoryLengthOffset + i*8).
    for tg in &fill.object_threadgroup_memory {
        if layout.object_threadgroup_memory_length_offset == 0 {
            return Err(IcbStatus::Args("icb_ers_no_object_tg_table"));
        }
        let off = layout.object_threadgroup_memory_length_offset as usize
            + (tg.index as usize) * ICB_TG_MEMORY_STRIDE;
        if off + 8 > size {
            return Err(IcbStatus::Args("icb_ers_object_tg_offset_oob"));
        }
        st64(&mut slot[off..], tg.length);
    }
    let args = layout.command_arguments_offset as usize;
    match fill.draw {
        IcbRenderDraw::Primitives {
            primitive_type,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => {
            if args + 0x22 > size {
                return Err(IcbStatus::Args("icb_ers_draw_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW);
            st16(&mut slot[args..], primitive_type);
            st64(&mut slot[args + 2..], vertex_start);
            st64(&mut slot[args + 0xa..], vertex_count);
            st64(&mut slot[args + 0x12..], instance_count);
            st64(&mut slot[args + 0x1a..], base_instance);
        }
        IcbRenderDraw::Indexed {
            primitive_type,
            index_type,
            index_buffer_ref,
            index_count,
            index_buffer_offset: _,
            index_wire_va,
            instance_count,
            base_vertex,
            base_instance,
        } => {
            if args + 0x38 > size {
                return Err(IcbStatus::Args("icb_ers_indexed_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW_INDEXED);
            st16(&mut slot[args..], primitive_type);
            st16(&mut slot[args + 2..], index_type);
            st32(&mut slot[args + 4..], index_buffer_ref);
            st64(&mut slot[args + 8..], index_count);
            st64(&mut slot[args + 0x10..], index_wire_va);
            st64(&mut slot[args + 0x18..], index_wire_va);
            st64(&mut slot[args + 0x20..], instance_count);
            st64(&mut slot[args + 0x28..], base_vertex as u64);
            st64(&mut slot[args + 0x30..], base_instance);
        }
        IcbRenderDraw::Patches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            patch_index_buffer_ref,
            patch_index_buffer_offset: _,
            patch_index_wire_va,
            instance_count,
            base_instance,
            tessellation_factor,
        } => {
            if args + ICB_DRAW_PATCHES_ARGS_LEN as usize > size {
                return Err(IcbStatus::Args("icb_ers_patches_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW_PATCHES);
            st16(&mut slot[args..], number_of_patch_control_points);
            st64(&mut slot[args + 2..], patch_start);
            st64(&mut slot[args + 0xa..], patch_count);
            st32(&mut slot[args + 0x12..], patch_index_buffer_ref);
            st64(&mut slot[args + 0x16..], patch_index_wire_va);
            st64(&mut slot[args + 0x1e..], patch_index_wire_va);
            st64(&mut slot[args + 0x26..], instance_count);
            st64(&mut slot[args + 0x2e..], base_instance);
            write_tessellation_factor(layout, &mut slot, &tessellation_factor)?;
        }
        IcbRenderDraw::IndexedPatches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            patch_index_buffer_ref,
            patch_index_buffer_offset: _,
            patch_index_wire_va,
            control_point_index_buffer_ref,
            control_point_index_buffer_offset: _,
            control_point_index_wire_va,
            instance_count,
            base_instance,
            tessellation_factor,
        } => {
            if args + ICB_DRAW_INDEXED_PATCHES_ARGS_LEN as usize > size {
                return Err(IcbStatus::Args("icb_ers_indexed_patches_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW_INDEXED_PATCHES);
            st16(&mut slot[args..], number_of_patch_control_points);
            st64(&mut slot[args + 2..], patch_start);
            st64(&mut slot[args + 0xa..], patch_count);
            st32(&mut slot[args + 0x12..], patch_index_buffer_ref);
            st64(&mut slot[args + 0x16..], patch_index_wire_va);
            st64(&mut slot[args + 0x1e..], patch_index_wire_va);
            st32(&mut slot[args + 0x26..], control_point_index_buffer_ref);
            st64(&mut slot[args + 0x2a..], control_point_index_wire_va);
            st64(&mut slot[args + 0x32..], control_point_index_wire_va);
            st64(&mut slot[args + 0x3a..], instance_count);
            st64(&mut slot[args + 0x42..], base_instance);
            write_tessellation_factor(layout, &mut slot, &tessellation_factor)?;
        }
        IcbRenderDraw::MeshThreads(mesh) | IcbRenderDraw::MeshThreadgroups(mesh) => {
            let threads = matches!(fill.draw, IcbRenderDraw::MeshThreads(_));
            if args + ICB_DRAW_MESH_ARGS_LEN as usize > size {
                return Err(IcbStatus::Args(if threads {
                    "icb_ers_mesh_threads_args_oob"
                } else {
                    "icb_ers_mesh_threadgroups_args_oob"
                }));
            }
            st32(
                &mut slot[type_off..],
                if threads {
                    ICB_CMD_TYPE_DRAW_MESH_THREADS
                } else {
                    ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS
                },
            );
            mesh.encode(&mut slot, args);
        }
    }
    let _ = (
        MTL_INDIRECT_CMD_DRAW,
        MTL_INDIRECT_CMD_DRAW_PATCHES,
        MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES,
    );
    Ok(slot)
}

/// Encode one compute command slot into ICB backing bytes (tests / fixtures).
#[cfg(test)]
pub fn encode_compute_command_slot(
    layout: &IcbCommandLayout,
    fill: &IcbComputeFill,
) -> Result<Vec<u8>, IcbStatus> {
    use crate::protocol::endian::{st32, st64};
    let size = layout.command_size as usize;
    if size == 0 {
        return Err(IcbStatus::Args("icb_ecs_zero_command_size"));
    }
    let mut slot = vec![0u8; size];
    let (cmd_type, gx, gy, gz, tx, ty, tz) = match fill.dispatch {
        IcbFillDispatch::ConcurrentThreadgroups {
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
        } => (
            ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS,
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
        ),
        IcbFillDispatch::ConcurrentThreads {
            threads_x,
            threads_y,
            threads_z,
            tg_x,
            tg_y,
            tg_z,
        } => (
            ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS,
            threads_x,
            threads_y,
            threads_z,
            tg_x,
            tg_y,
            tg_z,
        ),
    };
    let type_off = layout.command_type_offset as usize;
    if type_off + 4 > size {
        return Err(IcbStatus::Args("icb_ecs_type_offset_oob"));
    }
    st32(&mut slot[type_off..], cmd_type);
    if layout.pipeline_state_offset != 0 {
        let off = layout.pipeline_state_offset as usize;
        if off + 4 > size {
            return Err(IcbStatus::Args("icb_ecs_pipeline_offset_oob"));
        }
        st32(&mut slot[off..], fill.pipeline_ref);
    }
    for b in &fill.buffers {
        let off =
            layout.kernel_buffer_bind_offset as usize + (b.index as usize) * ICB_BUFFER_BIND_STRIDE;
        if off + ICB_BUFFER_BIND_STRIDE > size {
            return Err(IcbStatus::Args("icb_ecs_bind_offset_oob"));
        }
        st32(&mut slot[off..], b.buffer_ref);
        let va = if b.wire_va != 0 { b.wire_va } else { 0 };
        st64(&mut slot[off + 4..], va);
        st64(&mut slot[off + 0xc..], va);
        if b.has_attribute_stride {
            write_attribute_stride(layout, &mut slot, b.index, b.attribute_stride)?;
        }
    }
    // Barrier u32 (1 = setBarrier, 0 = clear).
    if layout.barrier_offset != 0 {
        let bo = layout.barrier_offset as usize;
        if bo + 4 > size {
            return Err(IcbStatus::Args("icb_ecs_barrier_offset_oob"));
        }
        st32(&mut slot[bo..], if fill.barrier { 1 } else { 0 });
    }
    // Threadgroup memory length table (u64 per index).
    for tg in &fill.threadgroup_memory {
        let off = layout.threadgroup_memory_length_offset as usize
            + (tg.index as usize) * ICB_TG_MEMORY_STRIDE;
        if off + 8 > size {
            return Err(IcbStatus::Args("icb_ecs_tg_offset_oob"));
        }
        st64(&mut slot[off..], tg.length);
    }
    let args = layout.command_arguments_offset as usize;
    if args + ICB_CONCURRENT_DISPATCH_ARGS_LEN > size {
        return Err(IcbStatus::Args("icb_ecs_dispatch_args_oob"));
    }
    st64(&mut slot[args..], gx as u64);
    st64(&mut slot[args + 8..], gy as u64);
    st64(&mut slot[args + 16..], gz as u64);
    st64(&mut slot[args + 24..], tx as u64);
    st64(&mut slot[args + 32..], ty as u64);
    st64(&mut slot[args + 40..], tz as u64);
    Ok(slot)
}

/// Load and decode a serializer-object ICB descriptor for `icb_ref` on the task object list.
pub fn load_icb_descriptor<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
) -> Result<IndirectCommandBufferDescriptor, IcbStatus> {
    if icb_ref == 0 {
        return Err(IcbStatus::Missing("icb_desc_ref_zero"));
    }
    // The two statuses this rail splits the ladder into, stated once: a tag that
    // is not serializer-object means the guest described something, wrongly, while a
    // missing entry or unreadable bytes mean it described nothing this device
    // can see yet.
    let (_entry, desc) = objects::resolve_descriptor(
        state,
        host,
        task_id,
        icb_ref,
        &[OBJECT_TYPE_SERIALIZER_OBJECT],
    )
    .map_err(|rung| {
        let slug = crate::observe::ladder_slugs!("icb")(rung);
        match rung {
            objects::LadderRung::NoListEntry
            | objects::LadderRung::DescRead { .. }
            | objects::LadderRung::NoTaskSpace => IcbStatus::Missing(slug),
            objects::LadderRung::WrongType { .. } => IcbStatus::BadDescriptor(slug),
        }
    })?;
    match decode_serializer_object_descriptor(&desc) {
        Ok(ResourceDescriptor::IndirectCommandBuffer(icb)) => {
            note_unapplied_icb_flags(task_id, icb_ref, &icb);
            Ok(icb)
        }
        Ok(_) => Err(IcbStatus::BadDescriptor("icb_desc_not_icb_body")),
        Err(_) => Err(IcbStatus::BadDescriptor(crate::observe::ladder_slug!(
            "icb",
            desc_decode
        ))),
    }
}

/// A flag the guest set on its indirect command buffer that this device decodes
/// and does not apply.
struct IcbFlagDropped(crate::runtime::decode::resource::IcbUnappliedFlag);

impl crate::observe::Decline for IcbFlagDropped {
    fn slug(&self) -> &'static str {
        self.0.slug()
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// Report every decoded ICB flag this device drops on the floor.
///
/// Eight of the ten attributed bits in the create body's flag word reach no
/// host setter: `supportRayTracing`, `supportDynamicAttributeStride` and the six
/// inherit-state flags Metal added in macOS 26. This device builds its host
/// `MTLIndirectCommandBufferDescriptor` without touching any of them, so each
/// one silently takes Metal's default instead of the guest's.
///
/// Counted rather than executed, deliberately. Six of the eight default *on* at
/// both ends, so on a descriptor the guest left alone nothing is lost and every
/// counter here is a healthy zero — which means a **non**-zero reading is the
/// measured argument for building that flag's setter, and says which one. Doing
/// it the other way round would mean writing eight `objc_msgSend` wrappers into
/// the Metal-only arm for a path a driven boot has never taken
/// (`runtime::icb` reads 0.00% on a driven x86 boot, and `icb_exec_seen` has
/// never fired on arm64 either).
///
/// This sits in [`load_icb_descriptor`] rather than in `materialize_metal_icb`
/// on purpose: the descriptor is decoded on both backends and only materialized
/// on one, so the count would otherwise be structurally zero on Vulkan for a
/// reason that has nothing to do with what the guest asked for.
fn note_unapplied_icb_flags(task_id: u32, icb_ref: u32, desc: &IndirectCommandBufferDescriptor) {
    use crate::observe::Decline as _;
    for flag in desc.unapplied_flags() {
        let decline = IcbFlagDropped(flag);
        crate::runtime::drain::note_store_route(decline.slug());
        crate::observe::Emit::decline("icb_desc_flag", &decline)
            .field("task", task_id)
            .field("icb", icb_ref)
            .field("flags", format!("{:#06x}", desc.flags))
            // The slug is already per flag, so the buffer is the only thing
            // left to key on: one line per ICB per flag, however many times a
            // cache miss reloads the descriptor.
            .fail_once(icb_ref as u64);
    }
}

// ---------------------------------------------------------------------------
// ICB registry: (task_id, icb_ref) → what the guest declared. Backend-free.
// ---------------------------------------------------------------------------

/// What the guest said about one ICB, with nothing of the host in it.
///
/// The descriptor and the command-memory span are the whole input to
/// [`decode_icb_command_range`], and that decode is the same on all three
/// pathways — so this lives here rather than inside the Metal object cache,
/// which is the only reason the Vulkan arm can hold ICB state at all.
///
/// Split out because the two halves have different lifetimes as well as
/// different portability: a descriptor change re-materializes the host object
/// but does not by itself say the guest re-pointed its command memory.
#[derive(Clone)]
struct IcbRecord {
    desc: IndirectCommandBufferDescriptor,
    /// Guest ICB backing buffer (the command slots the guest filled).
    command_memory: Option<IcbCommandMemory>,
}

fn icb_registry() -> &'static parking_lot::Mutex<HashMap<(u32, u32), IcbRecord>> {
    static REGISTRY: OnceLock<parking_lot::Mutex<HashMap<(u32, u32), IcbRecord>>> = OnceLock::new();
    REGISTRY.get_or_init(|| parking_lot::Mutex::new(HashMap::new()))
}

/// Load the guest's ICB descriptor and record it, on every pathway.
///
/// Returns the descriptor the caller should build against. When the create body
/// no longer matches what was recorded, the recorded command memory is dropped
/// with it: a re-created ICB of a different shape is not the one whose slots the
/// old span held, and decoding the old bytes at the new layout would read
/// whatever happened to be at those offsets rather than refusing.
pub fn resolve_icb_record<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
) -> Result<IndirectCommandBufferDescriptor, IcbStatus> {
    let desc = load_icb_descriptor(state, host, task_id, icb_ref)?;
    let mut reg = icb_registry().lock();
    match reg.get_mut(&(task_id, icb_ref)) {
        Some(rec)
            if rec.desc.max_command_count == desc.max_command_count
                && rec.desc.command_types == desc.command_types =>
        {
            Ok(rec.desc.clone())
        }
        slot => {
            // A refreshed descriptor starts with no command memory, and that
            // drops whatever `bind_icb_command_memory` had recorded for this
            // ref. Unobservable today because nothing binds it, but whoever
            // finds the wire record that does has to decide here whether a
            // create-body change invalidates the buffer too — the guest may
            // have changed only `maxCommandCount` and still be filling the
            // same slots.
            let command_memory = None;
            let rec = IcbRecord {
                desc: desc.clone(),
                command_memory,
            };
            match slot {
                Some(existing) => *existing = rec,
                None => {
                    reg.insert((task_id, icb_ref), rec);
                }
            }
            Ok(desc)
        }
    }
}

// ---------------------------------------------------------------------------
// Host ICB cache: (task_id, icb_ref) → filled Metal ICB + retained resources
// ---------------------------------------------------------------------------

/// Drop every recorded ICB and every host ICB the running rail built from one
/// (tests / task teardown).
///
/// One entry point for both maps: they are keyed alike and a registry entry
/// outliving its host object would name a descriptor no host indirect-command
/// buffer was built from. The rail's half is
/// [`crate::backend::Backend::forget_host_icbs`], which is free on a rail that
/// builds none.
pub fn clear_icb_cache() {
    icb_registry().lock().clear();
    crate::backend::Backend::forget_host_icbs(&crate::backend::selected());
}

/// Info-segment opcode for `PGSerializerInfoCommandEncoder icbHostResourceInfo:info:`.
///
/// Full wire record length `0x18` (8 B header + 16 B payload). Payload:
/// `icb_ref:u32 @0`, `buffer_ref:u32 @4`, `gpu_address:u64 @8`.
///
/// **The offsets are right and the last two names are wrong.** Apple's own bytes
/// say this record is a *query*: `+4` is the reply staging buffer and `+8` is
/// the offset into it where the host is being asked to write two `u64`s, which
/// is what the selector's `^{?=QQ}` out-parameter means. The reading below —
/// the ICB's backing buffer and its GPU address — has no derivation behind it;
/// it arrived with the initial import and nothing has tested it against a
/// captured record, because `PGSerializerInfoCommandEncoder` sits in the
/// divergence instrument's `UNCOVERED_CLASSES`.
///
/// The evidence is in [`reims_vgpu_wire::ops::info::Query`], which declares the
/// same three offsets under the other names. Shortest form: `+4` reads the same
/// value in all ten query fixtures, *including* the one whose queried object is
/// itself a buffer with a different ref — so it cannot be that object's backing
/// buffer.
///
/// Repaired: [`apply_icb_host_resource_info`] now declines by name rather than
/// binding the reply pair, and [`IcbHostResourceInfo`] carries the wire crate's
/// field names. The device still never writes the answer the guest asked for —
/// the two `u64`s are unattributed, and `runtime::heap_query` shows the shape a
/// reply takes. The rail is dormant, which is why the wrong reading survived
/// as long as it did: `runtime::icb` reads 0.00% on a driven boot.
///
/// The three constants below are the wire crate's, aliased rather than spelled,
/// so this file cannot drift from the declaration the fixtures pin.
pub const INFO_OP_ICB_HOST_RESOURCE: u32 =
    reims_vgpu_wire::ops::info::OPCODE_ICB_HOST_RESOURCE_INFO;
pub const INFO_OP_ICB_HOST_RESOURCE_RECORD_LEN: u32 = reims_vgpu_wire::ops::info::QUERY_TOTAL_LEN;
pub const INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN: usize =
    std::mem::size_of::<reims_vgpu_wire::ops::info::Query>();

/// Decoded `0x1d1` `icbHostResourceInfo:info:` payload.
///
/// The field names are [`reims_vgpu_wire::ops::info::Query`]'s, because this
/// record *is* that record — ten selectors write the identical 24 bytes and
/// differ only in opcode. This device used to declare the same three offsets a
/// second time under two other names, `buffer_ref` and `gpu_address`, which is
/// the drift the wire crate exists to catch: the offsets agreed and the meanings
/// did not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IcbHostResourceInfo {
    /// The ICB being asked about.
    pub icb_ref: u32,
    /// Where the *answer* goes: the scratch buffer the guest's command stream
    /// returned from `-getBufferBytes:alignment:buffer:offset:`.
    pub reply_buffer_ref: u32,
    /// Offset into [`Self::reply_buffer_ref`] for the two `u64`s the guest is
    /// asking the host to write.
    pub reply_offset: u64,
}

/// Decode `0x1d1` payload (16 bytes) or full record (24 bytes including header).
pub fn decode_icb_host_resource_info(bytes: &[u8]) -> Result<IcbHostResourceInfo, IcbStatus> {
    let payload = if bytes.len() >= INFO_OP_ICB_HOST_RESOURCE_RECORD_LEN as usize
        && ld32(&bytes[0..]) == INFO_OP_ICB_HOST_RESOURCE
    {
        &bytes[8..8 + INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN]
    } else if bytes.len() >= INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN {
        &bytes[..INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN]
    } else {
        return Err(IcbStatus::Args("icb_host_resource_info_short"));
    };
    // The three offsets are taken from the wire declaration rather than spelled
    // again, so a layout change there fails this build instead of silently
    // re-slicing the same bytes into different fields.
    use reims_vgpu_wire::ops::info::Query;
    let icb_ref = ld32(&payload[std::mem::offset_of!(Query, object_ref)..]);
    let reply_buffer_ref = ld32(&payload[std::mem::offset_of!(Query, reply_buffer_ref)..]);
    let reply_offset = ld64(&payload[std::mem::offset_of!(Query, reply_offset)..]);
    if icb_ref == 0 {
        return Err(IcbStatus::Args("icb_host_resource_info_ref_zero"));
    }
    Ok(IcbHostResourceInfo {
        icb_ref,
        reply_buffer_ref,
        reply_offset,
    })
}

/// The check [`decode_icb_command_range`] fails when an ICB has no command
/// memory bound, named here because [`icb_fill_outcome`] compares against it.
///
/// Spelled once so the raise site and the arm that classifies it cannot drift:
/// a literal in both places reads as two independent facts, and a rename of one
/// silently turns the classification into a forward.
pub const ICB_FILL_NO_COMMAND_MEMORY: &str = "icb_fill_no_command_memory";

/// What an ICB execute does with the outcome of filling its slots from the
/// guest's command memory, decided once for every pathway.
///
/// # Why this is not spelled at the call sites
///
/// It was, twice — the render arm in `runtime::draw::metal_icb` and the compute
/// arm in `runtime::compute_session` each carried
/// `Ok(()) | Err(IcbStatus::Missing(_)) => {}`. Two copies of one rule, and the
/// wildcard is what made them wrong: [`decode_icb_command_range`] raises
/// `Missing` under two different slugs, and only one of them was argued for.
///
/// # What each outcome means
///
/// - `Ok(())` — slots were filled from guest memory and the execute replays
///   the guest's own commands.
/// - [`ICB_FILL_NO_COMMAND_MEMORY`] — the ICB is registered but nothing bound
///   the buffer holding its command slots, so the execute runs an ICB with no
///   commands in it and **every command the guest encoded into it is lost**.
///   Control flow is unchanged — an empty execute is a no-op, and refusing here
///   would additionally skip the attachment writeback the caller does after —
///   but the loss is now counted and fail-visible instead of being swallowed as
///   an "empty shell" case. That phrase came from a reading in which opcode
///   `0x1d1` bound command memory; it is an info query, and since it stopped
///   being treated as a bind **no decode path binds command memory at all**
///   ([`bind_icb_command_memory`]'s only caller is
///   [`associate_icb_backing_buffer_ref`], which nothing outside tests calls).
///   So this is not a rare shape — it is what every ICB execute meets today,
///   and a counter reading zero here means no guest reached the rail rather
///   than that the rail worked.
/// - anything else — forwarded to the caller, which declines by the slug of the
///   check that refused. `icb_fill_not_cached` reaches this arm and is
///   unreachable in practice: both call sites run `resolve_metal_icb` first,
///   and [`resolve_icb_record`] inserts a record for every ref it is asked
///   about. It forwards rather than being swallowed because a fill against an
///   ICB the registry has never seen is a different loss from an empty one.
pub fn icb_fill_outcome(
    outcome: Result<(), IcbStatus>,
    task_id: u32,
    icb_ref: u32,
) -> Result<(), IcbStatus> {
    match outcome {
        Err(IcbStatus::Missing(slug)) if slug == ICB_FILL_NO_COMMAND_MEMORY => {
            crate::runtime::drain::note_store_route("icb_executed_without_command_memory");
            crate::observe::Emit::decline("icb_execute_empty", &IcbStatus::Missing(slug))
                .field("task", task_id)
                .field("icb", icb_ref)
                .fail_once(u64::from(icb_ref));
            Ok(())
        }
        other => other,
    }
}

/// Register guest command-memory GVA for an ICB (backing buffer for CPU fills).
///
/// Fills are not stream opcodes; the guest writes this buffer via
/// `PGSerializerIndirectComputeCommand`. Product re-decodes it at execute.
pub fn bind_icb_command_memory(
    task_id: u32,
    icb_ref: u32,
    mem: IcbCommandMemory,
) -> Result<(), IcbStatus> {
    if icb_ref == 0 || mem.gva == 0 || mem.byte_len == 0 {
        return Err(IcbStatus::Args("icb_bind_memory_bad_args"));
    }
    let mut reg = icb_registry().lock();
    let rec = reg
        .get_mut(&(task_id, icb_ref))
        .ok_or(IcbStatus::Missing("icb_bind_memory_not_cached"))?;
    rec.command_memory = Some(mem);
    Ok(())
}

/// Associate ICB command memory from a buffer object-list ref (sync path).
///
/// Resolves buffer GVA/size via the buffer descriptor (`handle << PAGE_SHIFT`).
/// Byte length is min(buffer size, command_size × max_command_count) from the
/// ICB create layout so oversize buffer allocations are truncated to the ICB.
pub fn associate_icb_backing_buffer_ref<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    buffer_ref: u32,
) -> Result<IcbCommandMemory, IcbStatus> {
    if icb_ref == 0 || buffer_ref == 0 {
        return Err(IcbStatus::Args("icb_associate_ref_zero"));
    }
    // Record the serializer-object create layout if it is not already recorded. This used
    // to materialize the host `MTLIndirectCommandBuffer` as a side effect, which
    // is why associating a backing buffer refused outright on the Vulkan arm —
    // the association is guest bookkeeping and needs no host object at all.
    let desc = resolve_icb_record(state, host, task_id, icb_ref)?;
    let (gva, buf_size) = buffer_gva_size(state, host, task_id, buffer_ref)?;
    let need = (desc.layout.command_size as u64).saturating_mul(desc.max_command_count as u64);
    if need == 0 {
        return Err(IcbStatus::Args("icb_associate_zero_layout_span"));
    }
    if buf_size < need {
        return Err(IcbStatus::Args("icb_associate_buffer_too_small"));
    }
    let mem = IcbCommandMemory {
        gva,
        byte_len: need,
    };
    bind_icb_command_memory(task_id, icb_ref, mem)?;
    Ok(mem)
}

/// Refuse info-segment `0x1d1` (`icbHostResourceInfo:info:`) by name.
///
/// **This record is a question, and this device has no answer for it.** The
/// selector's type encoding is `v32@0:8@16^{?=QQ}24`, so `info:` is a pointer to
/// two `u64` out-parameters: the guest names an ICB and a place to write two
/// words, and waits. Nothing here writes them — see
/// [`INFO_OP_ICB_HOST_RESOURCE`] for the full derivation and
/// [`reims_vgpu_wire::ops::info`] for the fixtures that settle it.
///
/// It used to read the reply pair as an answer instead of a question, and that
/// was worse than refusing. `reply_buffer_ref` went to
/// [`associate_icb_backing_buffer_ref`] as the ICB's command backing and
/// `reply_offset` became a command-memory GVA — so a guest whose scratch
/// allocator happened to return a resolvable buffer ref would have had *its own
/// reply staging area* bound as an ICB's command slots, and the next
/// `executeCommandsInBuffer:` would decode whatever sat there and run it as
/// draws. A refusal loses the guest's query; that lost the query and then
/// executed guest scratch as geometry.
///
/// What it would take to answer: the two words are unattributed. Nothing in the
/// captured fixtures varies them, because in a capture the stream *is* the
/// oracle. `runtime::heap_query` shows the shape a real reply takes.
pub fn apply_icb_host_resource_info<M: HostMemory + HostOps>(
    _state: &DeviceState,
    _host: &M,
    _task_id: u32,
    _info: &IcbHostResourceInfo,
) -> Result<IcbCommandMemory, IcbStatus> {
    Err(IcbStatus::Unsupported("icb_info_query_unanswered"))
}

/// Read attribute-stride u64 at `attributeStrideOffset + index*8`.
///
/// Returns `(stride, has)` — `has` is true when a stride table slot exists and
/// the stored value is non-zero, or when the slot exists and we treat any
/// stored value as authoritative (including 0 from host encode of
/// `has_attribute_stride` with stride 0 is rare; product uses non-zero for has).
fn read_attribute_stride(layout: &IcbCommandLayout, slot: &[u8], index: u32) -> (u64, bool) {
    let slots = icb_layout_attribute_stride_slot_count(layout);
    if slots == 0 || index >= slots || layout.attribute_stride_offset == 0 {
        return (0, false);
    }
    let off = layout.attribute_stride_offset as usize
        + (index as usize) * ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE;
    if off + 8 > slot.len() {
        return (0, false);
    }
    let stride = ld64(&slot[off..]);
    // Non-zero stride means the attributeStride API was used. Zero means unset
    // (plain setKernelBuffer/setVertexBuffer does not touch this table).
    if stride != 0 {
        (stride, true)
    } else {
        (0, false)
    }
}

#[cfg(test)]
fn write_attribute_stride(
    layout: &IcbCommandLayout,
    slot: &mut [u8],
    index: u32,
    stride: u64,
) -> Result<(), IcbStatus> {
    use crate::protocol::endian::st64;
    let slots = icb_layout_attribute_stride_slot_count(layout);
    if slots == 0 || index >= slots || layout.attribute_stride_offset == 0 {
        return Err(IcbStatus::Args("icb_attribute_stride_no_slot"));
    }
    let off = layout.attribute_stride_offset as usize
        + (index as usize) * ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE;
    if off + 8 > slot.len() {
        return Err(IcbStatus::Args("icb_attribute_stride_offset_oob"));
    }
    st64(&mut slot[off..], stride);
    Ok(())
}

fn buffer_gva_size<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
) -> Result<(u64, u64), IcbStatus> {
    objects::resolve_buffer_span(state, host, task_id, buffer_ref).map_err(
        |refusal| match refusal {
            objects::BufferSpanRefusal::Rung(rung) => {
                let slug = crate::observe::ladder_slugs!("icb_buffer")(rung);
                match rung {
                    objects::LadderRung::NoListEntry
                    | objects::LadderRung::DescRead { .. }
                    | objects::LadderRung::NoTaskSpace => IcbStatus::Missing(slug),
                    objects::LadderRung::WrongType { .. } => IcbStatus::BadDescriptor(slug),
                }
            }
            objects::BufferSpanRefusal::Decode => {
                IcbStatus::BadDescriptor(crate::observe::ladder_slug!("icb_buffer", desc_decode))
            }
            objects::BufferSpanRefusal::NoBacking => IcbStatus::Missing("icb_buffer_no_backing"),
        },
    )
}

/// Convert absolute bind VA → offset into buffer allocation (`handle << page_shift`).
///
/// PGSerializer stores `base+offset` in the bind VA field (not a separate offset).
/// `wire_va == 0` means base (offset 0). Fail-closed if VA is below base or past size.
fn offset_from_wire_va<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    wire_va: u64,
) -> Result<u64, IcbStatus> {
    if wire_va == 0 {
        return Ok(0);
    }
    let (base, size) = buffer_gva_size(state, host, task_id, buffer_ref)?;
    if wire_va < base {
        return Err(IcbStatus::Args("icb_wire_va_below_base"));
    }
    let off = wire_va - base;
    if off >= size {
        return Err(IcbStatus::Args("icb_wire_va_past_end"));
    }
    Ok(off)
}

/// Resolve wire VAs on a compute fill into buffer bind offsets (mutates in place).
pub fn resolve_compute_fill_offsets<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    fill: &mut IcbComputeFill,
) -> Result<(), IcbStatus> {
    for b in &mut fill.buffers {
        if b.wire_va != 0 {
            b.offset = offset_from_wire_va(state, host, task_id, b.buffer_ref, b.wire_va)?;
        }
    }
    Ok(())
}

/// Resolve wire VAs on a render fill into buffer bind / index offsets (mutates in place).
pub fn resolve_render_fill_offsets<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    fill: &mut IcbRenderFill,
) -> Result<(), IcbStatus> {
    for b in &mut fill.buffers {
        if b.wire_va != 0 {
            b.offset = offset_from_wire_va(state, host, task_id, b.buffer_ref, b.wire_va)?;
        }
    }
    match &mut fill.draw {
        IcbRenderDraw::Indexed {
            index_buffer_ref,
            index_wire_va,
            index_buffer_offset,
            ..
        } => {
            if *index_wire_va != 0 {
                *index_buffer_offset =
                    offset_from_wire_va(state, host, task_id, *index_buffer_ref, *index_wire_va)?;
            }
        }
        IcbRenderDraw::Patches {
            patch_index_buffer_ref,
            patch_index_wire_va,
            patch_index_buffer_offset,
            tessellation_factor,
            ..
        } => {
            if *patch_index_wire_va != 0 && *patch_index_buffer_ref != 0 {
                *patch_index_buffer_offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    *patch_index_buffer_ref,
                    *patch_index_wire_va,
                )?;
            }
            if tessellation_factor.wire_va != 0 && tessellation_factor.buffer_ref != 0 {
                tessellation_factor.offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    tessellation_factor.buffer_ref,
                    tessellation_factor.wire_va,
                )?;
            }
        }
        IcbRenderDraw::IndexedPatches {
            patch_index_buffer_ref,
            patch_index_wire_va,
            patch_index_buffer_offset,
            control_point_index_buffer_ref,
            control_point_index_wire_va,
            control_point_index_buffer_offset,
            tessellation_factor,
            ..
        } => {
            if *patch_index_wire_va != 0 && *patch_index_buffer_ref != 0 {
                *patch_index_buffer_offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    *patch_index_buffer_ref,
                    *patch_index_wire_va,
                )?;
            }
            if *control_point_index_wire_va != 0 {
                *control_point_index_buffer_offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    *control_point_index_buffer_ref,
                    *control_point_index_wire_va,
                )?;
            }
            if tessellation_factor.wire_va != 0 && tessellation_factor.buffer_ref != 0 {
                tessellation_factor.offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    tessellation_factor.buffer_ref,
                    tessellation_factor.wire_va,
                )?;
            }
        }
        IcbRenderDraw::Primitives { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => {}
    }
    Ok(())
}

/// One decoded, offset-resolved ICB command slot, ready for a backend to apply.
///
/// [`decode_icb_command_range`] returns these; what a backend does with one is
/// the only part of ICB execute that is backend-specific. The Metal arm fills a
/// real `MTLIndirectCommandBuffer` from them, the Vulkan arm replays them as
/// draws. Empty slots are not represented — the decoders skip them.
#[derive(Clone, Debug)]
pub enum IcbCommandFill {
    Compute(IcbComputeFill),
    Render(IcbRenderFill),
}

/// Decode guest command memory into host ICB fills for the given index range.
///
/// Dispatches compute vs render fills from wire `commandTypes` / slot
/// command-type tags, and resolves every wire VA into a buffer bind offset, so
/// the result names only refs and offsets. Nothing here touches a backend: this
/// is the half of ICB execute that is the same on all three pathways, and it is
/// portable so that the Vulkan arm has something to replay.
pub fn decode_icb_command_range<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    range_location: u64,
    range_length: u64,
) -> Result<Vec<IcbCommandFill>, IcbStatus> {
    use crate::runtime::decode::resource::{
        MTL_INDIRECT_CMD_CONCURRENT_DISPATCH, MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS,
        MTL_INDIRECT_CMD_DRAW, MTL_INDIRECT_CMD_DRAW_INDEXED,
        MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADS, MTL_INDIRECT_CMD_DRAW_PATCHES,
    };
    use crate::runtime::gva_mem;

    let (layout, max_kernel, max_vertex, max_fragment, command_types, max_cmds, mem) = {
        let reg = icb_registry().lock();
        let rec = reg
            .get(&(task_id, icb_ref))
            .ok_or(IcbStatus::Missing("icb_fill_not_cached"))?;
        let mem = rec
            .command_memory
            .ok_or(IcbStatus::Missing(ICB_FILL_NO_COMMAND_MEMORY))?;
        (
            rec.desc.layout,
            rec.desc.max_kernel_buffer_bind_count,
            rec.desc.max_vertex_buffer_bind_count,
            rec.desc.max_fragment_buffer_bind_count,
            rec.desc.command_types,
            rec.desc.max_command_count as u64,
            mem,
        )
    };
    if layout.command_size == 0 {
        return Err(IcbStatus::Args("icb_fill_zero_command_size"));
    }
    let end = range_location.saturating_add(range_length);
    if end > max_cmds {
        return Err(IcbStatus::Args("icb_fill_range_past_capacity"));
    }
    let need = end.saturating_mul(layout.command_size as u64);
    if need > mem.byte_len {
        return Err(IcbStatus::Args("icb_fill_range_past_memory"));
    }
    let mut bytes = vec![0u8; need as usize];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        mem.gva,
        &mut bytes,
        state.page_shift,
    )
    .map_err(|_| IcbStatus::MetalFailed("icb_fill_command_memory_read"))?;

    let is_compute = command_types
        & (MTL_INDIRECT_CMD_CONCURRENT_DISPATCH | MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS)
        != 0;
    let is_render = command_types
        & (MTL_INDIRECT_CMD_DRAW
            | MTL_INDIRECT_CMD_DRAW_INDEXED
            | MTL_INDIRECT_CMD_DRAW_PATCHES
            | MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES
            | MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS
            | MTL_INDIRECT_CMD_DRAW_MESH_THREADS)
        != 0;

    let mut out = Vec::new();
    for i in range_location..end {
        let off = (i as usize) * (layout.command_size as usize);
        let slot = &bytes[off..off + layout.command_size as usize];
        if is_compute || !is_render {
            // Prefer compute when ConcurrentDispatch bits are set; empty slots skip.
            if let Some(mut fill) = decode_compute_command_slot(&layout, slot, max_kernel)? {
                fill.command_index = i as u32;
                resolve_compute_fill_offsets(state, host, task_id, &mut fill)?;
                out.push(IcbCommandFill::Compute(fill));
                continue;
            }
        }
        if is_render {
            if let Some(mut fill) =
                decode_render_command_slot(&layout, slot, max_vertex, max_fragment)?
            {
                fill.command_index = i as u32;
                resolve_render_fill_offsets(state, host, task_id, &mut fill)?;
                out.push(IcbCommandFill::Render(fill));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
