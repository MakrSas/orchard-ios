//! Lifting the render-encoder records.
//!
//! # Forty-four opcodes, and the draws are where the shape lives
//!
//! Six of the eight draw shapes arrive in two encodings: a compact one whose
//! counts are sixteen bits and a wide one whose counts are sixty-four. That is
//! an encoding and not a meaning, so both lift to the same record and the
//! counts are widened. What is *not* widened away is
//! [`RenderKind::is_wide_encoding`], which a census can still ask.
//!
//! The one field where the two encodings genuinely disagree is `base_vertex`:
//! the compact form's is truncated to sixteen bits by Apple's serializer,
//! upstream of this device, while the wide form's is sign-extended and carries
//! the guest's whole value. The record carries `i64`, which is the width at
//! which both are representable.
//!
//! Field order is not shared between the encodings either — `Draw` puts the
//! primitive type first at 32 bits and `DrawInstanced` puts it last at 16 —
//! which is why every field comes from a wire view.
//!
//! # The singular and plural viewport and scissor records are one variant
//!
//! The plural forms are the singular record with a count in front, and the
//! element type is the singular record's whole payload. So both lift to a
//! borrowed slice, and the singular one is that slice at length one. This is
//! not true of the bind records — there the singular *is* the plural at
//! `count == 1` and shares its opcode — and stating the difference is what
//! keeps a reader from assuming a family rule the wire does not have.
//!
//! # Two ordinals are parsed and one is not
//!
//! `MTLIndexType` is parsed, because its two values differ by a factor of two
//! and reading the wrong one either overruns the index buffer or reads half the
//! indices. A store action is not, and neither is `MTLPrimitiveType`: which
//! ones a host can perform is a capability question the executor answers, and
//! folding an unknown ordinal onto a known one draws or stores the wrong thing
//! rather than refusing.

use super::{no_record, short, DecodeRefusal};
use crate::closure::Rail;
use crate::render::{IndexType, RenderKind, ShaderStage, StoreActionTarget};
use reims_vgpu_wire::op::Op;
use reims_vgpu_wire::ops::render as wire;
use reims_vgpu_wire::ops::render::{
    BufferBind, BufferStrideBind, RefBind, SamplerLodBind, ScissorRect, Viewport,
};
use reims_vgpu_wire::ops::render_pass::{self, RenderPassBody};

/// The index buffer an indexed draw reads, with the guest's ref.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexRef {
    pub buffer_ref: u32,
    pub offset: u64,
    pub index_type: IndexType,
}

/// A buffer window a draw reads its counts from, with the guest's ref.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectRef {
    pub buffer_ref: u32,
    pub offset: u64,
}

/// How many instances a draw runs, and where they start.
///
/// `None` is the plain form, whose record carries no instance count at all —
/// distinct from a guest asking for one instance, which is a count of one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Instancing {
    pub count: Option<u64>,
    pub base: u64,
}

/// A draw whose vertices come from the record's own counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Primitives {
    pub primitive: u32,
    pub vertex_start: u64,
    pub vertex_count: u64,
    pub instances: Instancing,
}

/// A draw whose vertices come from an index buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Indexed {
    pub primitive: u32,
    pub index: IndexRef,
    pub index_count: u64,
    pub instances: Instancing,
    pub base_vertex: i64,
}

/// A draw whose counts come from a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimitivesIndirect {
    pub primitive: u32,
    pub arguments: IndirectRef,
}

/// An indexed draw whose counts come from a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedIndirect {
    pub primitive: u32,
    pub index: IndexRef,
    pub arguments: IndirectRef,
}

/// One lifted draw.
///
/// Each variant carries **one named payload** rather than inline fields, for
/// the same reason [`RenderRecord`] does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawRecord {
    Primitives(Primitives),
    Indexed(Indexed),
    PrimitivesIndirect(PrimitivesIndirect),
    IndexedIndirect(IndexedIndirect),
}

/// A run of buffer bindings into one stage's argument table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindBuffers<'a> {
    pub stage: ShaderStage,
    /// The slot the first entry lands in.
    pub first: u32,
    pub entries: &'a [BufferBind],
}

/// The vertex stage's attribute-stride bind. There is no fragment form: the
/// API has no fragment attribute stride, which is why this payload names no
/// stage where the others do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindBuffersWithStride<'a> {
    pub first: u32,
    pub entries: &'a [BufferStrideBind],
}

/// A run of texture bindings into one stage's argument table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindTextures<'a> {
    pub stage: ShaderStage,
    pub first: u32,
    pub entries: &'a [RefBind],
}

/// A run of sampler bindings into one stage's argument table.
///
/// The same wire entry as [`BindTextures`] and a different argument table, so
/// they are two payloads rather than one — a run written into the wrong table
/// binds nothing the shader asked for and refuses nothing either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindSamplers<'a> {
    pub stage: ShaderStage,
    pub first: u32,
    pub entries: &'a [RefBind],
}

/// A run of sampler bindings that also carry a LOD clamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindSamplersWithLod<'a> {
    pub stage: ShaderStage,
    pub first: u32,
    pub entries: &'a [SamplerLodBind],
}

/// Move an already-bound buffer's offset, and its stride when the record
/// carries one. It names no buffer: the slot keeps whatever it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebindBufferOffset {
    pub stage: ShaderStage,
    pub index: u32,
    pub offset: u64,
    /// `None` for the form that does not carry the field at all — not a stride
    /// of zero, which is a stride the guest could state.
    pub stride: Option<u64>,
}

/// The render pipeline state the following draws run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetPipeline {
    pub pipeline_ref: u32,
}

/// The depth-stencil state the following draws test against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetDepthStencilState {
    pub state_ref: u32,
}

/// The pass descriptor, borrowed rather than copied: it is 592 bytes, and a
/// record that carried it by value would make every eight-byte variant that
/// size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteDescriptor<'a> {
    pub descriptor: &'a RenderPassBody,
}

/// `MTLCullMode`, as the record carries it.
///
/// A named payload for one `u64`, and the four rasterizer-state records are the
/// reason the rule is worth the noise: cull mode, winding, depth-clip mode and
/// fill mode are four wire-identical words, and a bare `u64` makes an executor
/// that pairs the wrong one with the wrong state field type-correct. That is
/// the exact defect the flat device command allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetCullMode {
    pub mode: u64,
}

/// `MTLWinding`, as the record carries it. See [`SetCullMode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetFrontFacingWinding {
    pub winding: u64,
}

/// `MTLDepthClipMode`, as the record carries it. See [`SetCullMode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetDepthClipMode {
    pub mode: u64,
}

/// `MTLTriangleFillMode`, as the record carries it. See [`SetCullMode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetTriangleFillMode {
    pub mode: u64,
}

/// Three `float`s, as the guest's bits. Bits rather than `f32` because a state
/// table has to compare, and float equality makes a NaN differ from itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetDepthBias {
    pub bias_bits: u32,
    pub slope_scale_bits: u32,
    pub clamp_bits: u32,
}

/// One `float`, as the guest's bits, for the reason [`SetDepthBias`]'s three
/// are. `setLineWidth:` shares its wire form with
/// `setTessellationFactorScale:`, so the opcode is the only thing that
/// separates a state this device carries from one it does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetLineWidth {
    pub width_bits: u32,
}

/// Four `float`s, as the guest's bits. See [`SetDepthBias`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetBlendColor {
    pub red_bits: u32,
    pub green_bits: u32,
    pub blue_bits: u32,
    pub alpha_bits: u32,
}

/// The two stencil reference values, which one record sets together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetStencilReference {
    pub front: u32,
    pub back: u32,
}

/// A store action, and which attachment it is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetStoreAction {
    pub target: StoreActionTarget,
    pub action: u64,
}

/// The visibility-result mode and the buffer offset it writes at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetVisibilityResultMode {
    pub mode: u64,
    pub offset: u64,
}

/// One lifted render record.
///
/// Each variant carries **one named payload** rather than inline fields, so a
/// consumer can take the record it handles by reference and cannot be handed a
/// different one. The two exceptions carry a slice of a type nothing else on
/// this rail carries — a `&[Viewport]` is not confusable with a
/// `&[ScissorRect]` — so a wrapper around them would name what the element type
/// already says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderRecord<'a> {
    Draw(DrawRecord),

    BindBuffers(BindBuffers<'a>),
    BindBuffersWithStride(BindBuffersWithStride<'a>),
    BindTextures(BindTextures<'a>),
    BindSamplers(BindSamplers<'a>),
    BindSamplersWithLod(BindSamplersWithLod<'a>),
    RebindBufferOffset(RebindBufferOffset),

    SetPipeline(SetPipeline),
    SetDepthStencilState(SetDepthStencilState),
    WriteDescriptor(WriteDescriptor<'a>),

    SetViewports(&'a [Viewport]),
    SetScissorRects(&'a [ScissorRect]),
    SetCullMode(SetCullMode),
    SetFrontFacingWinding(SetFrontFacingWinding),
    SetDepthClipMode(SetDepthClipMode),
    SetTriangleFillMode(SetTriangleFillMode),
    SetDepthBias(SetDepthBias),
    SetLineWidth(SetLineWidth),
    SetBlendColor(SetBlendColor),
    SetStencilReference(SetStencilReference),
    SetStoreAction(SetStoreAction),
    SetVisibilityResultMode(SetVisibilityResultMode),
}

impl RenderRecord<'_> {
    /// The stage a bind record wrote, if this is one.
    #[must_use]
    pub const fn stage(&self) -> Option<ShaderStage> {
        match self {
            Self::BindBuffers(r) => Some(r.stage),
            Self::BindTextures(r) => Some(r.stage),
            Self::BindSamplers(r) => Some(r.stage),
            Self::BindSamplersWithLod(r) => Some(r.stage),
            Self::RebindBufferOffset(r) => Some(r.stage),
            Self::BindBuffersWithStride(_) => Some(ShaderStage::Vertex),
            _ => None,
        }
    }
}

/// Lift a render record out of its bytes.
pub fn decode<'a>(op: &Op<'a>) -> Result<RenderRecord<'a>, DecodeRefusal> {
    let opcode = op.opcode();
    let Some(kind) = RenderKind::of_opcode(opcode) else {
        return Err(no_record(Rail::Render, opcode));
    };
    if kind.draw_shape().is_some() {
        return Ok(RenderRecord::Draw(draw(kind, op)?));
    }
    state(kind, op)
}

/// The stage a bind record names, which the opcode alone decides.
///
/// Unreachable for anything but a bind: `bind_stage` is total over the record
/// set and every caller here has already matched a bind arm. Falling back to
/// the vertex stage would be a guess; this is a decode failure and is reported
/// as one.
fn bind_stage(kind: RenderKind, op: &Op<'_>) -> Result<ShaderStage, DecodeRefusal> {
    kind.bind_stage().ok_or(DecodeRefusal::Unjudged {
        rail: Rail::Render,
        opcode: op.opcode(),
    })
}

fn state<'a>(kind: RenderKind, op: &Op<'a>) -> Result<RenderRecord<'a>, DecodeRefusal> {
    let opcode = op.opcode();
    let have = op.payload.len();
    let fail = |need: usize| short(Rail::Render, opcode, have, need);
    let entries = |entry: usize| bind_failure(op, entry);

    Ok(match kind {
        RenderKind::SetVertexBuffers | RenderKind::SetFragmentBuffers => {
            let (head, slots) =
                wire::buffer_binds(op).map_err(|_| entries(core::mem::size_of::<BufferBind>()))?;
            RenderRecord::BindBuffers(BindBuffers {
                stage: bind_stage(kind, op)?,
                first: head.first.get(),
                entries: slots,
            })
        }
        RenderKind::SetVertexBuffersWithStride => {
            let (head, slots) = wire::buffer_stride_binds(op)
                .map_err(|_| entries(core::mem::size_of::<BufferStrideBind>()))?;
            RenderRecord::BindBuffersWithStride(BindBuffersWithStride {
                first: head.first.get(),
                entries: slots,
            })
        }
        RenderKind::SetVertexTextures
        | RenderKind::SetFragmentTextures
        | RenderKind::SetVertexSamplers
        | RenderKind::SetFragmentSamplers => {
            let (head, slots) =
                wire::ref_binds(op).map_err(|_| entries(core::mem::size_of::<RefBind>()))?;
            let stage = bind_stage(kind, op)?;
            let first = head.first.get();
            if matches!(
                kind,
                RenderKind::SetVertexTextures | RenderKind::SetFragmentTextures
            ) {
                RenderRecord::BindTextures(BindTextures {
                    stage,
                    first,
                    entries: slots,
                })
            } else {
                RenderRecord::BindSamplers(BindSamplers {
                    stage,
                    first,
                    entries: slots,
                })
            }
        }
        RenderKind::SetVertexSamplersWithLod | RenderKind::SetFragmentSamplersWithLod => {
            let (head, slots) = wire::sampler_lod_binds(op)
                .map_err(|_| entries(core::mem::size_of::<SamplerLodBind>()))?;
            RenderRecord::BindSamplersWithLod(BindSamplersWithLod {
                stage: bind_stage(kind, op)?,
                first: head.first.get(),
                entries: slots,
            })
        }
        RenderKind::SetVertexBufferOffset | RenderKind::SetFragmentBufferOffset => {
            let r = wire::buffer_offset(op)
                .map_err(|_| fail(core::mem::size_of::<wire::BufferOffset>()))?;
            RenderRecord::RebindBufferOffset(RebindBufferOffset {
                stage: bind_stage(kind, op)?,
                index: r.index.get(),
                offset: r.offset.get(),
                stride: None,
            })
        }
        RenderKind::SetVertexBufferOffsetStride => {
            let r = wire::buffer_offset_stride(op)
                .map_err(|_| fail(core::mem::size_of::<wire::BufferOffsetStride>()))?;
            RenderRecord::RebindBufferOffset(RebindBufferOffset {
                stage: bind_stage(kind, op)?,
                index: r.index.get(),
                offset: r.offset.get(),
                stride: Some(r.attribute_stride.get()),
            })
        }
        RenderKind::SetRenderPipelineState => RenderRecord::SetPipeline(SetPipeline {
            pipeline_ref: wire::state_ref(op)
                .map_err(|_| fail(core::mem::size_of::<wire::StateRef>()))?
                .object_ref
                .get(),
        }),
        RenderKind::SetDepthStencilState => {
            RenderRecord::SetDepthStencilState(SetDepthStencilState {
                state_ref: wire::state_ref(op)
                    .map_err(|_| fail(core::mem::size_of::<wire::StateRef>()))?
                    .object_ref
                    .get(),
            })
        }
        RenderKind::WriteDescriptor => RenderRecord::WriteDescriptor(WriteDescriptor {
            descriptor: render_pass::render_pass(op)
                .map_err(|_| fail(core::mem::size_of::<RenderPassBody>()))?,
        }),
        RenderKind::SetViewport => RenderRecord::SetViewports(
            reims_vgpu_wire::view_slice::<Viewport>(op.payload, 1)
                .map_err(|_| fail(core::mem::size_of::<Viewport>()))?,
        ),
        RenderKind::SetViewports => {
            let (_, ports) = wire::set_viewports(op)
                .map_err(|_| counted(op, 4, core::mem::size_of::<Viewport>()))?;
            RenderRecord::SetViewports(ports)
        }
        RenderKind::SetScissorRect => RenderRecord::SetScissorRects(
            reims_vgpu_wire::view_slice::<ScissorRect>(op.payload, 1)
                .map_err(|_| fail(core::mem::size_of::<ScissorRect>()))?,
        ),
        RenderKind::SetScissorRects => {
            let (_, rects) = wire::set_scissor_rects(op)
                .map_err(|_| counted(op, 8, core::mem::size_of::<ScissorRect>()))?;
            RenderRecord::SetScissorRects(rects)
        }
        RenderKind::SetCullMode
        | RenderKind::SetFrontFacingWinding
        | RenderKind::SetDepthClipMode
        | RenderKind::SetTriangleFillMode => {
            let mode = wire::mode_state(op)
                .map_err(|_| fail(core::mem::size_of::<wire::ModeState>()))?
                .mode
                .get();
            match kind {
                RenderKind::SetCullMode => RenderRecord::SetCullMode(SetCullMode { mode }),
                RenderKind::SetFrontFacingWinding => {
                    RenderRecord::SetFrontFacingWinding(SetFrontFacingWinding { winding: mode })
                }
                RenderKind::SetDepthClipMode => {
                    RenderRecord::SetDepthClipMode(SetDepthClipMode { mode })
                }
                _ => RenderRecord::SetTriangleFillMode(SetTriangleFillMode { mode }),
            }
        }
        RenderKind::SetDepthBias => {
            let r = wire::set_depth_bias(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DepthBias>()))?;
            RenderRecord::SetDepthBias(SetDepthBias {
                bias_bits: r.bias.get().to_bits(),
                slope_scale_bits: r.slope_scale.get().to_bits(),
                clamp_bits: r.clamp.get().to_bits(),
            })
        }
        RenderKind::SetLineWidth => {
            let r = wire::float_state(op)
                .map_err(|_| fail(core::mem::size_of::<wire::FloatState>()))?;
            RenderRecord::SetLineWidth(SetLineWidth {
                width_bits: r.value.get().to_bits(),
            })
        }
        RenderKind::SetBlendColor => {
            let r = wire::set_blend_color(op)
                .map_err(|_| fail(core::mem::size_of::<wire::BlendColor>()))?;
            RenderRecord::SetBlendColor(SetBlendColor {
                red_bits: r.red.get().to_bits(),
                green_bits: r.green.get().to_bits(),
                blue_bits: r.blue.get().to_bits(),
                alpha_bits: r.alpha.get().to_bits(),
            })
        }
        RenderKind::SetStencilReference => {
            let r = wire::set_stencil_reference(op)
                .map_err(|_| fail(core::mem::size_of::<wire::StencilReference>()))?;
            RenderRecord::SetStencilReference(SetStencilReference {
                front: r.front.get(),
                back: r.back.get(),
            })
        }
        RenderKind::SetColorStoreAction => {
            let r = wire::set_color_store_action(op)
                .map_err(|_| fail(core::mem::size_of::<wire::ColorStoreAction>()))?;
            RenderRecord::SetStoreAction(SetStoreAction {
                target: StoreActionTarget::Color(r.index.get()),
                action: u64::from(r.store_action.get()),
            })
        }
        // The depth and stencil forms name their attachment by being
        // themselves, so they carry no index — and their action is 64 bits
        // where the colour form's is 32. Same statement, three records.
        RenderKind::SetDepthStoreAction | RenderKind::SetStencilStoreAction => {
            let action = wire::mode_state(op)
                .map_err(|_| fail(core::mem::size_of::<wire::ModeState>()))?
                .mode
                .get();
            RenderRecord::SetStoreAction(SetStoreAction {
                target: if matches!(kind, RenderKind::SetDepthStoreAction) {
                    StoreActionTarget::Depth
                } else {
                    StoreActionTarget::Stencil
                },
                action,
            })
        }
        RenderKind::SetVisibilityResultMode => {
            let r = wire::set_visibility_result_mode(op)
                .map_err(|_| fail(core::mem::size_of::<wire::VisibilityResult>()))?;
            RenderRecord::SetVisibilityResultMode(SetVisibilityResultMode {
                mode: r.mode.get(),
                offset: r.offset.get(),
            })
        }
        // Every draw kind was taken by `decode` before this function ran.
        _ => {
            return Err(DecodeRefusal::Unjudged {
                rail: Rail::Render,
                opcode,
            })
        }
    })
}

fn draw(kind: RenderKind, op: &Op<'_>) -> Result<DrawRecord, DecodeRefusal> {
    let opcode = op.opcode();
    let have = op.payload.len();
    let fail = |need: usize| short(Rail::Render, opcode, have, need);
    let index_type = |raw: u16| {
        IndexType::parse(raw).ok_or(DecodeRefusal::UndefinedOrdinal {
            rail: Rail::Render,
            opcode,
            field: "index_type",
            value: u32::from(raw),
        })
    };

    Ok(match kind {
        RenderKind::Draw => {
            let r = wire::draw(op).map_err(|_| fail(core::mem::size_of::<wire::Draw>()))?;
            DrawRecord::Primitives(Primitives {
                primitive: r.primitive_type.get(),
                vertex_start: u64::from(r.vertex_start.get()),
                vertex_count: u64::from(r.vertex_count.get()),
                instances: Instancing::default(),
            })
        }
        RenderKind::DrawWide => {
            let r =
                wire::draw_wide(op).map_err(|_| fail(core::mem::size_of::<wire::DrawWide>()))?;
            DrawRecord::Primitives(Primitives {
                primitive: r.primitive_type.get(),
                vertex_start: r.vertex_start.get(),
                vertex_count: r.vertex_count.get(),
                instances: Instancing::default(),
            })
        }
        RenderKind::DrawInstanced => {
            let r = wire::draw_instanced(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawInstanced>()))?;
            DrawRecord::Primitives(Primitives {
                primitive: u32::from(r.primitive_type.get()),
                vertex_start: u64::from(r.vertex_start.get()),
                vertex_count: u64::from(r.vertex_count.get()),
                instances: Instancing {
                    count: Some(u64::from(r.instance_count.get())),
                    base: 0,
                },
            })
        }
        RenderKind::DrawInstancedWide => {
            let r = wire::draw_instanced_wide(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawInstancedWide>()))?;
            DrawRecord::Primitives(Primitives {
                primitive: u32::from(r.primitive_type.get()),
                vertex_start: r.vertex_start.get(),
                vertex_count: r.vertex_count.get(),
                instances: Instancing {
                    count: Some(r.instance_count.get()),
                    base: 0,
                },
            })
        }
        RenderKind::DrawInstancedBase => {
            let r = wire::draw_instanced_base(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawInstancedBase>()))?;
            DrawRecord::Primitives(Primitives {
                primitive: u32::from(r.primitive_type.get()),
                vertex_start: u64::from(r.vertex_start.get()),
                vertex_count: u64::from(r.vertex_count.get()),
                instances: Instancing {
                    count: Some(u64::from(r.instance_count.get())),
                    base: u64::from(r.base_instance.get()),
                },
            })
        }
        RenderKind::DrawInstancedBaseWide => {
            let r = wire::draw_instanced_base_wide(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawInstancedBaseWide>()))?;
            DrawRecord::Primitives(Primitives {
                primitive: u32::from(r.primitive_type.get()),
                vertex_start: r.vertex_start.get(),
                vertex_count: r.vertex_count.get(),
                instances: Instancing {
                    count: Some(r.instance_count.get()),
                    base: r.base_instance.get(),
                },
            })
        }
        RenderKind::DrawIndexed => {
            let r = wire::draw_indexed(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawIndexed>()))?;
            DrawRecord::Indexed(Indexed {
                primitive: u32::from(r.primitive_type.get()),
                index: IndexRef {
                    buffer_ref: r.index_buffer_ref.get(),
                    offset: u64::from(r.index_buffer_offset.get()),
                    index_type: index_type(r.index_type.get())?,
                },
                index_count: u64::from(r.index_count.get()),
                instances: Instancing::default(),
                base_vertex: 0,
            })
        }
        RenderKind::DrawIndexedWide => {
            let r = wire::draw_indexed_wide(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawIndexedWide>()))?;
            DrawRecord::Indexed(Indexed {
                primitive: u32::from(r.primitive_type.get()),
                index: IndexRef {
                    buffer_ref: r.index_buffer_ref.get(),
                    offset: r.index_buffer_offset.get(),
                    index_type: index_type(r.index_type.get())?,
                },
                index_count: r.index_count.get(),
                instances: Instancing::default(),
                base_vertex: 0,
            })
        }
        RenderKind::DrawIndexedInstanced => {
            let r = wire::draw_indexed_instanced(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawIndexedInstanced>()))?;
            DrawRecord::Indexed(Indexed {
                primitive: u32::from(r.primitive_type.get()),
                index: IndexRef {
                    buffer_ref: r.index_buffer_ref.get(),
                    offset: u64::from(r.index_buffer_offset.get()),
                    index_type: index_type(r.index_type.get())?,
                },
                index_count: u64::from(r.index_count.get()),
                instances: Instancing {
                    count: Some(u64::from(r.instance_count.get())),
                    base: 0,
                },
                base_vertex: 0,
            })
        }
        RenderKind::DrawIndexedInstancedWide => {
            let r = wire::draw_indexed_instanced_wide(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawIndexedInstancedWide>()))?;
            DrawRecord::Indexed(Indexed {
                primitive: u32::from(r.primitive_type.get()),
                index: IndexRef {
                    buffer_ref: r.index_buffer_ref.get(),
                    offset: r.index_buffer_offset.get(),
                    index_type: index_type(r.index_type.get())?,
                },
                index_count: r.index_count.get(),
                instances: Instancing {
                    count: Some(r.instance_count.get()),
                    base: 0,
                },
                base_vertex: 0,
            })
        }
        RenderKind::DrawIndexedInstancedBase => {
            let r = wire::draw_indexed_instanced_base(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawIndexedInstancedBase>()))?;
            DrawRecord::Indexed(Indexed {
                primitive: u32::from(r.primitive_type.get()),
                index: IndexRef {
                    buffer_ref: r.index_buffer_ref.get(),
                    offset: u64::from(r.index_buffer_offset.get()),
                    index_type: index_type(r.index_type.get())?,
                },
                index_count: u64::from(r.index_count.get()),
                instances: Instancing {
                    count: Some(u64::from(r.instance_count.get())),
                    base: u64::from(r.base_instance.get()),
                },
                base_vertex: i64::from(r.base_vertex.get()),
            })
        }
        RenderKind::DrawIndexedInstancedBaseWide => {
            let r = wire::draw_indexed_instanced_base_wide(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawIndexedInstancedBaseWide>()))?;
            DrawRecord::Indexed(Indexed {
                primitive: u32::from(r.primitive_type.get()),
                index: IndexRef {
                    buffer_ref: r.index_buffer_ref.get(),
                    offset: r.index_buffer_offset.get(),
                    index_type: index_type(r.index_type.get())?,
                },
                index_count: r.index_count.get(),
                instances: Instancing {
                    count: Some(r.instance_count.get()),
                    base: r.base_instance.get(),
                },
                base_vertex: r.base_vertex.get(),
            })
        }
        RenderKind::DrawIndirect => {
            let r = wire::draw_indirect(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawIndirect>()))?;
            DrawRecord::PrimitivesIndirect(PrimitivesIndirect {
                primitive: u32::from(r.primitive_type.get()),
                arguments: IndirectRef {
                    buffer_ref: r.indirect_buffer_ref.get(),
                    offset: r.indirect_buffer_offset.get(),
                },
            })
        }
        RenderKind::DrawIndexedIndirect => {
            let r = wire::draw_indexed_indirect(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DrawIndexedIndirect>()))?;
            DrawRecord::IndexedIndirect(IndexedIndirect {
                primitive: u32::from(r.primitive_type.get()),
                index: IndexRef {
                    buffer_ref: r.index_buffer_ref.get(),
                    offset: r.index_buffer_offset.get(),
                    index_type: index_type(r.index_type.get())?,
                },
                arguments: IndirectRef {
                    buffer_ref: r.indirect_buffer_ref.get(),
                    offset: r.indirect_buffer_offset.get(),
                },
            })
        }
        // `decode` only calls this for a kind with a draw shape.
        _ => {
            return Err(DecodeRefusal::Unjudged {
                rail: Rail::Render,
                opcode,
            })
        }
    })
}

/// The refusal for a bind record whose head or entries did not fit.
fn bind_failure(op: &Op<'_>, entry_size: usize) -> DecodeRefusal {
    counted_at(op, core::mem::size_of::<wire::BindHeader>(), 4, entry_size)
}

/// The refusal for a counted record whose head sits entirely in front of the
/// array, with the count as its whole head.
fn counted(op: &Op<'_>, head_len: usize, entry_size: usize) -> DecodeRefusal {
    counted_at(op, head_len, 0, entry_size)
}

/// A counted record that did not fit, with the count read where it lives.
///
/// The head parsing and the array fitting are different failures, and only the
/// second has a count to report: "the guest asked for 200" beside "the record
/// held 12" is the pair that says which of the two is wrong. `count_at` is
/// where the count sits inside the head, because the bind records put a `first`
/// in front of theirs and the plural viewport and scissor records do not.
fn counted_at(op: &Op<'_>, head_len: usize, count_at: usize, entry_size: usize) -> DecodeRefusal {
    let have = op.payload.len();
    if have < head_len {
        return short(Rail::Render, op.opcode(), have, head_len);
    }
    // The plural scissor record's count is eight bytes wide; every other one is
    // four. Reading four of them either way is right: no legal count reaches
    // 2^32, and the wire view refuses a high word that is non-zero before this
    // is ever reached.
    let mut count = [0u8; 4];
    count.copy_from_slice(&op.payload[count_at..count_at + 4]);
    let count = u32::from_le_bytes(count);
    let need = head_len.saturating_add((count as usize).saturating_mul(entry_size));
    if have >= need {
        return short(Rail::Render, op.opcode(), have, need);
    }
    DecodeRefusal::CountOverruns {
        rail: Rail::Render,
        opcode: op.opcode(),
        count,
        have,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use reims_vgpu_wire::op::{op, OP_HEADER_LEN};

    fn record(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let total = (OP_HEADER_LEN + payload.len()) as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&total.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn lift(bytes: &[u8]) -> Result<RenderRecord<'_>, DecodeRefusal> {
        decode(&op(bytes, 0).expect("framed"))
    }

    /// The compact and wide encodings of one draw shape lift to the same
    /// record. That is the claim the widening rests on: the encoding is not a
    /// meaning, and a caller that had to match on it would be matching on how
    /// many bits a count arrived in.
    #[test]
    fn the_two_encodings_of_a_draw_lift_to_the_same_record() {
        let mut compact = 3u32.to_le_bytes().to_vec();
        compact.extend_from_slice(&7u16.to_le_bytes());
        compact.extend_from_slice(&9u16.to_le_bytes());
        let mut wide = 3u32.to_le_bytes().to_vec();
        wide.extend_from_slice(&7u64.to_le_bytes());
        wide.extend_from_slice(&9u64.to_le_bytes());

        let expected = RenderRecord::Draw(DrawRecord::Primitives(Primitives {
            primitive: 3,
            vertex_start: 7,
            vertex_count: 9,
            instances: Instancing::default(),
        }));
        assert_eq!(lift(&record(wire::OPCODE_DRAW, &compact)), Ok(expected));
        assert_eq!(lift(&record(wire::OPCODE_DRAW_WIDE, &wide)), Ok(expected));
    }

    /// The primitive type is first and 32 bits in the plain draw and last and
    /// 16 bits in the instanced one. Both reach the same field, which they
    /// could not if the layout were shared.
    #[test]
    fn the_primitive_type_moves_between_draw_records_and_still_lands() {
        let mut instanced = 7u16.to_le_bytes().to_vec();
        instanced.extend_from_slice(&9u16.to_le_bytes());
        instanced.extend_from_slice(&2u16.to_le_bytes());
        instanced.extend_from_slice(&3u16.to_le_bytes());
        assert_eq!(
            lift(&record(wire::OPCODE_DRAW_INSTANCED, &instanced)),
            Ok(RenderRecord::Draw(DrawRecord::Primitives(Primitives {
                primitive: 3,
                vertex_start: 7,
                vertex_count: 9,
                instances: Instancing {
                    count: Some(2),
                    base: 0
                },
            })))
        );
    }

    /// A plain draw carries no instance count, and that is `None` rather than
    /// one. A guest asking for a single instance said something the plain form
    /// cannot say.
    #[test]
    fn a_plain_draw_has_no_instance_count_rather_than_a_count_of_one() {
        let mut plain = 3u32.to_le_bytes().to_vec();
        plain.extend_from_slice(&0u16.to_le_bytes());
        plain.extend_from_slice(&3u16.to_le_bytes());
        let RenderRecord::Draw(DrawRecord::Primitives(Primitives { instances, .. })) =
            lift(&record(wire::OPCODE_DRAW, &plain)).expect("lifted")
        else {
            panic!("not a plain draw");
        };
        assert_eq!(instances.count, None);
    }

    /// The wide indexed base draw sign-extends its base vertex, so a value the
    /// compact form could not hold survives the lift.
    #[test]
    fn a_wide_base_vertex_below_the_compact_range_survives() {
        let mut payload = 3u16.to_le_bytes().to_vec();
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&5151u32.to_le_bytes());
        payload.extend_from_slice(&0x100u64.to_le_bytes());
        payload.extend_from_slice(&6u64.to_le_bytes());
        payload.extend_from_slice(&2u64.to_le_bytes());
        payload.extend_from_slice(&(-70_000i64).to_le_bytes());
        payload.extend_from_slice(&4u64.to_le_bytes());
        let bytes = record(wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE, &payload);
        let RenderRecord::Draw(DrawRecord::Indexed(Indexed { base_vertex, .. })) =
            lift(&bytes).expect("lifted")
        else {
            panic!("not an indexed draw");
        };
        assert_eq!(base_vertex, -70_000);
    }

    /// An index type outside `MTLIndexType` is refused by name. The two widths
    /// differ by a factor of two, so guessing either overruns the buffer or
    /// reads half the indices.
    #[test]
    fn an_undefined_index_type_is_refused() {
        let mut payload = 3u16.to_le_bytes().to_vec();
        payload.extend_from_slice(&9u16.to_le_bytes());
        payload.extend_from_slice(&5151u32.to_le_bytes());
        payload.extend_from_slice(&6u16.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        assert_eq!(
            lift(&record(wire::OPCODE_DRAW_INDEXED, &payload)),
            Err(DecodeRefusal::UndefinedOrdinal {
                rail: Rail::Render,
                opcode: wire::OPCODE_DRAW_INDEXED,
                field: "index_type",
                value: 9,
            })
        );
    }

    /// The singular viewport is the plural one at length one, and both borrow
    /// the guest's bytes. The plural record's count sits in front of the array;
    /// the singular record has no count at all.
    #[test]
    fn the_singular_and_plural_viewports_lift_to_one_borrowed_slice() {
        let one: Vec<u8> = [1.0f64, 2.0, 3.0, 4.0, 0.0, 1.0]
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let single = record(wire::OPCODE_SET_VIEWPORT, &one);
        let RenderRecord::SetViewports(ports) = lift(&single).expect("lifted") else {
            panic!("not viewports");
        };
        assert_eq!(ports.len(), 1);
        assert!(single.as_ptr_range().contains(&ports.as_ptr().cast::<u8>()));

        let mut plural_payload = 2u32.to_le_bytes().to_vec();
        plural_payload.extend_from_slice(&one);
        plural_payload.extend_from_slice(&one);
        let plural = record(wire::OPCODE_SET_VIEWPORTS, &plural_payload);
        let RenderRecord::SetViewports(ports) = lift(&plural).expect("lifted") else {
            panic!("not viewports");
        };
        assert_eq!(ports.len(), 2);
        assert_eq!(ports[0], ports[1]);
    }

    /// The three store-action records make the same statement about different
    /// attachments, and only the colour one carries a slot. The target comes
    /// from the opcode, which is the only place it could come from.
    #[test]
    fn the_three_store_actions_name_their_attachment_from_the_opcode() {
        let mut colour = 2u32.to_le_bytes().to_vec();
        colour.extend_from_slice(&3u32.to_le_bytes());
        assert_eq!(
            lift(&record(wire::OPCODE_SET_COLOR_STORE_ACTION, &colour)),
            Ok(RenderRecord::SetStoreAction(SetStoreAction {
                target: StoreActionTarget::Color(3),
                action: 2,
            }))
        );
        for (opcode, target) in [
            (
                wire::OPCODE_SET_DEPTH_STORE_ACTION,
                StoreActionTarget::Depth,
            ),
            (
                wire::OPCODE_SET_STENCIL_STORE_ACTION,
                StoreActionTarget::Stencil,
            ),
        ] {
            assert_eq!(
                lift(&record(opcode, &2u64.to_le_bytes())),
                Ok(RenderRecord::SetStoreAction(SetStoreAction {
                    target,
                    action: 2
                }))
            );
        }
    }

    /// Float state is carried as bits. A NaN blend colour has to stay equal to
    /// itself, which `f32` equality would not give.
    #[test]
    fn float_state_is_carried_as_bits_so_a_nan_compares_equal_to_itself() {
        let nan = f32::NAN.to_bits();
        let payload: Vec<u8> = [nan, nan, nan, nan]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = record(wire::OPCODE_SET_BLEND_COLOR, &payload);
        assert_eq!(lift(&bytes), lift(&bytes));
        assert_eq!(
            lift(&bytes),
            Ok(RenderRecord::SetBlendColor(SetBlendColor {
                red_bits: nan,
                green_bits: nan,
                blue_bits: nan,
                alpha_bits: nan,
            }))
        );
    }

    /// Every bind record's stage comes from its opcode, and the two stages stay
    /// apart. A record read at the wrong stage writes another table.
    #[test]
    fn every_bind_records_stage_comes_from_its_opcode() {
        let mut payload = 0u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&4242u32.to_le_bytes());
        for (opcode, stage) in [
            (wire::OPCODE_SET_VERTEX_TEXTURE, ShaderStage::Vertex),
            (wire::OPCODE_SET_FRAGMENT_TEXTURE, ShaderStage::Fragment),
        ] {
            let bytes = record(opcode, &payload);
            let lifted = lift(&bytes).expect("lifted");
            assert_eq!(lifted.stage(), Some(stage));
        }
    }

    /// Every kind in the vocabulary lifts a record. Driven off `RenderKind::ALL`
    /// so a kind added without a decode arm fails here rather than at a guest's
    /// expense.
    #[test]
    fn every_render_kind_lifts_a_record() {
        // Wide enough for the pass descriptor, which is the longest body here.
        let payload = [0u8; 1024];
        for kind in RenderKind::ALL {
            let bytes = record(kind.wire_opcode(), &payload);
            let lifted = lift(&bytes);
            assert!(lifted.is_ok(), "{kind:?} did not lift: {lifted:?}");
        }
    }

    /// Draw kinds lift draws and nothing else does, which is the split `decode`
    /// makes before it reads a single field.
    #[test]
    fn exactly_the_draw_kinds_lift_draws() {
        let payload = [0u8; 1024];
        for kind in RenderKind::ALL {
            let bytes = record(kind.wire_opcode(), &payload);
            let is_draw = matches!(lift(&bytes), Ok(RenderRecord::Draw(_)));
            assert_eq!(is_draw, kind.draw_shape().is_some(), "{kind:?}");
        }
    }

    /// A plural record whose count overruns reports the count and the bytes,
    /// and the count is read where that record puts it rather than at a
    /// family-wide offset.
    #[test]
    fn a_plural_count_that_overruns_is_reported_with_its_own_count() {
        let bytes = record(wire::OPCODE_SET_VIEWPORTS, &200u32.to_le_bytes());
        assert_eq!(
            lift(&bytes),
            Err(DecodeRefusal::CountOverruns {
                rail: Rail::Render,
                opcode: wire::OPCODE_SET_VIEWPORTS,
                count: 200,
                have: 4,
            })
        );

        let mut bind = 5u32.to_le_bytes().to_vec();
        bind.extend_from_slice(&200u32.to_le_bytes());
        assert_eq!(
            lift(&record(wire::OPCODE_SET_VERTEX_TEXTURE, &bind)),
            Err(DecodeRefusal::CountOverruns {
                rail: Rail::Render,
                opcode: wire::OPCODE_SET_VERTEX_TEXTURE,
                count: 200,
                have: 8,
            })
        );
    }

    /// A row the ledger settled *as a refusal* is refused by contract, not
    /// reported as an opcode nothing claims.
    ///
    /// This rail has four: the three `set*StoreActionOptions:` forms and the
    /// pass raster sample count. None of them has a `RenderKind`, because there
    /// is no record to lift — the settlement *is* the answer. What matters is
    /// which refusal comes back: `UnknownOpcode` would say this device has
    /// never seen the selector, and `Unjudged` would put a decided question
    /// back on the work queue. Both are the failure `no_record`'s doc names,
    /// and this is the render rail's instance of it.
    ///
    /// The count is asserted too. If a fifth row is settled as a refusal and
    /// someone gives it a `RenderKind` by reflex, that record would start
    /// lifting and the refusal would disappear without a test failing anywhere.
    #[test]
    fn every_render_row_the_ledger_settled_as_a_refusal_is_refused_by_contract() {
        use crate::closure::{Closure, LEDGER};

        let mut refused = 0;
        for row in LEDGER
            .iter()
            .filter(|o| o.rail == Rail::Render)
            .filter(|o| matches!(o.closure, Closure::Refused { .. }))
        {
            let Some(opcode) = row.opcode else { continue };
            refused += 1;
            assert_eq!(
                RenderKind::of_opcode(opcode),
                None,
                "render {opcode:#x} ({}) is settled as a refusal and has a record kind, so it \
                 lifts instead of being refused",
                row.selector
            );
            let bytes = record(opcode, &[0u8; 64]);
            assert_eq!(
                lift(&bytes),
                Err(DecodeRefusal::RefusedByContract {
                    rail: Rail::Render,
                    opcode
                }),
                "render {opcode:#x} ({}) must be refused by contract",
                row.selector
            );
        }
        assert_eq!(
            refused, 4,
            "the render rail settles four rows as refusals; a change to that number is a \
             contract change and not a test to update"
        );
    }
}
