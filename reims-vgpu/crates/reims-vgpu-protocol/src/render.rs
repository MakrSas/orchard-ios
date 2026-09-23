//! Which render-encoder record an opcode names.
//!
//! # The draws come in pairs, and the pair is an encoding
//!
//! Twelve of the fourteen draw opcodes are six selectors times a *compact* and
//! a *wide* form: `drawPrimitives:vertexStart:vertexCount:` writes `0x01` with
//! 16-bit counts, and `0x00` with 64-bit ones when they do not fit. The field
//! order is identical and the primitive type leads both.
//!
//! That is a wire encoding, not a semantic difference, so this module names
//! both — it is the layer that maps tags — and [`RenderKind::draw_shape`]
//! collapses them into the eight shapes a model actually distinguishes. The
//! model carries values at the wider width and never has to know which encoding
//! carried them; the census that counts records as the guest sent them uses the
//! kind.
//!
//! # Stage is opcode, not field
//!
//! `setVertexTexture:` and `setFragmentTexture:` are two opcodes with one
//! layout. **No wire field names the stage** — the opcode is the only thing
//! that does — so a decoder that dispatched on layout and then looked for a
//! stage would find none, and a model that carried "a texture bind" without a
//! stage would bind to whichever table it happened to reach.

use reims_vgpu_wire::ops::{render as wire, render_pass};

/// Which shader stage a bind writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

impl ShaderStage {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}

/// The eight draw shapes, with the compact/wide encoding collapsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DrawShape {
    Primitives,
    PrimitivesInstanced,
    PrimitivesInstancedBase,
    Indexed,
    IndexedInstanced,
    IndexedInstancedBase,
    PrimitivesIndirect,
    IndexedIndirect,
}

impl DrawShape {
    pub const ALL: &'static [DrawShape] = &[
        DrawShape::Primitives,
        DrawShape::PrimitivesInstanced,
        DrawShape::PrimitivesInstancedBase,
        DrawShape::Indexed,
        DrawShape::IndexedInstanced,
        DrawShape::IndexedInstancedBase,
        DrawShape::PrimitivesIndirect,
        DrawShape::IndexedIndirect,
    ];

    /// Whether the draw reads an index buffer.
    #[must_use]
    pub const fn is_indexed(self) -> bool {
        matches!(
            self,
            Self::Indexed
                | Self::IndexedInstanced
                | Self::IndexedInstancedBase
                | Self::IndexedIndirect
        )
    }

    /// Whether the counts come from a buffer rather than from the record.
    #[must_use]
    pub const fn is_indirect(self) -> bool {
        matches!(self, Self::PrimitivesIndirect | Self::IndexedIndirect)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Primitives => "primitives",
            Self::PrimitivesInstanced => "primitives_instanced",
            Self::PrimitivesInstancedBase => "primitives_instanced_base",
            Self::Indexed => "indexed",
            Self::IndexedInstanced => "indexed_instanced",
            Self::IndexedInstancedBase => "indexed_instanced_base",
            Self::PrimitivesIndirect => "primitives_indirect",
            Self::IndexedIndirect => "indexed_indirect",
        }
    }
}

/// The width of one index.
///
/// Two values, and they are the whole of `MTLIndexType`. A third ordinal is not
/// an index width this device can guess at: the two sizes differ by a factor of
/// two, so reading the wrong one either overruns the buffer or reads half the
/// indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndexType {
    Uint16,
    Uint32,
}

impl IndexType {
    /// Parse the record's ordinal.
    #[must_use]
    pub const fn parse(raw: u16) -> Option<IndexType> {
        match raw {
            0 => Some(IndexType::Uint16),
            1 => Some(IndexType::Uint32),
            _ => None,
        }
    }

    #[must_use]
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }

    /// The wire ordinal this width is spelled with.
    ///
    /// The inverse of [`IndexType::parse`], and it exists because the consumers
    /// downstream of a lifted draw still carry `MTLIndexType` as a raw `u32` —
    /// the Metal C ABI mirror declares it that way. Round-tripping through here
    /// keeps the two spellings derived from one table rather than restated at
    /// the boundary, which is where a `1` meaning `Uint32` and a `1` meaning
    /// "the second variant" would otherwise be told apart by nothing.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::Uint16 => 0,
            Self::Uint32 => 1,
        }
    }
}

/// Which attachment a store-action override names.
///
/// Derived from the opcode: three opcodes carry a store action, and only the
/// colour one carries a slot index — its two siblings name the depth and the
/// stencil attachment by being themselves. A target read from a field would
/// need a field the depth and stencil records do not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StoreActionTarget {
    Color(u32),
    Depth,
    Stencil,
}

/// The render-encoder record an opcode names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderKind {
    Draw,
    DrawWide,
    DrawInstanced,
    DrawInstancedWide,
    DrawInstancedBase,
    DrawInstancedBaseWide,
    DrawIndexed,
    DrawIndexedWide,
    DrawIndexedInstanced,
    DrawIndexedInstancedWide,
    DrawIndexedInstancedBase,
    DrawIndexedInstancedBaseWide,
    DrawIndirect,
    DrawIndexedIndirect,
    WriteDescriptor,
    SetBlendColor,
    SetColorStoreAction,
    SetDepthStencilState,
    SetDepthStoreAction,
    SetCullMode,
    SetDepthBias,
    SetDepthClipMode,
    SetLineWidth,
    SetFragmentBuffers,
    SetFragmentBufferOffset,
    SetFragmentSamplers,
    SetFragmentSamplersWithLod,
    SetFragmentTextures,
    SetFrontFacingWinding,
    SetRenderPipelineState,
    SetScissorRect,
    SetScissorRects,
    SetStencilReference,
    SetStencilStoreAction,
    SetTriangleFillMode,
    SetVertexBuffers,
    SetVertexBufferOffset,
    SetVertexSamplers,
    SetVertexSamplersWithLod,
    SetVertexTextures,
    SetViewport,
    SetViewports,
    SetVisibilityResultMode,
    SetVertexBuffersWithStride,
    SetVertexBufferOffsetStride,
}

impl RenderKind {
    pub const ALL: &'static [RenderKind] = &[
        RenderKind::Draw,
        RenderKind::DrawWide,
        RenderKind::DrawInstanced,
        RenderKind::DrawInstancedWide,
        RenderKind::DrawInstancedBase,
        RenderKind::DrawInstancedBaseWide,
        RenderKind::DrawIndexed,
        RenderKind::DrawIndexedWide,
        RenderKind::DrawIndexedInstanced,
        RenderKind::DrawIndexedInstancedWide,
        RenderKind::DrawIndexedInstancedBase,
        RenderKind::DrawIndexedInstancedBaseWide,
        RenderKind::DrawIndirect,
        RenderKind::DrawIndexedIndirect,
        RenderKind::WriteDescriptor,
        RenderKind::SetBlendColor,
        RenderKind::SetColorStoreAction,
        RenderKind::SetDepthStencilState,
        RenderKind::SetDepthStoreAction,
        RenderKind::SetCullMode,
        RenderKind::SetDepthBias,
        RenderKind::SetDepthClipMode,
        RenderKind::SetLineWidth,
        RenderKind::SetFragmentBuffers,
        RenderKind::SetFragmentBufferOffset,
        RenderKind::SetFragmentSamplers,
        RenderKind::SetFragmentSamplersWithLod,
        RenderKind::SetFragmentTextures,
        RenderKind::SetFrontFacingWinding,
        RenderKind::SetRenderPipelineState,
        RenderKind::SetScissorRect,
        RenderKind::SetScissorRects,
        RenderKind::SetStencilReference,
        RenderKind::SetStencilStoreAction,
        RenderKind::SetTriangleFillMode,
        RenderKind::SetVertexBuffers,
        RenderKind::SetVertexBufferOffset,
        RenderKind::SetVertexSamplers,
        RenderKind::SetVertexSamplersWithLod,
        RenderKind::SetVertexTextures,
        RenderKind::SetViewport,
        RenderKind::SetViewports,
        RenderKind::SetVisibilityResultMode,
        RenderKind::SetVertexBuffersWithStride,
        RenderKind::SetVertexBufferOffsetStride,
    ];

    #[must_use]
    pub const fn wire_opcode(self) -> u32 {
        match self {
            Self::Draw => wire::OPCODE_DRAW,
            Self::DrawWide => wire::OPCODE_DRAW_WIDE,
            Self::DrawInstanced => wire::OPCODE_DRAW_INSTANCED,
            Self::DrawInstancedWide => wire::OPCODE_DRAW_INSTANCED_WIDE,
            Self::DrawInstancedBase => wire::OPCODE_DRAW_INSTANCED_BASE,
            Self::DrawInstancedBaseWide => wire::OPCODE_DRAW_INSTANCED_BASE_WIDE,
            Self::DrawIndexed => wire::OPCODE_DRAW_INDEXED,
            Self::DrawIndexedWide => wire::OPCODE_DRAW_INDEXED_WIDE,
            Self::DrawIndexedInstanced => wire::OPCODE_DRAW_INDEXED_INSTANCED,
            Self::DrawIndexedInstancedWide => wire::OPCODE_DRAW_INDEXED_INSTANCED_WIDE,
            Self::DrawIndexedInstancedBase => wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE,
            Self::DrawIndexedInstancedBaseWide => wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE,
            Self::DrawIndirect => wire::OPCODE_DRAW_INDIRECT,
            Self::DrawIndexedIndirect => wire::OPCODE_DRAW_INDEXED_INDIRECT,
            Self::WriteDescriptor => render_pass::OPCODE_RENDER_PASS,
            Self::SetBlendColor => wire::OPCODE_SET_BLEND_COLOR,
            Self::SetColorStoreAction => wire::OPCODE_SET_COLOR_STORE_ACTION,
            Self::SetDepthStencilState => wire::OPCODE_SET_DEPTH_STENCIL_STATE,
            Self::SetDepthStoreAction => wire::OPCODE_SET_DEPTH_STORE_ACTION,
            Self::SetCullMode => wire::OPCODE_SET_CULL_MODE,
            Self::SetDepthBias => wire::OPCODE_SET_DEPTH_BIAS,
            Self::SetDepthClipMode => wire::OPCODE_SET_DEPTH_CLIP_MODE,
            Self::SetLineWidth => wire::OPCODE_SET_LINE_WIDTH,
            Self::SetFragmentBuffers => wire::OPCODE_SET_FRAGMENT_BUFFER,
            Self::SetFragmentBufferOffset => wire::OPCODE_SET_FRAGMENT_BUFFER_OFFSET,
            Self::SetFragmentSamplers => wire::OPCODE_SET_FRAGMENT_SAMPLER,
            Self::SetFragmentSamplersWithLod => wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD,
            Self::SetFragmentTextures => wire::OPCODE_SET_FRAGMENT_TEXTURE,
            Self::SetFrontFacingWinding => wire::OPCODE_SET_FRONT_FACING,
            Self::SetRenderPipelineState => wire::OPCODE_SET_RENDER_PIPELINE_STATE,
            Self::SetScissorRect => wire::OPCODE_SET_SCISSOR,
            Self::SetScissorRects => wire::OPCODE_SET_SCISSOR_RECTS,
            Self::SetStencilReference => wire::OPCODE_SET_STENCIL_REFERENCE,
            Self::SetStencilStoreAction => wire::OPCODE_SET_STENCIL_STORE_ACTION,
            Self::SetTriangleFillMode => wire::OPCODE_SET_TRIANGLE_FILL_MODE,
            Self::SetVertexBuffers => wire::OPCODE_SET_VERTEX_BUFFER,
            Self::SetVertexBufferOffset => wire::OPCODE_SET_VERTEX_BUFFER_OFFSET,
            Self::SetVertexSamplers => wire::OPCODE_SET_VERTEX_SAMPLER,
            Self::SetVertexSamplersWithLod => wire::OPCODE_SET_VERTEX_SAMPLER_LOD,
            Self::SetVertexTextures => wire::OPCODE_SET_VERTEX_TEXTURE,
            Self::SetViewport => wire::OPCODE_SET_VIEWPORT,
            Self::SetViewports => wire::OPCODE_SET_VIEWPORTS,
            Self::SetVisibilityResultMode => wire::OPCODE_SET_VISIBILITY_RESULT_MODE,
            Self::SetVertexBuffersWithStride => wire::OPCODE_SET_VERTEX_BUFFER_STRIDE,
            Self::SetVertexBufferOffsetStride => wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
        }
    }

    #[must_use]
    pub fn of_opcode(opcode: u32) -> Option<RenderKind> {
        RenderKind::ALL
            .iter()
            .copied()
            .find(|k| k.wire_opcode() == opcode)
    }

    /// The draw this record is, with the compact/wide encoding collapsed.
    #[must_use]
    pub const fn draw_shape(self) -> Option<DrawShape> {
        Some(match self {
            Self::Draw | Self::DrawWide => DrawShape::Primitives,
            Self::DrawInstanced | Self::DrawInstancedWide => DrawShape::PrimitivesInstanced,
            Self::DrawInstancedBase | Self::DrawInstancedBaseWide => {
                DrawShape::PrimitivesInstancedBase
            }
            Self::DrawIndexed | Self::DrawIndexedWide => DrawShape::Indexed,
            Self::DrawIndexedInstanced | Self::DrawIndexedInstancedWide => {
                DrawShape::IndexedInstanced
            }
            Self::DrawIndexedInstancedBase | Self::DrawIndexedInstancedBaseWide => {
                DrawShape::IndexedInstancedBase
            }
            Self::DrawIndirect => DrawShape::PrimitivesIndirect,
            Self::DrawIndexedIndirect => DrawShape::IndexedIndirect,
            _ => return None,
        })
    }

    /// Whether this opcode is the 64-bit-count encoding of its shape.
    #[must_use]
    pub const fn is_wide_encoding(self) -> bool {
        matches!(
            self,
            Self::DrawWide
                | Self::DrawInstancedWide
                | Self::DrawInstancedBaseWide
                | Self::DrawIndexedWide
                | Self::DrawIndexedInstancedWide
                | Self::DrawIndexedInstancedBaseWide
        )
    }

    /// The shader stage a bind record writes, if it is a bind.
    #[must_use]
    pub const fn bind_stage(self) -> Option<ShaderStage> {
        Some(match self {
            Self::SetVertexBuffers
            | Self::SetVertexBufferOffset
            | Self::SetVertexSamplers
            | Self::SetVertexSamplersWithLod
            | Self::SetVertexTextures
            | Self::SetVertexBuffersWithStride
            | Self::SetVertexBufferOffsetStride => ShaderStage::Vertex,
            Self::SetFragmentBuffers
            | Self::SetFragmentBufferOffset
            | Self::SetFragmentSamplers
            | Self::SetFragmentSamplersWithLod
            | Self::SetFragmentTextures => ShaderStage::Fragment,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{Rail, LEDGER};

    #[test]
    fn no_two_kinds_share_an_opcode() {
        for (i, a) in RenderKind::ALL.iter().enumerate() {
            for b in &RenderKind::ALL[i + 1..] {
                assert_ne!(a.wire_opcode(), b.wire_opcode(), "{a:?} and {b:?}");
            }
            assert_eq!(RenderKind::of_opcode(a.wire_opcode()), Some(*a));
        }
    }

    #[test]
    fn every_kind_is_a_judged_render_rail_operation() {
        for kind in RenderKind::ALL {
            let op = LEDGER
                .iter()
                .find(|o| o.rail == Rail::Render && o.opcode == Some(kind.wire_opcode()))
                .unwrap_or_else(|| panic!("{kind:?} has no ledger row"));
            assert!(
                !op.closure.blocks_cutover(),
                "{kind:?} is {}",
                op.closure.name()
            );
        }
    }

    /// Six shapes have two encodings each and two have one, which is fourteen
    /// draw opcodes over eight shapes. Every shape is reachable.
    #[test]
    fn the_draw_shapes_are_covered_and_the_pairing_is_six_plus_two() {
        let mut paired = 0;
        for shape in DrawShape::ALL {
            let kinds: Vec<_> = RenderKind::ALL
                .iter()
                .filter(|k| k.draw_shape() == Some(*shape))
                .collect();
            assert!(!kinds.is_empty(), "{shape:?} is unreachable");
            match kinds.len() {
                2 => {
                    paired += 1;
                    assert_eq!(kinds.iter().filter(|k| k.is_wide_encoding()).count(), 1);
                }
                1 => assert!(shape.is_indirect(), "{shape:?} has one encoding"),
                n => panic!("{shape:?} has {n} encodings"),
            }
        }
        assert_eq!(paired, 6);
        assert_eq!(
            RenderKind::ALL
                .iter()
                .filter(|k| k.draw_shape().is_some())
                .count(),
            14
        );
    }

    /// An indirect draw has no compact/wide pair, because its counts are not in
    /// the record at all.
    #[test]
    fn an_indirect_draw_has_no_encoding_pair() {
        for kind in [RenderKind::DrawIndirect, RenderKind::DrawIndexedIndirect] {
            assert!(!kind.is_wide_encoding());
            assert!(kind.draw_shape().expect("a draw").is_indirect());
        }
    }

    /// Every bind names a stage, and nothing else does. No wire field carries
    /// it, so this map is the only thing that does.
    #[test]
    fn exactly_the_bind_records_name_a_stage() {
        let staged: Vec<_> = RenderKind::ALL
            .iter()
            .filter(|k| k.bind_stage().is_some())
            .collect();
        assert_eq!(
            staged.len(),
            12,
            "five fragment forms and seven vertex ones"
        );
        for kind in staged {
            assert_eq!(kind.draw_shape(), None);
        }
        assert_eq!(RenderKind::SetViewport.bind_stage(), None);
        assert_eq!(
            RenderKind::SetVertexTextures.bind_stage(),
            Some(ShaderStage::Vertex)
        );
        assert_eq!(
            RenderKind::SetFragmentTextures.bind_stage(),
            Some(ShaderStage::Fragment)
        );
    }

    /// The stride binds exist on the vertex stage only, which is a property of
    /// the API rather than of this table: there is no fragment attribute
    /// stride.
    #[test]
    fn the_attribute_stride_binds_are_vertex_only() {
        for kind in [
            RenderKind::SetVertexBuffersWithStride,
            RenderKind::SetVertexBufferOffsetStride,
        ] {
            assert_eq!(kind.bind_stage(), Some(ShaderStage::Vertex));
        }
        assert!(!RenderKind::ALL.iter().any(|k| {
            k.bind_stage() == Some(ShaderStage::Fragment)
                && matches!(
                    k,
                    RenderKind::SetVertexBuffersWithStride
                        | RenderKind::SetVertexBufferOffsetStride
                )
        }));
    }
}
