//! The render-encoder payloads: draws, binds, and the fixed pipeline state.
//!
//! # A draw's footprint is three things from three places
//!
//! What a draw writes comes from [`crate::pass`] — the attachments, declared
//! before the first draw. What it reads through bindings comes from the
//! encoder's tables. What it reads *from its own record* is what this module
//! derives: an index buffer, an indirect argument block, or nothing.
//!
//! Keeping those apart is the point. A model that folded them together would
//! have to re-read the pass descriptor at every draw, and would have nowhere to
//! put the one extent that is exact.
//!
//! # The index buffer's extent is exact, and it is arithmetic the model owns
//!
//! An indexed draw reads `index_count` indices of `index_type` width starting
//! at `index_buffer_offset`. Both operands are in the record, so the byte range
//! is exact — two or four bytes an index, from an ordinal whose two values are
//! the whole of `MTLIndexType`.
//!
//! An *indirect* indexed draw is the case that cannot be exact: its index count
//! lives in the argument buffer, so the read widens to the whole index buffer.
//! Borrowing the direct form's arithmetic there would produce a range computed
//! from a count the record does not carry.
//!
//! # Values are carried at the wider width, and the encoding is not
//!
//! Six of the eight draw shapes arrive in a compact and a wide form, differing
//! only in whether the counts are 16 or 64 bits. That is an encoding, so the
//! payload carries `u64` and does not remember which form arrived;
//! [`reims_vgpu_protocol::render::RenderKind`] is where a census that cares
//! asks. `base_vertex` is the exception the wire forces: it is signed, and the
//! two encodings **disagree about what survives**. The compact form truncates
//! it to 16 bits — Apple's serializer does that, upstream of this device, so
//! `-70000` arrives as `0xee90` and the value is already lost. The wide form
//! sign-extends it to the full width and `-70000` arrives intact. So the
//! payload carries `i64`: it is the width at which both forms are
//! representable, and a model that carried the narrow one would take a loss
//! the wire did not.

use crate::access::{ByteRange, Participation, Participations};
use crate::bind::{BindSpan, IndirectSource};
use crate::identity::ResourceId;
pub use reims_vgpu_protocol::render::{
    DrawShape, IndexType, RenderKind, ShaderStage, StoreActionTarget,
};

/// The `MTLPrimitiveType` ordinal, carried verbatim.
///
/// Not parsed here. Which topologies a host supports is a capability question
/// and the executor answers it; folding an unknown ordinal onto a known one
/// would draw the wrong primitive rather than refuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrimitiveType(pub u32);

/// The index buffer an indexed draw reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexSource {
    pub buffer: ResourceId,
    pub offset: u64,
    pub index_type: IndexType,
}

impl IndexSource {
    /// The exact byte range `count` indices occupy.
    ///
    /// `None` on overflow, which widens rather than wrapping: a wrapped range
    /// is shorter than the read and a short range is a hazard edge that does
    /// not cover it.
    #[must_use]
    pub const fn range(self, count: u64) -> Option<ByteRange> {
        let Some(length) = count.checked_mul(self.index_type.bytes()) else {
            return None;
        };
        Some(ByteRange {
            offset: self.offset,
            length,
        })
    }
}

/// The bytes `MTLDrawPrimitivesIndirectArguments` occupies: four `uint32`.
pub const DRAW_INDIRECT_ARGS_BYTES: u64 = 16;
/// The bytes `MTLDrawIndexedPrimitivesIndirectArguments` occupies: five
/// `uint32`.
pub const DRAW_INDEXED_INDIRECT_ARGS_BYTES: u64 = 20;

/// One draw.
///
/// The instanced and base-instanced forms are fields rather than variants: they
/// are the same draw with more of the same counts, and a caller that wanted
/// "how many instances" would otherwise have to match three variants to find
/// out. The indexed and indirect distinctions *are* variants, because they
/// change which memory is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawOp {
    /// A direct, non-indexed draw.
    Primitives {
        primitive: PrimitiveType,
        vertex_start: u64,
        vertex_count: u64,
        instances: Instancing,
    },
    /// A direct, indexed draw.
    Indexed {
        primitive: PrimitiveType,
        index: IndexSource,
        index_count: u64,
        instances: Instancing,
        /// Signed, and carried at the width the wide encoding preserves.
        ///
        /// Not `i16`. The compact encoding's value is truncated to 16 bits by
        /// Apple's serializer before this device sees it, but the wide one is
        /// sign-extended and carries the guest's whole value — so narrowing
        /// here would discard, on the encoding that kept it, exactly what the
        /// other encoding lost.
        base_vertex: i64,
    },
    /// A non-indexed draw whose counts come from a buffer.
    PrimitivesIndirect {
        primitive: PrimitiveType,
        arguments: IndirectSource,
    },
    /// An indexed draw whose counts come from a buffer.
    IndexedIndirect {
        primitive: PrimitiveType,
        index: IndexSource,
        arguments: IndirectSource,
    },
}

/// How many instances a draw runs, and where they start.
///
/// `None` for both is the plain `drawPrimitives:` form. `Some` instance count
/// with a zero base is the `instanceCount:` form, and a non-zero base is the
/// `baseInstance:` one — but the model does not need to tell those apart, so it
/// does not: one instance starting at zero is one instance starting at zero
/// however the guest spelled it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instancing {
    pub count: Option<u64>,
    pub base: u64,
}

impl DrawOp {
    /// The shape this draw is.
    #[must_use]
    pub const fn shape(&self) -> DrawShape {
        match self {
            Self::Primitives { instances, .. } => match instances.count {
                None => DrawShape::Primitives,
                Some(_) if instances.base == 0 => DrawShape::PrimitivesInstanced,
                Some(_) => DrawShape::PrimitivesInstancedBase,
            },
            Self::Indexed { instances, .. } => match instances.count {
                None => DrawShape::Indexed,
                Some(_) if instances.base == 0 => DrawShape::IndexedInstanced,
                Some(_) => DrawShape::IndexedInstancedBase,
            },
            Self::PrimitivesIndirect { .. } => DrawShape::PrimitivesIndirect,
            Self::IndexedIndirect { .. } => DrawShape::IndexedIndirect,
        }
    }

    /// The index buffer this draw reads, and the exact range when the record
    /// carries the count.
    ///
    /// `Some((source, None))` is a real read with an unestablished extent —
    /// the indirect form, whose count is in the argument buffer. The caller
    /// widens to the whole buffer; it must not treat the absent range as an
    /// absent read.
    #[must_use]
    pub const fn index_read(&self) -> Option<(IndexSource, Option<ByteRange>)> {
        match *self {
            Self::Primitives { .. } | Self::PrimitivesIndirect { .. } => None,
            Self::Indexed {
                index, index_count, ..
            } => Some((index, index.range(index_count))),
            Self::IndexedIndirect { index, .. } => Some((index, None)),
        }
    }

    /// The argument buffer this draw reads its counts from, and how many bytes
    /// the block occupies.
    ///
    /// Exact in both cases: `MTLDrawPrimitivesIndirectArguments` and
    /// `MTLDrawIndexedPrimitivesIndirectArguments` are public structures, four
    /// and five `uint32`.
    #[must_use]
    pub const fn indirect_read(&self) -> Option<(IndirectSource, u64)> {
        match *self {
            Self::Primitives { .. } | Self::Indexed { .. } => None,
            Self::PrimitivesIndirect { arguments, .. } => {
                Some((arguments, DRAW_INDIRECT_ARGS_BYTES))
            }
            Self::IndexedIndirect { arguments, .. } => {
                Some((arguments, DRAW_INDEXED_INDIRECT_ARGS_BYTES))
            }
        }
    }
}

/// A scissor rectangle, in the guest's units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScissorRect {
    pub x: u64,
    pub y: u64,
    pub width: u64,
    pub height: u64,
}

/// A viewport, with its depth bounds as the guest's bits.
///
/// Six doubles, kept as bit patterns for the reason
/// [`crate::bind::LodClamp`] gives: a state table has to compare, and float
/// equality makes a NaN bound differ from itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Viewport {
    pub origin_x_bits: u64,
    pub origin_y_bits: u64,
    pub width_bits: u64,
    pub height_bits: u64,
    pub z_near_bits: u64,
    pub z_far_bits: u64,
}

/// A `float` piece of fixed-function state, as the guest's bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FloatBits(pub u32);

impl FloatBits {
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }

    #[must_use]
    pub const fn from_f32(value: f32) -> FloatBits {
        FloatBits(value.to_bits())
    }
}

/// A window of the transaction's viewport or scissor arena.
pub type StateSpan = BindSpan;

/// One render-encoder record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderOp {
    Draw(DrawOp),

    /// Consecutive buffer slots from `first`, on one stage.
    BindBuffers {
        stage: ShaderStage,
        first: u32,
        entries: BindSpan,
    },
    /// The vertex stage's attribute-stride buffer bind. There is no fragment
    /// form: the API has no fragment attribute stride.
    BindBuffersWithStride {
        first: u32,
        entries: BindSpan,
    },
    BindTextures {
        stage: ShaderStage,
        first: u32,
        entries: BindSpan,
    },
    BindSamplers {
        stage: ShaderStage,
        first: u32,
        entries: BindSpan,
    },
    BindSamplersWithLod {
        stage: ShaderStage,
        first: u32,
        entries: BindSpan,
    },
    /// Move an already-bound buffer's offset, and optionally its stride.
    RebindBufferOffset {
        stage: ShaderStage,
        index: u32,
        offset: u64,
        stride: Option<u64>,
    },

    SetPipeline {
        pipeline: ResourceId,
    },
    SetDepthStencilState {
        state: ResourceId,
    },
    /// The pass descriptor. Its content is [`crate::pass::PassDescriptor`],
    /// held by the encoder rather than inline: it is 592 bytes on the wire and
    /// a record that carried it by value would make every other variant that
    /// size.
    WriteDescriptor {
        descriptor: PassDescriptorSlot,
    },

    SetViewports(StateSpan),
    SetScissorRects(StateSpan),
    SetCullMode(u64),
    SetFrontFacingWinding(u64),
    SetDepthClipMode(u64),
    SetTriangleFillMode(u64),
    SetDepthBias {
        bias: FloatBits,
        slope_scale: FloatBits,
        clamp: FloatBits,
    },
    /// `setLineWidth:`. One float, and unlike the four ordinals above it is
    /// not a pipeline decision on any host: `VK_DYNAMIC_STATE_LINE_WIDTH` is
    /// Vulkan 1.0 core and Metal sets it on the encoder, so it is per-draw
    /// state everywhere. Whether a draw uses it at all is a joint fact of the
    /// fill mode and the topology, which only the executor holds.
    SetLineWidth(FloatBits),
    SetBlendColor {
        red: FloatBits,
        green: FloatBits,
        blue: FloatBits,
        alpha: FloatBits,
    },
    SetStencilReference {
        front: u32,
        back: u32,
    },
    /// A store-action override on one attachment.
    ///
    /// The ordinal is carried at `u64` because that is the width the wire uses
    /// for the depth and stencil forms; the colour form is 32 bits. Neither is
    /// narrowed and neither is parsed, for the reason [`PrimitiveType`] gives:
    /// which store actions a host can perform is the executor's question, and
    /// folding an unknown ordinal onto a known one stores the wrong thing
    /// rather than refusing.
    SetStoreAction {
        target: StoreActionTarget,
        action: u64,
    },
    /// The occlusion-query mode, and the offset into the pass's visibility
    /// buffer.
    ///
    /// It names no buffer — that is on the pass descriptor — so this record's
    /// own footprint is nothing, and the eight bytes it eventually writes are
    /// attributed when the encoder joins the two.
    SetVisibilityResultMode {
        mode: u64,
        offset: u64,
    },
}

/// A pass descriptor held in the transaction's descriptor arena.
///
/// One index, not a copy: the descriptor is 592 bytes on the wire and a variant
/// carrying it by value would make `RenderOp` that size for every record,
/// including the eight-byte ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PassDescriptorSlot(pub u32);

impl RenderOp {
    /// Whether this record draws.
    #[must_use]
    pub const fn is_draw(&self) -> bool {
        matches!(self, Self::Draw(_))
    }

    /// The memory this record names by itself.
    ///
    /// Empty for every bind and every piece of state, for the reason
    /// [`crate::compute`] gives at length: a bind writes a slot and touches no
    /// memory, and what a draw reads through those slots belongs to the
    /// encoder. What is here is only what the record's own fields name.
    ///
    /// A pass descriptor's own footprint is **not** here either, and that is
    /// not an omission: [`Self::WriteDescriptor`] carries a
    /// [`PassDescriptorSlot`], so the descriptor is in the transaction's arena
    /// and only [`crate::exec::ResolvedOperation::participations`] — which
    /// holds the arena — can reach it.
    #[must_use]
    pub fn participations(&self) -> Participations {
        let Self::Draw(draw) = self else {
            return Participations::NONE;
        };
        Participations::pair(
            draw.index_read().map(|(source, range)| {
                Participation::buffer_read(source.buffer, source.offset, range.map(|r| r.length))
            }),
            draw.indirect_read().map(|(source, bytes)| {
                Participation::buffer_read(source.buffer, source.offset, Some(bytes))
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessMode, ParticipationExtent};
    use crate::identity::{ObjectListRef, SlotGeneration};
    use crate::operation::{classify, OperationClass, OperationHome};
    use reims_vgpu_protocol::closure::{Rail, LEDGER};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn index(index_type: IndexType) -> IndexSource {
        IndexSource {
            buffer: res(5),
            offset: 0x100,
            index_type,
        }
    }

    fn arguments() -> IndirectSource {
        IndirectSource {
            buffer: res(6),
            offset: 0x200,
        }
    }

    fn is_refused(op: &reims_vgpu_protocol::closure::Op) -> bool {
        matches!(
            op.closure,
            reims_vgpu_protocol::closure::Closure::Refused { .. }
        )
    }

    /// The payload vocabulary is exactly the render-class operations that are
    /// judged and not refused.
    ///
    /// The exclusion is the same rule `crate::sync` states: a refused operation
    /// keeps its class, because refusing by name needs the name, and gains no
    /// payload, because there is nothing to execute.
    #[test]
    fn the_render_vocabulary_is_exactly_the_judged_render_operations() {
        let mut from_ledger: Vec<u32> = LEDGER
            .iter()
            .filter(|o| {
                o.rail == Rail::Render
                    && !is_refused(o)
                    && classify(o) == Some(OperationHome::Stream(OperationClass::Render))
            })
            .map(|o| {
                o.opcode
                    .expect("a classified render record carries an opcode")
            })
            .collect();
        from_ledger.sort_unstable();

        let mut from_kinds: Vec<u32> = RenderKind::ALL.iter().map(|k| k.wire_opcode()).collect();
        from_kinds.sort_unstable();

        assert_eq!(from_ledger, from_kinds);
    }

    /// Every settled refusal on this rail keeps its class and its route and
    /// gains no payload.
    ///
    /// A refusal is a record the device declines by name, so the vocabulary
    /// must not be able to represent it: a `RenderKind` for one would let a
    /// resolver build an operation for work that is refused, and the refusal
    /// would then have to be found again somewhere downstream.
    ///
    /// The four are the pass's default raster sample count, which this rail
    /// renders at one and refuses above, and the three store-action options
    /// records, whose only declared flag asks for programmable sample positions
    /// this rail does not set.
    #[test]
    fn every_refused_render_row_gains_no_payload() {
        let refused: Vec<_> = LEDGER
            .iter()
            .filter(|o| {
                o.rail == Rail::Render
                    && is_refused(o)
                    && classify(o) == Some(OperationHome::Stream(OperationClass::Render))
            })
            .map(|o| {
                let reims_vgpu_protocol::closure::Closure::Refused { route, .. } = o.closure else {
                    unreachable!("filtered")
                };
                (
                    o.opcode.expect("a refused render record carries an opcode"),
                    route,
                )
            })
            .collect();
        assert_eq!(
            refused,
            [
                (0x1e, "render_pass_raster_sample_count_dropped"),
                (0x67, "render_store_action_options_dropped"),
                (0x6a, "render_store_action_options_dropped"),
                (0x79, "render_store_action_options_dropped"),
            ]
        );
        for (opcode, _) in refused {
            assert_eq!(
                RenderKind::of_opcode(opcode),
                None,
                "{opcode:#x} is refused and the vocabulary can represent it"
            );
        }
    }

    /// Every draw shape is reachable from the payload, and the instancing
    /// fields are what select three of them.
    #[test]
    fn every_draw_shape_is_reachable() {
        let shapes = [
            DrawOp::Primitives {
                primitive: PrimitiveType(3),
                vertex_start: 0,
                vertex_count: 3,
                instances: Instancing::default(),
            },
            DrawOp::Primitives {
                primitive: PrimitiveType(3),
                vertex_start: 0,
                vertex_count: 3,
                instances: Instancing {
                    count: Some(2),
                    base: 0,
                },
            },
            DrawOp::Primitives {
                primitive: PrimitiveType(3),
                vertex_start: 0,
                vertex_count: 3,
                instances: Instancing {
                    count: Some(2),
                    base: 1,
                },
            },
            DrawOp::Indexed {
                primitive: PrimitiveType(3),
                index: index(IndexType::Uint16),
                index_count: 6,
                instances: Instancing::default(),
                base_vertex: 0,
            },
            DrawOp::Indexed {
                primitive: PrimitiveType(3),
                index: index(IndexType::Uint16),
                index_count: 6,
                instances: Instancing {
                    count: Some(2),
                    base: 0,
                },
                base_vertex: 0,
            },
            DrawOp::Indexed {
                primitive: PrimitiveType(3),
                index: index(IndexType::Uint16),
                index_count: 6,
                instances: Instancing {
                    count: Some(2),
                    base: 4,
                },
                base_vertex: -2,
            },
            DrawOp::PrimitivesIndirect {
                primitive: PrimitiveType(3),
                arguments: arguments(),
            },
            DrawOp::IndexedIndirect {
                primitive: PrimitiveType(3),
                index: index(IndexType::Uint32),
                arguments: arguments(),
            },
        ];
        let mut seen: Vec<DrawShape> = shapes.iter().map(|d| d.shape()).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), DrawShape::ALL.len());
    }

    /// A store-action ordinal wider than sixteen bits survives. The depth and
    /// stencil records carry theirs at 64 bits, so narrowing the payload would
    /// have folded a value the wire kept — the same shape of loss as the base
    /// vertex above, on a different field.
    #[test]
    fn a_store_action_wider_than_sixteen_bits_is_carried_whole() {
        let op = RenderOp::SetStoreAction {
            target: StoreActionTarget::Depth,
            action: 0x1_0000,
        };
        let RenderOp::SetStoreAction { action, .. } = op else {
            panic!("store action");
        };
        assert_eq!(action, 0x1_0000);
        assert!(action > u64::from(u16::MAX));
    }

    /// A base vertex the compact encoding could not have held survives the
    /// model. The wide encoding sign-extends it, so `-70000` is a value a guest
    /// can actually send, and an `i16` payload would have turned it into
    /// `0xee90` — the same loss the *other* encoding takes, applied to the one
    /// that did not.
    #[test]
    fn a_base_vertex_below_the_compact_encodings_range_is_carried_whole() {
        let draw = DrawOp::Indexed {
            primitive: PrimitiveType(3),
            index: index(IndexType::Uint16),
            index_count: 6,
            instances: Instancing::default(),
            base_vertex: -70_000,
        };
        let DrawOp::Indexed { base_vertex, .. } = draw else {
            panic!("indexed");
        };
        assert_eq!(base_vertex, -70_000);
        assert!(base_vertex < i64::from(i16::MIN));
    }

    /// The index range is exact and scales with the index width.
    #[test]
    fn an_indexed_draw_reads_an_exact_range() {
        for (width, bytes) in [(IndexType::Uint16, 2u64), (IndexType::Uint32, 4)] {
            let draw = DrawOp::Indexed {
                primitive: PrimitiveType(3),
                index: index(width),
                index_count: 6,
                instances: Instancing::default(),
                base_vertex: 0,
            };
            let (source, range) = draw.index_read().expect("indexed");
            assert_eq!(source.buffer, res(5));
            assert_eq!(
                range,
                Some(ByteRange {
                    offset: 0x100,
                    length: 6 * bytes
                })
            );
        }
    }

    /// An indirect indexed draw reads the index buffer and does not pretend to
    /// know how much of it.
    #[test]
    fn an_indirect_indexed_draw_reads_without_an_extent() {
        let draw = DrawOp::IndexedIndirect {
            primitive: PrimitiveType(3),
            index: index(IndexType::Uint32),
            arguments: arguments(),
        };
        let (source, range) = draw.index_read().expect("reads the index buffer");
        assert_eq!(source.buffer, res(5));
        assert_eq!(range, None);
        let accesses = RenderOp::Draw(draw).participations();
        assert_eq!(accesses.len(), 2);
        // No index count in the record, so the claim widens to the whole
        // buffer rather than narrowing on a guessed span.
        assert_eq!(accesses[0].extent, ParticipationExtent::Whole);
        assert_eq!(accesses[0].mode, AccessMode::Read);
    }

    /// Both argument blocks are public structures, so both extents are exact
    /// and they are not the same size.
    #[test]
    fn the_two_argument_blocks_are_exact_and_different() {
        let plain = DrawOp::PrimitivesIndirect {
            primitive: PrimitiveType(3),
            arguments: arguments(),
        };
        assert_eq!(
            plain.indirect_read().expect("indirect").1,
            DRAW_INDIRECT_ARGS_BYTES
        );
        let indexed = DrawOp::IndexedIndirect {
            primitive: PrimitiveType(3),
            index: index(IndexType::Uint16),
            arguments: arguments(),
        };
        assert_eq!(
            indexed.indirect_read().expect("indirect").1,
            DRAW_INDEXED_INDIRECT_ARGS_BYTES
        );
        assert_ne!(DRAW_INDIRECT_ARGS_BYTES, DRAW_INDEXED_INDIRECT_ARGS_BYTES);
        assert_eq!(DRAW_INDIRECT_ARGS_BYTES, 16);
        assert_eq!(DRAW_INDEXED_INDIRECT_ARGS_BYTES, 20);
    }

    /// A non-indexed direct draw reads nothing from its own record.
    #[test]
    fn a_direct_draw_names_no_memory() {
        let draw = DrawOp::Primitives {
            primitive: PrimitiveType(3),
            vertex_start: 0,
            vertex_count: 3,
            instances: Instancing::default(),
        };
        assert!(RenderOp::Draw(draw).participations().is_empty());
        assert_eq!(draw.index_read(), None);
        assert_eq!(draw.indirect_read(), None);
    }

    /// A bind names no memory, on either stage.
    #[test]
    fn no_bind_or_state_record_names_memory() {
        let ops = [
            RenderOp::BindBuffers {
                stage: ShaderStage::Vertex,
                first: 0,
                entries: BindSpan { start: 0, len: 1 },
            },
            RenderOp::BindTextures {
                stage: ShaderStage::Fragment,
                first: 0,
                entries: BindSpan { start: 0, len: 1 },
            },
            RenderOp::SetPipeline { pipeline: res(1) },
            RenderOp::SetViewports(StateSpan { start: 0, len: 1 }),
            RenderOp::SetVisibilityResultMode {
                mode: 1,
                offset: 0x40,
            },
        ];
        for op in ops {
            assert!(op.participations().is_empty());
            assert!(!op.is_draw());
        }
    }

    /// An index-width ordinal outside the two `MTLIndexType` values is refused.
    /// The widths differ by a factor of two, so guessing overruns the buffer or
    /// reads half the indices.
    #[test]
    fn an_unknown_index_width_is_refused() {
        assert_eq!(IndexType::parse(0), Some(IndexType::Uint16));
        assert_eq!(IndexType::parse(1), Some(IndexType::Uint32));
        for raw in [2u16, 3, u16::MAX] {
            assert_eq!(IndexType::parse(raw), None);
        }
        assert_eq!(IndexType::Uint16.bytes() * 2, IndexType::Uint32.bytes());
    }

    /// An index range that overflows widens rather than wrapping.
    #[test]
    fn an_overflowing_index_range_has_no_extent() {
        let source = IndexSource {
            buffer: res(5),
            offset: 0,
            index_type: IndexType::Uint32,
        };
        assert_eq!(source.range(u64::MAX), None);
        assert_eq!(
            source.range(2),
            Some(ByteRange {
                offset: 0,
                length: 8
            })
        );
    }

    /// A float piece of state compares as bits, so a NaN depth bias does not
    /// dirty the state on every record.
    #[test]
    fn float_state_compares_as_bits() {
        let nan = FloatBits::from_f32(f32::NAN);
        assert_eq!(nan, nan);
        assert!(nan.to_f32().is_nan());
        assert_ne!(FloatBits::from_f32(0.0), FloatBits::from_f32(1.0));
    }
}
