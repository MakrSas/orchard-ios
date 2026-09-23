//! Which compute-encoder record an opcode names.
//!
//! # Seventeen kinds over four groups
//!
//! Four dispatches, ten pieces of encoder state, two offset rebinds, and the
//! pass descriptor. The grouping is not cosmetic: a dispatch is the only kind
//! that consumes the encoder's accumulated state, and everything else is a
//! change to that state — so the model's dispatch has to know what was bound
//! and the binds have to know nothing about the dispatch.
//!
//! # What is not here
//!
//! The fence pair, the two barriers, the compressed-reinterpretation flush, the
//! two ICB executions and the seven GPU control-flow records. The first three
//! belong to other classes; the rest are unresolved. `encodeStartIf:` and its
//! six relatives are the largest unresolved family on this rail, and a payload
//! for them would have to state what a predicate does to the records that
//! follow — which is exactly the thing that is not established.

use reims_vgpu_wire::ops::compute as wire;

/// The compute-encoder record an opcode names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputeKind {
    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    DispatchThreadgroups,
    /// `dispatchThreads:threadsPerThreadgroup:` — the same record at a
    /// different opcode, and the opcode is the entire difference: the first
    /// count is threadgroups in one and threads in the other.
    DispatchThreads,
    /// `dispatchThreadgroupsWithIndirectBuffer:…threadsPerThreadgroup:`.
    DispatchThreadgroupsIndirect,
    /// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:`.
    DispatchThreadsIndirect,
    /// `setBuffer:offset:atIndex:` and its plural and `setBytes:` forms.
    SetBuffers,
    /// `setBuffer:offset:attributeStride:atIndex:` and its plural form.
    SetBuffersWithStride,
    /// `setBufferOffset:atIndex:`.
    SetBufferOffset,
    /// `setBufferOffset:attributeStride:atIndex:`.
    SetBufferOffsetStride,
    /// `setTexture:atIndex:` and its plural form.
    SetTextures,
    /// `setSamplerState:atIndex:` and its plural form.
    SetSamplers,
    /// `setSamplerState:lodMinClamp:lodMaxClamp:atIndex:` and its plural form.
    SetSamplersWithLod,
    /// `setComputePipelineState:`.
    SetPipelineState,
    /// `setStageInRegion:`.
    SetStageInRegion,
    /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`.
    SetStageInRegionIndirect,
    /// `setThreadgroupMemoryLength:atIndex:`.
    SetThreadgroupMemoryLength,
    /// `setImageblockWidth:height:`.
    SetImageblockSize,
    /// `writeDescriptor`, which is where the pass's dispatch type reaches the
    /// wire — `setCurrentDispatchType:` writes nothing at all.
    WriteDescriptor,
}

impl ComputeKind {
    pub const ALL: &'static [ComputeKind] = &[
        ComputeKind::DispatchThreadgroups,
        ComputeKind::DispatchThreads,
        ComputeKind::DispatchThreadgroupsIndirect,
        ComputeKind::DispatchThreadsIndirect,
        ComputeKind::SetBuffers,
        ComputeKind::SetBuffersWithStride,
        ComputeKind::SetBufferOffset,
        ComputeKind::SetBufferOffsetStride,
        ComputeKind::SetTextures,
        ComputeKind::SetSamplers,
        ComputeKind::SetSamplersWithLod,
        ComputeKind::SetPipelineState,
        ComputeKind::SetStageInRegion,
        ComputeKind::SetStageInRegionIndirect,
        ComputeKind::SetThreadgroupMemoryLength,
        ComputeKind::SetImageblockSize,
        ComputeKind::WriteDescriptor,
    ];

    #[must_use]
    pub const fn wire_opcode(self) -> u32 {
        match self {
            Self::DispatchThreadgroups => wire::OPCODE_DISPATCH_THREADGROUPS,
            Self::DispatchThreads => wire::OPCODE_DISPATCH_THREADS,
            Self::DispatchThreadgroupsIndirect => wire::OPCODE_DISPATCH_THREADGROUPS_INDIRECT,
            Self::DispatchThreadsIndirect => wire::OPCODE_DISPATCH_THREADS_INDIRECT,
            Self::SetBuffers => wire::OPCODE_SET_BUFFER,
            Self::SetBuffersWithStride => wire::OPCODE_SET_BUFFER_STRIDE,
            Self::SetBufferOffset => wire::OPCODE_SET_BUFFER_OFFSET,
            Self::SetBufferOffsetStride => wire::OPCODE_SET_BUFFER_OFFSET_STRIDE,
            Self::SetTextures => wire::OPCODE_SET_TEXTURE,
            Self::SetSamplers => wire::OPCODE_SET_SAMPLER,
            Self::SetSamplersWithLod => wire::OPCODE_SET_SAMPLER_LOD,
            Self::SetPipelineState => wire::OPCODE_SET_PIPELINE_STATE,
            Self::SetStageInRegion => wire::OPCODE_SET_STAGE_IN_REGION,
            Self::SetStageInRegionIndirect => wire::OPCODE_SET_STAGE_IN_REGION_INDIRECT,
            Self::SetThreadgroupMemoryLength => wire::OPCODE_SET_THREADGROUP_MEMORY_LENGTH,
            Self::SetImageblockSize => wire::OPCODE_SET_IMAGEBLOCK_SIZE,
            Self::WriteDescriptor => wire::OPCODE_WRITE_DESCRIPTOR,
        }
    }

    #[must_use]
    pub fn of_opcode(opcode: u32) -> Option<ComputeKind> {
        ComputeKind::ALL
            .iter()
            .copied()
            .find(|k| k.wire_opcode() == opcode)
    }

    /// Whether this record consumes the encoder's accumulated state.
    #[must_use]
    pub const fn is_dispatch(self) -> bool {
        matches!(
            self,
            Self::DispatchThreadgroups
                | Self::DispatchThreads
                | Self::DispatchThreadgroupsIndirect
                | Self::DispatchThreadsIndirect
        )
    }

    /// Whether the grid comes from a buffer rather than from the record.
    #[must_use]
    pub const fn reads_grid_from_a_buffer(self) -> bool {
        matches!(
            self,
            Self::DispatchThreadgroupsIndirect | Self::DispatchThreadsIndirect
        )
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DispatchThreadgroups => "dispatch_threadgroups",
            Self::DispatchThreads => "dispatch_threads",
            Self::DispatchThreadgroupsIndirect => "dispatch_threadgroups_indirect",
            Self::DispatchThreadsIndirect => "dispatch_threads_indirect",
            Self::SetBuffers => "set_buffers",
            Self::SetBuffersWithStride => "set_buffers_with_stride",
            Self::SetBufferOffset => "set_buffer_offset",
            Self::SetBufferOffsetStride => "set_buffer_offset_stride",
            Self::SetTextures => "set_textures",
            Self::SetSamplers => "set_samplers",
            Self::SetSamplersWithLod => "set_samplers_with_lod",
            Self::SetPipelineState => "set_pipeline_state",
            Self::SetStageInRegion => "set_stage_in_region",
            Self::SetStageInRegionIndirect => "set_stage_in_region_indirect",
            Self::SetThreadgroupMemoryLength => "set_threadgroup_memory_length",
            Self::SetImageblockSize => "set_imageblock_size",
            Self::WriteDescriptor => "write_descriptor",
        }
    }
}

/// The dispatch type a compute pass runs under.
///
/// It reaches the wire only through `writeDescriptor` — `setCurrentDispatchType:`
/// emits nothing under any capability this serializer has — so a reader
/// expecting a record per call would see none and conclude the guest never set
/// one.
///
/// `Serial` is the [`Default`] because it is the encoder's *starting* type
/// rather than a choice this crate makes on the guest's behalf: a pass that
/// never writes a descriptor is serial, and the capture reads `0` there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DispatchType {
    /// Dispatches in this pass are ordered against each other.
    #[default]
    Serial,
    /// Dispatches may overlap. Permission, not a requirement: a device that
    /// orders them anyway is conservative rather than wrong.
    Concurrent,
}

impl DispatchType {
    /// Parse the descriptor's word.
    ///
    /// The two values are the two the capture drove: the encoder's starting
    /// type reads `0` and `setCurrentDispatchType:1` reads `1`. Anything else
    /// is `None` — a dispatch type the model cannot name must not be executed
    /// as the one it is numerically nearest to.
    #[must_use]
    pub const fn parse(word: u32) -> Option<DispatchType> {
        match word {
            0 => Some(DispatchType::Serial),
            1 => Some(DispatchType::Concurrent),
            _ => None,
        }
    }

    /// The descriptor's word for this type.
    ///
    /// The inverse of [`Self::parse`], and the only way back to an ordinal —
    /// so a caller that has to hand this across an ABI boundary spells the
    /// value once here rather than re-deriving `0`/`1` at each edge.
    #[must_use]
    pub const fn word(self) -> u32 {
        match self {
            Self::Serial => 0,
            Self::Concurrent => 1,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Concurrent => "concurrent",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{Rail, LEDGER};

    #[test]
    fn no_two_kinds_share_an_opcode() {
        for (i, a) in ComputeKind::ALL.iter().enumerate() {
            for b in &ComputeKind::ALL[i + 1..] {
                assert_ne!(a.wire_opcode(), b.wire_opcode(), "{a:?} and {b:?}");
            }
            assert_eq!(ComputeKind::of_opcode(a.wire_opcode()), Some(*a));
        }
    }

    /// Every kind is a judged compute-rail operation.
    #[test]
    fn every_kind_is_a_judged_compute_rail_operation() {
        for kind in ComputeKind::ALL {
            let op = LEDGER
                .iter()
                .find(|o| o.rail == Rail::Compute && o.opcode == Some(kind.wire_opcode()))
                .unwrap_or_else(|| panic!("{kind:?} has no ledger row"));
            assert!(
                !op.closure.blocks_cutover(),
                "{kind:?} is {}",
                op.closure.name()
            );
        }
    }

    /// The two direct dispatches write the identical record and differ only in
    /// opcode, which is why they are two kinds rather than one with a flag: the
    /// first count means threadgroups in one and threads in the other, and
    /// nothing in the payload says which.
    #[test]
    fn the_two_direct_dispatches_are_one_layout_at_two_opcodes() {
        assert!(wire::is_dispatch(
            ComputeKind::DispatchThreadgroups.wire_opcode()
        ));
        assert!(wire::is_dispatch(
            ComputeKind::DispatchThreads.wire_opcode()
        ));
        assert_ne!(
            ComputeKind::DispatchThreadgroups.wire_opcode(),
            ComputeKind::DispatchThreads.wire_opcode()
        );
    }

    /// Exactly four kinds dispatch, and exactly two of those read their grid
    /// from a buffer — which is what makes them accesses rather than pure
    /// state.
    #[test]
    fn four_dispatch_and_two_of_them_read_a_buffer() {
        assert_eq!(
            ComputeKind::ALL.iter().filter(|k| k.is_dispatch()).count(),
            4
        );
        let indirect: Vec<_> = ComputeKind::ALL
            .iter()
            .filter(|k| k.reads_grid_from_a_buffer())
            .collect();
        assert_eq!(indirect.len(), 2);
        for kind in indirect {
            assert!(kind.is_dispatch());
        }
    }

    /// The dispatch type has two values and a third is refused rather than
    /// rounded.
    #[test]
    fn an_unnamed_dispatch_type_is_not_the_nearest_named_one() {
        assert_eq!(DispatchType::parse(0), Some(DispatchType::Serial));
        assert_eq!(DispatchType::parse(1), Some(DispatchType::Concurrent));
        for word in [2, 3, 0xffff_ffff] {
            assert_eq!(DispatchType::parse(word), None);
        }
    }

    /// The unresolved control-flow family gets no kind. It is the largest
    /// unresolved group on this rail and the one most likely to be reached for.
    #[test]
    fn the_control_flow_family_has_no_kind() {
        for opcode in [
            wire::OPCODE_START_DO_WHILE,
            wire::OPCODE_END_DO_WHILE,
            wire::OPCODE_START_WHILE,
            wire::OPCODE_END_WHILE,
            wire::OPCODE_START_IF,
            wire::OPCODE_START_ELSE,
            wire::OPCODE_END_IF,
        ] {
            assert_eq!(ComputeKind::of_opcode(opcode), None, "{opcode:#x}");
        }
    }
}
