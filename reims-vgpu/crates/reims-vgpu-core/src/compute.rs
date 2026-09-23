//! The compute-encoder payloads: what a bind changes, and what a dispatch
//! consumes.
//!
//! # A bind is state and a dispatch is an access, and only one of them touches
//! memory
//!
//! `setBuffer:offset:atIndex:` does not read the buffer. It writes a slot in
//! the encoder's binding table, and the memory is touched later, by whatever
//! dispatch runs with that slot still bound. That is the whole reason
//! [`ComputeOp::participations`] is empty for every bind: a model that
//! produced an access at the bind would order against memory the guest may
//! rebind before anything reads it.
//!
//! The footprint a dispatch reads *through* the table is the encoder's to
//! supply, from the slots that were live when the dispatch ran. This module is
//! the vocabulary of changes to those slots, and of the one thing a record
//! names on its own.
//!
//! # The one memory a dispatch record names itself
//!
//! An indirect dispatch reads its grid out of a buffer, and that read happens
//! on the GPU at dispatch time. It is a real access with a real extent and it
//! must be ordered — a compute pass that writes the arguments and then
//! dispatches from them is an ordinary pattern, and the write-then-read edge is
//! the only thing that makes it correct.
//!
//! The threadgroup-granular form's argument block is
//! `MTLDispatchThreadgroupsIndirectArguments`: three `uint32`, twelve bytes,
//! which is API contract. The thread-granular form is an SPI whose argument
//! layout is not established, so its extent widens to the whole buffer instead
//! of borrowing the twelve bytes from its sibling. Twelve bytes that turned out
//! to be twenty-four is a hazard edge that does not cover the read.
//!
//! # Extents are the compute encoder's own types
//!
//! A grid is not a copy size. [`crate::blit::Size3`] exists for transfer
//! geometry and this has [`ComputeExtent`], at the same width and deliberately
//! not the same type: one is threads and the other is texels, and a function
//! that accepted either would accept a grid where a copy region belongs.

use crate::access::{Participation, Participations};
pub use crate::bind::{BindSpan, BufferBinding, IndirectSource, LodClamp, ObjectBinding};
use crate::identity::ResourceId;
pub use reims_vgpu_protocol::compute::{ComputeKind, DispatchType};

/// A three-dimensional count of threads or threadgroups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeExtent {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

/// A three-dimensional origin in the compute encoder's coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeOrigin {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

/// The bytes `MTLDispatchThreadgroupsIndirectArguments` occupies.
///
/// Three `uint32`. This is the public API's own structure, which is why the
/// extent can be exact here and cannot be for the SPI form.
pub const DISPATCH_THREADGROUPS_INDIRECT_ARGS_BYTES: u64 = 12;

/// One dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchOp {
    /// A grid counted in threadgroups.
    Threadgroups {
        groups: ComputeExtent,
        threads_per_group: ComputeExtent,
    },
    /// A grid counted in threads, with non-uniform threadgroups at the edges.
    ///
    /// The identical record to [`Self::Threadgroups`] on the wire; the opcode
    /// is what says which count the first extent is. Two variants rather than a
    /// flag, because "threadgroups" and "threads" are different quantities and
    /// a caller that read the wrong one dispatches a grid the wrong size.
    Threads {
        threads: ComputeExtent,
        threads_per_group: ComputeExtent,
    },
    /// A threadgroup-granular grid read from a buffer.
    ThreadgroupsIndirect {
        source: IndirectSource,
        threads_per_group: ComputeExtent,
    },
    /// A thread-granular grid read from a buffer. Its selector carries no
    /// threadgroup size.
    ThreadsIndirect { source: IndirectSource },
}

impl DispatchOp {
    /// The buffer window this dispatch reads its grid from, if any.
    ///
    /// `Some((source, None))` means the buffer is read but the extent is not
    /// established — the caller widens to the whole resource. Returning the
    /// pair rather than an extent alone is what stops "no extent" being
    /// mistaken for "no read".
    #[must_use]
    pub const fn indirect_read(&self) -> Option<(IndirectSource, Option<u64>)> {
        match *self {
            Self::Threadgroups { .. } | Self::Threads { .. } => None,
            Self::ThreadgroupsIndirect { source, .. } => {
                Some((source, Some(DISPATCH_THREADGROUPS_INDIRECT_ARGS_BYTES)))
            }
            Self::ThreadsIndirect { source } => Some((source, None)),
        }
    }
}

/// One change to the encoder's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeOp {
    /// Consecutive buffer slots from `first`.
    BindBuffers { first: u32, entries: BindSpan },
    /// The same, from the selectors that carry an attribute stride.
    ///
    /// A variant rather than a flag on the entries: the record's opcode decides
    /// the entry shape for every entry it carries, so the shape is a property
    /// of the operation and reading it off an arena element would let one
    /// record hold both shapes.
    BindBuffersWithStride { first: u32, entries: BindSpan },
    /// Consecutive texture slots from `first`.
    BindTextures { first: u32, entries: BindSpan },
    /// Consecutive sampler slots from `first`.
    BindSamplers { first: u32, entries: BindSpan },
    /// The same, from the selectors that carry level-of-detail clamps.
    BindSamplersWithLod { first: u32, entries: BindSpan },
    /// Move an already-bound buffer's offset, and optionally its stride.
    ///
    /// It does not name a buffer: the slot keeps whatever it holds. A model
    /// that resolved a resource here would be resolving the binding table's
    /// contents at the wrong time.
    RebindBufferOffset {
        index: u32,
        offset: u64,
        stride: Option<u64>,
    },
    /// The pipeline the next dispatch runs.
    SetPipeline { pipeline: ResourceId },
    /// The stage-in region, from the record.
    SetStageInRegion {
        origin: ComputeOrigin,
        size: ComputeExtent,
    },
    /// The stage-in region, read from a buffer.
    ///
    /// Its extent is not established — no public structure names this layout —
    /// so the read widens to the whole buffer.
    SetStageInRegionIndirect { source: IndirectSource },
    /// Threadgroup memory for one slot.
    SetThreadgroupMemory { index: u32, length: u64 },
    /// The imageblock's pixel dimensions.
    SetImageblockSize { width: u32, height: u32 },
    /// The pass descriptor, which carries the dispatch type.
    WriteDescriptor { dispatch_type: DispatchType },
    /// A dispatch.
    Dispatch(DispatchOp),
}

impl ComputeOp {
    /// Which record this is, which is also which opcode carried it.
    #[must_use]
    pub const fn kind(&self) -> ComputeKind {
        match self {
            Self::BindBuffers { .. } => ComputeKind::SetBuffers,
            Self::BindBuffersWithStride { .. } => ComputeKind::SetBuffersWithStride,
            Self::BindTextures { .. } => ComputeKind::SetTextures,
            Self::BindSamplers { .. } => ComputeKind::SetSamplers,
            Self::BindSamplersWithLod { .. } => ComputeKind::SetSamplersWithLod,
            Self::RebindBufferOffset { stride: None, .. } => ComputeKind::SetBufferOffset,
            Self::RebindBufferOffset {
                stride: Some(_), ..
            } => ComputeKind::SetBufferOffsetStride,
            Self::SetPipeline { .. } => ComputeKind::SetPipelineState,
            Self::SetStageInRegion { .. } => ComputeKind::SetStageInRegion,
            Self::SetStageInRegionIndirect { .. } => ComputeKind::SetStageInRegionIndirect,
            Self::SetThreadgroupMemory { .. } => ComputeKind::SetThreadgroupMemoryLength,
            Self::SetImageblockSize { .. } => ComputeKind::SetImageblockSize,
            Self::WriteDescriptor { .. } => ComputeKind::WriteDescriptor,
            Self::Dispatch(DispatchOp::Threadgroups { .. }) => ComputeKind::DispatchThreadgroups,
            Self::Dispatch(DispatchOp::Threads { .. }) => ComputeKind::DispatchThreads,
            Self::Dispatch(DispatchOp::ThreadgroupsIndirect { .. }) => {
                ComputeKind::DispatchThreadgroupsIndirect
            }
            Self::Dispatch(DispatchOp::ThreadsIndirect { .. }) => {
                ComputeKind::DispatchThreadsIndirect
            }
        }
    }

    /// The memory this record names by itself.
    ///
    /// One entry at most: the only compute records that name memory are the
    /// two indirect dispatches and the indirect stage-in region, and each names
    /// one buffer. Everything a dispatch reads *through the binding table* is
    /// the encoder's to supply, and is deliberately not here — this answers
    /// "what did this record say", and the encoder answers "what was bound when
    /// it ran". Empty for every bind, because a bind touches no memory: that is
    /// the module's central claim and it is checked by a test over the whole
    /// vocabulary rather than trusted.
    #[must_use]
    pub const fn participations(&self) -> Participations {
        let (source, length) = match self {
            Self::Dispatch(dispatch) => match dispatch.indirect_read() {
                Some(pair) => pair,
                None => return Participations::NONE,
            },
            // No public structure names this argument layout, so the extent is
            // not established and the read is against the whole buffer.
            Self::SetStageInRegionIndirect { source } => (*source, None),
            _ => return Participations::NONE,
        };
        Participations::one(Participation::buffer_read(
            source.buffer,
            source.offset,
            length,
        ))
    }

    /// Whether this record consumes the encoder's accumulated state.
    #[must_use]
    pub const fn is_dispatch(&self) -> bool {
        matches!(self, Self::Dispatch(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessMode, ByteRange, ParticipationExtent};
    use crate::identity::{ObjectListRef, SlotGeneration};
    use crate::operation::{classify, OperationClass, OperationHome};
    use reims_vgpu_protocol::closure::{Rail, LEDGER};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn source() -> IndirectSource {
        IndirectSource {
            buffer: res(9),
            offset: 0x40,
        }
    }

    fn every_op() -> Vec<ComputeOp> {
        vec![
            ComputeOp::BindBuffers {
                first: 0,
                entries: BindSpan { start: 0, len: 2 },
            },
            ComputeOp::BindBuffersWithStride {
                first: 0,
                entries: BindSpan { start: 0, len: 2 },
            },
            ComputeOp::BindTextures {
                first: 1,
                entries: BindSpan { start: 0, len: 1 },
            },
            ComputeOp::BindSamplers {
                first: 2,
                entries: BindSpan { start: 0, len: 1 },
            },
            ComputeOp::BindSamplersWithLod {
                first: 2,
                entries: BindSpan { start: 0, len: 1 },
            },
            ComputeOp::RebindBufferOffset {
                index: 3,
                offset: 0x10,
                stride: None,
            },
            ComputeOp::RebindBufferOffset {
                index: 3,
                offset: 0x10,
                stride: Some(0x20),
            },
            ComputeOp::SetPipeline { pipeline: res(1) },
            ComputeOp::SetStageInRegion {
                origin: ComputeOrigin::default(),
                size: ComputeExtent::default(),
            },
            ComputeOp::SetStageInRegionIndirect { source: source() },
            ComputeOp::SetThreadgroupMemory {
                index: 0,
                length: 0x100,
            },
            ComputeOp::SetImageblockSize {
                width: 8,
                height: 8,
            },
            ComputeOp::WriteDescriptor {
                dispatch_type: DispatchType::Serial,
            },
            ComputeOp::Dispatch(DispatchOp::Threadgroups {
                groups: ComputeExtent::default(),
                threads_per_group: ComputeExtent::default(),
            }),
            ComputeOp::Dispatch(DispatchOp::Threads {
                threads: ComputeExtent::default(),
                threads_per_group: ComputeExtent::default(),
            }),
            ComputeOp::Dispatch(DispatchOp::ThreadgroupsIndirect {
                source: source(),
                threads_per_group: ComputeExtent::default(),
            }),
            ComputeOp::Dispatch(DispatchOp::ThreadsIndirect { source: source() }),
        ]
    }

    /// The vocabulary is exactly the judged compute-class operations. This is
    /// the half `reims_vgpu_protocol::compute` cannot assert, because it does
    /// not know which judged compute-rail records are fences, barriers or
    /// resource state.
    #[test]
    fn the_compute_vocabulary_is_exactly_the_judged_compute_operations() {
        let mut from_ledger: Vec<u32> = LEDGER
            .iter()
            .filter(|o| {
                o.rail == Rail::Compute
                    && classify(o) == Some(OperationHome::Stream(OperationClass::Compute))
            })
            .map(|o| {
                o.opcode
                    .expect("a classified compute record carries an opcode")
            })
            .collect();
        from_ledger.sort_unstable();

        let mut from_kinds: Vec<u32> = ComputeKind::ALL.iter().map(|k| k.wire_opcode()).collect();
        from_kinds.sort_unstable();

        assert_eq!(from_ledger, from_kinds);
    }

    /// Every variant maps to a distinct kind, and together they cover the
    /// vocabulary.
    #[test]
    fn every_kind_has_a_payload_and_every_payload_a_kind() {
        let mut kinds: Vec<ComputeKind> = every_op().iter().map(|o| o.kind()).collect();
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(kinds.len(), ComputeKind::ALL.len());
    }

    /// The module's central claim: a bind touches no memory, and only the
    /// indirect records do.
    #[test]
    fn only_the_indirect_records_name_memory() {
        for op in every_op() {
            let named = !op.participations().is_empty();
            let expected = matches!(
                op,
                ComputeOp::SetStageInRegionIndirect { .. }
                    | ComputeOp::Dispatch(
                        DispatchOp::ThreadgroupsIndirect { .. }
                            | DispatchOp::ThreadsIndirect { .. }
                    )
            );
            assert_eq!(named, expected, "{:?}", op.kind());
            for access in op.participations() {
                assert_eq!(access.mode, AccessMode::Read);
                assert_eq!(access.resource, res(9));
            }
        }
    }

    /// The public argument structure gives an exact extent; the SPI form does
    /// not, and does not borrow its sibling's.
    #[test]
    fn only_the_documented_argument_block_has_an_exact_extent() {
        let threadgroups = ComputeOp::Dispatch(DispatchOp::ThreadgroupsIndirect {
            source: source(),
            threads_per_group: ComputeExtent::default(),
        });
        assert_eq!(
            threadgroups.participations()[0].extent,
            ParticipationExtent::Range(ByteRange {
                offset: 0x40,
                length: DISPATCH_THREADGROUPS_INDIRECT_ARGS_BYTES,
            })
        );
        assert_eq!(DISPATCH_THREADGROUPS_INDIRECT_ARGS_BYTES, 12);

        for op in [
            ComputeOp::Dispatch(DispatchOp::ThreadsIndirect { source: source() }),
            ComputeOp::SetStageInRegionIndirect { source: source() },
        ] {
            // No established layout, so the claim widens to the whole buffer
            // rather than narrowing on a guessed span. The offset is not lost
            // by that — see `Participation::buffer_read`.
            assert_eq!(op.participations()[0].extent, ParticipationExtent::Whole);
        }
    }

    /// The offset rebind names no buffer, so nothing about the slot's contents
    /// is resolved a second time.
    #[test]
    fn an_offset_rebind_names_no_resource() {
        let op = ComputeOp::RebindBufferOffset {
            index: 3,
            offset: 0x10,
            stride: None,
        };
        assert!(op.participations().is_empty());
        assert!(!op.is_dispatch());
    }

    /// Exactly four variants dispatch.
    #[test]
    fn four_variants_dispatch() {
        assert_eq!(every_op().iter().filter(|o| o.is_dispatch()).count(), 4);
    }
}
