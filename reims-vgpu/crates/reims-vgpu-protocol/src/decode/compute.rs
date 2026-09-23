//! Lifting the compute-encoder records.
//!
//! # Seventeen records that agree about almost nothing
//!
//! Two dispatch selectors write byte-identical records and differ only in
//! opcode — which is the entire difference, because the first extent is
//! threadgroups in one and threads in the other. Two indirect records on the
//! same encoder put their ref and their offset in **opposite orders**. A
//! stage-in region writes its size before its origin, where `MTLRegion`
//! declares the origin first and every blit record writes it first.
//! `setThreadgroupMemoryLength:atIndex:` leads with the length and trails with
//! the slot, while `setBufferOffset:atIndex:` — the record beside it, and the
//! same shape of statement — leads with the slot.
//!
//! Nothing there is inferable from a neighbour, which is why every field comes
//! from a wire view. The one thing this module does add is the dispatch
//! distinction the opcode carries and the record does not: two variants rather
//! than a flag, because a caller that read the wrong one dispatches a grid of
//! the wrong size.
//!
//! # The bind entries are borrowed
//!
//! Five of the records carry a counted array, and it stays a window into the
//! guest's bytes. The model has an arena for them and appends resolved entries
//! into it; a copy here would be made before the ids exist and thrown away
//! immediately after.
//!
//! # An ordinal the API does not define is refused
//!
//! `writeDescriptor` carries the pass's dispatch type, and it is the one field
//! in this family that is an enumeration rather than a number. A value outside
//! it is refused rather than folded onto its nearest neighbour, which would
//! report a decision the guest did not make.

use super::{no_record, short, DecodeRefusal};
use crate::closure::Rail;
use crate::compute::{ComputeKind, DispatchType};
use reims_vgpu_wire::op::Op;
use reims_vgpu_wire::ops::compute as wire;
use reims_vgpu_wire::ops::compute::{BufferBind, BufferStrideBind, RefBind, SamplerLodBind};

/// A three-dimensional count of threads or threadgroups, as the record carries
/// it. Nothing is narrowed on this encoder, unlike almost everywhere else in
/// this protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Extent {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

/// A three-dimensional origin, as the record carries it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Origin {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

/// A buffer window a record reads its arguments from, with the guest's ref.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectRef {
    pub buffer_ref: u32,
    pub offset: u64,
}

/// A dispatch whose grid the record states in threadgroups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Threadgroups {
    pub groups: Extent,
    pub threads_per_group: Extent,
}

/// A dispatch whose grid the record states in threads.
///
/// Byte-identical to [`Threadgroups`] and a different statement: the field name
/// is the whole difference, and a caller that read the wrong one dispatches a
/// grid of the wrong size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Threads {
    pub threads: Extent,
    pub threads_per_group: Extent,
}

/// A dispatch whose threadgroup count comes from a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadgroupsIndirect {
    pub source: IndirectRef,
    pub threads_per_group: Extent,
}

/// A dispatch whose thread count *and* threadgroup size both come from a
/// buffer, which is why it carries no extent of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadsIndirect {
    pub source: IndirectRef,
}

/// One lifted dispatch.
///
/// Each variant carries **one named payload** rather than inline fields, for
/// the same reason [`ComputeRecord`] does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchRecord {
    Threadgroups(Threadgroups),
    Threads(Threads),
    ThreadgroupsIndirect(ThreadgroupsIndirect),
    ThreadsIndirect(ThreadsIndirect),
}

/// A run of buffer bindings starting at one argument-table slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindBuffers<'a> {
    /// The slot the first entry lands in.
    pub first: u32,
    pub entries: &'a [BufferBind],
}

/// A run of buffer bindings that also carry a vertex-attribute stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindBuffersWithStride<'a> {
    pub first: u32,
    pub entries: &'a [BufferStrideBind],
}

/// A run of texture bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindTextures<'a> {
    pub first: u32,
    pub entries: &'a [RefBind],
}

/// A run of sampler bindings.
///
/// The same wire entry as [`BindTextures`] and a different argument table, so
/// they are two payloads rather than one — a run written into the wrong table
/// binds nothing the kernel asked for and refuses nothing either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindSamplers<'a> {
    pub first: u32,
    pub entries: &'a [RefBind],
}

/// A run of sampler bindings that also carry a LOD clamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindSamplersWithLod<'a> {
    pub first: u32,
    pub entries: &'a [SamplerLodBind],
}

/// Move an already-bound buffer's offset, and its stride when the record
/// carries one. It names no buffer: the slot keeps whatever it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebindBufferOffset {
    pub index: u32,
    pub offset: u64,
    /// `None` for `setBufferOffset:atIndex:`, which does not carry the field at
    /// all — not a stride of zero, which is a stride the guest could state.
    pub stride: Option<u64>,
}

/// The compute pipeline state the following dispatches run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetPipeline {
    pub pipeline_ref: u32,
}

/// The stage-in region, as the record states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetStageInRegion {
    pub origin: Origin,
    pub size: Extent,
}

/// The stage-in region, read from a buffer instead of stated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetStageInRegionIndirect {
    pub source: IndirectRef,
}

/// Threadgroup memory reserved at one slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetThreadgroupMemory {
    pub index: u32,
    pub length: u64,
}

/// The imageblock dimensions the pass tiles with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetImageblockSize {
    pub width: u32,
    pub height: u32,
}

/// The pass's dispatch type, the one enumerated field on this encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteDescriptor {
    pub dispatch_type: DispatchType,
}

/// One lifted compute record.
///
/// Each variant carries **one named payload** rather than inline fields, so a
/// consumer can take the record it handles by reference and cannot be handed a
/// different one. Without that, the only thing an executor could be given is
/// the whole enum, and every arm would re-match to find out which record it was
/// already dispatched on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeRecord<'a> {
    BindBuffers(BindBuffers<'a>),
    BindBuffersWithStride(BindBuffersWithStride<'a>),
    BindTextures(BindTextures<'a>),
    BindSamplers(BindSamplers<'a>),
    BindSamplersWithLod(BindSamplersWithLod<'a>),
    RebindBufferOffset(RebindBufferOffset),
    SetPipeline(SetPipeline),
    SetStageInRegion(SetStageInRegion),
    SetStageInRegionIndirect(SetStageInRegionIndirect),
    SetThreadgroupMemory(SetThreadgroupMemory),
    SetImageblockSize(SetImageblockSize),
    WriteDescriptor(WriteDescriptor),
    Dispatch(DispatchRecord),
}

impl ComputeRecord<'_> {
    /// Which record this is.
    #[must_use]
    pub const fn kind(&self) -> ComputeKind {
        match self {
            Self::BindBuffers(_) => ComputeKind::SetBuffers,
            Self::BindBuffersWithStride(_) => ComputeKind::SetBuffersWithStride,
            Self::BindTextures(_) => ComputeKind::SetTextures,
            Self::BindSamplers(_) => ComputeKind::SetSamplers,
            Self::BindSamplersWithLod(_) => ComputeKind::SetSamplersWithLod,
            Self::RebindBufferOffset(RebindBufferOffset { stride: None, .. }) => {
                ComputeKind::SetBufferOffset
            }
            Self::RebindBufferOffset(RebindBufferOffset {
                stride: Some(_), ..
            }) => ComputeKind::SetBufferOffsetStride,
            Self::SetPipeline(_) => ComputeKind::SetPipelineState,
            Self::SetStageInRegion(_) => ComputeKind::SetStageInRegion,
            Self::SetStageInRegionIndirect(_) => ComputeKind::SetStageInRegionIndirect,
            Self::SetThreadgroupMemory(_) => ComputeKind::SetThreadgroupMemoryLength,
            Self::SetImageblockSize(_) => ComputeKind::SetImageblockSize,
            Self::WriteDescriptor(_) => ComputeKind::WriteDescriptor,
            Self::Dispatch(DispatchRecord::Threadgroups(_)) => ComputeKind::DispatchThreadgroups,
            Self::Dispatch(DispatchRecord::Threads(_)) => ComputeKind::DispatchThreads,
            Self::Dispatch(DispatchRecord::ThreadgroupsIndirect(_)) => {
                ComputeKind::DispatchThreadgroupsIndirect
            }
            Self::Dispatch(DispatchRecord::ThreadsIndirect(_)) => {
                ComputeKind::DispatchThreadsIndirect
            }
        }
    }
}

/// Lift a compute record out of its bytes.
pub fn decode<'a>(op: &Op<'a>) -> Result<ComputeRecord<'a>, DecodeRefusal> {
    let opcode = op.opcode();
    let Some(kind) = ComputeKind::of_opcode(opcode) else {
        return Err(no_record(Rail::Compute, opcode));
    };
    let have = op.payload.len();
    let fail = |need: usize| short(Rail::Compute, opcode, have, need);
    let entries = |need: usize| bind_failure(op, need);

    Ok(match kind {
        ComputeKind::DispatchThreadgroups | ComputeKind::DispatchThreads => {
            let r = wire::dispatch(op).map_err(|_| fail(core::mem::size_of::<wire::Dispatch>()))?;
            let first = Extent {
                width: r.groups_width.get(),
                height: r.groups_height.get(),
                depth: r.groups_depth.get(),
            };
            let threads_per_group = Extent {
                width: r.threads_width.get(),
                height: r.threads_height.get(),
                depth: r.threads_depth.get(),
            };
            ComputeRecord::Dispatch(if matches!(kind, ComputeKind::DispatchThreadgroups) {
                DispatchRecord::Threadgroups(Threadgroups {
                    groups: first,
                    threads_per_group,
                })
            } else {
                DispatchRecord::Threads(Threads {
                    threads: first,
                    threads_per_group,
                })
            })
        }
        ComputeKind::DispatchThreadgroupsIndirect => {
            let r = wire::dispatch_indirect(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DispatchIndirect>()))?;
            ComputeRecord::Dispatch(DispatchRecord::ThreadgroupsIndirect(ThreadgroupsIndirect {
                source: IndirectRef {
                    buffer_ref: r.indirect_buffer_ref.get(),
                    offset: r.indirect_buffer_offset.get(),
                },
                threads_per_group: Extent {
                    width: r.threads_width.get(),
                    height: r.threads_height.get(),
                    depth: r.threads_depth.get(),
                },
            }))
        }
        ComputeKind::DispatchThreadsIndirect => {
            let r = wire::dispatch_threads_indirect(op)
                .map_err(|_| fail(core::mem::size_of::<wire::DispatchThreadsIndirect>()))?;
            ComputeRecord::Dispatch(DispatchRecord::ThreadsIndirect(ThreadsIndirect {
                source: IndirectRef {
                    buffer_ref: r.indirect_buffer_ref.get(),
                    offset: r.indirect_buffer_offset.get(),
                },
            }))
        }
        ComputeKind::SetBuffers => {
            let (head, entries_slice) =
                wire::buffer_binds(op).map_err(|_| entries(core::mem::size_of::<BufferBind>()))?;
            ComputeRecord::BindBuffers(BindBuffers {
                first: head.first.get(),
                entries: entries_slice,
            })
        }
        ComputeKind::SetBuffersWithStride => {
            let (head, entries_slice) = wire::buffer_stride_binds(op)
                .map_err(|_| entries(core::mem::size_of::<BufferStrideBind>()))?;
            ComputeRecord::BindBuffersWithStride(BindBuffersWithStride {
                first: head.first.get(),
                entries: entries_slice,
            })
        }
        ComputeKind::SetTextures | ComputeKind::SetSamplers => {
            let (head, entries_slice) =
                wire::ref_binds(op).map_err(|_| entries(core::mem::size_of::<RefBind>()))?;
            let first = head.first.get();
            if matches!(kind, ComputeKind::SetTextures) {
                ComputeRecord::BindTextures(BindTextures {
                    first,
                    entries: entries_slice,
                })
            } else {
                ComputeRecord::BindSamplers(BindSamplers {
                    first,
                    entries: entries_slice,
                })
            }
        }
        ComputeKind::SetSamplersWithLod => {
            let (head, entries_slice) = wire::sampler_lod_binds(op)
                .map_err(|_| entries(core::mem::size_of::<SamplerLodBind>()))?;
            ComputeRecord::BindSamplersWithLod(BindSamplersWithLod {
                first: head.first.get(),
                entries: entries_slice,
            })
        }
        ComputeKind::SetBufferOffset => {
            let r = wire::set_buffer_offset(op)
                .map_err(|_| fail(core::mem::size_of::<wire::BufferOffset>()))?;
            ComputeRecord::RebindBufferOffset(RebindBufferOffset {
                index: r.index.get(),
                offset: r.offset.get(),
                stride: None,
            })
        }
        ComputeKind::SetBufferOffsetStride => {
            let r = wire::buffer_offset_stride(op)
                .map_err(|_| fail(core::mem::size_of::<wire::BufferOffsetStride>()))?;
            ComputeRecord::RebindBufferOffset(RebindBufferOffset {
                index: r.index.get(),
                offset: r.offset.get(),
                stride: Some(r.attribute_stride.get()),
            })
        }
        ComputeKind::SetPipelineState => {
            let r = wire::set_pipeline_state(op)
                .map_err(|_| fail(core::mem::size_of::<wire::Ref>()))?;
            ComputeRecord::SetPipeline(SetPipeline {
                pipeline_ref: r.object_ref.get(),
            })
        }
        ComputeKind::SetStageInRegion => {
            let r = wire::set_stage_in_region(op)
                .map_err(|_| fail(core::mem::size_of::<wire::StageInRegion>()))?;
            ComputeRecord::SetStageInRegion(SetStageInRegion {
                origin: Origin {
                    x: r.origin_x.get(),
                    y: r.origin_y.get(),
                    z: r.origin_z.get(),
                },
                size: Extent {
                    width: r.size_width.get(),
                    height: r.size_height.get(),
                    depth: r.size_depth.get(),
                },
            })
        }
        ComputeKind::SetStageInRegionIndirect => {
            let r = wire::set_stage_in_region_indirect(op)
                .map_err(|_| fail(core::mem::size_of::<wire::StageInRegionIndirect>()))?;
            ComputeRecord::SetStageInRegionIndirect(SetStageInRegionIndirect {
                source: IndirectRef {
                    buffer_ref: r.indirect_buffer_ref.get(),
                    offset: r.indirect_buffer_offset.get(),
                },
            })
        }
        ComputeKind::SetThreadgroupMemoryLength => {
            let r = wire::set_threadgroup_memory_length(op)
                .map_err(|_| fail(core::mem::size_of::<wire::ThreadgroupMemoryLength>()))?;
            ComputeRecord::SetThreadgroupMemory(SetThreadgroupMemory {
                index: r.index.get(),
                length: r.length.get(),
            })
        }
        ComputeKind::SetImageblockSize => {
            let r = wire::set_imageblock_size(op)
                .map_err(|_| fail(core::mem::size_of::<wire::ImageblockSize>()))?;
            ComputeRecord::SetImageblockSize(SetImageblockSize {
                width: r.width.get(),
                height: r.height.get(),
            })
        }
        ComputeKind::WriteDescriptor => {
            let r = wire::write_descriptor(op)
                .map_err(|_| fail(core::mem::size_of::<wire::PassDescriptor>()))?;
            let word = r.dispatch_type.get();
            let Some(dispatch_type) = DispatchType::parse(word) else {
                return Err(DecodeRefusal::UndefinedOrdinal {
                    rail: Rail::Compute,
                    opcode,
                    field: "dispatch_type",
                    value: word,
                });
            };
            ComputeRecord::WriteDescriptor(WriteDescriptor { dispatch_type })
        }
    })
}

/// The refusal for a bind record whose head or entries did not fit.
///
/// The head parses or it does not, and the two are different failures: a record
/// with no room for its `first`/`count` pair is short, while one whose count
/// asks for more entries than the record holds is a guest number that overruns.
/// Reporting both as "short" would lose the count, which is the only number
/// that says which of the guest and the record is wrong.
fn bind_failure(op: &Op<'_>, entry_size: usize) -> DecodeRefusal {
    let head = core::mem::size_of::<wire::BindHeader>();
    let have = op.payload.len();
    if have < head {
        return short(Rail::Compute, op.opcode(), have, head);
    }
    let mut count = [0u8; 4];
    count.copy_from_slice(&op.payload[4..8]);
    let count = u32::from_le_bytes(count);
    let need = head.saturating_add((count as usize).saturating_mul(entry_size));
    if have >= need {
        // The head fit and the entries fit; the view refused for a reason this
        // decoder cannot restate, so report what it could see.
        return short(Rail::Compute, op.opcode(), have, need);
    }
    DecodeRefusal::CountOverruns {
        rail: Rail::Compute,
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

    fn lift(bytes: &[u8]) -> Result<ComputeRecord<'_>, DecodeRefusal> {
        decode(&op(bytes, 0).expect("framed"))
    }

    fn u64s(values: &[u64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The two direct dispatches are byte-identical and the opcode is the whole
    /// difference. Lifted, they are different variants — because the first
    /// extent is threadgroups in one and threads in the other, and a caller
    /// that read the wrong one dispatches a grid of the wrong size.
    #[test]
    fn the_two_direct_dispatches_share_a_record_and_not_a_meaning() {
        let payload = u64s(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66]);
        let groups = Extent {
            width: 0x11,
            height: 0x22,
            depth: 0x33,
        };
        let threads_per_group = Extent {
            width: 0x44,
            height: 0x55,
            depth: 0x66,
        };
        assert_eq!(
            lift(&record(wire::OPCODE_DISPATCH_THREADGROUPS, &payload)),
            Ok(ComputeRecord::Dispatch(DispatchRecord::Threadgroups(
                Threadgroups {
                    groups,
                    threads_per_group,
                }
            )))
        );
        assert_eq!(
            lift(&record(wire::OPCODE_DISPATCH_THREADS, &payload)),
            Ok(ComputeRecord::Dispatch(DispatchRecord::Threads(Threads {
                threads: groups,
                threads_per_group,
            })))
        );
    }

    /// The two indirect records put their ref and offset in opposite orders,
    /// and both lift to the same `IndirectRef`. Distinct values, so a decoder
    /// that shared one order between them reads the ref as part of the offset.
    #[test]
    fn the_two_indirect_records_have_opposite_field_orders_and_one_meaning() {
        let mut dispatch = u64s(&[0x44, 0x55, 0x66, 0x1111]);
        dispatch.extend_from_slice(&5151u32.to_le_bytes());
        assert_eq!(
            lift(&record(
                wire::OPCODE_DISPATCH_THREADGROUPS_INDIRECT,
                &dispatch
            )),
            Ok(ComputeRecord::Dispatch(
                DispatchRecord::ThreadgroupsIndirect(ThreadgroupsIndirect {
                    source: IndirectRef {
                        buffer_ref: 5151,
                        offset: 0x1111,
                    },
                    threads_per_group: Extent {
                        width: 0x44,
                        height: 0x55,
                        depth: 0x66,
                    },
                })
            ))
        );

        let mut region = 5151u32.to_le_bytes().to_vec();
        region.extend_from_slice(&0x1111u64.to_le_bytes());
        assert_eq!(
            lift(&record(wire::OPCODE_SET_STAGE_IN_REGION_INDIRECT, &region)),
            Ok(ComputeRecord::SetStageInRegionIndirect(
                SetStageInRegionIndirect {
                    source: IndirectRef {
                        buffer_ref: 5151,
                        offset: 0x1111,
                    },
                }
            ))
        );
    }

    /// The stage-in region writes its size first, where `MTLRegion` declares
    /// the origin first. Six distinct values, so the transposition would show.
    #[test]
    fn the_stage_in_region_writes_its_size_before_its_origin() {
        let payload = u64s(&[0x44, 0x55, 0x66, 0x11, 0x22, 0x33]);
        assert_eq!(
            lift(&record(wire::OPCODE_SET_STAGE_IN_REGION, &payload)),
            Ok(ComputeRecord::SetStageInRegion(SetStageInRegion {
                origin: Origin {
                    x: 0x11,
                    y: 0x22,
                    z: 0x33
                },
                size: Extent {
                    width: 0x44,
                    height: 0x55,
                    depth: 0x66
                },
            }))
        );
    }

    /// A bind's entries are borrowed out of the guest's own bytes, and `first`
    /// comes from the head rather than being assumed zero.
    #[test]
    fn a_bind_borrows_its_entries_and_keeps_the_slot_it_starts_at() {
        let mut payload = 5u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&2u32.to_le_bytes());
        payload.extend_from_slice(&4242u32.to_le_bytes());
        payload.extend_from_slice(&4343u32.to_le_bytes());
        let bytes = record(wire::OPCODE_SET_TEXTURE, &payload);
        let ComputeRecord::BindTextures(BindTextures { first, entries }) =
            lift(&bytes).expect("lifted")
        else {
            panic!("not a texture bind");
        };
        assert_eq!(first, 5);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].object_ref.get(), 4343);
        assert!(bytes
            .as_ptr_range()
            .contains(&entries.as_ptr().cast::<u8>()));
    }

    /// The two rebind forms differ by one field, and the stride's presence is
    /// what tells them apart rather than a zero.
    #[test]
    fn the_offset_rebind_carries_a_stride_only_when_its_record_does() {
        let mut plain = 6u32.to_le_bytes().to_vec();
        plain.extend_from_slice(&0x5678u64.to_le_bytes());
        let mut strided = plain.clone();
        strided.extend_from_slice(&0x3456u64.to_le_bytes());
        assert_eq!(
            lift(&record(wire::OPCODE_SET_BUFFER_OFFSET, &plain)),
            Ok(ComputeRecord::RebindBufferOffset(RebindBufferOffset {
                index: 6,
                offset: 0x5678,
                stride: None,
            }))
        );
        assert_eq!(
            lift(&record(wire::OPCODE_SET_BUFFER_OFFSET_STRIDE, &strided)),
            Ok(ComputeRecord::RebindBufferOffset(RebindBufferOffset {
                index: 6,
                offset: 0x5678,
                stride: Some(0x3456),
            }))
        );
    }

    /// A dispatch type the API does not define is refused by name rather than
    /// folded onto the nearest one it does.
    #[test]
    fn an_undefined_dispatch_type_is_refused_and_names_its_field() {
        assert_eq!(
            lift(&record(wire::OPCODE_WRITE_DESCRIPTOR, &7u32.to_le_bytes())),
            Err(DecodeRefusal::UndefinedOrdinal {
                rail: Rail::Compute,
                opcode: wire::OPCODE_WRITE_DESCRIPTOR,
                field: "dispatch_type",
                value: 7,
            })
        );
        assert_eq!(
            lift(&record(wire::OPCODE_WRITE_DESCRIPTOR, &1u32.to_le_bytes())),
            Ok(ComputeRecord::WriteDescriptor(WriteDescriptor {
                dispatch_type: DispatchType::Concurrent,
            }))
        );
    }

    /// A count larger than the record it sits in reports the count and the
    /// bytes; a record with no room for the count at all is short.
    #[test]
    fn a_bind_count_that_overruns_is_told_apart_from_a_short_head() {
        let mut payload = 0u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&200u32.to_le_bytes());
        payload.extend_from_slice(&4242u32.to_le_bytes());
        assert_eq!(
            lift(&record(wire::OPCODE_SET_TEXTURE, &payload)),
            Err(DecodeRefusal::CountOverruns {
                rail: Rail::Compute,
                opcode: wire::OPCODE_SET_TEXTURE,
                count: 200,
                have: payload.len(),
            })
        );
        assert!(matches!(
            lift(&record(wire::OPCODE_SET_TEXTURE, &[0u8; 7])),
            Err(DecodeRefusal::Short { .. })
        ));
    }

    /// Every kind in the vocabulary lifts a record that reports that same kind.
    /// Driven off `ComputeKind::ALL`, so a kind added without a decode arm
    /// fails here rather than at a guest's expense.
    #[test]
    fn every_compute_kind_lifts_a_record_that_names_it() {
        // Long enough for the widest body, zero-filled: a `first`/`count` of
        // zero is a bind of no slots, which is a legal record.
        let payload = [0u8; 64];
        for kind in ComputeKind::ALL {
            let bytes = record(kind.wire_opcode(), &payload);
            let lifted = lift(&bytes).expect("every kind lifts");
            assert_eq!(lifted.kind(), *kind, "{kind:?}");
        }
    }

    /// An opcode this encoder does not carry lifts nothing, and the refusal
    /// says whether the ledger has ever heard of it.
    #[test]
    fn an_opcode_outside_the_encoder_lifts_nothing() {
        let unknown = 0x7fff;
        assert_eq!(
            lift(&record(unknown, &[])),
            Err(DecodeRefusal::UnknownOpcode {
                rail: Rail::Compute,
                opcode: unknown,
            })
        );
    }
}
