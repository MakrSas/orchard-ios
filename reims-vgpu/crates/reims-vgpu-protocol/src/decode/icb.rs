//! Lifting the indirect-command-buffer records.
//!
//! # Three judged records, and four more that decode to nothing on purpose
//!
//! Seven opcodes are indirect-command work. Three have settled contracts — the
//! render encoder's two execute forms and the blit encoder's optimize hint —
//! and this module lifts those. The compute encoder's execute pair, the ICB
//! copy and the reset are unresolved in the ledger, so they lift nothing and
//! the refusal says the contract is open rather than the opcode unknown.
//!
//! That is not a gap to close by writing four more arms. An unresolved row is a
//! row whose *behaviour* is not established; a decoder that lifted it would
//! hand the model an operation nobody can say what to do with, which is the
//! shape of guess the ledger exists to prevent.
//!
//! # The range is in commands, not bytes
//!
//! `withRange:` counts commands. Every other range in this protocol counts
//! bytes, and the two are not convertible — a command's size is the device's
//! representation, not the guest's. [`CommandRange`] is its own type for
//! exactly that reason, so a command index cannot reach a byte offset.

use super::{no_record, short, DecodeRefusal};
use crate::closure::Rail;
use reims_vgpu_wire::op::Op;
use reims_vgpu_wire::ops::{blit as wire_blit, render as wire_render};

/// A half-open range of commands inside an indirect command buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommandRange {
    pub location: u64,
    pub length: u64,
}

/// A buffer window an execute record reads its range from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectRef {
    pub buffer_ref: u32,
    pub offset: u64,
}

/// One lifted indirect-command record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcbRecord {
    /// Execute a literal range of commands.
    ExecuteRange {
        icb_ref: u32,
        commands: CommandRange,
    },
    /// Execute commands whose range comes from a buffer. The range's encoding
    /// inside that buffer is not established, so nothing here reads it.
    ExecuteIndirect {
        icb_ref: u32,
        arguments: IndirectRef,
    },
    /// Prepare a range of an ICB for reuse.
    Optimize {
        icb_ref: u32,
        commands: CommandRange,
    },
}

impl IcbRecord {
    /// The indirect command buffer this record is about.
    #[must_use]
    pub const fn icb_ref(&self) -> u32 {
        match *self {
            Self::ExecuteRange { icb_ref, .. }
            | Self::ExecuteIndirect { icb_ref, .. }
            | Self::Optimize { icb_ref, .. } => icb_ref,
        }
    }
}

/// Lift an indirect-command record out of its bytes.
pub fn decode(rail: Rail, op: &Op<'_>) -> Result<IcbRecord, DecodeRefusal> {
    let opcode = op.opcode();
    let have = op.payload.len();
    let fail = |need: usize| short(rail, opcode, have, need);
    match (rail, opcode) {
        (Rail::Render, wire_render::OPCODE_EXECUTE_COMMANDS_RANGE) => {
            let r = wire_render::execute_commands_range(op)
                .map_err(|_| fail(core::mem::size_of::<wire_render::ExecuteCommandsRange>()))?;
            Ok(IcbRecord::ExecuteRange {
                icb_ref: r.icb_ref.get(),
                commands: CommandRange {
                    location: r.range_location.get(),
                    length: r.range_length.get(),
                },
            })
        }
        (Rail::Render, wire_render::OPCODE_EXECUTE_COMMANDS_INDIRECT) => {
            let r = wire_render::execute_commands_indirect(op)
                .map_err(|_| fail(core::mem::size_of::<wire_render::ExecuteCommandsIndirect>()))?;
            Ok(IcbRecord::ExecuteIndirect {
                icb_ref: r.icb_ref.get(),
                arguments: IndirectRef {
                    buffer_ref: r.indirect_buffer_ref.get(),
                    offset: r.indirect_buffer_offset.get(),
                },
            })
        }
        (Rail::Blit, wire_blit::OPCODE_OPTIMIZE_ICB) => {
            let r = wire_blit::icb_range(op)
                .map_err(|_| fail(core::mem::size_of::<wire_blit::IcbRange>()))?;
            Ok(IcbRecord::Optimize {
                icb_ref: r.icb_ref.get(),
                commands: CommandRange {
                    location: r.range_location.get(),
                    length: r.range_length.get(),
                },
            })
        }
        _ => Err(no_record(rail, opcode)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use reims_vgpu_wire::op::{op, OP_HEADER_LEN};
    use reims_vgpu_wire::ops::compute as wire_compute;

    fn record(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let total = (OP_HEADER_LEN + payload.len()) as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&total.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn lift(rail: Rail, bytes: &[u8]) -> Result<IcbRecord, DecodeRefusal> {
        decode(rail, &op(bytes, 0).expect("framed"))
    }

    fn range_payload(icb: u32) -> Vec<u8> {
        let mut out = icb.to_le_bytes().to_vec();
        out.extend_from_slice(&4u64.to_le_bytes());
        out.extend_from_slice(&6u64.to_le_bytes());
        out
    }

    /// The two execute forms name the same ICB and differ in where the range
    /// comes from: one carries it, the other names a buffer and does not read
    /// it here.
    #[test]
    fn the_two_execute_forms_agree_on_the_icb_and_differ_on_the_range() {
        let bytes = record(
            wire_render::OPCODE_EXECUTE_COMMANDS_RANGE,
            &range_payload(7070),
        );
        assert_eq!(
            lift(Rail::Render, &bytes),
            Ok(IcbRecord::ExecuteRange {
                icb_ref: 7070,
                commands: CommandRange {
                    location: 4,
                    length: 6
                },
            })
        );

        let mut indirect = 7070u32.to_le_bytes().to_vec();
        indirect.extend_from_slice(&5151u32.to_le_bytes());
        indirect.extend_from_slice(&0x1111u64.to_le_bytes());
        let bytes = record(wire_render::OPCODE_EXECUTE_COMMANDS_INDIRECT, &indirect);
        let lifted = lift(Rail::Render, &bytes).expect("lifted");
        assert_eq!(lifted.icb_ref(), 7070);
        assert_eq!(
            lifted,
            IcbRecord::ExecuteIndirect {
                icb_ref: 7070,
                arguments: IndirectRef {
                    buffer_ref: 5151,
                    offset: 0x1111,
                },
            }
        );
    }

    /// The optimize hint carries a command range like the execute form, and it
    /// is a different operation rather than an execute the device skips.
    #[test]
    fn the_optimize_hint_is_its_own_record() {
        let bytes = record(wire_blit::OPCODE_OPTIMIZE_ICB, &range_payload(7070));
        assert_eq!(
            lift(Rail::Blit, &bytes),
            Ok(IcbRecord::Optimize {
                icb_ref: 7070,
                commands: CommandRange {
                    location: 4,
                    length: 6
                },
            })
        );
    }

    /// The four unresolved rows lift nothing, and the refusal says the contract
    /// is open. The compute pair is the case that matters: its records are
    /// byte-identical to the render encoder's, so a rail-free decoder would
    /// execute work whose behaviour nothing has established.
    #[test]
    fn the_unresolved_rows_are_unjudged_rather_than_lifted() {
        for (rail, opcode, payload) in [
            (
                Rail::Compute,
                wire_compute::OPCODE_EXECUTE_COMMANDS_RANGE,
                range_payload(7070),
            ),
            (
                Rail::Compute,
                wire_compute::OPCODE_EXECUTE_COMMANDS_INDIRECT,
                range_payload(7070),
            ),
            (Rail::Blit, wire_blit::OPCODE_RESET_ICB, range_payload(7070)),
            (Rail::Blit, wire_blit::OPCODE_COPY_ICB, alloc::vec![0u8; 32]),
        ] {
            assert_eq!(
                lift(rail, &record(opcode, &payload)),
                Err(DecodeRefusal::Unjudged { rail, opcode }),
                "{rail:?} {opcode:#x}"
            );
        }
    }

    /// The compute rail's execute-range opcode is the compute encoder's, and
    /// asking the render rail for it is a different question with a different
    /// answer. Neither rail answers for the other.
    #[test]
    fn the_two_rails_execute_opcodes_do_not_answer_for_each_other() {
        let payload = range_payload(7070);
        let compute_opcode = wire_compute::OPCODE_EXECUTE_COMMANDS_RANGE;
        assert_ne!(compute_opcode, wire_render::OPCODE_EXECUTE_COMMANDS_RANGE);
        assert!(lift(Rail::Render, &record(compute_opcode, &payload)).is_err());
        assert!(lift(
            Rail::Compute,
            &record(wire_render::OPCODE_EXECUTE_COMMANDS_RANGE, &payload)
        )
        .is_err());
    }

    /// A record too short for its body is refused with both numbers.
    #[test]
    fn a_short_record_is_refused() {
        let bytes = record(wire_render::OPCODE_EXECUTE_COMMANDS_RANGE, &[0u8; 19]);
        assert_eq!(
            lift(Rail::Render, &bytes),
            Err(DecodeRefusal::Short {
                rail: Rail::Render,
                opcode: wire_render::OPCODE_EXECUTE_COMMANDS_RANGE,
                have: 19,
                need: core::mem::size_of::<wire_render::ExecuteCommandsRange>(),
            })
        );
    }
}
