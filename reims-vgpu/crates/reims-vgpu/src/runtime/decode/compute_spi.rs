//! The compute-rail records the closure ledger has **not settled**.
//!
//! Eleven opcodes: the `MTLFence` update/wait pair, the seven control-flow SPI
//! markers and predicates, and the two `executeCommandsInBuffer:` forms.
//!
//! # Why they are decoded here and not in `reims-vgpu-protocol`
//!
//! An unresolved row has no established contract, so the layer that assigns
//! meaning to a wire tag may not give it a shape.
//! `reims_vgpu_protocol::decode::compute` lifts exactly the seventeen rows the
//! ledger has settled, `sync::fence_kind` deliberately answers `None` for this
//! rail's fence pair, and `decode::icb` has no compute arm at all. Each of
//! those is the correct answer for a decoder that promises a lifted record only
//! where one is established.
//!
//! # How this differs from [`super::blit_spi`], and it matters
//!
//! That module decodes its four rows in order to *decline* them: no executor
//! takes a value from it, and the decline's count is the measurement that will
//! settle the row. These eleven are not like that. Every one of them **drives
//! real work** — `runtime::fence_exec::execute_fence` for the pair,
//! `runtime::compute_session::MetalSession::{encode_control, encode_icb}` for
//! the other nine — on evidence this project gathered outside the ledger, and
//! that has been true since before the ledger existed.
//!
//! So the honest statement is not "these are declined pending evidence" but
//! "these are executed and the ledger has not caught up". Holding them here
//! rather than widening the protocol crate to cover them is what keeps that
//! distinction legible: a reader who wants to know which of this rail's records
//! rest on a settled contract can read the module a record came from.
//!
//! # It refuses every settled opcode by name
//!
//! No record on this rail has two readings. A settled row reaching [`decode`]
//! is [`DecodeStatus::ErrSettledElsewhere`], not a record — the routing in
//! `runtime::exec::handle_compute_record` disagreeing with the ledger, and it
//! is named rather than answered a second time.

use reims_vgpu_wire::ops::compute as wire;

/// Which unsettled record this is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    /// Never produced by [`decode`]; the `Default` a caller builds a command
    /// from before filling it in.
    #[default]
    Unknown,
    UpdateFence,
    WaitFence,
    ControlStartDoWhile,
    ControlEndDoWhile,
    ControlStartWhile,
    ControlEndWhile,
    ControlStartIf,
    ControlStartElse,
    ControlEndIf,
    ExecuteCommandsInBuffer,
    ExecuteCommandsInBufferIndirect,
}

/// Why this decoder refused a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    ErrShort,
    ErrUnknownOpcode,
    /// A row the ledger *has* settled, which a protocol decoder owns.
    ErrSettledElsewhere,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs keep the `compute_decode_` prefix the rail's decoder reported
    /// under, so a census taken across the cutover reads continuously.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ErrShort => "compute_decode_short",
            Self::ErrUnknownOpcode => "compute_decode_unknown_opcode",
            Self::ErrSettledElsewhere => "compute_decode_settled_elsewhere",
        })
    }
}

/// One decoded unsettled compute record.
///
/// A struct rather than an enum, because its two consumers are Metal encoder
/// methods that switch on [`Kind`] and read the fields their own arm needs.
/// The fields that are not this record's are zero, which is the flat shape the
/// settled records have left behind — and the reason to leave it here rather
/// than carry it forward: when a row is settled its record moves to the
/// protocol crate with a payload of its own.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Command {
    pub opcode: u32,
    pub kind: Kind,
    /// `updateFence:` / `waitForFence:`.
    pub fence_ref: u32,
    /// The four fields of a control-flow predicate. Only `encodeEndDoWhile:`,
    /// `encodeStartWhile:` and `encodeStartIf:` carry them; the four markers
    /// are the header alone.
    pub condition_buffer_ref: u32,
    pub condition_buffer_offset: u64,
    pub condition_comparison: u32,
    pub condition_reference_value: u32,
    /// The indirect-command-buffer execution's fields. `range_*` belong to the
    /// range form and `arguments_*` to the indirect one; neither form writes
    /// both.
    pub indirect_command_buffer_ref: u32,
    pub indirect_command_range_location: u64,
    pub indirect_command_range_length: u64,
    pub indirect_command_arguments_buffer_ref: u32,
    pub indirect_command_arguments_buffer_offset: u64,
}

/// Whether `opcode` is one of the eleven rows this module owns.
#[must_use]
pub fn is_unsettled(opcode: u32) -> bool {
    matches!(
        opcode,
        wire::OPCODE_UPDATE_FENCE
            | wire::OPCODE_WAIT_FOR_FENCE
            | wire::OPCODE_START_DO_WHILE
            | wire::OPCODE_END_DO_WHILE
            | wire::OPCODE_START_WHILE
            | wire::OPCODE_END_WHILE
            | wire::OPCODE_START_IF
            | wire::OPCODE_START_ELSE
            | wire::OPCODE_END_IF
            | wire::OPCODE_EXECUTE_COMMANDS_RANGE
            | wire::OPCODE_EXECUTE_COMMANDS_INDIRECT
    )
}

/// Decode one unsettled compute-rail record.
///
/// # Errors
///
/// [`DecodeStatus::ErrSettledElsewhere`] for a row a protocol decoder owns,
/// [`DecodeStatus::ErrUnknownOpcode`] for an opcode no compute row names, and
/// [`DecodeStatus::ErrShort`] for a record whose length is not its body's.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    let op = reims_vgpu_wire::op(command, 0).map_err(|_| DecodeStatus::ErrShort)?;
    let opcode = op.opcode();
    if !is_unsettled(opcode) {
        // The ledger is what says which of the two this is, so neither answer
        // is this module's opinion about an opcode.
        return Err(
            if reims_vgpu_protocol::closure::find(
                reims_vgpu_protocol::closure::Rail::Compute,
                opcode,
            )
            .is_some()
            {
                DecodeStatus::ErrSettledElsewhere
            } else {
                DecodeStatus::ErrUnknownOpcode
            },
        );
    }
    let command_length = op.length() as usize;
    let want = |need: u32| {
        if command_length == need as usize {
            Ok(())
        } else {
            Err(DecodeStatus::ErrShort)
        }
    };
    let mut out = Command {
        opcode,
        ..Default::default()
    };
    match opcode {
        wire::OPCODE_UPDATE_FENCE | wire::OPCODE_WAIT_FOR_FENCE => {
            want(wire::FENCE_TOTAL_LEN)?;
            let r = wire::fence(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = if opcode == wire::OPCODE_UPDATE_FENCE {
                Kind::UpdateFence
            } else {
                Kind::WaitFence
            };
            out.fence_ref = r.object_ref.get();
        }
        wire::OPCODE_START_WHILE | wire::OPCODE_START_IF | wire::OPCODE_END_DO_WHILE => {
            want(wire::CONTROL_FLOW_PREDICATE_TOTAL_LEN)?;
            let p = wire::control_flow_predicate(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = match opcode {
                wire::OPCODE_START_WHILE => Kind::ControlStartWhile,
                wire::OPCODE_START_IF => Kind::ControlStartIf,
                _ => Kind::ControlEndDoWhile,
            };
            out.condition_buffer_ref = p.buffer_ref.get();
            out.condition_buffer_offset = p.offset.get();
            out.condition_comparison = p.comparison.get();
            out.condition_reference_value = p.reference_value.get();
        }
        wire::OPCODE_START_DO_WHILE
        | wire::OPCODE_END_WHILE
        | wire::OPCODE_START_ELSE
        | wire::OPCODE_END_IF => {
            want(wire::CONTROL_FLOW_MARKER_TOTAL_LEN)?;
            out.kind = match opcode {
                wire::OPCODE_START_DO_WHILE => Kind::ControlStartDoWhile,
                wire::OPCODE_END_WHILE => Kind::ControlEndWhile,
                wire::OPCODE_START_ELSE => Kind::ControlStartElse,
                _ => Kind::ControlEndIf,
            };
        }
        wire::OPCODE_EXECUTE_COMMANDS_RANGE => {
            want(wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN)?;
            let e = wire::execute_commands_range(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommandsInBuffer;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.indirect_command_range_location = e.range_location.get();
            out.indirect_command_range_length = e.range_length.get();
        }
        _ => {
            want(wire::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN)?;
            let e = wire::execute_commands_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommandsInBufferIndirect;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.indirect_command_arguments_buffer_ref = e.indirect_buffer_ref.get();
            out.indirect_command_arguments_buffer_offset = e.indirect_buffer_offset.get();
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::closure::{Rail, LEDGER};

    fn record(opcode: u32, total_len: u32) -> Vec<u8> {
        let mut v = vec![0u8; total_len as usize];
        v[0..4].copy_from_slice(&opcode.to_le_bytes());
        v[4..8].copy_from_slice(&total_len.to_le_bytes());
        v
    }

    /// No settled compute row is decodable here.
    ///
    /// This is the claim that keeps "one record, one reading" structural rather
    /// than a property of the current routing. `runtime::exec` picks between
    /// this decoder and the protocol crate's by the ledger's class; if this one
    /// also answered a settled row, a routing mistake would be a silent second
    /// interpretation instead of a named refusal.
    #[test]
    fn no_settled_compute_row_is_a_record_this_decoder_owns() {
        let mut settled = 0;
        for op in LEDGER
            .iter()
            .filter(|o| o.rail == Rail::Compute)
            .filter(|o| !o.closure.blocks_cutover())
            .filter_map(|o| o.opcode)
        {
            settled += 1;
            assert_eq!(
                decode(&record(op, 256)),
                Err(DecodeStatus::ErrSettledElsewhere),
                "compute {op:#x} is settled and this decoder claimed it"
            );
        }
        assert!(
            settled >= 20,
            "the ledger settles most of this rail; {settled} rows is not that"
        );
    }

    /// Every unsettled compute row *is* one, and the two sets do not overlap.
    #[test]
    fn every_unsettled_compute_row_reaches_this_decoder() {
        for op in LEDGER
            .iter()
            .filter(|o| o.rail == Rail::Compute)
            .filter(|o| o.closure.blocks_cutover())
            .filter_map(|o| o.opcode)
        {
            // The unqualified residency pair is unsettled and is *not* this
            // module's: it is declined from its own lift in `runtime::exec`,
            // because it names no work this device performs.
            if matches!(op, 0x86 | 0x87) {
                assert!(!is_unsettled(op));
                continue;
            }
            assert!(
                is_unsettled(op),
                "compute {op:#x} is unsettled and no decoder owns it"
            );
        }
    }

    /// The predicate forms read four fields and the markers read none, and the
    /// length is what tells them apart — a marker sized as a predicate is a
    /// record this device cannot have received.
    #[test]
    fn a_control_marker_and_a_control_predicate_are_told_apart_by_their_length() {
        let marker = decode(&record(
            wire::OPCODE_START_DO_WHILE,
            wire::CONTROL_FLOW_MARKER_TOTAL_LEN,
        ))
        .expect("decoded");
        assert_eq!(marker.kind, Kind::ControlStartDoWhile);
        assert_eq!(marker.condition_buffer_ref, 0);

        assert_eq!(
            decode(&record(
                wire::OPCODE_START_DO_WHILE,
                wire::CONTROL_FLOW_PREDICATE_TOTAL_LEN,
            )),
            Err(DecodeStatus::ErrShort)
        );
        assert_eq!(
            decode(&record(
                wire::OPCODE_START_IF,
                wire::CONTROL_FLOW_MARKER_TOTAL_LEN,
            )),
            Err(DecodeStatus::ErrShort)
        );
    }

    /// The two indirect-command executions are told apart by which second
    /// operand they name: a range, or a buffer to read one from.
    #[test]
    fn the_two_command_buffer_executions_name_different_second_operands() {
        let range = decode(&record(
            wire::OPCODE_EXECUTE_COMMANDS_RANGE,
            wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN,
        ))
        .expect("decoded");
        let indirect = decode(&record(
            wire::OPCODE_EXECUTE_COMMANDS_INDIRECT,
            wire::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN,
        ))
        .expect("decoded");
        assert_eq!(range.kind, Kind::ExecuteCommandsInBuffer);
        assert_eq!(indirect.kind, Kind::ExecuteCommandsInBufferIndirect);
        // Each form leaves the other's operand alone rather than filling it
        // with a zero that reads as a stated value.
        assert_eq!(range.indirect_command_arguments_buffer_ref, 0);
        assert_eq!(indirect.indirect_command_range_length, 0);
    }

    /// The fence pair is one opcode apart and two directions.
    #[test]
    fn the_fence_pair_keeps_its_two_directions_apart() {
        assert_eq!(
            decode(&record(wire::OPCODE_UPDATE_FENCE, wire::FENCE_TOTAL_LEN))
                .expect("decoded")
                .kind,
            Kind::UpdateFence
        );
        assert_eq!(
            decode(&record(wire::OPCODE_WAIT_FOR_FENCE, wire::FENCE_TOTAL_LEN))
                .expect("decoded")
                .kind,
            Kind::WaitFence
        );
    }

    /// An opcode the compute ledger does not name at all is neither settled
    /// elsewhere nor a record here.
    #[test]
    fn an_opcode_no_compute_row_names_is_unknown_rather_than_settled() {
        assert_eq!(
            decode(&record(0xfff, 64)),
            Err(DecodeStatus::ErrUnknownOpcode)
        );
    }
}
