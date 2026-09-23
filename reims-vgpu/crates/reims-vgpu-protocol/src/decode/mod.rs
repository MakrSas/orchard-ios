//! Records lifted out of bytes, with the guest's own names still on them.
//!
//! # Two steps, and this is the first
//!
//! Turning a packet into work is two questions and they have different owners.
//! *What did the guest write* is a wire question with a contract answer, and it
//! is this. *Which object does this ref mean, and what is its content version*
//! is a question about device state, and it belongs to `reims-vgpu-core`, which
//! has the registries.
//!
//! Splitting them is what stops the second one being answered twice. A decoder
//! that resolved refs would need the object namespace, so it would either take
//! a reference to it — making every decode a borrow of live device state — or
//! resolve again later, which is the exact "resolve twice, get two answers"
//! shape the replacement exists to delete. So a record here carries the
//! guest's `u32` refs verbatim and the model resolves them once.
//!
//! # A refusal is typed, and the byte count is in it
//!
//! Every failure here is a wire fact: a record too short for its own body, an
//! opcode with no contract, a count that does not fit the record it is in.
//! Each one names what it saw, because a decode failure with no numbers in it
//! is a failure nobody can act on.

pub mod blit;
pub mod compute;
pub mod icb;
pub mod render;
pub mod residency;
pub mod resource_state;
pub mod sync;

use crate::closure::Rail;

/// The framed record every decoder here reads, and the element types the
/// records borrow arrays of.
///
/// Re-exported rather than left for a caller to reach past this crate for. A
/// layer above this one — the model — has to name a decoded record's entries to
/// resolve them, and the rule it works under is that everything it needs from
/// the wire comes through the layer that assigned the meaning. An element type
/// is a layout with no meaning of its own; which table it fills and what a slot
/// means is decided by the record carrying it, and that decision is made here.
pub use reims_vgpu_wire::op::{op, Op, OpStream, OP_HEADER_LEN};
pub use reims_vgpu_wire::ops::render::{
    BufferBind, BufferStrideBind, RefBind, SamplerLodBind, ScissorRect, Viewport,
};
pub use reims_vgpu_wire::{F32le, F64le, U16le, U32le, U64le};

pub use reims_vgpu_wire::ops::render_pass::{
    AttachmentPrefix, ColorAttachmentBody, DepthAttachmentBody, RenderPassBody,
    StencilAttachmentBody, RENDER_PASS_COLOR_ATTACHMENTS,
};

/// Why a record could not be lifted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeRefusal {
    /// The record is shorter than the body its opcode requires.
    Short {
        rail: Rail,
        opcode: u32,
        have: usize,
        need: usize,
    },
    /// The opcode belongs to no record this rail carries.
    ///
    /// Distinct from [`Self::Unjudged`]: this one is not in the rail's map at
    /// all, which is a stream that has gone wrong or a serializer this device
    /// has never seen. That is a different report from an opcode whose contract
    /// is merely open.
    UnknownOpcode { rail: Rail, opcode: u32 },
    /// The opcode is real and its contract is not established, so the model
    /// must not represent it.
    Unjudged { rail: Rail, opcode: u32 },
    /// The opcode's contract is established and the device refuses it.
    ///
    /// Distinct from [`Self::Unjudged`], which says nothing is known. This one
    /// says the row is settled and the settlement is a refusal, so a record
    /// that decoded perfectly must still not become an operation. Reporting the
    /// two the same way would make a deliberate refusal look like an open
    /// question and put it back on the work queue.
    RefusedByContract { rail: Rail, opcode: u32 },
    /// The record carries bytes and its selector takes no argument.
    ///
    /// Only reachable for the records whose whole body is the opcode. Every
    /// other record is a fixed body, and a longer buffer merely contains one —
    /// there is nothing to refuse. Reporting this as [`Self::Short`] would say
    /// the opposite of what happened.
    UnexpectedPayload {
        rail: Rail,
        opcode: u32,
        have: usize,
    },
    /// A field's value names nothing the API defines.
    ///
    /// The guest wrote an ordinal outside the enumeration the field carries.
    /// Refused here rather than carried on, because the whole reason a record
    /// becomes a typed value at this layer is that everything above it can stop
    /// asking whether the value means anything — and because the alternative,
    /// folding an unknown ordinal onto its nearest neighbour, reports a
    /// decision the guest did not make.
    UndefinedOrdinal {
        rail: Rail,
        opcode: u32,
        /// The record's own name for the field, so the report says which.
        field: &'static str,
        value: u32,
    },
    /// A counted array does not fit the record that declares it.
    ///
    /// The count is the guest's, so this is an ordinary hostile-input case and
    /// not a corrupt device. The declared count is reported beside the bytes
    /// available, because "the guest asked for 200" and "the record held 12" is
    /// the pair that identifies which of the two is wrong.
    CountOverruns {
        rail: Rail,
        opcode: u32,
        count: u32,
        have: usize,
    },
}

impl DecodeRefusal {
    /// The stable reason string for the failure channel.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Short { .. } => "decode_record_short",
            Self::UnknownOpcode { .. } => "decode_opcode_unknown",
            Self::Unjudged { .. } => "decode_opcode_unjudged",
            Self::RefusedByContract { .. } => "decode_opcode_refused_by_contract",
            Self::UnexpectedPayload { .. } => "decode_record_has_payload_and_takes_none",
            Self::UndefinedOrdinal { .. } => "decode_field_ordinal_undefined",
            Self::CountOverruns { .. } => "decode_count_overruns_record",
        }
    }

    #[must_use]
    pub const fn rail(self) -> Rail {
        match self {
            Self::Short { rail, .. }
            | Self::UnknownOpcode { rail, .. }
            | Self::Unjudged { rail, .. }
            | Self::RefusedByContract { rail, .. }
            | Self::UnexpectedPayload { rail, .. }
            | Self::UndefinedOrdinal { rail, .. }
            | Self::CountOverruns { rail, .. } => rail,
        }
    }

    #[must_use]
    pub const fn opcode(self) -> u32 {
        match self {
            Self::Short { opcode, .. }
            | Self::UnknownOpcode { opcode, .. }
            | Self::Unjudged { opcode, .. }
            | Self::RefusedByContract { opcode, .. }
            | Self::UnexpectedPayload { opcode, .. }
            | Self::UndefinedOrdinal { opcode, .. }
            | Self::CountOverruns { opcode, .. } => opcode,
        }
    }
}

/// Every refusal here is one check, and the fields are the numbers the check
/// saw.
///
/// The slug is [`DecodeRefusal::reason`] rather than a second set of strings —
/// a layer that may not depend on `observe` still has to name the refusal it
/// forwards, which is why `reason` is inherent, and the two spellings drifting
/// apart would mean a reader greps a name the log does not carry.
///
/// The rail and the opcode are on every variant because a refusal without them
/// says a record on some encoder somewhere did not lift, which is not a report
/// anyone can act on: the same opcode is a different record on a different
/// rail, and that is the first thing a reader has to know.
impl reims_vgpu_observe::Decline for DecodeRefusal {
    fn slug(&self) -> &'static str {
        self.reason()
    }

    fn fields(&self) -> alloc::vec::Vec<(&'static str, alloc::string::String)> {
        let mut f = alloc::vec![
            ("rail", alloc::format!("{:?}", self.rail())),
            ("opcode", alloc::format!("{:#x}", self.opcode())),
        ];
        match *self {
            Self::Short { have, need, .. } => {
                f.push(("have", alloc::string::ToString::to_string(&have)));
                f.push(("need", alloc::string::ToString::to_string(&need)));
            }
            Self::UnexpectedPayload { have, .. } => {
                f.push(("have", alloc::string::ToString::to_string(&have)))
            }
            Self::UndefinedOrdinal { field, value, .. } => {
                f.push(("field", alloc::string::ToString::to_string(&field)));
                f.push(("value", alloc::string::ToString::to_string(&value)));
            }
            Self::CountOverruns { count, have, .. } => {
                f.push(("count", alloc::string::ToString::to_string(&count)));
                f.push(("have", alloc::string::ToString::to_string(&have)));
            }
            // The three opcode judgements carry no third number: what happened
            // is entirely which of them it is, and the rail and opcode above
            // are the whole of the evidence.
            Self::UnknownOpcode { .. } | Self::Unjudged { .. } | Self::RefusedByContract { .. } => {
            }
        }
        f
    }
}

/// The refusal for an opcode this rail lifts no record for.
///
/// Three answers, and the difference is what a reader needs. An opcode the
/// ledger settled as [`crate::closure::Closure::Refused`] is refused *by
/// contract*: nothing is missing and nothing is to be built. One the ledger has
/// a row for but has not settled is unjudged, and the row says what is not yet
/// known. One with no row at all is a stream that has gone wrong, or a
/// serializer this device has never seen.
///
/// Collapsing the first two is the mistake worth naming: it would put a
/// deliberate refusal back on the work queue every time someone read the logs,
/// and it would let a genuinely open contract hide behind "we meant to do
/// that".
pub fn no_record(rail: Rail, opcode: u32) -> DecodeRefusal {
    match crate::closure::find(rail, opcode).map(|row| row.closure) {
        Some(crate::closure::Closure::Refused { .. }) => {
            DecodeRefusal::RefusedByContract { rail, opcode }
        }
        Some(_) => DecodeRefusal::Unjudged { rail, opcode },
        None => DecodeRefusal::UnknownOpcode { rail, opcode },
    }
}

/// Map a wire view error onto this layer's refusal.
///
/// The wire crate's error says a view did not fit; this layer's says which
/// record on which rail. Both facts are needed and neither is the other's.
pub(crate) fn short(rail: Rail, opcode: u32, have: usize, need: usize) -> DecodeRefusal {
    DecodeRefusal::Short {
        rail,
        opcode,
        have,
        need,
    }
}

/// Why a counted record whose count leads its head did not fit.
///
/// Two failures wear one `Err` in the wire crate, and they are not the same
/// news. A payload shorter than the head never carried a count at all, so the
/// honest report is how many bytes arrived against how many the head needs.
/// A payload long enough for the head that still did not fit is an array that
/// overran, and there the count is the fact worth having: "the guest asked for
/// two hundred" beside "the record held twelve" is the pair that says which of
/// the two is wrong.
///
/// Written once because the residency and barrier decoders both had it, under
/// two different names — so a reader could not tell they were one rule, and a
/// third counted record would have been a third copy. The count leads the head
/// in every record this serves; [`render::counted_at`] is the form for the
/// records that put a `first` in front of theirs.
pub(crate) fn counted_head(rail: Rail, op: &Op<'_>, head_len: usize) -> DecodeRefusal {
    let have = op.payload.len();
    if have < head_len {
        return short(rail, op.opcode(), have, head_len);
    }
    let mut count = [0u8; 4];
    count.copy_from_slice(&op.payload[..4]);
    DecodeRefusal::CountOverruns {
        rail,
        opcode: op.opcode(),
        count: u32::from_le_bytes(count),
        have,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every refusal reason is distinct, so a log line says which shape was
    /// seen.
    #[test]
    fn refusal_reasons_are_distinct() {
        let all = [
            DecodeRefusal::Short {
                rail: Rail::Blit,
                opcode: 1,
                have: 0,
                need: 1,
            },
            DecodeRefusal::UnknownOpcode {
                rail: Rail::Blit,
                opcode: 1,
            },
            DecodeRefusal::Unjudged {
                rail: Rail::Blit,
                opcode: 1,
            },
            DecodeRefusal::RefusedByContract {
                rail: Rail::Blit,
                opcode: 1,
            },
            DecodeRefusal::UnexpectedPayload {
                rail: Rail::Blit,
                opcode: 1,
                have: 4,
            },
            DecodeRefusal::UndefinedOrdinal {
                rail: Rail::Blit,
                opcode: 1,
                field: "field",
                value: 9,
            },
            DecodeRefusal::CountOverruns {
                rail: Rail::Blit,
                opcode: 1,
                count: 2,
                have: 3,
            },
        ];
        let mut seen: alloc::vec::Vec<&str> = all.iter().map(|r| r.reason()).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before);
        for refusal in all {
            assert_eq!(refusal.rail(), Rail::Blit);
            assert_eq!(refusal.opcode(), 1);
        }
    }
}
