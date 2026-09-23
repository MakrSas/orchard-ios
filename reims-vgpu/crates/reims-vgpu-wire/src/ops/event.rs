//! The event encoder's records — signal, wait, and wait-with-timeout.
//!
//! # No oracle backs this module
//!
//! Every other module here is pinned by bytes a serializer emitted. This one
//! is not: the event records reach the device on a segment family the userspace
//! serializer this crate drives does not produce, so no case can be added for
//! them and no fixture can hold them. The layout below comes from the device
//! contract instead, which puts the module in the crate's Tier 2 — the same
//! tier as [`crate::page_table`] — and the rule for that tier is to say so
//! rather than to let a reader assume the usual ground truth.
//!
//! What that changes in practice: the unit tests here synthesize buffers from
//! these constants, so they prove the arithmetic and prove nothing about the
//! layout. There is no manifest row, because the manifest is selector-indexed
//! and these records have no selector in the inventory.
//!
//! # Layout
//!
//! Two shapes over three opcodes. Signal and wait share theirs:
//!
//! ```text
//! payload +000  u32  event_ref
//! payload +004  u64  value
//! ```
//!
//! and the timeout form appends one field:
//!
//! ```text
//! payload +012  u32  timeout
//! ```
//!
//! The `u64` sits at `+4`, so this record is align-1 like every other and the
//! `le` scalars are what make that safe to view rather than to copy out.
//!
//! # The three opcodes are a window, and the window is the point
//!
//! `0x190`, `0x191`, `0x192` are contiguous, and the device's event
//! deserializer accepts exactly those three. What it refuses matters as much:
//! the blit encoder's fence pair is `0x13c`/`0x13d`, real opcodes in another
//! space, so an event reader that dispatched on "is it in some range" rather
//! than on the exact opcode would read a blit fence as an event signal.
//!
//! The meaning of each opcode is not assigned here — this crate reads bytes.
//! `reims_vgpu_protocol::sync` names which of the three is a signal and which
//! is a wait, and refuses the timeout form for the reason its ledger records.

use crate::le::{U32le, U64le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

/// Wait until an event reaches a value.
pub const OPCODE_WAIT_EVENT: u32 = 0x190;
/// Signal an event to a value.
pub const OPCODE_SIGNAL_EVENT: u32 = 0x191;
/// Wait until an event reaches a value, giving up after a timeout.
pub const OPCODE_WAIT_EVENT_TIMEOUT: u32 = 0x192;

/// Total wire length of a signal or a plain wait, header included.
pub const SIGNAL_WAIT_TOTAL_LEN: u32 = 20;
/// Total wire length of a wait-with-timeout, header included.
pub const WAIT_TIMEOUT_TOTAL_LEN: u32 = 24;

/// An event and the value the record names on it.
#[repr(C)]
#[derive(Debug)]
pub struct SignalWait {
    /// The guest's ref for the event object.
    pub event_ref: U32le,
    /// The generation being signalled, or waited for.
    pub value: U64le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for SignalWait {}

/// The same head with the timeout the third opcode carries.
///
/// A separate type rather than an `Option` field on [`SignalWait`]: the two
/// records have different lengths, and a view is a length check over the bytes
/// that are there. The head is repeated by value rather than nested so that
/// the whole record stays one flat `#[repr(C)]` body with no padding question.
#[repr(C)]
#[derive(Debug)]
pub struct WaitTimeout {
    pub event_ref: U32le,
    pub value: U64le,
    /// The wait's deadline, in whatever unit the contract fixes. Nothing here
    /// interprets it; the field is 32 bits where the two beside it are 32 and
    /// 64, which is the only claim this crate makes about it.
    pub timeout: U32le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for WaitTimeout {}

/// Whether an opcode carries the [`SignalWait`] shape.
#[inline]
#[must_use]
pub fn is_signal_wait(opcode: u32) -> bool {
    matches!(opcode, OPCODE_WAIT_EVENT | OPCODE_SIGNAL_EVENT)
}

/// View a signal or plain-wait record.
pub fn signal_wait<'a>(op: &Op<'a>) -> Result<&'a SignalWait, WireError> {
    debug_assert!(is_signal_wait(op.opcode()));
    view::<SignalWait>(op.payload)
}

/// View a wait-with-timeout record.
pub fn wait_timeout<'a>(op: &Op<'a>) -> Result<&'a WaitTimeout, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_WAIT_EVENT_TIMEOUT);
    view::<WaitTimeout>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OP_HEADER_LEN;

    /// Both bodies plus the shared header are the lengths this module declares.
    #[test]
    fn the_bodies_and_the_header_are_the_declared_lengths() {
        assert_eq!(
            core::mem::size_of::<SignalWait>() + OP_HEADER_LEN,
            SIGNAL_WAIT_TOTAL_LEN as usize
        );
        assert_eq!(
            core::mem::size_of::<WaitTimeout>() + OP_HEADER_LEN,
            WAIT_TIMEOUT_TOTAL_LEN as usize
        );
    }

    /// The timeout form is the shorter one plus exactly its extra field, and
    /// the shared fields sit at the same offsets in both. A timeout record read
    /// through the shorter view has to yield the same event and value.
    #[test]
    fn the_timeout_form_extends_the_shorter_one_rather_than_reordering_it() {
        assert_eq!(
            core::mem::offset_of!(SignalWait, event_ref),
            core::mem::offset_of!(WaitTimeout, event_ref)
        );
        assert_eq!(
            core::mem::offset_of!(SignalWait, value),
            core::mem::offset_of!(WaitTimeout, value)
        );
        assert_eq!(
            core::mem::size_of::<WaitTimeout>() - core::mem::size_of::<SignalWait>(),
            core::mem::size_of::<U32le>()
        );
    }

    /// The three opcodes are one contiguous window, and the blit encoder's
    /// fence pair is outside it. Written as a comparison against the other
    /// module's constants rather than as literals, so a renumbering there
    /// cannot leave this claim quietly false.
    #[test]
    fn the_event_window_is_contiguous_and_holds_no_blit_fence() {
        assert_eq!(OPCODE_SIGNAL_EVENT, OPCODE_WAIT_EVENT + 1);
        assert_eq!(OPCODE_WAIT_EVENT_TIMEOUT, OPCODE_SIGNAL_EVENT + 1);
        for foreign in [
            crate::ops::blit::OPCODE_UPDATE_FENCE,
            crate::ops::blit::OPCODE_WAIT_FOR_FENCE,
        ] {
            assert!(!(OPCODE_WAIT_EVENT..=OPCODE_WAIT_EVENT_TIMEOUT).contains(&foreign));
        }
    }

    /// A record one byte short of its body is refused rather than read.
    #[test]
    fn a_short_record_is_refused() {
        let short = [0u8; core::mem::size_of::<SignalWait>() - 1];
        assert!(view::<SignalWait>(&short).is_err());
        let short = [0u8; core::mem::size_of::<WaitTimeout>() - 1];
        assert!(view::<WaitTimeout>(&short).is_err());
    }
}
