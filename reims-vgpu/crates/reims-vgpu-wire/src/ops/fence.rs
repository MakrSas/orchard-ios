//! Opcode 13 — create fence.
//!
//! `-[PGSerializer newFenceWithAllocator:]`, which takes no descriptor at all.
//!
//! # Layout
//!
//! Total 12 bytes: the 8-byte [`crate::op::OpHeader`] then a 4-byte payload.
//!
//! ```text
//! payload +000  u32  object_ref
//! ```
//!
//! Every byte is written. That is the finding rather than a formality: the
//! selector has no argument but the allocator, so the shortest record this
//! serializer emits is exactly a ref, and any future build that made it longer
//! would be carrying something a guest could set and this crate could not see.
//! `tests::the_record_is_a_header_and_a_ref_and_nothing_else`
//! is what would say so.
//!
//! # What reads it today
//!
//! Nothing. `reims-vgpu` tracks fences as generation counters keyed by the raw
//! ref — `runtime::fence_exec` — and the encoder-side update and wait records
//! carry that ref directly, so the device never needs the creation record to
//! learn anything. It is covered here because absence of a consumer is not
//! absence of the command: a guest issues it, and a crate that claims to read
//! everything the serializer emits has to read this too.

use crate::le::U32le;
use crate::op::Op;
use crate::view::{view, Wire, WireError};

/// Opcode for fence creation, observed on
/// `-[PGSerializer newFenceWithAllocator:]`.
pub const OPCODE_NEW_FENCE: u32 = 13;

/// Total wire length of a fence-creation operation, header included.
pub const NEW_FENCE_TOTAL_LEN: u32 = 12;

/// Payload of a fence-creation record.
#[repr(C)]
#[derive(Debug)]
pub struct NewFenceBody {
    /// Ref the guest's object-ref allocator assigned to the new fence. Every
    /// later update, wait and delete names the fence by this value.
    pub object_ref: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for NewFenceBody {}

/// View the payload of a fence-creation record.
///
/// Refuses a record whose opcode is not [`OPCODE_NEW_FENCE`]; the caller is
/// expected to have dispatched on opcode already.
pub fn new_fence<'a>(op: &Op<'a>) -> Result<&'a NewFenceBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_NEW_FENCE);
    view::<NewFenceBody>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op, OP_HEADER_LEN};
    use core::mem::size_of;

    fn synth(object_ref: u32) -> [u8; 12] {
        let mut b = [0xAAu8; NEW_FENCE_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_NEW_FENCE.to_le_bytes());
        b[4..8].copy_from_slice(&NEW_FENCE_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&object_ref.to_le_bytes());
        b
    }

    #[test]
    fn the_record_is_a_header_and_a_ref_and_nothing_else() {
        assert_eq!(
            size_of::<NewFenceBody>() + OP_HEADER_LEN,
            NEW_FENCE_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<NewFenceBody>(), 1);
    }

    #[test]
    fn the_ref_reads_back_from_the_first_payload_word() {
        let buf = synth(0x2f);
        let o = op(&buf, 0).expect("well formed");
        assert_eq!(new_fence(&o).expect("fits").object_ref.get(), 0x2f);
    }

    #[test]
    fn a_truncated_fence_operation_is_refused_rather_than_read_short() {
        let buf = synth(1);
        let o = op(&buf, 0).expect("well formed");
        let short = Op {
            header: o.header,
            payload: &o.payload[..2],
            offset: 0,
        };
        assert!(matches!(
            new_fence(&short),
            Err(WireError::Short { need: 4, have: 2 })
        ));
    }
}
