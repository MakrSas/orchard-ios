//! Opcode 0x15 — create a texture placed inside a heap.
//!
//! `-[PGSerializer newTextureWithDescriptor:heap:offset:useOffset:allocator:]`.
//!
//! # Layout
//!
//! Total 60 bytes: the 8-byte [`crate::op::OpHeader`] then a 52-byte payload.
//!
//! ```text
//! payload +000  u32  object_ref
//! payload +004  u32  heap_ref
//! payload +008  32 bytes  the texture descriptor, `ops::texture`'s
//! payload +040  u8   use_offset  bit 0 only; bits [7:1] and +041..+043 unwritten
//! payload +044  u64  offset
//! ```
//!
//! The 32-byte middle is [`crate::ops::texture::TextureDescriptorBody`], the
//! same declaration `newTextureWithDescriptor:allocator:` uses after its own
//! ref — not a second reading of a similar record. Both records' fixtures run
//! through the same accessors, so a layout change fails on both rather than
//! leaving one reader right.
//!
//! # `use_offset` is one bit, and that has cost this project a texture
//!
//! It occupies a four-byte slot and the serializer writes **bit 0 of the first
//! byte and nothing else**. The other 31 bits are the guest's stale ring.
//!
//! That is measured, not inferred: under a `0xAA` arena fill the byte reads
//! `0xab` with the flag set and `0xaa` without, which is indistinguishable from
//! "the whole byte is written, and its constant part happens to equal the fill".
//! The oracle's second fill separates the two — see the crate's `AGENTS.md`.
//!
//! It matters because `reims-vgpu`'s `compute_exec` read that slot with a
//! 32-bit load and declined the whole texture when the result exceeded 1. On a
//! host arena the bytes above the flag are zero and the check passes; on a real
//! wire they are whatever the ring last held, so the bind was lost whenever
//! they were not.
//!
//! # How the layout was derived
//!
//! Perturbation across three cases: the offset moved between two distinctive
//! values, and `useOffset` moved between `YES` and `NO` with everything else
//! held. The embedded descriptor is derived in [`crate::ops::texture`].

use crate::le::{U32le, U64le};
use crate::op::Op;
use crate::ops::texture::TextureDescriptorBody;
use crate::view::{view, Wire, WireError};

/// Opcode for heap-placed texture creation, observed on
/// `-[PGSerializer newTextureWithDescriptor:heap:offset:useOffset:allocator:]`.
pub const OPCODE_NEW_HEAP_TEXTURE: u32 = 0x15;

/// Total wire length of a heap-texture-creation operation, header included.
pub const NEW_HEAP_TEXTURE_TOTAL_LEN: u32 = 60;

/// Payload of a heap-texture-creation record.
#[repr(C)]
#[derive(Debug)]
pub struct NewHeapTextureBody {
    /// Ref the guest's object-ref allocator assigned to the new texture.
    pub object_ref: U32le,
    /// Ref of the heap the texture is placed in. Observed: the oracle's stub
    /// heap answers `6565` to `-heapRef`, and `6565` is what lands here.
    pub heap_ref: U32le,
    /// The descriptor, byte for byte the one a plain creation carries.
    pub desc: TextureDescriptorBody,
    /// Bit 0 is `useOffset`; bits `[7:1]` are **not written**. Prefer
    /// [`NewHeapTextureBody::use_offset`].
    pub use_offset_bits: u8,
    /// Never written by the serializer; the guest's stale ring on a real wire.
    /// Named so nothing reads them by reaching past `use_offset_bits`.
    pub unwritten_after_use_offset: [u8; 3],
    /// Byte offset into the heap. Observed: `0x01234ab0` and `0x777000`, both
    /// verbatim, with all eight bytes written.
    ///
    /// Meaningful only when [`NewHeapTextureBody::use_offset`] is set — the
    /// serializer writes the offset either way, so a reader that ignores the
    /// flag places the texture where the guest did not ask.
    pub offset: U64le,
}

// SAFETY: `le` scalars, a `Wire` struct, a `u8` and a `[u8; 3]`, all align-1
// and valid for every byte pattern.
unsafe impl Wire for NewHeapTextureBody {}

impl NewHeapTextureBody {
    /// Whether the guest asked for the texture to be placed at
    /// [`NewHeapTextureBody::offset`].
    ///
    /// Observed: `YES` → the byte's bit 0 set, `NO` → clear, with every other
    /// bit of the four-byte slot unwritten in both. See the module doc.
    #[inline]
    pub fn use_offset(&self) -> bool {
        self.use_offset_bits & 0x1 != 0
    }
}

/// The same record under `-setSupportsTextureDescriptor2:`.
pub const OPCODE_NEW_HEAP_TEXTURE_WIDE: u32 = 0x38;

pub const NEW_HEAP_TEXTURE_WIDE_TOTAL_LEN: u32 = 68;

/// Payload of a heap-texture-creation record, wide descriptor.
///
/// Field for field [`NewHeapTextureBody`] with
/// [`crate::ops::texture::WideTextureDescriptorBody`] in the middle — including
/// the `useOffset` bit and its three unwritten neighbours, which are measured
/// here rather than carried over: `heap_texture_wide` and
/// `heap_texture_wide_no_offset` drive both values of the flag, because a bit
/// read as a four-byte word was a real bug on this record's narrow form.
#[repr(C)]
#[derive(Debug)]
pub struct NewHeapTextureWideBody {
    pub object_ref: U32le,
    pub heap_ref: U32le,
    pub desc: crate::ops::texture::WideTextureDescriptorBody,
    /// Bit 0 is `useOffset`; bits `[7:1]` are not written. Prefer
    /// [`NewHeapTextureWideBody::use_offset`].
    pub use_offset_bits: u8,
    /// Never written by the serializer.
    pub unwritten_after_use_offset: [u8; 3],
    pub offset: U64le,
}

// SAFETY: `le` scalars, a `Wire` struct, a `u8` and a `[u8; 3]`, all align-1
// and valid for every byte pattern.
unsafe impl Wire for NewHeapTextureWideBody {}

impl NewHeapTextureWideBody {
    /// Whether the guest asked for the texture to be placed at
    /// [`NewHeapTextureWideBody::offset`].
    #[inline]
    pub fn use_offset(&self) -> bool {
        self.use_offset_bits & 0x1 != 0
    }
}

/// View the payload of a heap-texture-creation record.
///
/// Refuses a record whose opcode is not [`OPCODE_NEW_HEAP_TEXTURE`]; the caller
/// is expected to have dispatched on opcode already.
pub fn new_heap_texture<'a>(op: &Op<'a>) -> Result<&'a NewHeapTextureBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_NEW_HEAP_TEXTURE);
    view::<NewHeapTextureBody>(op.payload)
}

pub fn new_heap_texture_wide<'a>(op: &Op<'a>) -> Result<&'a NewHeapTextureWideBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_NEW_HEAP_TEXTURE_WIDE);
    view::<NewHeapTextureWideBody>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op, OP_HEADER_LEN};
    use crate::ops::texture::TEXTURE_DESCRIPTOR_LEN;
    use core::mem::size_of;

    fn synth(heap_ref: u32, packed: u32, use_offset_byte: u8, offset: u64) -> [u8; 60] {
        let mut b = [0xAAu8; NEW_HEAP_TEXTURE_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_NEW_HEAP_TEXTURE.to_le_bytes());
        b[4..8].copy_from_slice(&NEW_HEAP_TEXTURE_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&48u32.to_le_bytes());
        b[12..16].copy_from_slice(&heap_ref.to_le_bytes());
        b[16..20].copy_from_slice(&packed.to_le_bytes());
        b[20..24].copy_from_slice(&0x1111u32.to_le_bytes());
        b[24..28].copy_from_slice(&0x2222u32.to_le_bytes());
        b[28..32].copy_from_slice(&1u32.to_le_bytes());
        b[32..34].copy_from_slice(&1u16.to_le_bytes());
        b[34..36].copy_from_slice(&1u16.to_le_bytes());
        b[36..38].copy_from_slice(&1u16.to_le_bytes());
        b[38..40].copy_from_slice(&0x0020u16.to_le_bytes());
        b[40..48].copy_from_slice(&0u64.to_le_bytes());
        b[48] = use_offset_byte;
        b[52..60].copy_from_slice(&offset.to_le_bytes());
        b
    }

    #[test]
    fn the_payload_is_exactly_the_record_minus_its_header() {
        assert_eq!(
            size_of::<NewHeapTextureBody>() + OP_HEADER_LEN,
            NEW_HEAP_TEXTURE_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<NewHeapTextureBody>(), 1);

        assert_eq!(
            size_of::<NewHeapTextureWideBody>() + OP_HEADER_LEN,
            NEW_HEAP_TEXTURE_WIDE_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<NewHeapTextureWideBody>(), 1);
        // The wide record is the narrow one plus the descriptor's eight bytes,
        // and nothing else moves: the flag and the offset keep their order and
        // their widths.
        assert_eq!(
            NEW_HEAP_TEXTURE_WIDE_TOTAL_LEN - NEW_HEAP_TEXTURE_TOTAL_LEN,
            8
        );
        assert_eq!(
            size_of::<NewHeapTextureWideBody>() - size_of::<NewHeapTextureBody>(),
            8
        );
    }

    #[test]
    fn the_embedded_descriptor_starts_where_the_heap_ref_ends() {
        // The offset arithmetic this record's tail depends on: two refs, then
        // the shared 32-byte body, then the flag. If any of the three moves,
        // `use_offset` and `offset` read the wrong bytes.
        assert_eq!(
            size_of::<U32le>() * 2 + TEXTURE_DESCRIPTOR_LEN,
            40,
            "the flag no longer sits at payload +40"
        );
    }

    #[test]
    fn the_record_reads_back_its_two_refs_the_descriptor_and_the_placement() {
        let buf = synth(6565, 0x0050_05c2, 0x01, 0x0123_4ab0);
        let o = op(&buf, 0).expect("well formed");
        let h = new_heap_texture(&o).expect("fits");

        assert_eq!(h.object_ref.get(), 48);
        assert_eq!(h.heap_ref.get(), 6565);
        assert_eq!(h.desc.texture_type(), 2);
        assert_eq!(h.desc.pixel_format(), 80);
        assert_eq!(h.desc.usage(), 5);
        assert_eq!(h.desc.width.get(), 0x1111);
        assert_eq!(h.desc.height.get(), 0x2222);
        assert_eq!(h.desc.storage_mode(), 2);
        assert!(h.use_offset());
        assert_eq!(h.offset.get(), 0x0123_4ab0);
    }

    #[test]
    fn use_offset_reads_one_bit_and_ignores_the_rest_of_its_slot() {
        // The bug this record taught: the three bytes after the flag, and the
        // seven bits above it, are the guest's ring. A reader that takes the
        // slot as a `u32` and range-checks it drops the texture whenever the
        // ring was not zero there.
        for (byte, expect) in [(0x00u8, false), (0x01, true), (0xfe, false), (0xff, true)] {
            let mut buf = synth(6565, 0x0050_05c2, byte, 0x777000);
            buf[49] = 0x5a;
            buf[50] = 0xa5;
            buf[51] = 0xff;
            let o = op(&buf, 0).expect("well formed");
            let h = new_heap_texture(&o).expect("fits");
            assert_eq!(h.use_offset(), expect, "byte {byte:#04x}");
            assert_eq!(h.offset.get(), 0x777000, "byte {byte:#04x}: offset moved");
        }
    }

    #[test]
    fn a_truncated_heap_texture_operation_is_refused_rather_than_read_short() {
        let buf = synth(1, 0, 0, 0);
        let o = op(&buf, 0).expect("well formed");
        let short = Op {
            header: o.header,
            payload: &o.payload[..40],
            offset: 0,
        };
        assert!(matches!(
            new_heap_texture(&short),
            Err(WireError::Short { need: 52, have: 40 })
        ));
    }
}
