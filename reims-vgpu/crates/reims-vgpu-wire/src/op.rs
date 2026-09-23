//! Record framing.
//!
//! Every record the serializer emits begins with the same 8-byte head:
//!
//! ```text
//! +000  u32  opcode
//! +004  u32  length   (the whole record, this header included)
//! +008  ...  payload  (length - 8 bytes, per-opcode)
//! ```
//!
//! # How this was derived, and why it is 8 rather than 12
//!
//! `PGSerializerCommandStream` requires `-getCommandBufferBytes:(size_t)`, and
//! the serializer asks for exactly the bytes one record needs before writing
//! it. Driving a `PGSerializerRenderCommandEncoder` through a conforming object
//! shows the second word equal to the requested size on every record:
//!
//! ```text
//! setScissorRect:          alloc=40  opcode=0x75  length=40
//! setCullMode:             alloc=16  opcode=0x6b  length=16
//! setFrontFacingWinding:   alloc=16  opcode=0x73  length=16
//! setBlendColorRed:...     alloc=24  opcode=0x65  length=24
//! drawPrimitives:...       alloc=16  opcode=0x01  length=16
//! ```
//!
//! So the payload starts at 8. An earlier version of this module read a third
//! header word, because the first *payload* word of an object-creation record
//! is the new object's ref and a 12-byte reading fits those records too. It
//! does not fit encoder records, which have no ref and would have lost the
//! first four bytes of every payload.
//!
//! Those five opcodes match the encoder table decode uses from this crate
//! (`OPCODE_SET_SCISSOR`, `OPCODE_SET_CULL_MODE`, `OPCODE_SET_FRONT_FACING`,
//! `OPCODE_SET_BLEND_COLOR`, `OPCODE_DRAW`), and decode takes
//! [`OP_HEADER_LEN`] from here rather than restating 8. Agreement between the
//! serializer allocation sizes and the shared header constant is why this is
//! stated rather than proposed.
//!
//! # This is not the FIFO packet header
//!
//! `reims_vgpu::runtime::decode::fifo` frames a different level of the protocol
//! — opcode `u16`, stamp count `u16`, then a total size and a completion stamp.
//! Its fields sit at offsets 4 and 8 too, meaning different things. Do not port
//! constants between the two.

use crate::le::U32le;
use crate::view::{split, view, Wire, WireError};

/// Bytes before a record's payload.
pub const OP_HEADER_LEN: usize = 8;

/// The fixed head of every serializer record.
#[repr(C)]
#[derive(Debug)]
pub struct OpHeader {
    pub opcode: U32le,
    /// Whole record including this header, matching the serializer's own
    /// allocation request.
    pub length: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars, so the struct is align-1
// and every 8-byte pattern is a valid `OpHeader`.
unsafe impl Wire for OpHeader {}

/// One record: its header and the payload bytes that belong to it.
#[derive(Debug)]
pub struct Op<'a> {
    pub header: &'a OpHeader,
    pub payload: &'a [u8],
    /// Byte offset of the record within the stream it came from, so a failure
    /// can name where in the capture it happened.
    pub offset: usize,
}

impl Op<'_> {
    #[inline]
    pub fn opcode(&self) -> u32 {
        self.header.opcode.get()
    }

    #[inline]
    pub fn length(&self) -> u32 {
        self.header.length.get()
    }
}

/// View a single record at the start of `buf`.
///
/// `length` is guest-controlled, so it is checked against both the header size
/// (below which the record cannot contain its own header) and the bytes
/// actually present.
pub fn op(buf: &[u8], offset: usize) -> Result<Op<'_>, WireError> {
    let header = view::<OpHeader>(buf)?;
    let length = header.length.get();
    let length_usize = length as usize;
    if length_usize < OP_HEADER_LEN || length_usize > buf.len() {
        return Err(WireError::BadLength {
            opcode: header.opcode.get(),
            length,
            remaining: buf.len(),
        });
    }
    let (_, rest) = split::<OpHeader>(&buf[..length_usize])?;
    Ok(Op {
        header,
        payload: rest,
        offset,
    })
}

/// Walk a buffer of back-to-back records.
///
/// Yields a typed error rather than ending quietly when the stream stops making
/// sense, then stops — a truncated capture and a malformed one must not look
/// alike, and neither may look like a clean end.
pub struct OpStream<'a> {
    buf: &'a [u8],
    offset: usize,
    stopped: bool,
}

impl<'a> OpStream<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            offset: 0,
            stopped: false,
        }
    }

    /// Bytes consumed so far. After the iterator ends without error this equals
    /// the buffer length; anything less is trailing bytes no record claimed.
    #[inline]
    pub fn consumed(&self) -> usize {
        self.offset
    }
}

impl<'a> Iterator for OpStream<'a> {
    type Item = Result<Op<'a>, WireError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stopped || self.offset >= self.buf.len() {
            return None;
        }
        match op(&self.buf[self.offset..], self.offset) {
            Ok(o) => {
                self.offset += o.length() as usize;
                Some(Ok(o))
            }
            Err(e) => {
                self.stopped = true;
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    /// Build a record head. Tests synthesize their own bytes; see AGENTS.md,
    /// "Two kinds of test" — this proves the view layer, not the layout, which
    /// only the oracle can settle.
    fn synth(opcode: u32, length: u32, payload: &[u8]) -> [u8; 64] {
        let mut b = [0u8; 64];
        b[0..4].copy_from_slice(&opcode.to_le_bytes());
        b[4..8].copy_from_slice(&length.to_le_bytes());
        b[8..8 + payload.len()].copy_from_slice(payload);
        b
    }

    #[test]
    fn the_header_is_exactly_eight_bytes_and_align_one() {
        assert_eq!(size_of::<OpHeader>(), OP_HEADER_LEN);
        assert_eq!(core::mem::align_of::<OpHeader>(), 1);
    }

    #[test]
    fn a_record_payload_is_length_minus_the_header() {
        let buf = synth(0x75, 16, &[0xde, 0xad, 0xbe, 0xef, 0, 0, 0, 0]);
        let o = op(&buf, 0).expect("well formed");
        assert_eq!(o.opcode(), 0x75);
        assert_eq!(o.length(), 16);
        assert_eq!(o.payload.len(), 8);
        assert_eq!(&o.payload[..4], &[0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn a_length_below_the_header_is_refused_rather_than_underflowing() {
        // 7 would make `length - 8` wrap if it were done as arithmetic.
        let buf = synth(3, 7, &[]);
        assert!(matches!(
            op(&buf, 0),
            Err(WireError::BadLength {
                opcode: 3,
                length: 7,
                remaining: 64
            })
        ));
    }

    #[test]
    fn a_length_past_the_buffer_is_refused_rather_than_read() {
        let buf = synth(3, 4096, &[]);
        assert!(matches!(
            op(&buf, 0),
            Err(WireError::BadLength { length: 4096, .. })
        ));
    }

    #[test]
    fn a_stream_walks_records_of_unequal_length_back_to_back() {
        // 44 then 36: the second starts 4-aligned but not 8-aligned, which is
        // the case `le` exists to make safe.
        let mut buf = [0u8; 80];
        buf[0..4].copy_from_slice(&1u32.to_le_bytes());
        buf[4..8].copy_from_slice(&44u32.to_le_bytes());
        buf[44..48].copy_from_slice(&0x75u32.to_le_bytes());
        buf[48..52].copy_from_slice(&36u32.to_le_bytes());

        let mut s = OpStream::new(&buf);
        let a = s.next().expect("first").expect("well formed");
        assert_eq!((a.opcode(), a.offset, a.payload.len()), (1, 0, 36));
        let b = s.next().expect("second").expect("well formed");
        assert_eq!((b.opcode(), b.offset, b.payload.len()), (0x75, 44, 28));
        assert!(s.next().is_none());
        assert_eq!(s.consumed(), 80);
    }

    #[test]
    fn a_stream_reports_the_malformed_record_instead_of_ending_quietly() {
        let mut buf = [0u8; 32];
        buf[0..4].copy_from_slice(&1u32.to_le_bytes());
        buf[4..8].copy_from_slice(&16u32.to_le_bytes());
        // Second record claims more than the buffer holds.
        buf[16..20].copy_from_slice(&2u32.to_le_bytes());
        buf[20..24].copy_from_slice(&999u32.to_le_bytes());

        let items: [Result<_, _>; 2] = {
            let mut s = OpStream::new(&buf);
            [s.next().expect("first"), s.next().expect("second")]
        };
        assert!(items[0].is_ok());
        assert!(matches!(
            items[1],
            Err(WireError::BadLength { opcode: 2, .. })
        ));
    }
}
