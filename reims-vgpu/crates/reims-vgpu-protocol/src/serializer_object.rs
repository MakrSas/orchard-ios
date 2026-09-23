//! The four-word header every serializer-object record starts with.
//!
//! # One header, two readers, and a relation only one of them could state
//!
//! A serializer object — a render pipeline, a compute pipeline, a sampler, a
//! depth-stencil state, an indirect command buffer — arrives as a record whose
//! first sixteen bytes are the same four words whatever the object is:
//!
//! ```text
//! 0x00  u32  object type
//! 0x04  u32  declared length — the whole record, header included
//! 0x08  u32  object id
//! 0x0c  u32  serialized payload length
//! ```
//!
//! The two pipeline decoders each read those four offsets with their own
//! literals and each spelled the same two checks — the type it expected, and
//! `declared == bytes.len()`. The compute one's own comment said so: *the same
//! four-word header its render sibling carries, so the same relation between its
//! two lengths holds.* A relation named in a comment beside a second copy of the
//! arithmetic is the shape this module exists to remove.
//!
//! # The header's length is where the body starts
//!
//! [`HEADER_LEN`] is not only a floor. For the property-list subtypes it is the
//! offset the first TLV sits at, and for the fixed-layout subtypes it is where
//! their struct begins — so a caller that walks a body derives its base from
//! here rather than writing sixteen again. The smallest record that can carry
//! anything at all is [`MIN_RECORD_LEN`], which is that plus one byte, and which
//! was a bare `17` with no derivation behind it.
//!
//! # What this does not decide
//!
//! Which types exist. The object type is compared against one the caller names,
//! because the set of subtypes is a table of opcodes and this crate deliberately
//! keeps no second copy of one — the same rule [`crate::fifo`] states.

use crate::endian::ld32;

/// Byte offset of the object type.
pub const OBJECT_TYPE: usize = 0x00;
/// Byte offset of the declared length: the whole record, header included.
pub const DECLARED_LEN: usize = OBJECT_TYPE + 4;
/// Byte offset of the object id.
pub const OBJECT_ID: usize = DECLARED_LEN + 4;
/// Byte offset of the serialized payload length.
///
/// **Not the same number as [`DECLARED_LEN`]'s word.** The declared length
/// covers the header and the payload padded to four bytes; this one is the same
/// payload unpadded. A record where
/// `declared != HEADER_LEN + round_up_4(payload)` is one whose two halves were
/// written by different ideas of how long it is.
pub const SERIALIZED_PAYLOAD_LEN: usize = OBJECT_ID + 4;
/// Bytes the header occupies, and therefore where the record's body starts.
pub const HEADER_LEN: usize = SERIALIZED_PAYLOAD_LEN + 4;
/// The smallest record that carries a body at all.
pub const MIN_RECORD_LEN: usize = HEADER_LEN + 1;

/// A serializer object's header, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub object_type: u32,
    /// The whole record's length, header included. Equal to the payload the
    /// caller handed in, which is what [`decode`] checks.
    pub declared: u32,
    pub object_id: u32,
    /// The payload's length unpadded. See [`SERIALIZED_PAYLOAD_LEN`].
    pub serialized_payload_len: u32,
}

/// Why a record's header is not one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Too short to hold the header and a body.
    Short { plen: usize },
    /// The record is not the type the caller asked for.
    ///
    /// Checked here rather than after the length, because a record of the wrong
    /// type is a routing mistake and a record of the right type with a bad
    /// length is a malformed one — and a caller that reported the second for the
    /// first would send a reader looking for corruption in a record that was
    /// merely handed to the wrong decoder.
    WrongType { expected: u32, found: u32 },
    /// The declared length is not the length of the bytes that arrived.
    ///
    /// Both directions are refused. Shorter means the record claims less than
    /// it was given, so the tail belongs to something else; longer means the
    /// walk that follows would read past the end.
    DeclaredLength { declared: u32, plen: usize },
}

/// Decode the header of a record the caller expects to be `expected_type`.
///
/// # Errors
///
/// [`Refusal`]: too short for a header and a body, the wrong object type, or a
/// declared length that disagrees with the bytes that arrived.
pub fn decode(bytes: &[u8], expected_type: u32) -> Result<Header, Refusal> {
    if bytes.len() < MIN_RECORD_LEN {
        return Err(Refusal::Short { plen: bytes.len() });
    }
    let object_type = ld32(&bytes[OBJECT_TYPE..]);
    if object_type != expected_type {
        return Err(Refusal::WrongType {
            expected: expected_type,
            found: object_type,
        });
    }
    let declared = ld32(&bytes[DECLARED_LEN..]);
    if declared as usize != bytes.len() {
        return Err(Refusal::DeclaredLength {
            declared,
            plen: bytes.len(),
        });
    }
    Ok(Header {
        object_type,
        declared,
        object_id: ld32(&bytes[OBJECT_ID..]),
        serialized_payload_len: ld32(&bytes[SERIALIZED_PAYLOAD_LEN..]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endian::st32;

    fn record(object_type: u32, len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; len];
        st32(&mut bytes[OBJECT_TYPE..], object_type);
        st32(&mut bytes[DECLARED_LEN..], len as u32);
        st32(&mut bytes[OBJECT_ID..], 0x42);
        st32(
            &mut bytes[SERIALIZED_PAYLOAD_LEN..],
            (len - HEADER_LEN) as u32,
        );
        bytes
    }

    /// The four words, and the derivation that makes the floor a floor.
    #[test]
    fn the_header_is_four_words_and_the_body_starts_after_it() {
        assert_eq!(HEADER_LEN, 16);
        assert_eq!(MIN_RECORD_LEN, HEADER_LEN + 1, "a header and one body byte");
        let bytes = record(0x0e, 24);
        assert_eq!(
            decode(&bytes, 0x0e),
            Ok(Header {
                object_type: 0x0e,
                declared: 24,
                object_id: 0x42,
                serialized_payload_len: 8,
            })
        );
    }

    /// The wrong type and a wrong length are different failures, and the type
    /// is judged first.
    ///
    /// A record handed to the wrong decoder is a routing mistake; reporting it
    /// as a length disagreement sends a reader looking for corruption that is
    /// not there.
    #[test]
    fn a_record_of_another_type_is_not_reported_as_a_bad_length() {
        let mut bytes = record(0x0b, 24);
        // Both wrong at once: the type is not the one asked for *and* the
        // declared length disagrees.
        st32(&mut bytes[DECLARED_LEN..], 99);
        assert_eq!(
            decode(&bytes, 0x0e),
            Err(Refusal::WrongType {
                expected: 0x0e,
                found: 0x0b,
            })
        );
    }

    /// A declared length that disagrees with the bytes is refused in both
    /// directions.
    #[test]
    fn a_declared_length_must_be_the_length_that_arrived() {
        for declared in [23u32, 25] {
            let mut bytes = record(0x0e, 24);
            st32(&mut bytes[DECLARED_LEN..], declared);
            assert_eq!(
                decode(&bytes, 0x0e),
                Err(Refusal::DeclaredLength { declared, plen: 24 }),
                "declared={declared}"
            );
        }
    }

    /// A record with a header and no body carries nothing to walk.
    #[test]
    fn a_header_with_no_body_is_short() {
        let bytes = record(0x0e, HEADER_LEN);
        assert_eq!(
            decode(&bytes, 0x0e),
            Err(Refusal::Short { plen: HEADER_LEN })
        );
    }
}
