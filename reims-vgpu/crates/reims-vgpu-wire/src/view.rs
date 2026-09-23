//! Checked, zero-copy views over guest bytes.
//!
//! A view is a length check followed by a pointer cast. It never copies, never
//! allocates, and never walks fields — reading a field is a load through the
//! reference at a compile-time-known offset.
//!
//! # The check is not parsing
//!
//! The bytes come from a virtual machine guest, so `length` and every other
//! length on the wire is attacker-controlled from this process's point of view.
//! An unchecked cast is a host out-of-bounds read driven by the guest, inside
//! the process that also holds QEMU's address space. Every constructor here is
//! therefore fallible, and the cost is one comparison per view — not the
//! field-by-field work the name "parsing" implies.

use core::fmt;
use core::mem::{align_of, size_of};

/// A type that may be viewed directly over wire bytes.
///
/// # Safety
///
/// Implementors must satisfy both of:
///
/// 1. **Every** byte pattern of `size_of::<Self>()` bytes is a valid `Self`.
///    Integers and arrays of them qualify. `bool`, `char`, references, and
///    `#[repr(u32)]` enums do **not** — an out-of-range guest value would be
///    an invalid value, which is undefined behaviour rather than a decode
///    error. Keep the raw scalar in the struct and expose a fallible accessor.
/// 2. `align_of::<Self>() == 1`, so no buffer offset can be misaligned. Build
///    structs out of [`crate::le`] scalars and this holds by construction;
///    [`Wire::ASSERT_ALIGN_1`] catches it at compile time if it does not.
pub unsafe trait Wire: Sized {
    /// Compile-time proof of requirement 2. Forced by every constructor, so a
    /// struct that picks up an over-aligned field fails to build rather than
    /// producing misaligned references at run time.
    #[doc(hidden)]
    const ASSERT_ALIGN_1: () = assert!(
        align_of::<Self>() == 1,
        "Wire types must be align-1; build the struct from crate::le scalars"
    );
}

/// Why a view could not be taken.
///
/// Carries the numbers a reader needs to tell a truncated capture from a
/// malformed one, because "it did not fit" alone cannot distinguish them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireError {
    /// The buffer is shorter than the type being viewed.
    Short { need: usize, have: usize },
    /// A view was requested at an offset past the end of the buffer.
    OutOfRange { offset: usize, len: usize },
    /// A record header declared a length that cannot be right.
    BadLength {
        opcode: u32,
        length: u32,
        remaining: usize,
    },
    /// A count times an element size overflowed `usize`.
    CountOverflow { count: usize, elem: usize },
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Short { need, have } => write!(f, "short: need {need}, have {have}"),
            Self::OutOfRange { offset, len } => {
                write!(f, "out of range: offset {offset} past len {len}")
            }
            Self::BadLength {
                opcode,
                length,
                remaining,
            } => write!(
                f,
                "bad length: opcode {opcode} declared {length}, {remaining} remaining"
            ),
            Self::CountOverflow { count, elem } => {
                write!(f, "count overflow: {count} x {elem}")
            }
        }
    }
}

/// View `T` at the start of `buf`.
#[inline]
pub fn view<T: Wire>(buf: &[u8]) -> Result<&T, WireError> {
    let () = T::ASSERT_ALIGN_1;
    let need = size_of::<T>();
    if buf.len() < need {
        return Err(WireError::Short {
            need,
            have: buf.len(),
        });
    }
    // SAFETY: the length is checked immediately above, and `T: Wire` promises
    // align-1 with every byte pattern valid, so the cast cannot produce a
    // misaligned reference or an invalid value.
    Ok(unsafe { &*(buf.as_ptr() as *const T) })
}

/// View `T` at `offset` bytes into `buf`.
#[inline]
pub fn view_at<T: Wire>(buf: &[u8], offset: usize) -> Result<&T, WireError> {
    let tail = buf.get(offset..).ok_or(WireError::OutOfRange {
        offset,
        len: buf.len(),
    })?;
    view::<T>(tail)
}

/// View `T` at the start of `buf` and return the bytes after it.
///
/// This is the shape most operations want: a fixed head followed by a
/// variable-length tail the caller interprets per opcode.
#[inline]
pub fn split<T: Wire>(buf: &[u8]) -> Result<(&T, &[u8]), WireError> {
    let head = view::<T>(buf)?;
    Ok((head, &buf[size_of::<T>()..]))
}

/// View `count` consecutive `T` at the start of `buf`.
#[inline]
pub fn view_slice<T: Wire>(buf: &[u8], count: usize) -> Result<&[T], WireError> {
    let () = T::ASSERT_ALIGN_1;
    let elem = size_of::<T>();
    let need = count
        .checked_mul(elem)
        .ok_or(WireError::CountOverflow { count, elem })?;
    if buf.len() < need {
        return Err(WireError::Short {
            need,
            have: buf.len(),
        });
    }
    // SAFETY: `need` bytes are present, and `T: Wire` promises align-1 with
    // every byte pattern valid, so `count` elements are all initialized and
    // legally aligned at 1-byte stride multiples of `elem`.
    Ok(unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const T, count) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::le::{U16le, U32le};

    #[repr(C)]
    #[derive(Debug, PartialEq)]
    struct Pair {
        a: U32le,
        b: U16le,
    }
    // SAFETY: both fields are align-1 all-bytes-valid `le` scalars.
    unsafe impl Wire for Pair {}

    #[test]
    fn a_view_reads_fields_without_copying_the_buffer() {
        let buf = [0x01, 0x02, 0x03, 0x04, 0xaa, 0xbb, 0xff];
        let p = view::<Pair>(&buf).expect("fits");
        assert_eq!(p.a.get(), 0x04030201);
        assert_eq!(p.b.get(), 0xbbaa);
        // The view aliases the caller's buffer rather than owning bytes.
        assert_eq!(p as *const Pair as *const u8, buf.as_ptr());
    }

    #[test]
    fn a_buffer_one_byte_too_short_is_refused_with_both_numbers() {
        let buf = [0u8; 5];
        assert_eq!(
            view::<Pair>(&buf),
            Err(WireError::Short { need: 6, have: 5 })
        );
    }

    #[test]
    fn a_view_survives_every_odd_offset_a_variable_length_stream_can_produce() {
        // The alignment hazard this crate exists to avoid: a 44-byte operation
        // leaves the next one on a 4-aligned but not 8-aligned address, and
        // nothing stops a 1- or 2-aligned one either.
        let buf = [0xffu8; 64];
        for offset in 0..16 {
            let p = view_at::<Pair>(&buf, offset).expect("fits at every offset");
            assert_eq!(p.a.get(), u32::MAX);
        }
    }

    #[test]
    fn split_hands_back_the_tail_for_the_caller_to_interpret() {
        let buf = [1, 0, 0, 0, 2, 0, 9, 9, 9];
        let (p, tail) = split::<Pair>(&buf).expect("fits");
        assert_eq!(p.a.get(), 1);
        assert_eq!(tail, &[9, 9, 9]);
    }

    #[test]
    fn view_slice_refuses_a_count_that_overflows_rather_than_wrapping() {
        let buf = [0u8; 16];
        assert_eq!(
            view_slice::<Pair>(&buf, usize::MAX),
            Err(WireError::CountOverflow {
                count: usize::MAX,
                elem: 6
            })
        );
        assert_eq!(view_slice::<Pair>(&buf, 2).expect("fits").len(), 2);
    }
}
