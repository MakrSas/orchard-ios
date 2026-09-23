//! Align-1 little-endian scalars.
//!
//! These exist so wire structs can have `align_of() == 1`. A `#[repr(C)]`
//! struct holding a bare `u64` has align 8, and taking a `&Struct` from a
//! buffer offset that is not 8-aligned is undefined behaviour — not a
//! theoretical one here, because operations are variable length and the
//! smallest observed texture operation is **44 bytes**, which is not a
//! multiple of 8. Two of those back to back put the second operation's `u64`
//! on a 4-aligned address.
//!
//! Wrapping each scalar in its byte array sidesteps the whole class: every
//! type below is `#[repr(transparent)]` over `[u8; N]`, so alignment is 1 and
//! any byte offset is a legal place to put one.
//!
//! Both guest pathways (x86_64 and arm64 macOS) are little-endian, so the
//! encoding is fixed rather than configurable; naming it in the type keeps a
//! future big-endian reader from assuming host order.

use core::fmt;

macro_rules! le_scalar {
    ($name:ident, $prim:ty, $len:expr) => {
        #[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name([u8; $len]);

        impl $name {
            /// Read the scalar in host order.
            #[inline]
            pub const fn get(self) -> $prim {
                <$prim>::from_le_bytes(self.0)
            }

            /// Build one from a host-order value. Used by tests that
            /// synthesize buffers; the decode path never constructs these.
            #[inline]
            pub const fn new(value: $prim) -> Self {
                Self(value.to_le_bytes())
            }

            /// The underlying wire bytes, in wire order.
            #[inline]
            pub const fn bytes(self) -> [u8; $len] {
                self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                // Wire values are read against hex dumps far more often than
                // against decimal ones, so lead with hex and keep decimal for
                // the fields that are counts.
                write!(f, "{:#x} ({})", self.get(), self.get())
            }
        }

        impl From<$prim> for $name {
            #[inline]
            fn from(value: $prim) -> Self {
                Self::new(value)
            }
        }

        impl From<$name> for $prim {
            #[inline]
            fn from(value: $name) -> Self {
                value.get()
            }
        }

        // SAFETY: `#[repr(transparent)]` over `[u8; $len]`, so the alignment is
        // 1 and every byte pattern of that length is a valid value.
        unsafe impl crate::view::Wire for $name {}
    };
}

le_scalar!(U16le, u16, 2);
le_scalar!(U32le, u32, 4);
le_scalar!(U64le, u64, 8);

// Signed scalars, for the fields Apple's API declares `NSInteger`. There is
// exactly one so far — a draw's `baseVertex`, which Metal documents as signed
// and which the serializer writes as two's complement in both of its widths
// (see `ops::render::DrawIndexedInstancedBase`). Reading one through `U16le`
// would turn a small negative offset into a value near 65535, which is a
// plausible index count and therefore a silent wrong draw rather than an
// obvious one.
le_scalar!(I16le, i16, 2);
le_scalar!(I64le, i64, 8);

/// Floats get their own macro: `Eq`/`Ord`/`Hash` stay bitwise (they are derived
/// on the byte array, not on the float, so `NaN != NaN` does not apply and two
/// wire values compare equal exactly when their bytes do), and `Debug` prints
/// the value rather than hex.
macro_rules! le_float {
    ($name:ident, $prim:ty, $len:expr) => {
        #[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name([u8; $len]);

        impl $name {
            #[inline]
            pub const fn get(self) -> $prim {
                <$prim>::from_le_bytes(self.0)
            }

            #[inline]
            pub const fn new(value: $prim) -> Self {
                Self(value.to_le_bytes())
            }

            #[inline]
            pub const fn bytes(self) -> [u8; $len] {
                self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.get())
            }
        }

        // SAFETY: `#[repr(transparent)]` over `[u8; $len]`, so alignment is 1.
        // Every byte pattern is a valid IEEE-754 value of this width — signaling
        // NaNs included, which is why the bytes are converted on read rather
        // than a `$prim` being stored in the struct.
        unsafe impl crate::view::Wire for $name {}
    };
}

le_float!(F32le, f32, 4);
le_float!(F64le, f64, 8);

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn every_le_scalar_is_align_one_and_exactly_its_wire_width() {
        // The whole point of the module. If this ever fails, `view()` is
        // handing out misaligned references.
        assert_eq!(align_of::<U16le>(), 1);
        assert_eq!(align_of::<U32le>(), 1);
        assert_eq!(align_of::<U64le>(), 1);
        assert_eq!(size_of::<U16le>(), 2);
        assert_eq!(size_of::<U32le>(), 4);
        assert_eq!(size_of::<U64le>(), 8);
    }

    #[test]
    fn scalars_read_little_endian_regardless_of_host_order() {
        assert_eq!(U32le([0x2c, 0x00, 0x00, 0x00]).get(), 44);
        assert_eq!(U16le([0x34, 0x12]).get(), 0x1234);
        assert_eq!(U64le([1, 0, 0, 0, 0, 0, 0, 0]).get(), 1);
        assert_eq!(U32le::new(0x1111).bytes(), [0x11, 0x11, 0x00, 0x00]);
    }
}
