//! What a bind record carries, shared by the encoders that carry it.
//!
//! # One vocabulary because the wire has one
//!
//! The render and compute encoders bind buffers, textures and samplers with the
//! *same entry layouts* at different opcodes — `reims-vgpu-wire` declares the
//! entry types once on the render module and the compute module re-uses them.
//! Two copies of these types in the model would be two chances to disagree
//! about what a slot holds, so there is one, and which table an entry lands in
//! is the operation's business rather than the entry's.
//!
//! # An unbound slot holds nothing
//!
//! A guest unbinds by naming no object, and every field here that can be
//! unbound is an `Option`. A resource id of zero is a resource, and a slot that
//! held one would order against it.

use crate::identity::ResourceId;
use crate::sync::ResourceSpan;

/// A window of a buffer an indirect record reads from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectSource {
    pub buffer: ResourceId,
    pub offset: u64,
}

/// A window of one of the transaction's bind-entry arenas.
///
/// Bind records carry a counted array, and the count is the guest's. The
/// entries live in the transaction's arena and the operation names a window, so
/// a bind of forty textures costs the same two `u32` as a bind of one.
pub type BindSpan = ResourceSpan;

/// One buffer slot's binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    /// `None` when the record bound nothing to the slot.
    ///
    /// A guest unbinds by naming no object, and the slot then holds nothing
    /// rather than holding a resource whose id happens to be zero.
    pub buffer: Option<ResourceId>,
    pub offset: u64,
    /// The attribute stride, for the records whose selector carries one.
    pub stride: Option<u64>,
}

/// A level-of-detail clamp, as the guest's bits.
///
/// The wire value is a 32-bit float and it is kept as its bit pattern rather
/// than as an `f32`. Two reasons, and the second is the load-bearing one: a
/// binding table has to be comparable to tell a redundant bind from a real one,
/// and `f32` equality says a NaN clamp differs from itself — so a guest that
/// bound one would produce a state change on every record, forever. The bits
/// compare as the guest wrote them.
///
/// Note the width: these are 32-bit where every viewport depth bound in this
/// protocol is 64-bit, and the two are not interchangeable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LodClamp(pub u32);

impl LodClamp {
    /// The value an executor passes to a host API.
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }

    #[must_use]
    pub const fn from_f32(value: f32) -> LodClamp {
        LodClamp(value.to_bits())
    }
}

/// One texture or sampler slot's binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectBinding {
    pub object: Option<ResourceId>,
    /// The level-of-detail clamps, for the sampler records that carry them.
    ///
    /// A pair rather than two fields so that "this record carried clamps" is
    /// one question.
    pub lod_clamps: Option<(LodClamp, LodClamp)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    /// A clamp compares as the guest's bits, so a NaN one is equal to itself
    /// and a redundant bind stays redundant.
    #[test]
    fn a_clamp_compares_as_bits_and_not_as_a_float() {
        let nan = LodClamp::from_f32(f32::NAN);
        assert_eq!(nan, nan);
        assert!(nan.to_f32().is_nan());
        assert_eq!(LodClamp::from_f32(0.25).to_f32(), 0.25);
        assert_ne!(LodClamp::from_f32(0.25), LodClamp::from_f32(0.75));
    }

    /// An unbound slot is not a resource whose id is zero.
    #[test]
    fn an_unbound_slot_holds_nothing() {
        let unbound = BufferBinding {
            buffer: None,
            offset: 0,
            stride: None,
        };
        let zero = BufferBinding {
            buffer: Some(res(0)),
            offset: 0,
            stride: None,
        };
        assert_ne!(unbound, zero);
    }
}
