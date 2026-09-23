//! What a residency declaration says, and which part of it a device may ignore.
//!
//! # The question this module exists to make answerable
//!
//! `useResource:usage:stages:` and its family declare that a resource will be
//! reached **indirectly** — through an argument buffer, an indirect command
//! buffer, or a heap — and say how. A driver that pages resources needs the
//! declaration to keep the memory resident. A device that resolves every
//! binding explicitly per draw appears not to need it at all, and this project
//! has answered the whole family by doing nothing on that reasoning.
//!
//! That reasoning holds for exactly one half of the declaration and the counter
//! behind it could not tell the halves apart. A resource declared **read** is
//! one the guest expects the device to have available; if the device already
//! binds everything it draws with, nothing is owed. A resource declared
//! **write** is one the guest expects the *GPU* to modify through a path the
//! device did not bind — and if that write does not happen, the guest reads
//! back stale content from a resource it believes the GPU just produced. The
//! two are not the same kind of silence, and one merged residency counter
//! reports them as one number.
//!
//! So this module gives the declaration a type: the usage bits classify into
//! the answer that decides whether the no-op is sound, and the stage set says
//! which shader stages the guest named. Neither selects behavior on its own —
//! it is a rail's job to decide what to do about a write declaration — but a
//! rail cannot decide what it cannot read, and until this existed the fields
//! were discarded at decode.
//!
//! # Where the values come from
//!
//! `MTLResourceUsage` and `MTLRenderStages` are decoded API enumerations. The
//! wire carries them narrowed: the `stages:`-qualified `useResource` packs
//! usage and stages into 16 bits each sharing one word, and the unqualified
//! form widens usage to 32 bits with no stages at all. Both reach the same
//! types here, because the semantic question does not change with the width.

/// `MTLResourceUsage`, as the wire carries it.
///
/// A newtype rather than a bare integer because the bits are guest data: a
/// value with bits outside the declared set is a thing that has to be
/// representable and reportable, not a value to mask down into whichever
/// declared bits it happens to share.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ResourceUsage(pub u32);

impl ResourceUsage {
    /// The resource is read through the indirect path.
    pub const READ: u32 = 1 << 0;
    /// The resource is written through the indirect path.
    pub const WRITE: u32 = 1 << 1;
    /// Sampled. Deprecated in the API in favour of `READ`, and carried
    /// separately here for exactly that reason: a guest still spelling it is
    /// telling this project which SDK generation it was built against, which a
    /// value folded into `READ` could not say.
    pub const SAMPLE: u32 = 1 << 2;

    const DECLARED: u32 = Self::READ | Self::WRITE | Self::SAMPLE;

    /// Whether the declaration includes a GPU write.
    pub const fn writes(self) -> bool {
        self.0 & Self::WRITE != 0
    }

    /// Bits outside the declared set, or zero.
    pub const fn undeclared_bits(self) -> u32 {
        self.0 & !Self::DECLARED
    }

    /// The classification a rail acts on.
    pub const fn classify(self) -> UsageClass {
        if self.undeclared_bits() != 0 {
            UsageClass::Undeclared
        } else if self.writes() {
            UsageClass::Writes
        } else if self.0 == 0 {
            UsageClass::Empty
        } else {
            UsageClass::ReadOnly
        }
    }
}

/// What a residency declaration's usage bits amount to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsageClass {
    /// No usage bit at all. The record still names resources, so this is a
    /// declaration of residency without a declared access — legal, and not the
    /// same as a read.
    Empty,
    /// Read and/or sample. A device that binds everything it draws with owes
    /// nothing further.
    ReadOnly,
    /// A GPU write through a path the device did not bind. Ignoring this loses
    /// content the guest expects to read back.
    Writes,
    /// Bits the API does not declare. Not narrowed into the declared ones —
    /// a guest writing an unknown bit is not writing the bits it shares.
    Undeclared,
}

impl UsageClass {
    /// A stable one-word name for a census route or a fail line.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::ReadOnly => "read",
            Self::Writes => "write",
            Self::Undeclared => "undeclared",
        }
    }
}

/// `MTLRenderStages`, as the wire carries it.
///
/// The `stages:`-qualified residency selectors and the fence pair both carry
/// this. The unqualified residency forms carry none, which is [`Self::NONE`]
/// and is a fact about the selector rather than about the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RenderStages(pub u32);

impl RenderStages {
    /// The selector carried no stage argument.
    pub const NONE: RenderStages = RenderStages(0);
    pub const VERTEX: u32 = 1 << 0;
    pub const FRAGMENT: u32 = 1 << 1;
    pub const TILE: u32 = 1 << 2;
    pub const OBJECT: u32 = 1 << 3;
    pub const MESH: u32 = 1 << 4;

    const DECLARED: u32 = Self::VERTEX | Self::FRAGMENT | Self::TILE | Self::OBJECT | Self::MESH;

    /// Bits outside the declared set, or zero.
    pub const fn undeclared_bits(self) -> u32 {
        self.0 & !Self::DECLARED
    }

    /// Whether the guest named a stage this device has no encoder for. Tile,
    /// object and mesh stages all reach records this project drops elsewhere,
    /// so a residency declaration naming one is a second sighting of the same
    /// gap rather than a new one.
    pub const fn names_unexecuted_stage(self) -> bool {
        self.0 & (Self::TILE | Self::OBJECT | Self::MESH) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the type: a write declaration must not be able to
    /// hide inside a read one.
    #[test]
    fn a_write_declaration_classifies_apart_from_a_read_one() {
        assert_eq!(
            ResourceUsage(ResourceUsage::READ).classify(),
            UsageClass::ReadOnly
        );
        assert_eq!(
            ResourceUsage(ResourceUsage::SAMPLE).classify(),
            UsageClass::ReadOnly
        );
        assert_eq!(
            ResourceUsage(ResourceUsage::READ | ResourceUsage::SAMPLE).classify(),
            UsageClass::ReadOnly
        );
        assert_eq!(
            ResourceUsage(ResourceUsage::WRITE).classify(),
            UsageClass::Writes
        );
        assert_eq!(
            ResourceUsage(ResourceUsage::READ | ResourceUsage::WRITE).classify(),
            UsageClass::Writes
        );
        assert_eq!(ResourceUsage(0).classify(), UsageClass::Empty);
    }

    /// An undeclared bit is not its declared neighbours. Masking `0x9` down to
    /// `READ` would report a guest asking for something unknown as a guest
    /// asking for the one case this device is sure it owes nothing on.
    #[test]
    fn an_undeclared_bit_does_not_become_the_bits_it_shares() {
        let u = ResourceUsage(ResourceUsage::READ | 0x8);
        assert_eq!(u.classify(), UsageClass::Undeclared);
        assert_eq!(u.undeclared_bits(), 0x8);
        // Even alongside a write: the reader needs to know the value was not
        // one this project has a contract for.
        assert_eq!(
            ResourceUsage(ResourceUsage::WRITE | 0x8000).classify(),
            UsageClass::Undeclared
        );
    }

    #[test]
    fn the_stages_this_device_has_no_encoder_for_are_named_together() {
        assert!(
            !RenderStages(RenderStages::VERTEX | RenderStages::FRAGMENT).names_unexecuted_stage()
        );
        for bit in [RenderStages::TILE, RenderStages::OBJECT, RenderStages::MESH] {
            assert!(RenderStages(bit).names_unexecuted_stage());
        }
        assert_eq!(RenderStages(0x40).undeclared_bits(), 0x40);
    }
}
