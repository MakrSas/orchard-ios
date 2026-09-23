//! What a serializer destroy record means: which kind of object ends, and
//! which reference names it.
//!
//! `CmdDeleteObject` carries exactly one of these, and every one of the eleven
//! writes the identical twelve-byte body — so **the kind is the opcode and
//! nothing else**, which is [`reims_vgpu_wire::ops::destroy`]'s claim and this
//! module's reason for existing: the wire crate has the layout, and assigning
//! the layout a meaning is this crate's.
//!
//! # The kind is what decides whether the model may act
//!
//! The eleven kinds do not share a closure. Sampler, depth-stencil,
//! render-pipeline and fence destroys retire state this device holds; function
//! and compute-pipeline destroys are proven no-ops on the cell that this device
//! keys both by content rather than by ref; and five kinds — buffer, texture,
//! heap, rasterization rate map and indirect command buffer — have never been
//! observed on any driven boot and are unresolved. That last group is why this
//! module answers a *kind* rather than a boolean: a caller that acted on all
//! eleven would be acting on five it has no evidence for, and one that refused
//! all eleven would refuse the 99.6% a real guest actually sends.
//!
//! So the ledger row for each kind is [`crate::closure::LEDGER`]'s, on the
//! serializer's [`crate::closure::Rail::Root`], and [`DestroyKind::settled`]
//! reads it there rather than restating it here. A kind that is added to this
//! enum without a row cannot be settled, which is the failure direction that
//! costs nothing.

use crate::closure::{find, Rail};
use reims_vgpu_wire::op::op;
use reims_vgpu_wire::ops::destroy as wire;

/// The kind of object a destroy record ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DestroyKind {
    Buffer,
    Texture,
    DepthStencilState,
    SamplerState,
    Function,
    ComputePipelineState,
    RenderPipelineState,
    Fence,
    Heap,
    RasterizationRateMap,
    IndirectCommandBuffer,
}

impl DestroyKind {
    /// Every kind, for the sweeps that must not miss one.
    pub const ALL: [Self; 11] = [
        Self::Buffer,
        Self::Texture,
        Self::DepthStencilState,
        Self::SamplerState,
        Self::Function,
        Self::ComputePipelineState,
        Self::RenderPipelineState,
        Self::Fence,
        Self::Heap,
        Self::RasterizationRateMap,
        Self::IndirectCommandBuffer,
    ];

    /// The kind an opcode names, or `None` for a number in the destroy span
    /// that no selector claims.
    ///
    /// Five of the sixteen numbers in `0x3e8`–`0x3f7` belong to no known
    /// selector. They are `None` rather than a twelfth kind: a record at one of
    /// them is something unmeasured, and naming it would be the guess the wire
    /// crate's module documentation declines to make.
    #[must_use]
    pub const fn of(opcode: u32) -> Option<Self> {
        Some(match opcode {
            wire::OPCODE_DELETE_BUFFER => Self::Buffer,
            wire::OPCODE_DELETE_TEXTURE => Self::Texture,
            wire::OPCODE_DELETE_DEPTH_STENCIL_STATE => Self::DepthStencilState,
            wire::OPCODE_DELETE_SAMPLER_STATE => Self::SamplerState,
            wire::OPCODE_DELETE_FUNCTION => Self::Function,
            wire::OPCODE_DELETE_COMPUTE_PIPELINE_STATE => Self::ComputePipelineState,
            wire::OPCODE_DELETE_RENDER_PIPELINE_STATE => Self::RenderPipelineState,
            wire::OPCODE_DELETE_FENCE => Self::Fence,
            wire::OPCODE_DELETE_HEAP => Self::Heap,
            wire::OPCODE_DELETE_RASTERIZATION_RATE_MAP => Self::RasterizationRateMap,
            wire::OPCODE_DELETE_INDIRECT_COMMAND_BUFFER => Self::IndirectCommandBuffer,
            _ => return None,
        })
    }

    /// The opcode this kind is written at.
    #[must_use]
    pub const fn opcode(self) -> u32 {
        match self {
            Self::Buffer => wire::OPCODE_DELETE_BUFFER,
            Self::Texture => wire::OPCODE_DELETE_TEXTURE,
            Self::DepthStencilState => wire::OPCODE_DELETE_DEPTH_STENCIL_STATE,
            Self::SamplerState => wire::OPCODE_DELETE_SAMPLER_STATE,
            Self::Function => wire::OPCODE_DELETE_FUNCTION,
            Self::ComputePipelineState => wire::OPCODE_DELETE_COMPUTE_PIPELINE_STATE,
            Self::RenderPipelineState => wire::OPCODE_DELETE_RENDER_PIPELINE_STATE,
            Self::Fence => wire::OPCODE_DELETE_FENCE,
            Self::Heap => wire::OPCODE_DELETE_HEAP,
            Self::RasterizationRateMap => wire::OPCODE_DELETE_RASTERIZATION_RATE_MAP,
            Self::IndirectCommandBuffer => wire::OPCODE_DELETE_INDIRECT_COMMAND_BUFFER,
        }
    }

    /// A stable one-word name, for a census line or a refusal.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Texture => "texture",
            Self::DepthStencilState => "depth_stencil_state",
            Self::SamplerState => "sampler_state",
            Self::Function => "function",
            Self::ComputePipelineState => "compute_pipeline_state",
            Self::RenderPipelineState => "render_pipeline_state",
            Self::Fence => "fence",
            Self::Heap => "heap",
            Self::RasterizationRateMap => "rasterization_rate_map",
            Self::IndirectCommandBuffer => "indirect_command_buffer",
        }
    }

    /// Whether the ledger has settled what destroying this kind does.
    ///
    /// Read from [`crate::closure::LEDGER`] rather than restated, so a row that
    /// changes changes this. A kind with no row at all answers `false`: an
    /// enum member the ledger has never heard of is exactly as unsettled as one
    /// it has recorded a question for.
    #[must_use]
    pub fn settled(self) -> bool {
        find(Rail::Root, self.opcode()).is_some_and(|op| !op.closure.blocks_cutover())
    }
}

/// The numbers the destroy family is written in.
///
/// Eleven of the sixteen are claimed. The span is named so that a record at one
/// of the other five refuses as unmeasured rather than as a foreign command —
/// see [`DestroyRefusal::UnclaimedOpcode`].
pub const SPAN: core::ops::RangeInclusive<u32> =
    wire::OPCODE_DELETE_BUFFER..=wire::OPCODE_DELETE_INDIRECT_COMMAND_BUFFER;

/// One destroy record, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Destroy {
    pub kind: DestroyKind,
    /// The reference the record names, in the task the *command* named.
    ///
    /// An object-list reference, measured rather than assumed:
    /// a sampler is constructed by looking this same number up in the guest's
    /// object list and requiring a serializer-object entry there. The number is
    /// not a second namespace, which is what a driven boot's finding that most
    /// destroys resolve to no live entry could be misread as — the guest clears
    /// its slot before it sends the destroy, and the name outlives the slot.
    pub object_ref: u32,
}

/// Why a destroy record did not decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DestroyRefusal {
    /// The bytes are not a record at all — too short for even an opcode.
    NotARecord,
    /// The opcode is outside the destroy family.
    NotADestroy { opcode: u32 },
    /// The opcode is inside the destroy *span* and no selector claims it.
    ///
    /// Five of the sixteen numbers in `0x3e8`–`0x3f7` are like this. The span
    /// is checked here and not in the wire crate on purpose: "these numbers are
    /// adjacent" is a layout fact and "an adjacent number is probably a delete"
    /// is a meaning, which is the guess `is_delete` declines to make and this
    /// refusal makes findable instead. A record at one of them is something
    /// unmeasured; a record at a render opcode is a command that does not
    /// belong in this packet at all, and only the second says the guest sent
    /// the wrong thing.
    UnclaimedOpcode { opcode: u32 },
    /// The record's body does not hold the one reference every destroy carries.
    RefUnreadable { opcode: u32 },
}

impl DestroyRefusal {
    /// The stable reason string for a failure channel.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::NotARecord => "destroy_not_a_record",
            Self::NotADestroy { .. } => "destroy_wrong_family",
            Self::UnclaimedOpcode { .. } => "destroy_unclaimed_opcode",
            Self::RefUnreadable { .. } => "destroy_ref_unreadable",
        }
    }
}

/// Decode the record a `CmdDeleteObject` carries.
///
/// # Errors
///
/// [`DestroyRefusal`], one variant per way the bytes are not a destroy.
pub fn decode(record: &[u8]) -> Result<Destroy, DestroyRefusal> {
    let op = op(record, 0).map_err(|_| DestroyRefusal::NotARecord)?;
    let opcode = op.opcode();
    let Some(kind) = DestroyKind::of(opcode) else {
        return Err(if SPAN.contains(&opcode) {
            DestroyRefusal::UnclaimedOpcode { opcode }
        } else {
            DestroyRefusal::NotADestroy { opcode }
        });
    };
    debug_assert!(wire::is_delete(opcode), "a named kind is a delete");
    let delete = wire::delete(&op).map_err(|_| DestroyRefusal::RefUnreadable { opcode })?;
    Ok(Destroy {
        kind,
        object_ref: delete.object_ref.get(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind round-trips through its own opcode.
    ///
    /// The hazard the wire crate's record documentation names: eleven selectors
    /// write one body and differ only in the opcode, so a table that mapped two
    /// of them to one kind would destroy the wrong thing and no byte in the
    /// record would say so.
    #[test]
    fn each_kind_is_its_own_opcode_and_no_two_share_one() {
        let mut seen = [0u32; DestroyKind::ALL.len()];
        for (slot, kind) in DestroyKind::ALL.iter().enumerate() {
            assert_eq!(
                DestroyKind::of(kind.opcode()),
                Some(*kind),
                "{} does not decode back to itself",
                kind.name()
            );
            assert!(
                !seen[..slot].contains(&kind.opcode()),
                "{} shares an opcode with an earlier kind",
                kind.name()
            );
            seen[slot] = kind.opcode();
        }
    }

    /// The five numbers in the span that no selector claims stay unnamed.
    #[test]
    fn an_unclaimed_number_in_the_destroy_span_is_no_kind() {
        for opcode in [0x3ec, 0x3f0, 0x3f2, 0x3f3, 0x3f5] {
            assert_eq!(
                DestroyKind::of(opcode),
                None,
                "{opcode:#x} is a number in the span that nothing has measured"
            );
        }
    }

    /// What the ledger has settled, asked of the ledger.
    ///
    /// The six a driven guest sends are settled and the five it never sends are
    /// not, and this asserts the partition rather than a count — a row that
    /// closes moves a kind across it and the assertion says which.
    #[test]
    fn the_kinds_a_driven_guest_sends_are_settled_and_the_five_it_never_sends_are_not() {
        let settled: Vec<&str> = DestroyKind::ALL
            .iter()
            .filter(|kind| kind.settled())
            .map(|kind| kind.name())
            .collect();
        assert_eq!(
            settled,
            vec![
                "depth_stencil_state",
                "sampler_state",
                "function",
                "compute_pipeline_state",
                "render_pipeline_state",
                "fence",
            ],
            "the settled set is not the six kinds a driven boot has observed"
        );
    }

    /// A record's opcode decides the kind, and its body the reference.
    #[test]
    fn a_record_decodes_to_the_kind_its_opcode_names_and_the_ref_its_body_holds() {
        let mut record = [0u8; wire::DELETE_TOTAL_LEN as usize];
        record[0..4].copy_from_slice(&wire::OPCODE_DELETE_SAMPLER_STATE.to_le_bytes());
        record[4..8].copy_from_slice(&wire::DELETE_TOTAL_LEN.to_le_bytes());
        record[8..12].copy_from_slice(&0x2a_u32.to_le_bytes());
        assert_eq!(
            decode(&record),
            Ok(Destroy {
                kind: DestroyKind::SamplerState,
                object_ref: 0x2a,
            })
        );

        // A record from another family is refused as the wrong family rather
        // than read at these offsets, which would answer some other record's
        // first argument as a reference and delete it.
        let mut render = record;
        render[0..4].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            decode(&render),
            Err(DestroyRefusal::NotADestroy { opcode: 1 })
        );

        // And a number inside the span that nothing claims is its own refusal.
        let mut unclaimed = record;
        unclaimed[0..4].copy_from_slice(&0x3ecu32.to_le_bytes());
        assert_eq!(
            decode(&unclaimed),
            Err(DestroyRefusal::UnclaimedOpcode { opcode: 0x3ec })
        );
    }
}
