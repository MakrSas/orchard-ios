//! Lifting the content-representation records.
//!
//! # The shape comes from the target, and there are only three
//!
//! Nine opcodes, and between them exactly three payloads: a ref, a ref with a
//! `slice:level:` pair, and nothing at all. So this decoder dispatches on the
//! *target* the vocabulary already derived rather than on the opcode a second
//! time — `content_request` is the one place that maps an opcode to a directive
//! and a target, and reading the opcode again here would be a second answer to
//! a question already answered.
//!
//! The record that carries nothing is the compute encoder's compressed-
//! reinterpretation flush, and its emptiness is checked rather than assumed:
//! the payload has to be empty, because a record with bytes in it is not the
//! record whose selector takes no argument.

use super::{no_record, short, DecodeRefusal};
use crate::closure::Rail;
use crate::resource_state::{content_request, ContentDirective, ContentTarget};
use reims_vgpu_wire::op::Op;
use reims_vgpu_wire::ops::blit as wire;

/// What the record named, with the guest's ref still on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordTarget {
    /// A ref alone. Not "slice 0, level 0": a whole-resource record covers
    /// every subresource, and the two must not converge here.
    WholeResource { object_ref: u32 },
    /// A texture and one of its subresources.
    SliceLevel {
        texture_ref: u32,
        slice: u16,
        level: u16,
    },
    /// The record named nothing, so its scope is the encoder that issued it.
    Encoder,
}

/// One lifted content-representation record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceStateRecord {
    pub directive: ContentDirective,
    pub target: RecordTarget,
}

impl ResourceStateRecord {
    /// The target shape the record was sent at.
    #[must_use]
    pub const fn target_kind(&self) -> ContentTarget {
        match self.target {
            RecordTarget::WholeResource { .. } => ContentTarget::WholeResource,
            RecordTarget::SliceLevel { .. } => ContentTarget::SliceLevel,
            RecordTarget::Encoder => ContentTarget::Encoder,
        }
    }
}

/// Lift a content-representation record out of its bytes.
pub fn decode(rail: Rail, op: &Op<'_>) -> Result<ResourceStateRecord, DecodeRefusal> {
    let opcode = op.opcode();
    let Some((directive, target)) = content_request(rail, opcode) else {
        return Err(no_record(rail, opcode));
    };
    let have = op.payload.len();
    let fail = |need: usize| short(rail, opcode, have, need);

    let target = match target {
        ContentTarget::WholeResource => {
            let r = wire::object_ref(op).map_err(|_| fail(core::mem::size_of::<wire::Ref>()))?;
            RecordTarget::WholeResource {
                object_ref: r.object_ref.get(),
            }
        }
        ContentTarget::SliceLevel => {
            let r = wire::ref_slice_level(op)
                .map_err(|_| fail(core::mem::size_of::<wire::RefSliceLevel>()))?;
            RecordTarget::SliceLevel {
                texture_ref: r.texture_ref.get(),
                slice: r.slice.get(),
                level: r.level.get(),
            }
        }
        // The selector takes no argument, so a payload here is not this record.
        ContentTarget::Encoder if have != 0 => {
            return Err(DecodeRefusal::UnexpectedPayload { rail, opcode, have })
        }
        ContentTarget::Encoder => RecordTarget::Encoder,
    };
    Ok(ResourceStateRecord { directive, target })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use reims_vgpu_wire::op::{op, OP_HEADER_LEN};

    fn record(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let total = (OP_HEADER_LEN + payload.len()) as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&total.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn lift(rail: Rail, bytes: &[u8]) -> Result<ResourceStateRecord, DecodeRefusal> {
        decode(rail, &op(bytes, 0).expect("framed"))
    }

    /// Every opcode the vocabulary names lifts a record whose directive and
    /// target are the ones the vocabulary assigned. Driven over the whole
    /// opcode space rather than over a written list, so a tenth record added to
    /// `content_request` cannot land here undecoded.
    #[test]
    fn every_content_opcode_lifts_the_directive_and_target_it_names() {
        let mut seen = 0usize;
        for rail in [Rail::Blit, Rail::Compute, Rail::Render] {
            for opcode in 0u32..0x200 {
                let Some((directive, target)) = content_request(rail, opcode) else {
                    continue;
                };
                seen += 1;
                let payload: &[u8] = match target {
                    ContentTarget::WholeResource => &[7, 0, 0, 0],
                    ContentTarget::SliceLevel => &[7, 0, 0, 0, 3, 0, 5, 0],
                    ContentTarget::Encoder => &[],
                };
                let lifted = lift(rail, &record(opcode, payload)).expect("lifted");
                assert_eq!(lifted.directive, directive);
                assert_eq!(lifted.target_kind(), target);
            }
        }
        assert_eq!(seen, 9);
    }

    /// `slice` and `level` are 16 bits each sharing one word, and they are
    /// unequal in the fixture on purpose: equal values could not tell a crossed
    /// pair from a correct one.
    #[test]
    fn a_slice_level_record_keeps_the_two_apart() {
        let bytes = record(
            wire::OPCODE_SYNCHRONIZE_TEXTURE,
            &[0x21, 0x43, 0, 0, 3, 0, 5, 0],
        );
        assert_eq!(
            lift(Rail::Blit, &bytes),
            Ok(ResourceStateRecord {
                directive: ContentDirective::Synchronize,
                target: RecordTarget::SliceLevel {
                    texture_ref: 0x4321,
                    slice: 3,
                    level: 5,
                },
            })
        );
    }

    /// A whole-resource record is not the subresource record with zeroes in it.
    /// The two forms exist because they mean different extents, and a decoder
    /// that folded them would publish one level where the guest asked for all
    /// of them.
    #[test]
    fn a_whole_resource_record_does_not_become_slice_zero_level_zero() {
        let whole = lift(
            Rail::Blit,
            &record(wire::OPCODE_SYNCHRONIZE_RESOURCE, &[7, 0, 0, 0]),
        )
        .expect("lifted");
        let sliced = lift(
            Rail::Blit,
            &record(wire::OPCODE_SYNCHRONIZE_TEXTURE, &[7, 0, 0, 0, 0, 0, 0, 0]),
        )
        .expect("lifted");
        assert_eq!(whole.directive, sliced.directive);
        assert_ne!(whole.target, sliced.target);
    }

    /// The flush names nothing, and a payload means the record is not it.
    #[test]
    fn the_encoder_flush_is_empty_and_a_payload_is_refused() {
        let opcode = crate::resource_state::OPCODE_COMPRESSED_REINTERPRETATION_FLUSH;
        assert_eq!(
            lift(Rail::Compute, &record(opcode, &[])),
            Ok(ResourceStateRecord {
                directive: ContentDirective::FlushCompressedReinterpretation,
                target: RecordTarget::Encoder,
            })
        );
        assert_eq!(
            lift(Rail::Compute, &record(opcode, &[0])),
            Err(DecodeRefusal::UnexpectedPayload {
                rail: Rail::Compute,
                opcode,
                have: 1,
            })
        );
    }

    /// A subresource record one byte short of its body is refused with both
    /// numbers rather than read out of the bytes that are there.
    #[test]
    fn a_short_record_is_refused_with_what_it_had_and_what_it_needed() {
        let bytes = record(wire::OPCODE_SYNCHRONIZE_TEXTURE, &[7, 0, 0, 0, 3, 0, 5]);
        assert_eq!(
            lift(Rail::Blit, &bytes),
            Err(DecodeRefusal::Short {
                rail: Rail::Blit,
                opcode: wire::OPCODE_SYNCHRONIZE_TEXTURE,
                have: 7,
                need: core::mem::size_of::<wire::RefSliceLevel>(),
            })
        );
    }

    /// The compute rail's flush opcode is not a blit opcode, and asking the
    /// blit rail for it lifts nothing. The rail parameter is what keeps two
    /// opcode spaces from answering for each other.
    #[test]
    fn the_flush_opcode_means_nothing_on_the_blit_rail() {
        let opcode = crate::resource_state::OPCODE_COMPRESSED_REINTERPRETATION_FLUSH;
        assert!(lift(Rail::Blit, &record(opcode, &[])).is_err());
    }
}
