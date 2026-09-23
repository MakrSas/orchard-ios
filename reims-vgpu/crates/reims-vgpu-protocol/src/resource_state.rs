//! What the content-representation records ask for.
//!
//! # Nine opcodes, five directives, and three targets
//!
//! `optimizeContentsForCPUAccess:`, `optimizeContentsForGPUAccess:`,
//! `synchronizeResource:` and `invalidateCompressedTexture:` each come in a
//! whole-resource form and a `slice:level:` form — eight opcodes over four
//! directives and two targets, regular enough that a table is the honest way to
//! write it.
//!
//! The ninth is the compute encoder's
//! `insertCompressedTextureReinterpretationFlush`, which names **no resource at
//! all**. Its target is the encoder, and giving it a target of its own is what
//! keeps "the record named nothing" from becoming "the record named resource
//! zero".
//!
//! # What these records are *about* is where content is, not what it is
//!
//! None of them changes a texel. They say which representation the guest is
//! about to use, or which one it no longer trusts. That makes them the
//! operations most exposed to a change of memory topology: every one of them is
//! a proven no-op today on a cell that reads "guest pages are written directly"
//! or "guest pages are the single copy of resource content", and both of those
//! are statements about the placement the current executor chose.
//!
//! The layer that gets to make placement decisions is the executor, so the
//! layer that gets to conclude "there is nothing to do" is the executor. This
//! module says what was asked.

use crate::closure::Rail;
use reims_vgpu_wire::ops::blit as wire;

/// What a content-representation record asks the device to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContentDirective {
    /// Make the guest's own pages current for this content.
    ///
    /// The one directive with a definable semantic effect: after it, a CPU read
    /// of the guest's pages must see what the GPU produced. Whether that costs
    /// anything depends on where the content actually is.
    Synchronize,
    /// Prepare the content for CPU access.
    OptimizeForCpu,
    /// Prepare the content for GPU access.
    OptimizeForGpu,
    /// The lossless-compression metadata for this content is no longer to be
    /// trusted.
    InvalidateCompressed,
    /// Everything the encoder has reinterpreted through a compressed view is to
    /// be made visible.
    ///
    /// The one directive that names no resource: the selector takes no
    /// argument, so its scope is the encoder.
    FlushCompressedReinterpretation,
}

impl ContentDirective {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Synchronize => "synchronize",
            Self::OptimizeForCpu => "optimize_for_cpu",
            Self::OptimizeForGpu => "optimize_for_gpu",
            Self::InvalidateCompressed => "invalidate_compressed",
            Self::FlushCompressedReinterpretation => "flush_compressed_reinterpretation",
        }
    }
}

/// What a record named.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContentTarget {
    /// The record carried a ref alone.
    WholeResource,
    /// The record carried a slice and a level.
    SliceLevel,
    /// The record carried nothing, and its scope is the encoder that issued it.
    Encoder,
}

/// The compute encoder's compressed-reinterpretation flush.
///
/// Named here for the same reason the event opcodes are: it is an opcode with a
/// meaning, and the meaning is this layer's to assign.
pub const OPCODE_COMPRESSED_REINTERPRETATION_FLUSH: u32 = 0xe3;

/// What an opcode on `rail` asks for, if it asks for anything.
///
/// The rail is a parameter and not an afterthought: the blit family and the
/// compute flush live in different opcode spaces, and a rail-free table would
/// be one collision away from reading a compute record as a blit one.
#[must_use]
pub fn content_request(rail: Rail, opcode: u32) -> Option<(ContentDirective, ContentTarget)> {
    use ContentDirective::*;
    use ContentTarget::*;
    Some(match (rail, opcode) {
        (Rail::Blit, wire::OPCODE_OPTIMIZE_FOR_CPU) => (OptimizeForCpu, WholeResource),
        (Rail::Blit, wire::OPCODE_OPTIMIZE_FOR_GPU) => (OptimizeForGpu, WholeResource),
        (Rail::Blit, wire::OPCODE_SYNCHRONIZE_RESOURCE) => (Synchronize, WholeResource),
        (Rail::Blit, wire::OPCODE_INVALIDATE_COMPRESSED_TEXTURE) => {
            (InvalidateCompressed, WholeResource)
        }
        (Rail::Blit, wire::OPCODE_OPTIMIZE_FOR_CPU_SLICE_LEVEL) => (OptimizeForCpu, SliceLevel),
        (Rail::Blit, wire::OPCODE_OPTIMIZE_FOR_GPU_SLICE_LEVEL) => (OptimizeForGpu, SliceLevel),
        (Rail::Blit, wire::OPCODE_SYNCHRONIZE_TEXTURE) => (Synchronize, SliceLevel),
        (Rail::Blit, wire::OPCODE_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL) => {
            (InvalidateCompressed, SliceLevel)
        }
        (Rail::Compute, OPCODE_COMPRESSED_REINTERPRETATION_FLUSH) => {
            (FlushCompressedReinterpretation, Encoder)
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{Closure, LEDGER};

    /// The two resource targets cover the same four directives, and the wire
    /// crate's own predicates agree about which opcodes are which shape.
    #[test]
    fn each_resource_directive_has_both_targets_and_the_shapes_agree() {
        for directive in [
            ContentDirective::Synchronize,
            ContentDirective::OptimizeForCpu,
            ContentDirective::OptimizeForGpu,
            ContentDirective::InvalidateCompressed,
        ] {
            let mut seen = [false; 2];
            for opcode in 0u32..0x200 {
                let Some((d, t)) = content_request(Rail::Blit, opcode) else {
                    continue;
                };
                if d != directive {
                    continue;
                }
                match t {
                    ContentTarget::WholeResource => {
                        assert!(wire::is_ref(opcode), "{opcode:#x} is not a bare ref record");
                        seen[0] = true;
                    }
                    ContentTarget::SliceLevel => {
                        assert!(
                            wire::is_ref_slice_level(opcode),
                            "{opcode:#x} is not a slice/level record"
                        );
                        seen[1] = true;
                    }
                    ContentTarget::Encoder => unreachable!("no blit record names the encoder"),
                }
            }
            assert_eq!(seen, [true, true], "{directive:?} is missing a form");
        }
    }

    /// The encoder-scoped flush names no resource, and it is the only one.
    #[test]
    fn exactly_one_directive_names_the_encoder() {
        let mut encoder_scoped = 0;
        for rail in [
            Rail::Blit,
            Rail::Compute,
            Rail::Render,
            Rail::Event,
            Rail::Info,
        ] {
            for opcode in 0u32..0x200 {
                if let Some((_, ContentTarget::Encoder)) = content_request(rail, opcode) {
                    encoder_scoped += 1;
                    assert_eq!(rail, Rail::Compute);
                    assert_eq!(opcode, OPCODE_COMPRESSED_REINTERPRETATION_FLUSH);
                }
            }
        }
        assert_eq!(encoder_scoped, 1);
    }

    /// Every opcode named here is a judged operation, and every one is a
    /// proven no-op.
    ///
    /// The second half is the part worth asserting: it is the premise of this
    /// module's documentation, and of `reims_vgpu_core::resource_state` keeping
    /// these as operations rather than dropping them. If one of them ever
    /// becomes implemented or unresolved, that reasoning needs reading again.
    #[test]
    fn every_content_request_is_a_judged_proven_no_op() {
        let mut count = 0;
        for rail in [
            Rail::Blit,
            Rail::Compute,
            Rail::Render,
            Rail::Event,
            Rail::Info,
        ] {
            for opcode in 0u32..0x200 {
                if content_request(rail, opcode).is_none() {
                    continue;
                }
                count += 1;
                let op = LEDGER
                    .iter()
                    .find(|o| o.rail == rail && o.opcode == Some(opcode))
                    .unwrap_or_else(|| panic!("{rail:?} {opcode:#x} has no ledger row"));
                assert!(
                    matches!(op.closure, Closure::ProvenNoOp { .. }),
                    "{rail:?} {opcode:#x} is {}",
                    op.closure.name()
                );
            }
        }
        assert_eq!(count, 9, "four directives in two targets, plus the flush");
    }
}
