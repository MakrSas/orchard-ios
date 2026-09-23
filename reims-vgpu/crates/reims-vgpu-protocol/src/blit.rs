//! Which transfer a blit opcode names.
//!
//! # Why this is a separate enumeration from the ledger
//!
//! [`crate::closure`] answers "does the device owe the guest anything for this
//! opcode, and has that been established". It says nothing about *shape*: a
//! buffer-to-buffer copy and a mipmap generation are one row each and look
//! identical from there. The model needs the shape, because the shape is what
//! decides which memory an operation touches and in which direction.
//!
//! So this is the second half of the same claim, and the two are joined by a
//! test rather than by convention: the kinds enumerated here are exactly the
//! blit-rail operations the ledger has judged and the operation vocabulary
//! classifies as transfers. An opcode that gains a contract without gaining a
//! kind fails that test, which is the only way "the vocabulary is exhaustive"
//! survives contact with a ledger that keeps changing.
//!
//! # What is deliberately absent
//!
//! The fence pair, the barrier-shaped residency and content-representation
//! records, the indirect-command-buffer family, and both `fillTexture:` forms.
//! The first three are other operation classes; the last two are unresolved and
//! must not be given a payload the model can execute, because executing a guess
//! about a write is worse than refusing it — the guest reads back content it
//! believes it wrote either way, and only the refusal says so.

use crate::pixel_format::BlitAspect;
use reims_vgpu_wire::ops::blit as wire;

// `MTLBlitOption`. A closed set of flag bits, which is what makes it this
// crate's to name: the first layer allowed to assign meaning to a wire tag.
pub const MTL_BLIT_OPTION_NONE: u32 = 0;
pub const MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL: u32 = 1 << 0;
pub const MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL: u32 = 1 << 1;
pub const MTL_BLIT_OPTION_ROW_LINEAR_PVRTC: u32 = 1 << 2;
/// Every bit the option word defines. A word outside it is not an option this
/// device declined to implement; it is a word this device cannot read.
pub const MTL_BLIT_OPTION_KNOWN_MASK: u32 = MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL
    | MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL
    | MTL_BLIT_OPTION_ROW_LINEAR_PVRTC;

/// Why an option word names no plane this device can copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionRefusal {
    /// A bit outside the defined set. Unknown stays unknown: a word this
    /// device cannot read is not a word it may ignore, because the bit it
    /// cannot read may be the one that says which bytes the guest meant.
    UnknownBits { options: u32 },
    /// `MTLBlitOptionRowLinearPVRTC`. The row-linear layout of a compressed
    /// PVRTC surface is a different addressing rule for the same bytes, and
    /// this device has no PVRTC path to apply it to.
    RowLinearPvrtc,
    /// Both plane bits at once. A copy addresses one plane or the whole texel,
    /// and "the depth plane and the stencil plane, interleaved how" is not a
    /// term the wire establishes.
    ConflictingAspects,
}

impl OptionRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownBits { .. } => "blit_options_unknown_bits",
            Self::RowLinearPvrtc => "blit_options_row_linear_pvrtc",
            Self::ConflictingAspects => "blit_options_conflicting_aspects",
        }
    }
}

/// The plane a blit's `MTLBlitOption` word selects.
///
/// Zero selects the whole texel, which for every format but the two combined
/// depth-stencil ones is the only plane there is.
///
/// # Errors
///
/// [`OptionRefusal`] for a word this device cannot read as a plane selection.
pub fn select_aspect(options: u32) -> Result<BlitAspect, OptionRefusal> {
    if options == MTL_BLIT_OPTION_NONE {
        return Ok(BlitAspect::Full);
    }
    if options & !MTL_BLIT_OPTION_KNOWN_MASK != 0 {
        return Err(OptionRefusal::UnknownBits { options });
    }
    if options & MTL_BLIT_OPTION_ROW_LINEAR_PVRTC != 0 {
        return Err(OptionRefusal::RowLinearPvrtc);
    }
    match (
        options & MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL != 0,
        options & MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL != 0,
    ) {
        (true, false) => Ok(BlitAspect::Depth),
        (false, true) => Ok(BlitAspect::Stencil),
        (false, false) => Ok(BlitAspect::Full),
        (true, true) => Err(OptionRefusal::ConflictingAspects),
    }
}

/// The transfer an opcode names.
///
/// One variant per record shape, not per selector: the `options:` forms of
/// buffer-to-texture and texture-to-buffer share their sibling's opcode and
/// length and carry the option in room the plain form already reserves, so they
/// are the same kind. The region copy is the exception — `options:` there is a
/// different opcode at a different length — and it keeps its own variant for
/// exactly that reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlitKind {
    /// `copyFromBuffer:…toTexture:…`, with or without `options:`.
    BufferToTexture,
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`.
    BufferToBuffer,
    /// `copyFromTexture:…toBuffer:…`, with or without `options:`.
    TextureToBuffer,
    /// `copyFromTexture:…toTexture:…` over a region.
    TextureRegion,
    /// The region copy's `options:` form, which is its own opcode and four
    /// bytes longer.
    TextureRegionOptions,
    /// `copyFromTexture:sourceSlice:sourceLevel:toTexture:…sliceCount:levelCount:`,
    /// and the whole-texture `copyFromTexture:toTexture:` that shares it.
    TextureSlices,
    /// `fillBuffer:range:value:`.
    FillBuffer,
    /// `fillBuffer:range:pattern4:`.
    FillBufferPattern4,
    /// `generateMipmapsForTexture:`.
    GenerateMipmaps,
}

impl BlitKind {
    pub const ALL: &'static [BlitKind] = &[
        BlitKind::BufferToTexture,
        BlitKind::BufferToBuffer,
        BlitKind::TextureToBuffer,
        BlitKind::TextureRegion,
        BlitKind::TextureRegionOptions,
        BlitKind::TextureSlices,
        BlitKind::FillBuffer,
        BlitKind::FillBufferPattern4,
        BlitKind::GenerateMipmaps,
    ];

    /// The opcode this kind is carried by, from the wire crate's constants.
    #[must_use]
    pub const fn wire_opcode(self) -> u32 {
        match self {
            Self::BufferToTexture => wire::OPCODE_COPY_BUFFER_TO_TEXTURE,
            Self::BufferToBuffer => wire::OPCODE_COPY_BUFFER_TO_BUFFER,
            Self::TextureToBuffer => wire::OPCODE_COPY_TEXTURE_TO_BUFFER,
            Self::TextureRegion => wire::OPCODE_COPY_TEXTURE_REGION,
            Self::TextureRegionOptions => wire::OPCODE_COPY_TEXTURE_REGION_OPTIONS,
            Self::TextureSlices => wire::OPCODE_COPY_TEXTURE_SLICES,
            Self::FillBuffer => wire::OPCODE_FILL_BUFFER,
            Self::FillBufferPattern4 => wire::OPCODE_FILL_BUFFER_PATTERN4,
            Self::GenerateMipmaps => wire::OPCODE_GENERATE_MIPMAPS,
        }
    }

    /// The kind an opcode names, or `None` if it names no transfer.
    ///
    /// `None` covers three different things — another operation class, an
    /// unresolved opcode, and an opcode with no contract at all — and this
    /// module is deliberately not the place that tells them apart. The ledger
    /// is.
    #[must_use]
    pub fn of_opcode(opcode: u32) -> Option<BlitKind> {
        BlitKind::ALL
            .iter()
            .copied()
            .find(|k| k.wire_opcode() == opcode)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::BufferToTexture => "buffer_to_texture",
            Self::BufferToBuffer => "buffer_to_buffer",
            Self::TextureToBuffer => "texture_to_buffer",
            Self::TextureRegion => "texture_region",
            Self::TextureRegionOptions => "texture_region_options",
            Self::TextureSlices => "texture_slices",
            Self::FillBuffer => "fill_buffer",
            Self::FillBufferPattern4 => "fill_buffer_pattern4",
            Self::GenerateMipmaps => "generate_mipmaps",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{Rail, LEDGER};

    /// Every option word this device can read, and every one it refuses.
    ///
    /// `select_aspect` is the only place a `MTLBlitOption` word becomes a plane
    /// selection, and both rails and every copy record ask it here. A zero and
    /// an absent options field are the same answer on purpose: a record with no
    /// `options:` field selects the whole texel, so the caller no longer needs
    /// to carry "this record has no options word" beside the word itself.
    #[test]
    fn an_option_word_selects_one_plane_or_names_why_it_cannot() {
        assert_eq!(select_aspect(MTL_BLIT_OPTION_NONE), Ok(BlitAspect::Full));
        assert_eq!(
            select_aspect(MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL),
            Ok(BlitAspect::Depth)
        );
        assert_eq!(
            select_aspect(MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL),
            Ok(BlitAspect::Stencil)
        );
        // Both plane bits at once names no plane a copy can address.
        assert_eq!(
            select_aspect(
                MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL
                    | MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL
            ),
            Err(OptionRefusal::ConflictingAspects)
        );
        // No PVRTC rail to apply a row-linear addressing rule to.
        assert_eq!(
            select_aspect(MTL_BLIT_OPTION_ROW_LINEAR_PVRTC),
            Err(OptionRefusal::RowLinearPvrtc)
        );
        // Unknown stays unknown: the bit this device cannot read may be the one
        // that says which bytes the guest meant.
        assert_eq!(
            select_aspect(1 << 8),
            Err(OptionRefusal::UnknownBits { options: 1 << 8 })
        );
        // Each refusal answers under its own name; a shared slug would report
        // three different losses as one.
        let slugs = [
            OptionRefusal::ConflictingAspects.slug(),
            OptionRefusal::RowLinearPvrtc.slug(),
            OptionRefusal::UnknownBits { options: 1 << 8 }.slug(),
        ];
        for (i, a) in slugs.iter().enumerate() {
            assert!(!slugs[i + 1..].contains(a), "{a} is two refusals' name");
        }
    }

    #[test]
    fn no_two_kinds_share_an_opcode() {
        for (i, a) in BlitKind::ALL.iter().enumerate() {
            for b in &BlitKind::ALL[i + 1..] {
                assert_ne!(a.wire_opcode(), b.wire_opcode(), "{a:?} and {b:?}");
            }
            assert_eq!(BlitKind::of_opcode(a.wire_opcode()), Some(*a));
        }
    }

    /// The two forms of the region copy really are two opcodes, and the two
    /// forms of the other copies really are one. Pinned because the whole
    /// variant set is shaped by that asymmetry, and a reader who assumed it was
    /// uniform would collapse the pair or split the singles.
    #[test]
    fn the_region_copy_is_the_only_split_options_form() {
        assert_ne!(
            BlitKind::TextureRegion.wire_opcode(),
            BlitKind::TextureRegionOptions.wire_opcode()
        );
        assert_eq!(
            wire::COPY_TEXTURE_REGION_OPTIONS_TOTAL_LEN - wire::COPY_TEXTURE_REGION_TOTAL_LEN,
            4
        );
        assert_eq!(
            wire::COPY_BUFFER_TO_TEXTURE_TOTAL_LEN,
            wire::COPY_TEXTURE_TO_BUFFER_TOTAL_LEN
        );
    }

    /// Every kind here is a judged blit-rail operation.
    ///
    /// The other direction — every judged transfer has a kind — cannot be
    /// asserted from this crate, because "which judged blit ops are transfers
    /// rather than fences, barriers or residency" is the operation vocabulary's
    /// classification and that lives above. `reims_vgpu_core::blit` closes it.
    #[test]
    fn every_kind_is_a_judged_blit_rail_operation() {
        for kind in BlitKind::ALL {
            let op = LEDGER
                .iter()
                .find(|o| o.rail == Rail::Blit && o.opcode == Some(kind.wire_opcode()))
                .unwrap_or_else(|| panic!("{kind:?} has no ledger row"));
            assert!(
                !op.closure.blocks_cutover(),
                "{kind:?} is {} and must not have a payload the model can execute",
                op.closure.name()
            );
        }
    }
}
