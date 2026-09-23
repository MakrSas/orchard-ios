//! Opcodes 7, 8 and 0x1b — create a view onto an existing texture.
//!
//! The three `-[PGSerializer newTextureViewWithPixelFormat:...]` forms. A guest
//! reinterpreting a texture's format, or narrowing it to a mip range or an
//! array slice, emits one of these; the window server does it constantly.
//!
//! # Three records, one growing layout
//!
//! Each form is the one before it plus a field, and the shared prefix is at the
//! same offsets in all three. That is worth stating because it is the kind of
//! family where a decoder reads the widest form's field out of the narrowest
//! record.
//!
//! ```text
//!                                       opcode 7   opcode 8   opcode 0x1b
//! payload +000  u32  object_ref            yes        yes         yes
//! payload +004  u32  base_texture_ref      yes        yes         yes
//! payload +008  u16  pixel_format          yes        yes         yes
//! payload +010  u16  texture_type          NEVER      yes         yes
//!                                          WRITTEN
//! payload +012  u64  level_base             --        yes         yes
//! payload +020  u64  level_count            --        yes         yes
//! payload +028  u64  slice_base             --        yes         yes
//! payload +036  u64  slice_count            --        yes         yes
//! payload +044  u8   swizzle_red            --         --         yes
//! payload +045  u8   swizzle_green          --         --         yes
//! payload +046  u8   swizzle_blue           --         --         yes
//! payload +047  u8   swizzle_alpha          --         --         yes
//! ```
//!
//! Record lengths are 20, 52 and 56 bytes, header included.
//!
//! **The two bytes at payload +010 of an opcode-7 record are never written.**
//! The format-only form has no texture type to carry, and the serializer leaves
//! the slot alone rather than zeroing it, so on a real wire it is whatever the
//! guest's ring last held. A decoder that reads a texture type out of an
//! opcode-7 record reads noise; this crate gives that form its own body with no
//! such field.
//!
//! # How the layout was derived
//!
//! Perturbation. The ranged form was captured twice with every field different
//! — format 70/10, normal-texture, level base 3/1, level count 2/7, slice base 5/2,
//! slice count 4/6 — so no two fields hold the same value in either case and a
//! view that swapped `levels` for `slices`, or a base for a count, reports a
//! pair no case produced. The Objective-C encoding declares both ranges as
//! `{_NSRange=QQ}`, which fixes them as `u64` pairs before a byte is read.
//!
//! The swizzle was captured twice with all four channels distinct in each, and
//! the second case is a permutation of the first rather than a shift, so the
//! channel order is pinned rather than merely consistent.

use crate::le::{U16le, U32le, U64le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

/// Opcode for the format-only view form.
pub const OPCODE_TEXTURE_VIEW: u32 = 7;
/// Opcode for the form that also carries a type and two ranges.
pub const OPCODE_TEXTURE_VIEW_RANGED: u32 = 8;
/// Opcode for the ranged form plus a channel swizzle.
pub const OPCODE_TEXTURE_VIEW_SWIZZLE: u32 = 0x1b;

/// Total wire length of an [`OPCODE_TEXTURE_VIEW`] record, header included.
pub const TEXTURE_VIEW_TOTAL_LEN: u32 = 20;
/// Bytes of that record the serializer actually writes, header included.
///
/// The format-only form allocates room for the texture type the wider forms
/// carry and never writes it, so its body stops two bytes short. See the
/// module doc.
pub const TEXTURE_VIEW_WRITTEN_LEN: u32 = 18;
/// Total wire length of an [`OPCODE_TEXTURE_VIEW_RANGED`] record.
pub const TEXTURE_VIEW_RANGED_TOTAL_LEN: u32 = 52;
/// Total wire length of an [`OPCODE_TEXTURE_VIEW_SWIZZLE`] record.
pub const TEXTURE_VIEW_SWIZZLE_TOTAL_LEN: u32 = 56;

/// Payload of the format-only view record, opcode 7.
///
/// Stops at the format deliberately: the two bytes after it are the slot the
/// wider forms use for a texture type, and this form never writes them.
#[repr(C)]
#[derive(Debug)]
pub struct TextureViewBody {
    /// Ref the guest's object-ref allocator assigned to the new view.
    pub object_ref: U32le,
    /// Ref of the texture being viewed. Observed: the oracle's stub texture
    /// answers `4242` to `-textureRef`, and `4242` is what lands here.
    pub base_texture_ref: U32le,
    /// `MTLPixelFormat` ordinal the view reinterprets through. Observed:
    /// RGBA8Unorm → 70.
    pub pixel_format: U16le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for TextureViewBody {}

/// Payload of the ranged view record, opcode 8.
#[repr(C)]
#[derive(Debug)]
pub struct TextureViewRangedBody {
    /// Ref the guest's object-ref allocator assigned to the new view.
    pub object_ref: U32le,
    /// Ref of the texture being viewed.
    pub base_texture_ref: U32le,
    /// `MTLPixelFormat` ordinal. Observed: RGBA8Unorm → 70, R8Unorm → 10.
    pub pixel_format: U16le,
    /// `MTLTextureType` ordinal. Observed: 2D → 2, 2DArray → 3.
    pub texture_type: U16le,
    /// `levels.location`. Observed: 3, and 1 in the second case.
    pub level_base: U64le,
    /// `levels.length`. Observed: 2, and 7.
    pub level_count: U64le,
    /// `slices.location`. Observed: 5, and 2.
    pub slice_base: U64le,
    /// `slices.length`. Observed: 4, and 6.
    pub slice_count: U64le,
}

// SAFETY: align-1 all-bytes-valid `le` scalars throughout.
unsafe impl Wire for TextureViewRangedBody {}

/// The four channels a swizzled view remaps through, in wire order.
///
/// `MTLTextureSwizzle` ordinals: Zero 0, One 1, Red 2, Green 3, Blue 4,
/// Alpha 5. Observed as `(2, 3, 4, 5)` for the identity swizzle and
/// `(5, 0, 1, 2)` for a permutation, which is what fixes the order as
/// red, green, blue, alpha rather than any rotation of it.
#[repr(C)]
#[derive(Debug)]
pub struct SwizzleChannels {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

// SAFETY: four `u8`, so align-1 and every byte pattern is valid.
unsafe impl Wire for SwizzleChannels {}

/// Payload of the swizzled view record, opcode 0x1b.
#[repr(C)]
#[derive(Debug)]
pub struct TextureViewSwizzleBody {
    /// Everything the ranged form carries, at the same offsets.
    pub ranged: TextureViewRangedBody,
    /// The channel remap.
    pub swizzle: SwizzleChannels,
}

// SAFETY: two align-1 `Wire` structs.
unsafe impl Wire for TextureViewSwizzleBody {}

/// Whether `opcode` is one of the three this module reads.
#[inline]
pub fn is_texture_view(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_TEXTURE_VIEW | OPCODE_TEXTURE_VIEW_RANGED | OPCODE_TEXTURE_VIEW_SWIZZLE
    )
}

/// View the payload of a format-only texture-view record.
pub fn texture_view<'a>(op: &Op<'a>) -> Result<&'a TextureViewBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_TEXTURE_VIEW);
    view::<TextureViewBody>(op.payload)
}

/// View the payload of a ranged texture-view record.
pub fn texture_view_ranged<'a>(op: &Op<'a>) -> Result<&'a TextureViewRangedBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_TEXTURE_VIEW_RANGED);
    view::<TextureViewRangedBody>(op.payload)
}

/// View the payload of a swizzled texture-view record.
pub fn texture_view_swizzle<'a>(op: &Op<'a>) -> Result<&'a TextureViewSwizzleBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_TEXTURE_VIEW_SWIZZLE);
    view::<TextureViewSwizzleBody>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op, OP_HEADER_LEN};
    use core::mem::size_of;

    fn synth_ranged(opcode: u32, total: u32) -> [u8; 56] {
        let mut b = [0xAAu8; 56];
        b[0..4].copy_from_slice(&opcode.to_le_bytes());
        b[4..8].copy_from_slice(&total.to_le_bytes());
        b[8..12].copy_from_slice(&57u32.to_le_bytes());
        b[12..16].copy_from_slice(&4242u32.to_le_bytes());
        b[16..18].copy_from_slice(&70u16.to_le_bytes());
        b[18..20].copy_from_slice(&2u16.to_le_bytes());
        b[20..28].copy_from_slice(&3u64.to_le_bytes());
        b[28..36].copy_from_slice(&2u64.to_le_bytes());
        b[36..44].copy_from_slice(&5u64.to_le_bytes());
        b[44..52].copy_from_slice(&4u64.to_le_bytes());
        b[52] = 2;
        b[53] = 3;
        b[54] = 4;
        b[55] = 5;
        b
    }

    #[test]
    fn each_form_is_its_body_plus_the_header() {
        assert_eq!(
            size_of::<TextureViewBody>() + OP_HEADER_LEN,
            TEXTURE_VIEW_WRITTEN_LEN as usize
        );
        assert_eq!(
            TEXTURE_VIEW_TOTAL_LEN - TEXTURE_VIEW_WRITTEN_LEN,
            2,
            "the format-only form's unwritten tail is no longer two bytes"
        );
        assert_eq!(
            size_of::<TextureViewRangedBody>() + OP_HEADER_LEN,
            TEXTURE_VIEW_RANGED_TOTAL_LEN as usize
        );
        assert_eq!(
            size_of::<TextureViewSwizzleBody>() + OP_HEADER_LEN,
            TEXTURE_VIEW_SWIZZLE_TOTAL_LEN as usize
        );
        for a in [
            core::mem::align_of::<TextureViewBody>(),
            core::mem::align_of::<TextureViewRangedBody>(),
            core::mem::align_of::<TextureViewSwizzleBody>(),
            core::mem::align_of::<SwizzleChannels>(),
        ] {
            assert_eq!(a, 1);
        }
    }

    #[test]
    fn the_shared_prefix_reads_the_same_bytes_in_all_three_forms() {
        // The failure this catches: a wider form's field creeping into the
        // narrower one's offsets, which every single-form test would pass.
        let buf = synth_ranged(OPCODE_TEXTURE_VIEW_RANGED, TEXTURE_VIEW_RANGED_TOTAL_LEN);
        let o = op(&buf[..52], 0).expect("well formed");
        let r = texture_view_ranged(&o).expect("fits");

        let mut narrow = buf;
        narrow[0..4].copy_from_slice(&OPCODE_TEXTURE_VIEW.to_le_bytes());
        narrow[4..8].copy_from_slice(&TEXTURE_VIEW_TOTAL_LEN.to_le_bytes());
        let o = op(&narrow[..20], 0).expect("well formed");
        let v = texture_view(&o).expect("fits");

        assert_eq!(v.object_ref.get(), r.object_ref.get());
        assert_eq!(v.base_texture_ref.get(), r.base_texture_ref.get());
        assert_eq!(v.pixel_format.get(), r.pixel_format.get());

        let mut wide = buf;
        wide[0..4].copy_from_slice(&OPCODE_TEXTURE_VIEW_SWIZZLE.to_le_bytes());
        wide[4..8].copy_from_slice(&TEXTURE_VIEW_SWIZZLE_TOTAL_LEN.to_le_bytes());
        let o = op(&wide, 0).expect("well formed");
        let s = texture_view_swizzle(&o).expect("fits");
        assert_eq!(s.ranged.object_ref.get(), r.object_ref.get());
        assert_eq!(s.ranged.level_base.get(), r.level_base.get());
        assert_eq!(s.ranged.slice_count.get(), r.slice_count.get());
    }

    #[test]
    fn the_format_only_form_has_no_field_over_the_bytes_it_never_writes() {
        // Payload +010 is the ranged form's texture type. In an opcode-7
        // record the serializer leaves it alone, so it is the guest's ring —
        // and the body must simply end before it. If a `texture_type` field
        // ever appears on `TextureViewBody`, this size check fails first.
        assert_eq!(size_of::<TextureViewBody>(), 10);
    }

    #[test]
    fn the_ranged_form_keeps_its_four_range_numbers_apart() {
        let buf = synth_ranged(OPCODE_TEXTURE_VIEW_RANGED, TEXTURE_VIEW_RANGED_TOTAL_LEN);
        let o = op(&buf[..52], 0).expect("well formed");
        let r = texture_view_ranged(&o).expect("fits");
        assert_eq!(r.object_ref.get(), 57);
        assert_eq!(r.base_texture_ref.get(), 4242);
        assert_eq!(r.pixel_format.get(), 70);
        assert_eq!(r.texture_type.get(), 2);
        assert_eq!(r.level_base.get(), 3);
        assert_eq!(r.level_count.get(), 2);
        assert_eq!(r.slice_base.get(), 5);
        assert_eq!(r.slice_count.get(), 4);
    }

    #[test]
    fn the_swizzle_channels_are_read_in_wire_order() {
        let buf = synth_ranged(OPCODE_TEXTURE_VIEW_SWIZZLE, TEXTURE_VIEW_SWIZZLE_TOTAL_LEN);
        let o = op(&buf, 0).expect("well formed");
        let s = texture_view_swizzle(&o).expect("fits");
        assert_eq!(s.swizzle.red, 2);
        assert_eq!(s.swizzle.green, 3);
        assert_eq!(s.swizzle.blue, 4);
        assert_eq!(s.swizzle.alpha, 5);
    }

    #[test]
    fn only_the_three_view_opcodes_answer_the_predicate() {
        for op in [
            OPCODE_TEXTURE_VIEW,
            OPCODE_TEXTURE_VIEW_RANGED,
            OPCODE_TEXTURE_VIEW_SWIZZLE,
        ] {
            assert!(is_texture_view(op), "{op:#x}");
        }
        // 9 and 0x0c are the buffer-backed and IOSurface-backed textures, which
        // arrive through the same object tag and are not views.
        for op in [0u32, 1, 3, 4, 9, 0x0c, 0x15, 0x1a, 0x1c] {
            assert!(!is_texture_view(op), "{op:#x}");
        }
    }

    #[test]
    fn a_truncated_record_of_each_form_is_refused_rather_than_read_short() {
        let narrow = synth_ranged(OPCODE_TEXTURE_VIEW, TEXTURE_VIEW_TOTAL_LEN);
        let on = op(&narrow[..20], 0).expect("well formed");
        for (payload, need, have) in [(4usize, 10usize, 4usize), (9, 10, 9)] {
            let short = Op {
                header: on.header,
                payload: &on.payload[..payload],
                offset: 0,
            };
            assert!(
                matches!(texture_view(&short), Err(WireError::Short { need: n, have: h })
                    if n == need && h == have),
                "payload {payload}"
            );
        }

        // A ranged body does not fit in a format-only record's payload, which
        // is the refusal that matters: the three share a prefix, so a decoder
        // that dispatched on the wrong opcode would otherwise read four range
        // numbers out of twelve bytes.
        let wide = synth_ranged(OPCODE_TEXTURE_VIEW_RANGED, TEXTURE_VIEW_RANGED_TOTAL_LEN);
        let ow = op(&wide[..52], 0).expect("well formed");
        let short = Op {
            header: ow.header,
            payload: &ow.payload[..12],
            offset: 0,
        };
        assert!(matches!(
            texture_view_ranged(&short),
            Err(WireError::Short { need: 44, have: 12 })
        ));

        let sw = synth_ranged(OPCODE_TEXTURE_VIEW_SWIZZLE, TEXTURE_VIEW_SWIZZLE_TOTAL_LEN);
        let os = op(&sw, 0).expect("well formed");
        let short = Op {
            header: os.header,
            payload: &os.payload[..44],
            offset: 0,
        };
        assert!(matches!(
            texture_view_swizzle(&short),
            Err(WireError::Short { need: 48, have: 44 })
        ));
    }
}
