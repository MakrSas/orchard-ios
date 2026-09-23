//! Lifting the transfer records.
//!
//! # Nine records, and the field orders do not rhyme
//!
//! `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` puts **both
//! refs first** and then the three `u64`, which is not the selector's order.
//! `copyFromTexture:…toBuffer:` narrows its options word to sixteen bits where
//! its buffer-to-texture sibling uses thirty-two. The region copy's `options:`
//! form is a different opcode at a different length while the other two copies'
//! `options:` forms share theirs.
//!
//! None of that is derivable from the selector, which is why every field here
//! comes from a `reims-vgpu-wire` view rather than from a hand-written offset:
//! the views are what the fixtures pin, and an offset written here would be a
//! second reading of the same bytes with nothing comparing the two.

use super::{no_record, short, DecodeRefusal};
use crate::blit::BlitKind;
use crate::closure::Rail;
use reims_vgpu_wire::op::Op;
use reims_vgpu_wire::ops::blit as wire;

/// A copy origin, as the record carries it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Origin {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

/// A copy size, as the record carries it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Size {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

/// A texture endpoint, with the guest's ref.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureEndpoint {
    pub texture_ref: u32,
    pub slice: u16,
    pub level: u16,
    pub origin: Origin,
}

/// A buffer-to-buffer copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferToBuffer {
    pub source_ref: u32,
    pub source_offset: u64,
    pub dest_ref: u32,
    pub dest_offset: u64,
    pub size: u64,
}

/// A buffer-to-texture copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferToTexture {
    pub source_ref: u32,
    pub source_offset: u64,
    pub bytes_per_row: u64,
    pub bytes_per_image: u64,
    pub size: Size,
    pub dest: TextureEndpoint,
    pub options: u32,
}

/// A texture-to-buffer copy.
///
/// `options` is sixteen bits here and thirty-two on [`BufferToTexture`]; the
/// widths are the records', not a choice made here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureToBuffer {
    pub source: TextureEndpoint,
    pub size: Size,
    pub dest_ref: u32,
    pub dest_offset: u64,
    pub bytes_per_row: u64,
    pub bytes_per_image: u64,
    pub options: u16,
}

/// A texture-to-texture region copy, with or without an `options:` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureRegion {
    pub source: TextureEndpoint,
    pub dest: TextureEndpoint,
    pub size: Size,
    /// Zero for the plain opcode, which does not carry the field at all.
    pub options: u32,
}

/// A slice-and-level range copy between two textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureSlices {
    pub source_ref: u32,
    pub source_slice: u16,
    pub source_level: u16,
    pub dest_ref: u32,
    pub dest_slice: u16,
    pub dest_level: u16,
    pub slice_count: u16,
    pub level_count: u16,
}

/// A buffer fill, in either of its two pattern widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillBuffer {
    pub buffer_ref: u32,
    pub location: u64,
    pub length: u64,
    /// One byte for `value:`, four for `pattern4:`. Widened here and
    /// distinguished by [`BlitRecord::kind`], because the two records are
    /// byte-identical apart from the width of this field and its opcode.
    pub pattern: FillPattern,
}

/// `generateMipmapsForTexture:`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerateMipmaps {
    pub texture_ref: u32,
}

/// One lifted transfer record.
///
/// Each variant carries **one named payload** rather than inline fields, so a
/// consumer can take the record it handles by reference and cannot be handed a
/// different one. The executor arms below the lift are functions of those
/// payloads, which is what keeps a nine-record dispatch from becoming a flat
/// struct with nine records' fields in it and a kind tag saying which are live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlitRecord {
    BufferToBuffer(BufferToBuffer),
    BufferToTexture(BufferToTexture),
    TextureToBuffer(TextureToBuffer),
    TextureRegion(TextureRegion),
    TextureSlices(TextureSlices),
    FillBuffer(FillBuffer),
    GenerateMipmaps(GenerateMipmaps),
}

/// What a fill writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillPattern {
    Byte(u8),
    Pattern4(u32),
}

impl BlitRecord {
    /// Which record this is.
    #[must_use]
    pub const fn kind(&self) -> BlitKind {
        match self {
            Self::BufferToBuffer(_) => BlitKind::BufferToBuffer,
            Self::BufferToTexture(_) => BlitKind::BufferToTexture,
            Self::TextureToBuffer(_) => BlitKind::TextureToBuffer,
            Self::TextureRegion(r) => {
                if r.options == 0 {
                    BlitKind::TextureRegion
                } else {
                    BlitKind::TextureRegionOptions
                }
            }
            Self::TextureSlices(_) => BlitKind::TextureSlices,
            Self::FillBuffer(FillBuffer {
                pattern: FillPattern::Byte(_),
                ..
            }) => BlitKind::FillBuffer,
            Self::FillBuffer(FillBuffer {
                pattern: FillPattern::Pattern4(_),
                ..
            }) => BlitKind::FillBufferPattern4,
            Self::GenerateMipmaps(_) => BlitKind::GenerateMipmaps,
        }
    }

    /// The guest refs this record reads from and writes to.
    ///
    /// Nine records name their endpoints under nine different field names, and
    /// a reader that wants "what did this copy touch" — a failure line, a
    /// residency census — otherwise has to restate the whole match to find out.
    /// `None` is an end the record does not have: a fill has no source, and
    /// `generateMipmapsForTexture:` reads and writes the same texture, so both
    /// ends are it.
    #[must_use]
    pub const fn refs(&self) -> (Option<u32>, Option<u32>) {
        match self {
            Self::BufferToBuffer(r) => (Some(r.source_ref), Some(r.dest_ref)),
            Self::BufferToTexture(r) => (Some(r.source_ref), Some(r.dest.texture_ref)),
            Self::TextureToBuffer(r) => (Some(r.source.texture_ref), Some(r.dest_ref)),
            Self::TextureRegion(r) => (Some(r.source.texture_ref), Some(r.dest.texture_ref)),
            Self::TextureSlices(r) => (Some(r.source_ref), Some(r.dest_ref)),
            Self::FillBuffer(r) => (None, Some(r.buffer_ref)),
            Self::GenerateMipmaps(r) => (Some(r.texture_ref), Some(r.texture_ref)),
        }
    }
}

/// Lift a transfer record out of its bytes.
///
/// Refuses an opcode that is not a transfer at all, one whose contract is not
/// established, and a record too short for the body its opcode requires. It
/// does not refuse anything else: a value out of range is the model's question,
/// and refusing here would report a wire error for a semantic one.
pub fn decode(op: &Op<'_>) -> Result<BlitRecord, DecodeRefusal> {
    let opcode = op.opcode();
    let Some(kind) = BlitKind::of_opcode(opcode) else {
        return Err(no_record(Rail::Blit, opcode));
    };
    let have = op.payload.len();
    let fail = |need: usize| short(Rail::Blit, opcode, have, need);

    Ok(match kind {
        BlitKind::BufferToBuffer => {
            let r = wire::copy_buffer_to_buffer(op)
                .map_err(|_| fail(core::mem::size_of::<wire::BufferToBuffer>()))?;
            BlitRecord::BufferToBuffer(BufferToBuffer {
                source_ref: r.source_ref.get(),
                source_offset: r.source_offset.get(),
                dest_ref: r.dest_ref.get(),
                dest_offset: r.dest_offset.get(),
                size: r.size.get(),
            })
        }
        BlitKind::BufferToTexture => {
            let r = wire::copy_buffer_to_texture(op)
                .map_err(|_| fail(core::mem::size_of::<wire::CopyBufferToTexture>()))?;
            BlitRecord::BufferToTexture(BufferToTexture {
                source_ref: r.source_ref.get(),
                source_offset: r.source_offset.get(),
                bytes_per_row: r.source_bytes_per_row.get(),
                bytes_per_image: r.source_bytes_per_image.get(),
                size: Size {
                    width: r.size_width.get(),
                    height: r.size_height.get(),
                    depth: r.size_depth.get(),
                },
                dest: TextureEndpoint {
                    texture_ref: r.dest_ref.get(),
                    slice: r.dest_slice.get(),
                    level: r.dest_level.get(),
                    origin: Origin {
                        x: r.dest_origin_x.get(),
                        y: r.dest_origin_y.get(),
                        z: r.dest_origin_z.get(),
                    },
                },
                options: r.options.get(),
            })
        }
        BlitKind::TextureToBuffer => {
            let r = wire::copy_texture_to_buffer(op)
                .map_err(|_| fail(core::mem::size_of::<wire::CopyTextureToBuffer>()))?;
            BlitRecord::TextureToBuffer(TextureToBuffer {
                source: TextureEndpoint {
                    texture_ref: r.source_ref.get(),
                    slice: r.source_slice.get(),
                    level: r.source_level.get(),
                    origin: Origin {
                        x: r.source_origin_x.get(),
                        y: r.source_origin_y.get(),
                        z: r.source_origin_z.get(),
                    },
                },
                size: Size {
                    width: r.size_width.get(),
                    height: r.size_height.get(),
                    depth: r.size_depth.get(),
                },
                dest_ref: r.dest_ref.get(),
                dest_offset: r.dest_offset.get(),
                bytes_per_row: r.dest_bytes_per_row.get(),
                bytes_per_image: r.dest_bytes_per_image.get(),
                options: r.options.get(),
            })
        }
        BlitKind::TextureRegion | BlitKind::TextureRegionOptions => {
            let (region, options) = if kind == BlitKind::TextureRegionOptions {
                let r = wire::copy_texture_region_options(op)
                    .map_err(|_| fail(core::mem::size_of::<wire::CopyTextureRegionOptions>()))?;
                (&r.region, r.options.get())
            } else {
                let r = wire::copy_texture_region(op)
                    .map_err(|_| fail(core::mem::size_of::<wire::CopyTextureRegion>()))?;
                (r, 0)
            };
            BlitRecord::TextureRegion(TextureRegion {
                source: TextureEndpoint {
                    texture_ref: region.source_ref.get(),
                    slice: region.source_slice.get(),
                    level: region.source_level.get(),
                    origin: Origin {
                        x: region.source_origin_x.get(),
                        y: region.source_origin_y.get(),
                        z: region.source_origin_z.get(),
                    },
                },
                dest: TextureEndpoint {
                    texture_ref: region.dest_ref.get(),
                    slice: region.dest_slice.get(),
                    level: region.dest_level.get(),
                    origin: Origin {
                        x: region.dest_origin_x.get(),
                        y: region.dest_origin_y.get(),
                        z: region.dest_origin_z.get(),
                    },
                },
                size: Size {
                    width: region.size_width.get(),
                    height: region.size_height.get(),
                    depth: region.size_depth.get(),
                },
                options,
            })
        }
        BlitKind::TextureSlices => {
            let r = wire::copy_texture_slices(op)
                .map_err(|_| fail(core::mem::size_of::<wire::CopyTextureSlices>()))?;
            BlitRecord::TextureSlices(TextureSlices {
                source_ref: r.source_ref.get(),
                source_slice: r.source_slice.get(),
                source_level: r.source_level.get(),
                dest_ref: r.dest_ref.get(),
                dest_slice: r.dest_slice.get(),
                dest_level: r.dest_level.get(),
                slice_count: r.slice_count.get(),
                level_count: r.level_count.get(),
            })
        }
        BlitKind::FillBuffer => {
            let r = wire::fill_buffer(op)
                .map_err(|_| fail(core::mem::size_of::<wire::FillBuffer>()))?;
            BlitRecord::FillBuffer(FillBuffer {
                buffer_ref: r.buffer_ref.get(),
                location: r.range_location.get(),
                length: r.range_length.get(),
                pattern: FillPattern::Byte(r.value),
            })
        }
        BlitKind::FillBufferPattern4 => {
            let r = wire::fill_buffer_pattern4(op)
                .map_err(|_| fail(core::mem::size_of::<wire::FillBufferPattern4>()))?;
            BlitRecord::FillBuffer(FillBuffer {
                buffer_ref: r.buffer_ref.get(),
                location: r.range_location.get(),
                length: r.range_length.get(),
                pattern: FillPattern::Pattern4(r.pattern.get()),
            })
        }
        BlitKind::GenerateMipmaps => {
            let r = wire::object_ref(op).map_err(|_| fail(core::mem::size_of::<wire::Ref>()))?;
            BlitRecord::GenerateMipmaps(GenerateMipmaps {
                texture_ref: r.object_ref.get(),
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use reims_vgpu_wire::op::{op, OP_HEADER_LEN};

    /// Build a record with `opcode` and `payload`, framed as the serializer
    /// frames one.
    fn record(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let total = (OP_HEADER_LEN + payload.len()) as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&total.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn decode_bytes(bytes: &[u8]) -> Result<BlitRecord, DecodeRefusal> {
        let view = op(bytes, 0).expect("framed");
        decode(&view)
    }

    fn u32s(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A buffer copy's refs come first and its three `u64` follow, which is not
    /// the selector's order. The fixture values are all distinct so a crossed
    /// field cannot read back correct.
    #[test]
    fn a_buffer_copy_lifts_its_fields_in_the_records_order() {
        let mut payload = u32s(&[5151, 5252]);
        for value in [0x1111u64, 0x2222, 0x3333] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let bytes = record(wire::OPCODE_COPY_BUFFER_TO_BUFFER, &payload);
        assert_eq!(
            decode_bytes(&bytes),
            Ok(BlitRecord::BufferToBuffer(BufferToBuffer {
                source_ref: 5151,
                source_offset: 0x1111,
                dest_ref: 5252,
                dest_offset: 0x2222,
                size: 0x3333,
            }))
        );
    }

    /// The two fill forms are byte-identical apart from the width of their last
    /// field, so the opcode is what tells them apart — and the lifted record
    /// keeps that difference rather than widening both to a word.
    #[test]
    fn the_two_fills_are_told_apart_by_opcode_and_stay_apart() {
        let mut narrow = u32s(&[5151]);
        narrow.extend_from_slice(&0x1100u64.to_le_bytes());
        narrow.extend_from_slice(&0x2200u64.to_le_bytes());
        let mut wide = narrow.clone();
        narrow.push(0x5a);
        wide.extend_from_slice(&0x89ab_cdefu32.to_le_bytes());

        let byte = decode_bytes(&record(wire::OPCODE_FILL_BUFFER, &narrow)).expect("lifted");
        let word = decode_bytes(&record(wire::OPCODE_FILL_BUFFER_PATTERN4, &wide)).expect("lifted");
        assert_eq!(byte.kind(), BlitKind::FillBuffer);
        assert_eq!(word.kind(), BlitKind::FillBufferPattern4);
        assert_ne!(byte, word);
        match (byte, word) {
            (
                BlitRecord::FillBuffer(FillBuffer {
                    pattern: FillPattern::Byte(b),
                    location,
                    ..
                }),
                BlitRecord::FillBuffer(FillBuffer {
                    pattern: FillPattern::Pattern4(w),
                    ..
                }),
            ) => {
                assert_eq!(b, 0x5a);
                assert_eq!(w, 0x89ab_cdef);
                assert_eq!(location, 0x1100);
            }
            _ => panic!("wrong shapes"),
        }
    }

    /// The region copy's `options:` form is a different opcode at a longer
    /// length, and the plain form carries no options field at all — so the
    /// plain one lifts a zero rather than reading four bytes that are not
    /// there.
    #[test]
    fn the_region_copys_two_opcodes_lift_to_one_record_with_and_without_options() {
        let mut region = u32s(&[7171, 7272]);
        for value in [0x11u64, 0x22, 0x33, 0x44, 0x55, 1, 0x66, 0x77, 0x88] {
            region.extend_from_slice(&value.to_le_bytes());
        }
        for value in [9u16, 10, 11, 12] {
            region.extend_from_slice(&value.to_le_bytes());
        }
        let plain =
            decode_bytes(&record(wire::OPCODE_COPY_TEXTURE_REGION, &region)).expect("lifted");
        assert_eq!(plain.kind(), BlitKind::TextureRegion);

        let mut with_options = region.clone();
        with_options.extend_from_slice(&4u32.to_le_bytes());
        let lifted = decode_bytes(&record(
            wire::OPCODE_COPY_TEXTURE_REGION_OPTIONS,
            &with_options,
        ))
        .expect("lifted");
        assert_eq!(lifted.kind(), BlitKind::TextureRegionOptions);

        match (plain, lifted) {
            (
                BlitRecord::TextureRegion(TextureRegion {
                    source, options: a, ..
                }),
                BlitRecord::TextureRegion(TextureRegion { options: b, .. }),
            ) => {
                assert_eq!(a, 0, "the plain form carries no options field");
                assert_eq!(b, 4);
                assert_eq!(source.texture_ref, 7171);
                assert_eq!(source.slice, 9);
                assert_eq!(source.origin.x, 0x11);
            }
            _ => panic!("wrong shapes"),
        }
    }

    /// `copyFromTexture:toBuffer:` reads two bytes of `options`, not four.
    ///
    /// It is the one copy record that narrows the field, and reading it four
    /// bytes wide once cost every depth-aspect copy on the rail: the two bytes
    /// past it belong to no field, so on a guest's wire they hold whatever the
    /// command ring last contained. They are poisoned here to stand in for
    /// that, and the sibling that really is four bytes wide is read beside it
    /// so this stays a per-record narrowing rather than a family rule.
    #[test]
    fn a_texture_to_buffer_copy_reads_no_byte_past_its_options() {
        const CTB_OPTIONS: usize = core::mem::offset_of!(wire::CopyTextureToBuffer, options);
        const CTB_LEN: usize = core::mem::size_of::<wire::CopyTextureToBuffer>();
        for written in [0u16, 4] {
            let mut payload = vec![0u8; CTB_LEN];
            for b in payload.iter_mut().skip(CTB_OPTIONS + 2) {
                *b = 0xAA;
            }
            payload[CTB_OPTIONS..CTB_OPTIONS + 2].copy_from_slice(&written.to_le_bytes());
            let bytes = record(wire::OPCODE_COPY_TEXTURE_TO_BUFFER, &payload);
            match decode_bytes(&bytes).expect("a well-formed copy lifts") {
                BlitRecord::TextureToBuffer(r) => assert_eq!(
                    r.options, written,
                    "options picked up a byte the serializer never wrote"
                ),
                other => panic!("{other:?}"),
            }
        }

        const CBT_OPTIONS: usize = core::mem::offset_of!(wire::CopyBufferToTexture, options);
        const CBT_LEN: usize = core::mem::size_of::<wire::CopyBufferToTexture>();
        let mut payload = vec![0u8; CBT_LEN];
        payload[CBT_OPTIONS..CBT_OPTIONS + 4].copy_from_slice(&0x0001_0000u32.to_le_bytes());
        match decode_bytes(&record(wire::OPCODE_COPY_BUFFER_TO_TEXTURE, &payload))
            .expect("the sibling lifts")
        {
            BlitRecord::BufferToTexture(r) => assert_eq!(r.options, 0x0001_0000),
            other => panic!("{other:?}"),
        }
    }

    /// The `options:` region copy is four bytes longer than its plain sibling,
    /// and a record written at the plain length is refused rather than having
    /// its options word read out of the bytes after it.
    #[test]
    fn a_region_copy_with_options_is_refused_at_the_plain_forms_length() {
        let plain_len = core::mem::size_of::<wire::CopyTextureRegion>();
        let with_options = core::mem::size_of::<wire::CopyTextureRegionOptions>();
        assert_eq!(with_options - plain_len, 4);
        assert!(decode_bytes(&record(
            wire::OPCODE_COPY_TEXTURE_REGION_OPTIONS,
            &vec![0u8; with_options]
        ))
        .is_ok());
        assert!(matches!(
            decode_bytes(&record(
                wire::OPCODE_COPY_TEXTURE_REGION_OPTIONS,
                &vec![0u8; plain_len]
            )),
            Err(DecodeRefusal::Short { .. })
        ));
    }

    /// The slice-and-level copy carries six `u16` after its two refs, and the
    /// record keeps them in the order the wire writes them.
    ///
    /// Every fixture value is distinct so a crossed pair cannot read back
    /// correct — the base slice and the base level of one end are adjacent
    /// halves of one word, and swapping them copies a real region that is not
    /// the one the guest named.
    #[test]
    fn a_slice_and_level_copy_lifts_its_six_counts_in_the_records_order() {
        let mut payload = u32s(&[2121, 3131]);
        for value in [1u16, 2, 3, 4, 5, 6] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        match decode_bytes(&record(wire::OPCODE_COPY_TEXTURE_SLICES, &payload))
            .expect("the record lifts")
        {
            BlitRecord::TextureSlices(r) => assert_eq!(
                r,
                TextureSlices {
                    source_ref: 2121,
                    source_slice: 1,
                    source_level: 2,
                    dest_ref: 3131,
                    dest_slice: 3,
                    dest_level: 4,
                    slice_count: 5,
                    level_count: 6,
                }
            ),
            other => panic!("{other:?}"),
        }
    }

    /// A record too short for its body is refused with the numbers in it, not
    /// read past its end.
    #[test]
    fn a_short_record_is_refused_with_both_lengths() {
        let bytes = record(wire::OPCODE_COPY_BUFFER_TO_BUFFER, &u32s(&[5151]));
        let err = decode_bytes(&bytes).expect_err("too short");
        match err {
            DecodeRefusal::Short {
                rail,
                opcode,
                have,
                need,
            } => {
                assert_eq!(rail, Rail::Blit);
                assert_eq!(opcode, wire::OPCODE_COPY_BUFFER_TO_BUFFER);
                assert_eq!(have, 4);
                assert!(need > have);
            }
            other => panic!("{other:?}"),
        }
    }

    /// An opcode whose contract is open and one that does not exist produce
    /// different refusals. Both are dropped work; only one of them is a bug in
    /// the stream.
    #[test]
    fn an_open_contract_and_an_unknown_opcode_report_differently() {
        // `fillTexture:…bytes:` is a real blit opcode with an unresolved row.
        let unjudged = decode_bytes(&record(0x140, &u32s(&[0]))).expect_err("unjudged");
        assert_eq!(
            unjudged,
            DecodeRefusal::Unjudged {
                rail: Rail::Blit,
                opcode: 0x140
            }
        );
        let unknown = decode_bytes(&record(0x1ff, &u32s(&[0]))).expect_err("unknown");
        assert_eq!(
            unknown,
            DecodeRefusal::UnknownOpcode {
                rail: Rail::Blit,
                opcode: 0x1ff
            }
        );
        assert_ne!(unjudged.reason(), unknown.reason());
    }

    /// Every transfer kind is reachable from bytes. A kind the decoder cannot
    /// produce is a payload nothing can construct.
    #[test]
    fn every_transfer_kind_decodes_from_bytes() {
        let cases: &[(u32, usize)] = &[
            (wire::OPCODE_COPY_BUFFER_TO_TEXTURE, 88),
            (wire::OPCODE_COPY_BUFFER_TO_BUFFER, 32),
            (wire::OPCODE_COPY_TEXTURE_TO_BUFFER, 88),
            (wire::OPCODE_COPY_TEXTURE_REGION, 88),
            (wire::OPCODE_COPY_TEXTURE_REGION_OPTIONS, 92),
            (wire::OPCODE_COPY_TEXTURE_SLICES, 20),
            (wire::OPCODE_FILL_BUFFER, 24),
            (wire::OPCODE_FILL_BUFFER_PATTERN4, 24),
            (wire::OPCODE_GENERATE_MIPMAPS, 4),
        ];
        let mut kinds = Vec::new();
        for &(opcode, payload_len) in cases {
            let mut payload = vec![0u8; payload_len];
            if opcode == wire::OPCODE_COPY_TEXTURE_REGION_OPTIONS {
                // A zero options word makes the record indistinguishable from
                // its plain sibling, which is the whole reason `kind` reads the
                // value rather than the opcode.
                payload[payload_len - 4] = 4;
            }
            let bytes = record(opcode, &payload);
            let lifted =
                decode_bytes(&bytes).unwrap_or_else(|e| panic!("{opcode:#x} did not lift: {e:?}"));
            kinds.push(lifted.kind());
        }
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(kinds.len(), BlitKind::ALL.len());
    }
}
