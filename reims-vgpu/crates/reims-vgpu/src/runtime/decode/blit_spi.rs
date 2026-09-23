//! The blit-rail records the closure ledger has **not settled**.
//!
//! Four opcodes: `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`,
//! `resetCommandsInBuffer:withRange:`, and the two
//! `fillTexture:level:slice:region:…` forms that `-setSupportsBlitEncoderSPI:`
//! gates.
//!
//! # Why they are decoded here and not in `reims-vgpu-protocol`
//!
//! An unresolved row has no established contract, so the layer that assigns
//! meaning to a wire tag may not give it a shape — `protocol::decode::blit`
//! refuses all four as `Unjudged`, which is the correct answer for a decoder
//! that promises a lifted record only where one is established.
//!
//! This device is not making that promise. It decodes these four in order to
//! *decline* them by name and with their extents, and that decline's count is
//! the measurement that will settle the row: whether a reset the device drops
//! leaves live commands a workload actually re-runs, and whether a texture fill
//! it drops is a region the guest reads back. A generic "unsupported opcode"
//! line would delete the instrument.
//!
//! So this module is deliberately narrow. It reads the fields the decline
//! reports and nothing else — no executor takes a value from here, and when a
//! row is settled the record moves to the protocol crate and this arm goes with
//! it.

use reims_vgpu_wire::ops::blit as wire;

/// Where a [`UnsettledRecord::TextureFill`] takes the value it writes.
///
/// The two forms are separate opcodes with different bodies, and which one a
/// workload issues decides which converter an executor would need — so the
/// decline keeps them apart rather than counting "a texture fill".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillSource {
    /// A clear colour the record carries, in the pixel format named beside it.
    Color,
    /// A pixel pattern the serializer staged into a buffer rather than putting
    /// it on the wire. The record carries no pixel data.
    Bytes,
}

/// A copy size, as the record carries it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Size {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

/// One decoded unsettled record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsettledRecord {
    /// `resetCommandsInBuffer:withRange:` and
    /// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`.
    ///
    /// One variant for both, because the decline is the same claim about both —
    /// a later `executeCommandsInBuffer:` runs commands this device left as it
    /// found them — and the opcode is what says which. `destination` is `None`
    /// for the reset, which names one buffer.
    IcbMutation {
        icb_ref: u32,
        destination: Option<u32>,
        range_location: u64,
        range_length: u64,
    },
    /// `fillTexture:level:slice:region:color:pixelFormat:` and
    /// `fillTexture:level:slice:region:bytes:length:`.
    TextureFill {
        source: FillSource,
        texture: u32,
        level: u16,
        slice: u16,
        size: Size,
    },
}

/// Why this decoder refused a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    ErrShort,
    ErrUnknownOpcode,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs keep the `blit_decode_` prefix the rail's decoder reported under,
    /// so a census taken across the cutover reads continuously.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ErrShort => "blit_decode_short",
            Self::ErrUnknownOpcode => "blit_decode_unknown_opcode",
        })
    }
}

/// Decode one unsettled blit-rail record.
///
/// # Errors
///
/// [`DecodeStatus::ErrUnknownOpcode`] for anything but the four rows this
/// module owns — including the settled ones, which have decoders of their own —
/// and [`DecodeStatus::ErrShort`] for a record too short for its body.
pub fn decode(command: &[u8]) -> Result<UnsettledRecord, DecodeStatus> {
    let op = reims_vgpu_wire::op(command, 0).map_err(|_| DecodeStatus::ErrShort)?;
    let command_length = op.length() as usize;
    let want = |need: u32| {
        if command_length == need as usize {
            Ok(())
        } else {
            Err(DecodeStatus::ErrShort)
        }
    };
    match op.opcode() {
        wire::OPCODE_RESET_ICB => {
            want(wire::ICB_RANGE_TOTAL_LEN)?;
            let r = wire::icb_range(&op).map_err(|_| DecodeStatus::ErrShort)?;
            Ok(UnsettledRecord::IcbMutation {
                icb_ref: r.icb_ref.get(),
                destination: None,
                range_location: r.range_location.get(),
                range_length: r.range_length.get(),
            })
        }
        wire::OPCODE_COPY_ICB => {
            want(wire::COPY_ICB_TOTAL_LEN)?;
            let c = wire::copy_icb(&op).map_err(|_| DecodeStatus::ErrShort)?;
            Ok(UnsettledRecord::IcbMutation {
                icb_ref: c.source_ref.get(),
                destination: Some(c.dest_ref.get()),
                range_location: c.range_location.get(),
                range_length: c.range_length.get(),
            })
        }
        wire::OPCODE_FILL_TEXTURE_COLOR => {
            want(wire::FILL_TEXTURE_COLOR_TOTAL_LEN)?;
            let f = wire::fill_texture_color(&op).map_err(|_| DecodeStatus::ErrShort)?;
            Ok(UnsettledRecord::TextureFill {
                source: FillSource::Color,
                texture: f.texture_ref.get(),
                level: f.level.get(),
                slice: f.slice.get(),
                size: Size {
                    width: f.size_width.get(),
                    height: f.size_height.get(),
                    depth: f.size_depth.get(),
                },
            })
        }
        wire::OPCODE_FILL_TEXTURE_BYTES => {
            want(wire::FILL_TEXTURE_BYTES_TOTAL_LEN)?;
            let f = wire::fill_texture_bytes(&op).map_err(|_| DecodeStatus::ErrShort)?;
            Ok(UnsettledRecord::TextureFill {
                source: FillSource::Bytes,
                texture: f.texture_ref.get(),
                level: f.level.get(),
                slice: f.slice.get(),
                size: Size {
                    width: f.size_width.get(),
                    height: f.size_height.get(),
                    depth: f.size_depth.get(),
                },
            })
        }
        _ => Err(DecodeStatus::ErrUnknownOpcode),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(opcode: u32, body_words: usize) -> Vec<u8> {
        let total = (8 + body_words * 4) as u32;
        let mut v = Vec::with_capacity(total as usize);
        v.extend_from_slice(&opcode.to_le_bytes());
        v.extend_from_slice(&total.to_le_bytes());
        v.resize(total as usize, 0);
        v
    }

    fn body_words(total_len: u32) -> usize {
        ((total_len - 8) / 4) as usize
    }

    /// The reset names one buffer and the copy names two, and the decline reads
    /// the difference — a copy counted as a reset would lose the destination
    /// that is the thing left stale.
    #[test]
    fn the_two_icb_mutations_are_told_apart_by_whether_they_name_a_destination() {
        let reset = decode(&record(
            wire::OPCODE_RESET_ICB,
            body_words(wire::ICB_RANGE_TOTAL_LEN),
        ))
        .expect("decoded");
        let copy = decode(&record(
            wire::OPCODE_COPY_ICB,
            body_words(wire::COPY_ICB_TOTAL_LEN),
        ))
        .expect("decoded");
        match (reset, copy) {
            (
                UnsettledRecord::IcbMutation {
                    destination: None, ..
                },
                UnsettledRecord::IcbMutation {
                    destination: Some(_),
                    ..
                },
            ) => {}
            other => panic!("{other:?}"),
        }
    }

    /// Both fills store their region's **size before its origin**, reversing
    /// `MTLRegion` and the selector's own type encoding.
    ///
    /// The decline reports the extent, and an extent read from the origin's
    /// words is a plausible number that is not the region: it would say a
    /// dropped 4×4 fill was a 0×0 one and read as a healthy zero. Distinct
    /// values so a crossed pair cannot read back correct.
    #[test]
    fn both_texture_fills_read_their_extent_from_the_size_and_not_the_origin() {
        for (opcode, total) in [
            (
                wire::OPCODE_FILL_TEXTURE_COLOR,
                wire::FILL_TEXTURE_COLOR_TOTAL_LEN,
            ),
            (
                wire::OPCODE_FILL_TEXTURE_BYTES,
                wire::FILL_TEXTURE_BYTES_TOTAL_LEN,
            ),
        ] {
            let mut v = record(opcode, body_words(total));
            let p = 8;
            v[p..p + 4].copy_from_slice(&4242u32.to_le_bytes());
            v[p + 4..p + 6].copy_from_slice(&3u16.to_le_bytes());
            v[p + 6..p + 8].copy_from_slice(&5u16.to_le_bytes());
            for (i, value) in [0x44u64, 0x55, 0x66, 0x11, 0x22, 0x33].iter().enumerate() {
                let at = p + 8 + i * 8;
                v[at..at + 8].copy_from_slice(&value.to_le_bytes());
            }
            assert_eq!(
                decode(&v).expect("decoded"),
                UnsettledRecord::TextureFill {
                    source: if opcode == wire::OPCODE_FILL_TEXTURE_COLOR {
                        FillSource::Color
                    } else {
                        FillSource::Bytes
                    },
                    texture: 4242,
                    level: 3,
                    slice: 5,
                    size: Size {
                        width: 0x44,
                        height: 0x55,
                        depth: 0x66,
                    },
                },
                "opcode {opcode:#x} read its extent from the origin"
            );
        }
    }

    /// The colour form and the bytes form are separate opcodes with separate
    /// bodies, and the decline counts them separately because they need
    /// different executors.
    #[test]
    fn the_two_texture_fills_keep_their_sources_apart() {
        let color = decode(&record(
            wire::OPCODE_FILL_TEXTURE_COLOR,
            body_words(wire::FILL_TEXTURE_COLOR_TOTAL_LEN),
        ))
        .expect("decoded");
        let bytes = decode(&record(
            wire::OPCODE_FILL_TEXTURE_BYTES,
            body_words(wire::FILL_TEXTURE_BYTES_TOTAL_LEN),
        ))
        .expect("decoded");
        assert!(matches!(
            color,
            UnsettledRecord::TextureFill {
                source: FillSource::Color,
                ..
            }
        ));
        assert!(matches!(
            bytes,
            UnsettledRecord::TextureFill {
                source: FillSource::Bytes,
                ..
            }
        ));
    }

    /// `optimizeIndirectCommandBuffer:withRange:` shares the reset's body and
    /// its row **is** settled, so it is lifted by the protocol crate and must
    /// not be decodable here — a second decoder for a settled row is the second
    /// semantic model the plan refuses.
    #[test]
    fn the_settled_optimize_hint_is_not_a_record_this_decoder_owns() {
        assert_eq!(
            decode(&record(
                wire::OPCODE_OPTIMIZE_ICB,
                body_words(wire::ICB_RANGE_TOTAL_LEN)
            )),
            Err(DecodeStatus::ErrUnknownOpcode)
        );
    }

    /// Every transfer opcode has a decoder in `reims-vgpu-protocol`, so none of
    /// them is decodable here either.
    #[test]
    fn no_transfer_opcode_is_a_record_this_decoder_owns() {
        for opcode in [
            wire::OPCODE_COPY_BUFFER_TO_BUFFER,
            wire::OPCODE_COPY_BUFFER_TO_TEXTURE,
            wire::OPCODE_COPY_TEXTURE_TO_BUFFER,
            wire::OPCODE_COPY_TEXTURE_REGION,
            wire::OPCODE_COPY_TEXTURE_REGION_OPTIONS,
            wire::OPCODE_COPY_TEXTURE_SLICES,
            wire::OPCODE_FILL_BUFFER,
            wire::OPCODE_FILL_BUFFER_PATTERN4,
            wire::OPCODE_GENERATE_MIPMAPS,
        ] {
            assert_eq!(
                decode(&record(opcode, 24)),
                Err(DecodeStatus::ErrUnknownOpcode),
                "opcode {opcode:#x}"
            );
        }
    }

    /// A record **one byte** short of its body is refused rather than read past
    /// its end, and the lengths this decoder checks are the records' own.
    ///
    /// One byte and not one word: a length check written against a rounded
    /// size admits a record the guest sized four bytes smaller than the layout,
    /// and the last field then reads whatever the ring held.
    #[test]
    fn a_record_short_of_its_body_is_refused() {
        for (opcode, total) in [
            (wire::OPCODE_RESET_ICB, wire::ICB_RANGE_TOTAL_LEN),
            (wire::OPCODE_COPY_ICB, wire::COPY_ICB_TOTAL_LEN),
            (
                wire::OPCODE_FILL_TEXTURE_COLOR,
                wire::FILL_TEXTURE_COLOR_TOTAL_LEN,
            ),
            (
                wire::OPCODE_FILL_TEXTURE_BYTES,
                wire::FILL_TEXTURE_BYTES_TOTAL_LEN,
            ),
        ] {
            let mut short = record(opcode, body_words(total));
            short.pop();
            let len = short.len() as u32;
            short[4..8].copy_from_slice(&len.to_le_bytes());
            assert_eq!(
                decode(&short),
                Err(DecodeStatus::ErrShort),
                "opcode {opcode:#x}"
            );
        }
    }
}
