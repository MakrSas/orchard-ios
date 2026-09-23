//! Opcode `0x32` — create or reset a rasterization rate map.
//!
//! Two selectors, one opcode. `-[PGSerializer
//! newRasterizationRateMapWithDescriptor:allocator:]` allocates the ref and
//! `-resetRasterizationRateMapWithDescriptor:existingID:allocator:` takes it as
//! an argument; the records are otherwise identical, which is measured rather
//! than assumed — both fixtures are driven off the same descriptor and differ
//! only in that word.
//!
//! Gated on `-setSupportsRasterizationRateMap:`, which defaults off. See this
//! crate's `AGENTS.md`: a selector measured silent at the default capability
//! state says nothing about Apple.
//!
//! # Layout
//!
//! **Variable length.** The record grows with the layer count and with each
//! layer's sample counts:
//!
//! ```text
//! payload +000  u32  object_ref
//! payload +004  u16  screen_width
//! payload +006  u16  screen_height
//! payload +008  u32  layer_count
//! payload +012  u32  unidentified_u32_a   reads 2 on every case
//! payload +016  u32  unidentified_u32_b   reads 0 on every case
//! payload +020  ..   layer_count x { u16 sample_width; u16 sample_height }
//!               ..   then, per layer in order:
//!                      sample_width  f32  horizontal quality
//!                      sample_height f32  vertical quality
//! ```
//!
//! Three lengths were captured and all three agree with that arithmetic: one
//! layer at 2x2 is 64 bytes, one layer at 4x3 is 76, two layers at 2x2 and 3x4
//! is 96.
//!
//! # The declared length is sixteen bytes longer than the record
//!
//! Every capture, at every layer and sample count, declares exactly sixteen
//! bytes more than the serializer writes — and the oracle's complementary-fill
//! measurement says those sixteen are **never written**. On a real wire they
//! hold whatever the guest's ring last contained, so nothing may read them.
//! [`quality_span`] returns the written extent, and it is deliberately not the
//! record length.
//!
//! `label` was the obvious candidate for what lives there and it is not:
//! `serializer_new_rasterization_rate_map_labelled` sets one and produces a
//! record byte-identical to the baseline apart from the ref. What would settle
//! it is a descriptor property this capture has no way to set — the tail is the
//! same size regardless of layer count, so it is not per-layer.
//!
//! # How the layout was derived
//!
//! Everything this record carries is a count of something else, so one
//! descriptor cannot separate the fields: a 64x64 screen with one layer at 2x2
//! puts 64, 1 and 2 on the wire and any of the three could be any field. Four
//! cases move one thing each — screen size to 320x200 (asymmetric, so width
//! cannot be read as height), sample count to 4x3, layer count to 2, and the
//! quality floats to eight distinct values. See `oracle/oracle.m`, the
//! `serializer_new_rasterization_rate_map*` family.
//!
//! The qualities are set explicitly in every case rather than left at whatever
//! `-initWithSampleCount:` leaves, because at the default they read `0.0` — and
//! a field that is zero in every fixture cannot be told from one the serializer
//! never writes.

use crate::le::{F32le, U16le, U32le};
use crate::op::Op;
use crate::view::{split, view_slice, Wire, WireError};

/// Opcode for rate-map creation and reset.
pub const OPCODE_NEW_RASTERIZATION_RATE_MAP: u32 = 0x32;

/// Bytes the record declares past its last written byte, at every size.
///
/// Named rather than inlined because it is the difference between the record's
/// declared length and the extent a reader may look at.
pub const UNWRITTEN_TAIL_LEN: usize = 16;

/// The fixed head of the record.
#[repr(C)]
#[derive(Debug)]
pub struct RateMapHead {
    /// The ref the serializer allocated, or — on the reset form — the one the
    /// caller passed. Fixture `serializer_reset_rasterization_rate_map` passes
    /// 6767 and reads it back here.
    pub object_ref: U32le,
    /// `MTLRasterizationRateMapDescriptor.screenSize`, narrowed to `u16` each
    /// from the `MTLSize` the API declares. 320x200 reads `0x0140`/`0x00c8`,
    /// which is what shows the pair is two `u16` and not one `u32`.
    pub screen_width: U16le,
    pub screen_height: U16le,
    pub layer_count: U32le,
    /// Reads 2 on every case, including the two-layer one and the 4x3 one, so
    /// it is neither the layer count nor a sample count.
    ///
    /// To settle it: `MTLRasterizationRateMap` exposes a `physicalGranularity`
    /// the descriptor does not, and a device whose granularity differs would
    /// move this if that is what it is. This host has one GPU.
    pub unidentified_u32_a: U32le,
    /// Reads 0 on every case. The serializer does write it — the
    /// complementary-fill measurement says so — so it is a zero it chose, not
    /// one it left. No descriptor property moves it.
    pub unidentified_u32_b: U32le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for RateMapHead {}

/// One layer's sample counts, which are also the lengths of its two quality
/// arrays.
#[repr(C)]
#[derive(Debug)]
pub struct LayerSampleCount {
    pub sample_width: U16le,
    pub sample_height: U16le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for LayerSampleCount {}

/// The head and the per-layer sample counts.
///
/// The quality floats follow, and are reached through [`layer_qualities`],
/// which needs the sample counts to find them.
pub fn rate_map<'a>(op: &Op<'a>) -> Result<(&'a RateMapHead, &'a [LayerSampleCount]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_NEW_RASTERIZATION_RATE_MAP);
    let (head, rest) = split::<RateMapHead>(op.payload)?;
    let layers = view_slice::<LayerSampleCount>(rest, head.layer_count.get() as usize)?;
    Ok((head, layers))
}

/// The horizontal and vertical quality arrays of one layer.
///
/// The layers are packed, not strided: layer `n`'s floats begin after every
/// float of every layer before it. Fixture
/// `serializer_new_rasterization_rate_map_two_layers` is what shows that — its
/// two layers have different sample counts, so a fixed stride would read layer
/// 1's horizontal qualities out of layer 0's vertical ones.
pub fn layer_qualities<'a>(
    op: &Op<'a>,
    layer: usize,
) -> Result<(&'a [F32le], &'a [F32le]), WireError> {
    let (head, layers) = rate_map(op)?;
    let count = head.layer_count.get() as usize;
    if layer >= count {
        return Err(WireError::OutOfRange {
            offset: layer,
            len: count,
        });
    }
    // Everything after the sample-count array; the floats start there.
    let mut rest = &op.payload[size_of::<RateMapHead>()
        + count
            .checked_mul(size_of::<LayerSampleCount>())
            .ok_or(WireError::CountOverflow {
                count,
                elem: size_of::<LayerSampleCount>(),
            })?..];
    for (index, entry) in layers.iter().enumerate() {
        let horizontal = entry.sample_width.get() as usize;
        let vertical = entry.sample_height.get() as usize;
        let h = view_slice::<F32le>(rest, horizontal)?;
        let v = view_slice::<F32le>(&rest[horizontal * size_of::<F32le>()..], vertical)?;
        if index == layer {
            return Ok((h, v));
        }
        rest = &rest[(horizontal + vertical) * size_of::<F32le>()..];
    }
    // Unreachable while `layers.len() == count`, which `rate_map` enforces.
    Err(WireError::OutOfRange {
        offset: layer,
        len: count,
    })
}

/// How many payload bytes the serializer actually wrote.
///
/// The record's declared length is [`UNWRITTEN_TAIL_LEN`] bytes longer than
/// this, and those bytes are the guest's stale ring on a real wire. A decoder
/// that wants to know it has consumed the record should compare against this
/// rather than against `op.payload.len()`.
pub fn quality_span(op: &Op<'_>) -> Result<usize, WireError> {
    let (head, layers) = rate_map(op)?;
    let mut span = size_of::<RateMapHead>()
        + (head.layer_count.get() as usize)
            .checked_mul(size_of::<LayerSampleCount>())
            .ok_or(WireError::CountOverflow {
                count: head.layer_count.get() as usize,
                elem: size_of::<LayerSampleCount>(),
            })?;
    for entry in layers {
        let floats = entry.sample_width.get() as usize + entry.sample_height.get() as usize;
        span += floats
            .checked_mul(size_of::<F32le>())
            .ok_or(WireError::CountOverflow {
                count: floats,
                elem: size_of::<F32le>(),
            })?;
    }
    Ok(span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op, OP_HEADER_LEN};

    /// Scratch a synthesized record is written into. Large enough for any
    /// layer set the tests below use; `synthesize` panics rather than
    /// truncating if it is not.
    type Scratch = [u8; 256];

    /// Build a record the way the serializer lays one out, from this module's
    /// own constants, and answer how many bytes it filled. Proves the
    /// arithmetic, not the layout — only the fixture test can do the latter.
    fn synthesize(into: &mut Scratch, layers: &[(u16, u16)]) -> usize {
        let mut at = OP_HEADER_LEN;
        let mut put = |bytes: &[u8], at: &mut usize| {
            into[*at..*at + bytes.len()].copy_from_slice(bytes);
            *at += bytes.len();
        };
        put(&7u32.to_le_bytes(), &mut at);
        put(&64u16.to_le_bytes(), &mut at);
        put(&48u16.to_le_bytes(), &mut at);
        put(&(layers.len() as u32).to_le_bytes(), &mut at);
        put(&2u32.to_le_bytes(), &mut at);
        put(&0u32.to_le_bytes(), &mut at);
        for (w, h) in layers {
            put(&w.to_le_bytes(), &mut at);
            put(&h.to_le_bytes(), &mut at);
        }
        let mut quality = 1.0f32;
        for (w, h) in layers {
            for _ in 0..(*w as usize + *h as usize) {
                put(&quality.to_le_bytes(), &mut at);
                quality += 1.0;
            }
        }
        put(&[0xaa; UNWRITTEN_TAIL_LEN], &mut at);
        let total = at;
        into[..4].copy_from_slice(&OPCODE_NEW_RASTERIZATION_RATE_MAP.to_le_bytes());
        into[4..8].copy_from_slice(&(total as u32).to_le_bytes());
        total
    }

    #[test]
    fn the_qualities_of_each_layer_start_after_every_layer_before_it() {
        let mut scratch: Scratch = [0; 256];
        let len = synthesize(&mut scratch, &[(2, 2), (3, 4)]);
        let bytes = &scratch[..len];
        let o = op(bytes, 0).expect("well formed");
        let (head, layers) = rate_map(&o).expect("fits");
        assert_eq!(head.layer_count.get(), 2);
        assert_eq!(layers.len(), 2);

        let (h0, v0) = layer_qualities(&o, 0).expect("layer 0");
        assert_eq!(h0.len(), 2);
        assert_eq!(v0.len(), 2);
        assert_eq!(h0[0].get(), 1.0);
        assert_eq!(v0[1].get(), 4.0);

        // Layer 1 starts at the fifth float, not at a fixed stride from layer 0.
        let (h1, v1) = layer_qualities(&o, 1).expect("layer 1");
        assert_eq!(h1.len(), 3);
        assert_eq!(v1.len(), 4);
        assert_eq!(h1[0].get(), 5.0);
        assert_eq!(v1[3].get(), 11.0);
    }

    #[test]
    fn the_written_span_stops_short_of_the_declared_length() {
        for layers in [&[(2u16, 2u16)][..], &[(4, 3)][..], &[(2, 2), (3, 4)][..]] {
            let mut scratch: Scratch = [0; 256];
            let len = synthesize(&mut scratch, layers);
            let o = op(&scratch[..len], 0).expect("well formed");
            assert_eq!(
                quality_span(&o).expect("fits") + UNWRITTEN_TAIL_LEN,
                o.payload.len(),
                "the tail is not {UNWRITTEN_TAIL_LEN} bytes for {layers:?}"
            );
        }
    }

    #[test]
    fn a_layer_index_past_the_count_is_refused_rather_than_read() {
        let mut scratch: Scratch = [0; 256];
        let len = synthesize(&mut scratch, &[(2, 2)]);
        let o = op(&scratch[..len], 0).expect("well formed");
        assert!(layer_qualities(&o, 1).is_err());
    }

    #[test]
    fn a_layer_count_the_payload_cannot_hold_is_refused() {
        let mut scratch: Scratch = [0; 256];
        let len = synthesize(&mut scratch, &[(2, 2)]);
        // The count is guest-controlled, so a record claiming 1000 layers must
        // be refused rather than read off the end of the payload.
        scratch[OP_HEADER_LEN + 8..OP_HEADER_LEN + 12].copy_from_slice(&1000u32.to_le_bytes());
        let o = op(&scratch[..len], 0).expect("well formed");
        assert!(rate_map(&o).is_err());
    }
}
