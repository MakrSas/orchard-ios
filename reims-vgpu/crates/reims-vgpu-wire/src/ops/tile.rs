//! Tile-shader records, `0x9b`–`0xa4`.
//!
//! The fourth bind stage. A `PGSerializerRenderCommandEncoder` carries vertex
//! and fragment tables, the compute encoder carries its own, and these are the
//! tile stage's — the imageblock/tile-memory pipeline Metal exposes on Apple
//! GPUs, dispatched from *inside* a render pass rather than from a compute one.
//!
//! # Every one of these was invisible until the capability was forced
//!
//! `-supportsTileShaders` reads **false** on a fresh serializer, and with it
//! false all twenty gated selectors run and write nothing. Every capture taken
//! before `withCapability` existed therefore saw silence, which is the outcome
//! the manifest turns into "Apple's serializer emits no operation for this
//! selector". Driven with the flag forced, fifteen of them emit and ten
//! distinct opcodes appear — the whole contiguous run `0x9b`–`0xa4`. See this
//! crate's `AGENTS.md`.
//!
//! The last of the ten was found late and is the reason the run reads as
//! contiguous rather than as nine plus a hole: `setThreadgroupMemoryLength:
//! offset:atIndex:` is the only selector in the family whose name does not say
//! "tile", so it was not in the first sweep and `0x9c` sat asserted as *not* a
//! tile opcode. A hole in an opcode run is a selector nobody has driven yet.
//!
//! # The bind records are the render encoder's, byte for byte
//!
//! `setTileBuffer:offset:atIndex:`, `setTileTexture:atIndex:`,
//! `setTileSamplerState:atIndex:` and their LOD and plural forms write the same
//! `[u32 first][u32 count]` head and the same entry widths as their vertex and
//! fragment counterparts, at tile opcodes. Rather than restate the layout this
//! module re-exports the declarations, the same way [`crate::ops::compute`]
//! does — one declaration, so drift between the three stages is impossible
//! rather than merely detectable.
//!
//! The singular/plural pairing that settles the leading word as a count rather
//! than a constant holds here too: `setTileTexture:atIndex:` at index 2 writes
//! `first = 2, count = 1` and one ref (`render_set_tile_texture`), while
//! `setTileTextures:withRange:` over range (7, 2) writes `first = 7, count = 2`
//! and two (`render_set_tile_textures_range`).
//!
//! # Relationship to `reims_vgpu::runtime::decode::render`
//!
//! All ten opcodes fall inside that module's accepted window, and every one of
//! them reached `Kind::OtherAccepted` — an `Ok` that means "nothing recognised
//! this" — until each was given a decode arm and a named counter there. A guest
//! that ran a tile shader had every tile bind and every tile dispatch silently
//! swallowed; it is now counted, still not executed.

use crate::le::{U32le, U64le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

pub use crate::ops::render::{BindHeader, BufferBind, BufferOffset, RefBind, SamplerLodBind};

// --- 0x9b dispatchThreadsPerTile: ------------------------------------------

pub const OPCODE_DISPATCH_THREADS_PER_TILE: u32 = 0x9b;
pub const DISPATCH_THREADS_PER_TILE_TOTAL_LEN: u32 = 32;

/// The threads-per-tile grid: an `MTLSize`, three `u64` unnarrowed.
///
/// Fixture `render_dispatch_threads_per_tile`, with `0x11`/`0x22`/`0x33` so no
/// two of the three can be confused and a dropped `depth` is visible. The whole
/// record is written — nothing here is left to the ring.
#[repr(C)]
#[derive(Debug)]
pub struct DispatchThreadsPerTile {
    pub width: U64le,
    pub height: U64le,
    pub depth: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DispatchThreadsPerTile {}

pub fn dispatch_threads_per_tile<'a>(op: &Op<'a>) -> Result<&'a DispatchThreadsPerTile, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DISPATCH_THREADS_PER_TILE);
    view::<DispatchThreadsPerTile>(op.payload)
}

// --- 0xa2 / 0xa3 dispatchThreadsPerTile:inRegion:[…] -----------------------

pub const OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION: u32 = 0xa2;
pub const OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX: u32 = 0xa3;

/// Both region forms allocate 84 bytes. Only `0xa3` writes the last four —
/// see [`dispatch_threads_per_tile_region_rt_index`].
pub const DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN: u32 = 84;

/// Offset of the render-target array index within `0xa3`'s payload.
pub const REGION_RT_INDEX_OFFSET: usize = 72;

/// A tile dispatch bounded to a region of the render target.
///
/// **The region is written origin-first, then size**, which is the order
/// `MTLRegion` declares and *not* what the sibling record in this protocol
/// does: [`crate::ops::compute::StageInRegion`] writes an `MTLRegion`'s size
/// before its origin. Two records, two orders, so neither can be inferred from
/// the other. Fixture `render_dispatch_threads_per_tile_in_region` carries
/// origin `0x44`/`0x55`/`0x66` and size `0x77`/`0x88`/`0x99`, distinct from
/// each other and from the threads-per-tile grid ahead of them, which is what
/// separates "the region reached the wire" from "the grid was written twice".
#[repr(C)]
#[derive(Debug)]
pub struct DispatchThreadsPerTileInRegion {
    pub width: U64le,
    pub height: U64le,
    pub depth: U64le,
    pub origin_x: U64le,
    pub origin_y: U64le,
    pub origin_z: U64le,
    pub region_width: U64le,
    pub region_height: U64le,
    pub region_depth: U64le,
}

// SAFETY: nine align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DispatchThreadsPerTileInRegion {}

#[inline]
pub fn is_dispatch_threads_per_tile_in_region(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION
            | OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX
    )
}

pub fn dispatch_threads_per_tile_in_region<'a>(
    op: &Op<'a>,
) -> Result<&'a DispatchThreadsPerTileInRegion, WireError> {
    debug_assert!(is_dispatch_threads_per_tile_in_region(op.opcode()));
    view::<DispatchThreadsPerTileInRegion>(op.payload)
}

/// The render-target array index, which **only `0xa3` writes**.
///
/// The two region forms are the same 84-byte allocation and the same nine
/// leading `u64`; they differ in that `0xa3` fills the trailing `u32` and
/// `0xa2` does not. That is measured, not assumed: the written-bit mask on
/// `render_dispatch_threads_per_tile_in_region` reads zero across exactly those
/// four bytes while every earlier byte reads one, and the `0xa3` fixture reads
/// one throughout.
///
/// So the field is not optional-and-defaulted, it is **absent**, and on a real
/// wire those four bytes hold whatever the guest's ring last contained. This
/// function returns `None` for `0xa2` rather than reading them. A decoder that
/// takes the index from a `0xa2` record is reading noise — the same shape as
/// the `copyFromTexture:toBuffer:` `options` bug and the heap texture's
/// `useOffset`.
///
/// Note the field is 32-bit where all nine before it are 64: the type encoding
/// says `I` (`dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:` is
/// `v92@0:8{?=QQQ}16{?={?=QQQ}{?=QQQ}}40I88`), and the fixture agrees —
/// `0xabc` lands in four bytes at payload `+72`.
pub fn dispatch_threads_per_tile_region_rt_index(op: &Op<'_>) -> Option<u32> {
    debug_assert!(is_dispatch_threads_per_tile_in_region(op.opcode()));
    if op.opcode() != OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX {
        return None;
    }
    crate::view::view_at::<U32le>(op.payload, REGION_RT_INDEX_OFFSET)
        .ok()
        .map(|v| v.get())
}

// --- 0x9d / 0x9f / 0xa0 / 0xa1 the tile bind records -----------------------

pub const OPCODE_SET_TILE_BUFFER: u32 = 0x9d;
pub const OPCODE_SET_TILE_SAMPLER: u32 = 0x9f;
pub const OPCODE_SET_TILE_SAMPLER_LOD: u32 = 0xa0;
pub const OPCODE_SET_TILE_TEXTURE: u32 = 0xa1;

/// A tile texture bind, one [`RefBind`] per slot.
///
/// Fixtures `render_set_tile_texture` (index 2, stub ref 4242) and
/// `render_set_tile_textures_range` (range (7, 2), refs 4242 and 4343 — two
/// *different* refs, which is what shows the entry stride is 4).
pub fn tile_texture_binds<'a>(op: &Op<'a>) -> Result<(&'a BindHeader, &'a [RefBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_TILE_TEXTURE);
    bind_entries::<RefBind>(op.payload)
}

/// A tile sampler bind without clamps, one [`RefBind`] per slot.
///
/// Fixtures `render_set_tile_sampler` (index 4) and
/// `render_set_tile_samplers_range` (range (6, 2)).
pub fn tile_sampler_binds<'a>(op: &Op<'a>) -> Result<(&'a BindHeader, &'a [RefBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_TILE_SAMPLER);
    bind_entries::<RefBind>(op.payload)
}

/// A tile sampler bind carrying LOD clamps.
///
/// **A different opcode from the plain tile sampler bind**, exactly as `0x80`
/// is from `0x7f` on the vertex stage — and that pairing is the one that cost
/// this project a silently unbound sampler, so it is worth restating: a decoder
/// that knows `0x9f` and not `0xa0` does not lose the clamps, it does not see
/// the bind at all.
///
/// The clamps are per entry. `render_set_tile_samplers_lod_range` binds two
/// slots with four distinct clamps (0.25/0.75 and 0.5/0.875) in a 40-byte
/// record, which is the eight-byte head plus two twelve-byte entries; the
/// singular `render_set_tile_sampler_lod` is the `count == 1` case.
pub fn tile_sampler_lod_binds<'a>(
    op: &Op<'a>,
) -> Result<(&'a BindHeader, &'a [SamplerLodBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_TILE_SAMPLER_LOD);
    bind_entries::<SamplerLodBind>(op.payload)
}

/// A tile buffer bind, one [`BufferBind`] per slot.
///
/// Fixtures `render_set_tile_buffer` (index 3, ref 5151 at `0x1234`),
/// `render_set_tile_buffers_range` (range (5, 2), two refs at two *different*
/// offsets `0x3456` and `0x4567`, which is what shows the entry stride is 12
/// rather than 16), and `render_set_tile_bytes`.
///
/// That last one is the point of the third fixture: `setTileBytes:length:atIndex:`
/// does **not** get an opcode of its own. It stages the caller's bytes through
/// the command stream and then writes this same `0x9d` record naming the
/// *staging* buffer's ref and offset — 8181 at `0x9999` in the fixture, which
/// are the harness's staging object rather than anything that case passed. The
/// vertex and fragment stages behave the same way.
pub fn tile_buffer_binds<'a>(op: &Op<'a>) -> Result<(&'a BindHeader, &'a [BufferBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_TILE_BUFFER);
    bind_entries::<BufferBind>(op.payload)
}

#[inline]
pub fn is_tile_bind(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_SET_TILE_BUFFER
            | OPCODE_SET_TILE_SAMPLER
            | OPCODE_SET_TILE_SAMPLER_LOD
            | OPCODE_SET_TILE_TEXTURE
    )
}

fn bind_entries<T: Wire>(payload: &[u8]) -> Result<(&BindHeader, &[T]), WireError> {
    let (head, rest) = crate::view::split::<BindHeader>(payload)?;
    let entries = crate::view::view_slice::<T>(rest, head.count.get() as usize)?;
    Ok((head, entries))
}

// --- 0x9e setTileBufferOffset:atIndex: -------------------------------------

pub const OPCODE_SET_TILE_BUFFER_OFFSET: u32 = 0x9e;
pub const SET_TILE_BUFFER_OFFSET_TOTAL_LEN: u32 = 20;

/// Re-point an already-bound tile buffer slot without naming the buffer again.
///
/// The vertex and fragment stages' [`BufferOffset`] record at a tile opcode, so
/// the layout is re-exported rather than restated. Fixture
/// `render_set_tile_buffer_offset` (index 4, offset `0x2345`). Note this is
/// *not* a [`BindHeader`] — the second word is the 64-bit offset, not a count.
pub fn tile_buffer_offset<'a>(op: &Op<'a>) -> Result<&'a BufferOffset, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_TILE_BUFFER_OFFSET);
    view::<BufferOffset>(op.payload)
}

// --- 0x9c setThreadgroupMemoryLength:offset:atIndex: -----------------------

pub const OPCODE_SET_TILE_THREADGROUP_MEMORY: u32 = 0x9c;
pub const SET_TILE_THREADGROUP_MEMORY_TOTAL_LEN: u32 = 28;

/// Imageblock memory for the tile stage: a length, an offset and a slot.
///
/// The one member of this family whose selector name does not say "tile", and
/// the reason `0x9c` was the single hole in the `0x9b`–`0xa4` run.
/// `setThreadgroupMemoryLength:offset:atIndex:` reads like the compute
/// encoder's namesake — which emits [`crate::ops::compute`]'s `0xd3` with no
/// capability at all — but on a render encoder it is the tile stage's
/// imageblock allocation and is gated on `-supportsTileShaders`. That gate was
/// narrowed rather than assumed: driven under `TileShaders` and `ImageBlocks`
/// together and then under `TileShaders` alone, and it emits either way.
///
/// **Not the compute record's layout.** `0xd3` is a `[length u64][index u32]`
/// pair with no offset; this is three fields and eight bytes longer, because
/// tile imageblock memory is sub-allocated from one block and the offset says
/// where. Fixture `render_set_threadgroup_memory_length` (`0x1234` at `0x2345`,
/// slot 5 — three distinct values, so a decoder that dropped the offset or
/// crossed it with the length cannot read back correct).
#[repr(C)]
#[derive(Debug)]
pub struct TileThreadgroupMemory {
    pub length: U64le,
    pub offset: U64le,
    pub index: U32le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for TileThreadgroupMemory {}

pub fn tile_threadgroup_memory<'a>(op: &Op<'a>) -> Result<&'a TileThreadgroupMemory, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_TILE_THREADGROUP_MEMORY);
    view::<TileThreadgroupMemory>(op.payload)
}

// --- 0xa4 getTileDimensions: -----------------------------------------------

pub const OPCODE_GET_TILE_DIMENSIONS: u32 = 0xa4;
pub const GET_TILE_DIMENSIONS_TOTAL_LEN: u32 = 20;

/// Where the device is expected to *write* the tile dimensions.
///
/// This record runs the protocol backwards, and it is the only one in the tile
/// family that does. `getTileDimensions:` declares `v24@0:8^{?=SS}16` — its one
/// argument is a pointer to two `u16`, so by the rule in this crate's
/// `AGENTS.md` it "fills a caller's buffer rather than the stream" and should
/// have been a legitimate `silent`. It is not. The serializer stages a
/// four-byte allocation through the command stream and emits a record naming
/// that buffer and offset, which is the guest asking the *host* to report the
/// tile width and height into guest memory.
///
/// So the type-encoding shortcut narrows to: a `^{...}` first argument means
/// the call does not put its argument *on* the wire — not that it emits
/// nothing. Fixture `render_get_tile_dimensions` carries the stub staging
/// buffer's ref 8181 and offset `0x9999`, which is what identifies the pair as
/// the staging buffer's rather than anything the case supplied.
///
/// A device that does not implement this leaves the guest reading whatever its
/// own ring last held, as a tile width and height. There is no default it could
/// be sitting at.
#[repr(C)]
#[derive(Debug)]
pub struct GetTileDimensions {
    /// The buffer the host writes the answer into.
    pub buffer_ref: U32le,
    /// Byte offset within it. The two `u16` go here, width first — the order
    /// the type encoding's `{?=SS}` declares. Nothing in the capture can
    /// confirm which is which, because the serializer writes neither; only the
    /// host does.
    pub offset: U64le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for GetTileDimensions {}

pub fn get_tile_dimensions<'a>(op: &Op<'a>) -> Result<&'a GetTileDimensions, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_GET_TILE_DIMENSIONS);
    view::<GetTileDimensions>(op.payload)
}

/// Every opcode this module claims.
#[inline]
pub fn is_tile_opcode(opcode: u32) -> bool {
    is_tile_bind(opcode)
        || is_dispatch_threads_per_tile_in_region(opcode)
        || matches!(
            opcode,
            OPCODE_DISPATCH_THREADS_PER_TILE
                | OPCODE_SET_TILE_BUFFER_OFFSET
                | OPCODE_SET_TILE_THREADGROUP_MEMORY
                | OPCODE_GET_TILE_DIMENSIONS
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OP_HEADER_LEN;
    use core::mem::size_of;

    #[test]
    fn the_bodies_and_the_total_lengths_agree() {
        assert_eq!(
            size_of::<DispatchThreadsPerTile>() + OP_HEADER_LEN,
            DISPATCH_THREADS_PER_TILE_TOTAL_LEN as usize
        );
        assert_eq!(
            size_of::<GetTileDimensions>() + OP_HEADER_LEN,
            GET_TILE_DIMENSIONS_TOTAL_LEN as usize
        );
        assert_eq!(
            size_of::<BufferOffset>() + OP_HEADER_LEN,
            SET_TILE_BUFFER_OFFSET_TOTAL_LEN as usize
        );
    }

    /// The region body is nine `u64`, and the record is four bytes longer than
    /// that on both opcodes. Those four are `0xa3`'s render-target array index,
    /// and on `0xa2` they are never written — so the body deliberately stops
    /// short of the record rather than covering it.
    #[test]
    fn the_region_body_stops_where_the_written_bytes_do() {
        assert_eq!(size_of::<DispatchThreadsPerTileInRegion>(), 72);
        assert_eq!(
            size_of::<DispatchThreadsPerTileInRegion>(),
            REGION_RT_INDEX_OFFSET
        );
        assert_eq!(
            size_of::<DispatchThreadsPerTileInRegion>() + OP_HEADER_LEN + size_of::<U32le>(),
            DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize
        );
    }

    /// The index is readable on `0xa3` and refused on `0xa2`, from the same
    /// bytes. A view that returned it for both would hand back ring contents.
    #[test]
    fn the_render_target_array_index_is_absent_on_the_plain_region_form() {
        let mut buf = [0u8; DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize];
        buf[REGION_RT_INDEX_OFFSET + OP_HEADER_LEN..][..4].copy_from_slice(&0xabcu32.to_le_bytes());

        for (opcode, want) in [
            (
                OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
                Some(0xabc),
            ),
            (OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION, None),
        ] {
            buf[..4].copy_from_slice(&opcode.to_le_bytes());
            buf[4..8].copy_from_slice(&DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN.to_le_bytes());
            let op = crate::op::op(&buf, 0).expect("well-formed region record");
            assert_eq!(dispatch_threads_per_tile_region_rt_index(&op), want);
        }
    }

    /// A count the record cannot hold is refused rather than read past. The
    /// bind entries are guest-controlled the same way every other stage's are.
    #[test]
    fn a_tile_bind_count_past_the_record_is_refused() {
        let mut buf = [0u8; 20];
        buf[..4].copy_from_slice(&OPCODE_SET_TILE_TEXTURE.to_le_bytes());
        buf[4..8].copy_from_slice(&20u32.to_le_bytes());
        buf[8..12].copy_from_slice(&0u32.to_le_bytes());
        buf[12..16].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
        let op = crate::op::op(&buf, 0).expect("well-formed header");
        assert!(matches!(
            tile_texture_binds(&op),
            Err(WireError::Short { .. })
        ));
    }

    #[test]
    fn every_tile_opcode_this_module_names_is_claimed() {
        for opcode in [
            OPCODE_DISPATCH_THREADS_PER_TILE,
            OPCODE_SET_TILE_BUFFER,
            OPCODE_SET_TILE_BUFFER_OFFSET,
            OPCODE_SET_TILE_THREADGROUP_MEMORY,
            OPCODE_SET_TILE_SAMPLER,
            OPCODE_SET_TILE_SAMPLER_LOD,
            OPCODE_SET_TILE_TEXTURE,
            OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
            OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
            OPCODE_GET_TILE_DIMENSIONS,
        ] {
            assert!(is_tile_opcode(opcode), "opcode {opcode:#x} unclaimed");
        }
        // The family is one contiguous run of ten, `0x9b`–`0xa4`, with no
        // holes. `0x9c` was the last one found and it sat here as an asserted
        // *gap* first — because its selector, `setThreadgroupMemoryLength:
        // offset:atIndex:`, is the only one in the family whose name does not
        // say "tile". A hole in an opcode run is a selector nobody has driven
        // yet, not a number Apple skipped.
        for opcode in 0x9bu32..=0xa4 {
            assert!(
                is_tile_opcode(opcode),
                "hole at {opcode:#x} in the tile run"
            );
        }
        assert!(
            !is_tile_opcode(0x9a),
            "0x9a is the vertex amplification count"
        );
        assert!(
            !is_tile_opcode(0xa5),
            "0xa5 is the vertex attribute-stride bind"
        );
        assert!(!is_tile_opcode(
            crate::ops::render::OPCODE_SET_VERTEX_TEXTURE
        ));
    }
}
