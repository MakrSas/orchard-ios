//! Blit encoder records.
//!
//! These are the records a `PGSerializerBlitCommandEncoder` writes through
//! `PGSerializerCommandStream`'s `-getCommandBufferBytes:`, each one the shared
//! 8-byte [`crate::op::OpHeader`] followed by a per-opcode payload. Every
//! layout below came from calling the Metal method with distinctive arguments
//! and reading the bytes back; the fixture that pins each is named in its doc.
//! See `oracle/oracle.m`'s `blitCases`.
//!
//! # The opcodes live in their own space
//!
//! Blit opcodes run `0x12c`–`0x143`, nowhere near the render encoder's
//! `0x00`–`0xa6`. That is a fact about this serializer's numbering and not an
//! assumption: every opcode below was read off a record Apple wrote.
//!
//! The run above `0x13e` is the six selectors `-setSupportsBlitEncoderSPI:`
//! gates, and it was invisible for as long as the capture drove this class with
//! the flag at its default. All sixteen capability flags default off, so the
//! run looked like it ended at `0x13e`. It does not, and the tail is where the
//! fills live — see [`FillBufferPattern4`].
//!
//! # Two resources, then the wide fields, then the narrow ones
//!
//! Every copy record has the same three-part shape, and it is **not** the
//! selector's argument order:
//!
//! 1. the source ref and the destination ref, both `u32`, first;
//! 2. the `u64` arguments — offsets, bytes-per-row, origins, sizes — in the
//!    order the selector declares them;
//! 3. the slice and level indices, `u16` each, at the tail.
//!
//! `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` is the
//! clearest case: the selector interleaves the two buffers with their offsets
//! and the record does not (see [`BufferToBuffer`]). Every case here uses a
//! different ref for source and destination, so a record that wrote one ref
//! into both slots, or swapped them, could not read back correct.
//!
//! # Three selectors have no opcode of their own
//!
//! `optimize:withCommand:`, `optimize:slice:level:withCommand:` and
//! `optimizeReset:withRange:withCommand:` write their `command:` argument into
//! the header's opcode field. Passing `0x77` produced a record with opcode
//! `0x77` and passing `0x55` produced one with opcode `0x55`
//! (`blit_optimize_with_command`, `blit_optimize_with_command_alt`), which is
//! why the second case exists — one observation cannot tell "the argument is
//! the opcode" from "the opcode happens to equal the argument".
//!
//! They are therefore generic emitters for the three record shapes [`Ref`],
//! [`RefSliceLevel`] and [`IcbRange`], and the fixed-opcode selectors that
//! share those shapes ([`OPCODE_OPTIMIZE_FOR_GPU`] and its five siblings) are
//! wrappers over them. Their manifest rows carry no opcode, because the opcode
//! is guest data rather than a property of the selector.
//!
//! # Relationship to `reims_vgpu::runtime::decode::blit`
//!
//! Recorded per record where the two disagree. Nothing here changes that
//! module; the divergences are findings.

use crate::le::{F64le, U16le, U32le, U64le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

// --- One resource ref ------------------------------------------------------

pub const OPCODE_GENERATE_MIPMAPS: u32 = 0x133;
pub const OPCODE_OPTIMIZE_FOR_CPU: u32 = 0x134;
pub const OPCODE_OPTIMIZE_FOR_GPU: u32 = 0x135;
pub const OPCODE_SYNCHRONIZE_RESOURCE: u32 = 0x13a;
pub const OPCODE_UPDATE_FENCE: u32 = 0x13c;
pub const OPCODE_WAIT_FOR_FENCE: u32 = 0x13d;
pub const OPCODE_INVALIDATE_COMPRESSED_TEXTURE: u32 = 0x142;

/// The whole payload of the seven single-ref blit records: 4 bytes, one ref, no
/// flags and nothing else.
///
/// Which object the ref names comes from the opcode, not from the record —
/// `updateFence:` names a fence and `generateMipmapsForTexture:` a texture, and
/// the bytes are identical. Fixtures `blit_generate_mipmaps`,
/// `blit_optimize_for_cpu`, `blit_optimize_for_gpu`,
/// `blit_synchronize_resource` (a *buffer*, ref 5151, where the other three are
/// the texture at 4242), `blit_update_fence` and `blit_wait_for_fence` (the
/// fence at 6464), and `blit_invalidate_compressed_texture`.
///
/// The last of those emits only under `-setSupportsBlitEncoderSPI:`; see
/// [`FillBufferPattern4`].
#[repr(C)]
#[derive(Debug)]
pub struct Ref {
    pub object_ref: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for Ref {}

pub const REF_TOTAL_LEN: u32 = 12;

#[inline]
pub fn is_ref(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_GENERATE_MIPMAPS
            | OPCODE_OPTIMIZE_FOR_CPU
            | OPCODE_OPTIMIZE_FOR_GPU
            | OPCODE_SYNCHRONIZE_RESOURCE
            | OPCODE_UPDATE_FENCE
            | OPCODE_WAIT_FOR_FENCE
            | OPCODE_INVALIDATE_COMPRESSED_TEXTURE
    )
}

pub fn object_ref<'a>(op: &Op<'a>) -> Result<&'a Ref, WireError> {
    view::<Ref>(op.payload)
}

// --- One texture, one slice, one level -------------------------------------

pub const OPCODE_OPTIMIZE_FOR_CPU_SLICE_LEVEL: u32 = 0x136;
pub const OPCODE_OPTIMIZE_FOR_GPU_SLICE_LEVEL: u32 = 0x137;
pub const OPCODE_SYNCHRONIZE_TEXTURE: u32 = 0x13b;
pub const OPCODE_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL: u32 = 0x143;

/// A texture and one of its subresources.
///
/// `slice` and `level` are **16 bits each**, though both are declared `Q` on
/// every selector that carries them. The record is 16 bytes total, so the two
/// share the single word at `+4`, and `(slice 3, level 5)` reads `03 00 05 00`
/// — unequal on purpose, because equal values could not tell the two apart.
/// Fixtures `blit_optimize_for_cpu_slice_level`,
/// `blit_optimize_for_gpu_slice_level`, `blit_synchronize_texture`,
/// `blit_invalidate_compressed_texture_slice_level`.
///
/// The last emits only under `-setSupportsBlitEncoderSPI:`; see
/// [`FillBufferPattern4`]. Note that its selector spells the arguments
/// `slice:level:` and the record keeps that order, which is the same order the
/// three above use despite two of *their* selectors spelling it the same way —
/// so the shape is shared rather than coincidental.
#[repr(C)]
#[derive(Debug)]
pub struct RefSliceLevel {
    pub texture_ref: U32le,
    pub slice: U16le,
    pub level: U16le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for RefSliceLevel {}

pub const REF_SLICE_LEVEL_TOTAL_LEN: u32 = 16;

#[inline]
pub fn is_ref_slice_level(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_OPTIMIZE_FOR_CPU_SLICE_LEVEL
            | OPCODE_OPTIMIZE_FOR_GPU_SLICE_LEVEL
            | OPCODE_SYNCHRONIZE_TEXTURE
            | OPCODE_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL
    )
}

pub fn ref_slice_level<'a>(op: &Op<'a>) -> Result<&'a RefSliceLevel, WireError> {
    view::<RefSliceLevel>(op.payload)
}

// --- An indirect command buffer and a command range ------------------------

pub const OPCODE_OPTIMIZE_ICB: u32 = 0x138;
pub const OPCODE_RESET_ICB: u32 = 0x139;

/// A range of commands inside an indirect command buffer.
///
/// The `NSRange` arrives as two `u64` in declaration order — location then
/// length — with no narrowing, which is what makes this shape different from
/// [`RefSliceLevel`] despite both being "a ref and two numbers". Fixtures
/// `blit_optimize_indirect_command_buffer` (range `0x3300`, `0x4400`) and
/// `blit_reset_commands_in_buffer` (range `0x1100`, `0x2200`); the two use
/// different ranges so a record carrying the wrong one is visible.
///
/// The ICB's ref reaches the record through `indirectCommandBufferRef`, which
/// is a different accessor from every other resource here.
#[repr(C)]
#[derive(Debug)]
pub struct IcbRange {
    pub icb_ref: U32le,
    pub range_location: U64le,
    pub range_length: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for IcbRange {}

pub const ICB_RANGE_TOTAL_LEN: u32 = 28;

#[inline]
pub fn is_icb_range(opcode: u32) -> bool {
    matches!(opcode, OPCODE_OPTIMIZE_ICB | OPCODE_RESET_ICB)
}

pub fn icb_range<'a>(op: &Op<'a>) -> Result<&'a IcbRange, WireError> {
    view::<IcbRange>(op.payload)
}

// --- 0x131 copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:

pub const OPCODE_COPY_ICB: u32 = 0x131;
pub const COPY_ICB_TOTAL_LEN: u32 = 40;

/// Copy a run of commands from one indirect command buffer into another.
///
/// Both refs lead, as everywhere in this family, and the destination *index* is
/// a `u64` beside the source range rather than a `u16` at the tail — it is a
/// command index, not a subresource index, and the record keeps the two kinds
/// apart. Fixture `blit_copy_indirect_command_buffer` (source 7171, range
/// `0x1100`/`0x2200`, destination 7272 at index `0x3300`).
#[repr(C)]
#[derive(Debug)]
pub struct CopyIcb {
    pub source_ref: U32le,
    pub dest_ref: U32le,
    pub range_location: U64le,
    pub range_length: U64le,
    pub dest_index: U64le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for CopyIcb {}

pub fn copy_icb<'a>(op: &Op<'a>) -> Result<&'a CopyIcb, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_COPY_ICB);
    view::<CopyIcb>(op.payload)
}

// --- 0x132 fillBuffer:range:value: -----------------------------------------

pub const OPCODE_FILL_BUFFER: u32 = 0x132;
pub const FILL_BUFFER_TOTAL_LEN: u32 = 32;

/// Fill a byte range of a buffer with one byte.
///
/// The record is 32 bytes and this body is 21 of the 24 payload bytes: the
/// three bytes after `value` are **never written**. They read `AA AA AA` in the
/// fixture because the oracle poisons its arena, and on a real wire they hold
/// whatever the ring last contained. There is no field there.
///
/// Fixture `blit_fill_buffer` (buffer 5151, range `0x1100`/`0x2200`, value
/// `0x5a` — chosen so the single byte is recognisable wherever it landed).
///
/// `fillBuffer:range:pattern4:`, the 32-bit sibling, emits **no record at all**
/// on this serializer: it returns without asking the buffer for anything. See
/// the manifest's exclusion for it.
#[repr(C)]
#[derive(Debug)]
pub struct FillBuffer {
    pub buffer_ref: U32le,
    pub range_location: U64le,
    pub range_length: U64le,
    /// A single byte, so there is no order to encode and no `le` wrapper for
    /// it. Align-1 and all-bytes-valid hold trivially, which is what the
    /// [`Wire`] contract asks of a field.
    pub value: u8,
}

// SAFETY: four align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for FillBuffer {}

pub fn fill_buffer<'a>(op: &Op<'a>) -> Result<&'a FillBuffer, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_FILL_BUFFER);
    view::<FillBuffer>(op.payload)
}

// --- 0x12d copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size: -----

pub const OPCODE_COPY_BUFFER_TO_BUFFER: u32 = 0x12d;
pub const COPY_BUFFER_TO_BUFFER_TOTAL_LEN: u32 = 40;

/// Buffer-to-buffer copy.
///
/// The record does not follow the selector's argument order. The selector
/// interleaves — buffer, offset, buffer, offset, size — and the record puts
/// **both refs first** and then the three `u64`s. Fixture
/// `blit_copy_buffer_to_buffer`: source 5151 at `0x1111`, destination 5252 at
/// `0x2222`, size `0x3333`, five values no two of which are equal.
#[repr(C)]
#[derive(Debug)]
pub struct BufferToBuffer {
    pub source_ref: U32le,
    pub dest_ref: U32le,
    pub source_offset: U64le,
    pub dest_offset: U64le,
    pub size: U64le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for BufferToBuffer {}

pub fn copy_buffer_to_buffer<'a>(op: &Op<'a>) -> Result<&'a BufferToBuffer, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_COPY_BUFFER_TO_BUFFER);
    view::<BufferToBuffer>(op.payload)
}

// --- 0x13e copyFromTexture:…sliceCount:levelCount: -------------------------

pub const OPCODE_COPY_TEXTURE_SLICES: u32 = 0x13e;
pub const COPY_TEXTURE_SLICES_TOTAL_LEN: u32 = 28;

/// Whole-subresource texture copy: six `u16` indices after the two refs.
///
/// `copyFromTexture:toTexture:` shares this opcode and is its default case —
/// the serializer reads the source texture's `arrayLength` and
/// `mipmapLevelCount` and writes them as the counts, with every index zero.
/// Fixture `blit_copy_texture_to_texture` produced `slice_count = 1`,
/// `level_count = 1` off a stub reporting exactly those, which is why the
/// expectation is read from the texture rather than transcribed.
///
/// The explicit form (`blit_copy_texture_slices`) uses 2, 3, 4, 5, 6, 7 — six
/// distinct values, so no pair of these six fields can be confused.
#[repr(C)]
#[derive(Debug)]
pub struct CopyTextureSlices {
    pub source_ref: U32le,
    pub dest_ref: U32le,
    pub source_slice: U16le,
    pub source_level: U16le,
    pub dest_slice: U16le,
    pub dest_level: U16le,
    pub slice_count: U16le,
    pub level_count: U16le,
}

// SAFETY: eight align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for CopyTextureSlices {}

pub fn copy_texture_slices<'a>(op: &Op<'a>) -> Result<&'a CopyTextureSlices, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_COPY_TEXTURE_SLICES);
    view::<CopyTextureSlices>(op.payload)
}

// --- 0x12f / 0x130 copyFromTexture:…sourceOrigin:sourceSize:…destinationOrigin:

pub const OPCODE_COPY_TEXTURE_REGION: u32 = 0x12f;
pub const COPY_TEXTURE_REGION_TOTAL_LEN: u32 = 96;
pub const OPCODE_COPY_TEXTURE_REGION_OPTIONS: u32 = 0x130;
pub const COPY_TEXTURE_REGION_OPTIONS_TOTAL_LEN: u32 = 100;

/// Region-to-region texture copy, without `options:`.
///
/// The three-part shape in its clearest form: two refs, nine `u64` (source
/// origin, size, destination origin, in selector order), then four `u16`. The
/// fixture `blit_copy_texture_region` uses `0x11`/`0x22`/`0x33` for the source
/// origin, `0x44`/`0x55`/`1` for the size and `0x66`/`0x77`/`0x88` for the
/// destination origin, so a swapped origin is visible rather than plausible.
///
/// `size_depth` is 1 rather than something distinctive because Metal will not
/// accept a 2D copy deeper than one slice; the field is located by its
/// neighbours moving, not by its own value.
#[repr(C)]
#[derive(Debug)]
pub struct CopyTextureRegion {
    pub source_ref: U32le,
    pub dest_ref: U32le,
    pub source_origin_x: U64le,
    pub source_origin_y: U64le,
    pub source_origin_z: U64le,
    pub size_width: U64le,
    pub size_height: U64le,
    pub size_depth: U64le,
    pub dest_origin_x: U64le,
    pub dest_origin_y: U64le,
    pub dest_origin_z: U64le,
    pub source_slice: U16le,
    pub source_level: U16le,
    pub dest_slice: U16le,
    pub dest_level: U16le,
}

// SAFETY: fifteen align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for CopyTextureRegion {}

pub fn copy_texture_region<'a>(op: &Op<'a>) -> Result<&'a CopyTextureRegion, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_COPY_TEXTURE_REGION);
    view::<CopyTextureRegion>(op.payload)
}

/// The same copy with `options:`, which is a **different opcode and a longer
/// record** — `0x130` at 100 bytes against `0x12f` at 96.
///
/// This is the one place in the blit family where the `options:` variant is not
/// its plain sibling with the field set: the other two copy selectors keep
/// their opcode and their length and carry `options` in room the plain form
/// already reserves. Fixture `blit_copy_texture_region_options` (options
/// `MTLBlitOptionRowLinearPVRTC` = 4, and slices/levels 9, 10, 11, 12 so
/// nothing collides with the option value).
#[repr(C)]
#[derive(Debug)]
pub struct CopyTextureRegionOptions {
    pub region: CopyTextureRegion,
    pub options: U32le,
}

// SAFETY: an align-1 `Wire` struct followed by an align-1 `le` scalar; every
// byte pattern of both is valid.
unsafe impl Wire for CopyTextureRegionOptions {}

pub fn copy_texture_region_options<'a>(
    op: &Op<'a>,
) -> Result<&'a CopyTextureRegionOptions, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_COPY_TEXTURE_REGION_OPTIONS);
    view::<CopyTextureRegionOptions>(op.payload)
}

// --- 0x12c copyFromBuffer:…toTexture:… -------------------------------------

pub const OPCODE_COPY_BUFFER_TO_TEXTURE: u32 = 0x12c;
pub const COPY_BUFFER_TO_TEXTURE_TOTAL_LEN: u32 = 96;

/// Buffer-to-texture copy, with or without `options:`.
///
/// **One opcode and one length for both selectors.** The plain form writes
/// `options = 0` into the same four bytes the `options:` form writes 4 into
/// (fixtures `blit_copy_buffer_to_texture` and
/// `blit_copy_buffer_to_texture_options`, both `0x12c` at 96 bytes), so the
/// plain selector is the `options: MTLBlitOptionNone` case rather than a
/// separate record.
///
/// `options` is four bytes wide here because four were written: the plain
/// form's poison-filled arena came back with `00 00 00 00` at `+84`, not with
/// two zeros and two `0xAA`. Its sibling [`CopyTextureToBuffer`] writes only
/// two at the same position, which is a real asymmetry in Apple's encoder and
/// not a reading error.
#[repr(C)]
#[derive(Debug)]
pub struct CopyBufferToTexture {
    pub source_ref: U32le,
    pub dest_ref: U32le,
    pub source_offset: U64le,
    pub source_bytes_per_row: U64le,
    pub source_bytes_per_image: U64le,
    pub size_width: U64le,
    pub size_height: U64le,
    pub size_depth: U64le,
    pub dest_origin_x: U64le,
    pub dest_origin_y: U64le,
    pub dest_origin_z: U64le,
    pub dest_slice: U16le,
    pub dest_level: U16le,
    pub options: U32le,
}

// SAFETY: fourteen align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for CopyBufferToTexture {}

pub fn copy_buffer_to_texture<'a>(op: &Op<'a>) -> Result<&'a CopyBufferToTexture, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_COPY_BUFFER_TO_TEXTURE);
    view::<CopyBufferToTexture>(op.payload)
}

// --- 0x12e copyFromTexture:…toBuffer:… -------------------------------------

pub const OPCODE_COPY_TEXTURE_TO_BUFFER: u32 = 0x12e;
pub const COPY_TEXTURE_TO_BUFFER_TOTAL_LEN: u32 = 96;

/// Texture-to-buffer copy, with or without `options:`.
///
/// One opcode and one length for both selectors, as with
/// [`CopyBufferToTexture`] — but `options` is **two** bytes here, not four.
/// `blit_copy_texture_to_buffer_options` came back `04 00 AA AA` at `+84`: the
/// serializer wrote two bytes and left the next two at the arena's poison. Two
/// bytes past the end of this body therefore belong to no field, and a view
/// that read a `u32` there would be reading whatever the guest's ring last
/// held.
///
/// The record is 96 bytes and this body is 86 of the 88 payload bytes, which is
/// the same relationship [`FillBuffer`] has to its record and for the same
/// reason.
#[repr(C)]
#[derive(Debug)]
pub struct CopyTextureToBuffer {
    pub source_ref: U32le,
    pub dest_ref: U32le,
    pub source_origin_x: U64le,
    pub source_origin_y: U64le,
    pub source_origin_z: U64le,
    pub size_width: U64le,
    pub size_height: U64le,
    pub size_depth: U64le,
    pub dest_offset: U64le,
    pub dest_bytes_per_row: U64le,
    pub dest_bytes_per_image: U64le,
    pub source_slice: U16le,
    pub source_level: U16le,
    pub options: U16le,
}

// SAFETY: fourteen align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for CopyTextureToBuffer {}

pub fn copy_texture_to_buffer<'a>(op: &Op<'a>) -> Result<&'a CopyTextureToBuffer, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_COPY_TEXTURE_TO_BUFFER);
    view::<CopyTextureToBuffer>(op.payload)
}

// --- 0x13f fillBuffer:range:pattern4: --------------------------------------

pub const OPCODE_FILL_BUFFER_PATTERN4: u32 = 0x13f;
pub const FILL_BUFFER_PATTERN4_TOTAL_LEN: u32 = 32;

/// Fill a byte range of a buffer with a repeating 32-bit pattern.
///
/// [`FillBuffer`]'s wide sibling, and the same layout to the byte: the two
/// records differ only in the width of their last field and in their opcode.
/// Where `fillBuffer:range:value:` writes one byte at `+20` and leaves the next
/// three at whatever the ring held, this writes all four — `written_mask` reads
/// `ff ff ff ff` across the word, against `ff 00 00 00` for the narrow form.
/// So the record is written end to end and there is no unwritten tail.
///
/// Fixtures `blit_fill_buffer_pattern4` (buffer 5151, range `0x3300`/`0x4400`,
/// pattern `0x89abcdef`) and `blit_fill_buffer_pattern4_alt` (buffer 5252,
/// range `0x5500`/`0x6600`, pattern `0x13572468`), which move every field.
///
/// # It emits only under `-setSupportsBlitEncoderSPI:`
///
/// With the flag off the selector returns having written nothing, and this
/// crate recorded that silence as `EMITS_NO_OPERATION` — a claim about Apple
/// that was wrong. The capture's per-flag attribution passes name
/// `BlitEncoderSPI` for this selector and the five below.
#[repr(C)]
#[derive(Debug)]
pub struct FillBufferPattern4 {
    pub buffer_ref: U32le,
    pub range_location: U64le,
    pub range_length: U64le,
    pub pattern: U32le,
}

// SAFETY: four align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for FillBufferPattern4 {}

pub fn fill_buffer_pattern4<'a>(op: &Op<'a>) -> Result<&'a FillBufferPattern4, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_FILL_BUFFER_PATTERN4);
    view::<FillBufferPattern4>(op.payload)
}

// --- 0x141 fillTexture:level:slice:region:color:[pixelFormat:] -------------

pub const OPCODE_FILL_TEXTURE_COLOR: u32 = 0x141;
pub const FILL_TEXTURE_COLOR_TOTAL_LEN: u32 = 100;

/// Clear a texture region to a colour.
///
/// **Two selectors, one record.** `fillTexture:level:slice:region:color:` and
/// `fillTexture:level:slice:region:color:pixelFormat:` write the same opcode,
/// the same length and the same fields; the only difference is where
/// [`FillTextureColor::pixel_format`] comes from. That is derived rather than
/// assumed — the two forms' bodies are byte-identical in the fixtures apart
/// from that one word.
///
/// # The region is stored size-first
///
/// `MTLRegion` declares `origin` then `size`, and the selector's type encoding
/// says so (`{?={?=QQQ}{?=QQQ}}`). The record reverses them: width, height and
/// depth occupy `+8`..`+31` and the origin follows at `+32`. Every case uses an
/// origin and a size no two components of which are equal, so a reader that
/// took the two the other way round could not read back correct.
///
/// # `pixel_format` has two sources and is sixteen bits
///
/// Declared `Q` on the `pixelFormat:` selector and narrowed to `u16` on the
/// wire. When the guest names a format the record carries it —
/// `blit_fill_texture_color_pixel_format` asked for RGBA16Float and got 115,
/// `..._alt` asked for R8Unorm and got 10. When the guest does *not*, the
/// serializer reads the format off the texture: `blit_fill_texture_color`
/// against a BGRA8Unorm texture reads 80 and
/// `blit_fill_texture_color_r8_texture` against an R8Unorm one reads 10, which
/// is what separates "the serializer asked the texture" from "the field is a
/// constant".
///
/// The two bytes at `+90` are never written and belong to no field.
///
/// Fixtures `blit_fill_texture_color` (texture 4242, level 3, slice 5, region
/// `0x11`/`0x22`/`0x33` by `0x44`/`0x55`/1, colour 0.25/0.5/0.75/1.0) and
/// `blit_fill_texture_color_alt`, which moves every one of those.
#[repr(C)]
#[derive(Debug)]
pub struct FillTextureColor {
    pub texture_ref: U32le,
    pub level: U16le,
    pub slice: U16le,
    pub size_width: U64le,
    pub size_height: U64le,
    pub size_depth: U64le,
    pub origin_x: U64le,
    pub origin_y: U64le,
    pub origin_z: U64le,
    /// `MTLClearColor` is four `double`s, and they reach the wire as four
    /// `double`s rather than being narrowed the way the integer arguments are.
    pub color_red: F64le,
    pub color_green: F64le,
    pub color_blue: F64le,
    pub color_alpha: F64le,
    pub pixel_format: U16le,
}

// SAFETY: fourteen align-1 all-bytes-valid `le` scalars. `F64le` is
// `#[repr(transparent)]` over `[u8; 8]` and converts on read, so a signaling
// NaN in guest bytes is a value rather than undefined behaviour.
unsafe impl Wire for FillTextureColor {}

pub fn fill_texture_color<'a>(op: &Op<'a>) -> Result<&'a FillTextureColor, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_FILL_TEXTURE_COLOR);
    view::<FillTextureColor>(op.payload)
}

// --- 0x140 fillTexture:level:slice:region:bytes:length: --------------------

pub const OPCODE_FILL_TEXTURE_BYTES: u32 = 0x140;
pub const FILL_TEXTURE_BYTES_TOTAL_LEN: u32 = 84;

/// Fill a texture region with a caller-supplied pixel pattern.
///
/// The same leading fields as [`FillTextureColor`] — ref, level, slice, then
/// the size-before-origin region — and then the pattern, which does **not**
/// travel inline. The serializer stages the caller's bytes into a buffer and
/// names it: `bytes_ref` is the staging buffer's ref and `bytes_offset` is the
/// offset within it, the same mechanism `setVertexBytes:length:atIndex:` and
/// its four siblings use, and recognisable in the fixtures as the same ref and
/// offset those records carry.
///
/// So a reader of this record must fetch `length` bytes from the named buffer;
/// the record itself carries no pixel data, and the length is the guest's, not
/// derivable from the region.
///
/// The record is written end to end.
///
/// Fixtures `blit_fill_texture_bytes` (8 bytes) and
/// `blit_fill_texture_bytes_alt` (4), which is what shows `length` is a field
/// rather than a constant and that the record's own length does not track it.
#[repr(C)]
#[derive(Debug)]
pub struct FillTextureBytes {
    pub texture_ref: U32le,
    pub level: U16le,
    pub slice: U16le,
    pub size_width: U64le,
    pub size_height: U64le,
    pub size_depth: U64le,
    pub origin_x: U64le,
    pub origin_y: U64le,
    pub origin_z: U64le,
    pub bytes_ref: U32le,
    pub bytes_offset: U64le,
    pub length: U64le,
}

// SAFETY: twelve align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for FillTextureBytes {}

pub fn fill_texture_bytes<'a>(op: &Op<'a>) -> Result<&'a FillTextureBytes, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_FILL_TEXTURE_BYTES);
    view::<FillTextureBytes>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    use crate::op::OP_HEADER_LEN;

    /// Every record whose length the serializer fixes: body plus header must be
    /// exactly it. This is the one line that catches a field of the wrong
    /// width, because a slipped field moves the total.
    #[test]
    fn each_fixed_length_record_is_its_body_plus_the_header() {
        for (name, body, total) in [
            ("Ref", size_of::<Ref>(), REF_TOTAL_LEN),
            (
                "RefSliceLevel",
                size_of::<RefSliceLevel>(),
                REF_SLICE_LEVEL_TOTAL_LEN,
            ),
            ("IcbRange", size_of::<IcbRange>(), ICB_RANGE_TOTAL_LEN),
            ("CopyIcb", size_of::<CopyIcb>(), COPY_ICB_TOTAL_LEN),
            (
                "BufferToBuffer",
                size_of::<BufferToBuffer>(),
                COPY_BUFFER_TO_BUFFER_TOTAL_LEN,
            ),
            (
                "CopyTextureSlices",
                size_of::<CopyTextureSlices>(),
                COPY_TEXTURE_SLICES_TOTAL_LEN,
            ),
            (
                "CopyTextureRegion",
                size_of::<CopyTextureRegion>(),
                COPY_TEXTURE_REGION_TOTAL_LEN,
            ),
            (
                "CopyTextureRegionOptions",
                size_of::<CopyTextureRegionOptions>(),
                COPY_TEXTURE_REGION_OPTIONS_TOTAL_LEN,
            ),
            (
                "CopyBufferToTexture",
                size_of::<CopyBufferToTexture>(),
                COPY_BUFFER_TO_TEXTURE_TOTAL_LEN,
            ),
            (
                "FillBufferPattern4",
                size_of::<FillBufferPattern4>(),
                FILL_BUFFER_PATTERN4_TOTAL_LEN,
            ),
            (
                "FillTextureBytes",
                size_of::<FillTextureBytes>(),
                FILL_TEXTURE_BYTES_TOTAL_LEN,
            ),
        ] {
            assert_eq!(
                body + OP_HEADER_LEN,
                total as usize,
                "{name}: body {body} + header does not make {total}"
            );
        }
    }

    /// The two records whose body is *shorter* than their payload, because the
    /// serializer leaves bytes at the end unwritten.
    ///
    /// A future edit that "tidies" either by widening the last field to fill
    /// the record would make the view read uninitialized guest memory, so the
    /// gap is asserted rather than left as a comment.
    #[test]
    fn the_two_records_with_unwritten_tails_stay_short_of_their_length() {
        assert_eq!(
            size_of::<FillBuffer>() + OP_HEADER_LEN + 3,
            FILL_BUFFER_TOTAL_LEN as usize,
            "fillBuffer: leaves exactly three bytes after `value` unwritten"
        );
        assert_eq!(
            size_of::<CopyTextureToBuffer>() + OP_HEADER_LEN + 2,
            COPY_TEXTURE_TO_BUFFER_TOTAL_LEN as usize,
            "copyFromTexture:toBuffer: leaves exactly two bytes after `options` unwritten"
        );
        assert_eq!(
            size_of::<FillTextureColor>() + OP_HEADER_LEN + 2,
            FILL_TEXTURE_COLOR_TOTAL_LEN as usize,
            "fillTexture:…:color: leaves exactly two bytes after `pixel_format` unwritten"
        );
    }

    /// The pattern fill is the byte fill with a wider last field.
    ///
    /// Two records of the same length whose bodies differ by three bytes, which
    /// is the whole of the difference between them: `fillBuffer:range:value:`
    /// writes one byte where `fillBuffer:range:pattern4:` writes four. A future
    /// edit that widened `FillBuffer::value` to a `u32` "for symmetry" would
    /// make it read three bytes of the guest's ring, and this is the line that
    /// says so.
    #[test]
    fn the_pattern_fill_is_the_byte_fill_with_three_more_written_bytes() {
        assert_eq!(
            size_of::<FillBufferPattern4>(),
            size_of::<FillBuffer>() + 3,
            "the two fills differ only in the width of their last field"
        );
        assert_eq!(FILL_BUFFER_PATTERN4_TOTAL_LEN, FILL_BUFFER_TOTAL_LEN);
    }

    /// The two texture fills share their leading fields.
    ///
    /// Ref, level, slice and the size-before-origin region are common to both,
    /// and only the tail differs — a colour and a format against a staged
    /// buffer, its offset and a length. Asserted by offset rather than by
    /// reading, because the shared prefix is what makes it safe to describe the
    /// region's reversal once for both.
    #[test]
    fn the_two_texture_fills_agree_on_their_shared_prefix() {
        use core::mem::offset_of;
        for (name, a, b) in [
            (
                "level",
                offset_of!(FillTextureColor, level),
                offset_of!(FillTextureBytes, level),
            ),
            (
                "slice",
                offset_of!(FillTextureColor, slice),
                offset_of!(FillTextureBytes, slice),
            ),
            (
                "size_width",
                offset_of!(FillTextureColor, size_width),
                offset_of!(FillTextureBytes, size_width),
            ),
            (
                "origin_x",
                offset_of!(FillTextureColor, origin_x),
                offset_of!(FillTextureBytes, origin_x),
            ),
        ] {
            assert_eq!(a, b, "{name} sits at a different offset in the two fills");
        }
        // The region is stored size-first, reversing `MTLRegion`. Both fills.
        assert!(offset_of!(FillTextureColor, size_width) < offset_of!(FillTextureColor, origin_x));
        assert!(offset_of!(FillTextureBytes, size_width) < offset_of!(FillTextureBytes, origin_x));
    }

    /// The `options:` form of the region copy is its plain form plus one word,
    /// laid out at the same offsets.
    #[test]
    fn the_region_options_record_extends_the_plain_one() {
        assert_eq!(
            size_of::<CopyTextureRegionOptions>(),
            size_of::<CopyTextureRegion>() + 4
        );
        assert_eq!(
            COPY_TEXTURE_REGION_OPTIONS_TOTAL_LEN,
            COPY_TEXTURE_REGION_TOTAL_LEN + 4
        );
    }

    /// No opcode belongs to two shape predicates.
    ///
    /// The three multi-opcode shapes are dispatched by predicate rather than by
    /// a match arm per opcode, so an overlap would route a record to a view
    /// whose length check cannot catch it — `Ref` is 4 bytes and fits inside
    /// every other body here.
    #[test]
    fn no_blit_opcode_answers_two_shape_predicates() {
        for opcode in 0x100u32..=0x180 {
            let hits = [
                is_ref(opcode),
                is_ref_slice_level(opcode),
                is_icb_range(opcode),
                opcode == OPCODE_COPY_ICB,
                opcode == OPCODE_FILL_BUFFER,
                opcode == OPCODE_COPY_BUFFER_TO_BUFFER,
                opcode == OPCODE_COPY_TEXTURE_SLICES,
                opcode == OPCODE_COPY_TEXTURE_REGION,
                opcode == OPCODE_COPY_TEXTURE_REGION_OPTIONS,
                opcode == OPCODE_COPY_BUFFER_TO_TEXTURE,
                opcode == OPCODE_COPY_TEXTURE_TO_BUFFER,
                opcode == OPCODE_FILL_BUFFER_PATTERN4,
                opcode == OPCODE_FILL_TEXTURE_COLOR,
                opcode == OPCODE_FILL_TEXTURE_BYTES,
            ]
            .into_iter()
            .filter(|hit| *hit)
            .count();
            assert!(
                hits <= 1,
                "opcode {opcode:#x} answers {hits} shape predicates"
            );
        }
    }

    /// Every accessor refuses a payload one byte short rather than reading past
    /// it. The bytes are guest-controlled, so this is the whole reason the
    /// constructors are fallible.
    #[test]
    fn a_short_payload_is_refused_by_every_accessor() {
        macro_rules! refuses_short {
            ($f:path, $ty:ty, $opcode:expr) => {{
                let need = size_of::<$ty>();
                let mut rec = [0u8; 128];
                let total = (OP_HEADER_LEN + need - 1) as u32;
                rec[..4].copy_from_slice(&($opcode as u32).to_le_bytes());
                rec[4..8].copy_from_slice(&total.to_le_bytes());
                let op = crate::op::op(&rec[..total as usize], 0).expect("header fits");
                assert!(
                    matches!($f(&op), Err(WireError::Short { .. })),
                    "{} accepted a payload one byte short",
                    stringify!($f)
                );
            }};
        }

        refuses_short!(object_ref, Ref, OPCODE_UPDATE_FENCE);
        refuses_short!(ref_slice_level, RefSliceLevel, OPCODE_SYNCHRONIZE_TEXTURE);
        refuses_short!(icb_range, IcbRange, OPCODE_RESET_ICB);
        refuses_short!(copy_icb, CopyIcb, OPCODE_COPY_ICB);
        refuses_short!(fill_buffer, FillBuffer, OPCODE_FILL_BUFFER);
        refuses_short!(
            copy_buffer_to_buffer,
            BufferToBuffer,
            OPCODE_COPY_BUFFER_TO_BUFFER
        );
        refuses_short!(
            copy_texture_slices,
            CopyTextureSlices,
            OPCODE_COPY_TEXTURE_SLICES
        );
        refuses_short!(
            copy_texture_region,
            CopyTextureRegion,
            OPCODE_COPY_TEXTURE_REGION
        );
        refuses_short!(
            copy_texture_region_options,
            CopyTextureRegionOptions,
            OPCODE_COPY_TEXTURE_REGION_OPTIONS
        );
        refuses_short!(
            copy_buffer_to_texture,
            CopyBufferToTexture,
            OPCODE_COPY_BUFFER_TO_TEXTURE
        );
        refuses_short!(
            copy_texture_to_buffer,
            CopyTextureToBuffer,
            OPCODE_COPY_TEXTURE_TO_BUFFER
        );
        refuses_short!(
            fill_buffer_pattern4,
            FillBufferPattern4,
            OPCODE_FILL_BUFFER_PATTERN4
        );
        refuses_short!(
            fill_texture_color,
            FillTextureColor,
            OPCODE_FILL_TEXTURE_COLOR
        );
        refuses_short!(
            fill_texture_bytes,
            FillTextureBytes,
            OPCODE_FILL_TEXTURE_BYTES
        );
    }
}
