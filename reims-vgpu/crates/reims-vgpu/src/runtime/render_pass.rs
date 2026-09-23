//! The render pass's attachments, and what this device can bind of one.
//!
//! # This is a model, not a decoder
//!
//! Every render record is lifted by `reims_vgpu_protocol::decode::render` or,
//! for the rows the closure ledger has not settled, by
//! [`crate::runtime::decode::render_spi`]. What lives here is what neither of
//! those owns: the *device's* attachment types, the subresource coordinates it
//! can and cannot bind, and the three lifts that turn a wire attachment body
//! into one.
//!
//! The distinction is why this module is no longer under `decode/`. A decoder
//! answers "what did the guest write"; this answers "what can this device do
//! with it", and the second question is a capability rather than a layout.
//! `attachment_subresource_is_bindable` is the whole of it: a slice, a depth
//! plane or an unresolved mip level is a coordinate the guest named and this
//! device would silently render past, and the refusal is a statement about the
//! rails rather than about the bytes.
//!
//! The offsets are still here because two of the three lifts read at them —
//! `decode_color_attachment` takes a payload and a slot index — and because
//! the tests that pin the wire layout build their fixtures from them.

use reims_vgpu_wire::ops::render as wire;
use reims_vgpu_wire::ops::render_pass as wire_pass;

use reims_vgpu_wire::OP_HEADER_LEN;

// Layout lengths for fixed-size records and bind tables. Opcodes live in
// `reims_vgpu_wire::ops::{render,render_pass,tile}`; this module maps them into
// product `Kind`/`Command` and does not re-export wire opcode constants.
/// Compact `drawPrimitives:vertexStart:vertexCount:` payload length (`alloc(1, 8)`).
/// Checked exactly: a `0x1` record of any other size is not a form this contract knows.
pub const DRAW_COMPACT_PAYLOAD_LEN: usize = 8;
/// Compact draw total length including the shared op header.
pub const DRAW_COMPACT_CMD_LEN: usize = OP_HEADER_LEN + DRAW_COMPACT_PAYLOAD_LEN;
// The two records' *field* offsets are not named here at all: both arms decode
// through `wire::execute_commands_indirect` / `_range` and read the fields off
// the view, so an offset beside them would be a name for something no code
// asks. The lengths above stay because the arms length-check before viewing.

/// Render-pass attachment layout, taken from the wire structs' own fields.
///
/// The three sections are contiguous, so each record's extent is the distance to
/// the one after it and is never written down separately: depth is
/// `[0x00, 0x28)`, stencil is `[0x28, 0x4c)`, and the color slots run from 0x4c
/// at `PASS_COLOR_ATTACH_STRIDE` each. A single "depth/stencil stride" constant
/// used to state both of the first two as 0x28, which is right for depth and
/// 4 bytes too long for stencil — that spare word is color slot 0's texture ref.
///
/// Offsets are `offset_of!` / `size_of!` on
/// [`reims_vgpu_wire::ops::render_pass`]. Attachment decode maps wire attachment
/// bodies rather than hand-loading fields; `level` is sixteen bits with `slice`
/// immediately above it (a former product colour-arm u32 load swallowed the
/// slice).
#[cfg(test)]
pub(crate) const PASS_DEPTH_ATTACH_OFF: usize = 0x00;
pub const PASS_STENCIL_ATTACH_OFF: usize = core::mem::size_of::<wire_pass::DepthAttachmentBody>();
pub const PASS_COLOR_ATTACH_OFF: usize =
    PASS_STENCIL_ATTACH_OFF + core::mem::size_of::<wire_pass::StencilAttachmentBody>();
pub const PASS_COLOR_ATTACH_STRIDE: usize = core::mem::size_of::<wire_pass::ColorAttachmentBody>();
#[cfg(test)]
pub(crate) const PASS_ATTACH_TEXREF: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, texture_ref);
#[cfg(test)]
pub(crate) const PASS_ATTACH_RESOLVEREF: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, resolve_texture_ref);
pub const PASS_ATTACH_LEVEL: usize = core::mem::offset_of!(wire_pass::AttachmentPrefix, level);
#[cfg(test)]
pub(crate) const PASS_ATTACH_SLICE: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, slice);
#[cfg(test)]
pub(crate) const PASS_ATTACH_DEPTH_PLANE: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, depth_plane);
#[cfg(test)]
pub(crate) const PASS_ATTACH_LOAD_ACTION: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, load_action);
#[cfg(test)]
pub(crate) const PASS_ATTACH_STORE_ACTION: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, store_action);
#[cfg(test)]
pub(crate) const PASS_ATTACH_CLEAR_COLOR: usize =
    core::mem::offset_of!(wire_pass::ColorAttachmentBody, clear_color_bits);
#[cfg(test)]
pub(crate) const PASS_DEPTH_ATTACH_CLEAR_DEPTH: usize =
    core::mem::offset_of!(wire_pass::DepthAttachmentBody, clear_depth_bits);
#[cfg(test)]
pub(crate) const PASS_STENCIL_ATTACH_CLEAR_STENCIL: usize =
    core::mem::offset_of!(wire_pass::StencilAttachmentBody, clear_stencil);
pub const PASS_MAX_COLOR_ATTACHMENTS: usize = wire_pass::RENDER_PASS_COLOR_ATTACHMENTS;

// The five load/store ordinals this record carries are declared in
// `protocol::pass_action`, not here: both backends and the Metal C ABI mirror
// consume them, and while they lived in this decoder the mirror's copy was the
// only spelling the encode path could reach.
pub const PASS_MIN_PAYLOAD: usize = PASS_COLOR_ATTACH_OFF + PASS_COLOR_ATTACH_STRIDE;
/// Count width of `setScissorRects:count:` — eight bytes, not the four used by
/// `setViewports:count:`. The element is the singular scissor payload.
pub const SCISSOR_RECTS_COUNT_LEN: usize = core::mem::size_of::<wire::SetScissorRects>();
/// Bytes one LOD-bearing sampler entry occupies: ref, then two `f32` clamps.
pub const SAMPLER_LOD_BIND_ENTRY_SIZE: usize = core::mem::size_of::<wire::SamplerLodBind>();

/// Residency record head: `count:u32` at `+0` on both forms.
///
/// Four wire opcodes, in two pairs. `wire::OPCODE_USE_HEAP` (`0x1b`) and
/// `wire::OPCODE_USE_RESOURCE` (`0x89`) are the `stages:`-qualified forms the
/// render encoder declares itself; `wire::OPCODE_USE_HEAPS_NO_STAGES` (`0x86`)
/// and `wire::OPCODE_USE_RESOURCES_NO_STAGES` (`0x87`) are the unqualified ones
/// it inherits. All four reach this rail and all four count as one hint.
#[cfg(test)]
pub(crate) const RESIDENCY_COUNT: usize = core::mem::offset_of!(wire::UseResource, count);
/// `useResource:` packs `usage` and `stages` into the word at `+4` as two
/// `u16`s, so its refs begin at `+8` — the size of the head the view declares.
#[cfg(test)]
pub(crate) const USE_RESOURCE_REFS: usize = core::mem::size_of::<wire::UseResource>();
/// `useHeap:` has no `usage` at all: `stages` sits alone at `+4` as a `u16` and
/// the refs begin at `+6`. That offset is deliberately not a multiple of four —
/// reading this record with the resource record's layout skips the first heap.
/// Two heads that differ by one field is exactly the pair that must not be two
/// numbers here, so both are the view's own size.
#[cfg(test)]
pub(crate) const USE_HEAP_REFS: usize = core::mem::size_of::<wire::UseHeap>();
/// The inherited forms take no `stages:`, so `useHeaps:count:` is a bare count
/// with its refs at `+4` and `useResources:count:usage:` puts them at `+8`.
/// Three head sizes across four opcodes, which is why each reads its own.
#[cfg(test)]
pub(crate) const USE_HEAPS_NO_STAGES_REFS: usize = core::mem::size_of::<wire::UseHeapsNoStages>();
#[cfg(test)]
pub(crate) const USE_RESOURCES_NO_STAGES_REFS: usize =
    core::mem::size_of::<wire::UseResourcesNoStages>();

/// Multi-entry bind header: `first:u32 @0`, `count:u32 @4`, entries after it.
#[cfg(test)]
pub(crate) const BIND_FIRST: usize = core::mem::offset_of!(wire::BindHeader, first);
#[cfg(test)]
pub(crate) const BIND_COUNT: usize = core::mem::offset_of!(wire::BindHeader, count);
pub const BIND_ENTRIES: usize = core::mem::size_of::<wire::BindHeader>();
pub const BUFFER_BIND_ENTRY_SIZE: usize = core::mem::size_of::<wire::BufferBind>();
/// The same entry with a `u64` attribute stride appended. See
/// [`wire::OPCODE_SET_VERTEX_BUFFER_STRIDE`].
pub const BUFFER_STRIDE_BIND_ENTRY_SIZE: usize = core::mem::size_of::<wire::BufferStrideBind>();
/// `setVertexAmplificationCount:viewMappings:`: a four-byte count, then one
/// `MTLVertexAmplificationViewMapping` (two `u32`) per view.
#[cfg(test)]
pub(crate) const AMPLIFICATION_COUNT_LEN: usize =
    core::mem::size_of::<wire::VertexAmplificationHeader>();
#[cfg(test)]
pub(crate) const AMPLIFICATION_MAPPING_SIZE: usize = core::mem::size_of::<wire::ViewMapping>();
pub const REF_BIND_ENTRY_SIZE: usize = core::mem::size_of::<wire::RefBind>();

/// Bytes a bind record needs for `count` entries of `entry_size`, or `None` if
/// no record could be that long.
///
/// **A bind record is bounded by its own length and by nothing else.** This
/// replaced a `MAX_BIND_ENTRIES = 32` cap that had no citation and was not
/// Apple's: `setVertexTextures:withRange:` over a range of 40 produces a
/// 176-byte record (fixture `render_set_vertex_textures_range_40`), which that
/// cap refused with `ErrBadLength` — dropping all forty binds rather than the
/// eight that would not fit a table. Metal's own limit is 128 textures per
/// stage, so 32 was not even the API's number.
///
/// The count stays guest-controlled and is never trusted before this check:
/// nothing is allocated or read until the entries are known to be inside the
/// record the guest itself sized.
#[inline]
pub fn bind_record_len(count: u32, entry_size: usize) -> Option<usize> {
    (count as usize)
        .checked_mul(entry_size)
        .and_then(|n| n.checked_add(BIND_ENTRIES))
}
/// set*BufferOffset: index:u32 @0, offset:u64 @4 (payload 12; full cmd 0x14).
#[cfg(test)]
pub(crate) const BUFFER_OFFSET_INDEX: usize = core::mem::offset_of!(wire::BufferOffset, index);
#[cfg(test)]
pub(crate) const BUFFER_OFFSET_VALUE: usize = core::mem::offset_of!(wire::BufferOffset, offset);
#[cfg(test)]
pub(crate) const BUFFER_OFFSET_PAYLOAD_LEN: usize = core::mem::size_of::<wire::BufferOffset>();
/// One scissor rectangle's extent. Its four fields are not named here: both
/// scissor arms read them off `wire::ScissorRect` through the view.
#[cfg(test)]
pub(crate) const SCISSOR_PAYLOAD_LEN: usize = core::mem::size_of::<wire::ScissorRect>();

// Supported window is the full C-accepted encoder range 0x00..=0x98 minus rejected.

/// One color attachment from a render-pass descriptor (0x1a).
///
/// # Whether a slot is bound is `texture_ref != 0`, and only that
///
/// All three attachment shapes carried a `present: bool` beside `texture_ref`
/// that every decode path set to `texture_ref != 0` — the derived copy of a
/// field sitting next to the field it is derived from. The two could disagree
/// only by construction, and did: a bound-but-textureless attachment is not a
/// thing this decoder can produce, yet a caller could build one and a consumer
/// reading `present` would honour it. One call site had already written both
/// halves in one expression (`!att.present || att.texture_ref == 0`), which is
/// the same test twice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColorAttachment {
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    /// Array slice, sixteen bits on the wire directly above `level`.
    pub slice: u32,
    /// Depth plane of a 3D attachment, sixteen bits above `slice`.
    pub depth_plane: u32,
    pub load_action: u16,
    pub store_action: u16,
    /// `MTLClearColor` as RGBA doubles. The attachment pixel format decides
    /// whether those components are continuous values or integer counts; an
    /// integer clear of `1.0` means `1`, not the format's normalized maximum.
    pub clear_color: [f64; 4],
}

/// Depth attachment from a render-pass descriptor (slot @0x00).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthAttachment {
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    /// Array slice and depth plane, the two sixteen-bit fields above `level`.
    ///
    /// All three attachment shapes share one 28-byte prefix — that is what
    /// `reims_vgpu_wire::ops::render_pass::AttachmentPrefix` is — so these are
    /// here for the same reason they are on [`ColorAttachment`]: a depth buffer
    /// bound at slice 5 is as real as a colour target bound there, and a field
    /// nothing decodes is a field nothing can report.
    pub slice: u32,
    pub depth_plane: u32,
    pub load_action: u16,
    pub store_action: u16,
    pub clear_depth: f64,
}

/// A scissor rectangle in target texels, as `MTLScissorRect` declares one.
///
/// A type because the four numbers used to travel as four loose fields here, as
/// an `Option<(u32, u32, u32, u32)>` through two request structs, and as four
/// and then six adjacent `u32` parameters into the coverage census — where the
/// rect sat next to the target extent and every permutation of the six
/// compiled. `ScissorResource` and `MTLScissorRect` are the two backends' own
/// ABI shapes and stay; this is the one the decode produces and the device
/// carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScissorRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ScissorRect {
    /// Whether this rect reaches every texel of a `target_w` x `target_h`
    /// attachment. A draw that does could have written anywhere in it, so
    /// nothing downstream can bound its writes by the scissor.
    pub fn covers(&self, target_w: u32, target_h: u32) -> bool {
        self.x == 0 && self.y == 0 && self.width >= target_w && self.height >= target_h
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Lift one wire rect. Shared by the singular and plural scissor opcodes, which
/// carry the identical element and differ only in how many of it follow.
pub(crate) fn scissor_from_wire(r: &wire::ScissorRect) -> ScissorRect {
    ScissorRect {
        x: r.x.get() as u32,
        y: r.y.get() as u32,
        width: r.width.get() as u32,
        height: r.height.get() as u32,
    }
}

/// Lift one wire viewport, in the `[originX, originY, width, height, znear,
/// zfar]` order both backends read it back in. Shared by the singular and
/// plural viewport opcodes for the same reason as [`scissor_from_wire`].
pub(crate) fn viewport_from_wire(v: &wire::Viewport) -> [f64; 6] {
    [
        v.origin_x.get(),
        v.origin_y.get(),
        v.width.get(),
        v.height.get(),
        v.znear.get(),
        v.zfar.get(),
    ]
}

/// The subresource coordinates and resolve target shared by all three
/// attachment shapes, lifted so the arms that read them cannot drift apart.
///
/// They are one 28-byte prefix on the wire
/// ([`reims_vgpu_wire::ops::render_pass::AttachmentPrefix`]), and this device
/// had two arms reading it with two copies of the same four-line check. A
/// third copy is what the colour arm would have needed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttachSubresource {
    pub level: u32,
    pub slice: u32,
    pub depth_plane: u32,
    pub resolve_texture_ref: u32,
}

impl From<DepthAttachment> for AttachSubresource {
    fn from(a: DepthAttachment) -> Self {
        Self {
            level: a.level,
            slice: a.slice,
            depth_plane: a.depth_plane,
            resolve_texture_ref: a.resolve_texture_ref,
        }
    }
}

impl From<StencilAttachment> for AttachSubresource {
    fn from(a: StencilAttachment) -> Self {
        Self {
            level: a.level,
            slice: a.slice,
            depth_plane: a.depth_plane,
            resolve_texture_ref: a.resolve_texture_ref,
        }
    }
}

impl From<ColorAttachment> for AttachSubresource {
    fn from(a: ColorAttachment) -> Self {
        Self {
            level: a.level,
            slice: a.slice,
            depth_plane: a.depth_plane,
            resolve_texture_ref: a.resolve_texture_ref,
        }
    }
}

/// Whether the arm asking [`attachment_subresource_is_bindable`] can render into
/// a mip level other than zero.
///
/// The arms genuinely differ, which is why this is a parameter rather than a
/// second predicate. The colour rail materializes a level's own plane inside the
/// guest allocation — `TextureDescriptor::level_gva` gives its address, stride
/// and geometry, and `render_target`'s linear rung has rendered into one since
/// texture-view mip views existed. The depth/stencil rail has no such rung: it would
/// bind level 0 and the guest would read a level it never wrote.
///
/// Making it an enum rather than a `bool` is the point — an arm has to say which
/// it is, and it cannot say it by accident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelSupport {
    /// This arm renders into level 0 and nothing else.
    LevelZeroOnly,
    /// This arm resolves the named level's own plane.
    AnyLevel,
}

/// Whether this device can honour an attachment's subresource as decoded.
///
/// Slice 0, plane 0, no multisample resolve for callers using this predicate, and a level the
/// caller's rail can reach. `slice` and `depth_plane` joined the test when they became decodable: a
/// depth buffer bound at slice 5 was previously read as slice 0 and silently accepted.
///
/// It lives beside the structs it reads because four arms apply it — the stream
/// decode that admits an attachment into a pass, once per aspect, and the Metal
/// rail that builds a host-side buffer for one. **Every hand-written copy of it
/// that has existed was missing a term.** The rail's tested `level` and
/// `resolve_texture_ref` only, so the two `u16` fields above `level` were
/// checked in one place and not the other. The colour arm's tested `level`,
/// `slice` and `depth_plane` and not `resolve_texture_ref`, so a multisample
/// colour pass — the attachment multisampled, `storeAction =
/// MultisampleResolve`, `resolveTexture` naming where the single-sampled result
/// goes — was admitted, rendered at one sample into the attachment, and left
/// the resolve target the guest goes on to read holding whatever it held.
///
/// That is why it takes [`AttachSubresource`] rather than any one attachment
/// type: a fifth arm gets the whole rule or does not compile.
pub fn attachment_subresource_is_bindable(s: AttachSubresource, levels: LevelSupport) -> bool {
    let level_ok = match levels {
        LevelSupport::LevelZeroOnly => s.level == 0,
        LevelSupport::AnyLevel => true,
    };
    level_ok && s.slice == 0 && s.depth_plane == 0 && s.resolve_texture_ref == 0
}

/// Whether a colour attachment's directly-addressed coordinates are bindable.
///
/// A resolve texture is a second attachment and an end-of-pass operation, not
/// a coordinate of the multisample source. Keep it intact so the backend can
/// encode or precisely refuse that operation. Depth and stencil continue to
/// use [`attachment_subresource_is_bindable`] because their backend request
/// types do not yet carry resolve destinations.
pub fn color_attachment_subresource_is_bindable(s: AttachSubresource) -> bool {
    s.slice == 0 && s.depth_plane == 0
}

/// Stencil attachment from a render-pass descriptor (slot @0x28).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StencilAttachment {
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    /// See [`DepthAttachment::slice`]; the prefix is the same 28 bytes.
    pub slice: u32,
    pub depth_plane: u32,
    pub load_action: u16,
    pub store_action: u16,
    pub clear_stencil: u32,
}

/// Decode color attachment slot `index` from a render-pass payload.
///
/// # `level` is sixteen bits and `slice` is the sixteen above it
///
/// This read was `ld32` at `PASS_ATTACH_LEVEL` for as long as the function has
/// existed, with a comment on [`decode_depth_attachment`] stating the rule as
/// "the archive uses u16 for depth/stencil level (color uses u32)". Nothing had
/// ever checked that; Apple's own bytes say all three shapes are identical here,
/// and the four bytes at `+0x08` are `level` then `slice`.
///
/// So a pass rendering into array slice 1 — a cube face, a texture-array layer,
/// a layered shadow map — reported mip level 65536 and lost its slice entirely.
/// Both are decoded now.
pub(crate) fn color_from_wire(c: &wire_pass::ColorAttachmentBody) -> ColorAttachment {
    let p = &c.prefix;
    ColorAttachment {
        texture_ref: p.texture_ref.get(),
        resolve_texture_ref: p.resolve_texture_ref.get(),
        level: u32::from(p.level.get()),
        slice: u32::from(p.slice.get()),
        depth_plane: u32::from(p.depth_plane.get()),
        load_action: p.load_action.get(),
        store_action: p.store_action.get(),
        clear_color: c.clear_color(),
    }
}

pub(crate) fn depth_from_wire(d: &wire_pass::DepthAttachmentBody) -> DepthAttachment {
    let p = &d.prefix;
    DepthAttachment {
        texture_ref: p.texture_ref.get(),
        resolve_texture_ref: p.resolve_texture_ref.get(),
        level: u32::from(p.level.get()),
        slice: u32::from(p.slice.get()),
        depth_plane: u32::from(p.depth_plane.get()),
        load_action: p.load_action.get(),
        store_action: p.store_action.get(),
        clear_depth: d.clear_depth(),
    }
}

pub(crate) fn stencil_from_wire(s: &wire_pass::StencilAttachmentBody) -> StencilAttachment {
    let p = &s.prefix;
    StencilAttachment {
        texture_ref: p.texture_ref.get(),
        resolve_texture_ref: p.resolve_texture_ref.get(),
        level: u32::from(p.level.get()),
        slice: u32::from(p.slice.get()),
        depth_plane: u32::from(p.depth_plane.get()),
        load_action: p.load_action.get(),
        store_action: p.store_action.get(),
        clear_stencil: s.clear_stencil.get(),
    }
}

pub fn decode_color_attachment(payload: &[u8], index: usize) -> ColorAttachment {
    let base = PASS_COLOR_ATTACH_OFF + index * PASS_COLOR_ATTACH_STRIDE;
    match reims_vgpu_wire::view_at::<wire_pass::ColorAttachmentBody>(payload, base) {
        Ok(c) => color_from_wire(c),
        Err(_) => ColorAttachment::default(),
    }
}

/// Decode the depth attachment (fixed slot @0).
pub fn decode_depth_attachment(payload: &[u8]) -> DepthAttachment {
    match reims_vgpu_wire::view::<wire_pass::DepthAttachmentBody>(payload) {
        Ok(d) if payload.len() >= PASS_STENCIL_ATTACH_OFF => depth_from_wire(d),
        _ => DepthAttachment::default(),
    }
}

/// Decode the stencil attachment (fixed slot after depth).
pub fn decode_stencil_attachment(payload: &[u8]) -> StencilAttachment {
    match reims_vgpu_wire::view_at::<wire_pass::StencilAttachmentBody>(
        payload,
        PASS_STENCIL_ATTACH_OFF,
    ) {
        Ok(s) => stencil_from_wire(s),
        Err(_) => StencilAttachment::default(),
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::endian::{st16, st32, st64};

    #[test]
    fn depth_and_stencil_pass_slots() {
        use reims_vgpu_protocol::pass_action::{MTL_LOAD_ACTION_CLEAR, MTL_STORE_ACTION_STORE};
        let mut payload = vec![0u8; PASS_MIN_PAYLOAD];
        // depth @0
        st32(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            77,
        );
        st16(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_LOAD_ACTION..],
            MTL_LOAD_ACTION_CLEAR,
        );
        st16(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_STORE_ACTION..],
            MTL_STORE_ACTION_STORE,
        );
        st64(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_DEPTH_ATTACH_CLEAR_DEPTH..],
            0.5f64.to_bits(),
        );
        // stencil @0x28
        st32(
            &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            88,
        );
        st32(
            &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_STENCIL_ATTACH_CLEAR_STENCIL..],
            9,
        );
        let d = decode_depth_attachment(&payload);
        assert_eq!(d.texture_ref, 77);
        assert!((d.clear_depth - 0.5).abs() < 1e-9);
        let s = decode_stencil_attachment(&payload);
        assert_eq!(s.texture_ref, 88);
        assert_eq!(s.clear_stencil, 9);
    }

    /// Each of the first two records ends where the next one begins: depth
    /// `[0x00, 0x28)`, stencil `[0x28, 0x4c)`. A payload that carries both in
    /// full — and not one byte of the color section — must decode both.
    ///
    /// A shared `PASS_DEPTH_STENCIL_ATTACH_STRIDE = 0x28` used to give the
    /// stencil record the depth record's length, so the decoder demanded 0x50
    /// bytes to read a 0x24-byte record and sliced 4 bytes past its end, over
    /// color slot 0's texture ref. This payload is exactly `PASS_COLOR_ATTACH_OFF`
    /// long, so the old guard rejected it and returned a defaulted attachment.
    #[test]
    fn depth_and_stencil_records_end_where_the_next_section_begins() {
        let mut payload = vec![0u8; PASS_COLOR_ATTACH_OFF];
        st32(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            31,
        );
        st64(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_DEPTH_ATTACH_CLEAR_DEPTH..],
            0.25f64.to_bits(),
        );
        st32(
            &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            32,
        );
        st32(
            &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_STENCIL_ATTACH_CLEAR_STENCIL..],
            0xfe,
        );
        let d = decode_depth_attachment(&payload);
        assert_eq!(
            d.texture_ref, 31,
            "depth record is complete at {PASS_STENCIL_ATTACH_OFF} bytes"
        );
        assert!((d.clear_depth - 0.25).abs() < 1e-9);
        let s = decode_stencil_attachment(&payload);
        assert_eq!(
            s.texture_ref, 32,
            "stencil record is complete at {PASS_COLOR_ATTACH_OFF} bytes"
        );
        assert_eq!(s.clear_stencil, 0xfe);
    }

    /// Whether a scissor reaches the whole target is one rule, and it had been
    /// written twice in opposite polarities.
    ///
    /// The draw-coverage census asked
    /// `x == 0 && y == 0 && w >= target_w && h >= target_h`; the partial-store
    /// path asked `x > 0 || y > 0 || w < width || h < height` and acted on the
    /// negation. Two spellings of one predicate, in two files, over four
    /// numbers that travelled loose. Each row below is a case where a term
    /// dropped from either spelling would change the answer.
    #[test]
    fn a_scissor_covers_its_target_only_when_every_term_says_so() {
        let full = ScissorRect {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
        };
        assert!(full.covers(800, 600), "an exact fit covers");
        assert!(
            ScissorRect {
                width: 900,
                height: 700,
                ..full
            }
            .covers(800, 600),
            "a scissor larger than the target still covers it"
        );
        for (name, rect) in [
            ("offset x", ScissorRect { x: 1, ..full }),
            ("offset y", ScissorRect { y: 1, ..full }),
            ("narrow", ScissorRect { width: 799, ..full }),
            (
                "short",
                ScissorRect {
                    height: 599,
                    ..full
                },
            ),
        ] {
            assert!(
                !rect.covers(800, 600),
                "a scissor {name} must not read as covering the target"
            );
        }

        // A zero-extent rect draws nothing; the stream decode drops it and
        // keeps the previous scissor rather than binding an empty one.
        assert!(!full.is_empty());
        assert!(ScissorRect { width: 0, ..full }.is_empty());
        assert!(ScissorRect { height: 0, ..full }.is_empty());
    }

    /// All four fields of the prefix decide bindability, and both attachment
    /// shapes hand all four to the rule.
    ///
    /// The rule had two consumers and the second carried its own copy testing
    /// `level` and `resolve_texture_ref` only — so a depth buffer bound at
    /// slice 5 was refused by the stream decode and would have been accepted by
    /// the Metal rail. This drives each field on its own, from both shapes, so
    /// a consumer that reconstructs three of the four fails here rather than at
    /// a guest that binds an array layer.
    ///
    /// `level` is driven from both [`LevelSupport`] arms, because it is the one
    /// field whose answer depends on which rail is asking: a colour attachment's
    /// level resolves to its own plane in the guest allocation and a depth
    /// attachment's does not. Every other field must refuse on both arms — an arm
    /// that reads `AnyLevel` as "anything goes" fails here.
    #[test]
    fn every_field_of_the_attachment_prefix_decides_bindability() {
        for levels in [LevelSupport::LevelZeroOnly, LevelSupport::AnyLevel] {
            assert!(
                attachment_subresource_is_bindable(AttachSubresource::default(), levels),
                "{levels:?}: the whole texture at level 0, slice 0, plane 0 with no resolve is bindable"
            );
        }
        let mip = AttachSubresource {
            level: 1,
            ..AttachSubresource::default()
        };
        assert!(
            !attachment_subresource_is_bindable(mip, LevelSupport::LevelZeroOnly),
            "a rail that only renders level 0 must refuse a level the guest named"
        );
        assert!(
            attachment_subresource_is_bindable(mip, LevelSupport::AnyLevel),
            "a rail that resolves the named level's own plane must admit it"
        );

        for (name, sub) in [
            (
                "slice",
                AttachSubresource {
                    slice: 5,
                    ..AttachSubresource::default()
                },
            ),
            (
                "depth_plane",
                AttachSubresource {
                    depth_plane: 2,
                    ..AttachSubresource::default()
                },
            ),
            (
                "resolve_texture_ref",
                AttachSubresource {
                    resolve_texture_ref: 99,
                    ..AttachSubresource::default()
                },
            ),
        ] {
            for levels in [LevelSupport::LevelZeroOnly, LevelSupport::AnyLevel] {
                assert!(
                    !attachment_subresource_is_bindable(sub, levels),
                    "{levels:?}: a non-default {name} must refuse the attachment on its own"
                );
            }
        }

        // And both shapes hand all four to the rule: a field a conversion drops
        // arrives as 0, which is the value that admits.
        let all_four = AttachSubresource {
            level: 1,
            slice: 5,
            depth_plane: 2,
            resolve_texture_ref: 99,
        };
        assert_eq!(
            AttachSubresource::from(DepthAttachment {
                texture_ref: 77,
                level: 1,
                slice: 5,
                depth_plane: 2,
                resolve_texture_ref: 99,
                ..DepthAttachment::default()
            }),
            all_four
        );
        assert_eq!(
            AttachSubresource::from(StencilAttachment {
                texture_ref: 88,
                level: 1,
                slice: 5,
                depth_plane: 2,
                resolve_texture_ref: 99,
                ..StencilAttachment::default()
            }),
            all_four
        );
    }

    /// The store-action options and the tessellation factor buffer.
    ///
    /// `0x67`/`0x6a`/`0x79` sit one opcode above the three store actions and
    /// look like longer forms of them. They are not, and the difference is a
    /// *width*: the options are a `u64` where the action is a `u32`, so the
    /// colour form's attachment index moves from payload `+4` to `+8` and the
    /// record grows from 16 bytes to 20. A decoder that reused
    /// `ColorStoreAction` here would read the index out of the options' high
    /// half and report attachment 0 for every one of them.
    ///
    /// `0x7a` is checked in the same test because it makes the opposite
    /// mistake available: it names a buffer with an offset, so it reads like a
    /// bind, and a reader that took a `BindHeader` would call the ref `first`
    /// and the low half of the offset `count`.
    /// A colour attachment's `level` is sixteen bits, and `slice` is the
    /// sixteen above it.
    ///
    /// This arm read `ld32` at [`PASS_ATTACH_LEVEL`] for as long as it existed,
    /// under a comment on the depth arm that stated the rule as "the archive
    /// uses u16 for depth/stencil level (color uses u32)". Apple's own bytes say
    /// all three attachment shapes are identical through their prefix, so the
    /// wide read returned `level | (slice << 16)`: a pass rendering into array
    /// slice 1 reported mip level 65536 and lost the slice.
    ///
    /// The synthetic here is what a cube-face pass looks like — level 1, slice
    /// 5 — and the two fields are read apart. The `0xffff` case is the same
    /// claim at the boundary: a slice that fills its half must not reach the
    /// level at all.
    #[test]
    fn a_colour_attachments_level_does_not_swallow_its_slice() {
        for (level, slice, plane) in [(1u16, 5u16, 2u16), (0, 0xffff, 0), (0xffff, 0, 0)] {
            let total = OP_HEADER_LEN + PASS_MIN_PAYLOAD;
            let mut cmd = vec![0u8; total];
            st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
            st32(&mut cmd[4..], total as u32);
            let slot = OP_HEADER_LEN + PASS_COLOR_ATTACH_OFF;
            st32(&mut cmd[slot + PASS_ATTACH_TEXREF..], 7);
            cmd[slot + PASS_ATTACH_LEVEL..slot + PASS_ATTACH_LEVEL + 2]
                .copy_from_slice(&level.to_le_bytes());
            cmd[slot + PASS_ATTACH_SLICE..slot + PASS_ATTACH_SLICE + 2]
                .copy_from_slice(&slice.to_le_bytes());
            cmd[slot + PASS_ATTACH_DEPTH_PLANE..slot + PASS_ATTACH_DEPTH_PLANE + 2]
                .copy_from_slice(&plane.to_le_bytes());
            let att = decode_color_attachment(&cmd[OP_HEADER_LEN..], 0);
            assert_eq!(att.level, u32::from(level), "level took the slice's bits");
            assert_eq!(att.slice, u32::from(slice), "slice went unread");
            assert_eq!(att.depth_plane, u32::from(plane), "depth plane went unread");
        }
    }
}
