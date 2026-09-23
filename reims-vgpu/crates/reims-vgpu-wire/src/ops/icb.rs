//! Opcode `0x36` — create indirect command buffer.
//!
//! `-[PGSerializer newIndirectCommandBufferWithDescriptor:layout:
//! maxCommandCount:options:allocator:]`, the only creation record whose
//! descriptor half the type encoding does not describe.
//!
//! # Layout
//!
//! Total 88 bytes: the 8-byte [`crate::op::OpHeader`] then an 80-byte payload.
//!
//! ```text
//! payload +000  u32  command_types    MTLIndirectCommandType, verbatim
//! payload +004  u8   max_vertex_buffer_bind_count
//! payload +005  u8   max_fragment_buffer_bind_count
//! payload +006  u8   max_kernel_buffer_bind_count
//! payload +007  u8   max_object_buffer_bind_count
//! payload +008  u8   max_mesh_buffer_bind_count
//! payload +009  u8   max_kernel_threadgroup_memory_bind_count
//! payload +010  u8   max_object_threadgroup_memory_bind_count
//! payload +011  u8   unidentified_u8_a   reads 1 on every case
//! payload +012  u8   unidentified_u8_b   reads 1 on every case
//! payload +013  u8   never written by the serializer
//! payload +014  u16  flags               see [`NewIcbBody::inherit_buffers`] &c
//! payload +016  u32  unidentified_u32    reads 0 on every case
//! payload +020  ..   layout, 52 bytes    [`IcbLayout`]
//! payload +072  u32  max_command_count
//! payload +076  u16  options             MTLResourceOptions, narrowed
//! payload +078  u16  never written by the serializer
//! ```
//!
//! # This record does not carry the new object's ref
//!
//! Every other creation record in this protocol begins its payload with the ref
//! the serializer's allocator handed out — texture, sampler, depth/stencil, the
//! three texture views, both backed textures, the heap-placed texture, the
//! fence and the rate map all do, and [`crate::ops::texture`] documents that as
//! the shape of the family. This one does not, and the first payload word is
//! `commandTypes` instead.
//!
//! That is measured, not inferred from a gap. Sixteen cases were driven through
//! one capture, the serializer's allocator handed each a different ref (68
//! through 83, consecutive), and **the returned ref appears at no offset in any
//! of the sixteen records** — searched as a `u32` at every byte position, not
//! just the aligned ones. The four zero bytes at payload `+016` are the only
//! candidate slot and they read 0 in all sixteen.
//!
//! A host therefore cannot name the object this record creates from the record.
//! What would settle how the pairing is meant to work is a guest boot: drive a
//! guest that creates an indirect command buffer and then names it in a later
//! record — `optimizeIndirectCommandBuffer:withRange:` and the `0x1d1` info
//! query both carry an ICB ref — and see whether the ref it uses is the one an
//! allocation-ordered counter would predict.
//!
//! # How the layout was derived
//!
//! The `layout:` argument's 52 bytes come free from the Objective-C type
//! encoding, which declares it `^{?=SSSSIIIIIIIIIII}` — four `u16` then eleven
//! `u32`. Two cases with different seeds show all fifteen reach the wire
//! verbatim, in order, at payload `+020`.
//!
//! The descriptor half has no such source: `MTLIndirectCommandBufferDescriptor`
//! is an object, and nothing in the encoding says which of its properties the
//! serializer copies or where. Every field named above moved in a case that
//! changed exactly that property off a common baseline, and every field *not*
//! named is one that no case moved. See `oracle/oracle.m`, the
//! `serializer_new_indirect_command_buffer*` family.
//!
//! # The flag word's five unattributed bits
//!
//! Bits 0–5, 7, 8, 9 and 10 each moved with one BOOL property, and each of the
//! eleven BOOLs `MTLIndirectCommandBufferDescriptor` declares was inverted from
//! its default in its own case — so no bit was named from a guess about
//! ordering. What is left is bits 6 and 11–14, which read **1** in every case
//! and which none of the eleven properties moves; bit 15 is never written at
//! all. Note in particular that bit 6 sits *between* two named bits, so the
//! assignment is not a contiguous run and cannot be extended by counting.
//!
//! One property is measured to reach the wire **nowhere**:
//! `supportColorAttachmentMapping`, set to `YES`, produces a record byte-for-
//! byte identical to the baseline. So a guest asking for colour-attachment
//! mapping emits a record that does not say so.

use crate::le::{U16le, U32le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

/// Opcode for indirect-command-buffer creation.
pub const OPCODE_NEW_INDIRECT_COMMAND_BUFFER: u32 = 0x36;

/// Total wire length, header included.
pub const NEW_INDIRECT_COMMAND_BUFFER_TOTAL_LEN: u32 = 88;

/// The `layout:` argument, copied onto the wire verbatim.
///
/// The type encoding `^{?=SSSSIIIIIIIIIII}` fixes the widths and the order and
/// gives no names, so the fields are positional here. Naming them would be a
/// claim this crate cannot support — see this crate's `AGENTS.md` on naming the
/// unknown. Fixtures `serializer_new_indirect_command_buffer` (seed `0x11`) and
/// `serializer_new_indirect_command_buffer_count_options` (seed `0x44`) carry
/// fifteen distinct values each, so a field read at the wrong offset reads the
/// wrong neighbour rather than a plausible value.
///
/// To settle what they mean: `MTLIndirectCommandBufferDescriptor` has no
/// fifteen-field companion struct in the public SDK, so this is an SPI layout.
/// The experiment is a guest boot that issues an ICB creation and a later
/// `MTLIndirectCommandBuffer` use whose behaviour depends on one of them.
#[repr(C)]
#[derive(Debug)]
pub struct IcbLayout {
    /// The `SSSS` half of the encoding, in declaration order.
    pub words16: [U16le; 4],
    /// The `IIIIIIIIIII` half, in declaration order.
    pub words32: [U32le; 11],
}

// SAFETY: two arrays of align-1 all-bytes-valid `le` scalars; an array of an
// align-1 type is align-1 and every byte pattern is a valid inhabitant.
unsafe impl Wire for IcbLayout {}

/// The whole payload of `0x36`.
#[repr(C)]
#[derive(Debug)]
pub struct NewIcbBody {
    /// `MTLIndirectCommandType`, an option set the serializer carries
    /// unchanged: `MTLIndirectCommandTypeDraw` (1) reads 1, and
    /// `DrawIndexed | DrawPatches | ConcurrentDispatch |
    /// ConcurrentDispatchThreads` (0x66) reads `0x66`. So a decoder must treat
    /// an out-of-range bit as guest data rather than as impossible.
    pub command_types: U32le,
    pub max_vertex_buffer_bind_count: u8,
    pub max_fragment_buffer_bind_count: u8,
    pub max_kernel_buffer_bind_count: u8,
    pub max_object_buffer_bind_count: u8,
    pub max_mesh_buffer_bind_count: u8,
    /// Note the default: this reads **31** on a descriptor nothing set, where
    /// the six counts beside it read 0. `serializer_new_indirect_command_buffer_
    /// stage_bind_counts` moves it to 10, which is what separates a non-zero
    /// default from a constant.
    pub max_kernel_threadgroup_memory_bind_count: u8,
    pub max_object_threadgroup_memory_bind_count: u8,
    /// Reads 1 in every case. Every `MTLIndirectCommandBufferDescriptor`
    /// property with a setter was driven and none moves it.
    ///
    /// To settle it: drive a serializer whose `-setSupports…` capability set
    /// differs, or a guest whose ICB is created against a device with a
    /// different bind-count limit — this is the shape of a field carrying a
    /// limit the descriptor does not expose.
    pub unidentified_u8_a: u8,
    /// Reads 1 in every case, on the same evidence as
    /// [`NewIcbBody::unidentified_u8_a`] and with the same experiment
    /// outstanding.
    pub unidentified_u8_b: u8,
    /// **Never written by the serializer.** On a real wire this holds whatever
    /// the guest's ring last contained, so nothing may read it. Measured by the
    /// oracle's complementary-fill passes; see this crate's `AGENTS.md`.
    pub never_written: u8,
    /// The BOOL properties, one bit each. Use the accessors below rather than
    /// this word: bit 15 is never written and bits 6 and 11–14 are unidentified.
    pub flags: U16le,
    /// Reads 0 in every case, and is the only slot the missing object ref could
    /// have occupied — see this module's doc.
    pub unidentified_u32: U32le,
    pub layout: IcbLayout,
    /// Declared `Q` by the selector and written as four bytes: the two above
    /// the low half read zero and the serializer *does* write them, which is
    /// what separates this field from [`NewIcbBody::options`] beside it.
    pub max_command_count: U32le,
    /// `MTLResourceOptions`, declared `Q` and narrowed to **sixteen bits** on
    /// the wire. `MTLResourceStorageModePrivate` (32) reads 32.
    ///
    /// The width is measured rather than assumed from the neighbour: the two
    /// bytes above it are never written, so a `u32` here reads the guest's
    /// stale ring in its top half. That is exactly the shape of the
    /// `copyFromTexture:toBuffer:` `options` bug this project has already
    /// shipped once — see this crate's `AGENTS.md`.
    pub options: U16le,
    /// **Never written by the serializer.** The two bytes above `options`.
    pub never_written_tail: U16le,
}

// SAFETY: align-1 `le` scalars and one align-1 `Wire` struct; every byte
// pattern is a valid inhabitant of all of them.
unsafe impl Wire for NewIcbBody {}

/// Bit positions in [`NewIcbBody::flags`], each derived from the one case that
/// inverted that property from the value the baseline read back.
pub mod flag {
    /// `inheritPipelineState`. Baseline 0 → bit set when `YES`.
    pub const INHERIT_PIPELINE_STATE: u16 = 1 << 0;
    /// `inheritBuffers`. Baseline 0.
    pub const INHERIT_BUFFERS: u16 = 1 << 1;
    /// `supportRayTracing`. Baseline 0.
    pub const SUPPORT_RAY_TRACING: u16 = 1 << 2;
    /// `supportDynamicAttributeStride`. Baseline 0.
    pub const SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE: u16 = 1 << 3;
    /// `inheritDepthStencilState`. Baseline **1** — clearing it is what moved
    /// this bit, which is why the case sets the property to `NO`.
    pub const INHERIT_DEPTH_STENCIL_STATE: u16 = 1 << 4;
    /// `inheritDepthBias`. Baseline 1.
    pub const INHERIT_DEPTH_BIAS: u16 = 1 << 5;
    /// `inheritDepthClipMode`. Baseline 1. Note it is bit **7**, not bit 6:
    /// the run of inherit bits is not contiguous.
    pub const INHERIT_DEPTH_CLIP_MODE: u16 = 1 << 7;
    /// `inheritCullMode`. Baseline 1.
    pub const INHERIT_CULL_MODE: u16 = 1 << 8;
    /// `inheritFrontFacingWinding`. Baseline 1.
    pub const INHERIT_FRONT_FACING_WINDING: u16 = 1 << 9;
    /// `inheritTriangleFillMode`. Baseline 1.
    pub const INHERIT_TRIANGLE_FILL_MODE: u16 = 1 << 10;

    /// Bits 6 and 11–14: set in every fixture, moved by no property. Bit 15 is
    /// excluded because the serializer never writes it.
    pub const UNIDENTIFIED: u16 = (1 << 6) | (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14);

    /// Bit 15, which the serializer never writes. A decoder that reads it is
    /// reading the guest's stale ring.
    pub const NEVER_WRITTEN: u16 = 1 << 15;
}

impl NewIcbBody {
    /// Whether `flags` has `bit` set. `bit` is one of the [`flag`] constants.
    #[inline]
    pub fn has_flag(&self, bit: u16) -> bool {
        self.flags.get() & bit != 0
    }

    /// The bits this crate has not identified, as they read on the wire.
    ///
    /// Exposed so a caller can notice a value that differs from the `0x7840`
    /// every fixture carries — which would mean one of them is a real field
    /// this capture never made move. [`flag::NEVER_WRITTEN`] is masked off
    /// because it is noise rather than data.
    #[inline]
    pub fn unidentified_flags(&self) -> u16 {
        self.flags.get() & flag::UNIDENTIFIED
    }
}

/// View the payload of an indirect-command-buffer creation.
pub fn new_indirect_command_buffer<'a>(op: &Op<'a>) -> Result<&'a NewIcbBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_NEW_INDIRECT_COMMAND_BUFFER);
    view::<NewIcbBody>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OP_HEADER_LEN;

    #[test]
    fn the_body_and_the_header_account_for_the_whole_record() {
        assert_eq!(
            size_of::<NewIcbBody>() + OP_HEADER_LEN,
            NEW_INDIRECT_COMMAND_BUFFER_TOTAL_LEN as usize
        );
        assert_eq!(size_of::<IcbLayout>(), 52);
    }

    #[test]
    fn every_named_flag_bit_is_distinct_and_none_is_unidentified() {
        let named = [
            flag::INHERIT_PIPELINE_STATE,
            flag::INHERIT_BUFFERS,
            flag::SUPPORT_RAY_TRACING,
            flag::SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE,
            flag::INHERIT_DEPTH_STENCIL_STATE,
            flag::INHERIT_DEPTH_BIAS,
            flag::INHERIT_DEPTH_CLIP_MODE,
            flag::INHERIT_CULL_MODE,
            flag::INHERIT_FRONT_FACING_WINDING,
            flag::INHERIT_TRIANGLE_FILL_MODE,
        ];
        let mut union = 0u16;
        for bit in named {
            assert_eq!(bit.count_ones(), 1, "{bit:#06x} is not a single bit");
            assert_eq!(union & bit, 0, "{bit:#06x} is claimed twice");
            assert_eq!(
                bit & flag::UNIDENTIFIED,
                0,
                "{bit:#06x} is named and unidentified at once"
            );
            union |= bit;
        }
        // Every bit is either named, unidentified, or the one the serializer
        // never writes. A future build that adds a property lands in the gap
        // this asserts does not exist.
        assert_eq!(union | flag::UNIDENTIFIED | flag::NEVER_WRITTEN, u16::MAX);
    }

    #[test]
    fn a_payload_one_byte_short_is_refused_rather_than_read() {
        let bytes = [0u8; 79];
        assert!(view::<NewIcbBody>(&bytes).is_err());
    }
}
