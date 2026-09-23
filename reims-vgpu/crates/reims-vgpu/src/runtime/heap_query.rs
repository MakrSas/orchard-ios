//! `CmdHeapTextureSizeAndAlign` request decode and host requirement query.
//!
//! The command carries a `PGSerializedTextureDescriptor` record. The recovered
//! Apple host routine reconstructs an `MTLTextureDescriptor` and returns the
//! device's `MTLSizeAndAlign` as two little-endian `u64`s.

use crate::protocol::fifo;
use reims_vgpu_wire::ops::texture as wire;

/// The request's framing — its three words, the reply floor, and the embedded
/// record's tag — belongs to [`crate::protocol::fifo`], which derives each
/// offset from the one before it. Re-exported rather than restated so the
/// callers that bound this command keep one set of numbers.
pub use fifo::{
    SizeAndAlign, HEAP_TEXTURE_REPLY_LEN as REPLY_LEN,
    HEAP_TEXTURE_REQUEST_HEADER_LEN as REQUEST_HEADER_LEN,
    HEAP_TEXTURE_SERIALIZED_LEN as SERIALIZED_TEXTURE_LEN,
    HEAP_TEXTURE_SERIALIZED_TAG as SERIALIZED_TEXTURE_TAG,
};

pub const TEXTURE_BODY_LEN: usize = wire::TEXTURE_DESCRIPTOR_LEN;
/// The same descriptor once the guest's serializer has a texture-descriptor
/// capability on: eight bytes wider, with `usage` promoted out of the packed
/// word into a `u32` and a four-channel swizzle appended.
///
/// Two different flags select it depending on the record — `SwizzledTextures`
/// for the plain creation, `TextureDescriptor2` for the four that embed it —
/// and neither is a length the caller may infer. Every record carrying this
/// body has an opcode of its own; see
/// [`crate::runtime::decode::resource::decode_heap_texture`].
pub const WIDE_TEXTURE_BODY_LEN: usize = wire::WIDE_TEXTURE_DESCRIPTOR_LEN;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureDescriptor {
    pub texture_type: u8,
    pub framebuffer_only: bool,
    pub is_drawable: bool,
    pub allow_gpu_optimized_contents: bool,
    /// `MTLTextureUsage` mask. Eight bits of the packed word on the narrow
    /// body and a field of its own on the wide one, so it is held at the wider
    /// of the two — narrowing it here would drop any bit above 7 that a wide
    /// descriptor sets.
    pub usage: u32,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mipmap_level_count: u16,
    pub sample_count: u16,
    pub array_length: u16,
    pub resource_options: u16,
    pub protection_options: u64,
    /// `MTLTextureSwizzleChannels` as four raw `MTLTextureSwizzle` ordinals in
    /// red, green, blue, alpha order — the same encoding the texture-view swizzle
    /// view carries, so [`crate::protocol::pixel_format::swizzle_plan`] reads
    /// both.
    ///
    /// `None` on the narrow body, which has no such field at all. That is not
    /// the same as the identity: a reader must not turn an absent swizzle into
    /// a present one.
    pub swizzle: Option<[u8; 4]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub task_id: u32,
    pub reply_gva: u64,
    pub reply_len: u64,
    pub descriptor: TextureDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryError {
    ShortPayload,
    BadReplyLength,
    BadSerializerLength,
    UnknownSerializerTag,
    BadDescriptorLength,
    UnknownTextureType,
    UnknownPixelFormat,
    UnknownUsage,
    UnknownResourceOptions,
    UnsupportedProtectionOptions,
    NoMetalDevice,
    ZeroRequirement,
    /// The request names a task id that resolves to no active task, so there is
    /// nowhere to write the reply. Checked by the caller in `runtime/drain/mod.rs`
    /// rather than here — the vocabulary still owns the reason, because the
    /// alternative is one untyped `reason=bad_task` sitting inside an event
    /// family where every other reason is typed.
    BadTask,
}

impl crate::observe::Decline for QueryError {
    /// Every slug carries the `heap_query_` prefix.
    ///
    /// Not decoration: these names are generic enough (`short_payload`,
    /// `unknown_pixel_format`, `unknown_usage`) that they describe checks half
    /// the crate also makes. `unknown_pixel_format` in fact **collided** with
    /// `TranslateReason`'s, and no per-enum uniqueness test could see it — both
    /// enums were internally consistent, and a grep of the fail log for that
    /// slug would have returned a mix of two unrelated subsystems' refusals.
    /// The prefix is what makes the check nameable at crate scope.
    fn slug(&self) -> &'static str {
        match self {
            Self::ShortPayload => "heap_query_short_payload",
            Self::BadReplyLength => "heap_query_bad_reply_length",
            Self::BadSerializerLength => "heap_query_bad_serializer_length",
            Self::UnknownSerializerTag => "heap_query_unknown_serializer_tag",
            Self::BadDescriptorLength => "heap_query_bad_descriptor_length",
            Self::UnknownTextureType => "heap_query_unknown_texture_type",
            Self::UnknownPixelFormat => "heap_query_unknown_pixel_format",
            Self::UnknownUsage => "heap_query_unknown_usage",
            Self::UnknownResourceOptions => "heap_query_unknown_resource_options",
            Self::UnsupportedProtectionOptions => "heap_query_unsupported_protection_options",
            Self::NoMetalDevice => "heap_query_no_metal_device",
            Self::ZeroRequirement => "heap_query_zero_requirement",
            Self::BadTask => "heap_query_bad_task",
        }
    }
}

impl From<fifo::HeapTextureRefusal> for QueryError {
    /// The framing refusals, named in this vocabulary.
    ///
    /// The mapping is total and one-to-one; it exists only because the reasons
    /// this command reports under are `heap_query_`-prefixed at crate scope
    /// (see [`QueryError::slug`]) while the framing that produces them is the
    /// protocol's. `HeapTextureRefusal::slug` states the same four slugs, so a
    /// caller on either side of the boundary logs one name for one fact.
    fn from(refusal: fifo::HeapTextureRefusal) -> Self {
        match refusal {
            fifo::HeapTextureRefusal::Short(_) => Self::ShortPayload,
            fifo::HeapTextureRefusal::ReplyDestination { .. } => Self::BadReplyLength,
            fifo::HeapTextureRefusal::SerializerLength { .. } => Self::BadSerializerLength,
            fifo::HeapTextureRefusal::SerializerTag { .. } => Self::UnknownSerializerTag,
        }
    }
}

/// Decode a `CmdHeapTextureSizeAndAlign` request.
///
/// The framing is the protocol's; what is left here is turning the embedded
/// body into fields, which is this device's reading of the descriptor.
///
/// # Errors
///
/// [`QueryError`] for a request whose framing does not hold, or a body this
/// device cannot read.
pub fn decode_request(payload: &[u8]) -> Result<Request, QueryError> {
    let request = fifo::decode_heap_texture_query(payload)?;
    Ok(Request {
        task_id: request.raw_task,
        reply_gva: request.reply_gva,
        reply_len: request.reply_len,
        descriptor: decode_serialized_texture_descriptor(request.descriptor)?,
    })
}

/// Decode the shared 32-byte `PGSerializedTextureDescriptor` body.
///
/// The same body is embedded in heap-texture resource opcode `0x15` and in the
/// buffer-backed opcode 9; keeping one decoder prevents the query and resource
/// paths from drifting.
///
/// Read through `reims_vgpu_wire`'s view rather than at offsets restated here,
/// so a field this device names is the field Apple's bytes derived. Two of the
/// names below are this device's reading and not the wire crate's:
/// `framebuffer_only` and `is_drawable` are `packed[5:4]`, which that crate
/// carries as `unidentified_flags` because neither is an `MTLTextureDescriptor`
/// property and no perturbation has moved either. They read 0 on every fixture.
pub fn decode_serialized_texture_descriptor(body: &[u8]) -> Result<TextureDescriptor, QueryError> {
    if body.len() != TEXTURE_BODY_LEN {
        return Err(QueryError::BadDescriptorLength);
    }
    let d: &wire::TextureDescriptorBody =
        reims_vgpu_wire::view(body).map_err(|_| QueryError::BadDescriptorLength)?;
    let packed = d.packed.get();
    Ok(TextureDescriptor {
        texture_type: d.texture_type(),
        framebuffer_only: packed & (1 << 4) != 0,
        is_drawable: packed & (1 << 5) != 0,
        allow_gpu_optimized_contents: d.allow_gpu_optimized_contents(),
        usage: d.usage() as u32,
        pixel_format: d.pixel_format(),
        width: d.width.get(),
        height: d.height.get(),
        depth: d.depth.get(),
        mipmap_level_count: d.mipmap_level_count.get(),
        sample_count: d.sample_count.get(),
        array_length: d.array_length.get(),
        resource_options: d.resource_options.get(),
        // The wire crate calls this word `unidentified_u64` and has moved it in
        // no capture. `protection_options` is this device's ported reading of
        // it; `query_size_and_align` refuses a non-zero one rather than acting
        // on the name, which is the only safe thing to do with a field whose
        // meaning nothing has confirmed.
        protection_options: d.unidentified_u64.get(),
        swizzle: None,
    })
}

/// Decode the 40-byte wide `PGSerializedTextureDescriptor` body.
///
/// The wide form is not a longer narrow one: `usage` leaves the packed word for
/// a `u32` of its own and four swizzle ordinals trail, so every field after the
/// first byte sits at a different offset. Which of the two a record carries is
/// a property of its **opcode**, never of its length — see
/// [`crate::runtime::decode::resource::decode_heap_texture`].
///
/// The fortieth byte is declared and never written, so nothing reads it.
pub fn decode_wide_serialized_texture_descriptor(
    body: &[u8],
) -> Result<TextureDescriptor, QueryError> {
    if body.len() != WIDE_TEXTURE_BODY_LEN {
        return Err(QueryError::BadDescriptorLength);
    }
    let d: &wire::WideTextureDescriptorBody =
        reims_vgpu_wire::view(body).map_err(|_| QueryError::BadDescriptorLength)?;
    Ok(TextureDescriptor {
        texture_type: d.texture_type(),
        // The same two bits the narrow form carries. Bit 7 is a third flag that
        // exists only here — the narrow serializer never writes it — and it is
        // unnamed in both crates, so it is not read.
        framebuffer_only: d.type_and_flags & (1 << 4) != 0,
        is_drawable: d.type_and_flags & (1 << 5) != 0,
        allow_gpu_optimized_contents: d.allow_gpu_optimized_contents(),
        usage: d.usage.get(),
        pixel_format: d.pixel_format.get(),
        width: d.width.get(),
        height: d.height.get(),
        depth: d.depth.get(),
        mipmap_level_count: d.mipmap_level_count.get(),
        sample_count: d.sample_count.get(),
        array_length: d.array_length.get(),
        resource_options: d.resource_options.get(),
        protection_options: d.unidentified_u64.get(),
        swizzle: Some([
            d.swizzle_red,
            d.swizzle_green,
            d.swizzle_blue,
            d.swizzle_alpha,
        ]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_request_fixture() -> Vec<u8> {
        let words = [
            0x1u32, 0x162200, 0x0, 0x10, 0x0, 0x28, 0x16, 0x28, 0x7d0342, 0xb4, 0x87, 0x1, 0x10001,
            0x200001, 0x0, 0x0,
        ];
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    #[test]
    fn decodes_live_heap_texture_query() {
        let request = decode_request(&live_request_fixture()).unwrap();
        assert_eq!(request.task_id, 1);
        assert_eq!(request.reply_gva, 0x162200);
        assert_eq!(request.reply_len, 16);
        assert_eq!(
            request.descriptor,
            TextureDescriptor {
                texture_type: 2,
                framebuffer_only: false,
                is_drawable: false,
                allow_gpu_optimized_contents: true,
                usage: 3,
                pixel_format: 125,
                width: 180,
                height: 135,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 1,
                resource_options: 0x20,
                protection_options: 0,
                // The narrow body has no swizzle field at all. `None` rather
                // than the identity: see [`TextureDescriptor::swizzle`].
                swizzle: None,
            }
        );
    }

    #[test]
    fn rejects_unknown_serializer_version() {
        let mut payload = live_request_fixture();
        payload[24..28].copy_from_slice(&0x99u32.to_le_bytes());
        assert_eq!(
            decode_request(&payload),
            Err(QueryError::UnknownSerializerTag)
        );
    }

    /// The wire decode above, carried through to the host driver.
    ///
    /// Here rather than beside the driver call, because what it pins is that
    /// *this* fixture's descriptor survives decode into something Apple's
    /// driver will place — the two halves are only right together.
    #[cfg(target_os = "macos")]
    #[test]
    fn native_query_returns_nonzero_requirement() {
        let request = decode_request(&live_request_fixture()).unwrap();
        let result =
            crate::backend::heap_placement::heap_texture_size_and_align(&request.descriptor)
                .unwrap();
        assert!(result.size >= 180 * 135 * 16);
        assert!(result.align.is_power_of_two());
    }
}
