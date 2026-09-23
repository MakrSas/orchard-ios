//! Command-stream bytes, for the tests that want the model driven the way
//! production drives it.
//!
//! # Why this is a module and not a helper in whichever test needed it first
//!
//! Two suites want the same thing: [`crate::walk`] asks whether a stream
//! becomes the transaction it describes, and [`crate::schedule`] asks whether
//! transactions mean the same thing however they are scheduled. The second used
//! to state its accesses directly, which is a fine way to reach an access shape
//! the registry would not produce — and a poor way to answer "does the batch a
//! guest actually sends still schedule the same". For that it needs bytes, and
//! a second copy of the record framing in a second test module is the
//! duplication this crate's layering exists to prevent.
//!
//! Nothing here invents a layout. A record is an opcode, a length and a
//! payload, which is [`reims_vgpu_wire::op`]'s shape; a segment is a header and
//! its records, which is [`reims_vgpu_protocol::segment`]'s. What this module
//! adds is the writing side of both, which no production path needs because
//! production only ever reads them.

use reims_vgpu_protocol::segment::{SegmentKind, SegmentLifetime};
use reims_vgpu_wire::ops::segment::SEGMENT_HEADER_LEN;

/// A builder, the identity a session would have stamped on it, and the envelope
/// fields a session would have taken from the packet.
///
/// [`crate::exec::ExecBuilder`] cannot state where a packet arrived or what
/// completion word it carries — those are the envelope's, and the envelope is
/// [`crate::session::SessionModel::admit`]'s. A test that wants a finished
/// [`DeviceTransaction`] has to choose them somewhere. Choosing them here, once,
/// keeps every suite from re-deriving the pairing and keeps the choice visible
/// as a test's.
pub(crate) struct At<'u> {
    builder: crate::exec::ExecBuilder<'u>,
    identity: crate::identity::TransactionIdentity,
    stamp_waits: Vec<crate::identity::StampWait>,
    completion: Option<crate::identity::CompletionStamp>,
}

impl<'u> At<'u> {
    pub(crate) fn new(domain: u32, ingress: u64) -> Self {
        Self {
            builder: crate::exec::ExecBuilder::new(),
            identity: identity(domain, ingress),
            stamp_waits: Vec::new(),
            completion: None,
        }
    }

    /// A completion-stamp point the packet's envelope waits for.
    pub(crate) fn wait_for(&mut self, wait: crate::identity::StampWait) {
        self.stamp_waits.push(wait);
    }

    /// The completion word the packet publishes.
    pub(crate) fn publish_stamp(&mut self, stamp: crate::identity::CompletionStamp) {
        self.completion = Some(stamp);
    }

    pub(crate) fn finish(
        self,
    ) -> Result<crate::transaction::DeviceTransaction, crate::stream::StreamRefusal> {
        Ok(crate::transaction::DeviceTransaction {
            identity: self.identity,
            stamp_waits: self.stamp_waits,
            completion: self.completion,
            payload: crate::transaction::Payload::Exec(self.builder.finish()?),
        })
    }
}

impl<'u> core::ops::Deref for At<'u> {
    type Target = crate::exec::ExecBuilder<'u>;
    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl<'u> core::ops::DerefMut for At<'u> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

/// The identity a first-generation session would stamp for `domain`/`ingress`.
///
/// Channel sequence tracks the ingress ordinal, which is what a single-channel
/// arrival order produces; a suite that needs the two to differ states its own.
pub(crate) fn identity(domain: u32, ingress: u64) -> crate::identity::TransactionIdentity {
    crate::identity::TransactionIdentity {
        session: crate::identity::SessionGeneration::FIRST,
        domain: crate::identity::ChannelId(domain),
        domain_sequence: crate::identity::ChannelSequence(ingress),
        ingress: crate::identity::IngressOrdinal(ingress),
    }
}

/// A resolver that answers every ref.
///
/// Resolution is not what the suites using this are about: an unresolvable ref
/// would report a missing object where the question is about framing, ordering
/// or scheduling.
pub(crate) struct Everything;

impl crate::resolve::RefResolver for Everything {
    fn resource(&self, object_ref: u32) -> Option<crate::identity::ResourceId> {
        Some(crate::identity::ResourceId {
            slot: crate::identity::ObjectListRef(object_ref),
            generation: crate::identity::SlotGeneration(1),
        })
    }
}

/// One record, framed the way the serializer frames one.
pub(crate) fn record(opcode: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let length = (reims_vgpu_protocol::decode::OP_HEADER_LEN + payload.len()) as u32;
    out.extend_from_slice(&opcode.to_le_bytes());
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// `generateMipmapsForTexture:`, the shortest blit record that names a
/// resource.
pub(crate) fn generate_mipmaps(texture: u32) -> Vec<u8> {
    record(
        reims_vgpu_wire::ops::blit::OPCODE_GENERATE_MIPMAPS,
        &texture.to_le_bytes(),
    )
}

/// `setVertexBuffers:offsets:withRange:`, binding `refs` at consecutive slots
/// from `first`.
///
/// The head is `[first][count]` and each entry is a ref and a `u64` offset, so
/// the entry stride is twelve rather than sixteen.
pub(crate) fn bind_vertex_buffers(first: u32, refs: &[u32]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&first.to_le_bytes());
    payload.extend_from_slice(&(refs.len() as u32).to_le_bytes());
    for &buffer in refs {
        payload.extend_from_slice(&buffer.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());
    }
    record(
        reims_vgpu_wire::ops::render::OPCODE_SET_VERTEX_BUFFER,
        &payload,
    )
}

/// `drawPrimitives:vertexStart:vertexCount:`, the draw that names no memory of
/// its own — so everything it declares came from the encoder's bound slots.
pub(crate) fn draw_primitives() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&3u32.to_le_bytes());
    payload.extend_from_slice(&0u16.to_le_bytes());
    payload.extend_from_slice(&3u16.to_le_bytes());
    record(reims_vgpu_wire::ops::render::OPCODE_DRAW, &payload)
}

/// `synchronizeResource:`, the other single-ref blit record, so a stream can
/// name two resources without repeating one opcode.
pub(crate) fn synchronize_resource(resource: u32) -> Vec<u8> {
    record(
        reims_vgpu_wire::ops::blit::OPCODE_SYNCHRONIZE_RESOURCE,
        &resource.to_le_bytes(),
    )
}

/// One self-contained segment, with the length `-endEncoding` fills in.
pub(crate) fn segment_bytes(wire_type: u8, records: &[Vec<u8>]) -> Vec<u8> {
    segment_bytes_with(wire_type, SegmentLifetime::SELF_CONTAINED, records)
}

/// One segment, with both encoder-lifetime bits stated.
pub(crate) fn segment_bytes_with(
    wire_type: u8,
    lifetime: SegmentLifetime,
    records: &[Vec<u8>],
) -> Vec<u8> {
    let body: usize = records.iter().map(Vec::len).sum();
    let mut out = Vec::new();
    out.extend_from_slice(&((SEGMENT_HEADER_LEN + body) as u32).to_le_bytes());
    out.push(wire_type);
    out.push(u8::from(lifetime.continues_previous));
    out.push(u8::from(lifetime.continues_into_next));
    // The byte the serializer never writes. Filled with something other than
    // zero, because a reader that took it for a field would then answer
    // differently.
    out.push(0xaa);
    for r in records {
        out.extend_from_slice(r);
    }
    out
}

/// A one-segment blit stream over the resources `refs` names.
/// A one-segment render stream that binds `refs` and draws.
///
/// The draw is `drawPrimitives:vertexStart:vertexCount:`, which names no memory
/// of its own — so every access the transaction carries came through the
/// encoder's binding table, at [`crate::access::AccessMode::Unknown`], which is
/// what an unreflected bound slot contributes and what conflicts with
/// everything.
pub(crate) fn render_stream(refs: &[u32]) -> Vec<u8> {
    let mut records = vec![bind_vertex_buffers(0, refs)];
    records.push(draw_primitives());
    segment_bytes(SegmentKind::Render.wire_type(), &records)
}

pub(crate) fn blit_stream(refs: &[u32]) -> Vec<u8> {
    let records: Vec<Vec<u8>> = refs
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            if i % 2 == 0 {
                generate_mipmaps(r)
            } else {
                synchronize_resource(r)
            }
        })
        .collect();
    segment_bytes(SegmentKind::Blit.wire_type(), &records)
}

/// A lifecycle owner holding a dedicated resource in every slot `slots` names.
///
/// The real registry rather than a stub, because what a participation becomes
/// — which backing, which window, which content versions it consumes and
/// produces — is [`crate::lifecycle::Lifecycle`]'s answer, and a stub that
/// returned no version would hand a schedule a trace with nothing in it to
/// disagree about.
///
/// Dedicated rather than heap-placed, and generously sized: what varies between
/// the suites using this is the records, not the storage under them.
pub(crate) fn registry(
    task: crate::identity::TaskId,
    slots: &[u32],
) -> crate::lifecycle::Lifecycle {
    use crate::access::{BackingId, ByteRange};
    use crate::lifecycle::{Lifecycle, LifecycleOp, Storage};

    let mut model = Lifecycle::new();
    // The effects are the caller's obligation and there are none here: a task
    // definition owes no transfer and frees no storage.
    let _ = model
        .apply(&LifecycleOp::DefineTask {
            task,
            kernel: false,
            directory: crate::identity::DirectoryFrame(0x1000),
        })
        .expect("a fresh task");
    for &slot in slots {
        let _ = model
            .apply(&LifecycleOp::CreateResource {
                task,
                slot: crate::identity::ObjectListRef(slot),
                storage: Storage::Dedicated {
                    backing: BackingId(u64::from(slot)),
                    extent: ByteRange {
                        offset: 0,
                        length: 1 << 30,
                    },
                },
            })
            .expect("a fresh slot");
    }
    model
}
