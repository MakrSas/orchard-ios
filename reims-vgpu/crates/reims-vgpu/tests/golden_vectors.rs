//! Golden-vector and cross-module smoke tests for protocol packages.
//!
//! The vectors were originally extracted from the C decoder unit-test matrices
//! under `host/utils`. That tree is deleted, so there is nothing left to run a
//! differential C↔Rust comparison against: these expectations are now the
//! source of truth for the values they cover, not a copy of one.

use reims_vgpu::protocol::endian::{st32, st64};
use reims_vgpu::protocol::pixel_format;

/// Never share the live product logs with a concurrent boot.
fn isolate_logs() {
    reims_vgpu::observe::redirect_logs_for_tests();
}

#[test]
fn pixel_format_c_matrix_rows() {
    isolate_logs();
    // IOSurface row expectations, from the deleted C pixel-format matrix, read
    // through the mapper rail's own row-bytes rule rather than a second copy of
    // it that only this vector reached.
    use reims_vgpu::protocol::iosurface_pages::packed_span_estimate;
    assert_eq!(
        packed_span_estimate(pixel_format::MTL_FORMAT_BGRA8_UNORM, 200, 1),
        Some(896)
    );
    assert_eq!(
        packed_span_estimate(pixel_format::MTL_FORMAT_RGBA16_FLOAT, 200, 1),
        Some(1664)
    );
}

/// The same vector, through the framing that now reads it.
///
/// This device's own segment/record framer is gone;
/// `reims_vgpu_protocol::segment::SegmentStream` locates segments and
/// `reims_vgpu_wire::op::OpStream` locates the records inside one. The bytes are
/// unchanged, so the vector still says what it always said.
#[test]
fn stream_segment_record_roundtrip_shape() {
    use reims_vgpu::protocol::segment::{
        SegmentBody, SegmentKind, SegmentStream, SEGMENT_HEADER_LEN,
    };
    isolate_logs();
    let mut payload = Vec::new();
    // record: opcode 0x12d, length 0x28
    let mut rec = vec![0u8; 0x28];
    st32(&mut rec[0..], 0x12d);
    st32(&mut rec[4..], 0x28);
    payload.extend_from_slice(&rec);

    let mut stream_bytes = Vec::new();
    let seg_len = (SEGMENT_HEADER_LEN + payload.len()) as u32;
    let mut hdr = [0u8; 8];
    st32(&mut hdr[0..], seg_len);
    hdr[4] = SegmentKind::Blit.wire_type();
    stream_bytes.extend_from_slice(&hdr);
    stream_bytes.extend_from_slice(&payload);

    let segs: Vec<_> = SegmentStream::new(&stream_bytes)
        .expect("the vector is addressable")
        .collect::<Result<_, _>>()
        .expect("a well-formed stream frames");
    assert_eq!(segs.len(), 1);
    let SegmentBody::Encoder { kind, commands } = segs[0].body else {
        panic!("a blit segment carries records");
    };
    assert_eq!(kind, SegmentKind::Blit);
    let op = reims_vgpu::protocol::decode::op(commands, 0).expect("the record frames");
    assert_eq!(op.opcode(), 0x12d);
    // 0x12d is `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`,
    // and the lift that reads it in production is the protocol crate's. The
    // vector's bytes are unchanged, so it still says what it always said — it
    // now says it about the decoder the rail actually calls.
    let record = reims_vgpu::protocol::decode::blit::decode(&op).expect("the vector lifts");
    assert!(matches!(
        record,
        reims_vgpu::protocol::decode::blit::BlitRecord::BufferToBuffer(_)
    ));
}

/// The same golden vector, through the lift that now reads it.
///
/// This device's own event decoder is gone; `reims_vgpu_protocol::decode::sync`
/// lifts the event rail for every encoder at once. The bytes are unchanged, so
/// the vector still says what it always said — a signal at ref 3 with value
/// `0x42` — and it now says it about the decoder in production.
#[test]
fn event_signal_golden() {
    use reims_vgpu::protocol::closure::Rail;
    use reims_vgpu::protocol::decode::sync::{decode, SyncRecord};
    isolate_logs();
    let mut v = vec![0u8; 0x14];
    st32(&mut v[0..], 0x191);
    st32(&mut v[4..], 0x14);
    st32(&mut v[8..], 3);
    st64(&mut v[12..], 0x42);
    let op = reims_vgpu::protocol::decode::op(&v, 0).expect("the vector frames");
    let SyncRecord::Event(record) = decode(Rail::Event, &op).expect("the vector lifts") else {
        panic!("0x191 is the event signal and lifts as one");
    };
    assert_eq!(record.event_ref, 3);
    assert_eq!(record.value, 0x42);
}

#[test]
fn corpus_property_random_decode_no_panic() {
    isolate_logs();
    // Smoke fuzz: random-ish buffers through all byte parsers.
    let seeds: &[&[u8]] = &[
        &[],
        &[0],
        &[0xff; 7],
        &[0x00; 64],
        &[0x12, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00],
    ];
    for s in seeds {
        if let Ok(segments) = reims_vgpu::protocol::segment::SegmentStream::new(s) {
            for framed in segments {
                let _ = framed;
            }
        }
        if let Ok(op) = reims_vgpu::protocol::decode::op(s, 0) {
            let _ = reims_vgpu::protocol::decode::blit::decode(&op);
            let _ = reims_vgpu::runtime::decode::blit_spi::decode(s);
            let _ = reims_vgpu::protocol::decode::sync::decode(
                reims_vgpu::protocol::closure::Rail::Event,
                &op,
            );
        }
        let _ = reims_vgpu::runtime::decode::compute_spi::decode(s);
        let _ = reims_vgpu::runtime::decode::render_spi::decode(s);
        if let Ok(op) = reims_vgpu::protocol::decode::op(s, 0) {
            let _ = reims_vgpu::protocol::decode::render::decode(&op);
        }
        let _ = reims_vgpu::runtime::decode::resource::decode_list_object_entry(s);
    }
}
