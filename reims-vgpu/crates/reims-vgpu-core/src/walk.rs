//! From a guest's command-stream bytes to one transaction.
//!
//! # The link that was missing
//!
//! [`crate::exec::ExecBuilder`] takes `begin_segment`, `record`, `end_segment`
//! and `finish`, in wire order, and produces the only [`ExecWork`] this
//! crate can make. Every one of those calls had to come from somewhere, and
//! until this module the only somewhere was a test calling them by hand. So the
//! model could resolve a record and could place a record, and had no way to be
//! handed a stream.
//!
//! [`exec`] is that way. It is the whole path — bytes, segments, records,
//! operations, accesses, transaction — and it is one function because every
//! seam inside it is a place where two halves could disagree about where a
//! record was, which is exactly the class of defect the replacement exists to
//! remove.
//!
//! # It owns none of the four parses it drives
//!
//! Segments come from [`reims_vgpu_protocol::segment::SegmentStream`]. Records
//! come from `reims_vgpu_wire::op::OpStream`, through the protocol crate's
//! re-export. Meaning comes from [`crate::resolve::operation`]. Placement,
//! ordering and access derivation come from the builder. This module contains
//! no byte arithmetic at all; it is the composition, and the reason it is worth
//! writing down separately is that the composition is where a rail can be taken
//! from the wrong place.
//!
//! The rail is the segment's. `resolve::operation` is handed a rail rather than
//! deriving one, and the only defensible source for it is the encoder class the
//! guest wrote the record into —
//! [`reims_vgpu_protocol::segment::SegmentKind::rail`], from the type byte in
//! the header immediately above the record. Taking it from anywhere else, such
//! as a previous segment or a per-packet default, reads one encoder's commands
//! as another's.
//!
//! # A record that does not resolve refuses the transaction
//!
//! Not the record — the transaction. Dropping one record and executing the rest
//! is a wrong frame presented as a right one: a draw without its pipeline, a
//! blit without its barrier. The closure ledger is what makes this a schedule
//! rather than a wall, and it already says so —
//! [`reims_vgpu_protocol::closure::Closure::blocks_cutover`] holds for every
//! unresolved row, so a stream refusing here while rows remain open is the
//! ledger's prediction rather than a surprise.
//!
//! # An encoder is not a segment
//!
//! A guest may split one encoder across several segments, and the contract for
//! it is established rather than guessed: the `beginSegment:` `BOOL` lands at
//! `+5` of the header it opens, and the serializer then reaches *back* into the
//! preceding header to mark `+6`. The two are one edge recorded from both ends
//! — "this segment continues the encoder above" and "that encoder continues
//! below" — which is why one non-zero byte cannot be read for the direction,
//! and why [`reims_vgpu_protocol::segment::SegmentLifetime`] is one value
//! rather than two bools a seam could split.
//!
//! This module does nothing with it beyond passing it whole.
//! [`crate::stream::StreamCursor`] holds the encoder, so it is the cursor that
//! decides whether a segment's end ends anything, and the walker's loop is the
//! same either way. That is the point of the split: a walker that knew when an
//! encoder survives would be a second copy of the encoder state machine.

use crate::exec::{ExecBuilder, ExecWork};
use crate::resolve::{self, RefResolver, ResolveRefusal};
use crate::stream::{ProtectionOptions, StreamRefusal};
use reims_vgpu_protocol::decode::OpStream;
use reims_vgpu_protocol::segment::{FramedSegment, FramingRefusal, SegmentBody, SegmentStream};

/// Where in a stream something was refused.
///
/// Both coordinates, because neither alone finds the record: the segment index
/// is what a report names an encoder by, and the byte offset is what finds the
/// bytes in a capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamSite {
    /// Which segment, counting from zero.
    pub segment: u32,
    /// Byte offset within the whole stream.
    pub offset: u32,
}

/// Why a command stream did not become a transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkRefusal {
    /// The stream did not divide into segments.
    Framing(FramingRefusal),
    /// A segment's window did not divide into records.
    ///
    /// The wire error is not carried: it names a byte length against a buffer
    /// length, and the site names which buffer. Both halves of a report exist,
    /// and neither is a `WireError` this crate would then have to give a
    /// reason string to on wire's behalf.
    RecordFraming { at: StreamSite },
    /// A record did not become an operation.
    ///
    /// Includes the ledger's own answers — an opcode with no row, an open row,
    /// a row settled as a refusal. Those are contract answers rather than
    /// defects, and the whole transaction still refuses; see the module
    /// documentation.
    Resolve {
        at: StreamSite,
        refusal: ResolveRefusal,
    },
    /// The builder refused the operation's placement, ordering or access.
    ///
    /// `record` is the refused record's own opcode when a record is what was
    /// being placed, and `None` for the three placements that are a *segment*'s
    /// — a protection envelope, an encoder beginning, a segment ending — which
    /// have a site and no record. **It is the field a window refusal cannot be
    /// diagnosed without**: `StreamRefusal::Access`'s reason says a
    /// participation did not fit its resource's extent, and whether that is a
    /// record naming bytes past the end or two coordinate spaces meeting turns
    /// on which record it was. A driven macos-15 boot refuses seven exec
    /// packets a boot this way, all at one site, and the site alone names no
    /// record.
    Place {
        at: StreamSite,
        record: Option<u32>,
        refusal: StreamRefusal,
    },
    /// The stream ended with an encoder or a protection envelope unfinished.
    Unfinished { refusal: StreamRefusal },
}

impl WalkRefusal {
    /// The stable reason string for the failure channel.
    ///
    /// Each arm that wraps an owner's refusal reports the owner's own reason
    /// rather than a second name for it, so a log line says which check
    /// refused and not merely that the walk did.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Framing(inner) => inner.reason(),
            Self::RecordFraming { .. } => "walk_record_framing_refused",
            Self::Resolve { refusal, .. } => refusal.reason(),
            Self::Place { refusal, .. } | Self::Unfinished { refusal } => refusal.reason(),
        }
    }

    /// Where it happened, for the arms that have a place.
    ///
    /// [`Self::Framing`] carries the framing layer's own offset and
    /// [`Self::Unfinished`] is about the stream's end rather than a point
    /// inside it.
    #[must_use]
    pub const fn site(self) -> Option<StreamSite> {
        match self {
            Self::RecordFraming { at } | Self::Resolve { at, .. } | Self::Place { at, .. } => {
                Some(at)
            }
            Self::Framing(_) | Self::Unfinished { .. } => None,
        }
    }
}

impl From<FramingRefusal> for WalkRefusal {
    fn from(inner: FramingRefusal) -> Self {
        Self::Framing(inner)
    }
}

/// Walk one command buffer into a transaction that is still being built.
///
/// **One EXEC packet carries a *table* of command buffers, not one.** Its
/// header declares `cmdbuf_count` descriptors of `{gva, length}`, and all of
/// them belong to the single submission the packet is — one ordering position,
/// one completion word, one access list. So the walk of a buffer and the
/// finishing of a transaction are two steps, and this is the first: it takes
/// the builder by `&mut` so a caller with several buffers walks each into the
/// same one.
///
/// [`exec`] is the whole of the second step plus this, kept for the caller that
/// genuinely has one buffer, and it is where the builder is consumed.
///
/// # Errors
///
/// Any [`WalkRefusal`]. The transaction is all-or-nothing across *every* buffer
/// — see the module documentation for why a single unresolvable record refuses
/// the whole of it, and note that the reason does not weaken when the record is
/// in the second buffer of five: they are one submission, and a partial one is
/// a frame drawn from state the guest did not ask for.
pub fn command_buffer(
    bytes: &[u8],
    resolver: &impl RefResolver,
    source: &mut (impl crate::access::AccessSource + ?Sized),
    builder: &mut ExecBuilder,
) -> Result<(), WalkRefusal> {
    for framed in SegmentStream::new(bytes)? {
        let framed = framed?;
        segment(&framed, resolver, source, builder)?;
    }
    Ok(())
}

/// Walk one EXEC's single command stream into the transaction it describes.
///
/// The builder is consumed: what comes out is either the finished transaction
/// or a refusal, and never a half-written builder a caller could submit
/// anyway. A caller whose packet declares more than one command buffer uses
/// [`command_buffer`] per buffer and finishes once — see that function for why
/// the two steps are separate.
///
/// # Errors
///
/// Any [`WalkRefusal`]. The transaction is all-or-nothing — see the module
/// documentation for why a single unresolvable record refuses the whole of it.
pub fn exec(
    bytes: &[u8],
    resolver: &impl RefResolver,
    source: &mut (impl crate::access::AccessSource + ?Sized),
    mut builder: ExecBuilder,
) -> Result<ExecWork, WalkRefusal> {
    command_buffer(bytes, resolver, source, &mut builder)?;
    builder
        .finish()
        .map_err(|refusal| WalkRefusal::Unfinished { refusal })
}

/// One segment's worth of the walk.
fn segment(
    framed: &FramedSegment<'_>,
    resolver: &impl RefResolver,
    source: &mut (impl crate::access::AccessSource + ?Sized),
    builder: &mut ExecBuilder,
) -> Result<(), WalkRefusal> {
    let at = StreamSite {
        segment: framed.index,
        offset: framed.offset,
    };
    let (kind, commands) = match framed.body {
        SegmentBody::ProtectionEnvelope { options } => {
            // The envelope arms the segment after it, which is the cursor's
            // rule and not restated here.
            return builder
                .protection_envelope(ProtectionOptions(options))
                .map_err(|refusal| WalkRefusal::Place {
                    at,
                    record: None,
                    refusal,
                });
        }
        SegmentBody::Encoder { kind, commands } => (kind, commands),
    };
    builder
        .begin_encoder(kind, framed.lifetime)
        .map_err(|refusal| WalkRefusal::Place {
            at,
            record: None,
            refusal,
        })?;
    let mut records = OpStream::new(commands);
    loop {
        // Taken before the step, so it is the offset the record starts at
        // whether or not the record turns out to be readable. A refused one is
        // not a value and cannot be asked where it began.
        let started = records.consumed();
        let Some(record) = records.next() else { break };
        // The window's offsets are inside the segment; a report has to name the
        // stream. The cast is exact: the framing layer established that the
        // stream's length fits a `u32`, and the window is inside it.
        let at = StreamSite {
            segment: framed.index,
            offset: framed.commands_offset + started as u32,
        };
        let Ok(view) = record else {
            return Err(WalkRefusal::RecordFraming { at });
        };
        // Taken before the view is consumed, and only so a refusal can name it.
        let record = view.opcode();
        let resolved = resolve::operation(kind.rail(), &view, resolver, builder.arenas_mut())
            .map_err(|refusal| WalkRefusal::Resolve { at, refusal })?;
        builder
            .record(resolved, source)
            .map_err(|refusal| WalkRefusal::Place {
                at,
                record: Some(record),
                refusal,
            })?;
    }
    builder
        .end_segment()
        .map(|_| ())
        .map_err(|refusal| WalkRefusal::Place {
            at,
            record: None,
            refusal,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::StubRegistry;
    use crate::exec::ExecArenas;
    use crate::identity::ChannelId;
    use crate::testing::{
        bind_vertex_buffers, draw_primitives, generate_mipmaps, record, segment_bytes,
        segment_bytes_with, Everything,
    };
    use reims_vgpu_protocol::segment::{
        SegmentKind, SegmentLifetime, SEGMENT_TYPE_PROTECTION_OPTIONS,
    };
    use reims_vgpu_wire::ops::render::OPCODE_SET_LINE_WIDTH;
    use reims_vgpu_wire::ops::segment::SEGMENT_HEADER_LEN;

    const DOMAIN: ChannelId = ChannelId(3);

    /// The encoder outlives this segment.
    const HOLDS: SegmentLifetime = SegmentLifetime {
        continues_previous: false,
        continues_into_next: true,
    };

    /// This segment continues the encoder above and ends it.
    const TAKES: SegmentLifetime = SegmentLifetime {
        continues_previous: true,
        continues_into_next: false,
    };

    /// This segment continues the encoder above and passes it on.
    const RELAYS: SegmentLifetime = SegmentLifetime {
        continues_previous: true,
        continues_into_next: true,
    };

    fn builder() -> ExecBuilder<'static> {
        ExecBuilder::new()
    }

    fn line_width(width: f32) -> Vec<u8> {
        record(OPCODE_SET_LINE_WIDTH, &width.to_le_bytes())
    }

    /// A compute-encoder scope barrier: a four-byte payload of which the
    /// serializer writes the first two.
    fn barrier() -> Vec<u8> {
        record(
            reims_vgpu_wire::ops::compute::OPCODE_MEMORY_BARRIER_SCOPE,
            &[1, 0, 0xaa, 0xaa],
        )
    }

    /// Two command buffers of one packet walk into one transaction, in order.
    ///
    /// **The wire fact the one-buffer signature could not express.** An EXEC
    /// packet's header declares a table of `{gva, length}` descriptors and every
    /// buffer in it belongs to the single submission the packet is — one
    /// ordering position, one completion word, one access list. Walked into two
    /// transactions they would be two positions the guest never asked for, and
    /// the second's records would be ordered against the first's rather than
    /// after them.
    ///
    /// Order is asserted and not just membership: the records of the first
    /// buffer precede the records of the second, because that is what "one
    /// submission" means for a stream the guest wrote in an order.
    #[test]
    fn two_command_buffers_of_one_packet_walk_into_one_transaction_in_order() {
        let first = segment_bytes(
            SegmentKind::Render.wire_type(),
            &[bind_vertex_buffers(0, &[5151]), draw_primitives()],
        );
        let second = segment_bytes(
            SegmentKind::Render.wire_type(),
            &[bind_vertex_buffers(0, &[6262]), draw_primitives()],
        );

        let mut b = builder();
        let mut source = StubRegistry(DOMAIN);
        command_buffer(&first, &Everything, &mut source, &mut b).expect("first buffer");
        command_buffer(&second, &Everything, &mut source, &mut b).expect("second buffer");
        let tx = b.finish().expect("both buffers ended their encoders");

        let named: Vec<u64> = tx
            .accesses
            .iter()
            .filter_map(|a| match a.key {
                crate::access::AccessKey::Range(r, _)
                | crate::access::AccessKey::Subresource(r, _)
                | crate::access::AccessKey::Whole(r) => Some(r.backing.0),
                crate::access::AccessKey::DomainOnly | crate::access::AccessKey::Heap(_) => None,
            })
            .collect();
        assert!(
            named.contains(&5151) && named.contains(&6262),
            "one access list carries both buffers' memory: {named:?}"
        );
        assert_eq!(
            named.iter().position(|b| *b == 5151),
            Some(0),
            "and the first buffer's is first: {named:?}"
        );

        // The same two buffers walked separately are two transactions, which is
        // what this exists to stop being the only option.
        let alone = exec(&first, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("a well-framed stream");
        assert!(
            alone.accesses.len() < tx.accesses.len(),
            "one buffer alone is a smaller transaction than the two together"
        );
    }

    /// A draw declares the buffers the binds before it named — through decode,
    /// resolution and the encoder, from bytes.
    ///
    /// `drawPrimitives:` names no memory of its own, so every access this
    /// transaction carries came out of the encoder's binding tables. The whole
    /// path had a gap here: the bind records resolved, the draw resolved, and
    /// nothing joined the two.
    #[test]
    fn a_draw_declares_the_buffers_the_binds_before_it_named() {
        let bytes = segment_bytes(
            SegmentKind::Render.wire_type(),
            &[
                bind_vertex_buffers(0, &[5151, 6262]),
                bind_vertex_buffers(7, &[7373]),
                draw_primitives(),
            ],
        );
        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("a well-framed stream");

        let mut named: Vec<u64> = tx
            .accesses
            .iter()
            .filter_map(|a| match a.key {
                crate::access::AccessKey::Range(r, _)
                | crate::access::AccessKey::Subresource(r, _)
                | crate::access::AccessKey::Whole(r) => Some(r.backing.0),
                crate::access::AccessKey::Heap(_) | crate::access::AccessKey::DomainOnly => None,
            })
            .collect();
        named.sort_unstable();
        assert_eq!(named, vec![5151, 6262, 7373]);
    }

    /// The same stream without the draw declares nothing: a bind writes a slot
    /// and touches no memory.
    #[test]
    fn binds_with_no_draw_after_them_declare_nothing() {
        let bytes = segment_bytes(
            SegmentKind::Render.wire_type(),
            &[bind_vertex_buffers(0, &[5151, 6262])],
        );
        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("a well-framed stream");
        assert!(tx.accesses.is_empty());
    }

    /// The whole path, from bytes to a transaction whose accesses came from the
    /// records that named them.
    #[test]
    fn a_stream_becomes_the_transaction_its_segments_describe() {
        let mut bytes = segment_bytes(
            SegmentKind::Render.wire_type(),
            &[line_width(2.5), line_width(1.25)],
        );
        bytes.extend_from_slice(&segment_bytes(
            SegmentKind::Blit.wire_type(),
            &[generate_mipmaps(4242)],
        ));

        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("a well-framed stream");

        assert_eq!(tx.streams.len(), 2);
        assert_eq!(tx.record_count(), 3);
        let positions: Vec<_> = tx.records().map(|r| r.at).collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(
            positions.iter().map(|p| p.segment).collect::<Vec<_>>(),
            [0, 0, 1]
        );
        // The mipmap generation is the only record here that names a resource,
        // and the access on the transaction is the one it named.
        assert_eq!(tx.accesses.len(), 1);
        assert_eq!(
            tx.accesses[0].key,
            crate::access::AccessKey::Whole(crate::access::ResourceKey {
                backing: crate::access::BackingId(4242),
                heap: None,
            })
        );
        assert_eq!(tx.accesses[0].domain, DOMAIN);
    }

    /// The rail a record is read on is the encoder the guest wrote it into.
    ///
    /// The same four opcode bytes are a blit record and nothing at all on the
    /// render rail. A walk that carried a rail from anywhere but the segment
    /// header immediately above the record — a previous segment, a per-packet
    /// default — would read one encoder's commands as another's, and the only
    /// evidence it had done so would be the frame.
    #[test]
    fn the_rail_a_record_is_read_on_is_the_encoder_it_was_written_into() {
        let mipmaps = generate_mipmaps(4242);
        let inside_blit = segment_bytes(
            SegmentKind::Blit.wire_type(),
            std::slice::from_ref(&mipmaps),
        );
        let inside_render = segment_bytes(SegmentKind::Render.wire_type(), &[mipmaps]);

        assert_eq!(
            exec(
                &inside_blit,
                &Everything,
                &mut StubRegistry(DOMAIN),
                builder()
            )
            .expect("a blit record in a blit segment")
            .record_count(),
            1
        );

        let refused = exec(
            &inside_render,
            &Everything,
            &mut StubRegistry(DOMAIN),
            builder(),
        )
        .expect_err("a blit record is not a render record");
        assert!(matches!(refused, WalkRefusal::Resolve { .. }));
        assert_eq!(
            refused.site(),
            Some(StreamSite {
                segment: 0,
                offset: SEGMENT_HEADER_LEN as u32,
            })
        );
    }

    /// The envelope's value reaches the segment it arms, so a report can say
    /// the guest asked for a protection domain this device does not provide.
    #[test]
    fn a_protection_envelope_reaches_the_segment_it_arms() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&segment_bytes(SEGMENT_TYPE_PROTECTION_OPTIONS, &[]));
        // The envelope's window is its payload, not a record.
        let at = bytes.len() - SEGMENT_HEADER_LEN;
        bytes[at..at + 4].copy_from_slice(&((SEGMENT_HEADER_LEN + 8) as u32).to_le_bytes());
        bytes.extend_from_slice(&0x44u64.to_le_bytes());
        bytes.extend_from_slice(&segment_bytes(
            SegmentKind::Blit.wire_type(),
            &[generate_mipmaps(7)],
        ));

        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("an envelope and the segment it arms");
        assert_eq!(tx.streams.len(), 1);
        assert_eq!(
            tx.streams[0].begin.protection,
            Some(ProtectionOptions(0x44))
        );
        assert!(tx.streams[0].begin.demands_protection());
    }

    /// An envelope with nothing after it armed nothing, and a dropped
    /// protection request is loss the stream's end has to report.
    #[test]
    fn an_envelope_at_the_end_of_a_stream_refuses_the_transaction() {
        let mut bytes = segment_bytes(SEGMENT_TYPE_PROTECTION_OPTIONS, &[]);
        bytes[..4].copy_from_slice(&((SEGMENT_HEADER_LEN + 8) as u32).to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());

        assert_eq!(
            exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder()),
            Err(WalkRefusal::Unfinished {
                refusal: StreamRefusal::ProtectionEnvelopeUnclaimed,
            })
        );
    }

    /// One record the model cannot represent refuses the whole transaction.
    ///
    /// Executing the rest would be a draw without its pipeline or a blit
    /// without its barrier, presented as a finished frame.
    #[test]
    fn one_unrepresentable_record_refuses_the_whole_transaction() {
        let good = line_width(2.5);
        let bad = record(0xffff_ff00, &[]);
        let bytes = segment_bytes(
            SegmentKind::Render.wire_type(),
            &[good.clone(), bad, line_width(1.0)],
        );

        let refused = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect_err("an opcode the render rail carries no record for");
        assert_eq!(
            refused.site(),
            Some(StreamSite {
                segment: 0,
                offset: (SEGMENT_HEADER_LEN + good.len()) as u32,
            })
        );
        assert_eq!(refused.reason(), "decode_opcode_unknown");
    }

    /// A record whose framing does not fit its segment stops the walk where it
    /// started, not where the length pointed.
    #[test]
    fn a_record_that_overruns_its_segment_names_where_it_began() {
        let good = line_width(2.5);
        let mut bytes = segment_bytes(SegmentKind::Render.wire_type(), std::slice::from_ref(&good));
        // Extend the segment by four bytes that cannot be a record header.
        let length = u32::from_le_bytes(bytes[..4].try_into().expect("four bytes"));
        bytes[..4].copy_from_slice(&(length + 4).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);

        assert_eq!(
            exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder()),
            Err(WalkRefusal::RecordFraming {
                at: StreamSite {
                    segment: 0,
                    offset: (SEGMENT_HEADER_LEN + good.len()) as u32,
                },
            })
        );
    }

    /// A framing the stream layer refuses never reaches the model.
    #[test]
    fn a_stream_that_does_not_frame_refuses_before_any_record() {
        let bytes = [0u8; 4];
        assert_eq!(
            exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder()),
            Err(WalkRefusal::Framing(FramingRefusal::ShortHeader {
                at: 0,
                remaining: 4,
            }))
        );
    }

    /// An encoder the guest splits across two segments is one encoder.
    ///
    /// The records of both segments land in one `ResolvedStream`, with one
    /// opening and one protection state, and the segment indices still say
    /// where each record was written. Opening a fresh encoder for the second
    /// segment instead would attribute its records to a pass the guest never
    /// opened.
    #[test]
    fn an_encoder_split_across_segments_is_one_encoder() {
        let mut bytes = segment_bytes_with(
            SegmentKind::Blit.wire_type(),
            HOLDS,
            &[generate_mipmaps(1), generate_mipmaps(2)],
        );
        bytes.extend_from_slice(&segment_bytes_with(
            SegmentKind::Blit.wire_type(),
            TAKES,
            std::slice::from_ref(&generate_mipmaps(3)),
        ));

        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("both ends of the continuation are declared");
        assert_eq!(tx.streams.len(), 1, "one encoder, not two");
        assert_eq!(tx.record_count(), 3);
        // Where a record was written and which encoder ran it are different
        // questions, and the positions still answer the first.
        let positions: Vec<_> = tx.records().map(|r| r.at).collect();
        assert_eq!(
            positions.iter().map(|p| p.segment).collect::<Vec<_>>(),
            [0, 0, 1]
        );
        assert!(positions.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(tx.accesses.len(), 3);
    }

    /// An encoder may be relayed through as many segments as the guest writes.
    #[test]
    fn an_encoder_may_be_relayed_through_several_segments() {
        let mut bytes = segment_bytes_with(
            SegmentKind::Compute.wire_type(),
            HOLDS,
            std::slice::from_ref(&barrier()),
        );
        bytes.extend_from_slice(&segment_bytes_with(
            SegmentKind::Compute.wire_type(),
            RELAYS,
            std::slice::from_ref(&barrier()),
        ));
        bytes.extend_from_slice(&segment_bytes_with(
            SegmentKind::Compute.wire_type(),
            TAKES,
            std::slice::from_ref(&barrier()),
        ));

        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("a relayed encoder");
        assert_eq!(tx.streams.len(), 1);
        assert_eq!(tx.record_count(), 3);
        assert_eq!(
            tx.records().map(|r| r.at.segment).collect::<Vec<_>>(),
            [0, 1, 2]
        );
    }

    /// Half an edge is a refusal, in either direction.
    ///
    /// The contract records a continuation from both headers. A stream where
    /// only one end declares it has two headers disagreeing about whether an
    /// encoder is still alive, and picking either reading attributes records to
    /// a pass on a guess.
    #[test]
    fn half_a_continuation_edge_is_refused_from_either_side() {
        let record = generate_mipmaps(1);
        let blit = SegmentKind::Blit.wire_type();

        // Offered and not taken: the second segment opens on top of an encoder
        // that is still alive.
        let mut offered = segment_bytes_with(blit, HOLDS, std::slice::from_ref(&record));
        offered.extend_from_slice(&segment_bytes_with(
            blit,
            SegmentLifetime::SELF_CONTAINED,
            std::slice::from_ref(&record),
        ));
        assert_eq!(
            exec(&offered, &Everything, &mut StubRegistry(DOMAIN), builder())
                .expect_err("an encoder was abandoned")
                .reason(),
            "stream_encoder_begin_while_open"
        );

        // Taken and never offered.
        let mut taken = segment_bytes_with(
            blit,
            SegmentLifetime::SELF_CONTAINED,
            std::slice::from_ref(&record),
        );
        taken.extend_from_slice(&segment_bytes_with(
            blit,
            TAKES,
            std::slice::from_ref(&record),
        ));
        assert_eq!(
            exec(&taken, &Everything, &mut StubRegistry(DOMAIN), builder())
                .expect_err("nothing offered a continuation")
                .reason(),
            "stream_continuation_without_encoder"
        );

        // Offered by one family and claimed by another.
        let mut crossed = segment_bytes_with(blit, HOLDS, std::slice::from_ref(&record));
        crossed.extend_from_slice(&segment_bytes_with(
            SegmentKind::Compute.wire_type(),
            TAKES,
            std::slice::from_ref(&barrier()),
        ));
        assert_eq!(
            exec(&crossed, &Everything, &mut StubRegistry(DOMAIN), builder())
                .expect_err("a compute segment cannot continue a blit encoder")
                .reason(),
            "stream_continuation_kind_mismatch"
        );
    }

    /// An encoder held open at the end of a stream is records with no
    /// `-endEncoding` behind them.
    #[test]
    fn an_encoder_held_open_at_the_end_of_a_stream_is_refused() {
        let bytes = segment_bytes_with(
            SegmentKind::Blit.wire_type(),
            HOLDS,
            std::slice::from_ref(&generate_mipmaps(1)),
        );
        assert_eq!(
            exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder()),
            Err(WalkRefusal::Unfinished {
                refusal: StreamRefusal::EncoderNeverEnded(SegmentKind::Blit),
            })
        );
    }

    /// An empty stream is an empty transaction, not a refusal. A guest may
    /// submit a command buffer that encoded nothing.
    #[test]
    fn an_empty_stream_is_an_empty_transaction() {
        let tx = exec(&[], &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("a stream with no segments");
        assert_eq!(tx.record_count(), 0);
        assert!(tx.streams.is_empty());
        assert!(tx.accesses.is_empty());
    }

    /// The arenas a resolver fills are the builder's, so a record that files a
    /// variable-length entry names a window the finished transaction can read
    /// back.
    #[test]
    fn resolution_files_into_the_transactions_own_arenas() {
        // A default set is what a fresh builder starts from; the walk must not
        // hand a resolver anything else, or a window filed during resolution
        // would name an arena nobody keeps.
        assert_eq!(ExecArenas::default().resources.len(), 0);
        let bytes = segment_bytes(
            SegmentKind::Blit.wire_type(),
            &[generate_mipmaps(9), generate_mipmaps(10)],
        );
        let tx = exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("two single-ref records");
        assert_eq!(tx.record_count(), 2);
        assert_eq!(tx.accesses.len(), 2);
    }

    /// A buffer the guest corrupted is refused by name, never read past, and
    /// never panics.
    ///
    /// # Why a sweep and not more cases
    ///
    /// Every length, count and offset in a command stream is the guest's, and
    /// this walk is the first thing that touches them. The named cases above
    /// each cover one arrangement someone thought of; what they cannot cover is
    /// the arithmetic between them — an offset added to a window base, a count
    /// multiplied by an entry size, a subtraction that is only non-negative
    /// because the length was checked two layers up. Those fail as a panic in a
    /// debug build and as a wrong read in a release one, and the second is a
    /// guest reading another guest's memory.
    ///
    /// So this takes a well-formed stream and breaks it: bit flips, corrupted
    /// lengths, truncations, and buffers that were never a stream at all. The
    /// claim is not that any particular one refuses — several of these mutations
    /// produce perfectly valid streams — but that **every outcome is one of the
    /// two the signature admits**, with a reason string when it is a refusal.
    ///
    /// The seeds are [`crate::schedule::Rng`]'s, so seed *n* is the same buffer
    /// on every machine and a failure is a bug report rather than a rumour.
    #[test]
    fn a_corrupted_stream_is_refused_by_name_and_never_read_past() {
        use crate::schedule::Rng;

        // A stream with every shape the walk has an arm for: two encoders, a
        // protection envelope, records with and without refs.
        let well_formed = || {
            let mut bytes = segment_bytes(
                SegmentKind::Render.wire_type(),
                &[line_width(2.5), line_width(1.0)],
            );
            let mut envelope = segment_bytes(SEGMENT_TYPE_PROTECTION_OPTIONS, &[]);
            let at = envelope.len() - SEGMENT_HEADER_LEN;
            envelope[at..at + 4].copy_from_slice(&((SEGMENT_HEADER_LEN + 8) as u32).to_le_bytes());
            envelope.extend_from_slice(&0x44u64.to_le_bytes());
            bytes.extend_from_slice(&envelope);
            bytes.extend_from_slice(&segment_bytes(
                SegmentKind::Blit.wire_type(),
                &[generate_mipmaps(7), generate_mipmaps(9)],
            ));
            bytes
        };
        // The base stream is itself walkable, or the sweep is corrupting
        // something that was already broken.
        let base = well_formed();
        exec(&base, &Everything, &mut StubRegistry(DOMAIN), builder())
            .expect("the stream the sweep corrupts is one the walk accepts");

        let mut walked = 0usize;
        let mut refused = 0usize;
        for seed in 0..2048u64 {
            let mut rng = Rng::new(seed);
            let mut bytes = match seed % 4 {
                // A buffer that was never a stream.
                0 => (0..rng.below(96) + 1)
                    .map(|_| u8::try_from(rng.next() % 256).expect("masked"))
                    .collect(),
                // A truncation, which is what a short DMA looks like.
                1 => {
                    let mut b = well_formed();
                    b.truncate(rng.below(b.len()));
                    b
                }
                // Bit flips, one to four of them.
                _ => {
                    let mut b = well_formed();
                    for _ in 0..=rng.below(4) {
                        let at = rng.below(b.len());
                        b[at] ^= 1u8 << (rng.next() % 8);
                    }
                    b
                }
            };
            // And a trailing byte often enough to exercise the tail arms.
            if rng.next().is_multiple_of(5) {
                bytes.push(u8::try_from(rng.next() % 256).expect("masked"));
            }

            match exec(&bytes, &Everything, &mut StubRegistry(DOMAIN), builder()) {
                Ok(tx) => {
                    // Whatever came out is a transaction the rest of the model
                    // may rely on: its records are in a strictly ascending
                    // order and its stream list agrees with its record count.
                    let positions: Vec<_> = tx.records().map(|r| r.at).collect();
                    assert!(
                        positions.windows(2).all(|w| w[0] < w[1]),
                        "seed {seed}: an accepted stream is out of order"
                    );
                    assert_eq!(
                        positions.len(),
                        tx.record_count(),
                        "seed {seed}: the record count disagrees with the records"
                    );
                    walked += 1;
                }
                Err(refusal) => {
                    assert!(
                        !refusal.reason().is_empty(),
                        "seed {seed}: a refusal with no name"
                    );
                    refused += 1;
                }
            }
        }
        // Both outcomes have to happen, or the sweep is measuring one arm. A
        // corruption that never refuses means the walk accepts anything; one
        // that always refuses means the mutations are too coarse to be about
        // this walk at all.
        assert!(
            walked > 0 && refused > 0,
            "walked {walked}, refused {refused}"
        );
        println!("corrupted streams: {walked} walked, {refused} refused");
    }

    /// Every refusal reason is distinct where this module owns it, and is the
    /// owner's own where it does not.
    #[test]
    fn walk_refusal_reasons_name_the_check_that_refused() {
        let framing = FramingRefusal::UnknownType {
            at: 0,
            wire_type: 9,
        };
        assert_eq!(WalkRefusal::Framing(framing).reason(), framing.reason());
        let inner = StreamRefusal::RecordOutsideEncoder;
        assert_eq!(
            WalkRefusal::Unfinished { refusal: inner }.reason(),
            inner.reason()
        );
        let site = StreamSite {
            segment: 0,
            offset: 0,
        };
        assert_eq!(
            WalkRefusal::RecordFraming { at: site }.reason(),
            "walk_record_framing_refused"
        );
    }
}
