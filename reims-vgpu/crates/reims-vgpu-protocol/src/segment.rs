//! What a segment-type byte means.
//!
//! # Why the meaning lives here and the bytes live in wire
//!
//! `reims_vgpu_wire::ops::segment` owns the header layout and the four type
//! values its fixtures measured, plus the protection-options envelope. It
//! cannot say that type `1` "is a compute encoder whose records are read on the
//! compute rail", because that is a meaning and wire does not assign meanings.
//! This module does, and it takes every value it can from wire rather than
//! restating a number, so a fixture that moved one would break the parse rather
//! than leaving two disagreeing copies.
//!
//! # The one value wire does not name
//!
//! Wire has driven the render, compute, blit and info encoders and measured
//! `0`, `1`, `2` and `4`. It deliberately does not name `3`: no fixture wrote
//! it. Two independent facts establish it anyway. The deserializer constructs
//! record decoders for the contiguous set `0..=3` and rejects new
//! non-continuation types at `4` and above, so `3` is a constructed decoder
//! rather than a hole; and the remaining encoder class in that set is the event
//! encoder, which [`crate::closure`] now carries as [`Rail::Event`] with its own
//! records. So `3` is named here, at the layer that assigns meaning, with the
//! derivation attached — and not in wire, which would have to pretend a fixture
//! wrote it.
//!
//! # Unknown is a refusal, not a variant
//!
//! [`segment_role`] returns `None` for every other byte. A segment whose family
//! is unknown has an unknown record framing, so walking it reads guest data as
//! commands; and a catch-all variant would hand it ordering and a completion
//! obligation the device cannot honour. The caller refuses.
//!
//! # Where the segments are, and why that is the same owner
//!
//! [`SegmentStream`] divides a command stream into its segments. That is byte
//! arithmetic over a layout wire owns, but the *result* of it is not: the walk
//! has to know that a type-5 window is a protection value rather than records,
//! and that a type it does not recognise cannot be skipped on its length. Both
//! are meanings, so the walk belongs beside the map that assigns them and
//! [`SegmentBody`] can be the two answers with their windows already read —
//! rather than a raw type byte every caller re-interprets.
//!
//! Records inside a segment are **not** re-framed here.
//! [`reims_vgpu_wire::op::OpStream`] already walks back-to-back records with
//! typed errors, and a second copy of a record header parse is exactly the
//! duplication this layering exists to prevent.

use crate::closure::Rail;
use reims_vgpu_wire::ops::segment::{
    protection_options_envelope, segment_header, PROTECTION_OPTIONS_ENVELOPE_LEN,
    SEGMENT_TYPE_BLIT, SEGMENT_TYPE_COMPUTE, SEGMENT_TYPE_INFO, SEGMENT_TYPE_RENDER,
};

/// Bytes a segment header allocates.
///
/// Wire's, re-exported for the same reason the type bytes are: a caller that
/// has to build or bound a segment reaches the framing layer for it, and a
/// second literal is a number that can disagree with the framer. The
/// allocation, not the struct — the header's eighth byte belongs to no field.
pub use reims_vgpu_wire::ops::segment::SEGMENT_HEADER_LEN;

/// The segment type that introduces a protection-options envelope.
///
/// Wire's, re-exported: an envelope has no [`SegmentKind`], so it is the one
/// role whose byte a caller cannot reach through [`SegmentKind::wire_type`].
pub use reims_vgpu_wire::ops::segment::SEGMENT_TYPE_PROTECTION_OPTIONS;

/// The event encoder's segment type.
///
/// Derived rather than measured; see the module documentation.
pub const SEGMENT_TYPE_EVENT: u8 = 3;

/// A record-bearing segment: one encoder's worth of commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SegmentKind {
    Render,
    Compute,
    Blit,
    Event,
    Info,
}

impl SegmentKind {
    pub const ALL: &'static [SegmentKind] = &[
        SegmentKind::Render,
        SegmentKind::Compute,
        SegmentKind::Blit,
        SegmentKind::Event,
        SegmentKind::Info,
    ];

    /// The byte this family writes into the segment header.
    #[must_use]
    pub const fn wire_type(self) -> u8 {
        match self {
            Self::Render => SEGMENT_TYPE_RENDER,
            Self::Compute => SEGMENT_TYPE_COMPUTE,
            Self::Blit => SEGMENT_TYPE_BLIT,
            Self::Event => SEGMENT_TYPE_EVENT,
            Self::Info => SEGMENT_TYPE_INFO,
        }
    }

    /// The rail whose records may appear inside this segment.
    ///
    /// One-to-one, and that is the point: a rail is how an opcode is read, a
    /// segment kind is where the record was found, and a model that trusted
    /// either alone would read one encoder's commands as another's.
    #[must_use]
    pub const fn rail(self) -> Rail {
        match self {
            Self::Render => Rail::Render,
            Self::Compute => Rail::Compute,
            Self::Blit => Rail::Blit,
            Self::Event => Rail::Event,
            Self::Info => Rail::Info,
        }
    }

    /// The segment kind whose records are read on `rail`, if any.
    ///
    /// [`Rail::Root`] has none: its records arrive as object-list payloads and
    /// never inside a command stream.
    #[must_use]
    pub const fn of_rail(rail: Rail) -> Option<SegmentKind> {
        Some(match rail {
            Rail::Render => Self::Render,
            Rail::Compute => Self::Compute,
            Rail::Blit => Self::Blit,
            Rail::Event => Self::Event,
            Rail::Info => Self::Info,
            Rail::Root => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Render => "render",
            Self::Compute => "compute",
            Self::Blit => "blit",
            Self::Event => "event",
            Self::Info => "info",
        }
    }
}

/// What a segment header introduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SegmentRole {
    /// An encoder's records follow.
    Encoder(SegmentKind),
    /// A protection-options envelope, which arms the *next* encoder segment and
    /// carries no records of its own.
    ProtectionEnvelope,
}

/// Parse a segment-type byte, or refuse it.
#[must_use]
pub const fn segment_role(wire_type: u8) -> Option<SegmentRole> {
    Some(match wire_type {
        SEGMENT_TYPE_RENDER => SegmentRole::Encoder(SegmentKind::Render),
        SEGMENT_TYPE_COMPUTE => SegmentRole::Encoder(SegmentKind::Compute),
        SEGMENT_TYPE_BLIT => SegmentRole::Encoder(SegmentKind::Blit),
        SEGMENT_TYPE_EVENT => SegmentRole::Encoder(SegmentKind::Event),
        SEGMENT_TYPE_INFO => SegmentRole::Encoder(SegmentKind::Info),
        SEGMENT_TYPE_PROTECTION_OPTIONS => SegmentRole::ProtectionEnvelope,
        _ => return None,
    })
}

/// Why a byte range is not a command stream.
///
/// Every variant names an arrangement whose records cannot be located, so
/// walking past it would read guest data as commands. None is repaired: a
/// stream that stops making sense stops here, and the caller reports which
/// check stopped it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramingRefusal {
    /// The buffer is longer than a segment offset inside it can name.
    ///
    /// A segment's length is a `u32`, so a stream that cannot be addressed by
    /// one cannot be walked by one either.
    StreamTooLong { length: usize },
    /// Fewer bytes remain than a segment header occupies.
    ShortHeader { at: u32, remaining: u32 },
    /// A segment claims a length that cannot hold its own header.
    LengthBelowHeader { at: u32, length: u32 },
    /// A segment claims a length past the end of the stream.
    LengthPastStreamEnd {
        at: u32,
        length: u32,
        remaining: u32,
    },
    /// A segment type with no established contract.
    ///
    /// Its record framing is unknown, so its length is the only thing about it
    /// that can be read — and skipping it on that length would hand the next
    /// segment an encoder state derived from bytes nothing here understands.
    UnknownType { at: u32, wire_type: u8 },
    /// A protection envelope whose window is not the eight bytes its payload
    /// occupies.
    EnvelopeWindowNotItsPayload { at: u32, length: u32 },
}

impl FramingRefusal {
    /// The stable reason string for the failure channel.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::StreamTooLong { .. } => "framing_stream_too_long",
            Self::ShortHeader { .. } => "framing_segment_short_header",
            Self::LengthBelowHeader { .. } => "framing_segment_length_below_header",
            Self::LengthPastStreamEnd { .. } => "framing_segment_length_past_stream_end",
            Self::UnknownType { .. } => "framing_segment_type_unknown",
            Self::EnvelopeWindowNotItsPayload { .. } => "framing_envelope_window_not_payload",
        }
    }
}

/// Every framing refusal is one check, and the fields are the numbers it saw.
///
/// The slug is [`FramingRefusal::reason`] rather than a second set of strings,
/// for the reason `DecodeRefusal`'s is: a layer that may not depend on
/// `observe` still has to name the refusal it forwards, and two spellings can
/// drift.
///
/// `at` is on every variant that has one because a stream that stopped framing
/// stopped *somewhere*, and the offset is what turns "this stream is malformed"
/// into a place a reader can look.
impl reims_vgpu_observe::Decline for FramingRefusal {
    fn slug(&self) -> &'static str {
        self.reason()
    }

    fn fields(&self) -> alloc::vec::Vec<(&'static str, alloc::string::String)> {
        use alloc::string::ToString;
        match *self {
            Self::StreamTooLong { length } => alloc::vec![("length", length.to_string())],
            Self::ShortHeader { at, remaining } => {
                alloc::vec![("at", at.to_string()), ("remaining", remaining.to_string())]
            }
            Self::LengthBelowHeader { at, length } => {
                alloc::vec![("at", at.to_string()), ("length", length.to_string())]
            }
            Self::LengthPastStreamEnd {
                at,
                length,
                remaining,
            } => alloc::vec![
                ("at", at.to_string()),
                ("length", length.to_string()),
                ("remaining", remaining.to_string())
            ],
            Self::UnknownType { at, wire_type } => alloc::vec![
                ("at", at.to_string()),
                ("wire_type", alloc::format!("{wire_type:#x}"))
            ],
            Self::EnvelopeWindowNotItsPayload { at, length } => {
                alloc::vec![("at", at.to_string()), ("length", length.to_string())]
            }
        }
    }
}

/// What a framed segment carries after its header.
///
/// The two arms are [`SegmentRole`]'s two answers with their windows already
/// read, so a caller cannot ask an envelope for records or an encoder for a
/// protection value. The role byte is parsed once, here, and the window that
/// goes with it is established in the same step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentBody<'a> {
    /// One encoder's records, as the bytes they occupy.
    ///
    /// Iterating them is [`reims_vgpu_wire::op::OpStream`]'s job: a record's
    /// header is wire's layout and there is no second copy of it here.
    Encoder {
        kind: SegmentKind,
        commands: &'a [u8],
    },
    /// The `protectionOptions:` argument, verbatim.
    ///
    /// Read rather than skipped. This device provides no protection domain, so
    /// nothing acts on the value — but "we do not act on it" and "we cannot
    /// read it" are different claims, and only a value that has been read lets
    /// a report make the first one.
    ProtectionEnvelope { options: u64 },
}

/// Whether a segment's encoder reaches past the segment, in either direction.
///
/// One value rather than two bools passed side by side, because a continuation
/// *is* the pair: the serializer writes the `beginSegment:` `BOOL` at `+5` of
/// the header it opens and then reaches back to mark `+6` of the **preceding**
/// header, so the edge is recorded from both ends and one non-zero byte cannot
/// be read for its direction. A seam that took the two halves separately could
/// carry one and drop the other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentLifetime {
    /// Whether this segment's records continue the encoder the previous
    /// segment left open.
    ///
    /// The `BOOL` first argument of `-beginSegment:protectionOptions:`, at `+5`
    /// of the header that call opens. Read as set or clear rather than kept as
    /// a byte: the wire carries a `BOOL`, and a guest that writes `2` means by
    /// it what one that writes `1` means.
    pub continues_previous: bool,
    /// Whether the encoder outlives this segment, so the next one may continue
    /// it.
    ///
    /// The other half of the same edge. The serializer does not write this into
    /// the header it is opening; it marks `+6` of the preceding one.
    pub continues_into_next: bool,
}

impl SegmentLifetime {
    /// Neither end of a continuation is declared: one segment, one encoder.
    pub const SELF_CONTAINED: Self = Self {
        continues_previous: false,
        continues_into_next: false,
    };
}

/// One segment of a command stream, located and parsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramedSegment<'a> {
    /// Position in the stream, counting from zero.
    pub index: u32,
    /// Byte offset of the header within the stream.
    pub offset: u32,
    /// Byte offset of [`SegmentBody::Encoder::commands`] within the stream, so
    /// a per-record offset taken inside the window can be made absolute.
    pub commands_offset: u32,
    /// Whether this segment's encoder reaches past it, in either direction.
    pub lifetime: SegmentLifetime,
    pub body: SegmentBody<'a>,
}

/// Walk a command stream into its segments.
///
/// Yields a typed refusal rather than ending quietly when the stream stops
/// making sense, then stops — a truncated stream and a malformed one must not
/// look alike, and neither may look like a clean end.
///
/// # The index is counted, not searched
///
/// Each segment's position is the count of the ones before it, which this
/// iterator already knows. Deriving it instead by re-walking the stream to find
/// the offset makes the walk quadratic in the number of segments, and a
/// measured boot of this guest puts ninety-five thousand of them in a stream.
#[derive(Clone, Debug)]
pub struct SegmentStream<'a> {
    bytes: &'a [u8],
    cursor: usize,
    index: u32,
    stopped: bool,
}

impl<'a> SegmentStream<'a> {
    /// # Errors
    ///
    /// [`FramingRefusal::StreamTooLong`] if an offset in `bytes` cannot be
    /// named by the `u32` a segment header measures lengths in.
    pub fn new(bytes: &'a [u8]) -> Result<Self, FramingRefusal> {
        if u32::try_from(bytes.len()).is_err() {
            return Err(FramingRefusal::StreamTooLong {
                length: bytes.len(),
            });
        }
        Ok(Self {
            bytes,
            cursor: 0,
            index: 0,
            stopped: false,
        })
    }

    /// Bytes consumed so far. After the iterator ends without a refusal this
    /// equals the buffer length.
    #[must_use]
    pub const fn consumed(&self) -> usize {
        self.cursor
    }

    /// How many segments have been yielded.
    #[must_use]
    pub const fn segments(&self) -> u32 {
        self.index
    }

    fn step(&mut self) -> Result<FramedSegment<'a>, FramingRefusal> {
        // Checked in `new`, and the cursor only ever advances within the
        // buffer, so both casts below are exact.
        let at = self.cursor as u32;
        let remaining = self.bytes.len() - self.cursor;
        // Against the *allocation*, not the struct. The header's eighth byte
        // belongs to no field, so wire's view is satisfied by seven — and a
        // seven-byte tail is a truncated header, not a segment whose length
        // happens to be wrong.
        if remaining < SEGMENT_HEADER_LEN {
            return Err(FramingRefusal::ShortHeader {
                at,
                remaining: remaining as u32,
            });
        }
        let header = segment_header(&self.bytes[self.cursor..])
            .expect("the allocation is present, and the fields are shorter than it");
        let length = header.length.get();
        let length_usize = length as usize;
        if length_usize < SEGMENT_HEADER_LEN {
            return Err(FramingRefusal::LengthBelowHeader { at, length });
        }
        if length_usize > remaining {
            return Err(FramingRefusal::LengthPastStreamEnd {
                at,
                length,
                remaining: remaining as u32,
            });
        }
        let Some(role) = segment_role(header.segment_type) else {
            return Err(FramingRefusal::UnknownType {
                at,
                wire_type: header.segment_type,
            });
        };
        let lifetime = SegmentLifetime {
            continues_previous: header.begin_flag != 0,
            continues_into_next: header.unidentified_u8 != 0,
        };
        let commands_offset = self.cursor + SEGMENT_HEADER_LEN;
        let window = &self.bytes[commands_offset..self.cursor + length_usize];
        let body = match role {
            SegmentRole::Encoder(kind) => SegmentBody::Encoder {
                kind,
                commands: window,
            },
            SegmentRole::ProtectionEnvelope => {
                // The envelope's window *is* its payload: one segment header,
                // then eight bytes of `protectionOptions:`. A window of any
                // other size is not the burst this role names, and reading the
                // first eight bytes of it anyway would take a protection value
                // out of whatever else the guest put there.
                if window.len() != PROTECTION_OPTIONS_ENVELOPE_LEN {
                    return Err(FramingRefusal::EnvelopeWindowNotItsPayload { at, length });
                }
                let envelope = protection_options_envelope(window)
                    .expect("the window is exactly the payload's length");
                SegmentBody::ProtectionEnvelope {
                    options: envelope.protection_options.get(),
                }
            }
        };
        let out = FramedSegment {
            index: self.index,
            offset: at,
            commands_offset: commands_offset as u32,
            lifetime,
            body,
        };
        self.cursor += length_usize;
        self.index += 1;
        Ok(out)
    }
}

impl<'a> Iterator for SegmentStream<'a> {
    type Item = Result<FramedSegment<'a>, FramingRefusal>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stopped || self.cursor >= self.bytes.len() {
            return None;
        }
        match self.step() {
            Ok(segment) => Some(Ok(segment)),
            Err(refusal) => {
                self.stopped = true;
                Some(Err(refusal))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind round-trips, and nothing outside the six established values
    /// parses at all.
    #[test]
    fn the_byte_map_is_a_bijection_over_the_established_values() {
        for &kind in SegmentKind::ALL {
            assert_eq!(
                segment_role(kind.wire_type()),
                Some(SegmentRole::Encoder(kind))
            );
        }
        assert_eq!(
            segment_role(SEGMENT_TYPE_PROTECTION_OPTIONS),
            Some(SegmentRole::ProtectionEnvelope)
        );
        for byte in 0u8..=255 {
            let established = SegmentKind::ALL.iter().any(|k| k.wire_type() == byte)
                || byte == SEGMENT_TYPE_PROTECTION_OPTIONS;
            assert_eq!(segment_role(byte).is_some(), established, "{byte:#x}");
        }
    }

    /// The values are wire's, not this module's — with `3` the stated
    /// exception. A test rather than a comment, because the failure mode is two
    /// copies drifting silently.
    #[test]
    fn the_measured_values_come_from_wire() {
        assert_eq!(SegmentKind::Render.wire_type(), SEGMENT_TYPE_RENDER);
        assert_eq!(SegmentKind::Compute.wire_type(), SEGMENT_TYPE_COMPUTE);
        assert_eq!(SegmentKind::Blit.wire_type(), SEGMENT_TYPE_BLIT);
        assert_eq!(SegmentKind::Info.wire_type(), SEGMENT_TYPE_INFO);
        assert_eq!(SegmentKind::Event.wire_type(), 3);
    }

    /// `Info` is `4`, not the next integer after blit. A device that had
    /// guessed the sequence would read every info segment as an event one.
    #[test]
    fn info_is_four_and_the_gap_belongs_to_events() {
        assert_eq!(SEGMENT_TYPE_INFO, 4);
        assert_eq!(
            segment_role(3),
            Some(SegmentRole::Encoder(SegmentKind::Event))
        );
    }

    /// Rails and segments correspond one-to-one, except the root rail, which
    /// has no segment because its records never enter a command stream.
    #[test]
    fn rails_and_segments_correspond_except_at_the_root() {
        for &kind in SegmentKind::ALL {
            assert_eq!(SegmentKind::of_rail(kind.rail()), Some(kind));
            assert_ne!(kind.rail(), Rail::Root);
        }
        assert_eq!(SegmentKind::of_rail(Rail::Root), None);
    }

    /// A segment header, as the serializer leaves it after `-endEncoding`.
    ///
    /// Synthesized from this crate's own constants: it proves the walk is
    /// self-consistent and deliberately cannot prove the offsets are Apple's.
    /// `reims_vgpu_wire::ops::segment` pins those against the capture, and
    /// `reims-vgpu-core`'s fixture suite walks a stream built out of it.
    fn header(wire_type: u8, length: u32, previous: bool, next: bool) -> [u8; SEGMENT_HEADER_LEN] {
        let mut out = [0u8; SEGMENT_HEADER_LEN];
        out[..4].copy_from_slice(&length.to_le_bytes());
        out[4] = wire_type;
        out[5] = u8::from(previous);
        out[6] = u8::from(next);
        // The eighth byte is the one the serializer never writes. Filled with
        // something other than zero here, because a walk that read it as a
        // field would then answer differently.
        out[7] = 0xaa;
        out
    }

    fn record(opcode: u32, length: u32) -> alloc::vec::Vec<u8> {
        let mut out = alloc::vec::Vec::new();
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&length.to_le_bytes());
        out.resize(length as usize, 0);
        out
    }

    /// Segments come out in wire order, with the window each encoder's records
    /// occupy and nothing of the header around them.
    #[test]
    fn a_stream_divides_into_the_segments_its_lengths_describe() {
        let first = record(0x01, 16);
        let second = record(0x75, 40);
        let mut bytes = alloc::vec::Vec::new();
        bytes.extend_from_slice(&header(
            SEGMENT_TYPE_RENDER,
            (SEGMENT_HEADER_LEN + first.len() + second.len()) as u32,
            false,
            false,
        ));
        bytes.extend_from_slice(&first);
        bytes.extend_from_slice(&second);
        let compute = record(0x02, 24);
        bytes.extend_from_slice(&header(
            SEGMENT_TYPE_COMPUTE,
            (SEGMENT_HEADER_LEN + compute.len()) as u32,
            false,
            false,
        ));
        bytes.extend_from_slice(&compute);

        let mut walk = SegmentStream::new(&bytes).expect("a short stream");
        let a = walk.next().expect("a first segment").expect("well framed");
        assert_eq!(a.index, 0);
        assert_eq!(a.offset, 0);
        assert_eq!(a.commands_offset, SEGMENT_HEADER_LEN as u32);
        let SegmentBody::Encoder { kind, commands } = a.body else {
            panic!("a render segment carries records");
        };
        assert_eq!(kind, SegmentKind::Render);
        assert_eq!(commands.len(), first.len() + second.len());

        let b = walk.next().expect("a second segment").expect("well framed");
        assert_eq!(b.index, 1);
        assert_eq!(
            b.offset,
            (SEGMENT_HEADER_LEN + first.len() + second.len()) as u32
        );
        let SegmentBody::Encoder { kind, commands } = b.body else {
            panic!("a compute segment carries records");
        };
        assert_eq!(kind, SegmentKind::Compute);
        assert_eq!(commands.len(), compute.len());

        assert!(walk.next().is_none());
        assert_eq!(walk.consumed(), bytes.len());
        assert_eq!(walk.segments(), 2);
    }

    /// The records of a segment are wire's to frame, and the offsets they
    /// report are inside the window rather than inside the stream.
    #[test]
    fn a_segments_window_is_exactly_the_records_written_into_it() {
        let first = record(0x01, 16);
        let second = record(0x75, 40);
        let mut bytes = alloc::vec::Vec::new();
        bytes.extend_from_slice(&header(
            SEGMENT_TYPE_BLIT,
            (SEGMENT_HEADER_LEN + first.len() + second.len()) as u32,
            false,
            false,
        ));
        bytes.extend_from_slice(&first);
        bytes.extend_from_slice(&second);

        let segment = SegmentStream::new(&bytes)
            .expect("a short stream")
            .next()
            .expect("a segment")
            .expect("well framed");
        let SegmentBody::Encoder { commands, .. } = segment.body else {
            panic!("a blit segment carries records");
        };
        let seen: alloc::vec::Vec<(u32, usize)> = reims_vgpu_wire::op::OpStream::new(commands)
            .map(|r| {
                let r = r.expect("well framed");
                (r.opcode(), r.offset + segment.commands_offset as usize)
            })
            .collect();
        assert_eq!(seen, [(0x01, 8), (0x75, 24)]);
    }

    /// The envelope's window is its payload, and the value read out of it is
    /// the guest's.
    #[test]
    fn a_protection_envelope_yields_the_value_rather_than_being_skipped() {
        let mut bytes = alloc::vec::Vec::new();
        bytes.extend_from_slice(&header(
            SEGMENT_TYPE_PROTECTION_OPTIONS,
            (SEGMENT_HEADER_LEN + PROTECTION_OPTIONS_ENVELOPE_LEN) as u32,
            false,
            false,
        ));
        bytes.extend_from_slice(&0x44u64.to_le_bytes());
        bytes.extend_from_slice(&header(
            SEGMENT_TYPE_BLIT,
            SEGMENT_HEADER_LEN as u32,
            false,
            false,
        ));

        let seen: alloc::vec::Vec<SegmentBody> = SegmentStream::new(&bytes)
            .expect("a short stream")
            .map(|s| s.expect("well framed").body)
            .collect();
        assert_eq!(
            seen,
            [
                SegmentBody::ProtectionEnvelope { options: 0x44 },
                SegmentBody::Encoder {
                    kind: SegmentKind::Blit,
                    commands: &[],
                },
            ]
        );
    }

    /// An envelope window of any other size is not the burst the role names.
    /// Reading its first eight bytes anyway would take a protection value out
    /// of whatever else the guest put there.
    #[test]
    fn an_envelope_window_that_is_not_the_payload_is_refused() {
        for length in [
            SEGMENT_HEADER_LEN as u32,
            (SEGMENT_HEADER_LEN + PROTECTION_OPTIONS_ENVELOPE_LEN + 1) as u32,
        ] {
            let mut bytes = alloc::vec::Vec::new();
            bytes.extend_from_slice(&header(
                SEGMENT_TYPE_PROTECTION_OPTIONS,
                length,
                false,
                false,
            ));
            bytes.resize(length as usize, 0);
            assert_eq!(
                SegmentStream::new(&bytes)
                    .expect("a short stream")
                    .next()
                    .expect("a segment"),
                Err(FramingRefusal::EnvelopeWindowNotItsPayload { at: 0, length })
            );
        }
    }

    /// Both encoder-lifetime bits are located, and neither is confused with the
    /// other or with the type beside them.
    #[test]
    fn the_two_encoder_lifetime_bits_read_independently() {
        for previous in [false, true] {
            for next in [false, true] {
                let bytes = header(SEGMENT_TYPE_BLIT, SEGMENT_HEADER_LEN as u32, previous, next);
                let segment = SegmentStream::new(&bytes)
                    .expect("a short stream")
                    .next()
                    .expect("a segment")
                    .expect("well framed");
                assert_eq!(
                    segment.lifetime,
                    SegmentLifetime {
                        continues_previous: previous,
                        continues_into_next: next,
                    }
                );
                assert_eq!(
                    segment.body,
                    SegmentBody::Encoder {
                        kind: SegmentKind::Blit,
                        commands: &[],
                    }
                );
            }
        }
    }

    /// A guest may put any non-zero byte in a `BOOL`, and `2` means set.
    #[test]
    fn any_non_zero_lifetime_byte_is_set() {
        let mut bytes = header(SEGMENT_TYPE_BLIT, SEGMENT_HEADER_LEN as u32, false, false);
        bytes[5] = 2;
        bytes[6] = 0x80;
        let segment = SegmentStream::new(&bytes)
            .expect("a short stream")
            .next()
            .expect("a segment")
            .expect("well framed");
        assert!(segment.lifetime.continues_previous);
        assert!(segment.lifetime.continues_into_next);
    }

    /// Every malformed framing is a named refusal, and the walk stops there
    /// rather than resynchronising on bytes it has already shown it cannot
    /// read.
    #[test]
    fn malformed_framings_are_named_refusals_and_the_walk_stops() {
        let short = [0u8; SEGMENT_HEADER_LEN - 1];
        assert_eq!(
            SegmentStream::new(&short)
                .expect("a short stream")
                .next()
                .expect("a segment"),
            Err(FramingRefusal::ShortHeader {
                at: 0,
                remaining: SEGMENT_HEADER_LEN as u32 - 1,
            })
        );

        let below = header(
            SEGMENT_TYPE_BLIT,
            SEGMENT_HEADER_LEN as u32 - 1,
            false,
            false,
        );
        assert_eq!(
            SegmentStream::new(&below)
                .expect("a short stream")
                .next()
                .expect("a segment"),
            Err(FramingRefusal::LengthBelowHeader {
                at: 0,
                length: SEGMENT_HEADER_LEN as u32 - 1,
            })
        );

        let past = header(
            SEGMENT_TYPE_BLIT,
            SEGMENT_HEADER_LEN as u32 + 1,
            false,
            false,
        );
        assert_eq!(
            SegmentStream::new(&past)
                .expect("a short stream")
                .next()
                .expect("a segment"),
            Err(FramingRefusal::LengthPastStreamEnd {
                at: 0,
                length: SEGMENT_HEADER_LEN as u32 + 1,
                remaining: SEGMENT_HEADER_LEN as u32,
            })
        );

        // A type with no contract has no record framing, so the length that
        // would skip it is a number this layer cannot vouch for.
        let mut unknown = alloc::vec::Vec::new();
        unknown.extend_from_slice(&header(9, SEGMENT_HEADER_LEN as u32, false, false));
        unknown.extend_from_slice(&header(
            SEGMENT_TYPE_BLIT,
            SEGMENT_HEADER_LEN as u32,
            false,
            false,
        ));
        let mut walk = SegmentStream::new(&unknown).expect("a short stream");
        assert_eq!(
            walk.next().expect("a segment"),
            Err(FramingRefusal::UnknownType {
                at: 0,
                wire_type: 9,
            })
        );
        assert!(
            walk.next().is_none(),
            "the walk resumed past a framing it had refused"
        );
    }

    /// Every refusal reason is distinct, so a log line identifies which check
    /// stopped the walk.
    #[test]
    fn framing_refusal_reasons_are_distinct() {
        let all = [
            FramingRefusal::StreamTooLong { length: 0 },
            FramingRefusal::ShortHeader {
                at: 0,
                remaining: 0,
            },
            FramingRefusal::LengthBelowHeader { at: 0, length: 0 },
            FramingRefusal::LengthPastStreamEnd {
                at: 0,
                length: 0,
                remaining: 0,
            },
            FramingRefusal::UnknownType {
                at: 0,
                wire_type: 0,
            },
            FramingRefusal::EnvelopeWindowNotItsPayload { at: 0, length: 0 },
        ];
        let mut seen: alloc::vec::Vec<&str> = all.iter().map(|r| r.reason()).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before);
    }
}
