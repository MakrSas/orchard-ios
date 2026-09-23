//! Segments, encoders, and the order a record has inside an EXEC.
//!
//! # The boundary is the first payload type, because everything else sits in it
//!
//! [`crate::operation`] names eleven classes of resolved operation. Eight of
//! them are records inside an encoder; the boundary class is the thing that
//! says which encoder those records belong to and in what order they run. It
//! has to be first: a `Render` payload with no encoder is a draw with no pass,
//! and a vocabulary that resolves draws before it resolves the encoder they sit
//! in would have to discover the encoder later, from position, which is exactly
//! the "resolve twice" shape the replacement exists to remove.
//!
//! Being first is also why it has no payload type. A boundary opens or closes
//! an encoder, and this module's state machine is where that happens — so the
//! opening lives on [`crate::exec::ResolvedStream`] and the close is a
//! [`SegmentEnd`], not a [`crate::exec::ResolvedOperation`] a caller could
//! record at a position *inside* the encoder it opens.
//!
//! # A segment type is parsed once, and not here
//!
//! [`reims_vgpu_protocol::segment`] owns that parse: it is a wire tag becoming
//! a meaning, which is the protocol layer's job, and it returns `None` for
//! every byte with no established contract. This module takes the resulting
//! [`SegmentKind`] as given and never sees the byte again — so there is one
//! place a new segment family is admitted, and it is not the state machine.
//!
//! # Ordering, and the parallelism this deliberately does not claim
//!
//! Records inside one encoder are ordered; that is contract. Whether two
//! *encoders* may run concurrently is not established for this wire, so
//! [`StreamOrder`] gives every record a total order within its EXEC and offers
//! no way to express "these two segments are independent". A later seam that
//! proves parallel encoding adds the independence claim there; until then the
//! model reduces parallelism rather than inventing it, which is the direction
//! the plan requires when a concurrency proof is missing.
//!
//! # What an open encoder means for admission
//!
//! [`StreamCursor`] is the state machine: it refuses a record with no open
//! encoder, a record whose rail disagrees with the segment it is inside, a
//! begin while another encoder is open, and an end with nothing to end. Each of
//! those is a wire shape that cannot be executed, so each is a typed refusal
//! and none is silently tolerated — the alternative is a draw attributed to
//! whichever pass happened to be open, which is a wrong frame rather than a
//! missing one.

use reims_vgpu_protocol::closure::Rail;
pub use reims_vgpu_protocol::segment::{segment_role, SegmentKind, SegmentLifetime, SegmentRole};

/// The protection-options value a guest attached to the segment that follows.
///
/// Carried, not honoured. This device provides no protection domain, so an
/// executor cannot make the guarantee the value asks for; carrying it keeps
/// "we do not act on it" separable from "we cannot read it", and
/// [`SegmentBegin::demands_protection`] is what a report asks so the difference
/// stays visible on the failure channel instead of being lost at decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProtectionOptions(pub u64);

impl ProtectionOptions {
    /// The envelope is emitted only for a non-zero value, so a zero here came
    /// from somewhere other than the guest's argument.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

/// The resolved form of an encoder opening.
///
/// The header's `continues_previous` bit is not here, and cannot be: a
/// `SegmentBegin` exists exactly when a *new* encoder opens, which is exactly
/// when that bit is clear. Carrying it would be carrying a constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentBegin {
    pub kind: SegmentKind,
    /// The value from a preceding protection envelope, if one armed this
    /// segment.
    pub protection: Option<ProtectionOptions>,
}

impl SegmentBegin {
    /// Whether the guest asked for a protection domain this device does not
    /// provide.
    #[must_use]
    pub fn demands_protection(&self) -> bool {
        self.protection.is_some_and(|p| !p.is_none())
    }
}

/// Which segment, and which record inside it.
///
/// [`Ord`] is the execution order, and it is total: the model has no way to
/// say two positions are unordered, because nothing has proved that they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamPosition {
    pub segment: u32,
    pub record: u32,
}

/// A total order over the records of one EXEC.
///
/// A separate type rather than a bare comparison so that the claim "records in
/// this EXEC are totally ordered" has one owner, and so a later seam that
/// proves encoder independence changes this type instead of every comparison
/// site.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamOrder;

impl StreamOrder {
    /// Whether `earlier` must complete its ordering obligations before `later`.
    ///
    /// Always true for distinct ascending positions. That is the conservative
    /// reading and it is deliberate.
    #[must_use]
    pub fn precedes(self, earlier: StreamPosition, later: StreamPosition) -> bool {
        earlier < later
    }

    /// Whether the two may be reordered with respect to each other.
    #[must_use]
    pub fn independent(self, _a: StreamPosition, _b: StreamPosition) -> bool {
        false
    }
}

/// Why a stream shape cannot be admitted.
///
/// Every variant names a wire arrangement that has no executable meaning, and
/// each one is reported rather than repaired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamRefusal {
    /// A segment type with no established contract.
    UnknownSegmentType(u8),
    /// A begin arrived while an encoder was still open, and did not claim to
    /// continue it.
    ///
    /// Either the previous segment declared the encoder outlives it and this
    /// one did not take the offer, or the two disagree outright. Both are half
    /// an edge, and an encoder abandoned mid-stream is records with no
    /// `-endEncoding` behind them.
    EncoderStillOpen(SegmentKind),
    /// A segment claimed to continue an encoder, and none was open.
    ContinuationWithoutEncoder(SegmentKind),
    /// A segment claimed to continue an encoder of a different family.
    ContinuationKindMismatch {
        open: SegmentKind,
        claimed: SegmentKind,
    },
    /// A segment claimed to continue an encoder the previous segment did not
    /// offer.
    ///
    /// The contract records a continuation from both ends. One end alone is a
    /// stream whose two headers disagree about whether an encoder survived,
    /// and picking either reading attributes records to a pass on a guess.
    ContinuationNotOffered(SegmentKind),
    /// A record arrived outside any encoder.
    RecordOutsideEncoder,
    /// A record decoded on one rail was found inside another rail's segment.
    RailMismatch { segment: SegmentKind, record: Rail },
    /// An end arrived with no encoder open.
    EndWithoutBegin,
    /// A protection envelope armed nothing, because another envelope or the end
    /// of the stream followed it.
    ProtectionEnvelopeUnclaimed,
    /// The stream ended with an encoder still open.
    EncoderNeverEnded(SegmentKind),
    /// A record named memory the access source could not place.
    ///
    /// The whole transaction refuses. A record whose participation cannot
    /// become an access is one the scheduler cannot order, and admitting it
    /// with the access dropped is a hazard edge that does not get built —
    /// which is a race and not a slowdown.
    Access(crate::access::AccessRefusal),
}

impl StreamRefusal {
    /// The stable reason string for the failure channel.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::UnknownSegmentType(_) => "stream_segment_type_unknown",
            Self::EncoderStillOpen(_) => "stream_encoder_begin_while_open",
            Self::ContinuationWithoutEncoder(_) => "stream_continuation_without_encoder",
            Self::ContinuationKindMismatch { .. } => "stream_continuation_kind_mismatch",
            Self::ContinuationNotOffered(_) => "stream_continuation_not_offered",
            Self::RecordOutsideEncoder => "stream_record_outside_encoder",
            Self::RailMismatch { .. } => "stream_record_rail_mismatch",
            Self::EndWithoutBegin => "stream_encoder_end_without_begin",
            Self::ProtectionEnvelopeUnclaimed => "stream_protection_envelope_unclaimed",
            Self::EncoderNeverEnded(_) => "stream_encoder_never_ended",
            // The owner's own reason, not a second name for it.
            Self::Access(refusal) => refusal.reason,
        }
    }
}

/// What opening a segment did.
///
/// Two answers rather than one, because a continuation opens no encoder: there
/// is no new [`SegmentBegin`] to make, and making one anyway would give the
/// same encoder two openings and two protection states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SegmentOpening {
    /// A new encoder opened at this position.
    Opened(StreamPosition, SegmentBegin),
    /// This segment's records join the encoder already open.
    Continued(StreamPosition),
}

/// What ending a segment did to the encoder inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SegmentEnd {
    /// The encoder ended with the segment, having recorded this many records.
    ///
    /// The count is the model's, not the wire's: the wire fills in a byte
    /// length at `-endEncoding` and says nothing about how many records it
    /// covered. A count the model derived is checkable against a re-walk;
    /// a length copied from the header is not.
    ///
    /// It counts the *encoder*, not the segment. An encoder that spanned three
    /// segments reports all of its records once, here, at the end of the third.
    EncoderEnded { records: u32 },
    /// The segment ended and its encoder did not, having recorded this many
    /// records so far. The next segment may continue it — and must, or the
    /// encoder is abandoned.
    EncoderHeld { records: u32 },
}

/// Where an encoder is in its life.
///
/// Three states rather than two, because "inside a segment" and "alive between
/// segments" admit different things and collapsing them would let a record
/// land where no segment is open, or let a continuation be claimed without a
/// segment boundary in between.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Inside a segment, taking records.
    Open {
        kind: SegmentKind,
        records: u32,
        /// Whether *this* segment declared the encoder outlives it. Per
        /// segment, not per encoder: an encoder spanning three segments
        /// re-declares it on each of the first two.
        outlives_segment: bool,
    },
    /// Between segments, with the encoder alive and waiting to be continued.
    Held {
        kind: SegmentKind,
        records: u32,
    },
    Closed,
}

/// The encoder state machine for one EXEC's command stream.
///
/// Fed in wire order; every call either returns the resolved boundary or a
/// refusal. It holds no bytes and no host state, so a test can drive it from a
/// list of events and get the same answers the ingress path would.
#[derive(Debug)]
pub struct StreamCursor {
    phase: Phase,
    pending_protection: Option<ProtectionOptions>,
    segment: u32,
    record: u32,
}

impl Default for StreamCursor {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamCursor {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            phase: Phase::Closed,
            pending_protection: None,
            segment: 0,
            record: 0,
        }
    }

    /// A protection envelope was seen. It arms the next encoder begin.
    ///
    /// Two in a row is a refusal: the second would silently replace the first,
    /// and a dropped protection request is exactly the class of loss that has
    /// to be typed.
    pub fn protection_envelope(&mut self, options: ProtectionOptions) -> Result<(), StreamRefusal> {
        if self.pending_protection.is_some() {
            return Err(StreamRefusal::ProtectionEnvelopeUnclaimed);
        }
        if let Phase::Open { kind, .. } | Phase::Held { kind, .. } = self.phase {
            return Err(StreamRefusal::EncoderStillOpen(kind));
        }
        self.pending_protection = Some(options);
        Ok(())
    }

    /// A segment header opened an encoder, named by its type byte.
    ///
    /// The parse and the state machine are separate entry points because a
    /// caller that has already been told which encoder this is — a stream
    /// walker, which cannot cut a segment's window without knowing — would
    /// otherwise have to re-encode the kind into a byte for this to parse it
    /// back. See [`Self::begin_kind`].
    pub fn begin(
        &mut self,
        wire_type: u8,
        lifetime: SegmentLifetime,
    ) -> Result<SegmentOpening, StreamRefusal> {
        let kind = match segment_role(wire_type) {
            Some(SegmentRole::Encoder(kind)) => kind,
            // The envelope has its own entry point; arriving here means the
            // caller framed it as an encoder, which it is not.
            Some(SegmentRole::ProtectionEnvelope) | None => {
                return Err(StreamRefusal::UnknownSegmentType(wire_type))
            }
        };
        self.begin_kind(kind, lifetime)
    }

    /// A segment header opened an encoder whose kind is already established.
    ///
    /// # Both ends of a continuation, or neither
    ///
    /// A continuation is one edge written from two places: this header's `+5`
    /// says it continues what came before, and the *previous* header's `+6`,
    /// which the serializer went back and marked, says the encoder survived to
    /// be continued. This checks both. A stream where only one is set has two
    /// headers disagreeing about whether an encoder is still alive, and reading
    /// either one alone attributes records to a pass on a guess.
    pub fn begin_kind(
        &mut self,
        kind: SegmentKind,
        lifetime: SegmentLifetime,
    ) -> Result<SegmentOpening, StreamRefusal> {
        let at = StreamPosition {
            segment: self.segment,
            record: self.record,
        };
        if lifetime.continues_previous {
            match self.phase {
                Phase::Held {
                    kind: open,
                    records,
                } => {
                    if open != kind {
                        return Err(StreamRefusal::ContinuationKindMismatch {
                            open,
                            claimed: kind,
                        });
                    }
                    // Re-declared per segment: the offer this one makes says
                    // nothing about whether the encoder survives *it*.
                    self.phase = Phase::Open {
                        kind,
                        records,
                        outlives_segment: lifetime.continues_into_next,
                    };
                    return Ok(SegmentOpening::Continued(at));
                }
                // A segment is still open, so no boundary has been crossed and
                // there is nothing to continue across one.
                Phase::Open { .. } => return Err(StreamRefusal::ContinuationNotOffered(kind)),
                Phase::Closed => return Err(StreamRefusal::ContinuationWithoutEncoder(kind)),
            }
        }
        match self.phase {
            Phase::Open { kind, .. } | Phase::Held { kind, .. } => {
                return Err(StreamRefusal::EncoderStillOpen(kind))
            }
            Phase::Closed => {}
        }
        self.phase = Phase::Open {
            kind,
            records: 0,
            outlives_segment: lifetime.continues_into_next,
        };
        Ok(SegmentOpening::Opened(
            at,
            SegmentBegin {
                kind,
                protection: self.pending_protection.take(),
            },
        ))
    }

    /// A record was decoded on `rail` inside the open encoder.
    ///
    /// Returns the record's position, which is what an [`crate::access`] intent
    /// and a hazard edge are keyed by.
    pub fn record(&mut self, rail: Rail) -> Result<StreamPosition, StreamRefusal> {
        let Phase::Open { kind, records, .. } = &mut self.phase else {
            return Err(StreamRefusal::RecordOutsideEncoder);
        };
        if kind.rail() != rail {
            return Err(StreamRefusal::RailMismatch {
                segment: *kind,
                record: rail,
            });
        }
        *records += 1;
        let at = StreamPosition {
            segment: self.segment,
            record: self.record,
        };
        self.record += 1;
        Ok(at)
    }

    /// The segment's records are over.
    ///
    /// The encoder ends here unless the header declared it outlives the
    /// segment, in which case it stays open for the next one to continue.
    /// Either way this is a segment boundary, so the position counter advances
    /// and the per-segment record counter restarts — an encoder spanning three
    /// segments still gives its records three distinct segment indices, because
    /// where a record was written is not the same question as which encoder ran
    /// it.
    pub fn end(&mut self) -> Result<SegmentEnd, StreamRefusal> {
        let Phase::Open {
            kind,
            records,
            outlives_segment,
        } = self.phase
        else {
            return Err(StreamRefusal::EndWithoutBegin);
        };
        self.segment += 1;
        self.record = 0;
        if outlives_segment {
            self.phase = Phase::Held { kind, records };
            return Ok(SegmentEnd::EncoderHeld { records });
        }
        self.phase = Phase::Closed;
        Ok(SegmentEnd::EncoderEnded { records })
    }

    /// The stream is over. Anything still open or still armed is a refusal.
    pub fn finish(self) -> Result<u32, StreamRefusal> {
        if let Phase::Open { kind, .. } | Phase::Held { kind, .. } = self.phase {
            return Err(StreamRefusal::EncoderNeverEnded(kind));
        }
        if self.pending_protection.is_some() {
            return Err(StreamRefusal::ProtectionEnvelopeUnclaimed);
        }
        Ok(self.segment)
    }

    /// The kind of the open encoder, if one is open.
    #[must_use]
    pub const fn open_encoder(&self) -> Option<SegmentKind> {
        match self.phase {
            Phase::Open { kind, .. } => Some(kind),
            // An encoder held between segments takes no records, so there is no
            // encoder for one to be admitted by.
            Phase::Held { .. } | Phase::Closed => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One segment, one encoder: the shape every stream this guest has been
    /// measured emitting uses.
    const SELF: SegmentLifetime = SegmentLifetime::SELF_CONTAINED;

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

    /// Each kind admits its own rail's records and no other's.
    #[test]
    fn a_segment_admits_exactly_its_own_rail() {
        for &kind in SegmentKind::ALL {
            for &other in SegmentKind::ALL {
                let mut c = StreamCursor::new();
                c.begin(kind.wire_type(), SELF).expect("opens");
                let got = c.record(other.rail());
                if kind == other {
                    assert!(got.is_ok());
                } else {
                    assert_eq!(
                        got,
                        Err(StreamRefusal::RailMismatch {
                            segment: kind,
                            record: other.rail(),
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn records_are_positioned_in_wire_order_across_segments() {
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Render.wire_type(), SELF)
            .expect("open");
        let a = c.record(Rail::Render).expect("rec");
        let b = c.record(Rail::Render).expect("rec");
        assert_eq!(c.end(), Ok(SegmentEnd::EncoderEnded { records: 2 }));
        c.begin(SegmentKind::Blit.wire_type(), SELF).expect("open");
        let d = c.record(Rail::Blit).expect("rec");
        assert_eq!(c.end(), Ok(SegmentEnd::EncoderEnded { records: 1 }));
        assert_eq!(c.finish(), Ok(2));

        assert!(a < b && b < d);
        let order = StreamOrder;
        assert!(order.precedes(a, d));
        assert!(!order.precedes(d, a));
        assert!(!order.independent(a, d));
    }

    /// Every malformed arrangement is a named refusal rather than a tolerated
    /// one, because each of them would otherwise attribute work to the wrong
    /// encoder or drop it.
    #[test]
    fn malformed_arrangements_are_named_refusals() {
        let mut c = StreamCursor::new();
        assert_eq!(
            c.record(Rail::Render),
            Err(StreamRefusal::RecordOutsideEncoder)
        );
        assert_eq!(c.end(), Err(StreamRefusal::EndWithoutBegin));
        assert_eq!(c.begin(9, SELF), Err(StreamRefusal::UnknownSegmentType(9)));

        c.begin(SegmentKind::Render.wire_type(), SELF)
            .expect("open");
        assert_eq!(
            c.begin(SegmentKind::Compute.wire_type(), SELF),
            Err(StreamRefusal::EncoderStillOpen(SegmentKind::Render))
        );
        assert_eq!(
            c.finish(),
            Err(StreamRefusal::EncoderNeverEnded(SegmentKind::Render))
        );
    }

    /// The envelope arms the next begin, exactly once.
    #[test]
    fn a_protection_envelope_arms_the_segment_after_it() {
        let mut c = StreamCursor::new();
        c.protection_envelope(ProtectionOptions(0x44))
            .expect("armed");
        let SegmentOpening::Opened(_, begin) =
            c.begin(SegmentKind::Blit.wire_type(), SELF).expect("open")
        else {
            panic!("a fresh encoder opens")
        };
        assert_eq!(begin.protection, Some(ProtectionOptions(0x44)));
        assert!(begin.demands_protection());
        c.end().expect("end");

        // Not the one after that.
        let SegmentOpening::Opened(_, next) =
            c.begin(SegmentKind::Blit.wire_type(), SELF).expect("open")
        else {
            panic!("a fresh encoder opens")
        };
        assert_eq!(next.protection, None);
        assert!(!next.demands_protection());
    }

    /// A second envelope would silently replace the first, and a protection
    /// request the device drops without a word is the loss this refuses.
    #[test]
    fn an_envelope_that_arms_nothing_is_refused() {
        let mut c = StreamCursor::new();
        c.protection_envelope(ProtectionOptions(1)).expect("armed");
        assert_eq!(
            c.protection_envelope(ProtectionOptions(2)),
            Err(StreamRefusal::ProtectionEnvelopeUnclaimed)
        );
        assert_eq!(c.finish(), Err(StreamRefusal::ProtectionEnvelopeUnclaimed));
    }

    /// An envelope inside an open encoder is not the burst the contract
    /// describes; it precedes a segment header, it does not interrupt one.
    #[test]
    fn an_envelope_may_not_arrive_inside_an_encoder() {
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Compute.wire_type(), SELF)
            .expect("open");
        assert_eq!(
            c.protection_envelope(ProtectionOptions(1)),
            Err(StreamRefusal::EncoderStillOpen(SegmentKind::Compute))
        );
    }

    /// An encoder split across segments is one encoder, and its record count is
    /// the encoder's rather than the last segment's.
    #[test]
    fn an_encoder_may_span_segments_when_both_headers_say_so() {
        let mut c = StreamCursor::new();
        let opening = c.begin(SegmentKind::Blit.wire_type(), HOLDS).expect("open");
        assert!(matches!(opening, SegmentOpening::Opened(..)));
        let a = c.record(Rail::Blit).expect("rec");
        assert_eq!(c.end(), Ok(SegmentEnd::EncoderHeld { records: 1 }));

        // No second opening: the encoder is the one already running.
        let opening = c
            .begin(SegmentKind::Blit.wire_type(), RELAYS)
            .expect("continue");
        let SegmentOpening::Continued(_) = opening else {
            panic!("a continuation opens no encoder")
        };
        let b = c.record(Rail::Blit).expect("rec");
        assert_eq!(c.end(), Ok(SegmentEnd::EncoderHeld { records: 2 }));

        c.begin(SegmentKind::Blit.wire_type(), TAKES)
            .expect("continue");
        let d = c.record(Rail::Blit).expect("rec");
        assert_eq!(
            c.end(),
            Ok(SegmentEnd::EncoderEnded { records: 3 }),
            "the count is the encoder's, not the last segment's"
        );
        assert_eq!(c.finish(), Ok(3));

        // Where a record was written is still answered per segment.
        assert_eq!([a.segment, b.segment, d.segment], [0, 1, 2]);
        assert!(a < b && b < d);
    }

    /// A continuation is one edge recorded from both headers, so half of one is
    /// a refusal from either side. Reading either half alone attributes records
    /// to a pass on a guess.
    #[test]
    fn half_a_continuation_edge_is_refused_from_either_side() {
        // Claimed with nothing open.
        let mut c = StreamCursor::new();
        assert_eq!(
            c.begin(SegmentKind::Blit.wire_type(), TAKES),
            Err(StreamRefusal::ContinuationWithoutEncoder(SegmentKind::Blit))
        );

        // Claimed against an encoder that never offered it.
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Blit.wire_type(), SELF).expect("open");
        c.end().expect("end");
        assert_eq!(
            c.begin(SegmentKind::Blit.wire_type(), TAKES),
            Err(StreamRefusal::ContinuationWithoutEncoder(SegmentKind::Blit))
        );

        // Offered, then claimed by another family.
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Blit.wire_type(), HOLDS).expect("open");
        c.end().expect("held");
        assert_eq!(
            c.begin(SegmentKind::Compute.wire_type(), TAKES),
            Err(StreamRefusal::ContinuationKindMismatch {
                open: SegmentKind::Blit,
                claimed: SegmentKind::Compute,
            })
        );

        // Offered and not taken: the encoder is abandoned.
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Blit.wire_type(), HOLDS).expect("open");
        c.end().expect("held");
        assert_eq!(
            c.begin(SegmentKind::Blit.wire_type(), SELF),
            Err(StreamRefusal::EncoderStillOpen(SegmentKind::Blit))
        );

        // Claimed without a segment boundary in between: a begin arriving
        // inside a segment that is still open has nothing to continue *across*.
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Blit.wire_type(), HOLDS).expect("open");
        assert_eq!(
            c.begin(SegmentKind::Blit.wire_type(), TAKES),
            Err(StreamRefusal::ContinuationNotOffered(SegmentKind::Blit))
        );

        // A held encoder takes no records: it is alive, and no segment is open
        // for one to be written into.
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Blit.wire_type(), HOLDS).expect("open");
        c.end().expect("held");
        assert_eq!(
            c.record(Rail::Blit),
            Err(StreamRefusal::RecordOutsideEncoder)
        );
        assert_eq!(c.end(), Err(StreamRefusal::EndWithoutBegin));

        // Held open to the end of the stream.
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Blit.wire_type(), HOLDS).expect("open");
        c.end().expect("held");
        assert_eq!(
            c.finish(),
            Err(StreamRefusal::EncoderNeverEnded(SegmentKind::Blit))
        );
    }

    /// `continues_into_next` is re-read on every segment of a spanning encoder.
    ///
    /// A cursor that recorded it once, at the opening, would let a relayed
    /// encoder be continued forever — or refuse the segment that legitimately
    /// ends it.
    #[test]
    fn the_offer_is_re_declared_by_each_segment_of_a_spanning_encoder() {
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Blit.wire_type(), HOLDS).expect("open");
        c.end().expect("held");
        // This one takes the offer and makes none.
        c.begin(SegmentKind::Blit.wire_type(), TAKES)
            .expect("continue");
        assert_eq!(c.end(), Ok(SegmentEnd::EncoderEnded { records: 0 }));
        assert_eq!(
            c.begin(SegmentKind::Blit.wire_type(), TAKES),
            Err(StreamRefusal::ContinuationWithoutEncoder(SegmentKind::Blit))
        );
    }

    /// A protection envelope cannot arm a segment that opens no encoder.
    #[test]
    fn an_envelope_may_not_arm_a_continuation() {
        let mut c = StreamCursor::new();
        c.begin(SegmentKind::Blit.wire_type(), HOLDS).expect("open");
        c.end().expect("held");
        assert_eq!(
            c.protection_envelope(ProtectionOptions(1)),
            Err(StreamRefusal::EncoderStillOpen(SegmentKind::Blit))
        );
    }

    /// Every refusal reason is distinct, so a log line identifies which shape
    /// was seen.
    #[test]
    fn refusal_reasons_are_distinct() {
        let all = [
            StreamRefusal::UnknownSegmentType(0),
            StreamRefusal::EncoderStillOpen(SegmentKind::Render),
            StreamRefusal::RecordOutsideEncoder,
            StreamRefusal::RailMismatch {
                segment: SegmentKind::Render,
                record: Rail::Blit,
            },
            StreamRefusal::EndWithoutBegin,
            StreamRefusal::ProtectionEnvelopeUnclaimed,
            StreamRefusal::EncoderNeverEnded(SegmentKind::Render),
            StreamRefusal::ContinuationWithoutEncoder(SegmentKind::Render),
            StreamRefusal::ContinuationKindMismatch {
                open: SegmentKind::Render,
                claimed: SegmentKind::Blit,
            },
            StreamRefusal::ContinuationNotOffered(SegmentKind::Render),
        ];
        let mut seen: Vec<&str> = all.iter().map(|r| r.reason()).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before);
    }

    struct Rng(u64);

    impl Rng {
        const fn new(seed: u64) -> Self {
            Self(seed ^ 0x9E37_79B9_7F4A_7C15)
        }

        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: u64) -> u64 {
            if bound == 0 {
                return 0;
            }
            self.next() % bound
        }
    }

    const KINDS: [SegmentKind; 5] = [
        SegmentKind::Render,
        SegmentKind::Compute,
        SegmentKind::Blit,
        SegmentKind::Event,
        SegmentKind::Info,
    ];

    const RAILS: [Rail; 5] = [
        Rail::Render,
        Rail::Compute,
        Rail::Blit,
        Rail::Event,
        Rail::Info,
    ];

    /// Everything a caller of the cursor can be affected by.
    ///
    /// The private fields, because a refusal that moved the position counter
    /// would be invisible from outside until the next accepted record landed
    /// somewhere nothing expects it.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct State {
        phase: Phase,
        pending: Option<ProtectionOptions>,
        segment: u32,
        record: u32,
    }

    fn state(c: &StreamCursor) -> State {
        State {
            phase: c.phase,
            pending: c.pending_protection,
            segment: c.segment,
            record: c.record,
        }
    }

    #[derive(Default)]
    struct Census {
        opened: u32,
        continued: u32,
        records: u32,
        ended: u32,
        held: u32,
        finished: u32,
        refusals: std::collections::BTreeMap<&'static str, u32>,
    }

    impl Census {
        fn refusal(&mut self, refusal: StreamRefusal) {
            *self.refusals.entry(refusal.reason()).or_insert(0) += 1;
        }
    }

    /// **Every position is distinct and moves forward, an encoder's record
    /// count survives its continuations, and a refusal moves nothing.**
    ///
    /// Not checked against a second state machine, because a second state
    /// machine would be this one written twice and would agree with it about
    /// everything including a mistake. What it is checked against is the
    /// *history*: the whole list of positions the run handed out, the whole
    /// count of records the run accepted, and a snapshot of the cursor taken
    /// before every call.
    ///
    /// The refusal snapshot is the sharp one. Six of the ten refusals are
    /// decided part-way through a call — after a kind has been read, after a
    /// phase has been matched — and a refusal that had already moved the
    /// segment counter or taken the pending envelope would leave the next
    /// accepted record at a position nothing expects, with no failure anywhere
    /// to attribute it to.
    ///
    /// Floors on all ten refusal reasons, so "this stream shape is unreachable"
    /// cannot pass as "this stream shape was never tried".
    #[test]
    fn a_position_is_handed_out_once_and_a_refusal_hands_out_nothing() {
        let mut census = Census::default();
        for seed in 0..600u64 {
            let mut rng = Rng::new(seed + 1);
            let mut cursor = StreamCursor::new();
            let mut positions: Vec<StreamPosition> = Vec::new();
            // Records accepted since the current encoder opened, which a
            // continuation must carry across a segment boundary.
            let mut in_encoder = 0u32;
            let mut ends = 0u32;
            // Which encoder the driver believes is alive and whether it is
            // inside a segment. Followed from the cursor's answers, never
            // predicted: it steers the driver towards streams a serializer
            // would actually write and asserts nothing.
            let mut alive: Option<(SegmentKind, bool)> = None;

            for _ in 0..20 {
                let before = state(&cursor);
                // A quarter of the steps are whatever, which is what keeps all
                // ten refusals reachable; the rest follow the shape a stream
                // has, which is what gets records past admission at all.
                let step = if rng.below(4) == 0 {
                    rng.below(10)
                } else {
                    match alive {
                        None => {
                            if rng.below(8) == 0 {
                                0
                            } else {
                                3
                            }
                        }
                        Some((_, true)) => {
                            if rng.below(3) == 0 {
                                8
                            } else {
                                6
                            }
                        }
                        Some((_, false)) => 3,
                    }
                };
                let refusal = match step {
                    0 => cursor
                        .protection_envelope(ProtectionOptions(rng.below(3) + 1))
                        .err(),
                    1 => {
                        // The byte door, so an unknown segment type is driven
                        // through the same call a real header takes.
                        let byte = rng.below(0x30) as u8;
                        let lifetime = SegmentLifetime {
                            continues_previous: rng.below(3) == 0,
                            continues_into_next: rng.below(3) == 0,
                        };
                        match cursor.begin(byte, lifetime) {
                            Ok(opening) => {
                                took(&mut census, opening, &mut in_encoder);
                                None
                            }
                            Err(refusal) => Some(refusal),
                        }
                    }
                    2..=4 => {
                        // A held encoder is continued by naming its own kind;
                        // anything else is a stream nobody writes, and the
                        // random quarter above is where those come from.
                        let (kind, continues_previous) = match alive {
                            Some((held, false)) if rng.below(4) != 0 => (held, true),
                            _ => (
                                KINDS[rng.below(KINDS.len() as u64) as usize],
                                rng.below(4) == 0,
                            ),
                        };
                        let lifetime = SegmentLifetime {
                            continues_previous,
                            continues_into_next: rng.below(2) == 0,
                        };
                        match cursor.begin_kind(kind, lifetime) {
                            Ok(opening) => {
                                took(&mut census, opening, &mut in_encoder);
                                alive = Some((kind, true));
                                None
                            }
                            Err(refusal) => Some(refusal),
                        }
                    }
                    5..=7 => {
                        let rail = match alive {
                            Some((kind, true)) if rng.below(5) != 0 => kind.rail(),
                            _ => RAILS[rng.below(RAILS.len() as u64) as usize],
                        };
                        match cursor.record(rail) {
                            Ok(at) => {
                                census.records += 1;
                                in_encoder += 1;
                                // The open encoder took it, so it was open and
                                // it was this rail's.
                                assert_eq!(
                                    before.phase_kind().map(SegmentKind::rail),
                                    Some(rail),
                                    "seed {seed}: a record was taken by the wrong encoder"
                                );
                                positions.push(at);
                                None
                            }
                            Err(refusal) => {
                                match refusal {
                                    StreamRefusal::RecordOutsideEncoder => assert_eq!(
                                        before.phase_kind(),
                                        None,
                                        "seed {seed}: an open encoder refused a record as if none \
                                         were open"
                                    ),
                                    StreamRefusal::RailMismatch { segment, .. } => {
                                        assert_eq!(before.phase_kind(), Some(segment));
                                        assert_ne!(segment.rail(), rail);
                                    }
                                    other => panic!("seed {seed}: a record refused as {other:?}"),
                                }
                                Some(refusal)
                            }
                        }
                    }
                    _ => match cursor.end() {
                        Ok(SegmentEnd::EncoderEnded { records }) => {
                            census.ended += 1;
                            ends += 1;
                            alive = None;
                            assert_eq!(
                                records, in_encoder,
                                "seed {seed}: the encoder's record count is not what it took"
                            );
                            in_encoder = 0;
                            None
                        }
                        Ok(SegmentEnd::EncoderHeld { records }) => {
                            census.held += 1;
                            ends += 1;
                            alive = alive.map(|(kind, _)| (kind, false));
                            assert_eq!(
                                records, in_encoder,
                                "seed {seed}: a held encoder's count is not what it took"
                            );
                            None
                        }
                        Err(refusal) => Some(refusal),
                    },
                };

                if let Some(refusal) = refusal {
                    census.refusal(refusal);
                    assert_eq!(
                        state(&cursor),
                        before,
                        "seed {seed}: {refusal:?} moved the cursor"
                    );
                }
            }

            // Every position handed out is distinct and later than the one
            // before it, over the whole run rather than per call.
            let mut sorted = positions.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                positions.len(),
                "seed {seed}: a position was handed out twice"
            );
            assert!(
                positions.windows(2).all(|w| w[0] < w[1]),
                "seed {seed}: positions went backwards"
            );
            // And a position's record index is its index *within its segment*,
            // which is what makes an encoder spanning three segments give its
            // records three distinct segment indices rather than one long
            // count. Checked by grouping, because the property is about the
            // grouping and not about the sequence.
            let mut by_segment: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
            for at in &positions {
                by_segment.entry(at.segment).or_default().push(at.record);
            }
            for (segment, records) in &by_segment {
                assert_eq!(
                    records,
                    &(0..records.len() as u32).collect::<Vec<u32>>(),
                    "seed {seed}: segment {segment} does not number its records from zero"
                );
            }

            let before = state(&cursor);
            match cursor.finish() {
                Ok(segments) => {
                    census.finished += 1;
                    assert_eq!(
                        segments, ends,
                        "seed {seed}: the segment count is not the segments that ended"
                    );
                    assert_eq!(before.phase, Phase::Closed);
                    assert_eq!(before.pending, None);
                }
                Err(refusal) => {
                    census.refusal(refusal);
                    match refusal {
                        StreamRefusal::EncoderNeverEnded(kind) => {
                            assert_eq!(before.phase_kind_including_held(), Some(kind));
                        }
                        StreamRefusal::ProtectionEnvelopeUnclaimed => {
                            assert!(before.pending.is_some());
                            assert_eq!(before.phase, Phase::Closed);
                        }
                        other => panic!("seed {seed}: finish refused as {other:?}"),
                    }
                }
            }
        }

        assert!(census.opened > 1000, "{}", census.opened);
        assert!(census.continued > 500, "{}", census.continued);
        assert!(census.records > 2000, "{}", census.records);
        assert!(census.ended > 800, "{}", census.ended);
        assert!(census.held > 800, "{}", census.held);
        assert!(census.finished > 50, "{}", census.finished);

        // Every refusal this module names, reached. A reason with no count is
        // a stream shape the sweep never built, and a floor that only says
        // "some refusals happened" would let nine of the ten go undriven.
        for reason in [
            "stream_segment_type_unknown",
            "stream_encoder_begin_while_open",
            "stream_continuation_without_encoder",
            "stream_continuation_kind_mismatch",
            "stream_continuation_not_offered",
            "stream_record_outside_encoder",
            "stream_record_rail_mismatch",
            "stream_encoder_end_without_begin",
            "stream_protection_envelope_unclaimed",
            "stream_encoder_never_ended",
        ] {
            let count = census.refusals.get(reason).copied().unwrap_or(0);
            assert!(count > 50, "{reason} was reached {count} times");
        }
    }

    fn took(census: &mut Census, opening: SegmentOpening, in_encoder: &mut u32) {
        match opening {
            SegmentOpening::Opened(..) => {
                census.opened += 1;
                // A new encoder starts with no records, whatever the last one
                // had.
                *in_encoder = 0;
            }
            SegmentOpening::Continued(_) => census.continued += 1,
        }
    }

    impl State {
        /// The kind of the encoder that would take a record right now.
        const fn phase_kind(self) -> Option<SegmentKind> {
            match self.phase {
                Phase::Open { kind, .. } => Some(kind),
                Phase::Held { .. } | Phase::Closed => None,
            }
        }

        /// The kind of the encoder that is alive, open or held.
        const fn phase_kind_including_held(self) -> Option<SegmentKind> {
            match self.phase {
                Phase::Open { kind, .. } | Phase::Held { kind, .. } => Some(kind),
                Phase::Closed => None,
            }
        }
    }
}
