//! Does the closure ledger describe *these* decoders?
//!
//! `reims_vgpu_protocol::closure` records one outcome per decodable operation,
//! and its own tests keep the row set equal to the serializer's selector
//! manifest. Neither of those touches this crate. So the ledger could say a
//! render opcode is implemented while this rail's decoder refuses it as
//! unknown, and nothing would notice: one is a statement about the protocol,
//! the other is a match arm, and until now they were only ever read by people.
//!
//! These tests are the join. They drive the real per-rail decoders across every
//! ledger row and assert the two agree about which operations exist. They say
//! nothing about whether an outcome is the *right* one — no test can — only
//! that the ledger and the decoders are describing the same opcode space.
//!
//! # What "recognised" means here
//!
//! Each rail refuses an opcode outside its own accepted window with a distinct
//! status (`ErrUnknownOpcode`, `ErrUnsupportedOpcode`), separately from
//! refusing a record whose payload is too short for its layout (`ErrShort`,
//! `ErrBadLength`). Only the first pair is a claim about the opcode, so that is
//! what is asserted: a generously sized zero payload is offered and the decoder
//! may still refuse its *contents*, because a record of zeroes is not a record
//! any of these selectors would actually write.

use reims_vgpu_protocol::closure::{Rail, LEDGER};

/// A record header for `opcode` followed by `payload` zero bytes.
fn zero_record(opcode: u32, payload: usize) -> Vec<u8> {
    let total = reims_vgpu_wire::OP_HEADER_LEN + payload;
    let mut v = vec![0u8; total];
    v[0..4].copy_from_slice(&opcode.to_le_bytes());
    v[4..8].copy_from_slice(&(total as u32).to_le_bytes());
    v
}

/// Wide enough for every head in these families plus a counted entry array;
/// the point is to reach the opcode arm, not to satisfy a layout.
const GENEROUS_PAYLOAD: usize = 256;

/// Candidate records for one opcode, for the tests that need a decode to
/// *succeed* rather than merely not be refused by opcode.
///
/// Several of these families check the payload length exactly — "the product
/// refuses slack the guest did not size for" — so one generous buffer reaches
/// almost none of them. The probe walks the plausible lengths instead, with and
/// without a leading `1` for the count-led families, and the caller takes
/// whichever spelling decodes. Nothing here claims to synthesise a *meaningful*
/// record; it only has to reach the arm.
fn probe_records(opcode: u32) -> impl Iterator<Item = Vec<u8>> {
    (0..=GENEROUS_PAYLOAD).flat_map(move |len| {
        let plain = zero_record(opcode, len);
        let counted = (len >= 4).then(|| {
            let mut v = zero_record(opcode, len);
            v[reims_vgpu_wire::OP_HEADER_LEN..reims_vgpu_wire::OP_HEADER_LEN + 4]
                .copy_from_slice(&1u32.to_le_bytes());
            v
        });
        core::iter::once(plain).chain(counted)
    })
}

/// Every compute-rail row reaches exactly one decoder, and `classify` is what
/// says which.
///
/// The same two claims the blit test below makes, on the rail whose split is
/// wider. The seventeen records the ledger has settled are lifted by
/// `reims_vgpu_protocol::decode::compute`, the two barriers by `decode::sync`,
/// the compressed-reinterpretation flush by `decode::resource_state`, the
/// unqualified residency pair by `decode::residency`, and the eleven rows the
/// ledger has **not settled** by this crate's `decode::compute_spi`.
///
/// No row is dropped by every decoder, and no row is claimed by two. The second
/// is the claim a single decoder could not make, and it is what catches a
/// settled row growing a second reading of its bytes — which on this rail is
/// the specific failure the cutover removed, since the eleven unsettled rows
/// keep a decoder of their own and drive real work from it.
#[test]
fn every_compute_row_reaches_exactly_one_decoder_and_the_ledger_picks_it() {
    use reims_vgpu_protocol::decode::{compute, residency, resource_state, sync, DecodeRefusal};
    for op in LEDGER.iter().filter(|o| o.rail == Rail::Compute) {
        let Some(opcode) = op.opcode else { continue };
        let bytes = zero_record(opcode, GENEROUS_PAYLOAD);
        let framed = reims_vgpu_protocol::decode::op(&bytes, 0).expect("framed");
        // "Claims this record" means: does not refuse it for being an opcode
        // this decoder does not own. A refusal for a payload of zeroes is a
        // statement about the bytes, not about ownership.
        fn disowned<T>(r: &Result<T, DecodeRefusal>) -> bool {
            matches!(
                r,
                Err(DecodeRefusal::UnknownOpcode { .. }) | Err(DecodeRefusal::Unjudged { .. })
            )
        }
        let claimants: Vec<&str> = [
            ("record", !disowned(&compute::decode(&framed))),
            ("sync", !disowned(&sync::decode(Rail::Compute, &framed))),
            (
                "resource_state",
                !disowned(&resource_state::decode(Rail::Compute, &framed)),
            ),
            // `lift` rather than `decode`: the residency pair is unsettled, so
            // `decode` refuses it on principle, and the question here is which
            // decoder owns the *layout*.
            (
                "residency",
                !disowned(&residency::lift(Rail::Compute, &framed)),
            ),
            (
                "unsettled",
                !matches!(
                    super::compute_spi::decode(&bytes),
                    Err(super::compute_spi::DecodeStatus::ErrUnknownOpcode)
                        | Err(super::compute_spi::DecodeStatus::ErrSettledElsewhere)
                ),
            ),
        ]
        .into_iter()
        .filter_map(|(name, claimed)| claimed.then_some(name))
        .collect();
        assert_eq!(
            claimants.len(),
            1,
            "the closure ledger records compute {opcode:#x} ({}) and {} decoder(s) claim it: \
             {claimants:?} — a row with none is work this device drops in silence, and a row \
             with two is the second reading of one record's bytes the cutover exists to remove",
            op.selector,
            claimants.len(),
        );
    }
}

/// Every blit-rail row reaches exactly one decoder, and `classify` is what
/// says which.
///
/// The rail no longer has *a* decoder. A record's class comes from the ledger
/// row, and each class is lifted by the layer that owns its layout: the nine
/// transfers by `reims_vgpu_protocol::decode::blit`, fences by `decode::sync`,
/// the indirect-command hint by `decode::icb`, the content directives by
/// `decode::resource_state`, and the four rows the ledger has **not settled**
/// by this crate's `decode::blit_spi`.
///
/// So the claim the old single-decoder test made — "the decoder recognises
/// every row" — is now two claims, and both are worth more than the one was:
/// no row is dropped by every decoder, and no row is claimed by two. The
/// second is the one a single decoder could not make at all, and it is the one
/// that catches a settled row growing a second reading of its bytes.
#[test]
fn every_blit_row_reaches_exactly_one_decoder_and_the_ledger_picks_it() {
    use reims_vgpu_protocol::decode::{blit, icb, resource_state, sync};
    for op in LEDGER.iter().filter(|o| o.rail == Rail::Blit) {
        let Some(opcode) = op.opcode else { continue };
        let bytes = zero_record(opcode, GENEROUS_PAYLOAD);
        let framed = reims_vgpu_protocol::decode::op(&bytes, 0).expect("framed");
        // "Claims this record" means: does not refuse it for being an opcode
        // this decoder does not own. A refusal for a payload of zeroes is a
        // statement about the bytes, and a generously sized zero record is not
        // a record any of these selectors would write.
        let owns = |refused_opcode: bool| !refused_opcode;
        let claimants: Vec<&str> = [
            (
                "transfer",
                owns(matches!(
                    blit::decode(&framed),
                    Err(reims_vgpu_protocol::decode::DecodeRefusal::UnknownOpcode { .. })
                        | Err(reims_vgpu_protocol::decode::DecodeRefusal::Unjudged { .. })
                )),
            ),
            (
                "sync",
                owns(matches!(
                    sync::decode(Rail::Blit, &framed),
                    Err(reims_vgpu_protocol::decode::DecodeRefusal::UnknownOpcode { .. })
                        | Err(reims_vgpu_protocol::decode::DecodeRefusal::Unjudged { .. })
                )),
            ),
            (
                "icb",
                owns(matches!(
                    icb::decode(Rail::Blit, &framed),
                    Err(reims_vgpu_protocol::decode::DecodeRefusal::UnknownOpcode { .. })
                        | Err(reims_vgpu_protocol::decode::DecodeRefusal::Unjudged { .. })
                )),
            ),
            (
                "resource_state",
                owns(matches!(
                    resource_state::decode(Rail::Blit, &framed),
                    Err(reims_vgpu_protocol::decode::DecodeRefusal::UnknownOpcode { .. })
                        | Err(reims_vgpu_protocol::decode::DecodeRefusal::Unjudged { .. })
                )),
            ),
            (
                "unsettled",
                owns(matches!(
                    super::blit_spi::decode(&bytes),
                    Err(super::blit_spi::DecodeStatus::ErrUnknownOpcode)
                )),
            ),
        ]
        .into_iter()
        .filter_map(|(name, claimed)| claimed.then_some(name))
        .collect();
        assert_eq!(
            claimants.len(),
            1,
            "the closure ledger records blit {opcode:#x} ({}) and {} decoder(s) claim it: {claimants:?} \
             — a row with none is work this device drops in silence, and a row with two is \
             the second reading of one record's bytes the cutover exists to remove",
            op.selector,
            claimants.len(),
        );
    }
}

/// The event rail's records lift, and its one refused row still refuses.
///
/// The sibling tests above drive *this crate's* per-rail decoders, because
/// those are the ones that could disagree with the ledger. The event rail no
/// longer has one: `runtime::exec::handle_event_record` lifts through
/// `reims_vgpu_protocol::decode::sync`, which is inside the crate that owns the
/// ledger and is checked against it there.
///
/// What is still this crate's to check is the *pairing* — that the rail the
/// device names when it lifts an event segment is the rail whose records lift,
/// and that the row the ledger settled as refused is refused rather than
/// unjudged. Those two are what a reader of `event_record` refusals is told,
/// and getting either wrong turns a settled decision into an open question.
#[test]
fn the_event_rail_lifts_its_records_and_refuses_its_one_settled_row() {
    use reims_vgpu_protocol::decode::sync::decode;
    use reims_vgpu_protocol::decode::{op, DecodeRefusal};
    let mut refused = 0usize;
    let mut lifted = 0usize;
    for row in LEDGER.iter().filter(|o| o.rail == Rail::Event) {
        let opcode = row.opcode.expect("every event row names an opcode");
        // The rail is the one `handle_event_record` passes, written here rather
        // than taken from the row: a test that read the rail off the row it is
        // checking would pass whatever rail the device actually used.
        // Reduced to `Result<(), DecodeRefusal>` inside the closure: a lifted
        // record borrows the bytes it was lifted from, and only whether it
        // lifted is being asked here.
        let outcomes: Vec<Result<(), DecodeRefusal>> = probe_records(opcode)
            .map(|bytes| {
                let framed = op(&bytes, 0).expect("probe records frame");
                decode(Rail::Event, &framed).map(|_| ())
            })
            .collect();
        if outcomes
            .iter()
            .all(|o| matches!(o, Err(DecodeRefusal::RefusedByContract { .. })))
        {
            refused += 1;
            continue;
        }
        assert!(
            outcomes.iter().any(Result::is_ok),
            "the closure ledger records event {opcode:#x} and the protocol decoder lifts no \
             record for it at any payload length: {:?}",
            outcomes.first()
        );
        lifted += 1;
    }
    assert_eq!(
        refused, 1,
        "one event row is settled as refused — the bounded wait — and it is the only one"
    );
    assert_eq!(lifted, 2, "the signal and the unbounded wait both lift");
}

/// The packet half of the same join.
///
/// `protocol::packets` records an outcome per FIFO packet class, and the
/// serializer manifest cannot enumerate that space — there is no runtime to ask
/// which packets exist, only the dispatch table this device transcribed. So the
/// enumeration lives in `model::regs` and this is what keeps the ledger equal to
/// it in both directions: a command declared here without a row is a packet
/// class nobody has judged, and a row naming an opcode no constant declares is a
/// judgement about a command this device cannot receive.
mod packets {
    use crate::model::{CHILD_COMMANDS, CHILD_OP_MAX, ROOT_COMMANDS};
    use reims_vgpu_core::control::ControlKind;
    use reims_vgpu_protocol::packets::{find, Channel, LEDGER, OPCODE_CEILING};
    use std::collections::BTreeSet;

    /// The child opcodes the ledger judges retired, read the one way this
    /// device reads them.
    ///
    /// The set used to be a constant beside the command constants, and the test
    /// below that walked it was asking the ledger whether it agreed with a copy
    /// of itself. It is the ledger's now, so the question these tests can still
    /// ask is the one that was always the point: whether the *commands* this
    /// device dispatches and the retired set are disjoint, and whether every
    /// judged row is a packet this device can receive at all.
    fn retired_slots() -> BTreeSet<u16> {
        LEDGER
            .iter()
            .filter(|p| p.channel == Channel::Child)
            .filter(|p| ControlKind::of(p.channel, p.opcode) == Some(ControlKind::RetiredSlot))
            .map(|p| p.opcode)
            .collect()
    }

    /// The retired set is not empty and not everything.
    ///
    /// Without this a `ControlKind` change that stopped calling anything a
    /// retired slot would leave the two tests below trivially true: an empty
    /// set is disjoint from every command list and adds no receivable opcode.
    #[test]
    fn the_ledger_judges_a_retired_set_at_all() {
        assert_eq!(
            retired_slots().len(),
            15,
            "the reference host routes fifteen child slots to its shared \
             deprecated handler"
        );
    }

    #[test]
    fn every_declared_command_has_a_ledger_row() {
        for (name, op) in CHILD_COMMANDS {
            assert!(
                find(Channel::Child, *op).is_some(),
                "CHILD_OP_{name} ({op:#04x}) is a command this device dispatches and the closure \
                 ledger does not judge"
            );
        }
        for (name, op) in ROOT_COMMANDS {
            assert!(
                find(Channel::Root, *op).is_some(),
                "ROOT_OP_{name} ({op:#04x}) is a command this device dispatches and the closure \
                 ledger does not judge"
            );
        }
    }

    #[test]
    fn no_ledger_row_names_a_command_this_device_cannot_receive() {
        let child: BTreeSet<u16> = CHILD_COMMANDS
            .iter()
            .map(|(_, op)| *op)
            .chain(retired_slots())
            .collect();
        let root: BTreeSet<u16> = ROOT_COMMANDS.iter().map(|(_, op)| *op).collect();
        for p in LEDGER {
            let known = match p.channel {
                Channel::Child => child.contains(&p.opcode),
                Channel::Root => root.contains(&p.opcode),
            };
            assert!(
                known,
                "the ledger judges {} {:#04x} and no constant in model::regs declares it",
                p.channel.name(),
                p.opcode
            );
        }
    }

    /// Two transcriptions of one number the reference host reads off its own
    /// dispatch table. They were taken separately, and a disagreement means one
    /// of them is describing a table this device does not dispatch against.
    #[test]
    fn the_two_dispatch_ceilings_agree() {
        assert_eq!(CHILD_OP_MAX, OPCODE_CEILING);
    }

    /// The four opcodes the drain answers as queries are the four the model
    /// calls questions.
    ///
    /// `runtime::drain::query_request` asks
    /// [`reims_vgpu_core::query::QueryKind::of`] which question a packet is,
    /// and reports `query_not_a_query_packet` when the answer is `None` — a
    /// reading that means this device's dispatch table and the closure ledger
    /// disagree about what a query is, and one the drain cannot check for
    /// itself because it only ever sees one opcode at a time. This is where it
    /// is checked, in both directions: every query the ledger judges has an arm
    /// here, and every arm has a question.
    #[test]
    fn the_query_arms_and_the_models_questions_are_the_same_four() {
        use reims_vgpu_core::query::QueryKind;
        // The opcodes `process_root_packet` and `process_child_packet` answer
        // with a reply, written out because a test that derived them from the
        // same source the drain does would agree with the drain by
        // construction.
        let arms = [
            (Channel::Root, crate::model::ROOT_OP_DEVICE_INFO_TAHOE),
            (Channel::Root, crate::model::ROOT_OP_DEVICE_INFO_MONTEREY),
            (Channel::Child, crate::model::CHILD_OP_GET_COMPUTE_INFO),
            (
                Channel::Child,
                crate::model::CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN,
            ),
        ];
        for (channel, op) in arms {
            assert!(
                QueryKind::of(channel, op).is_some(),
                "the drain answers {} {op:#04x} with a reply and the model does not call it a \
                 question, so `query_request` would refuse a packet the guest is blocked on",
                channel.name()
            );
        }
        let judged: BTreeSet<(Channel, u16)> = LEDGER
            .iter()
            .filter(|p| QueryKind::of(p.channel, p.opcode).is_some())
            .map(|p| (p.channel, p.opcode))
            .collect();
        assert_eq!(
            judged,
            arms.into_iter().collect::<BTreeSet<_>>(),
            "the model's questions and the drain's query arms are not the same set"
        );
    }

    /// A slot cannot be both a live command and one the host retired, on either
    /// side of the join.
    #[test]
    fn no_live_command_is_also_a_retired_slot() {
        let retired = retired_slots();
        for (name, op) in CHILD_COMMANDS {
            assert!(
                !retired.contains(op),
                "CHILD_OP_{name} is also listed as a retired slot, so the drain would give one \
                 number two arms and the retired one would swallow a live command"
            );
        }
    }
}

/// Every render-rail row reaches exactly one decoder, and the ledger is what
/// says which.
///
/// The rail no longer has *a* decoder. Forty-five rows are lifted by
/// `reims_vgpu_protocol::decode::render`, the fence pair and the three barriers
/// by `decode::sync`, the four residency declarations by `decode::residency`,
/// and the thirty-one rows the ledger has **not settled** by
/// [`super::render_spi`] — which refuses every settled opcode by name, so the
/// two halves cannot both claim a record.
///
/// Replaces `the_render_opcode_table_is_exactly_apples_render_manifest` and
/// `the_accepted_window_ends_where_apples_render_manifest_does`, and is
/// strictly stronger than either: those asked whether one decoder recognised
/// every row, which cannot see a row claimed twice. This asks both questions —
/// no row is dropped by all of them, and no row is claimed by two.
#[test]
fn every_render_row_reaches_exactly_one_decoder_and_the_ledger_picks_it() {
    use reims_vgpu_protocol::decode::{render, residency, sync, DecodeRefusal};
    for op in LEDGER.iter().filter(|o| o.rail == Rail::Render) {
        let Some(opcode) = op.opcode else { continue };
        let bytes = zero_record(opcode, GENEROUS_PAYLOAD);
        let framed = reims_vgpu_protocol::decode::op(&bytes, 0).expect("framed");
        // "Claims this record" means: does not refuse it for being an opcode
        // this decoder does not own. A refusal for a payload of zeroes is a
        // statement about the bytes, not about ownership.
        //
        // `RefusedByContract` is a disowning here and it is the render rail
        // that makes the distinction matter: this rail has four `Refused` rows
        // and the other two have none, so this is the first test to ask what a
        // contract refusal claims. It claims nothing. Every protocol decoder
        // answers `RefusedByContract` for such a row — that is what the ledger
        // told it to answer — so counting it as ownership would make all of
        // them claim the same record, which says nothing about which one owns
        // its layout. `render_spi` owns those four, because for a refusal the
        // *value* is the evidence: which options word, or which sample count.
        fn disowned<T>(r: &Result<T, DecodeRefusal>) -> bool {
            matches!(
                r,
                Err(DecodeRefusal::UnknownOpcode { .. })
                    | Err(DecodeRefusal::Unjudged { .. })
                    | Err(DecodeRefusal::RefusedByContract { .. })
            )
        }
        let claimants: Vec<&str> = [
            ("record", !disowned(&render::decode(&framed))),
            ("sync", !disowned(&sync::decode(Rail::Render, &framed))),
            // `lift` rather than `decode`: all four residency rows are
            // unsettled, so `decode` refuses them on principle, and the
            // question here is which decoder owns the *layout*.
            (
                "residency",
                !disowned(&residency::lift(Rail::Render, &framed)),
            ),
            (
                "unsettled",
                !matches!(
                    super::render_spi::decode(&bytes),
                    Err(super::render_spi::DecodeStatus::ErrUnknownOpcode)
                        | Err(super::render_spi::DecodeStatus::ErrSettledElsewhere)
                ),
            ),
        ]
        .into_iter()
        .filter_map(|(name, claimed)| claimed.then_some(name))
        .collect();
        assert_eq!(
            claimants.len(),
            1,
            "the closure ledger records render {opcode:#x} ({}) and {} decoder(s) claim it: \
             {claimants:?} — a row with none is work this device drops in silence, and a row \
             with two is the second reading of one record's bytes the cutover exists to remove",
            op.selector,
            claimants.len(),
        );
    }
}
