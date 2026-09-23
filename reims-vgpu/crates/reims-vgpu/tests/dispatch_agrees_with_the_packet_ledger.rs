//! The device's opcode dispatch and the protocol crate's packet ledger are two
//! tables of the same numbers. This is the only thing that compares them.
//!
//! # Why it matters more than a tidiness check
//!
//! The ledger is what the replacement routes by: `classify` reads a row and
//! answers a payload class, and a packet with no row gets no class and may not
//! be claimed. `process_root_packet` and `process_child_packet` are what the
//! guest actually reaches today. Every difference between the two sets is a
//! defect the cutover would deliver:
//!
//! - an opcode production handles with no ledger row is work the guest gets
//!   today and would be refused after the switch;
//! - a ledger row production never reaches is a command the ledger promises an
//!   outcome for and the device answers with nothing.
//!
//! The model's own suite (`every_packet_reaches_one_class`) pins `classify`
//! against each payload class's `Kind::of`. Both of those live in the
//! replacement crates. Nothing looked at the device.
//!
//! # The retired rows are the one asymmetry, and it is stated
//!
//! Fifteen child rows are `Closure::ProvenNoOp` against the reference host's
//! single shared retired-command handler: the packet is accepted, its payload
//! is ignored and its stamps retire. Production has no named constant for those
//! and reaches them through its default arm, which is that same behaviour. So
//! they are expected on the ledger's side of the difference and nowhere else —
//! asserted as an exact set rather than subtracted, so a row that stopped being
//! retired would fail here instead of being quietly forgiven.
//!
//! # And the rows that block the cutover are counted, not skipped
//!
//! Five child rows are `Closure::Unresolved`, and `classify` answers `None` for
//! each. Production handles all five today. That is the cutover's actual
//! remaining gap and it is asserted here so it shrinks visibly and cannot grow
//! unnoticed.

use reims_vgpu_protocol::closure::Closure;
use reims_vgpu_protocol::packets::{Channel, LEDGER};
use reims_vgpu_testkit::dispatch::{opcodes_named_in, u16_constants};
use std::collections::BTreeSet;

const PREFIXES: &[&str] = &["ROOT_OP_", "CHILD_OP_"];

fn regs() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/regs.rs"))
        .expect("the device's opcode constants")
}

fn drain() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/runtime/drain/mod.rs"
    ))
    .expect("the device's packet dispatch")
}

/// The opcodes one dispatch function names.
fn handled(function: &str) -> BTreeSet<u16> {
    let constants = u16_constants(&regs(), PREFIXES).expect("the opcode constants are readable");
    assert!(
        constants.len() > 20,
        "the constant scan found {} opcodes, which is not this device's opcode space — a scan \
         that matches nothing passes every comparison below",
        constants.len()
    );
    let named = opcodes_named_in(&drain(), function, &constants)
        .unwrap_or_else(|e| panic!("scanning {function}: {e}"));
    assert!(
        !named.is_empty(),
        "{function} names no opcode constant, so the comparison below is vacuous"
    );
    named.into_values().collect()
}

fn ledger(channel: Channel) -> BTreeSet<u16> {
    LEDGER
        .iter()
        .filter(|p| p.channel == channel)
        .map(|p| p.opcode)
        .collect()
}

/// Rows whose whole contract is the shared retired-command handler, which
/// production reaches through its default arm rather than a named constant.
fn retired(channel: Channel) -> BTreeSet<u16> {
    LEDGER
        .iter()
        .filter(|p| p.channel == channel && matches!(p.closure, Closure::ProvenNoOp { .. }))
        .map(|p| p.opcode)
        .collect()
}

#[test]
fn the_root_dispatch_handles_exactly_the_root_rows() {
    let handled = handled("process_root_packet");
    let ledger = ledger(Channel::Root);
    assert!(
        retired(Channel::Root).is_empty(),
        "the root channel has no retired slots, so its two tables are equal with no asymmetry \
         to state"
    );
    assert_eq!(
        handled.difference(&ledger).copied().collect::<Vec<_>>(),
        Vec::<u16>::new(),
        "the device handles root opcodes the ledger has no row for; the replacement would refuse \
         work the guest gets today"
    );
    assert_eq!(
        ledger.difference(&handled).copied().collect::<Vec<_>>(),
        Vec::<u16>::new(),
        "the ledger has root rows the device's dispatch never reaches"
    );
}

#[test]
fn the_child_dispatch_handles_exactly_the_child_rows_that_are_not_retired() {
    let handled = handled("process_child_packet");
    let ledger = ledger(Channel::Child);
    let retired = retired(Channel::Child);
    assert_eq!(
        handled.difference(&ledger).copied().collect::<Vec<_>>(),
        Vec::<u16>::new(),
        "the device handles child opcodes the ledger has no row for; the replacement would refuse \
         work the guest gets today"
    );
    assert_eq!(
        ledger.difference(&handled).copied().collect::<Vec<_>>(),
        retired.iter().copied().collect::<Vec<_>>(),
        "the only child rows the dispatch does not name are the retired slots, whose contract is \
         the default arm's behaviour"
    );
    assert!(
        !retired.is_empty(),
        "no retired rows, so the assertion above compared two empty sets"
    );
}

/// What is left before every admitted packet can be routed by class.
///
/// Not a fixed number for its own sake: the count is what makes the gap visible
/// in a suite run, and a row closed without this being updated is a passing
/// test that no longer describes the device.
#[test]
fn the_rows_that_block_routing_are_the_ones_production_still_answers_alone() {
    let blocking: Vec<(Channel, u16, &str)> = LEDGER
        .iter()
        .filter(|p| p.closure.blocks_cutover())
        .map(|p| (p.channel, p.opcode, p.name))
        .collect();
    assert_eq!(
        blocking,
        vec![
            (Channel::Child, 0x00, "CmdDebug"),
            (Channel::Child, 0x09, "CmdDisplaySleepState"),
            (Channel::Child, 0x0a, "CmdDisplaySetProperties"),
            (Channel::Child, 0x3d, "CmdDelay"),
        ],
        "the unresolved rows changed; the cutover's remaining gap is not what this says it is"
    );

    // Every one of them is a packet the guest sends and this device answers
    // today, which is why they block rather than merely being unwritten: the
    // replacement cannot claim them and production cannot stop.
    let handled = handled("process_child_packet");
    for (_, opcode, name) in blocking {
        assert!(
            handled.contains(&opcode),
            "{name} ({opcode:#04x}) blocks the cutover but the device does not handle it either, \
             so the row is unreachable rather than open"
        );
    }
}
