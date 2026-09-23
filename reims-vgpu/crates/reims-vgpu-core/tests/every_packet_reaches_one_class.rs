//! Every packet the ledger admits reaches exactly one payload class's resolver.
//!
//! # Two tables of the same opcodes, and one of them is in another crate
//!
//! [`classify`] says which class a packet is, and each class has its own way in:
//! a lifecycle command through `lifecycle::operation`, a question through
//! `query::request_words`, a frame through `present::resolve`, a control command
//! through `control::resolve`, and GPU work through the stream walker.
//!
//! Three of those ask `classify` before answering, so for them the partition is
//! enforced at its source and what this census pins is a different thing: that
//! each kind's own arm list — twelve opcodes for lifecycle, four for query,
//! twenty-three for control — *is* `classify`'s arm list for that class. Those
//! are two tables of the same numbers, and a row moved in one and not the other
//! is a packet with a class and no kind, which is admitted and then has nothing
//! to do with itself.
//!
//! [`PresentForm::of`] asks nothing. It lives in the protocol crate, which
//! cannot see `classify` at all, so its three opcodes are a genuinely
//! independent table — and this is the only thing that compares them. The exec
//! opcode is the same case: the walker is handed a payload rather than asked
//! whether it wants one, so the only thing naming it is `classify` and the
//! literal here.
//!
//! # The cutover half
//!
//! A row the ledger has not closed gets no class, and nothing may claim it: a
//! resolver that took one would be this device acting on a contract the ledger
//! says is not established. That is the direction the open rows are counted in,
//! and it is checked for every row rather than for the ones anyone remembered.
//!
//! # What is not claimed
//!
//! That the resolvers *succeed*. Most are handed a zero payload here, and a
//! command whose operation is not in its own packet — see
//! `ResolveRefusal::NeedsStorage` — names what is missing
//! rather than producing one. Claiming the packet is the question; what the
//! answer is belongs to each owner's own tests.

use reims_vgpu_core::control::ControlKind;
use reims_vgpu_core::lifecycle::LifecycleKind;
use reims_vgpu_core::query::QueryKind;
use reims_vgpu_core::transaction::{classify, PayloadClass};
use reims_vgpu_protocol::packets::{Channel, LEDGER};
use reims_vgpu_protocol::present::PresentForm;

/// Which classes' resolvers claim this packet, in `PayloadClass` order.
fn claimed_by(channel: Channel, opcode: u16) -> Vec<PayloadClass> {
    let mut out = Vec::new();
    // Exec has no `Kind::of`: it is one opcode, and the stream walker is handed
    // a payload rather than asked whether it wants one. `classify` is the only
    // thing that names it, which is why it is compared against the literal here
    // rather than against a second table.
    if (channel, opcode) == (Channel::Child, 0x37) {
        out.push(PayloadClass::Exec);
    }
    if LifecycleKind::of(channel, opcode).is_some() {
        out.push(PayloadClass::ResourceLifecycle);
    }
    if QueryKind::of(channel, opcode).is_some() {
        out.push(PayloadClass::Query);
    }
    if PresentForm::of(channel, opcode).is_some() {
        out.push(PayloadClass::Present);
    }
    if ControlKind::of(channel, opcode).is_some() {
        out.push(PayloadClass::Control);
    }
    out
}

/// Every classified packet is claimed by the one class `classify` names, and by
/// no other.
///
/// The `<= 1` half holds by construction for the three kinds that consult
/// `classify`; it is a real check for the present forms, whose table is in
/// another crate, and for the exec opcode.
///
/// The equality half is a real check for all five: it is what says each kind's
/// arm list and `classify`'s arm list are the same list.
#[test]
fn a_classified_packet_is_claimed_by_exactly_its_own_class() {
    let mut counts = [0usize; 5];
    for p in LEDGER {
        let claimed = claimed_by(p.channel, p.opcode);
        assert!(
            claimed.len() <= 1,
            "{} {:#04x} is claimed by {claimed:?}",
            p.channel.name(),
            p.opcode
        );
        let Some(class) = classify(p.channel, p.opcode) else {
            // An unjudged or open row has no class, and nothing may claim it:
            // a resolver that took one would be this device acting on a
            // contract the ledger says is not established.
            assert!(
                claimed.is_empty(),
                "{} {:#04x} blocks cutover and is claimed by {claimed:?}",
                p.channel.name(),
                p.opcode
            );
            continue;
        };
        assert_eq!(
            claimed,
            vec![class],
            "{} {:#04x}",
            p.channel.name(),
            p.opcode
        );
        counts[class as usize] += 1;
    }
    // The partition's shape, so a row moving between classes is visible rather
    // than merely still consistent.
    let total: usize = counts.iter().sum();
    assert_eq!(
        total,
        LEDGER
            .iter()
            .filter(|p| classify(p.channel, p.opcode).is_some())
            .count(),
        "every classified row was counted once"
    );
    assert!(
        counts.iter().all(|n| *n > 0),
        "every class has at least one packet: {counts:?}"
    );
}

/// A packet's resolver refuses every packet that is not its own.
///
/// The mirror of the census above, and driven rather than derived: the
/// resolvers are handed every row in the ledger and each must refuse all but
/// its own class.
///
/// `present::resolve` is the one that can genuinely disagree, because it reaches
/// `PresentForm::of` and nothing else. `control::resolve` is the one that can go
/// *silent*: `classify`'s last arm is a catch-all, so every judged row it did
/// not match above becomes `Control`, while `ControlKind::of` names seven
/// opcodes plus whatever the ledger records as a proven no-op. A row that stops
/// being a proven no-op keeps its class and loses its kind — classified,
/// admitted, and unresolvable — and this is what says none currently has.
///
/// The lifecycle half is the stronger claim: not merely that a kind exists, but
/// that `operation` sends it to a join that owns it. A `NotA…` refusal here
/// would mean the dispatcher's arm list and the joins' disagree.
#[test]
fn each_resolver_refuses_the_other_classes_packets() {
    let payload = [0u8; 64];
    for p in LEDGER {
        let mine = classify(p.channel, p.opcode);
        let is = |class| mine == Some(class);

        let present = reims_vgpu_core::present::resolve(p.channel, p.opcode, &payload);
        assert_eq!(
            present.is_ok(),
            is(PayloadClass::Present),
            "{} {:#04x} present: {present:?}",
            p.channel.name(),
            p.opcode
        );

        let control = reims_vgpu_core::control::resolve(p.channel, p.opcode, &payload);
        assert_eq!(
            control.is_ok(),
            is(PayloadClass::Control),
            "{} {:#04x} control: {control:?}",
            p.channel.name(),
            p.opcode
        );

        // The two that reach their join through a kind. A kind is only produced
        // for the right class, so the check is that no other class produces one
        // — and, where one is produced, that the join takes the packet.
        match LifecycleKind::of(p.channel, p.opcode) {
            Some(kind) => {
                assert!(is(PayloadClass::ResourceLifecycle));
                let op = reims_vgpu_core::lifecycle::operation(
                    kind,
                    &payload,
                    &Everything,
                    &EveryMapping,
                );
                // Either an operation, or a named reason the operation is not
                // in the packet. Never "this is not my command".
                assert!(
                    !matches!(
                        op,
                        Err(reims_vgpu_core::lifecycle::ResolveRefusal::NotAResourceList { .. }
                            | reims_vgpu_core::lifecycle::ResolveRefusal::NotAnObjectReference { .. }
                            | reims_vgpu_core::lifecycle::ResolveRefusal::NotABackingRetirement { .. }
                            | reims_vgpu_core::lifecycle::ResolveRefusal::NotAMapNotice { .. })
                    ),
                    "{} {:#04x} reached a join that does not own it: {op:?}",
                    p.channel.name(),
                    p.opcode
                );
            }
            None => assert!(!is(PayloadClass::ResourceLifecycle)),
        }

        match QueryKind::of(p.channel, p.opcode) {
            Some(kind) => {
                assert!(is(PayloadClass::Query));
                // A zero payload is long enough for the two info layouts and is
                // not a heap-texture request, whose record has to frame. Both
                // outcomes are the question's own, which is all this claims.
                let _ = reims_vgpu_core::query::request_words(kind, &payload);
            }
            None => assert!(!is(PayloadClass::Query)),
        }
    }
}

/// A resolver that answers about every object and every mapping, so a refusal
/// here is about the command and never about what happens to be live.
struct Everything;

impl reims_vgpu_core::resolve::RefResolver for Everything {
    fn resource(&self, object_ref: u32) -> Option<reims_vgpu_core::identity::ResourceId> {
        Some(reims_vgpu_core::identity::ResourceId {
            slot: reims_vgpu_core::identity::ObjectListRef(object_ref),
            generation: reims_vgpu_core::identity::SlotGeneration(1),
        })
    }
}

// One namespace for every task, which this suite is entitled to: it asks which
// join a kind reaches, not which task's slots a ref is in.
impl reims_vgpu_core::resolve::TaskNamespaces for Everything {
    fn resource(
        &self,
        _task: reims_vgpu_core::identity::TaskId,
        object_ref: u32,
    ) -> Option<reims_vgpu_core::identity::ResourceId> {
        reims_vgpu_core::resolve::RefResolver::resource(self, object_ref)
    }
}

struct EveryMapping;

impl reims_vgpu_core::resolve::MappingResolver for EveryMapping {
    fn backing(
        &self,
        mapping: reims_vgpu_core::identity::MappingId,
    ) -> Option<reims_vgpu_core::access::BackingId> {
        Some(reims_vgpu_core::access::BackingId(u64::from(mapping.0)))
    }
}
