//! Seam 2's exit: serial and parallel schedules mean the same thing.
//!
//! The sweep below is only worth anything if the schedules it produces
//! actually differ, so that is asserted first and separately. A property test
//! whose generator happens to produce one order is a test that passes for the
//! wrong reason, and it passes just as happily after the property breaks.

use super::*;
use crate::access::{AccessIntent, AccessKey, AccessMode, ByteRange, ResourceKey, StubRegistry};
use crate::exec::{ExecBuilder, ResolvedOperation};
use crate::identity::{
    ChannelId, ChannelSequence, CompletionStamp, ObjectListRef, SlotGeneration, StampWait,
    TransactionIdentity,
};
use crate::prereq::Diagnosis;
use crate::stream::{SegmentKind, SegmentLifetime};
use crate::sync::{EventKind, EventOp, FenceKind, FenceOp};
use crate::transaction::{DeviceTransaction, Payload, PayloadClass};

fn res(slot: u32) -> ResourceId {
    ResourceId {
        slot: ObjectListRef(slot),
        generation: SlotGeneration(1),
    }
}

fn builder(domain: u32, ingress: u64) -> crate::testing::At<'static> {
    crate::testing::At::new(domain, ingress)
}

/// A whole-backing access. Writes are always whole here, which is what keeps
/// two publishers of one backing hazard-ordered — see
/// [`Ineligible::UnorderedVersionRace`].
fn whole(domain: u32, backing: u64, mode: AccessMode) -> AccessIntent {
    AccessIntent {
        domain: ChannelId(domain),
        key: AccessKey::Whole(ResourceKey {
            backing: BackingId(backing),
            heap: None,
        }),
        mode,
        api_stages: 0,
        input_content_version: None,
        output_content_version: None,
    }
}

/// A whole-backing write that also claims the region's next content version.
fn produces(domain: u32, backing: u64, to: u64) -> AccessIntent {
    AccessIntent {
        output_content_version: Some(ContentVersion(to)),
        ..whole(domain, backing, AccessMode::Write)
    }
}

fn ranged(domain: u32, backing: u64, offset: u64) -> AccessIntent {
    AccessIntent {
        domain: ChannelId(domain),
        key: AccessKey::Range(
            ResourceKey {
                backing: BackingId(backing),
                heap: None,
            },
            ByteRange { offset, length: 64 },
        ),
        mode: AccessMode::Read,
        api_stages: 0,
        input_content_version: None,
        output_content_version: None,
    }
}

// ---------------------------------------------------------------- workloads

/// A batch that is a straight hazard chain: every transaction writes the same
/// backing, so the dependency graph totally orders them.
fn chain(length: u64) -> Vec<DeviceTransaction> {
    (1..=length)
        .map(|n| {
            let mut b = builder(1, n);
            b.declare_access(produces(1, 1, n));
            b.publish_stamp(CompletionStamp {
                slot: StampSlot(1),
                value: StampValue(u32::try_from(n).expect("small")),
            });
            b.finish().expect("frozen")
        })
        .collect()
}

/// A batch of transactions that touch nothing in common, so the dependency
/// graph orders none of them.
fn independent(count: u64) -> Vec<DeviceTransaction> {
    (1..=count)
        .map(|n| {
            let mut b = builder(1, n);
            b.declare_access(produces(1, n, 1));
            b.finish().expect("frozen")
        })
        .collect()
}

/// A mixed workload: several domains, shared and private backings, reads and
/// writes, event signals answered by later waits, fence updates, and a
/// completion stamp per domain.
///
/// Backings are partitioned by domain because [`crate::access::requires_edge`]
/// refuses to order accesses in different domains — which is correct, and
/// which means a version reservation on a backing two domains write would have
/// no legal order at all.
fn mixed(seed: u64, count: u64) -> Vec<DeviceTransaction> {
    const DOMAINS: u64 = 3;
    let mut rng = Rng::new(seed ^ 0xC0FF_EE00);
    let mut batch = Vec::new();
    // Highest value signalled into each event so far, so a wait can only name
    // a point an earlier transaction already produced.
    let mut signalled: Vec<(ResourceId, u64)> = Vec::new();
    // Next content version per backing, and the next stamp value per domain.
    let mut version = [0u64; 12];
    let mut stamp = [0u32; DOMAINS as usize];

    for n in 1..=count {
        let domain = u32::try_from(n % DOMAINS).expect("small") + 1;
        let mut b = builder(domain, n);

        // Zero to two reads of backings this domain owns.
        for _ in 0..(rng.next() % 3) {
            let backing = (u64::from(domain) * 4) + (rng.next() % 4);
            b.declare_access(ranged(domain, backing, rng.next() % 4 * 64));
        }
        // One write, sometimes, claiming the region's next content version.
        if !rng.next().is_multiple_of(3) {
            let backing = (u64::from(domain) * 4) + (rng.next() % 4);
            let slot = usize::try_from(backing).expect("small") % version.len();
            version[slot] += 1;
            b.declare_access(produces(domain, backing, version[slot]));
        }
        // A wait for a point some earlier transaction has already signalled.
        if !signalled.is_empty() && rng.next().is_multiple_of(3) {
            let (event, value) = signalled[rng.below(signalled.len())];
            b.require(Prerequisite::Event { event, value });
        }
        // A wait for a stamp value some earlier transaction already owes.
        if rng.next().is_multiple_of(4) {
            let owed = stamp[usize::try_from(domain).expect("small") - 1];
            if owed > 0 {
                b.wait_for(StampWait {
                    slot: StampSlot(domain),
                    value: StampValue(owed),
                });
            }
        }
        // Records: an event signal, or a fence update on a blit encoder.
        match rng.next() % 3 {
            0 => {
                let event = res(u32::try_from(rng.next() % 3).expect("small") + 20);
                let at = signalled
                    .iter()
                    .filter(|(e, _)| *e == event)
                    .map(|(_, v)| *v)
                    .max()
                    .unwrap_or(0);
                b.begin_segment(
                    SegmentKind::Event.wire_type(),
                    SegmentLifetime::SELF_CONTAINED,
                )
                .expect("event encoder opens");
                b.record(
                    ResolvedOperation::Event(EventOp {
                        kind: EventKind::Signal,
                        event,
                        value: at + 1,
                    }),
                    &mut StubRegistry(ChannelId(domain)),
                )
                .expect("a signal records");
                b.end_segment().expect("event encoder closes");
                signalled.push((event, at + 1));
            }
            1 => {
                b.begin_segment(
                    SegmentKind::Blit.wire_type(),
                    SegmentLifetime::SELF_CONTAINED,
                )
                .expect("blit encoder opens");
                b.record(
                    ResolvedOperation::Fence(FenceOp {
                        kind: FenceKind::Update,
                        fence: res(u32::try_from(rng.next() % 2).expect("small") + 30),
                        stages: None,
                    }),
                    &mut StubRegistry(ChannelId(domain)),
                )
                .expect("a fence update records");
                b.end_segment().expect("blit encoder closes");
            }
            _ => {}
        }
        // A completion stamp, on the domain's own slot, advancing.
        let slot = usize::try_from(domain).expect("small") - 1;
        stamp[slot] += 1;
        b.publish_stamp(CompletionStamp {
            slot: StampSlot(domain),
            value: StampValue(stamp[slot]),
        });
        batch.push(b.finish().expect("frozen"));
    }
    batch
}

/// A batch whose accesses came from Apple's record shapes rather than from this
/// file.
///
/// Every workload above states its accesses with [`ExecBuilder::declare_access`]
/// — which is the right way to reach an access shape the registry would not
/// produce, and the wrong way to answer "does the batch a guest actually sends
/// still schedule the same". This one answers that: bytes go through
/// [`crate::walk::exec`], each record states its own participation, and the
/// hazard graph is built from whatever comes out. Nothing here names an access.
///
/// The refs interleave on purpose. Transaction `n` touches ref `n % 3` and ref
/// `n % 2`, so the batch is neither a chain nor independent: some pairs
/// conflict and some do not, which is the shape a schedule can reorder without
/// being free to reorder everything.
///
/// One domain. Two channels writing memory they share is
/// [`Ineligible::UnorderedVersionRace`], and what this workload adds is the
/// records rather than the domain split the ones above already cover.
fn from_records(count: u64) -> Vec<DeviceTransaction> {
    const TASK: crate::identity::TaskId = crate::identity::TaskId(1);
    const SLOTS: &[u32] = &[1, 2, 3, 10, 11];
    // One registry for the whole batch, which is the point: content versions
    // are the session's, so transaction `n`'s write reserves the version after
    // whatever `n - 1` reserved. A registry per transaction would hand every
    // one of them version 1 and leave the trace with nothing to disagree about.
    let mut model = crate::testing::registry(TASK, SLOTS);
    (1..=count)
        .map(|n| {
            let bytes = crate::testing::blit_stream(&[
                u32::try_from(n % 3).expect("small") + 1,
                u32::try_from(n % 2).expect("small") + 10,
            ]);
            let work = crate::walk::exec(
                &bytes,
                &crate::testing::Everything,
                &mut model.task_access(TASK, ChannelId(1)),
                crate::exec::ExecBuilder::new(),
            )
            .expect("a stream of records the ledger has judged");
            DeviceTransaction {
                identity: crate::testing::identity(1, n),
                stamp_waits: Vec::new(),
                completion: Some(CompletionStamp {
                    slot: StampSlot(1),
                    value: StampValue(u32::try_from(n).expect("small")),
                }),
                payload: Payload::Exec(work),
            }
        })
        .collect()
}

/// The same, from render streams whose accesses are all
/// [`AccessMode::Unknown`].
///
/// A different access shape from every other workload in this file, and the one
/// a guest's render encoder actually produces. A bound slot contributes
/// `Unknown` until a pipeline publishes what its shader does with it, and
/// `Unknown` is the only mode that conflicts with a *reader* — so a batch of
/// these compiles edges no `Read`/`Write` workload can, and the reference
/// interpreter has to mean the same thing about them.
fn from_render_records(count: u64) -> Vec<DeviceTransaction> {
    const TASK: crate::identity::TaskId = crate::identity::TaskId(1);
    const SLOTS: &[u32] = &[1, 2, 3, 10, 11];
    let mut model = crate::testing::registry(TASK, SLOTS);
    (1..=count)
        .map(|n| {
            let bytes = crate::testing::render_stream(&[
                u32::try_from(n % 3).expect("small") + 1,
                u32::try_from(n % 2).expect("small") + 10,
            ]);
            let work = crate::walk::exec(
                &bytes,
                &crate::testing::Everything,
                &mut model.task_access(TASK, ChannelId(1)),
                crate::exec::ExecBuilder::new(),
            )
            .expect("a stream of records the ledger has judged");
            DeviceTransaction {
                identity: crate::testing::identity(1, n),
                stamp_waits: Vec::new(),
                completion: Some(CompletionStamp {
                    slot: StampSlot(1),
                    value: StampValue(u32::try_from(n).expect("small")),
                }),
                payload: Payload::Exec(work),
            }
        })
        .collect()
}

// ------------------------------------------------------------- the sweep

/// The sweep is only meaningful if the seeds actually reach different
/// schedules, so that is its own assertion rather than a hope.
#[test]
fn independent_transactions_reach_many_completion_orders() {
    let batch = independent(6);
    eligible(&batch).expect("independent work is eligible");
    let orders: std::collections::BTreeSet<Vec<IngressOrdinal>> = (0..64u64)
        .map(|seed| parallel(&batch, seed).order())
        .collect();
    assert!(
        orders.len() > 8,
        "the seeds reached {} distinct orders of six independent transactions; \
         a sweep that only finds one is not sweeping",
        orders.len()
    );
}

/// A batch whose classes are mixed: EXECs, a lifecycle synchronise that
/// produces content, and control commands that carry only their envelopes.
///
/// The scheduler used to take `ExecTransaction`, so a batch could contain
/// nothing else — and a device whose channel carries a delete between two draws
/// had no reference to be checked against for that sequence at all. Every
/// transaction here writes the same backing, so the dependency graph totally
/// orders the ones that touch memory and the control commands float.
fn mixed_classes() -> Vec<DeviceTransaction> {
    let mut batch = Vec::new();
    for n in 1..=7u64 {
        let identity = crate::testing::identity(1, n);
        let completion = Some(CompletionStamp {
            slot: StampSlot(1),
            value: StampValue(u32::try_from(n).expect("small")),
        });
        let accesses = vec![produces(1, 1, n)];
        batch.push(match n % 5 {
            0 => DeviceTransaction {
                identity,
                stamp_waits: Vec::new(),
                completion,
                payload: Payload::Control(crate::control::ControlOp::Inert {
                    kind: crate::control::ControlKind::Nop,
                }),
            },
            1 => {
                let mut b = ExecBuilder::new();
                b.declare_access(accesses[0]);
                DeviceTransaction {
                    identity,
                    stamp_waits: Vec::new(),
                    completion,
                    payload: Payload::Exec(b.finish().expect("frozen")),
                }
            }
            2 => DeviceTransaction {
                identity,
                stamp_waits: Vec::new(),
                completion,
                payload: Payload::ResourceLifecycle(
                    crate::transaction::LifecyclePayload::new(
                        crate::lifecycle::LifecycleOp::Synchronize {
                            task: crate::identity::TaskId(1),
                            resources: vec![res(1)],
                        },
                        accesses.into_iter().map(|a| (res(1), a)).collect(),
                    )
                    .expect("every access is for the one resource the op names"),
                ),
            },
            // A frame. Its accesses are reads, so it publishes no version and
            // its whole contribution to the trace is which mapping it showed.
            //
            // **One present, deliberately.** Two presents on one domain read
            // different surfaces and two reads compile no hazard edge, so
            // nothing in this plane orders them — and the device's answer to
            // that is `PresentStream`'s, which refuses an out-of-order queue
            // rather than ordering the transactions. The interpreter has no
            // stream (an image count is a host fact a guest cannot observe), so
            // a batch with two presents reports a `Divergence::Presentation`
            // that is this reference's silence rather than a scheduling defect.
            // Recorded here rather than asserted, because asserting either
            // answer would be choosing one.
            3 => {
                let packet = crate::present::PresentPacket {
                    form: crate::present::PresentForm::SwapMapping,
                    mapping: crate::identity::MappingId(u32::try_from(n).expect("small")),
                    task: None,
                };
                DeviceTransaction {
                    identity,
                    stamp_waits: Vec::new(),
                    completion,
                    payload: Payload::Present(
                        crate::transaction::PresentPayload::new(
                            packet,
                            vec![(packet.mapping, whole(1, 3, AccessMode::Read))],
                        )
                        .expect("one read of the packet's own target"),
                    ),
                }
            }
            // A question. This interpreter has no answer source, so every one
            // of them stalls — which is the observation under test: a stalled
            // query publishes its completion word and no content, and a
            // schedule that lost either would be a guest reading a destination
            // nothing wrote or blocking on a word that never came.
            _ => {
                let request = crate::query::resolve(
                    crate::query::QueryKind::HeapTextureSizeAndAlign,
                    crate::query::RequestWords::HeapTexture,
                    crate::query::ReplyDestination {
                        backing: BackingId(4),
                        bytes: ByteRange {
                            offset: 0,
                            length: 4096,
                        },
                    },
                )
                .expect("its own layout");
                DeviceTransaction {
                    identity,
                    stamp_waits: Vec::new(),
                    completion,
                    payload: Payload::Query(crate::transaction::QueryPayload::new(
                        request,
                        ChannelId(1),
                        Some(ContentVersion(n)),
                    )),
                }
            }
        });
    }
    batch
}

/// A completion word is the envelope's, so the packet that publishes one an
/// EXEC waits for need not be an EXEC.
///
/// The wait graph used to see only EXECs, which made this batch look like a
/// wait nothing answers — and `eligible` would have refused a schedule the
/// guest is entitled to.
#[test]
fn a_control_command_can_answer_an_execs_stamp_wait() {
    let producer = DeviceTransaction {
        identity: crate::testing::identity(1, 1),
        stamp_waits: Vec::new(),
        completion: Some(CompletionStamp {
            slot: StampSlot(1),
            value: StampValue(5),
        }),
        payload: Payload::Control(crate::control::ControlOp::Inert {
            kind: crate::control::ControlKind::Nop,
        }),
    };
    let mut b = builder(1, 2);
    b.wait_for(StampWait {
        slot: StampSlot(1),
        value: StampValue(5),
    });
    let waiter = b.finish().expect("frozen");
    let batch = vec![producer, waiter];
    eligible(&batch).expect("the control command answers the wait");

    // And the reference runs both, in order, with the stamp published before
    // the waiter is allowed to proceed.
    let run = serial(&batch);
    assert_eq!(run.order(), vec![IngressOrdinal(1), IngressOrdinal(2)]);
    assert!(run.stalled.is_empty());
}

/// The claim the envelope exists to make: ordering and publication are owed to
/// every class equally, so a batch that mixes them schedules the same however
/// it is run.
#[test]
fn a_batch_of_mixed_classes_schedules_equivalently() {
    let batch = mixed_classes();
    // All five classes, because the claim is about the envelope and an envelope
    // that only two classes carry is not one.
    let mut classes: Vec<PayloadClass> = batch.iter().map(DeviceTransaction::class).collect();
    classes.sort_unstable();
    classes.dedup();
    assert_eq!(
        classes,
        vec![
            PayloadClass::Exec,
            PayloadClass::ResourceLifecycle,
            PayloadClass::Query,
            PayloadClass::Present,
            PayloadClass::Control,
        ],
        "the workload has to actually mix classes"
    );
    assert!(
        batch
            .iter()
            .any(|tx| tx.class() == PayloadClass::Control && tx.accesses().is_empty()),
        "and a control command that touches nothing has to be in it"
    );
    eligible(&batch).expect("a mixed batch is eligible");
    let reference = serial(&batch);
    // Each of the three observation kinds only these classes can produce has to
    // be in the reference trace, or the comparison below is checking stamps.
    for (kind, present) in [
        (
            "a version published",
            reference
                .trace
                .iter()
                .any(|o| matches!(o, crate::interpret::Observation::VersionPublished { .. })),
        ),
        (
            "a frame presented",
            reference
                .trace
                .iter()
                .any(|o| matches!(o, crate::interpret::Observation::FramePresented { .. })),
        ),
        (
            "a query left unanswered",
            reference
                .trace
                .iter()
                .any(|o| matches!(o, crate::interpret::Observation::QueryUnanswered { .. })),
        ),
    ] {
        assert!(present, "the reference trace has no {kind}");
    }
    // And the comparison actually reads them. The assertion above says the
    // reference trace *contains* a stalled query; that a stall reaches
    // `equivalent` at all is a separate fact, and it was not true --- the
    // summary collected the stalls and nothing compared them, so this sweep
    // looked like it covered queries while proving nothing about them.
    let mut answered = reference.clone();
    answered
        .trace
        .retain(|o| !matches!(o, crate::interpret::Observation::QueryUnanswered { .. }));
    assert_eq!(
        equivalent(&reference, &answered).expect_err("a stall that vanished is a divergence"),
        Divergence::QueriesUnanswered {
            serial: Summary::of(&reference.trace).unanswered,
            parallel: Vec::new(),
        },
        "a schedule that answered a query the serial run stalled reads the same \
         everywhere else: same versions, same stamps, same releases"
    );

    // The reference is checked for the two structural properties too, and not
    // only the schedules compared against it. A stamp that went backwards is
    // wrong however the work ran, and `Summary` records where a monotone point
    // came to rest --- so a regressing reference leaves no trace in the outcome
    // comparison and would have been declared equivalent to a correct
    // schedule. Republishing a value the slot already reached is the smallest
    // regression there is.
    let (slot, value) = reference
        .trace
        .iter()
        .find_map(|o| match *o {
            crate::interpret::Observation::StampPublished { slot, value } => Some((slot, value)),
            _ => None,
        })
        .expect("the reference publishes a stamp");
    let mut regressed = reference.clone();
    regressed
        .trace
        .push(crate::interpret::Observation::StampPublished { slot, value });
    assert!(
        matches!(
            equivalent(&regressed, &reference).expect_err("a regressing reference is a divergence"),
            Divergence::NonMonotonePublication { .. }
        ),
        "a reference that republished a value it had reached was accepted"
    );

    let mut orders = std::collections::BTreeSet::new();
    for seed in 0..64u64 {
        let run = parallel(&batch, seed);
        assert!(run.stalled.is_empty(), "seed {seed} stalled");
        equivalent(&reference, &run).unwrap_or_else(|d| {
            panic!("seed {seed} gave a mixed batch a different meaning: {d:?}")
        });
        orders.insert(run.order());
    }
    assert!(
        orders.len() > 1,
        "the control commands touch nothing, so some seed has to run one out of \
         ingress order — a sweep that finds one order proves nothing"
    );
}

/// And a chain must reach exactly one, or the dependency graph is not ordering
/// what it claims to.
#[test]
fn a_hazard_chain_reaches_exactly_one_order_and_an_identical_trace() {
    let batch = chain(6);
    eligible(&batch).expect("a chain is eligible");
    let reference = serial(&batch);
    for seed in 0..64u64 {
        let run = parallel(&batch, seed);
        assert_eq!(
            run.order(),
            (1..=6).map(IngressOrdinal).collect::<Vec<_>>(),
            "seed {seed} reordered a hazard chain"
        );
        assert_eq!(
            run.trace, reference.trace,
            "a totally ordered batch has one trace, not an equivalent one"
        );
    }
}

/// Seam 2's exit.
#[test]
fn every_permitted_schedule_means_what_the_serial_one_meant() {
    let mut reordered = 0usize;
    for workload in 0..24u64 {
        let batch = mixed(workload, 14);
        eligible(&batch).unwrap_or_else(|e| panic!("workload {workload} is ineligible: {e:?}"));
        let reference = serial(&batch);
        let mut orders = std::collections::BTreeSet::new();
        for seed in 0..24u64 {
            let run = parallel(&batch, seed);
            assert!(
                run.stalled.is_empty(),
                "workload {workload} seed {seed} stalled at {:?}",
                run.stalled
            );
            assert_eq!(
                run.order().len(),
                batch.len(),
                "workload {workload} seed {seed} left work unrun"
            );
            equivalent(&reference, &run).unwrap_or_else(|d| {
                panic!("workload {workload} seed {seed} diverged: {d:?}");
            });
            orders.insert(run.order());
        }
        if orders.len() > 1 {
            reordered += 1;
        }
        assert_eq!(
            parallel_with(&batch, |_| 0).order(),
            reference.order(),
            "workload {workload}: taking the lowest ready ordinal every time \
             must reproduce ingress order, or the readiness service is \
             withholding a transaction the serial run could execute"
        );
    }
    assert!(
        reordered >= 20,
        "only {reordered} of 24 workloads were reordered at all; a sweep over \
         schedules that are all the same schedule proves nothing"
    );
}

/// Seam 2's exit, over a batch built from command-stream bytes.
///
/// The sweep above proves the property over access sets this file states. This
/// one proves it over access sets the *records* state: the same batch a guest
/// would send, walked into transactions by the same path production will use,
/// with every hazard edge derived from a participation rather than declared.
///
/// A schedule that reordered these would be the first evidence that the
/// derivation and the declaration disagree about what a record touches — and
/// the declaration is the one with a test, which is exactly why it cannot be
/// the only input.
#[test]
fn a_batch_built_from_records_schedules_the_way_a_declared_one_does() {
    let batch = from_records(12);
    eligible(&batch).expect("one domain, judged records");
    // The point of the workload: every transaction's accesses came from its
    // records, and there are some.
    assert!(
        batch.iter().all(|tx| !tx.accesses().is_empty()),
        "a record named a resource and the transaction carries no access for it"
    );
    assert!(
        batch
            .iter()
            .all(|tx| tx.exec().expect("an EXEC").record_count() == 2),
        "the walk lost a record"
    );
    // And the registry gave them versions, which is what puts anything in the
    // trace for two orders to disagree about. A source that returned none would
    // leave a trace of stamps alone, and stamps publish in channel order under
    // every schedule — so the sweep would pass without testing the accesses.
    assert!(
        batch
            .iter()
            .any(|tx| crate::exec::published_versions(tx.accesses())
                .next()
                .is_some()),
        "no transaction publishes a content version; the trace has nothing in \
         it that a reordering could move"
    );

    let reference = serial(&batch);
    let mut orders = std::collections::BTreeSet::new();
    for seed in 0..32u64 {
        let run = parallel(&batch, seed);
        assert!(
            run.stalled.is_empty(),
            "seed {seed} stalled at {:?}",
            run.stalled
        );
        assert_eq!(
            run.order().len(),
            batch.len(),
            "seed {seed} left work unrun"
        );
        equivalent(&reference, &run).unwrap_or_else(|d| panic!("seed {seed} diverged: {d:?}"));
        orders.insert(run.order());
    }
    // The derived hazard edges must order *something* and not everything: a
    // batch that reached one order would prove the accesses were too coarse,
    // and one that reached every order would prove they were absent.
    assert!(
        orders.len() > 1,
        "the derived accesses admitted exactly one schedule, so the sweep \
         proved nothing about them"
    );
    assert_eq!(
        parallel_with(&batch, |_| 0).order(),
        reference.order(),
        "taking the lowest ready ordinal every time must reproduce ingress order"
    );
}

/// Seam 2's exit over the access shape a render encoder produces.
///
/// Every other workload in this file declares `Read` or `Write`. A guest's
/// draw declares neither: what a bound slot contributes is the pipeline's
/// answer, nothing has published one, and the honest answer until then is
/// `Unknown` — the one mode that conflicts with a reader as well as a writer.
///
/// So this batch exercises edges the rest of the sweep cannot reach, and it
/// exercises them through the whole path: framed bytes, decoded records, the
/// encoder's binding table, and participations placed by the registry that owns
/// the names.
#[test]
fn a_batch_of_draws_schedules_the_way_a_declared_one_does() {
    let batch = from_render_records(12);
    eligible(&batch).expect("one domain, judged records");
    // The workload is only this workload if the accesses are the encoder's.
    assert!(
        batch.iter().all(|tx| !tx.accesses().is_empty()
            && tx.accesses().iter().all(|a| a.mode == AccessMode::Unknown)),
        "a draw's declared accesses are its bound slots, at Unknown"
    );
    assert!(
        batch
            .iter()
            .any(|tx| crate::exec::published_versions(tx.accesses())
                .next()
                .is_some()),
        "Unknown writes, so a draw publishes a version; without one the trace          has nothing a reordering could move"
    );

    let reference = serial(&batch);
    let mut orders = std::collections::BTreeSet::new();
    for seed in 0..32u64 {
        let run = parallel(&batch, seed);
        assert!(
            run.stalled.is_empty(),
            "seed {seed} stalled at {:?}",
            run.stalled
        );
        assert_eq!(
            run.order().len(),
            batch.len(),
            "seed {seed} left work unrun"
        );
        equivalent(&reference, &run).unwrap_or_else(|d| panic!("seed {seed} diverged: {d:?}"));
        orders.insert(run.order());
    }
    assert!(
        orders.len() > 1,
        "the derived accesses admitted exactly one schedule, so the sweep          proved nothing about them"
    );
    assert_eq!(
        parallel_with(&batch, |_| 0).order(),
        reference.order(),
        "taking the lowest ready ordinal every time must reproduce ingress order"
    );
}

/// The claim ordered publication adds to the exit: however the work finishes,
/// each channel tells the guest about it in channel order.
#[test]
fn a_channel_publishes_in_its_own_order_however_the_schedule_runs() {
    let mut ever_held = false;
    for workload in 0..24u64 {
        let batch = mixed(workload, 14);
        let reference = serial(&batch);
        for seed in 0..24u64 {
            let run = parallel(&batch, seed);
            for domain in reference.domains() {
                assert_eq!(
                    run.published_by(domain),
                    reference.published_by(domain),
                    "workload {workload} seed {seed} published channel {domain:?} differently"
                );
            }
            ever_held |= run.blocked.iter().any(|(_, held)| *held > 0);
        }
    }
    assert!(
        ever_held,
        "no schedule ever finished work ahead of its channel's head, so the \
         FIFO was never asked to hold anything and this proves nothing"
    );
}

/// And a schedule that finishes in channel order costs the FIFO nothing.
#[test]
fn a_hazard_chain_never_holds_a_position() {
    let batch = chain(6);
    let run = parallel_with(&batch, |_| 0);
    assert!(run.blocked.is_empty());
}

#[test]
fn the_equivalence_relation_rejects_a_channel_that_published_out_of_order() {
    let batch = chain(3);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    broken.releases.swap(0, 2);
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::PublicationOrder { .. })
    ));
}

/// The compiler's cost is proportional to what overlaps, and the census is how
/// that is checked rather than asserted. Independent work compiles no edges.
#[test]
fn independent_work_compiles_no_hazard_edges() {
    let batch = independent(8);
    let run = parallel(&batch, 0);
    assert_eq!(run.census.edges, 0);
    assert_eq!(run.census.accesses, 8);
    assert_eq!(
        run.census.domain_only_comparisons, 0,
        "every access named a backing, so none of them met the domain"
    );
}

// -------------------------------------------------------------- eligibility

#[test]
fn a_wait_for_a_later_packets_stamp_has_no_serial_meaning() {
    let mut waiter = builder(1, 1);
    waiter.wait_for(StampWait {
        slot: StampSlot(1),
        value: StampValue(5),
    });
    let waiter = waiter.finish().expect("frozen");
    let mut producer = builder(1, 2);
    producer.publish_stamp(CompletionStamp {
        slot: StampSlot(1),
        value: StampValue(5),
    });
    let producer = producer.finish().expect("frozen");
    assert_eq!(
        eligible(&[waiter, producer]),
        Err(Ineligible::ForwardExplicitWait {
            waiter: IngressOrdinal(1),
            point: WaitPoint::Stamp {
                slot: StampSlot(1),
                value: StampValue(5)
            },
            producer: IngressOrdinal(2),
        })
    );
}

#[test]
fn a_wait_nothing_produces_has_no_serial_meaning() {
    let mut waiter = builder(1, 1);
    waiter.require(Prerequisite::Event {
        event: res(7),
        value: 3,
    });
    let batch = [waiter.finish().expect("frozen")];
    assert_eq!(
        eligible(&batch),
        Err(Ineligible::UnansweredWait {
            waiter: IngressOrdinal(1),
            point: WaitPoint::Event {
                event: res(7),
                value: 3
            },
        })
    );
    // And the wait graph says the same thing, because there is one answer to
    // this question and two callers of it.
    let mut graph = WaitGraph::new();
    graph.admit(&batch[0]);
    assert!(matches!(
        graph.diagnose().as_slice(),
        [Diagnosis::Unproduced { .. }]
    ));
}

#[test]
fn an_encoder_scoped_fence_prerequisite_is_outside_the_comparison() {
    let mut b = builder(1, 1);
    b.require(Prerequisite::Fence { fence: res(3) });
    assert_eq!(
        eligible(&[b.finish().expect("frozen")]),
        Err(Ineligible::FencePrerequisite {
            waiter: IngressOrdinal(1)
        })
    );
}

#[test]
fn transactions_out_of_ingress_order_are_refused_before_anything_else() {
    let batch = vec![
        builder(1, 5).finish().expect("frozen"),
        builder(1, 2).finish().expect("frozen"),
    ];
    assert_eq!(
        eligible(&batch),
        Err(Ineligible::OutOfIngressOrder {
            at: IngressOrdinal(2),
            after: IngressOrdinal(5),
        })
    );
}

/// `eligible` says ingress order is a legal schedule for the batch. It does not
/// say the interpreter will accept every transaction in it — nothing hands it
/// the interpreter's generation to compare against — so a refusal is a state
/// both runs reach.
///
/// `serial` withdraws the refused position. `parallel_with` published it, and
/// a completion word for work the model refused is the one thing publication
/// exists to withhold: the guest is told its transaction finished. The
/// equivalence relation caught the divergence only because both runs are
/// compared; a production executor built on the parallel path alone would have
/// had nothing to compare with.
#[test]
fn a_refused_transaction_publishes_nothing_however_the_schedule_runs() {
    let mut b = ExecBuilder::new();
    b.declare_access(whole(1, 1, AccessMode::Read));
    let batch = vec![DeviceTransaction {
        identity: TransactionIdentity {
            // Not the generation `Interpreter::new` starts at, which is what
            // makes the interpreter refuse. One transaction, so the batch is
            // not a mixed one and `eligible` has nothing to say about it.
            session: SessionGeneration::FIRST.next(),
            domain: ChannelId(1),
            domain_sequence: ChannelSequence(1),
            ingress: IngressOrdinal(1),
        },
        stamp_waits: Vec::new(),
        completion: Some(CompletionStamp {
            slot: StampSlot(1),
            value: StampValue(1),
        }),
        payload: Payload::Exec(b.finish().expect("frozen")),
    }];
    eligible(&batch).expect("one transaction, in ingress order, waiting on nothing");

    let serial_run = serial(&batch);
    assert!(
        serial_run.releases.is_empty(),
        "the reference publishes nothing for a refused transaction"
    );
    for seed in 0..8 {
        let parallel_run = parallel(&batch, seed);
        assert!(
            parallel_run.releases.is_empty(),
            "seed {seed}: a refused transaction published a completion word"
        );
        assert_eq!(
            equivalent(&serial_run, &parallel_run),
            Ok(()),
            "seed {seed}"
        );
    }
}

#[test]
fn a_second_generation_in_one_batch_is_refused() {
    let first = builder(1, 1).finish().expect("frozen");
    let mut b = ExecBuilder::new();
    b.declare_access(whole(1, 1, AccessMode::Read));
    let second = DeviceTransaction {
        identity: TransactionIdentity {
            session: SessionGeneration::FIRST.next(),
            domain: ChannelId(1),
            domain_sequence: ChannelSequence(2),
            ingress: IngressOrdinal(2),
        },
        stamp_waits: Vec::new(),
        completion: None,
        payload: Payload::Exec(b.finish().expect("frozen")),
    };
    assert_eq!(
        eligible(&[first, second]),
        Err(Ineligible::MixedGeneration {
            expected: SessionGeneration::FIRST,
            found: SessionGeneration::FIRST.next(),
        })
    );
}

/// Two writers of disjoint ranges of one backing are not a race. This is the
/// case that used to be one, because a version claim named a whole backing
/// while the access naming the bytes named a range; now the claim *is* the
/// access's region and the two histories are independent.
#[test]
fn two_publishers_of_disjoint_regions_of_one_backing_are_independent() {
    let batch: Vec<_> = [(1u64, 0u64), (2, 512)]
        .into_iter()
        .map(|(n, offset)| {
            let mut b = builder(1, n);
            b.declare_access(AccessIntent {
                mode: AccessMode::Write,
                output_content_version: Some(ContentVersion(1)),
                ..ranged(1, 1, offset)
            });
            b.finish().expect("frozen")
        })
        .collect();
    eligible(&batch).expect("disjoint regions have no shared history");
    let reference = serial(&batch);
    for seed in 0..16u64 {
        equivalent(&reference, &parallel(&batch, seed))
            .unwrap_or_else(|d| panic!("seed {seed} diverged: {d:?}"));
    }
}

/// What is left of the race, and it is real: two channels writing memory they
/// share. `requires_edge` orders nothing across domains — correctly, because
/// the guest supplied no ordering — so that region's version sequence has two
/// legal answers and this comparison declines to pick one.
#[test]
fn two_channels_writing_shared_memory_have_no_legal_version_order() {
    let batch: Vec<_> = [1u64, 2]
        .into_iter()
        .map(|n| {
            let mut b = builder(u32::try_from(n).expect("small"), n);
            b.declare_access(produces(u32::try_from(n).expect("small"), 1, 1));
            b.finish().expect("frozen")
        })
        .collect();
    assert_eq!(
        eligible(&batch),
        Err(Ineligible::UnorderedVersionRace {
            backing: BackingId(1),
            first: IngressOrdinal(1),
            second: IngressOrdinal(2),
        })
    );
}

// ------------------------------------------------- the relation itself bites

/// A relation that accepts everything proves nothing, so each arm is shown to
/// reject something.
#[test]
fn the_equivalence_relation_rejects_a_reordered_content_history() {
    let batch = chain(3);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    broken.trace.swap(0, 2);
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::ContentHistory { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_stamp_that_came_to_rest_elsewhere() {
    let batch = chain(3);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    for observation in &mut broken.trace {
        if let Observation::StampPublished { value, .. } = observation {
            *value = StampValue(value.0 + 100);
        }
    }
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::StampOutcome { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_stamp_that_goes_backwards() {
    let batch = chain(3);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    // Republish the first slot value after the last one, which is exactly what
    // a device that overwrote instead of advancing would show a guest.
    broken.trace.push(Observation::StampPublished {
        slot: StampSlot(1),
        value: StampValue(1),
    });
    let last = broken.spans.last_mut().expect("a span");
    last.1.end = broken.trace.len();
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::NonMonotonePublication { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_publication_split_by_another_transaction() {
    let batch = chain(3);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    // Two transactions' completion windows overlap: one made its versions
    // visible while another was still making its own visible.
    broken.spans[1].1.start = broken.spans[0].1.start;
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::SplitPublication { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_stamp_published_before_its_versions() {
    let batch = chain(1);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    broken.trace.swap(0, 1);
    assert!(
        matches!(
            equivalent(&reference, &broken),
            Err(Divergence::StampBeforeVersions { .. })
        ),
        "a guest that polled the stamp and then read the content must not be \
         able to see the flag without the bytes"
    );
}

#[test]
fn the_equivalence_relation_rejects_a_transaction_that_did_not_run() {
    let batch = chain(3);
    let reference = serial(&batch);
    let mut broken = reference.clone();
    broken.spans.pop();
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::DifferentTransactions { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_a_missing_fence_update() {
    let mut b = builder(1, 1);
    b.begin_segment(
        SegmentKind::Blit.wire_type(),
        SegmentLifetime::SELF_CONTAINED,
    )
    .expect("blit encoder opens");
    b.record(
        ResolvedOperation::Fence(FenceOp {
            kind: FenceKind::Update,
            fence: res(30),
            stages: None,
        }),
        &mut StubRegistry(ChannelId(1)),
    )
    .expect("a fence update records");
    b.end_segment().expect("blit encoder closes");
    let work = b.finish().expect("frozen");
    let reference = serial(&[work]);
    assert_eq!(reference.trace.len(), 1);
    let mut broken = reference.clone();
    broken.trace.clear();
    broken.spans[0].1 = 0..0;
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::FenceUpdates { .. })
    ));
}

#[test]
fn the_equivalence_relation_rejects_an_event_that_came_to_rest_elsewhere() {
    let mut b = builder(1, 1);
    b.begin_segment(
        SegmentKind::Event.wire_type(),
        SegmentLifetime::SELF_CONTAINED,
    )
    .expect("event encoder opens");
    b.record(
        ResolvedOperation::Event(EventOp {
            kind: EventKind::Signal,
            event: res(20),
            value: 4,
        }),
        &mut StubRegistry(ChannelId(1)),
    )
    .expect("a signal records");
    b.end_segment().expect("event encoder closes");
    let work = b.finish().expect("frozen");
    let reference = serial(&[work]);
    let mut broken = reference.clone();
    broken.trace[0] = Observation::EventAdvanced {
        event: res(20),
        to: 9,
    };
    assert!(matches!(
        equivalent(&reference, &broken),
        Err(Divergence::EventOutcome { .. })
    ));
}
