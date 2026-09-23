//! The deadlock class, driven: at every reachable state of a batch, something
//! can move.
//!
//! # What is being claimed
//!
//! The architecture this replaces could reach a state where transaction A held
//! ownership while waiting for B, and B needed that same ownership to record,
//! submit, complete or publish — an ownership cycle neither the Vulkan
//! dependency nor the API contract required. The replacement's answer is
//! structural: dependency compilation never waits, readiness is published
//! rather than waited on, the submission gate returns an answer instead of
//! blocking, and publication is a per-channel order rather than a lock.
//!
//! Structural claims of that shape are worth driving anyway, because the way
//! they fail is not a single wrong line but two owners that each behave
//! correctly and together admit a state where nothing is enabled. So this runs
//! whole batches through
//! [`SessionModel`](reims_vgpu_core::session::SessionModel) and
//! [`SubmitGate`](reims_vgpu_core::submit::SubmitGate) under forced adverse
//! interleavings and asserts the one property that matters:
//!
//! > **While work remains, at least one action is enabled.**
//!
//! # Why the batches cannot deadlock by construction
//!
//! The generator only writes waits that point *backwards* in ingress order, so
//! no guest-authored cycle is present — a mutual wait is
//! [`prereq::Diagnosis::Cycle`](reims_vgpu_core::prereq::Diagnosis) and a
//! contract answer rather than a device defect, and it is tested at its own
//! owner. What is under test here is whether the *device's* own service
//! handoffs can invent a cycle that the guest did not write.
//!
//! # What this cannot distinguish, and where that is tested instead
//!
//! Deadlock-freedom needs only the *weak* form of the submission rule — release
//! a waiter once its producer has finished. The rule this device actually
//! implements is the strong one, release once the producer has been *submitted*,
//! and the difference is throughput rather than progress: a driver that only
//! ever released on completion would still finish every batch here. So that
//! half of the claim lives in `submit`'s own tests, and a mutation that removes
//! it fails there and passes here. Recorded so a later reader does not take
//! this test's silence for coverage.
//!
//! # The interleavings
//!
//! Three policies, because a scheduler that only ever takes the oldest enabled
//! action is the one arrangement most likely to be accidentally safe: oldest
//! first, newest first, and pseudorandom. Each runs over the same generated
//! batches.

use reims_vgpu_core::access::{AccessIntent, AccessKey, AccessMode, BackingId, ResourceKey};
use reims_vgpu_core::exec::ExecWork;
use reims_vgpu_core::identity::{
    ChannelId, CompletionStamp, IngressOrdinal, ObjectListRef, ResourceId, SessionId,
    SlotGeneration, StampSlot, StampValue, StampWait,
};
use reims_vgpu_core::pipeline::PipelineState;
use reims_vgpu_core::session::{Packet, SessionModel};
use reims_vgpu_core::submit::{Admission, PhysicalQueue, QueueMap, SubmitGate};
use reims_vgpu_core::transaction::Payload;
use reims_vgpu_protocol::packets::Channel;
use std::collections::{BTreeMap, BTreeSet};

/// The interleaving the driver forces at every handoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    Oldest,
    Newest,
    Pseudorandom,
}

impl Policy {
    const ALL: [Policy; 3] = [Policy::Oldest, Policy::Newest, Policy::Pseudorandom];

    fn pick(self, choices: &[usize], rng: &mut Rng) -> usize {
        match self {
            Self::Oldest => choices[0],
            Self::Newest => choices[choices.len() - 1],
            Self::Pseudorandom => choices[rng.below(choices.len())],
        }
    }
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

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next() % bound as u64) as usize
    }
}

/// The identity a generated pipeline number resolves to.
fn pipeline_id(n: u32) -> ResourceId {
    ResourceId {
        slot: ObjectListRef(n),
        generation: SlotGeneration(1),
    }
}

/// One generated transaction, before it is admitted.
struct Spec {
    channel: ChannelId,
    /// The backing it touches, and how.
    backing: u64,
    mode: AccessMode,
    /// The stamp it waits for, and which earlier transaction publishes it.
    wait: Option<(StampWait, usize)>,
    /// Whether that wait is forwarded to the device as a timeline wait rather
    /// than held on the CPU.
    ///
    /// The distinction is what makes the submission gate load-bearing. A
    /// CPU-held wait is a readiness prerequisite, and readiness will not call
    /// the waiter ready until the producer has *completed* — stronger than the
    /// gate's rule, so the gate never sees it. A forwarded wait leaves the
    /// waiter submittable while its producer is still running, which is the
    /// state in which submitting it first can strand it on a shared path.
    forwarded: bool,
    completion: Option<CompletionStamp>,
    /// A pipeline this transaction binds, if it binds one.
    ///
    /// A fourth wait class, and the only one whose release does not come from
    /// another transaction: a compilation finishes on its own schedule, and the
    /// batch cannot make it happen by running anything. So a state whose only
    /// enabled action is "finish a compilation" is a state the other three wait
    /// classes cannot produce.
    pipeline: Option<u32>,
}

/// How many distinct pipelines a batch's transactions may bind.
///
/// Small, so several transactions share one and a single compilation releases
/// more than one waiter.
const PIPELINES: u32 = 3;

/// A batch whose every wait points backwards, so nothing the guest wrote is a
/// cycle.
fn batch(seed: u64, count: usize, channels: u32) -> Vec<Spec> {
    let mut rng = Rng::new(seed);
    let mut specs: Vec<Spec> = Vec::with_capacity(count);
    for i in 0..count {
        let channel = ChannelId(rng.below(channels as usize) as u32 + 1);
        // Four backings, so hazard chains form and independent work exists.
        let backing = rng.below(4) as u64;
        let mode = if rng.below(3) == 0 {
            AccessMode::Read
        } else {
            AccessMode::Write
        };
        // Every transaction publishes into its own slot, so a producer is
        // unambiguous.
        let completion = Some(CompletionStamp {
            slot: StampSlot(i as u32),
            value: StampValue(1),
        });
        let forwarded = rng.below(2) == 0;
        // Half of them wait on an earlier transaction's stamp.
        //
        // A forwarded wait deliberately names a producer this transaction has
        // no memory hazard with. A waiter that also touches the producer's
        // backing is already ordered behind it by the hazard compiler, and
        // readiness then holds it until the producer *completes* — so the
        // submission rule would never be reached and the test would be green on
        // a gate with nothing in it. The case the rule exists for is exactly a
        // cross-packet timeline wait with no hazard under it.
        let candidates: Vec<usize> = (0..i)
            .filter(|&j| !forwarded || specs[j].backing != backing)
            .collect();
        let wait = if !candidates.is_empty() && rng.below(2) == 0 {
            let producer = candidates[rng.below(candidates.len())];
            Some((
                StampWait {
                    slot: StampSlot(producer as u32),
                    value: StampValue(1),
                },
                producer,
            ))
        } else {
            None
        };
        // A third of them bind a pipeline. Not all, because a batch in which
        // every transaction waits on a compilation would make the pipeline
        // action the only one for its whole first phase and stop exercising the
        // interleaving between the other three.
        let pipeline = (rng.below(3) == 0).then(|| rng.below(PIPELINES as usize) as u32);
        specs.push(Spec {
            channel,
            backing,
            mode,
            wait,
            forwarded,
            completion,
            pipeline,
        });
    }
    specs
}

/// What driving a batch actually exercised, so the sweep can prove it was not
/// trivially green.
#[derive(Clone, Copy, Debug, Default)]
struct Exercised {
    /// Times the submission gate held a transaction.
    gate_holds: usize,
    /// States where the only enabled action was to complete something — the
    /// states in which a wrong gate or a wrong publication order would be a
    /// deadlock rather than a slowdown.
    only_completable: usize,
    /// States where a channel was holding a finished position behind an
    /// unfinished one.
    publication_blocked: usize,
    /// States where finishing a compilation was the only enabled action.
    ///
    /// The states the fourth wait class exists to reach: no transaction could
    /// be submitted and none could be completed, and the batch was not stuck —
    /// it was waiting for something no transaction produces.
    only_compilable: usize,
}

impl Exercised {
    fn absorb(&mut self, other: Exercised) {
        self.gate_holds += other.gate_holds;
        self.only_completable += other.only_completable;
        self.publication_blocked += other.publication_blocked;
        self.only_compilable += other.only_compilable;
    }
}

/// Drive one batch to completion under one policy, asserting that a state with
/// work remaining and nothing enabled never occurs.
fn drive(seed: u64, count: usize, channels: u32, queues: u32, policy: Policy) -> Exercised {
    drive_withdrawing(seed, count, channels, queues, policy, false)
}

/// The same drive, optionally taking one transaction out before the loop
/// starts.
///
/// **A withdrawal is the one action that removes work without completing it**,
/// and it is the action a pipeline that will never build or a device loss
/// forces. The transaction holds a position in the publication order, the
/// dependency graph and the readiness service, and the last of those is the
/// only thing that discharges another transaction's hazard wait — so a
/// withdrawal that released the first plane and not the third strands every
/// later transaction sharing its backing. That is a hang, it was real, and this
/// driver is where it should have shown.
///
/// The transaction taken out is chosen so that **nothing waits on its
/// completion word**. A stamp wait on a withdrawn producer can never be
/// satisfied — the work never ran and no word is published for it, which is the
/// contract — so withdrawing a producer would be a legitimate stall rather than
/// a defect, and the property under test here is the *hazard* obligation.
fn drive_withdrawing(
    seed: u64,
    count: usize,
    channels: u32,
    queues: u32,
    policy: Policy,
    withdraw_one: bool,
) -> Exercised {
    let specs = batch(seed, count, channels);
    // Nothing may wait on the one taken out; see the doc above. The last such
    // transaction, so it has predecessors to have taken hazard waits on.
    let doomed = withdraw_one
        .then(|| {
            (0..specs.len()).rfind(|i| specs.iter().all(|s| s.wait.map(|(_, p)| p) != Some(*i)))
        })
        .flatten();
    let mut model = SessionModel::new(SessionId(1));
    let mut map = QueueMap::new();
    for c in 1..=channels {
        model
            .open_channel(ChannelId(c))
            .expect("each domain is opened once");
        // Several logical channels share each physical path, which is the
        // arrangement the submission rule is about.
        map.assign(ChannelId(c), PhysicalQueue(c % queues));
    }
    let mut gate = SubmitGate::new(map);

    // Every pipeline a spec names, declared and stepped as far as the compiling
    // layer takes it on its own. The last step is the driver's action below, so
    // a transaction binding one is admitted with a wait that only that action
    // can discharge.
    let mut compiling: BTreeSet<u32> = specs.iter().filter_map(|s| s.pipeline).collect();
    let generation = model.generation();
    for id in &compiling {
        let pipeline = pipeline_id(*id);
        assert!(model.pipelines().declare(pipeline, generation));
        for step in [PipelineState::Translating, PipelineState::Compiling] {
            assert!(model.pipelines().advance(pipeline, step));
        }
    }

    let mut ordinals: Vec<IngressOrdinal> = Vec::with_capacity(count);
    let mut of_ordinal: BTreeMap<IngressOrdinal, usize> = BTreeMap::new();
    for (i, spec) in specs.iter().enumerate() {
        let packet = Packet {
            session: model.generation(),
            channel: Channel::Child,
            domain: spec.channel,
            // The EXEC: the packet class that carries accesses.
            opcode: 0x37,
            stamp_waits: spec
                .wait
                .filter(|_| !spec.forwarded)
                .map(|(w, _)| w)
                .into_iter()
                .collect(),
            completion: spec.completion,
            payload: Payload::Exec(ExecWork {
                accesses: vec![AccessIntent {
                    key: AccessKey::Whole(ResourceKey {
                        backing: BackingId(spec.backing),
                        heap: None,
                    }),
                    mode: spec.mode,
                    domain: spec.channel,
                    api_stages: 0,
                    input_content_version: None,
                    output_content_version: None,
                }],
                pipeline_leases: spec.pipeline.map(pipeline_id).into_iter().collect(),
                ..ExecWork::default()
            }),
        };
        let admitted = model.admit(&packet).expect("a well-formed EXEC");
        let ordinal = admitted.transaction.identity.ingress;
        let producers: Vec<IngressOrdinal> = spec
            .wait
            .filter(|_| spec.forwarded)
            .map(|(_, p)| ordinals[p])
            .into_iter()
            .collect();
        gate.admit(ordinal, spec.channel, &producers);
        of_ordinal.insert(ordinal, i);
        ordinals.push(ordinal);
    }

    let mut withdrawn: BTreeSet<IngressOrdinal> = BTreeSet::new();
    if let Some(i) = doomed {
        let _ = model.withdraw(ordinals[i]);
        gate.retire(ordinals[i]);
        withdrawn.insert(ordinals[i]);
    }

    let mut rng = Rng::new(seed ^ 0x5EED);
    let mut ready: BTreeSet<IngressOrdinal> = BTreeSet::new();
    let mut submitted: BTreeSet<IngressOrdinal> = BTreeSet::new();
    let mut completed: BTreeSet<IngressOrdinal> = BTreeSet::new();
    // Generous, and finite: an interleaving that needed more than this would be
    // livelock, which is the same failure as deadlock with a longer trace.
    let budget = count * count * 8 + 64;
    let mut exercised = Exercised::default();

    for step in 0..budget {
        ready.extend(model.take_ready());
        let submittable: Vec<IngressOrdinal> = ready
            .iter()
            .copied()
            .filter(|o| !submitted.contains(o))
            .filter(|o| gate.gate(*o) == Admission::Go)
            .collect();
        let completable: Vec<IngressOrdinal> = submitted.difference(&completed).copied().collect();

        if completed.len() + withdrawn.len() == count {
            assert!(
                submittable.is_empty() && completable.is_empty(),
                "everything completed and something is still enabled"
            );
            exercised.gate_holds = gate.holds();
            return exercised;
        }
        if submittable.is_empty() && !completable.is_empty() {
            exercised.only_completable += 1;
        }
        if submittable.is_empty() && completable.is_empty() && !compiling.is_empty() {
            exercised.only_compilable += 1;
        }
        if !model.publisher().blocked().is_empty() {
            exercised.publication_blocked += 1;
        }

        assert!(
            !submittable.is_empty() || !completable.is_empty() || !compiling.is_empty(),
            "seed {seed} policy {policy:?} step {step}: work remains and \
             nothing is enabled. ready={} submitted={} completed={}/{count} \
             withdrawn={} blocked={:?} gate_holds={}",
            ready.len(),
            submitted.len(),
            completed.len(),
            withdrawn.len(),
            model.publisher().blocked(),
            gate.holds()
        );

        // A compilation finishing is not something the batch produces, so it
        // is available whenever one is outstanding. Taking it only when nothing
        // else is enabled would make every state trivially progressive and stop
        // testing whether a compilation *releases* anything; taking it
        // sometimes when other actions exist is what builds the backlog.
        if !compiling.is_empty()
            && (submittable.is_empty() && completable.is_empty() || rng.below(3) == 0)
        {
            let ids: Vec<u32> = compiling.iter().copied().collect();
            let indices: Vec<usize> = (0..ids.len()).collect();
            let chosen = ids[policy.pick(&indices, &mut rng)];
            assert!(
                model.pipeline_ready(pipeline_id(chosen)),
                "a compiling pipeline may become ready"
            );
            compiling.remove(&chosen);
            continue;
        }

        // Prefer whichever the policy can act on, and when both are available
        // let the interleaving decide — a driver that always completed before
        // submitting would never build a backlog.
        let take_submit = if submittable.is_empty() {
            false
        } else if completable.is_empty() {
            true
        } else {
            rng.below(2) == 0
        };

        if take_submit {
            let indices: Vec<usize> = (0..submittable.len()).collect();
            let chosen = submittable[policy.pick(&indices, &mut rng)];
            gate.submitted(chosen);
            submitted.insert(chosen);
        } else {
            let indices: Vec<usize> = (0..completable.len()).collect();
            let chosen = completable[policy.pick(&indices, &mut rng)];
            let epoch = model.epoch();
            let _ = model
                .complete(epoch, chosen)
                .expect("no device loss in this harness");
            gate.retire(chosen);
            completed.insert(chosen);
            ready.remove(&chosen);
        }
    }

    panic!(
        "seed {seed} policy {policy:?}: {} of {count} completed ({} withdrawn) \
         within the step budget, which is livelock rather than progress",
        completed.len(),
        withdrawn.len()
    );
}

/// The claim, over a spread of shapes: batches whose channels share physical
/// paths, batches whose channels do not, and batches with one channel.
#[test]
fn a_batch_always_has_something_it_can_do() {
    let mut total = Exercised::default();
    for policy in Policy::ALL {
        for seed in 0..24u64 {
            // Several channels onto fewer physical queues: the arrangement the
            // submission rule exists for.
            total.absorb(drive(seed, 12, 4, 2, policy));
            // One physical queue for everything: every same-channel and
            // cross-channel producer shares the path.
            total.absorb(drive(seed, 10, 3, 1, policy));
            // One channel: every transaction is in one publication order.
            total.absorb(drive(seed, 8, 1, 1, policy));
            // A path per channel: the gate holds only on same-channel
            // producers.
            total.absorb(drive(seed, 10, 4, 4, policy));
        }
    }
    // Non-vacuity. A sweep in which nothing was ever held and every state had a
    // submission available would be green on an implementation with no rules in
    // it at all, so each of the three situations the property is about has to
    // have actually occurred.
    assert!(
        total.gate_holds > 0,
        "the submission gate never held anything, so the rule was never under test"
    );
    assert!(
        total.only_completable > 0,
        "no state had completion as its only enabled action, so a wrong \
         publication order would have shown as a slowdown rather than a stop"
    );
    assert!(
        total.publication_blocked > 0,
        "no channel ever held a finished position behind an unfinished one"
    );
    assert!(
        total.only_compilable > 0,
        "no state had a compilation as its only enabled action, so a          `pipeline_ready` that released nothing would have passed"
    );
}

/// The same claim, with one transaction withdrawn before anything runs.
///
/// A withdrawal is the only action that removes work without completing it, and
/// it is what a pipeline that will never build and a device loss both force. The
/// withdrawn transaction was holding a position in three planes, and only one of
/// them is the channel's publication order — the other two order later work, and
/// leaving either behind strands every transaction that shares its backing.
///
/// The batches are the same shapes as the sweep above, so the difference between
/// the two tests is the withdrawal and nothing else.
#[test]
fn a_batch_with_a_withdrawn_transaction_still_finishes() {
    let mut total = Exercised::default();
    for policy in Policy::ALL {
        for seed in 0..24u64 {
            total.absorb(drive_withdrawing(seed, 12, 4, 2, policy, true));
            total.absorb(drive_withdrawing(seed, 10, 3, 1, policy, true));
            total.absorb(drive_withdrawing(seed, 8, 1, 1, policy, true));
            total.absorb(drive_withdrawing(seed, 10, 4, 4, policy, true));
        }
    }
    // Non-vacuity, as the sweep above: a run in which nothing was ever held and
    // no state needed a completion would pass on an implementation with no
    // rules in it.
    assert!(total.gate_holds > 0);
    assert!(total.only_completable > 0);
    assert!(total.publication_blocked > 0);
    assert!(total.only_compilable > 0);
}

/// The sweep is only worth running if the interleavings actually differ. A
/// suite where every policy produced the same order would be one test wearing
/// three hats.
#[test]
fn the_policies_reach_different_completion_orders() {
    let orders: Vec<Vec<u64>> = Policy::ALL.iter().map(|p| order(7, *p)).collect();
    assert!(
        orders[0] != orders[1] || orders[1] != orders[2],
        "three policies produced one order: {orders:?}"
    );
}

/// The completion order one policy reaches on one batch, for the sweep's own
/// non-vacuity check.
fn order(seed: u64, policy: Policy) -> Vec<u64> {
    let specs = batch(seed, 10, 3);
    let mut model = SessionModel::new(SessionId(1));
    let mut map = QueueMap::new();
    for c in 1..=3u32 {
        model.open_channel(ChannelId(c)).expect("once");
        map.assign(ChannelId(c), PhysicalQueue(c % 2));
    }
    let mut gate = SubmitGate::new(map);
    let mut ordinals: Vec<IngressOrdinal> = Vec::new();
    for spec in &specs {
        let packet = Packet {
            session: model.generation(),
            channel: Channel::Child,
            domain: spec.channel,
            opcode: 0x37,
            stamp_waits: spec
                .wait
                .filter(|_| !spec.forwarded)
                .map(|(w, _)| w)
                .into_iter()
                .collect(),
            completion: spec.completion,
            payload: Payload::Exec(ExecWork {
                accesses: vec![AccessIntent {
                    key: AccessKey::Whole(ResourceKey {
                        backing: BackingId(spec.backing),
                        heap: None,
                    }),
                    mode: spec.mode,
                    domain: spec.channel,
                    api_stages: 0,
                    input_content_version: None,
                    output_content_version: None,
                }],
                ..ExecWork::default()
            }),
        };
        let ordinal = model
            .admit(&packet)
            .expect("well-formed")
            .transaction
            .identity
            .ingress;
        let producers: Vec<IngressOrdinal> = spec
            .wait
            .filter(|_| spec.forwarded)
            .map(|(_, p)| ordinals[p])
            .into_iter()
            .collect();
        gate.admit(ordinal, spec.channel, &producers);
        ordinals.push(ordinal);
    }

    let mut rng = Rng::new(seed ^ 0xA11CE);
    let mut ready: BTreeSet<IngressOrdinal> = BTreeSet::new();
    let mut submitted: BTreeSet<IngressOrdinal> = BTreeSet::new();
    let mut completed: Vec<u64> = Vec::new();
    let mut done: BTreeSet<IngressOrdinal> = BTreeSet::new();
    for _ in 0..1024 {
        if done.len() == specs.len() {
            break;
        }
        ready.extend(model.take_ready());
        let submittable: Vec<IngressOrdinal> = ready
            .iter()
            .copied()
            .filter(|o| !submitted.contains(o))
            .filter(|o| gate.gate(*o) == Admission::Go)
            .collect();
        let completable: Vec<IngressOrdinal> = submitted.difference(&done).copied().collect();
        if !completable.is_empty() && (submittable.is_empty() || rng.below(2) == 0) {
            let indices: Vec<usize> = (0..completable.len()).collect();
            let chosen = completable[policy.pick(&indices, &mut rng)];
            let epoch = model.epoch();
            let _ = model
                .complete(epoch, chosen)
                .expect("no device loss in this harness");
            gate.retire(chosen);
            done.insert(chosen);
            ready.remove(&chosen);
            completed.push(chosen.0);
        } else if !submittable.is_empty() {
            let indices: Vec<usize> = (0..submittable.len()).collect();
            let chosen = submittable[policy.pick(&indices, &mut rng)];
            gate.submitted(chosen);
            submitted.insert(chosen);
        }
    }
    completed
}
