//! A parallel schedule the dependency graph permits, and what it has to mean.
//!
//! # The claim this module exists to fail on
//!
//! The replacement's entire risk is that running transactions concurrently
//! stops meaning what running them one at a time meant. [`crate::interpret`]
//! defines the second thing. This defines the first: it drives
//! [`crate::depend::DependencyGraph`] and [`crate::ready::Scheduler`] over a
//! batch and completes transactions in an order those two permit — an
//! arbitrary one, chosen by a seed, so a test can sweep the space rather than
//! check one lucky interleaving.
//!
//! Then [`equivalent`] says whether the two agree, and the point of writing
//! that relation down is that it is *not* "the traces are equal". Getting it
//! wrong in the strict direction makes the test fail for reorderings a device
//! is allowed to do; getting it wrong in the loose direction makes it pass for
//! ones it is not.
//!
//! # Completion order is the whole schedule
//!
//! A transaction's effects become visible at completion and never before, so a
//! model that applies effects atomically at completion loses nothing by not
//! representing "started but not finished". Every interleaving of overlapping
//! execution is observationally some completion order, and every completion
//! order this model can produce is a linear extension of the dependency
//! relation. So the space this sweeps is the space.
//!
//! # What may differ, and why
//!
//! A **region's** content versions are compared in order. Two transactions
//! that make aliasing memory current are ordered by their hazard edge, so that
//! region's versions have one legal sequence and a device that produced the
//! other sequence would be showing the guest stale bytes under a fresh version.
//! Two transactions writing *disjoint* regions of one backing are ordered by
//! nothing and need not be: their histories are separate, which is why the
//! region is part of the observation rather than folded into the backing.
//!
//! A **channel's releases** are compared in order. Work may finish in any order
//! the dependency graph permits, but each channel tells the guest about it in
//! channel order — that is [`crate::publish`]'s contract, and comparing the
//! release sequence is how a schedule that violated it is caught even when the
//! resulting slot values happen to agree.
//!
//! An event generation, and a stamp slot two channels share, are compared by
//! their **final value**, and every publication must advance. These are
//! monotone points that independent work races to on purpose: two packets that
//! touch no common memory carry no ordering, and the guest that submitted them
//! asked for none. Requiring a fixed order across channels would be requiring
//! the device to serialise work the guest deliberately left independent.
//!
//! A transaction's completion-visible observations must be **contiguous** in
//! the trace, and its completion word must land after them and never inside
//! them. A guest that polled the stamp and then read the content must not be
//! able to see the flag without the bytes; another transaction's versions
//! landing inside one's completion window is the same failure wearing a
//! different shape.
//!
//! # The batch has to be one a serial run could execute
//!
//! Ingress order is only a legal schedule when every explicit wait names a
//! producer that already ran. A batch where a packet waits for a later
//! packet's stamp has no serial meaning to compare against — the serial
//! interpreter refuses it as unmeetable, which is correct and is not a
//! divergence. [`eligible`] names that condition, and three others, rather
//! than letting a generator quietly produce batches the comparison cannot
//! speak about.

use crate::access::{AccessKey, BackingId, ContentVersion};
use crate::depend::{Census, DependencyGraph};
use crate::exec::{Prerequisite, VersionPublication};
use crate::identity::{
    ChannelId, ChannelSequence, IngressOrdinal, ResourceId, SessionGeneration, StampSlot,
    StampValue,
};
use crate::interpret::{Interpreter, Observation, Refusal};
use crate::prereq::{WaitGraph, WaitPoint};
use crate::publish::{Publisher, Release};
use crate::ready::Scheduler;
use crate::transaction::DeviceTransaction;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

/// Why a batch has no serial meaning to compare a parallel run against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ineligible {
    /// Transactions were not handed over in ingress order, which is the order
    /// the hazard compiler is admitted in.
    OutOfIngressOrder {
        at: IngressOrdinal,
        after: IngressOrdinal,
    },
    /// A packet waits for a point produced by a packet that arrives no earlier
    /// than it does. Legal for a guest to submit and legal for the device to
    /// run; simply not something ingress order executes, so the serial
    /// interpreter refuses it and there is nothing to compare.
    ForwardExplicitWait {
        waiter: IngressOrdinal,
        point: WaitPoint,
        producer: IngressOrdinal,
    },
    /// A packet waits for a point nothing in the batch produces.
    UnansweredWait {
        waiter: IngressOrdinal,
        point: WaitPoint,
    },
    /// A packet carries an encoder-scoped fence prerequisite. The serial
    /// interpreter answers those from whether any packet has ever updated the
    /// fence, which is order-sensitive state neither graph carries; a batch
    /// that uses one is outside what this comparison can speak about.
    FencePrerequisite { waiter: IngressOrdinal },
    /// Two packets make aliasing memory current without a hazard edge between
    /// them, so that region's version sequence has no single legal order.
    ///
    /// Version claims are derived from write accesses and aliasing writes in
    /// one domain always produce an edge, so what is left is two *channels*
    /// writing memory they share. The guest supplied no ordering between them
    /// and neither does this device; such a batch is not one this comparison
    /// can judge, and saying so is better than picking a winner.
    UnorderedVersionRace {
        backing: BackingId,
        first: IngressOrdinal,
        second: IngressOrdinal,
    },
    /// The batch spans more than one semantic generation, so a serial run
    /// would refuse part of it for reasons that are not about scheduling.
    MixedGeneration {
        expected: SessionGeneration,
        found: SessionGeneration,
    },
}

impl Ineligible {
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::OutOfIngressOrder { .. } => "schedule_out_of_ingress_order",
            Self::ForwardExplicitWait { .. } => "schedule_forward_explicit_wait",
            Self::UnansweredWait { .. } => "schedule_unanswered_wait",
            Self::FencePrerequisite { .. } => "schedule_fence_prerequisite",
            Self::UnorderedVersionRace { .. } => "schedule_unordered_version_race",
            Self::MixedGeneration { .. } => "schedule_mixed_generation",
        }
    }
}

/// Whether ingress order is a legal schedule for this batch, so that a serial
/// run and a parallel one are comparable at all.
///
/// # Errors
///
/// The first condition the batch fails, in the order the variants are
/// declared: shape before content, so a batch that is out of ingress order is
/// reported as that rather than as whatever the misordering made of its waits.
pub fn eligible(batch: &[DeviceTransaction]) -> Result<(), Ineligible> {
    for pair in batch.windows(2) {
        if pair[1].identity.ingress <= pair[0].identity.ingress {
            return Err(Ineligible::OutOfIngressOrder {
                at: pair[1].identity.ingress,
                after: pair[0].identity.ingress,
            });
        }
    }
    if let Some(first) = batch.first() {
        for tx in batch {
            if tx.identity.session != first.identity.session {
                return Err(Ineligible::MixedGeneration {
                    expected: first.identity.session,
                    found: tx.identity.session,
                });
            }
        }
    }
    for tx in batch {
        if tx.exec().is_some_and(|e| {
            e.prerequisites()
                .iter()
                .any(|p| matches!(p, Prerequisite::Fence { .. }))
        }) {
            return Err(Ineligible::FencePrerequisite {
                waiter: tx.identity.ingress,
            });
        }
    }

    let mut waits = WaitGraph::new();
    for tx in batch {
        waits.admit(tx);
    }
    for diagnosis in waits.diagnose() {
        if let crate::prereq::Diagnosis::Unproduced { waiter, point } = diagnosis {
            return Err(Ineligible::UnansweredWait { waiter, point });
        }
    }
    for (waiter, point, producer) in earliest_producers(&waits) {
        if producer >= waiter {
            return Err(Ineligible::ForwardExplicitWait {
                waiter,
                point,
                producer,
            });
        }
    }

    version_races(batch)?;
    Ok(())
}

/// For each unmet wait, the earliest transaction that discharges it.
///
/// A wait needs *one* producer that ran before it, not every producer to have.
/// A later packet that also discharges the point is ordinary — an event is
/// signalled repeatedly and every signal past the waited-for value discharges
/// it — so the earliest producer is both what decides eligibility and the only
/// prerequisite worth carrying: depending on the rest would order work the
/// guest left independent.
fn earliest_producers(waits: &WaitGraph) -> Vec<(IngressOrdinal, WaitPoint, IngressOrdinal)> {
    let mut out: Vec<(IngressOrdinal, WaitPoint, IngressOrdinal)> = Vec::new();
    for (waiter, point, producer) in waits.edges() {
        match out.iter_mut().find(|(w, p, _)| *w == waiter && *p == point) {
            Some(slot) => slot.2 = slot.2.min(producer),
            None => out.push((waiter, point, producer)),
        }
    }
    out
}

/// The one eligibility question that needs the hazard compiler to answer it.
///
/// Two version claims over memory that may alias must be hazard-ordered, or
/// their region's history has two legal sequences and no reason to prefer
/// either. Since versions are derived from write accesses, two aliasing claims
/// in one domain always *are* ordered — [`crate::access::requires_edge`] says
/// so — which leaves exactly one shape this can still catch: two channels
/// writing memory they share. That is real, it has no ordering the guest
/// supplied, and it is named rather than assumed away.
fn version_races(batch: &[DeviceTransaction]) -> Result<(), Ineligible> {
    let mut graph = DependencyGraph::new();
    let mut publishers: Vec<(IngressOrdinal, VersionPublication)> = Vec::new();
    let mut ordered: HashMap<IngressOrdinal, Vec<IngressOrdinal>> = HashMap::new();
    for tx in batch {
        ordered.insert(
            tx.identity.ingress,
            graph.admit(tx.identity.ingress, tx.accesses()),
        );
        publishers.extend(
            crate::exec::published_versions(tx.accesses()).map(|p| (tx.identity.ingress, p)),
        );
    }
    // Reachability over hazard edges, which point backwards, so one pass in
    // ingress order settles it.
    let mut reaches: HashMap<IngressOrdinal, std::collections::BTreeSet<IngressOrdinal>> =
        HashMap::new();
    for tx in batch {
        let mut set = std::collections::BTreeSet::new();
        for earlier in &ordered[&tx.identity.ingress] {
            set.insert(*earlier);
            if let Some(theirs) = reaches.get(earlier) {
                set.extend(theirs.iter().copied());
            }
        }
        reaches.insert(tx.identity.ingress, set);
    }
    for (at, (second, later)) in publishers.iter().enumerate() {
        for (first, earlier) in &publishers[..at] {
            if first == second || !earlier.region.may_alias(later.region) {
                continue;
            }
            if !reaches[second].contains(first) {
                return Err(Ineligible::UnorderedVersionRace {
                    backing: later.backing,
                    first: *first,
                    second: *second,
                });
            }
        }
    }
    Ok(())
}

/// One execution of a batch.
#[derive(Clone, Debug, Default)]
pub struct Run {
    /// Every observation, in the order it became visible.
    pub trace: Vec<Observation>,
    /// Completion order, and where each transaction's completion-visible
    /// observations sit. Its stamp is not in this window: a stamp becomes
    /// visible at publication, which is a later event.
    pub spans: Vec<(IngressOrdinal, Range<usize>)>,
    /// What each channel released, in the order it released it.
    pub releases: Vec<(ChannelId, Release)>,
    /// Where each transaction's completion stamp landed in the trace, for the
    /// stamps that advanced their slot and so produced an observation.
    pub stamp_at: Vec<(IngressOrdinal, usize)>,
    /// Transactions the scheduler never released. Empty for an eligible batch;
    /// a non-empty one is a defect in the readiness service, not in the batch.
    pub stalled: Vec<IngressOrdinal>,
    /// The most positions each channel ever held behind an unfinished head.
    ///
    /// What ordered publication cost this schedule. Zero for every channel
    /// means the work also *finished* in channel order and the FIFO never had
    /// to hold anything — which is a fact about the schedule, not about the
    /// contract.
    pub blocked: Vec<(ChannelId, usize)>,
    /// What compiling this batch's hazards cost.
    pub census: Census,
}

impl Run {
    /// Completion order.
    #[must_use]
    pub fn order(&self) -> Vec<IngressOrdinal> {
        self.spans.iter().map(|(o, _)| *o).collect()
    }

    /// One channel's releases, in publication order.
    #[must_use]
    pub fn published_by(&self, domain: ChannelId) -> Vec<Release> {
        self.releases
            .iter()
            .filter(|(d, _)| *d == domain)
            .map(|(_, r)| *r)
            .collect()
    }

    /// Every channel that published anything, in a stable order.
    #[must_use]
    pub fn domains(&self) -> Vec<ChannelId> {
        let mut out: Vec<_> = self.releases.iter().map(|(d, _)| *d).collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Which transaction holds each channel position.
fn positions(batch: &[DeviceTransaction]) -> HashMap<(ChannelId, ChannelSequence), IngressOrdinal> {
    batch
        .iter()
        .map(|tx| {
            (
                (tx.identity.domain, tx.identity.domain_sequence),
                tx.identity.ingress,
            )
        })
        .collect()
}

/// Pay the stamps a channel just released, in the order it released them.
///
/// The one place a completion word becomes readable, so that the serial run
/// and the parallel one cannot differ in *how* they publish — only in when.
fn pay(
    releases: Vec<Release>,
    domain: ChannelId,
    interpreter: &mut Interpreter,
    mut scheduler: Option<&mut Scheduler>,
    at: &HashMap<(ChannelId, ChannelSequence), IngressOrdinal>,
    run: &mut Run,
) {
    for release in releases {
        if let Some(stamp) = release.stamp {
            let before = interpreter.trace().len();
            interpreter.publish(stamp);
            if interpreter.trace().len() > before {
                run.stamp_at.push((at[&(domain, release.sequence)], before));
            }
            if let Some(scheduler) = scheduler.as_deref_mut() {
                scheduler.publish(stamp);
            }
        }
        run.releases.push((domain, release));
    }
}

/// Run the batch one transaction at a time in ingress order.
///
/// The reference. No scheduler, no readiness, nothing to be concurrent about —
/// but publication still goes through the same [`Publisher`], because ordered
/// guest publication is a contract and not a consequence of concurrency.
#[must_use]
pub fn serial(batch: &[DeviceTransaction]) -> Run {
    let at = positions(batch);
    let mut interpreter = Interpreter::new();
    let mut publisher = Publisher::new();
    let mut run = Run::default();
    for tx in batch {
        publisher.admit(tx.identity.domain, tx.identity.domain_sequence);
        let start = interpreter.trace().len();
        let owed = interpreter.complete(tx);
        run.spans
            .push((tx.identity.ingress, start..interpreter.trace().len()));
        let released = match owed {
            Ok(stamp) => publisher.complete(tx.identity.domain, tx.identity.domain_sequence, stamp),
            // A refused position never publishes, and must not hold the ones
            // behind it.
            Err(_) => publisher.withdraw(tx.identity.domain, tx.identity.domain_sequence),
        };
        pay(
            released,
            tx.identity.domain,
            &mut interpreter,
            None,
            &at,
            &mut run,
        );
        note_blocked(&publisher, &mut run);
    }
    run.trace = interpreter.trace().to_vec();
    run
}

/// Keep the high-water mark of held positions per channel.
fn note_blocked(publisher: &Publisher, run: &mut Run) {
    for (domain, held) in publisher.blocked() {
        match run.blocked.iter_mut().find(|(d, _)| *d == domain) {
            Some(slot) => slot.1 = slot.1.max(held),
            None => run.blocked.push((domain, held)),
        }
    }
    run.blocked.sort_unstable();
}

/// Run the batch through the dependency graph and readiness service,
/// completing whichever ready transaction `seed` picks.
///
/// Both graphs are driven exactly as production would drive them: hazards from
/// [`DependencyGraph::admit`], explicit event waits as ordinal prerequisites,
/// stamp waits through the readiness service's own stamp machinery, and
/// retirement on completion. What the seed chooses is only which of the
/// transactions the two of them have *already declared ready* goes next.
#[must_use]
pub fn parallel(batch: &[DeviceTransaction], seed: u64) -> Run {
    let mut rng = Rng::new(seed);
    parallel_with(batch, move |ready| rng.below(ready.len()))
}

/// The same, with the choice of which ready transaction goes next handed to
/// the caller.
///
/// `pick` receives the transactions the dependency graph and the readiness
/// service have *already declared ready* and returns an index into them. It
/// cannot make an illegal schedule, only choose among the legal ones — which
/// is what makes "always take the lowest ordinal" a way to ask whether ingress
/// order is still reachable.
#[must_use]
pub fn parallel_with(
    batch: &[DeviceTransaction],
    mut pick: impl FnMut(&[IngressOrdinal]) -> usize,
) -> Run {
    let by_ordinal: HashMap<IngressOrdinal, &DeviceTransaction> =
        batch.iter().map(|tx| (tx.identity.ingress, tx)).collect();
    let at = positions(batch);

    // Explicit event waits become ordinal prerequisites. Eligibility has
    // already established that every producer is earlier, so this cannot
    // introduce a forward edge.
    let mut waits = WaitGraph::new();
    for tx in batch {
        waits.admit(tx);
    }
    let mut explicit: HashMap<IngressOrdinal, Vec<IngressOrdinal>> = HashMap::new();
    for (waiter, point, producer) in earliest_producers(&waits) {
        if matches!(point, WaitPoint::Event { .. }) {
            explicit.entry(waiter).or_default().push(producer);
        }
    }

    let mut graph = DependencyGraph::new();
    let mut scheduler = Scheduler::new();
    let mut publisher = Publisher::new();
    let mut pool: Vec<IngressOrdinal> = Vec::new();
    for tx in batch {
        publisher.admit(tx.identity.domain, tx.identity.domain_sequence);
        let mut prerequisites = graph.admit(tx.identity.ingress, tx.accesses());
        prerequisites.extend(explicit.remove(&tx.identity.ingress).unwrap_or_default());
        prerequisites.sort_unstable();
        prerequisites.dedup();

        scheduler.admit(
            tx.identity.ingress,
            &prerequisites,
            &tx.stamp_waits,
            // A batch is compared for schedule equivalence, and a pipeline that
            // is still compiling is not a property of the schedule: every arm
            // of the comparison would hold on the same transaction. Pipeline
            // readiness is tested where it lives, in `crate::ready`.
            &[],
            tx.completion,
        );
    }

    let mut interpreter = Interpreter::new();
    let mut run = Run::default();
    loop {
        pool.extend(scheduler.take_ready());
        if pool.is_empty() {
            break;
        }
        pool.sort_unstable();
        let ordinal = pool.remove(pick(&pool));
        let tx = by_ordinal[&ordinal];
        let start = interpreter.trace().len();
        let owed = interpreter.complete(tx);
        run.spans.push((ordinal, start..interpreter.trace().len()));

        // Completion releases hazard dependents at once, because they wait for
        // the work. The stamp it owes is paid only when this channel's
        // publication order reaches it, and whatever waits on that stamp is
        // released then and not before.
        //
        // The scheduling side of that happens whether the interpreter accepted
        // the transaction or refused it: the ordinal is no longer pending and
        // the hazard graph no longer holds its accesses either way. What the
        // outcome decides is publication.
        let scheduled = scheduler.complete(ordinal);
        graph.retire(ordinal);
        let released = match owed {
            Ok(_) => publisher.complete(tx.identity.domain, tx.identity.domain_sequence, scheduled),
            // `serial`'s arm, and for its reason: a refused position never
            // publishes, and must not hold the ones behind it. Readiness is
            // not the interpreter — `eligible` says ingress order is a legal
            // schedule for the batch, not that the interpreter will accept
            // every transaction in it — so a refusal here is a state this loop
            // reaches rather than one it may assert away. Publishing it would
            // hand the guest a completion word for work the model refused, and
            // would diverge from the serial run that is the reference.
            Err(_) => publisher.withdraw(tx.identity.domain, tx.identity.domain_sequence),
        };
        pay(
            released,
            tx.identity.domain,
            &mut interpreter,
            Some(&mut scheduler),
            &at,
            &mut run,
        );
        note_blocked(&publisher, &mut run);
    }
    run.stalled = scheduler.stalled();
    run.trace = interpreter.trace().to_vec();
    run.census = graph.census();
    run
}

/// A way the two runs meant different things.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Divergence {
    /// A transaction ran in one and not the other.
    DifferentTransactions {
        serial: Vec<IngressOrdinal>,
        parallel: Vec<IngressOrdinal>,
    },
    /// A backing's content versions were published in a different order, or a
    /// different set of them was published.
    ContentHistory {
        backing: BackingId,
        region: AccessKey,
        serial: Vec<ContentVersion>,
        parallel: Vec<ContentVersion>,
    },
    /// A completion word came to rest at a different value.
    StampOutcome {
        slot: StampSlot,
        serial: Option<StampValue>,
        parallel: Option<StampValue>,
    },
    /// An event generation came to rest at a different value.
    EventOutcome {
        event: ResourceId,
        serial: u64,
        parallel: u64,
    },
    /// A monotone point was published a value that did not advance it.
    NonMonotonePublication { at: usize },
    /// A fence was updated a different number of times.
    FenceUpdates {
        fence: ResourceId,
        serial: usize,
        parallel: usize,
    },
    /// A channel published a different sequence of positions, or published
    /// them in a different order.
    PublicationOrder {
        domain: ChannelId,
        serial: Vec<Release>,
        parallel: Vec<Release>,
    },
    /// A transaction was refused in one run and not the other.
    Refusals {
        serial: Vec<(IngressOrdinal, Refusal)>,
        parallel: Vec<(IngressOrdinal, Refusal)>,
    },
    /// A channel showed a different sequence of frames.
    Presentation {
        domain: ChannelId,
        serial: Vec<crate::identity::MappingId>,
        parallel: Vec<crate::identity::MappingId>,
    },
    /// A transaction's lifetime operation was declined in one run and not the
    /// other. Its stamp is owed either way — see
    /// [`crate::interpret::Observation::OperationDeclined`] — so this is a
    /// divergence in what the model did, not in what the guest was told to
    /// wait for.
    Declined {
        serial: Vec<(IngressOrdinal, crate::lifecycle::Refusal)>,
        parallel: Vec<(IngressOrdinal, crate::lifecycle::Refusal)>,
    },
    /// A query completed with no answer written under one schedule and with
    /// one under the other.
    ///
    /// The failure this names has no other symptom. The completion word is
    /// published either way, so a guest that polls the stamp and then reads
    /// the destination is handed a fresh flag over bytes nobody wrote --- and
    /// every other part of this comparison agrees, because the versions, the
    /// stamps, the events and the releases are all identical. Compared by
    /// ingress like the two channels above it: which transaction stalled is
    /// the fact, and the order the stalls reached the trace in is not.
    QueriesUnanswered {
        serial: Vec<(IngressOrdinal, crate::query::QueryKind, crate::query::Stall)>,
        parallel: Vec<(IngressOrdinal, crate::query::QueryKind, crate::query::Stall)>,
    },
    /// One transaction's publications were interrupted by another's.
    SplitPublication { ordinal: IngressOrdinal },
    /// A transaction published its completion stamp before its content
    /// versions.
    StampBeforeVersions { ordinal: IngressOrdinal },
    /// A published version was beaten by newer content in one run and not the
    /// other, or lost a different amount of it.
    ///
    /// Guest-visible, which is why it is a divergence and not a note: the
    /// bytes of a beaten write are never readable. A schedule where one write
    /// lands and another where it is overwritten are two different pictures on
    /// the screen — and if a hazard edge orders the two writers, neither
    /// schedule may produce it.
    ContentBeaten {
        backing: BackingId,
        region: AccessKey,
        serial: Vec<(ContentVersion, u64)>,
        parallel: Vec<(ContentVersion, u64)>,
    },
}

impl Divergence {
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::DifferentTransactions { .. } => "diverge_different_transactions",
            Self::ContentHistory { .. } => "diverge_content_history",
            Self::StampOutcome { .. } => "diverge_stamp_outcome",
            Self::EventOutcome { .. } => "diverge_event_outcome",
            Self::NonMonotonePublication { .. } => "diverge_non_monotone_publication",
            Self::FenceUpdates { .. } => "diverge_fence_updates",
            Self::PublicationOrder { .. } => "diverge_publication_order",
            Self::Refusals { .. } => "diverge_refusals",
            Self::Declined { .. } => "diverge_declined",
            Self::Presentation { .. } => "diverge_presentation",
            Self::SplitPublication { .. } => "diverge_split_publication",
            Self::StampBeforeVersions { .. } => "diverge_stamp_before_versions",
            Self::ContentBeaten { .. } => "diverge_content_beaten",
            Self::QueriesUnanswered { .. } => "diverge_queries_unanswered",
        }
    }
}

/// Whether a parallel run means the same thing the serial one did.
///
/// See the module documentation for which parts are compared in order and
/// which by outcome, and why each is which.
///
/// # Errors
///
/// The first difference found, in a fixed order so the answer does not depend
/// on hash iteration.
pub fn equivalent(serial: &Run, parallel: &Run) -> Result<(), Divergence> {
    let mut mine = serial.order();
    let mut theirs = parallel.order();
    mine.sort_unstable();
    theirs.sort_unstable();
    if mine != theirs {
        return Err(Divergence::DifferentTransactions {
            serial: mine,
            parallel: theirs,
        });
    }

    let left = Summary::of(&serial.trace);
    let right = Summary::of(&parallel.trace);
    for (key, versions) in &left.content {
        let theirs = right.content.get(key).cloned().unwrap_or_default();
        if *versions != theirs {
            return Err(Divergence::ContentHistory {
                backing: key.0,
                region: key.1,
                serial: versions.clone(),
                parallel: theirs,
            });
        }
    }
    for key in right.content.keys() {
        if !left.content.contains_key(key) {
            return Err(Divergence::ContentHistory {
                backing: key.0,
                region: key.1,
                serial: Vec::new(),
                parallel: right.content[key].clone(),
            });
        }
    }
    for key in left.beaten.keys().chain(right.beaten.keys()) {
        let (a, b) = (
            left.beaten.get(key).cloned().unwrap_or_default(),
            right.beaten.get(key).cloned().unwrap_or_default(),
        );
        if a != b {
            return Err(Divergence::ContentBeaten {
                backing: key.0,
                region: key.1,
                serial: a,
                parallel: b,
            });
        }
    }
    for slot in left.stamps.keys().chain(right.stamps.keys()) {
        let (a, b) = (
            left.stamps.get(slot).copied(),
            right.stamps.get(slot).copied(),
        );
        if a != b {
            return Err(Divergence::StampOutcome {
                slot: *slot,
                serial: a,
                parallel: b,
            });
        }
    }
    for event in left.events.keys().chain(right.events.keys()) {
        let (a, b) = (
            left.events.get(event).copied().unwrap_or(0),
            right.events.get(event).copied().unwrap_or(0),
        );
        if a != b {
            return Err(Divergence::EventOutcome {
                event: *event,
                serial: a,
                parallel: b,
            });
        }
    }
    for fence in left.fences.keys().chain(right.fences.keys()) {
        let (a, b) = (
            left.fences.get(fence).copied().unwrap_or(0),
            right.fences.get(fence).copied().unwrap_or(0),
        );
        if a != b {
            return Err(Divergence::FenceUpdates {
                fence: *fence,
                serial: a,
                parallel: b,
            });
        }
    }
    let (mut refused_left, mut refused_right) = (left.refusals.clone(), right.refusals.clone());
    refused_left.sort_by_key(|(o, _)| *o);
    refused_right.sort_by_key(|(o, _)| *o);
    if refused_left != refused_right {
        return Err(Divergence::Refusals {
            serial: refused_left,
            parallel: refused_right,
        });
    }

    let (mut declined_left, mut declined_right) = (left.declined.clone(), right.declined.clone());
    declined_left.sort_by_key(|(o, _)| *o);
    declined_right.sort_by_key(|(o, _)| *o);
    if declined_left != declined_right {
        return Err(Divergence::Declined {
            serial: declined_left,
            parallel: declined_right,
        });
    }

    let (mut stalled_left, mut stalled_right) = (left.unanswered.clone(), right.unanswered.clone());
    stalled_left.sort_by_key(|(o, _, _)| *o);
    stalled_right.sort_by_key(|(o, _, _)| *o);
    if stalled_left != stalled_right {
        return Err(Divergence::QueriesUnanswered {
            serial: stalled_left,
            parallel: stalled_right,
        });
    }

    for domain in left.presented.keys().chain(right.presented.keys()) {
        let (a, b) = (
            left.presented.get(domain).cloned().unwrap_or_default(),
            right.presented.get(domain).cloned().unwrap_or_default(),
        );
        if a != b {
            return Err(Divergence::Presentation {
                domain: *domain,
                serial: a,
                parallel: b,
            });
        }
    }

    let mut domains = serial.domains();
    domains.extend(parallel.domains());
    domains.sort_unstable();
    domains.dedup();
    for domain in domains {
        let (a, b) = (serial.published_by(domain), parallel.published_by(domain));
        if a != b {
            return Err(Divergence::PublicationOrder {
                domain,
                serial: a,
                parallel: b,
            });
        }
    }

    // Both runs, and the serial one first. These two are properties of the
    // *model*, not of an interleaving: a completion window that another
    // transaction published inside, or a stamp that went backwards, is wrong
    // however the work was scheduled. Checking only the parallel run made the
    // reference an oracle nothing tested --- and a serial run that regressed a
    // stamp would have gone unseen twice over, because `Summary` records where
    // a monotone point *came to rest* and a regression leaves no trace in it.
    // So the outcome comparison above cannot see it either, and a broken
    // reference would have been declared equivalent to a correct schedule.
    monotone(serial)?;
    atomic_publication(serial)?;
    monotone(parallel)?;
    atomic_publication(parallel)
}

/// Every publication into a monotone point advances it.
fn monotone(run: &Run) -> Result<(), Divergence> {
    let mut stamps: HashMap<StampSlot, StampValue> = HashMap::new();
    let mut events: HashMap<ResourceId, u64> = HashMap::new();
    for (at, observation) in run.trace.iter().enumerate() {
        match *observation {
            Observation::StampPublished { slot, value } => {
                if stamps.get(&slot).is_some_and(|at| !value.follows(*at)) {
                    return Err(Divergence::NonMonotonePublication { at });
                }
                stamps.insert(slot, value);
            }
            Observation::EventAdvanced { event, to } => {
                if events.get(&event).is_some_and(|at| to <= *at) {
                    return Err(Divergence::NonMonotonePublication { at });
                }
                events.insert(event, to);
            }
            Observation::VersionPublished { .. }
            | Observation::VersionBeaten { .. }
            | Observation::FenceUpdated { .. }
            | Observation::Refused { .. }
            | Observation::OperationDeclined { .. }
            | Observation::FramePresented { .. }
            | Observation::QueryUnanswered { .. }
            | Observation::DiscardDeclined { .. } => {}
        }
    }
    Ok(())
}

/// What a transaction makes visible at completion is made visible all at once,
/// and its completion word comes after it.
fn atomic_publication(run: &Run) -> Result<(), Divergence> {
    let mut spans: Vec<_> = run.spans.clone();
    spans.sort_by_key(|(_, r)| r.start);
    let mut covered = 0usize;
    for (ordinal, span) in &spans {
        if span.start < covered {
            return Err(Divergence::SplitPublication { ordinal: *ordinal });
        }
        covered = span.end;
        if run.trace[span.clone()]
            .iter()
            .any(|o| matches!(o, Observation::StampPublished { .. }))
        {
            // A stamp is not visible at completion. One inside a completion
            // window is the failure a guest sees as a fresh flag over stale
            // bytes.
            return Err(Divergence::StampBeforeVersions { ordinal: *ordinal });
        }
    }
    for (ordinal, at) in &run.stamp_at {
        if !matches!(run.trace[*at], Observation::StampPublished { .. }) {
            return Err(Divergence::StampBeforeVersions { ordinal: *ordinal });
        }
        let end = spans
            .iter()
            .find(|(o, _)| o == ordinal)
            .map_or(0, |(_, span)| span.end);
        if *at < end {
            return Err(Divergence::StampBeforeVersions { ordinal: *ordinal });
        }
    }
    Ok(())
}

/// What a trace came to, per observable location.
#[derive(Debug, Default)]
struct Summary {
    /// Per (backing, region), because two disjoint regions of one backing
    /// are two independent histories.
    content: BTreeMap<(BackingId, AccessKey), Vec<ContentVersion>>,
    /// Per (backing, region), each write that newer content beat and how many
    /// bytes of it survived. Part of the outcome because a beaten write is a
    /// picture the guest never sees.
    beaten: BTreeMap<(BackingId, AccessKey), Vec<(ContentVersion, u64)>>,
    stamps: BTreeMap<StampSlot, StampValue>,
    events: BTreeMap<ResourceId, u64>,
    fences: BTreeMap<ResourceId, usize>,
    refusals: Vec<(IngressOrdinal, Refusal)>,
    declined: Vec<(IngressOrdinal, crate::lifecycle::Refusal)>,
    /// Per domain, the frames handed to the display in order. Two channels'
    /// presents have no ordering obligation to each other, so this is not one
    /// interleaved sequence.
    presented: BTreeMap<ChannelId, Vec<crate::identity::MappingId>>,
    /// Queries that completed with no answer written. Part of the outcome
    /// because the guest reads its destination either way: a schedule that
    /// answered and one that stalled hand it different bytes while publishing
    /// the same completion word.
    unanswered: Vec<(IngressOrdinal, crate::query::QueryKind, crate::query::Stall)>,
}

impl Summary {
    fn of(trace: &[Observation]) -> Self {
        let mut out = Self::default();
        for observation in trace {
            match *observation {
                Observation::VersionPublished {
                    backing,
                    region,
                    version,
                } => {
                    out.content
                        .entry((backing, region))
                        .or_default()
                        .push(version);
                }
                // Where the point came to rest, which on a monotone location
                // is the furthest value published into it and not the last
                // one written. Taking the last would make a trace that
                // regressed a slot compare equal to one that did not, and
                // that regression is exactly what [`monotone`] exists to
                // catch.
                Observation::StampPublished { slot, value } => {
                    let at = out.stamps.entry(slot).or_insert(value);
                    *at = at.later(value);
                }
                Observation::EventAdvanced { event, to } => {
                    let at = out.events.entry(event).or_insert(to);
                    *at = (*at).max(to);
                }
                Observation::FenceUpdated { fence } => {
                    *out.fences.entry(fence).or_insert(0) += 1;
                }
                Observation::VersionBeaten {
                    backing,
                    region,
                    version,
                    landed,
                } => {
                    out.beaten
                        .entry((backing, region))
                        .or_default()
                        .push((version, landed));
                }
                Observation::Refused { ingress, reason } => out.refusals.push((ingress, reason)),
                // Part of the outcome for the same reason a refusal is: it
                // lands on the always-on failure channel, and a schedule under
                // which an operation is declined and one under which it is not
                // do not mean the same thing.
                Observation::OperationDeclined { ingress, reason } => {
                    out.declined.push((ingress, reason));
                }
                Observation::FramePresented { domain, mapping } => {
                    out.presented.entry(domain).or_default().push(mapping);
                }
                Observation::QueryUnanswered {
                    ingress,
                    kind,
                    reason,
                } => out.unanswered.push((ingress, kind, reason)),
                // Not part of the outcome, and the variant's own doc says
                // why: whether a spare copy existed when a discard completed
                // is a function of which transfers had finished, so two
                // schedules the graph permits may decline different discards
                // and neither is wrong --- the bytes a later read returns are
                // the same either way. Requiring them to match would be the
                // strict-direction error the module doc warns about. It is in
                // the trace because it is guest work the device declined, not
                // because it is a difference in meaning.
                Observation::DiscardDeclined { .. } => {}
            }
        }
        out
    }
}

/// A reproducible choice of interleaving.
///
/// Deliberately not a general random source: what a schedule sweep needs is
/// that seed *n* names the same interleaving on every machine and every run,
/// so a failure is a bug report rather than a rumour. The stream walk's hostile
/// input sweep wants the same guarantee for the same reason, so it uses this
/// one rather than growing a second xorshift with a different constant.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) const fn new(seed: u64) -> Self {
        // Any non-zero state; a zero seed is a legitimate thing for a caller
        // to pass and xorshift is stuck there.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(crate) fn below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        usize::try_from(self.next() % bound as u64).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests;
