//! The fixed executor: a population that exists before the work does, and
//! work that goes to exactly one member of it.
//!
//! # Fixed means fixed
//!
//! The population is decided at construction and never grows. That is the
//! whole claim: no worker, command pool, or per-workload host object is
//! created because a packet arrived. A thread-per-unit design is not slower in
//! a way a benchmark shows — it is a design where the cost of a frame depends
//! on how many packets the guest happened to send, and where teardown has to
//! find an unbounded set of things to stop.
//!
//! # Work goes to *its* worker, not to a free one
//!
//! A recording's command pool, descriptor arena and continued encoder state
//! belong to one worker. Sending its continuation to a different worker would
//! mean recording into a pool that another thread owns, so an assignment names
//! its worker and dispatch honours that name exactly. Nothing here
//! load-balances an assignment that has already been made; [`FixedExecutor::assign`]
//! is where a *new* chain picks its worker, and it is the only place a choice
//! happens.
//!
//! # A refusal returns the work
//!
//! Every rejection hands back the complete [`Assignment`], payload included.
//! An executor that refused by returning an error code would leave the native
//! handles inside the request owned by nothing — the classic version of this
//! bug is a closure that captured them being dropped on the error path, which
//! leaks silently and cannot be found from the leak. So the error type carries
//! the work, and a caller that ignores it does not compile.
//!
//! # Dispatch does not wait
//!
//! Handing work to a worker returns a [`Receipt`] immediately. Nothing here
//! blocks, joins, or waits for a worker, because a coordinator that waits for
//! a recording is a coordinator that cannot schedule the work which would
//! release it.

use crate::identity::IngressOrdinal;

/// One member of the fixed population.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkerId(pub u16);

/// Work, and the worker that owns what it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assignment<T> {
    pub worker: WorkerId,
    /// The transaction this work belongs to, so a receipt can be matched to it
    /// without the payload having to be inspected.
    pub ingress: IngressOrdinal,
    pub payload: T,
}

/// Proof that a worker accepted work. Not a result and not a handle to one:
/// the work is running, and what it produces arrives on its own path.
///
/// Not `Clone`, no public constructor, and consumed by
/// [`FixedExecutor::finished`], because occupancy is the number placement is
/// decided from and a receipt completed twice makes it quietly wrong. The
/// `occupancy > 0` assertion catches that only for a worker holding one piece
/// of work; a worker holding two absorbs the second completion silently and
/// the panic then lands on whichever *other* receipt arrives next, naming a
/// caller that did nothing. So the second completion is unrepresentable
/// instead, and the assertion is left as the backstop for the case a type
/// cannot reach: a receipt handed to a different population.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a receipt never completed leaves its worker occupied forever"]
pub struct Receipt {
    worker: WorkerId,
    ingress: IngressOrdinal,
}

impl Receipt {
    /// The worker that took the work.
    #[must_use]
    pub const fn worker(&self) -> WorkerId {
        self.worker
    }

    /// The transaction the work belongs to, so a completion can be matched to
    /// it without the payload being inspected.
    #[must_use]
    pub const fn ingress(&self) -> IngressOrdinal {
        self.ingress
    }
}

/// Why a worker did not take the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// The assignment names a worker this population does not have. A forged
    /// or stale identity, and the one case where guessing a substitute would
    /// record into a pool belonging to someone else.
    UnknownWorker { worker: WorkerId },
    /// The worker has stopped taking work — teardown, or its epoch ended.
    WorkerStopped { worker: WorkerId },
}

impl Rejection {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownWorker { .. } => "executor_unknown_worker",
            Self::WorkerStopped { .. } => "executor_worker_stopped",
        }
    }
}

/// A refusal, with the work it refused.
///
/// `#[must_use]` because dropping one is exactly the leak this type exists to
/// prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a refused assignment owns its payload; dropping it leaks whatever the payload holds"]
pub struct Returned<T> {
    pub reason: Rejection,
    pub assignment: Assignment<T>,
}

#[derive(Clone, Copy, Debug)]
struct Worker {
    id: WorkerId,
    stopped: bool,
    occupancy: usize,
    accepted: usize,
}

/// A fixed population of workers and the rule for reaching exactly one.
#[derive(Debug)]
pub struct FixedExecutor {
    workers: Vec<Worker>,
}

impl FixedExecutor {
    /// Build the population.
    ///
    /// # Panics
    ///
    /// If `population` is zero. An executor with no workers accepts nothing
    /// and would turn every dispatch into a refusal that reads like a
    /// transient failure.
    #[must_use]
    pub fn with_population(population: u16) -> Self {
        assert!(
            population > 0,
            "a fixed executor with no workers runs nothing"
        );
        Self {
            workers: (0..population)
                .map(|n| Worker {
                    id: WorkerId(n),
                    stopped: false,
                    occupancy: 0,
                    accepted: 0,
                })
                .collect(),
        }
    }

    /// The population, which does not change.
    #[must_use]
    pub fn population(&self) -> usize {
        self.workers.len()
    }

    /// Pick a worker for a chain of work that has no predecessor.
    ///
    /// The only place a choice is made. Least-occupied, with the lowest
    /// identity breaking ties, so the same sequence of assignments produces
    /// the same placement on every run — an executor whose placement depends
    /// on timing cannot be tested for placement.
    ///
    /// Returns `None` when every worker has stopped.
    #[must_use]
    pub fn assign(&self) -> Option<WorkerId> {
        self.workers
            .iter()
            .filter(|w| !w.stopped)
            .min_by_key(|w| (w.occupancy, w.id))
            .map(|w| w.id)
    }

    /// Hand work to the worker it names.
    ///
    /// Does not wait, does not choose, and does not substitute. The named
    /// worker takes it or the work comes back.
    ///
    /// # Errors
    ///
    /// The complete assignment, with the reason. See [`Returned`].
    pub fn dispatch<T>(&mut self, assignment: Assignment<T>) -> Result<Receipt, Returned<T>> {
        let named = assignment.worker;
        let Some(worker) = self.workers.iter_mut().find(|w| w.id == named) else {
            return Err(Returned {
                reason: Rejection::UnknownWorker { worker: named },
                assignment,
            });
        };
        if worker.stopped {
            return Err(Returned {
                reason: Rejection::WorkerStopped { worker: named },
                assignment,
            });
        }
        worker.occupancy += 1;
        worker.accepted += 1;
        Ok(Receipt {
            worker: named,
            ingress: assignment.ingress,
        })
    }

    /// A worker finished the work a receipt named.
    ///
    /// # Panics
    ///
    /// If the receipt names a worker this population does not have, or one
    /// that was not occupied. Both mean a receipt was forged or completed
    /// twice, and either would make occupancy — the number placement is
    /// decided from — quietly wrong.
    pub fn finished(&mut self, receipt: Receipt) {
        let worker = self
            .workers
            .iter_mut()
            .find(|w| w.id == receipt.worker)
            .expect("a receipt from a worker this population does not have");
        assert!(
            worker.occupancy > 0,
            "{:?} finished work it was not given",
            receipt.worker
        );
        worker.occupancy -= 1;
    }

    /// Stop a worker taking new work. It keeps what it already has.
    pub fn stop(&mut self, worker: WorkerId) {
        if let Some(w) = self.workers.iter_mut().find(|w| w.id == worker) {
            w.stopped = true;
        }
    }

    /// Work a worker has been given and has not finished.
    #[must_use]
    pub fn occupancy(&self, worker: WorkerId) -> usize {
        self.workers
            .iter()
            .find(|w| w.id == worker)
            .map_or(0, |w| w.occupancy)
    }

    /// Work each worker has ever accepted, in identity order. The placement
    /// record, so an accidental hot worker is visible rather than inferred.
    #[must_use]
    pub fn accepted(&self) -> Vec<usize> {
        self.workers.iter().map(|w| w.accepted).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work(worker: WorkerId, ingress: u64, payload: &'static str) -> Assignment<&'static str> {
        Assignment {
            worker,
            ingress: IngressOrdinal(ingress),
            payload,
        }
    }

    /// The claim the type is named for: nothing is created because work
    /// arrived.
    #[test]
    fn the_population_does_not_grow_with_the_work() {
        let mut e = FixedExecutor::with_population(3);
        for n in 1..=64 {
            let worker = e.assign().expect("a live worker");
            let _receipt = e.dispatch(work(worker, n, "record")).expect("accepted");
        }
        assert_eq!(e.population(), 3, "sixty-four packets, three workers");
        assert_eq!(
            e.accepted(),
            vec![22, 21, 21],
            "and the placement is a fact rather than a hope"
        );
    }

    /// An assignment already made is honoured exactly, so continued encoder
    /// state stays with the pool that owns it.
    #[test]
    fn dispatch_goes_to_the_named_worker_and_never_to_a_freer_one() {
        let mut e = FixedExecutor::with_population(4);
        let pinned = WorkerId(2);
        for n in 1..=5 {
            let receipt = e
                .dispatch(work(pinned, n, "continuation"))
                .expect("accepted");
            assert_eq!(receipt.worker(), pinned);
        }
        assert_eq!(e.occupancy(pinned), 5);
        assert_eq!(
            e.accepted(),
            vec![0, 0, 5, 0],
            "three idle workers did not tempt it"
        );
    }

    /// The ownership rule, and the reason the error type is shaped as it is.
    #[test]
    fn an_unknown_worker_returns_the_whole_assignment() {
        let mut e = FixedExecutor::with_population(2);
        let refused = e
            .dispatch(work(WorkerId(9), 1, "native handles"))
            .expect_err("no such worker");
        assert_eq!(
            refused.reason,
            Rejection::UnknownWorker {
                worker: WorkerId(9)
            }
        );
        assert_eq!(
            refused.assignment.payload, "native handles",
            "the caller gets its handles back rather than losing them in a \
             dropped closure"
        );
    }

    #[test]
    fn a_stopped_worker_returns_the_whole_assignment() {
        let mut e = FixedExecutor::with_population(2);
        e.stop(WorkerId(0));
        let refused = e
            .dispatch(work(WorkerId(0), 1, "native handles"))
            .expect_err("stopped");
        assert_eq!(
            refused.reason,
            Rejection::WorkerStopped {
                worker: WorkerId(0)
            }
        );
        assert_eq!(refused.assignment.payload, "native handles");
        assert_eq!(
            e.assign(),
            Some(WorkerId(1)),
            "a new chain goes to a worker that is still taking work"
        );
    }

    #[test]
    fn an_executor_with_every_worker_stopped_assigns_nothing() {
        let mut e = FixedExecutor::with_population(2);
        e.stop(WorkerId(0));
        e.stop(WorkerId(1));
        assert_eq!(e.assign(), None);
    }

    /// A stopped worker keeps what it has: teardown does not lose in-flight
    /// work, it stops taking more.
    #[test]
    fn stopping_a_worker_does_not_abandon_the_work_it_holds() {
        let mut e = FixedExecutor::with_population(2);
        let receipt = e
            .dispatch(work(WorkerId(0), 1, "in flight"))
            .expect("accepted");
        e.stop(WorkerId(0));
        assert_eq!(e.occupancy(WorkerId(0)), 1);
        e.finished(receipt);
        assert_eq!(e.occupancy(WorkerId(0)), 0);
    }

    /// Placement is decided from occupancy, so a receipt completed twice would
    /// corrupt every later placement rather than fail where it happened. It
    /// cannot be spelled: `Receipt` is not `Clone` and `finished` consumes it,
    /// so this test is the *other* forgery --- a receipt taken from one
    /// population and completed against another, which no type can prevent.
    #[test]
    #[should_panic(expected = "finished work it was not given")]
    fn completing_another_populations_receipt_is_loud() {
        let mut mine = FixedExecutor::with_population(1);
        let mut theirs = FixedExecutor::with_population(1);
        let receipt = theirs
            .dispatch(work(WorkerId(0), 1, "elsewhere"))
            .expect("accepted");
        mine.finished(receipt);
    }

    #[test]
    #[should_panic(expected = "runs nothing")]
    fn a_population_of_none_is_a_contract_violation() {
        let _ = FixedExecutor::with_population(0);
    }

    /// Placement is deterministic, so a test may assert it.
    #[test]
    fn the_same_sequence_of_assignments_places_the_same_way() {
        let place = || {
            let mut e = FixedExecutor::with_population(3);
            let mut receipts = Vec::new();
            for n in 1..=7 {
                let worker = e.assign().expect("a live worker");
                receipts.push(e.dispatch(work(worker, n, "record")).expect("accepted"));
                if n % 3 == 0 {
                    e.finished(receipts.remove(0));
                }
            }
            e.accepted()
        };
        assert_eq!(place(), place());
    }
}
