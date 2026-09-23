//! Event/fence synchronization planner.
//!
//! A pure decision function and nothing else: given the generation currently
//! stored for a `(task, domain, ref)` and a requested signal/wait, decide
//! whether to advance the stored value, treat the operation as already
//! satisfied, or leave it pending. The storage itself is
//! [`crate::model::DeviceState::fence_generations`]; this module holds no state
//! and performs no I/O, so the whole event/fence contract is testable without a
//! device.

pub const FENCE_INITIAL_GENERATION: u64 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Domain {
    #[default]
    Unknown = 0,
    Event = 1,
    BlitFence = 2,
    ComputeFence = 3,
    RenderFence = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Decision {
    #[default]
    Invalid = 0,
    SignalUpdate = 1,
    SignalNoop = 2,
    WaitSatisfied = 3,
    WaitPending = 4,
}

/// Why the [`Decision`] came out the way it did.
///
/// Finer-grained than `Decision` on purpose: "signal ignored because it repeated
/// the current value" and "signal ignored because it went backwards" are the
/// same decision and a different contract, and only this field separates them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Reason {
    #[default]
    Invalid = 0,
    SignalFirst = 1,
    SignalAdvance = 2,
    SignalEqualIgnored = 3,
    SignalStaleIgnored = 4,
    WaitReached = 5,
    WaitMissingSignal = 6,
    WaitBelowTarget = 7,
    FenceUpdateFirst = 9,
    FenceUpdateAdvance = 10,
    FenceUpdateAtMax = 11,
    FenceWaitReached = 12,
    FenceWaitMissingUpdate = 13,
    BadFenceDomain = 14,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Signal,
    Wait,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceAction {
    Update,
    Wait,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Plan {
    pub decision: Decision,
    pub reason: Reason,
    /// Caller must write `update_value` back to the generation store.
    pub updates_state: bool,
    pub update_value: u64,
}

impl Plan {
    fn signal(reason: Reason, update_value: u64) -> Self {
        Self {
            decision: Decision::SignalUpdate,
            reason,
            updates_state: true,
            update_value,
        }
    }

    fn noop(reason: Reason) -> Self {
        Self {
            decision: Decision::SignalNoop,
            reason,
            updates_state: false,
            update_value: 0,
        }
    }

    fn decided(decision: Decision, reason: Reason) -> Self {
        Self {
            decision,
            reason,
            updates_state: false,
            update_value: 0,
        }
    }
}

fn is_fence_domain(domain: Domain) -> bool {
    matches!(
        domain,
        Domain::BlitFence | Domain::ComputeFence | Domain::RenderFence
    )
}

/// Plan a guest event signal or wait.
///
/// Signals carry an explicit wire value and advance monotonically; a repeated or
/// backwards value is ignored rather than rejected. A wait is satisfied once the
/// stored value reaches the target.
///
/// # There is no bounded wait to plan
///
/// `waitForEvent:value:timeoutMS:` is refused where its contract is settled —
/// `reims_vgpu_protocol::sync::event_kind` gives it no kind, so
/// `decode::sync` refuses the record and it never reaches a planner. This
/// function took a `has_timeout` flag and answered `WaitTimeoutUnsupported` for
/// it, which put one settled row's refusal in two places: the closure ledger
/// where it belongs, and here, where it was found first and read as a gap in
/// the planner rather than as a decision about the wire.
pub fn plan_event(kind: EventKind, value: u64, current: Option<u64>) -> Plan {
    match kind {
        EventKind::Signal => match current {
            None => Plan::signal(Reason::SignalFirst, value),
            Some(cur) if value > cur => Plan::signal(Reason::SignalAdvance, value),
            Some(cur) if value == cur => Plan::noop(Reason::SignalEqualIgnored),
            Some(_) => Plan::noop(Reason::SignalStaleIgnored),
        },
        EventKind::Wait => match current {
            Some(cur) if cur >= value => {
                Plan::decided(Decision::WaitSatisfied, Reason::WaitReached)
            }
            Some(_) => Plan::decided(Decision::WaitPending, Reason::WaitBelowTarget),
            None => Plan::decided(Decision::WaitPending, Reason::WaitMissingSignal),
        },
    }
}

/// Plan an encoder fence update or wait.
///
/// Unlike events, fences carry no wire value: an update is an implicit
/// increment of a generation counter that starts at
/// [`FENCE_INITIAL_GENERATION`], and a wait is satisfied by the existence of any
/// prior update (the drain is in-order, so a generation that exists has already
/// been reached).
pub fn plan_fence(action: FenceAction, domain: Domain, current: Option<u64>) -> Plan {
    if !is_fence_domain(domain) {
        return Plan::decided(Decision::Invalid, Reason::BadFenceDomain);
    }
    match action {
        FenceAction::Update => match current {
            None => Plan::signal(Reason::FenceUpdateFirst, FENCE_INITIAL_GENERATION),
            Some(u64::MAX) => Plan::noop(Reason::FenceUpdateAtMax),
            Some(cur) => Plan::signal(Reason::FenceUpdateAdvance, cur + 1),
        },
        FenceAction::Wait => match current {
            Some(_) => Plan::decided(Decision::WaitSatisfied, Reason::FenceWaitReached),
            None => Plan::decided(Decision::WaitPending, Reason::FenceWaitMissingUpdate),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_first_and_advance() {
        let plan = plan_event(EventKind::Signal, 5, None);
        assert!(plan.updates_state);
        assert_eq!(plan.update_value, 5);
        assert_eq!(plan.reason, Reason::SignalFirst);

        let plan2 = plan_event(EventKind::Signal, 7, Some(5));
        assert_eq!(plan2.reason, Reason::SignalAdvance);
        assert_eq!(plan2.update_value, 7);

        let plan3 = plan_event(EventKind::Signal, 5, Some(5));
        assert!(!plan3.updates_state);
        assert_eq!(plan3.reason, Reason::SignalEqualIgnored);

        let plan4 = plan_event(EventKind::Signal, 4, Some(5));
        assert!(!plan4.updates_state);
        assert_eq!(plan4.reason, Reason::SignalStaleIgnored);
    }

    #[test]
    fn wait_pending_and_satisfied() {
        let p = plan_event(EventKind::Wait, 3, None);
        assert_eq!(p.decision, Decision::WaitPending);
        assert_eq!(p.reason, Reason::WaitMissingSignal);

        let p2 = plan_event(EventKind::Wait, 3, Some(3));
        assert_eq!(p2.decision, Decision::WaitSatisfied);

        let p3 = plan_event(EventKind::Wait, 3, Some(2));
        assert_eq!(p3.reason, Reason::WaitBelowTarget);
    }

    #[test]
    fn fence_generation() {
        let p = plan_fence(FenceAction::Update, Domain::BlitFence, None);
        assert_eq!(p.update_value, FENCE_INITIAL_GENERATION);
        let p2 = plan_fence(FenceAction::Update, Domain::BlitFence, Some(1));
        assert_eq!(p2.update_value, 2);
        let p3 = plan_fence(FenceAction::Update, Domain::BlitFence, Some(u64::MAX));
        assert!(!p3.updates_state);
        assert_eq!(p3.reason, Reason::FenceUpdateAtMax);

        let w = plan_fence(FenceAction::Wait, Domain::ComputeFence, Some(4));
        assert_eq!(w.decision, Decision::WaitSatisfied);
        let w2 = plan_fence(FenceAction::Wait, Domain::RenderFence, None);
        assert_eq!(w2.decision, Decision::WaitPending);
    }

    #[test]
    fn fence_rejects_non_fence_domains() {
        for d in [Domain::Event, Domain::Unknown] {
            let bad = plan_fence(FenceAction::Update, d, None);
            assert_eq!(bad.decision, Decision::Invalid);
            assert_eq!(bad.reason, Reason::BadFenceDomain);
            assert!(!bad.updates_state);
        }
    }
}
