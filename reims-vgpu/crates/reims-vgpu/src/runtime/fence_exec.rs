//! Product-path event + encoder fence sync (event / blit / compute / render).
//!
//! Uses [`crate::runtime::plan::event_sync`] for planning and
//! [`DeviceState::fence_generations`] for storage. Unsatisfied waits are
//! soft-pending and do not block the drain (unified-memory in-order path).
//! Event timeouts are fail-closed as unsupported (no deferred timer).

use crate::model::{
    DeviceState, FENCE_DOMAIN_BLIT, FENCE_DOMAIN_COMPUTE, FENCE_DOMAIN_EVENT, FENCE_DOMAIN_RENDER,
};
use crate::runtime::plan::event_sync::{self, Decision, Domain, EventKind, FenceAction};
use reims_vgpu_protocol::decode::sync::EventRecord;
use reims_vgpu_protocol::sync::EventKind as RecordEventKind;

/// Outcome of a product-path event or encoder fence operation.
///
/// `Unsupported` **carries the check that refused**. Seven distinct causes reach
/// it — a bad fence domain, an event on the fence path, either wait-with-timeout
/// form, either invalid plan, an unknown event kind, and a blit reason forwarded
/// by the encoder remap — and while the reason was named in a hand-rolled
/// `format!` at each site, the *value* lost it the moment it was returned. So
/// `blit_exec`'s remap back into `BlitStatus` flattened all seven into one
/// `fence_unsupported` slug, and no caller could tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceStatus {
    Ok,
    /// Wait with no prior update/signal, or signal value not yet reached (soft).
    Pending,
    /// Zero ref.
    Missing,
    /// Refused; the payload is the registered slug naming which check refused.
    Unsupported(&'static str),
}

impl crate::observe::Refusal for FenceStatus {
    /// `Ok` and `Pending` are control flow. `Missing` is `ref == 0` — the
    /// genuinely-unbound case `AGENTS.md` carves out by name, and the guest polls
    /// it, so logging it would flood.
    ///
    /// One caveat a reader needs: `exec`'s blit-fence arm remaps
    /// `BlitFailure::MissingResource` into `Missing`, which is a real failure
    /// rather than an unbound ref. It is not silent — the blit status it came
    /// from carries `fence_missing` in its own `reason` — but the two meanings
    /// do share this variant.
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok | Self::Pending | Self::Missing => None,
            Self::Unsupported(slug) => Some(slug),
        }
    }
}

/// Emit a refused [`FenceStatus`] once per `(reason, ref)` and return it
/// unchanged, so a refusing site is one `return refused(..)` rather than a
/// `format!` beside a bare variant.
///
/// This replaces the file's own `fail_once` + reason-slug latch, which duplicated
/// [`crate::observe::Emit::fail_once`] down to the `HashSet`. The latch key
/// changes with the move, deliberately: it was the reason alone, so the *second*
/// ref to hit a class was silent and a guest with two bad fences was
/// indistinguishable from a guest with one. Keying on the ref is what `AGENTS.md`
/// prescribes, and the per-op fail counters (`event_ops_fail`, `*_fences_fail`)
/// still carry ongoing magnitude.
///
/// Runs on the drain worker, off the QEMU main core.
fn refused(
    status: FenceStatus,
    reference: u32,
    fields: impl FnOnce(crate::observe::Emit) -> crate::observe::Emit,
) -> FenceStatus {
    if let Some(e) = crate::observe::Emit::refusal("fence_exec_fail", &status) {
        fields(e).fail_once(u64::from(reference));
    }
    status
}

/// Map event-sync domain to the compact tag stored on [`DeviceState`].
pub fn domain_tag(domain: Domain) -> Option<u8> {
    match domain {
        Domain::Event => Some(FENCE_DOMAIN_EVENT),
        Domain::BlitFence => Some(FENCE_DOMAIN_BLIT),
        Domain::ComputeFence => Some(FENCE_DOMAIN_COMPUTE),
        Domain::RenderFence => Some(FENCE_DOMAIN_RENDER),
        Domain::Unknown => None,
    }
}

/// Execute fence update or wait on the given encoder domain (blit/compute/render).
pub fn execute_fence(
    state: &mut DeviceState,
    task_id: u32,
    domain: Domain,
    fence_ref: u32,
    action: FenceAction,
) -> FenceStatus {
    if fence_ref == 0 {
        return FenceStatus::Missing;
    }
    let Some(tag) = domain_tag(domain) else {
        return refused(
            FenceStatus::Unsupported("fence_domain_unknown"),
            fence_ref,
            |e| {
                e.field("task", task_id)
                    .field("domain", format!("{domain:?}"))
                    .field("action", format!("{action:?}"))
            },
        );
    };
    if domain == Domain::Event {
        return refused(
            FenceStatus::Unsupported("fence_event_in_fence_path"),
            fence_ref,
            |e| {
                e.field("task", task_id)
                    .field("action", format!("{action:?}"))
            },
        );
    }
    let current = state.fence_generation(task_id, tag, fence_ref);
    let plan = event_sync::plan_fence(action, domain, current);
    if plan.updates_state {
        state.set_fence_generation(task_id, tag, fence_ref, plan.update_value);
    }
    match plan.decision {
        Decision::SignalUpdate | Decision::SignalNoop | Decision::WaitSatisfied => FenceStatus::Ok,
        Decision::WaitPending => FenceStatus::Pending,
        Decision::Invalid => refused(
            FenceStatus::Unsupported("fence_plan_invalid"),
            fence_ref,
            |e| {
                e.field("task", task_id)
                    .field("domain", format!("{domain:?}"))
                    .field("action", format!("{action:?}"))
            },
        ),
    }
}

/// Execute a decoded ch-event segment command (signal / wait / wait-timeout).
///
/// Signal advances the Event-domain table with the explicit wire value (monotonic
/// advance only). Wait is satisfied when the stored value is present and
/// `>= target`. Wait-with-timeout is unsupported (no host timer). Soft-pending
/// waits do not block drain.
pub fn execute_event(state: &mut DeviceState, task_id: u32, cmd: &EventRecord) -> FenceStatus {
    if cmd.event_ref == 0 {
        return FenceStatus::Missing;
    }
    // Two kinds, and no third. The device's own decoder had an `Unknown` kind
    // that a well-formed record could carry, so "which of signal and wait is
    // this" was a question the executor had to re-ask and could be told the
    // answer to twice. `reims_vgpu_protocol::sync::EventKind` has the two the
    // wire has: a record that is neither never becomes an `EventRecord`.
    let kind = match cmd.kind {
        RecordEventKind::Signal => EventKind::Signal,
        RecordEventKind::Wait => EventKind::Wait,
    };
    let tag = FENCE_DOMAIN_EVENT;
    let current = state.fence_generation(task_id, tag, cmd.event_ref);
    let plan = event_sync::plan_event(kind, cmd.value, current);
    if plan.updates_state {
        state.set_fence_generation(task_id, tag, cmd.event_ref, plan.update_value);
    }
    match plan.decision {
        Decision::SignalUpdate | Decision::SignalNoop | Decision::WaitSatisfied => FenceStatus::Ok,
        Decision::WaitPending => FenceStatus::Pending,
        Decision::Invalid => refused(
            FenceStatus::Unsupported("event_plan_invalid"),
            cmd.event_ref,
            |e| e.field("task", task_id).field("value", cmd.value),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        DeviceId, FENCE_DOMAIN_BLIT, FENCE_DOMAIN_COMPUTE, FENCE_DOMAIN_EVENT, FENCE_DOMAIN_RENDER,
        PAGE_SHIFT_ARM64E,
    };
    /// An event record as the protocol crate lifts one.
    ///
    /// Built rather than encoded and decoded: these tests are about what
    /// `execute_event` does with a record, and running the bytes through a
    /// decoder to get one made every case here depend on the decoder agreeing
    /// about a layout it is not this file's job to check. Whether the lift is
    /// right is `reims_vgpu_protocol::decode::sync`'s own tests' question, and
    /// `runtime::decode::ledger` asks whether this device's rail matches it.
    ///
    /// There is no timeout parameter, because there is no bounded-wait record:
    /// `waitForEvent:value:timeoutMS:` is refused at the lift and cannot become
    /// an `EventRecord` at all.
    fn event_cmd(kind: RecordEventKind, event_ref: u32, value: u64) -> EventRecord {
        EventRecord {
            kind,
            event_ref,
            value,
        }
    }

    #[test]
    fn blit_compute_render_domains_independent() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        // Same ref id, three domains.
        assert_eq!(
            execute_fence(&mut state, 1, Domain::BlitFence, 5, FenceAction::Update),
            FenceStatus::Ok
        );
        assert_eq!(
            execute_fence(&mut state, 1, Domain::ComputeFence, 5, FenceAction::Update,),
            FenceStatus::Ok
        );
        assert_eq!(
            execute_fence(&mut state, 1, Domain::RenderFence, 5, FenceAction::Update,),
            FenceStatus::Ok
        );
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_BLIT, 5), Some(1));
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_COMPUTE, 5), Some(1));
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_RENDER, 5), Some(1));
        // Wait each domain satisfied independently.
        for d in [Domain::BlitFence, Domain::ComputeFence, Domain::RenderFence] {
            assert_eq!(
                execute_fence(&mut state, 1, d, 5, FenceAction::Wait),
                FenceStatus::Ok
            );
        }
        // Wait on never-updated ref is pending.
        assert_eq!(
            execute_fence(&mut state, 1, Domain::ComputeFence, 9, FenceAction::Wait),
            FenceStatus::Pending
        );
    }

    #[test]
    fn zero_ref_missing() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert_eq!(
            execute_fence(&mut state, 1, Domain::RenderFence, 0, FenceAction::Update,),
            FenceStatus::Missing
        );
        let cmd = event_cmd(RecordEventKind::Signal, 0, 1);
        assert_eq!(execute_event(&mut state, 1, &cmd), FenceStatus::Missing);
    }

    #[test]
    fn event_signal_then_wait() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let sig = event_cmd(RecordEventKind::Signal, 7, 100);
        assert_eq!(execute_event(&mut state, 1, &sig), FenceStatus::Ok);
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_EVENT, 7), Some(100));

        let wait_ok = event_cmd(RecordEventKind::Wait, 7, 100);
        assert_eq!(execute_event(&mut state, 1, &wait_ok), FenceStatus::Ok);

        // Wait for higher value is soft-pending.
        let wait_hi = event_cmd(RecordEventKind::Wait, 7, 101);
        assert_eq!(execute_event(&mut state, 1, &wait_hi), FenceStatus::Pending);

        // Advance signal, then wait satisfied.
        let sig2 = event_cmd(RecordEventKind::Signal, 7, 101);
        assert_eq!(execute_event(&mut state, 1, &sig2), FenceStatus::Ok);
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_EVENT, 7), Some(101));
        assert_eq!(execute_event(&mut state, 1, &wait_hi), FenceStatus::Ok);
    }

    #[test]
    fn event_stale_signal_noop_and_independent_of_fence() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert_eq!(
            execute_event(&mut state, 1, &event_cmd(RecordEventKind::Signal, 3, 50)),
            FenceStatus::Ok
        );
        // Stale / equal: no regression.
        assert_eq!(
            execute_event(&mut state, 1, &event_cmd(RecordEventKind::Signal, 3, 40)),
            FenceStatus::Ok
        );
        assert_eq!(
            execute_event(&mut state, 1, &event_cmd(RecordEventKind::Signal, 3, 50)),
            FenceStatus::Ok
        );
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_EVENT, 3), Some(50));

        // Same ref on blit fence domain is independent.
        assert_eq!(
            execute_fence(&mut state, 1, Domain::BlitFence, 3, FenceAction::Update),
            FenceStatus::Ok
        );
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_BLIT, 3), Some(1));
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_EVENT, 3), Some(50));
    }

    #[test]
    fn event_wait_missing_signal_pending() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let wait = event_cmd(RecordEventKind::Wait, 99, 1);
        assert_eq!(execute_event(&mut state, 1, &wait), FenceStatus::Pending);
    }

    /// Every refusal on these two paths must name a *different* check, or the
    /// coarse status is back and the log cannot say which one fired.
    ///
    /// Two of the causes this used to walk are gone rather than untested. The
    /// bounded event wait is refused at the lift now — `event_kind` gives
    /// `waitForEvent:value:timeoutMS:` no kind, so it never becomes a record —
    /// and the encoder-fence path's timeout arm was answering a `Decision`
    /// `plan_fence` cannot return. What is left is the set `execute_fence` can
    /// still produce.
    #[test]
    fn no_two_fence_checks_answer_with_the_same_reason() {
        use crate::observe::Refusal;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);

        let mut seen: Vec<&'static str> = Vec::new();
        let mut record = |st: FenceStatus| {
            seen.push(st.refusal().unwrap_or_else(|| {
                panic!("expected a refusal, got {st:?}");
            }));
        };

        // Fence path: an unknown domain, and an event ref on the encoder-fence
        // path.
        record(execute_fence(
            &mut state,
            1,
            Domain::Unknown,
            4,
            FenceAction::Update,
        ));
        record(execute_fence(
            &mut state,
            1,
            Domain::Event,
            4,
            FenceAction::Update,
        ));
        let mut unique = seen.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            seen.len(),
            "two fence checks share a reason: {seen:?}"
        );
        assert!(
            seen.contains(&"fence_domain_unknown") && seen.contains(&"fence_event_in_fence_path"),
            "the reasons no longer name their checks: {seen:?}"
        );
    }

    /// `Ok`, `Pending` and a zero ref are control flow, not refusals. Logging
    /// them would flood the always-on sink on every poll — I2's carve-out, made
    /// a compile-time `match` rather than a comment.
    #[test]
    fn success_pending_and_unbound_refs_are_never_logged() {
        use crate::observe::Refusal;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);

        assert_eq!(
            execute_fence(&mut state, 1, Domain::BlitFence, 5, FenceAction::Update).refusal(),
            None,
            "a successful update is not a refusal"
        );
        assert_eq!(
            execute_fence(&mut state, 1, Domain::ComputeFence, 9, FenceAction::Wait).refusal(),
            None,
            "a soft-pending wait is re-polled every drain; logging it floods"
        );
        assert_eq!(
            execute_fence(&mut state, 1, Domain::RenderFence, 0, FenceAction::Update,).refusal(),
            None,
            "ref==0 is the genuinely-unbound case AGENTS.md carves out"
        );
    }

    /// The encoder remap in `blit_exec` re-derives a blit reason from the fence
    /// status. It used to write a flat `fence_unsupported`, collapsing all seven
    /// causes into one slug; the reason now rides in the value, so the specific
    /// check survives the hop.
    #[test]
    fn a_refusal_carries_its_reason_across_the_blit_remap() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let st = execute_fence(&mut state, 1, Domain::Unknown, 7, FenceAction::Update);
        assert_eq!(st, FenceStatus::Unsupported("fence_domain_unknown"));

        let remapped = crate::runtime::blit_exec::blit_status_from_fence(st);
        assert_eq!(
            remapped.reason(),
            Some("fence_domain_unknown"),
            "the remap flattened the fence reason; got {remapped:?}"
        );
    }
}
