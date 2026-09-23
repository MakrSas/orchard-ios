//! The synchronisation vocabulary: which opcode is a fence, an event, or a
//! barrier, and what a barrier's scope word says.
//!
//! # Three rails number their fences separately, and that is the hazard
//!
//! `updateFence:` is `0x18` on the render rail, `0x13c` on the blit rail and
//! `0xd4` on the compute one, and those numbers are nowhere near each other. A
//! table that matched one rail's pair against another rail's records would read
//! a blit fence as a render one — the exact mistake the wire crate's per-module
//! opcode constants exist to prevent — so every lookup here takes the rail as
//! well as the opcode, and there is no rail-free form of it to reach for.
//!
//! # The compute pair is absent, not zero
//!
//! `0xd4`/`0xd5` are unresolved in the ledger, so no kind is returned for them
//! and the model refuses a compute fence rather than executing it as the render
//! one it resembles. That absence is asserted by a test, not left to be noticed.

use crate::closure::Rail;
use reims_vgpu_wire::ops::{blit as wire_blit, compute as wire_compute, render as wire_render};

/// The event-encoder opcodes.
///
/// Re-exported rather than restated. The event encoder is the one segment
/// family no fixture pins, so its records sit in the wire crate's unverified
/// tier — but they are still a byte layout with opcodes attached, and a byte
/// layout is that crate's whether or not an oracle backs it. Naming the numbers
/// twice would let the view and the meaning drift apart, which for
/// [`OPCODE_WAIT_EVENT_TIMEOUT`] would mean a refusal aimed at an opcode the
/// device no longer reads there.
///
/// What stays here is the meaning: which of the three is a signal, which is a
/// wait, and which is refused.
pub use reims_vgpu_wire::ops::event::{
    OPCODE_SIGNAL_EVENT, OPCODE_WAIT_EVENT, OPCODE_WAIT_EVENT_TIMEOUT,
};

/// Which side of a fence a record is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FenceKind {
    /// The encoder reaches this point and the fence is updated.
    Update,
    /// The encoder waits for the fence before continuing.
    Wait,
}

/// The fence side an opcode names on `rail`, if any.
#[must_use]
pub fn fence_kind(rail: Rail, opcode: u32) -> Option<FenceKind> {
    match rail {
        Rail::Render if opcode == wire_render::OPCODE_UPDATE_FENCE => Some(FenceKind::Update),
        Rail::Render if opcode == wire_render::OPCODE_WAIT_FOR_FENCE => Some(FenceKind::Wait),
        Rail::Blit if opcode == wire_blit::OPCODE_UPDATE_FENCE => Some(FenceKind::Update),
        Rail::Blit if opcode == wire_blit::OPCODE_WAIT_FOR_FENCE => Some(FenceKind::Wait),
        _ => None,
    }
}

/// Which side of an event a record is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventKind {
    /// Advance the event to a value.
    Signal,
    /// Wait, unbounded, for the event to reach a value.
    Wait,
}

/// The event side an opcode names, if any.
///
/// The timeout form returns `None`. It is a refused row: a bounded wait needs a
/// clock this device does not run against the guest's, and executing it as the
/// unbounded wait it is not would turn a guest's timeout into a hang.
#[must_use]
pub fn event_kind(rail: Rail, opcode: u32) -> Option<EventKind> {
    if rail != Rail::Event {
        return None;
    }
    match opcode {
        OPCODE_SIGNAL_EVENT => Some(EventKind::Signal),
        OPCODE_WAIT_EVENT => Some(EventKind::Wait),
        _ => None,
    }
}

/// What a barrier record names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BarrierKind {
    /// A counted list of resource refs.
    Resources,
    /// A scope word covering a whole class of memory.
    Scope,
    /// The render rail's `textureBarrier`, which names nothing at all.
    Texture,
}

/// The barrier an opcode names on `rail`, if any.
#[must_use]
pub fn barrier_kind(rail: Rail, opcode: u32) -> Option<BarrierKind> {
    match rail {
        Rail::Render if opcode == wire_render::OPCODE_MEMORY_BARRIER_RESOURCES => {
            Some(BarrierKind::Resources)
        }
        Rail::Render if opcode == wire_render::OPCODE_MEMORY_BARRIER_SCOPE => {
            Some(BarrierKind::Scope)
        }
        Rail::Render if opcode == wire_render::OPCODE_TEXTURE_BARRIER => Some(BarrierKind::Texture),
        Rail::Compute if opcode == wire_compute::OPCODE_MEMORY_BARRIER_RESOURCES => {
            Some(BarrierKind::Resources)
        }
        Rail::Compute if opcode == wire_compute::OPCODE_MEMORY_BARRIER_SCOPE => {
            Some(BarrierKind::Scope)
        }
        _ => None,
    }
}

/// `MTLBarrierScope`, as the wire carries it.
///
/// Three bits are established, each by a fixture that set it and moved the byte
/// alone: buffers, textures, and render targets. Nothing above them is defined
/// by the API, which is why [`Self::undeclared_bits`] exists — a value outside
/// the set is a guest asking for something this vocabulary does not name, and
/// masking it down to its declared neighbours would report it as a scope the
/// device is sure it understands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BarrierScope(pub u32);

impl BarrierScope {
    pub const BUFFERS: u32 = 1 << 0;
    pub const TEXTURES: u32 = 1 << 1;
    pub const RENDER_TARGETS: u32 = 1 << 2;

    const DECLARED: u32 = Self::BUFFERS | Self::TEXTURES | Self::RENDER_TARGETS;

    /// Bits outside the declared set, or zero.
    #[must_use]
    pub const fn undeclared_bits(self) -> u32 {
        self.0 & !Self::DECLARED
    }

    /// Whether the scope names any memory at all.
    ///
    /// A zero scope is a barrier over nothing. It is legal to send and it
    /// orders nothing, which is different from a barrier the device chose not
    /// to honour.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{Closure, LEDGER};

    /// Every fence, event and barrier this module names is a judged operation.
    #[test]
    fn every_named_operation_is_judged() {
        let named: &[(Rail, u32)] = &[
            (Rail::Render, wire_render::OPCODE_UPDATE_FENCE),
            (Rail::Render, wire_render::OPCODE_WAIT_FOR_FENCE),
            (Rail::Blit, wire_blit::OPCODE_UPDATE_FENCE),
            (Rail::Blit, wire_blit::OPCODE_WAIT_FOR_FENCE),
            (Rail::Render, wire_render::OPCODE_MEMORY_BARRIER_RESOURCES),
            (Rail::Render, wire_render::OPCODE_MEMORY_BARRIER_SCOPE),
            (Rail::Render, wire_render::OPCODE_TEXTURE_BARRIER),
            (Rail::Compute, wire_compute::OPCODE_MEMORY_BARRIER_RESOURCES),
            (Rail::Compute, wire_compute::OPCODE_MEMORY_BARRIER_SCOPE),
            (Rail::Event, OPCODE_SIGNAL_EVENT),
            (Rail::Event, OPCODE_WAIT_EVENT),
        ];
        for &(rail, opcode) in named {
            let op = LEDGER
                .iter()
                .find(|o| o.rail == rail && o.opcode == Some(opcode))
                .unwrap_or_else(|| panic!("{rail:?} {opcode:#x} has no ledger row"));
            assert!(
                !op.closure.blocks_cutover(),
                "{rail:?} {opcode:#x} is {}",
                op.closure.name()
            );
        }
    }

    /// The compute fence pair is unresolved, so it has no kind. A model that
    /// returned the render pair's answer here would execute a compute fence as
    /// a render one on the strength of the two selectors sharing a name.
    #[test]
    fn the_unresolved_compute_fence_pair_has_no_kind() {
        for opcode in [0xd4, 0xd5] {
            assert_eq!(fence_kind(Rail::Compute, opcode), None);
            let row = LEDGER
                .iter()
                .find(|o| o.rail == Rail::Compute && o.opcode == Some(opcode))
                .expect("row");
            assert!(
                matches!(row.closure, Closure::Unresolved { .. }),
                "if the compute pair has been resolved, this module owes it a kind"
            );
        }
    }

    /// A rail's fence numbers are its own, and no rail answers another's.
    #[test]
    fn no_rail_answers_another_rails_fence_numbers() {
        let render = [
            wire_render::OPCODE_UPDATE_FENCE,
            wire_render::OPCODE_WAIT_FOR_FENCE,
        ];
        let blit = [
            wire_blit::OPCODE_UPDATE_FENCE,
            wire_blit::OPCODE_WAIT_FOR_FENCE,
        ];
        for opcode in render {
            assert!(fence_kind(Rail::Render, opcode).is_some());
            assert_eq!(fence_kind(Rail::Blit, opcode), None);
            assert_eq!(fence_kind(Rail::Compute, opcode), None);
        }
        for opcode in blit {
            assert!(fence_kind(Rail::Blit, opcode).is_some());
            assert_eq!(fence_kind(Rail::Render, opcode), None);
        }
    }

    /// The bounded wait is refused and stays out of the vocabulary.
    #[test]
    fn the_timeout_wait_has_no_kind() {
        assert_eq!(event_kind(Rail::Event, OPCODE_WAIT_EVENT_TIMEOUT), None);
        let row = LEDGER
            .iter()
            .find(|o| o.rail == Rail::Event && o.opcode == Some(OPCODE_WAIT_EVENT_TIMEOUT))
            .expect("row");
        assert!(matches!(row.closure, Closure::Refused { .. }));
    }

    /// The event window is contiguous, and the opcodes below and above it are
    /// not events. The blit fence pair is the pointed case: real opcodes in a
    /// different space that an event decoder must not accept.
    #[test]
    fn the_event_window_is_three_contiguous_opcodes_and_nothing_else() {
        assert_eq!(OPCODE_SIGNAL_EVENT, OPCODE_WAIT_EVENT + 1);
        assert_eq!(OPCODE_WAIT_EVENT_TIMEOUT, OPCODE_SIGNAL_EVENT + 1);
        for outside in [
            OPCODE_WAIT_EVENT - 1,
            OPCODE_WAIT_EVENT_TIMEOUT + 1,
            wire_blit::OPCODE_UPDATE_FENCE,
            wire_blit::OPCODE_WAIT_FOR_FENCE,
        ] {
            assert_eq!(event_kind(Rail::Event, outside), None, "{outside:#x}");
        }
    }

    #[test]
    fn an_undeclared_scope_bit_does_not_become_its_neighbours() {
        assert_eq!(BarrierScope(BarrierScope::BUFFERS).undeclared_bits(), 0);
        assert_eq!(BarrierScope(0x8).undeclared_bits(), 0x8);
        assert_eq!(
            BarrierScope(BarrierScope::BUFFERS | 0x80).undeclared_bits(),
            0x80
        );
        assert!(BarrierScope::default().is_empty());
    }

    /// The scope the serial-dispatch barrier emits is buffers and textures
    /// together, which is what makes it a two-bit value rather than a
    /// single-bit one.
    #[test]
    fn the_serial_dispatch_scope_is_two_declared_bits() {
        let serial = BarrierScope(3);
        assert_eq!(serial.undeclared_bits(), 0);
        assert_eq!(serial.0, BarrierScope::BUFFERS | BarrierScope::TEXTURES);
    }
}
