//! Indirect command buffers: executing a range of one, and the hint about one.
//!
//! # A command range is not a byte range
//!
//! `executeCommandsInBuffer:withRange:` names an `NSRange` of *commands*, and
//! the bytes those commands occupy are not derivable from it — the encoding of
//! one ICB command is a host and format question this layer cannot see. So the
//! range is carried in its own units, in its own type, and the ICB's
//! participation is against the whole buffer.
//!
//! That is coarser than the transfer classes manage and it is not a shortcut. A
//! range in commands turned into a range in bytes by assuming a stride would be
//! an assumption with a hazard edge resting on it.
//!
//! # Three judged operations, and the rest of the family is unresolved
//!
//! The two render-rail executions are implemented; the blit rail's optimize
//! hint is a proven no-op on a cell that says no host ICB is materialised. The
//! ICB copy, the ICB reset and the two compute-rail executions are unresolved
//! and get no payload — and the reason the copy and the reset are unresolved is
//! worth repeating, because it is the sharpest case in the ledger: dropping
//! either leaves the destination holding *stale commands*, which then execute.
//! That is not lost work; it is wrong work, and it is why they must not be
//! given a shape the model can run.

use crate::access::{AccessMode, Participation, ParticipationExtent, Participations};
use crate::bind::IndirectSource;
use crate::identity::ResourceId;

/// A half-open range of commands inside an indirect command buffer.
///
/// Its own type rather than [`crate::access::ByteRange`], and the difference is
/// the module's central claim: the units are commands. A function that accepted
/// either would let a command index reach a byte offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommandRange {
    pub location: u64,
    pub length: u64,
}

impl CommandRange {
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.length == 0
    }
}

/// What an ICB operation does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcbOp {
    /// Execute a literal range of commands.
    ExecuteRange {
        icb: ResourceId,
        commands: CommandRange,
    },
    /// Execute commands whose range comes from a buffer.
    ///
    /// The range's own encoding in that buffer is not established, so the
    /// argument read has no extent — the same shape the SPI dispatch takes in
    /// [`crate::compute`], and for the same reason.
    ExecuteIndirect {
        icb: ResourceId,
        arguments: IndirectSource,
    },
    /// Ask the device to prepare a range of an ICB for reuse.
    ///
    /// A hint. It costs speed and not semantics to skip, which is what its
    /// ledger row establishes — but the *skipping* is an executor's decision
    /// against the representation it actually materialised, so the operation
    /// reaches the model like any other. Same rule as the barriers.
    Optimize {
        icb: ResourceId,
        commands: CommandRange,
    },
}

impl IcbOp {
    /// The indirect command buffer this operation is about.
    #[must_use]
    pub const fn icb(&self) -> ResourceId {
        match *self {
            Self::ExecuteRange { icb, .. }
            | Self::ExecuteIndirect { icb, .. }
            | Self::Optimize { icb, .. } => icb,
        }
    }

    /// Whether the operation runs the commands.
    #[must_use]
    pub const fn executes(&self) -> bool {
        matches!(
            self,
            Self::ExecuteRange { .. } | Self::ExecuteIndirect { .. }
        )
    }

    /// The argument buffer, for the form whose range comes from one.
    #[must_use]
    pub const fn argument_read(&self) -> Option<IndirectSource> {
        match *self {
            Self::ExecuteIndirect { arguments, .. } => Some(arguments),
            Self::ExecuteRange { .. } | Self::Optimize { .. } => None,
        }
    }

    /// What this operation touches.
    ///
    /// The ICB is read whole — the command range is in commands and the bytes
    /// are not derivable from it — and the argument buffer, where there is one,
    /// is read with no extent for the same reason its encoding is unestablished.
    ///
    /// **What is deliberately absent: the resources the ICB's own commands
    /// touch.** Those are inside the buffer, they are the guest's to have
    /// declared through residency, and this device's residency rows are all
    /// unresolved. An operation here that claimed to know them would be
    /// claiming the contract that is open.
    #[must_use]
    pub fn participations(&self) -> Participations {
        let commands = Participation {
            resource: self.icb(),
            extent: ParticipationExtent::Whole,
            mode: if self.executes() {
                AccessMode::Read
            } else {
                // An optimize hint may rewrite the device's representation of
                // the commands. Nothing establishes that it does not, and a
                // read-only claim that turned out to be a write is a missing
                // edge rather than a redundant one.
                AccessMode::ReadWrite
            },
            api_stages: NO_STAGES,
        };
        Participations::pair(
            Some(commands),
            self.argument_read().map(|arguments| Participation {
                resource: arguments.buffer,
                extent: ParticipationExtent::Whole,
                mode: AccessMode::Read,
                api_stages: NO_STAGES,
            }),
        )
    }
}

/// An ICB operation declares no shader stage.
const NO_STAGES: u32 = 0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};
    use crate::operation::{classify, OperationClass, OperationHome};
    use reims_vgpu_protocol::closure::{Closure, Rail, LEDGER};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn every_op() -> [IcbOp; 3] {
        [
            IcbOp::ExecuteRange {
                icb: res(1),
                commands: CommandRange {
                    location: 2,
                    length: 3,
                },
            },
            IcbOp::ExecuteIndirect {
                icb: res(1),
                arguments: IndirectSource {
                    buffer: res(2),
                    offset: 0x40,
                },
            },
            IcbOp::Optimize {
                icb: res(1),
                commands: CommandRange {
                    location: 0,
                    length: 8,
                },
            },
        ]
    }

    /// Three judged operations in this class, and the payload has three forms.
    ///
    /// Counted from the ledger rather than asserted as a number alone, so a row
    /// that closes shows up here as a missing payload rather than as a silent
    /// gain.
    #[test]
    fn the_judged_indirect_command_operations_are_three() {
        let judged: Vec<_> = LEDGER
            .iter()
            .filter(|o| classify(o) == Some(OperationHome::Stream(OperationClass::IndirectCommand)))
            .map(|o| (o.rail, o.opcode.expect("carries an opcode")))
            .collect();
        assert_eq!(
            judged,
            vec![
                (Rail::Render, 0x14),
                (Rail::Render, 0x15),
                (Rail::Blit, 0x138)
            ]
        );
        assert_eq!(every_op().len(), judged.len());
    }

    /// The unresolved half of the family stays out, and the two whose failure
    /// mode is *executing stale commands* are named explicitly — those are the
    /// rows a future reader is most likely to want to wave through.
    #[test]
    fn the_mutating_indirect_operations_are_unresolved_and_have_no_payload() {
        for (rail, opcode) in [
            (Rail::Blit, 0x131), // copyIndirectCommandBuffer
            (Rail::Blit, 0x139), // resetCommandsInBuffer
            (Rail::Compute, 0xe4),
            (Rail::Compute, 0xe5),
        ] {
            let op = LEDGER
                .iter()
                .find(|o| o.rail == rail && o.opcode == Some(opcode))
                .unwrap_or_else(|| panic!("{rail:?} {opcode:#x} has no row"));
            assert!(
                matches!(op.closure, Closure::Unresolved { .. }),
                "{rail:?} {opcode:#x} is {} and now owes the model a payload",
                op.closure.name()
            );
            assert_eq!(classify(op), None);
        }
    }

    /// An execution reads the ICB whole, because the range is in commands.
    #[test]
    fn an_execution_reads_the_whole_command_buffer() {
        let op = IcbOp::ExecuteRange {
            icb: res(1),
            commands: CommandRange {
                location: 4,
                length: 2,
            },
        };
        let parts = op.participations();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].resource, res(1));
        assert_eq!(parts[0].extent, ParticipationExtent::Whole);
        assert_eq!(parts[0].mode, AccessMode::Read);
    }

    /// The indirect form reads a second buffer, with no extent.
    #[test]
    fn the_indirect_form_reads_its_argument_buffer() {
        let op = IcbOp::ExecuteIndirect {
            icb: res(1),
            arguments: IndirectSource {
                buffer: res(2),
                offset: 0x40,
            },
        };
        let parts = op.participations();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1].resource, res(2));
        assert_eq!(parts[1].mode, AccessMode::Read);
        assert_eq!(parts[1].extent, ParticipationExtent::Whole);
    }

    /// The optimize hint may rewrite the device's representation, so it is
    /// read-write rather than a read.
    #[test]
    fn the_optimize_hint_is_conservatively_a_write() {
        let op = IcbOp::Optimize {
            icb: res(1),
            commands: CommandRange::default(),
        };
        assert!(!op.executes());
        assert_eq!(op.participations()[0].mode, AccessMode::ReadWrite);
    }

    /// A command range is not a byte range, and the type says so: there is no
    /// conversion between them here at all.
    #[test]
    fn a_command_range_stays_in_commands() {
        let range = CommandRange {
            location: 0x1100,
            length: 0x2200,
        };
        assert!(!range.is_empty());
        assert!(CommandRange::default().is_empty());
        // The whole claim: nothing in this module turns one into a byte offset.
        for op in every_op() {
            for part in op.participations() {
                assert_eq!(part.extent, ParticipationExtent::Whole);
            }
        }
    }
}
