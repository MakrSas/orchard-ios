//! The FIFO packet half of the refusal-closure ledger.
//!
//! [`crate::closure`] records what the device owes the guest for every
//! *serializer* operation — the records inside an EXEC's command stream. Those
//! are only half of what a guest sends. The other half arrives as FIFO packets:
//! task and channel lifecycle, resource mapping and teardown, display and
//! cursor control, queries whose replies the guest blocks on, and the EXEC
//! packet that carries the stream at all. The same four outcomes apply and for
//! the same reason, so the vocabulary is shared and only the key changes.
//!
//! # Why the key is a channel and an opcode
//!
//! One flat 16-bit opcode space is dispatched by two tables. `0x2d` is the
//! Monterey-era device-info query on the root channel and a retired slot on a
//! child channel, and both readings are correct — so an opcode alone is not a
//! key here any more than it is one rail over.
//!
//! # The three kinds of slot
//!
//! The reference host bounds the opcode at [`OPCODE_CEILING`] before indexing
//! its table, which splits the space three ways and the ledger carries two of
//! them. A slot with a command is a row. A slot the reference host routes to
//! its one shared retired handler is a row too, and a
//! [`crate::closure::Closure::ProvenNoOp`] one: that handler accepts the
//! packet, does nothing with the payload and retires the stamps, so doing
//! exactly that is fidelity rather than a gap. An unassigned slot is neither —
//! it has no command to judge, and a packet carrying one is a guest asking for
//! something this host generation does not have.

use crate::closure::{Closure, Counts};

/// Which dispatch table a packet is routed through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Channel {
    /// The device's own channel: task, FIFO and object-list lifecycle, and the
    /// device-info queries.
    Root,
    /// A guest-defined channel: everything a task submits.
    Child,
}

impl Channel {
    pub const ALL: &'static [Channel] = &[Channel::Root, Channel::Child];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Child => "child",
        }
    }
}

/// The highest opcode the reference host will index its dispatch table with.
///
/// Above it there is no handler at all, so a packet carrying one is a corrupt
/// header or a desynced ring rather than an unimplemented command — which is a
/// different thing to report and a different thing to fix.
pub const OPCODE_CEILING: u16 = 0x40;

/// One FIFO packet class and its recorded outcome.
#[derive(Clone, Copy, Debug)]
pub struct Packet {
    pub channel: Channel,
    pub opcode: u16,
    /// The protocol's own name for the command, or `retired slot` for a slot
    /// the reference host routes to its shared retired handler.
    pub name: &'static str,
    pub closure: Closure,
}

/// Tally the whole packet ledger, or one channel's part of it.
pub fn counts(channel: Option<Channel>) -> Counts {
    let mut c = Counts::default();
    for p in LEDGER
        .iter()
        .filter(|p| channel.is_none_or(|ch| p.channel == ch))
    {
        match p.closure {
            Closure::Implemented { .. } => c.implemented += 1,
            Closure::ProvenNoOp { .. } => c.proven_noop += 1,
            Closure::Refused { .. } => c.refused += 1,
            Closure::Unresolved { .. } => c.unresolved += 1,
        }
    }
    c
}

/// Every packet class whose outcome is not established.
pub fn blocking() -> impl Iterator<Item = &'static Packet> {
    LEDGER.iter().filter(|p| p.closure.blocks_cutover())
}

/// The row for one packet class, if the ledger has one.
pub fn find(channel: Channel, opcode: u16) -> Option<&'static Packet> {
    LEDGER
        .iter()
        .find(|p| p.channel == channel && p.opcode == opcode)
}

/// The retired slots, whose one shared handler is the whole contract.
const RETIRED: Closure = Closure::ProvenNoOp {
    cell: "the reference host's one shared retired-command handler",
    evidence: "that handler accepts the packet, does nothing with the payload and retires the \
               stamps, so doing exactly that is fidelity. The record exists to say a guest is \
               still emitting a command its own host generation retired",
};

/// The ledger. See [`crate::closure`] for what the four outcomes mean and what
/// may not be spelled as one.
pub const LEDGER: &[Packet] = &[
    // ---- root channel ----------------------------------------------------
    Packet {
        channel: Channel::Root,
        opcode: 0x01,
        name: "CmdDisplaySetSharedStatePage",
        closure: Closure::Implemented {
            evidence: "registers the display pipe's shared-state page; the same command as the \
                       child form and not a second meaning for the number",
        },
    },
    Packet {
        channel: Channel::Root,
        opcode: 0x20,
        name: "CmdDeleteTask",
        closure: Closure::Implemented {
            evidence: "retires the task slot, so a later DefineTask2 cannot inherit the previous \
                       task's page directory",
        },
    },
    Packet {
        channel: Channel::Root,
        opcode: 0x2d,
        name: "CmdDeviceInfo (Monterey)",
        closure: Closure::Implemented {
            evidence: "the Monterey-era device-info query, still answered for a guest old enough \
                       to ask; the root arm runs before the retired-slot arm that also claims \
                       this number on a child channel",
        },
    },
    Packet {
        channel: Channel::Root,
        opcode: 0x30,
        name: "CmdDefineFIFO",
        closure: Closure::Implemented {
            evidence: "opens a channel and its ring; the channel is the submission ordering \
                       domain every EXEC on it belongs to",
        },
    },
    Packet {
        channel: Channel::Root,
        opcode: 0x31,
        name: "CmdFreeFIFO",
        closure: Closure::Implemented {
            evidence: "closes a channel and its ring",
        },
    },
    Packet {
        channel: Channel::Root,
        opcode: 0x33,
        name: "CmdSetObjectList",
        closure: Closure::Implemented {
            evidence: "establishes the task's object-list, the ref space every decoded command \
                       resolves its objects out of",
        },
    },
    Packet {
        channel: Channel::Root,
        opcode: 0x38,
        name: "CmdDefineTask2",
        closure: Closure::Implemented {
            evidence: "establishes a task and its page directory",
        },
    },
    Packet {
        channel: Channel::Root,
        opcode: 0x3a,
        name: "CmdDeviceInfo (Tahoe)",
        closure: Closure::Implemented {
            evidence: "the current device-info query, answered from the device's declared \
                       capabilities",
        },
    },
    // ---- child channel ---------------------------------------------------
    Packet {
        channel: Channel::Child,
        opcode: 0x00,
        name: "CmdDebug",
        closure: Closure::Unresolved {
            question: "a host-side trace marker carrying whatever the guest wants logged. Nothing \
                       in the device model moves, but the payload contract has not been decoded, \
                       so 'nothing is owed' is a reasonable guess rather than a proof",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x01,
        name: "CmdDisplaySetSharedStatePage",
        closure: Closure::Implemented {
            evidence: "registers the display pipe's shared-state page; without it the display \
                       never comes online",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x02,
        name: "CmdDisplayOnlineAck",
        closure: Closure::Implemented {
            evidence: "the guest acknowledging the display came online",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x04,
        name: "CmdCursorGlyph",
        closure: Closure::Implemented {
            evidence: "loads the cursor image and hands it to the host window",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x05,
        name: "CmdCursorShow",
        closure: Closure::Implemented {
            evidence: "shows or hides the cursor at its sampled position",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x06,
        name: "CmdDisplayTransaction2_DEPRECATED",
        closure: Closure::Implemented {
            evidence: "one of the three present forms; they differ only in where the surface word \
                       sits in the trailer",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x07,
        name: "CmdDisplayTransaction3",
        closure: Closure::Implemented {
            evidence: "the current present form; its trailer carries a gamma table beside the \
                       surface",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x08,
        name: "CmdDisplaySwapMapping",
        closure: Closure::Implemented {
            evidence: "the arm/EFI-era present form; same present path as 0x06 and 0x07",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x09,
        name: "CmdDisplaySleepState",
        closure: Closure::Unresolved {
            question: "a display entering or leaving sleep. Nothing in this device's display \
                       model tracks it, so a guest that sleeps a panel and finds it still lit is \
                       looking at this slot",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x0a,
        name: "CmdDisplaySetProperties",
        closure: Closure::Unresolved {
            question: "a property key, value and word count forwarded to the display nub on the \
                       reference host. No property is applied here and which ones a guest sets is \
                       unmeasured",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x1e,
        name: "CmdNOP",
        closure: Closure::Implemented {
            evidence: "a fence carrying stamps and no payload. Retiring the stamps is the whole \
                       obligation and the drain does that for every accepted packet; a payload \
                       here would be a command that grew a form this arm does not decode, and it \
                       is named rather than dropped",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x20,
        name: "CmdDeleteTask",
        closure: Closure::Implemented {
            evidence: "the child form of the root command at the same opcode",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x22,
        name: "CmdUnmapMemory",
        closure: Closure::Implemented {
            evidence: "retires a task GPU-VA mapping",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x25,
        name: "CmdDeleteResource",
        closure: Closure::Implemented {
            evidence: "retires the object-table entry the resource layer allocated. Not the same \
                       command as 0x28, which names a different ref space",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x28,
        name: "CmdDeleteObject",
        closure: Closure::Implemented {
            evidence: "retires the object-list name the destroy record's reference holds, and the \
                       task-local registry entry its kind keeps. The record is self-describing --- \
                       one of eleven opcodes over an identical twelve-byte body --- so the kind is \
                       the record's and each kind is settled on its own row on the serializer \
                       rail; a kind whose row is not settled refuses this packet by name through \
                       `lifecycle::ResolveRefusal::UnsettledDestroy` rather than holding the whole \
                       class out of the model. Five kinds are in that state --- buffer, texture, \
                       heap, rasterization rate map and indirect command buffer --- and no driven \
                       boot has ever seen one. What settled the row was the ref space, measured \
                       twice. First: of 2203 destroys this device retires something for, 2123 \
                       named no live object-list entry, which reads like a second namespace and is \
                       not --- `objects::resolve_sampler_state` constructs a sampler by looking \
                       that same number up in the guest's object list and requiring a \
                       serializer-object entry there, so the spaces are one and the reading is \
                       about time: the guest clears its slot before it sends the destroy. Second, \
                       the question that follows from that: could the semantic model *name* the \
                       reference, which is answered from the device's own namespace before the \
                       guest's list is consulted --- zero of 1984, because nothing on the sampler \
                       construction path had ever declared a name. Taking the name at \
                       construction moved it to 2006 named against 2006 retired, on a driven \
                       macos-15 boot with no line on the failure channel",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x33,
        name: "CmdSetObjectList",
        closure: Closure::Implemented {
            evidence: "the child form of the root command at the same opcode",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x34,
        name: "CmdInvalidateResources",
        closure: Closure::Implemented {
            evidence: "drops the device's cached view of the named resources' guest pages",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x35,
        name: "CmdSynchronizeResources",
        closure: Closure::Implemented {
            evidence: "the guest is about to CPU-read the named resources, so every guest-page \
                       write already submitted has to have executed before the packet completes",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x36,
        name: "CmdDeleteIOSurfaceBacking2",
        closure: Closure::Implemented {
            evidence: "retires an IOSurface backing and the mappings that named it",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x37,
        name: "CmdExecIndirect2",
        closure: Closure::Implemented {
            evidence: "the GPU-work packet: a resource table and a counted, ordered list of \
                       serialized-storage chunks, whose records are the operations \
                       `crate::closure` judges",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x38,
        name: "CmdDefineTask2",
        closure: Closure::Implemented {
            evidence: "the child form of the root command at the same opcode",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x39,
        name: "CmdMapMemory2",
        closure: Closure::Implemented {
            evidence: "establishes a task GPU-VA mapping over guest pages",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x3b,
        name: "CmdGetComputeInfo",
        closure: Closure::Implemented {
            evidence: "a query whose reply is written before the stamp retires, because the guest \
                       blocks on it — a compute pipeline creation stalls without the answer",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x3c,
        name: "CmdReplacePhysical",
        closure: Closure::Implemented {
            evidence: "the guest re-pointing one task-local resource at different guest pages; \
                       the resource's held address resolution and authoritative frame are retired \
                       so nothing keeps reading the old pages",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x3d,
        name: "CmdDelay",
        closure: Closure::Unresolved {
            question: "the guest asking the channel to be held before the next command runs. This \
                       device continues immediately, which reorders nothing — the stamps still \
                       retire in submission order — but a guest that used the delay to let \
                       something settle does not get it, and what it was waiting for is unknown",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x3e,
        name: "CmdSynchronizeAndDiscardResources",
        closure: Closure::Implemented {
            evidence: "the synchronise half is 0x35's obligation and is met. The discard half is \
                       a hint that the contents are no longer needed; ignoring it costs memory \
                       and not correctness, which is the same reading 0x3f carries",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x3f,
        name: "CmdDiscardResources",
        closure: Closure::Implemented {
            evidence: "releases each named resource's transfer backing; prepare or synchronize \
                       recreates it lazily. A malformed payload is still named, because it says \
                       the guest and this device disagree about a record layout the two commands \
                       that do act on it share",
        },
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x40,
        name: "CmdHeapTextureSizeAndAlign",
        closure: Closure::Implemented {
            evidence: "a query answered through the reply GVA the request names. Nothing in the \
                       device model moves, but the guest blocks on the reply, so a refusal is a \
                       stall rather than a dropped command",
        },
    },
    // ---- the retired slots ----------------------------------------------
    Packet {
        channel: Channel::Child,
        opcode: 0x03,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x1f,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x21,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x23,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x24,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x26,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x27,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x29,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x2a,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x2b,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x2c,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x2d,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x2e,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x2f,
        name: "retired slot",
        closure: RETIRED,
    },
    Packet {
        channel: Channel::Child,
        opcode: 0x32,
        name: "retired slot",
        closure: RETIRED,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn no_packet_class_is_judged_twice() {
        let mut seen = BTreeSet::new();
        for p in LEDGER {
            assert!(
                seen.insert((p.channel, p.opcode)),
                "two rows for {:?} {:#04x}",
                p.channel,
                p.opcode
            );
        }
    }

    /// A row above the ceiling is a judgement about a value the reference host
    /// refuses before it reaches a table, so it can never be acted on.
    #[test]
    fn every_row_is_inside_the_dispatch_ceiling() {
        for p in LEDGER {
            assert!(
                p.opcode <= OPCODE_CEILING,
                "{:?} {:#04x} is above the dispatch ceiling",
                p.channel,
                p.opcode
            );
        }
    }

    #[test]
    fn every_row_states_its_reasoning() {
        for p in LEDGER {
            let (a, b) = match p.closure {
                Closure::Implemented { evidence } => (evidence, evidence),
                Closure::ProvenNoOp { cell, evidence } => (cell, evidence),
                Closure::Refused { route, evidence } => (route, evidence),
                Closure::Unresolved { question } => (question, question),
            };
            assert!(
                a.len() > 8 && b.len() > 8 && !p.name.is_empty(),
                "{:?} {:#04x} does not say why",
                p.channel,
                p.opcode
            );
        }
    }

    #[test]
    fn counts_cover_the_whole_ledger() {
        assert_eq!(counts(None).total(), LEDGER.len());
        let per_channel: usize = Channel::ALL.iter().map(|&c| counts(Some(c)).total()).sum();
        assert_eq!(per_channel, LEDGER.len());
        assert_eq!(counts(None).unresolved, blocking().count());
    }

    #[test]
    fn find_answers_from_the_ledger() {
        for p in LEDGER {
            assert_eq!(find(p.channel, p.opcode).expect("findable").name, p.name);
        }
        assert!(find(Channel::Root, 0x00).is_none());
    }

    #[test]
    fn report_the_blocking_set() {
        let c = counts(None);
        println!(
            "packet ledger: {} classes — {} implemented, {} proven no-op, {} refused, {} unresolved",
            c.total(),
            c.implemented,
            c.proven_noop,
            c.refused,
            c.unresolved
        );
        for p in blocking() {
            println!(
                "  BLOCKING {:5} {:#04x} {}",
                p.channel.name(),
                p.opcode,
                p.name
            );
        }
    }
}
