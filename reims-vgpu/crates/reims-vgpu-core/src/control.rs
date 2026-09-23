//! The control transaction: the packets whose whole effect is on the device's
//! own plumbing, and the ones whose whole effect is retiring a stamp.
//!
//! # The class that could rot into a bucket
//!
//! [`crate::transaction::PayloadClass::Control`] is the one payload class a
//! command could be dropped into for lack of anywhere better, which is why
//! [`crate::transaction::classify`] refuses an unestablished command outright
//! rather than calling it control. This module is the other half of that
//! defence: `Control` has an exhaustive vocabulary too, checked against the
//! ledger, so a command cannot arrive at the class and then quietly have no
//! behavior. If a new opcode becomes established and reaches `Control`, the
//! totality test asks what it is.
//!
//! # An acknowledged no-op is an obligation, not an absence
//!
//! Fifteen of the twenty-three control packets are slots a previous version of
//! this interface used and this one does not. They still arrive, and each one
//! still carries stamp waits and a completion word — so "does nothing" means
//! *the payload* does nothing, and the transaction still waits, still takes its
//! ordering position, and still publishes. That is the same envelope every
//! other class has, which is why they are transactions here rather than a
//! special case at ingress: the moment a retired slot got a shortcut, its
//! completion would stop being ordered against the channel like everything
//! else's. `CmdNOP` is the same shape and is not retired — it is the command
//! whose entire purpose is that envelope.
//!
//! # Channel definition is a real transition and not bookkeeping
//!
//! A channel is the submission ordering domain every packet on it belongs to.
//! Opening one and freeing one are therefore transitions of
//! [`crate::session::SessionModel`]: a packet naming a domain no definition
//! opened is refused at ingress, because admitting it would give it an ordering
//! position and a completion obligation in a publication order nothing drains,
//! and the guest would wait on that word forever. Freeing refuses while
//! positions are outstanding for the mirror-image reason.
//!
//! # Display and cursor state is named, and its content is not here
//!
//! A shared state page is a backing and a window of it; a cursor glyph is
//! pixels somewhere and a hotspot. Which of those this crate can hold is
//! decided by what it can name: the page's [`crate::access::BackingId`] and
//! range, yes; the image, no. So the display and cursor kinds are named,
//! carried, and left for the layer that owns a display — and naming them is not
//! nothing, because it is what stops them being read as "control, therefore
//! nothing happens".

use crate::identity::ChannelId;
use crate::transaction::{classify, is_acknowledged_noop, PayloadClass};
use reims_vgpu_protocol::packets::Channel;

/// Which control command a packet is.
///
/// Exhaustive over the packet classes [`classify`] calls
/// [`PayloadClass::Control`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ControlKind {
    /// The page the display's shared state is read from and written to.
    DisplaySharedStatePage,
    /// The guest acknowledging that a display came online.
    DisplayOnlineAck,
    /// New cursor pixels and their hotspot.
    CursorGlyph,
    /// Whether the cursor is drawn.
    CursorShow,
    /// Open a submission ordering domain and its ring.
    DefineChannel,
    /// End one.
    FreeChannel,
    /// The command whose entire obligation is its own envelope: wait for its
    /// stamp waits, take its position, publish its completion.
    Nop,
    /// A slot a previous version of this interface used. The payload does
    /// nothing; the envelope is the same as every other transaction's.
    RetiredSlot,
}

impl ControlKind {
    /// The control command a packet is, or `None` if it is not a control
    /// packet.
    #[must_use]
    pub fn of(channel: Channel, opcode: u16) -> Option<Self> {
        if classify(channel, opcode) != Some(PayloadClass::Control) {
            return None;
        }
        use Channel::{Child, Root};
        Some(match (channel, opcode) {
            (Root | Child, 0x01) => Self::DisplaySharedStatePage,
            (Child, 0x02) => Self::DisplayOnlineAck,
            (Child, 0x04) => Self::CursorGlyph,
            (Child, 0x05) => Self::CursorShow,
            (Root, 0x30) => Self::DefineChannel,
            (Root, 0x31) => Self::FreeChannel,
            (Child, 0x1e) => Self::Nop,
            // Everything else the ledger judged into this class is a retired
            // slot, and the ledger is asked rather than the opcode: a numeric
            // range would adopt a live command that happened to land inside it.
            _ if is_acknowledged_noop(channel, opcode) => Self::RetiredSlot,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DisplaySharedStatePage => "display_shared_state_page",
            Self::DisplayOnlineAck => "display_online_ack",
            Self::CursorGlyph => "cursor_glyph",
            Self::CursorShow => "cursor_show",
            Self::DefineChannel => "define_channel",
            Self::FreeChannel => "free_channel",
            Self::Nop => "nop",
            Self::RetiredSlot => "retired_slot",
        }
    }

    /// Whether this command's payload changes nothing at all.
    ///
    /// True for the retired slots and for `CmdNOP`. It is deliberately **not**
    /// a licence to skip the transaction: the envelope — the waits, the
    /// ordering position, the completion word — is owed identically, and this
    /// only says the payload has no other effect.
    #[must_use]
    pub const fn payload_is_inert(self) -> bool {
        matches!(self, Self::RetiredSlot | Self::Nop)
    }

    /// The channel-lifetime transition this command is, if it is one.
    #[must_use]
    pub const fn channel_transition(self) -> Option<ChannelTransition> {
        match self {
            Self::DefineChannel => Some(ChannelTransition::Open),
            Self::FreeChannel => Some(ChannelTransition::Free),
            _ => None,
        }
    }
}

/// What a channel-lifetime command does to a submission domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelTransition {
    Open,
    Free,
}

/// One resolved control operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlOp {
    /// Open or free a submission domain. Applied through
    /// [`crate::session::SessionModel::open_channel`] and
    /// [`crate::session::SessionModel::retire_channel`], which own the
    /// ordering consequences.
    Channel {
        transition: ChannelTransition,
        domain: ChannelId,
    },
    /// Display or cursor state. Named here and owned by the layer that has a
    /// display; the kind travels so a reader cannot mistake it for a command
    /// with no effect.
    Display { kind: ControlKind },
    /// The payload does nothing. The envelope is owed in full.
    Inert { kind: ControlKind },
}

impl ControlOp {
    #[must_use]
    pub const fn kind(self) -> ControlKind {
        match self {
            Self::Channel {
                transition: ChannelTransition::Open,
                ..
            } => ControlKind::DefineChannel,
            Self::Channel {
                transition: ChannelTransition::Free,
                ..
            } => ControlKind::FreeChannel,
            Self::Display { kind } | Self::Inert { kind } => kind,
        }
    }
}

/// Why a control packet's bytes did not become an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveRefusal {
    /// The packet is not a control packet at all.
    NotControl { channel: Channel, opcode: u16 },
    /// A channel-lifetime command whose payload cannot hold its one word.
    Payload(reims_vgpu_protocol::fifo::ShortPayload),
}

impl ResolveRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NotControl { .. } => "control_not_a_control_packet",
            Self::Payload(_) => reims_vgpu_protocol::fifo::ShortPayload::SLUG,
        }
    }
}

/// Turn one control packet into the operation it names.
///
/// **The join between the ledger's twenty-three control rows and the three-arm
/// vocabulary that describes them.** [`ControlKind::of`] said which command a
/// packet is and [`ControlOp`] said what the model does about it, and nothing
/// carried a packet from one to the other — so `Payload::Control` could be
/// described and could not be built from a guest's bytes.
///
/// Only one of the twenty-three reads a payload word: a channel definition or
/// free names the domain it opens or ends. The rest carry nothing this crate
/// reads — a cursor's pixels and a display's shared-state page belong to the
/// layer that has a display, and the retired slots and `CmdNOP` carry nothing
/// at all — so their operations are the kind and no more. That is not a
/// shortcut: it is what [`ControlOp::Display`] and [`ControlOp::Inert`] already
/// say, and this function is where the claim becomes checkable against every
/// row the ledger judged.
///
/// # Errors
///
/// [`ResolveRefusal`]: a packet that is not control, or a channel-lifetime
/// command too short to name a domain.
pub fn resolve(channel: Channel, opcode: u16, payload: &[u8]) -> Result<ControlOp, ResolveRefusal> {
    let Some(kind) = ControlKind::of(channel, opcode) else {
        return Err(ResolveRefusal::NotControl { channel, opcode });
    };
    // Everything but the two channel-lifetime commands: the kind is the whole
    // operation. Both questions are asked of `ControlKind` rather than
    // restated here, so a row that stops being a channel command or stops
    // being a no-op changes one answer and not two.
    let Some(transition) = kind.channel_transition() else {
        return Ok(if kind.payload_is_inert() {
            ControlOp::Inert { kind }
        } else {
            ControlOp::Display { kind }
        });
    };
    let domain = reims_vgpu_protocol::fifo::decode_channel_lifetime(payload)
        .map_err(ResolveRefusal::Payload)?;
    Ok(ControlOp::Channel {
        transition,
        domain: ChannelId(domain),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::packets::LEDGER;

    /// The claim the module docs make and cannot check by being read: the
    /// class that could rot into a bucket has an exhaustive vocabulary.
    #[test]
    fn every_control_packet_has_exactly_one_kind() {
        let mut seen: Vec<ControlKind> = Vec::new();
        for p in LEDGER {
            let kind = ControlKind::of(p.channel, p.opcode);
            let is_control = classify(p.channel, p.opcode) == Some(PayloadClass::Control);
            assert_eq!(
                kind.is_some(),
                is_control,
                "{} {:#04x} ({}) is classified {:?} and resolves to {:?}",
                p.channel.name(),
                p.opcode,
                p.name,
                classify(p.channel, p.opcode),
                kind
            );
            if let Some(k) = kind {
                seen.push(k);
            }
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 8, "eight control commands, and no ninth");
    }

    /// The retired slots are exactly the acknowledged no-ops, and no live
    /// command is one.
    #[test]
    fn the_retired_slots_are_the_acknowledged_noops() {
        let retired: Vec<_> = LEDGER
            .iter()
            .filter(|p| ControlKind::of(p.channel, p.opcode) == Some(ControlKind::RetiredSlot))
            .collect();
        assert_eq!(retired.len(), 15, "the reference host's retired slots");
        for p in retired {
            assert!(is_acknowledged_noop(p.channel, p.opcode));
        }
        assert_eq!(
            ControlKind::of(Channel::Child, 0x04),
            Some(ControlKind::CursorGlyph),
            "a live command is not a retired slot because it is control"
        );
    }

    /// Inert is about the payload and never about the transaction.
    #[test]
    fn an_inert_payload_is_still_a_transaction() {
        assert!(ControlKind::Nop.payload_is_inert());
        assert!(ControlKind::RetiredSlot.payload_is_inert());
        for live in [
            ControlKind::DisplaySharedStatePage,
            ControlKind::DisplayOnlineAck,
            ControlKind::CursorGlyph,
            ControlKind::CursorShow,
            ControlKind::DefineChannel,
            ControlKind::FreeChannel,
        ] {
            assert!(!live.payload_is_inert(), "{}", live.name());
        }
    }

    #[test]
    fn only_the_channel_commands_are_channel_transitions() {
        assert_eq!(
            ControlKind::DefineChannel.channel_transition(),
            Some(ChannelTransition::Open)
        );
        assert_eq!(
            ControlKind::FreeChannel.channel_transition(),
            Some(ChannelTransition::Free)
        );
        for other in [
            ControlKind::DisplaySharedStatePage,
            ControlKind::DisplayOnlineAck,
            ControlKind::CursorGlyph,
            ControlKind::CursorShow,
            ControlKind::Nop,
            ControlKind::RetiredSlot,
        ] {
            assert_eq!(other.channel_transition(), None, "{}", other.name());
        }
    }

    #[test]
    fn a_packet_that_is_not_control_has_no_kind() {
        assert_eq!(ControlKind::of(Channel::Child, 0x37), None, "the EXEC");
        assert_eq!(ControlKind::of(Channel::Child, 0x25), None, "a delete");
        assert_eq!(ControlKind::of(Channel::Root, 0x3a), None, "a query");
        assert_eq!(ControlKind::of(Channel::Child, 0x06), None, "a present");
        assert_eq!(ControlKind::of(Channel::Child, 0x3d), None, "unresolved");
    }

    #[test]
    fn every_kind_names_itself_once() {
        let ops = [
            ControlOp::Channel {
                transition: ChannelTransition::Open,
                domain: ChannelId(1),
            },
            ControlOp::Channel {
                transition: ChannelTransition::Free,
                domain: ChannelId(1),
            },
            ControlOp::Display {
                kind: ControlKind::CursorShow,
            },
            ControlOp::Inert {
                kind: ControlKind::Nop,
            },
        ];
        let mut names: Vec<&str> = ops.iter().map(|o| o.kind().name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    /// The claim `resolve` exists to make checkable: every control row the
    /// ledger judged becomes an operation from bytes, and the operation is the
    /// command the row names.
    ///
    /// Before it, `ControlKind::of` said which command a packet is and
    /// `ControlOp` said what the model does about it, with nothing carrying a
    /// packet from one to the other.
    #[test]
    fn every_control_packet_resolves_from_bytes_to_its_own_operation() {
        let channel_id = 5u32;
        let payload = channel_id.to_le_bytes();
        let mut channels = 0;
        let mut inert = 0;
        let mut display = 0;
        for p in LEDGER {
            let Some(kind) = ControlKind::of(p.channel, p.opcode) else {
                assert_eq!(
                    resolve(p.channel, p.opcode, &payload),
                    Err(ResolveRefusal::NotControl {
                        channel: p.channel,
                        opcode: p.opcode
                    }),
                    "{} {:#04x} is not control and must not resolve to one",
                    p.channel.name(),
                    p.opcode
                );
                continue;
            };
            let op = resolve(p.channel, p.opcode, &payload).expect("a control packet");
            assert_eq!(
                op.kind(),
                kind,
                "{} {:#04x} resolved to a different command",
                p.channel.name(),
                p.opcode
            );
            match op {
                ControlOp::Channel { domain, .. } => {
                    assert_eq!(domain, ChannelId(channel_id), "the domain is word zero");
                    channels += 1;
                }
                ControlOp::Inert { .. } => inert += 1,
                ControlOp::Display { .. } => display += 1,
            }
        }
        assert_eq!(
            (channels, inert, display),
            (2, 16, 5),
            "two channel-lifetime commands, sixteen inert payloads (the fifteen \
             retired slots and CmdNOP) and five display commands"
        );
    }

    /// A channel definition names its domain, and a free ends the same one.
    /// They share a payload because they are the two ends of one lifetime.
    #[test]
    fn a_channel_command_names_the_domain_it_opens_or_ends() {
        let payload = 9u32.to_le_bytes();
        assert_eq!(
            resolve(Channel::Root, 0x30, &payload),
            Ok(ControlOp::Channel {
                transition: ChannelTransition::Open,
                domain: ChannelId(9),
            })
        );
        assert_eq!(
            resolve(Channel::Root, 0x31, &payload),
            Ok(ControlOp::Channel {
                transition: ChannelTransition::Free,
                domain: ChannelId(9),
            })
        );
    }

    /// A channel command with nothing to read names no domain, and is refused
    /// rather than opening channel zero.
    #[test]
    fn a_channel_command_too_short_to_name_a_domain_is_refused() {
        for opcode in [0x30u16, 0x31] {
            for plen in 0..4usize {
                let refusal = resolve(Channel::Root, opcode, &vec![0u8; plen][..])
                    .expect_err("no domain to name");
                assert_eq!(
                    refusal,
                    ResolveRefusal::Payload(reims_vgpu_protocol::fifo::ShortPayload {
                        plen,
                        need: 4
                    }),
                    "{opcode:#04x} at {plen}"
                );
                assert_eq!(refusal.slug(), "fifo_payload_short");
            }
        }
        // And the commands that read no payload do not care.
        assert!(resolve(Channel::Child, 0x1e, &[]).is_ok());
    }
}
