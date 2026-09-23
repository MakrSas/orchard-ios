//! Ingress: the one place a packet becomes a transaction, or does not become
//! one at all.
//!
//! # Refusal happens here or it happens too late
//!
//! A packet whose contract is not established must be refused *before* it is
//! given an ingress ordinal, an ordering position and a completion obligation.
//! Once it has those, every mechanism downstream will honour them — the hazard
//! compiler will order against its accesses, the scheduler will hold work for
//! its stamp, and something will eventually have to publish a completion for
//! work that was never described. A refusal at ingress costs the guest one
//! command; a refusal after admission costs it the channel.
//!
//! So [`SessionModel::admit`] decides in one place, from the closure ledger,
//! and the refusal it returns names which of the reasons applied.
//!
//! # What a session owns and what it does not
//!
//! It owns the ordinal counters, the per-channel sequences, the hazard graph,
//! the readiness service and each channel's publication order. It owns no
//! resources, no pipelines and no host objects, and it cannot: those live
//! behind a lease whose identity carries this session's generation, and the
//! crate they live in is not this one.

use crate::control::{ChannelTransition, ControlOp};
use crate::depend::DependencyGraph;
use crate::identity::{
    ChannelId, ChannelSequence, CompletionStamp, DeviceEpoch, IngressOrdinal, ResourceId,
    SessionGeneration, SessionId, StampWait, TransactionIdentity,
};
use crate::publish::{Publisher, Release, RetireRefusal};
use crate::ready::Scheduler;
use crate::retire::Lifetime;
use crate::transaction::{classify, DeviceTransaction, Payload, PayloadClass};
use reims_vgpu_protocol::packets::{find, Channel};
use std::collections::{BTreeMap, BTreeSet};

/// What ending a pipeline's life did.
///
/// # Why `took` is a field and not the emptiness of the list
///
/// Both doors that end a pipeline used to answer `Vec::new()` for two
/// different facts: the step was not a legal one, or it was and nothing was
/// parked on the pipeline. A driven boot made the cost of that concrete —
/// 170 retirements against 116 the table actually took, a difference of 54
/// readable only by subtracting an occupancy line from a route counter and
/// knowing to. The 54 are ordinary: a guest deleting a render pipeline this
/// device never drew with names a slot the table has no entry for. Ordinary is
/// exactly what has to be *nameable*, or the day it stops being ordinary looks
/// like nothing at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[must_use = "work stranded by a pipeline ending holds its channel's publication head until it is withdrawn"]
pub struct Ended {
    /// Whether the table took the step.
    pub took: bool,
    /// The transactions that can therefore never be ready. Always empty when
    /// `took` is false: a step the table refused released nothing.
    pub stranded: Vec<IngressOrdinal>,
}

/// Why a packet did not become a transaction.
///
/// Each variant is one check, never shared, so a reader can tell which one
/// refused. The slug is the name it reaches a failure channel under; this crate
/// does not own a failure channel, and a caller that has one renders these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No dispatch table entry: the opcode names no command on this channel.
    UnknownCommand { channel: Channel, opcode: u16 },
    /// A command with no established contract. Admitting it would promise
    /// ordering and completion for work the model cannot describe.
    UnestablishedContract { channel: Channel, opcode: u16 },
    /// The host device incarnation ended and no replacement exists yet.
    /// Admitting would promise ordering and completion on a device that is not
    /// there. Not a guest error and not a semantic one, which is why it is
    /// neither of the other two.
    DeviceLost { epoch: DeviceEpoch },
    /// A replacement device was asked for while the current one is live.
    DeviceNotLost { epoch: DeviceEpoch },
    /// A host completion arrived for an incarnation that is no longer the one
    /// executing. Its transaction was stranded by
    /// [`SessionModel::device_lost`], which is the only thing that takes an
    /// *accepted and submitted* transaction out, so there is no position left
    /// to publish and nothing was lost by saying so.
    ///
    /// Not a caller error. Submission is not completion: work handed to a host
    /// before the loss can still report back after it, and a caller that could
    /// not have known is exactly who this is for.
    CompletionAfterLoss {
        submitted_under: DeviceEpoch,
        current: DeviceEpoch,
    },
    /// The packet named a submission domain no channel definition opened.
    /// Admitting it would give it an ordering position in a publication order
    /// nothing will ever drain, which is a completion word the guest waits on
    /// forever.
    ChannelNotOpen { channel: ChannelId },
    /// A channel definition named a domain that is already open. Silently
    /// reopening would reset a publication order that still has positions in
    /// it.
    ChannelAlreadyOpen { channel: ChannelId },
    /// The packet arrived after the semantic lifetime it names was closed.
    /// Not an error in the guest: a reset races in-flight submissions, and the
    /// contract is that the closed generation stops accepting rather than that
    /// the guest stops sending.
    GenerationClosed {
        named: SessionGeneration,
        current: SessionGeneration,
    },
    /// The packet binds a pipeline it can never use: one this device refused
    /// to build, or one this generation does not have.
    ///
    /// Refused rather than admitted with a wait, because the wait would never
    /// resolve. Admitting it and withdrawing it later costs the guest the same
    /// frame and costs this device a position it has to remember to take back.
    PipelineUnusable(crate::pipeline::LeaseRefusal),
    /// The opcode declares one payload class and the decoded payload is
    /// another. Admitting it would order the packet as the wrong kind of work
    /// against a namespace that does not own what it names.
    PayloadMismatch {
        channel: Channel,
        opcode: u16,
        declared: PayloadClass,
        decoded: PayloadClass,
    },
}

impl Refusal {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownCommand { .. } => "ingress_unknown_command",
            Self::UnestablishedContract { .. } => "ingress_unestablished_contract",
            Self::DeviceLost { .. } => "ingress_device_lost",
            Self::DeviceNotLost { .. } => "ingress_device_not_lost",
            Self::CompletionAfterLoss { .. } => "ingress_completion_after_loss",
            Self::ChannelNotOpen { .. } => "ingress_channel_not_open",
            Self::ChannelAlreadyOpen { .. } => "ingress_channel_already_open",
            Self::PayloadMismatch { .. } => "ingress_payload_mismatch",
            Self::PipelineUnusable(refusal) => refusal.slug(),
            Self::GenerationClosed { .. } => "ingress_generation_closed",
        }
    }
}

/// Why a control operation's transition did not happen.
///
/// Two variants because there are two owners, and neither refusal is invented
/// here: opening is this model's own [`Refusal::ChannelAlreadyOpen`] and
/// freeing is the publisher's [`RetireRefusal::LivePositions`]. Folding them
/// into one reason would lose which half of a channel's lifetime went wrong,
/// and restating either would be a second copy of a check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlRefusal {
    Open(Refusal),
    Free(FreeRefusal),
}

impl ControlRefusal {
    /// The name this reaches a failure channel under: the forwarded owner's,
    /// unchanged.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Open(refusal) => refusal.slug(),
            Self::Free(refusal) => refusal.slug(),
        }
    }
}

/// Why a channel could not be freed.
///
/// Two owners answer this and they answer different things, which is why it is
/// not one of them: the session knows whether a domain was ever opened, and
/// the publisher knows whether an open one still owes publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeRefusal {
    /// The command named a submission domain no channel definition opened.
    ///
    /// The mirror of [`Refusal::ChannelAlreadyOpen`], which the opening door
    /// has always refused, and the same fact [`SessionModel::admit`] refuses a
    /// packet for. Answering `Ok` told a guest that a free succeeded for a
    /// channel that never existed — and told it twice for a double free —
    /// while the FIFO it meant to free stayed open with nothing on any failure
    /// channel to say so.
    NotOpen(Refusal),
    /// The channel still owes publication the guest is waiting on.
    Owed(RetireRefusal),
    /// The command named the root domain, whose lifetime is the device's.
    ///
    /// [`ChannelId::ROOT`] is opened by [`SessionModel::new`] because there is
    /// no packet that could open it — it is the domain a `CmdDefineFifo`
    /// arrives *on*. Freeing it would leave the model with no domain for the
    /// commands that open every other one, and the guest with a device that
    /// stops answering the FIFO it is still writing to.
    IsRoot,
}

impl FreeRefusal {
    /// The forwarded owner's name, unchanged.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NotOpen(refusal) => refusal.slug(),
            Self::Owed(refusal) => refusal.slug(),
            Self::IsRoot => "session_root_channel_not_freeable",
        }
    }
}

/// Whether this session has a host device incarnation to execute on.
///
/// Separate from the epoch identity: the identity says *which* incarnation,
/// this says whether there is one. A session between a loss and its
/// replacement has an epoch that names something dead, and admitting work
/// against it would be promising execution on a device that does not exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceState {
    Live,
    Lost,
}

/// A packet as ingress receives it, before it is a transaction.
///
/// It carries no position. That is the point: the ordinal and the channel
/// sequence are consumed under the arrival this call *is*, so a caller that
/// could state them could state ones that never happened. [`SessionModel::admit`]
/// assigns them and stamps the payload with them.
#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    pub channel: Channel,
    /// The channel this arrived on. [`Packet::channel`] says which dispatch
    /// table; this says which ordering domain, and the root channel is one
    /// domain like any other.
    pub domain: ChannelId,
    /// The semantic lifetime this packet was **read under**.
    ///
    /// Not the one it will be admitted into: a reset races the drain, so a
    /// packet that left the ring before the reset and reaches ingress after it
    /// names a lifetime that has closed. Nothing else can tell — the guest's
    /// packet carries no generation, and by the time this model sees it, its own
    /// generation has already moved — so the reader states the one it was
    /// holding when it took the bytes, which is the one fact the reader has and
    /// this plane does not.
    ///
    /// See [`Refusal::GenerationClosed`], and
    /// [`crate::interpret::Refusal::StaleGeneration`], which is the serial
    /// reference's spelling of the same rule.
    pub session: SessionGeneration,
    pub opcode: u16,
    pub stamp_waits: Vec<StampWait>,
    pub completion: Option<CompletionStamp>,
    /// The decoded work and everything it touches. Resolution is the caller's:
    /// it needs the namespaces, and this is the ordering plane.
    pub payload: Payload,
}

/// What a device loss ended.
///
/// The epoch and the work are one value because they are one event: an epoch
/// that died without its stranded transactions being taken out is a set of
/// channels that never publish again, and a list of stranded transactions
/// without the epoch is a retirement queue that does not know what to abandon.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "the guest is owed a typed reason for every packet the loss stranded"]
pub struct DeviceLoss {
    /// The incarnation that ended.
    pub epoch: DeviceEpoch,
    /// Transactions admitted into it that can never complete, in ingress order.
    /// Already withdrawn from every plane; what is left is to name each one.
    pub stranded: Vec<IngressOrdinal>,
    /// Completion words the withdrawals released — work that had *already*
    /// completed and was waiting behind a stranded position for its channel's
    /// head. Those are not lost: the host delivered them before the device
    /// died and the guest is owed them.
    pub released: Vec<Release>,
}

/// What a guest reset closed.
///
/// The generation and the pipelines are one value because they are one event.
/// A generation that closed without its pipelines being taken out leaves a
/// table that grows with every reset and host objects nothing frees; a list of
/// pipelines without the new generation says nothing about what may now be
/// named.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "the closed generation's pipelines are host objects nothing else will destroy, and its stranded work is a channel head nothing else will release"]
pub struct Reset {
    /// The lifetime the guest may now name things in.
    pub generation: SessionGeneration,
    /// Pipelines of the closed generation that a host object exists for, in id
    /// order. Destroyed rather than abandoned: the handles are live and merely
    /// unnameable — see
    /// [`crate::pipeline::PipelineTable::generation_closed`].
    pub destroy: Vec<ResourceId>,
    /// Transactions that were waiting for a pipeline the closed generation
    /// took away, in ingress order and each named once.
    ///
    /// Accepted work is not invalidated by a reset — but a transaction parked
    /// on a pipeline is parked on a name, and the name is what a reset ends.
    /// The compile that lands afterwards finds no entry and releases nobody,
    /// so left alone these hold their channels' publication heads forever.
    ///
    /// They come back rather than being withdrawn here, which is
    /// [`SessionModel::pipeline_refused`]'s division of labour and not
    /// [`SessionModel::device_lost`]'s: the work is not dead, its *lease* is,
    /// and the caller withdraws each one with a typed reason on its failure
    /// channel — see [`SessionModel::withdraw`].
    pub stranded: Vec<IngressOrdinal>,
}

/// What admitting a packet produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Admitted {
    pub transaction: DeviceTransaction,
    /// Earlier transactions this one must not overtake.
    pub hazard_waits: Vec<IngressOrdinal>,
    /// Whether it may begin immediately.
    pub ready: bool,
}

/// The ordering and readiness plane for one semantic lifetime.
#[derive(Debug)]
pub struct SessionModel {
    id: SessionId,
    generation: SessionGeneration,
    epoch: DeviceEpoch,
    device: DeviceState,
    next_ingress: IngressOrdinal,
    /// The domains a channel definition has opened. Separate from
    /// `channel_sequence` because a channel that is open and has carried no
    /// packet is a real state, and one that has carried packets and been freed
    /// must stop being nameable even while its positions drain.
    open_channels: BTreeSet<ChannelId>,
    channel_sequence: BTreeMap<ChannelId, ChannelSequence>,
    graph: DependencyGraph,
    /// The pipelines whose host build a device loss took, waiting for the
    /// replacement incarnation that can start them again.
    ///
    /// Held here because nothing else can enumerate them: the table knows
    /// which builds it lost and the caller knows how to start one, and between
    /// the loss and [`SessionModel::recreate_device`] there is no device to
    /// start them against.
    rebuilding: Vec<ResourceId>,
    /// The pipeline objects this session's work binds.
    ///
    /// Held here rather than beside this plane, because the waits a transaction
    /// is admitted with are the table's answer about that transaction's own
    /// leases. A table on the other side of the boundary means a caller states
    /// the waits, and a caller that states them can state ones the payload does
    /// not lease or omit ones it does.
    pipelines: crate::pipeline::PipelineTable,
    scheduler: Scheduler,
    publisher: Publisher,
    /// Which channel position each admitted transaction holds, so completion
    /// can find its publication domain without the caller carrying it back.
    position: BTreeMap<IngressOrdinal, (ChannelId, ChannelSequence)>,
    refusals: usize,
}

impl SessionModel {
    #[must_use]
    pub fn new(id: SessionId) -> Self {
        Self {
            id,
            generation: SessionGeneration::FIRST,
            epoch: DeviceEpoch::FIRST,
            device: DeviceState::Live,
            next_ingress: IngressOrdinal::default().next(),
            // The root domain, open from the start. See [`ChannelId::ROOT`]:
            // every other domain is opened by a command the guest sends on this
            // one, so a model that required a command for this one would refuse
            // the command that opens the rest — along with every task
            // definition, object-list bind and device query the root FIFO
            // carries.
            open_channels: BTreeSet::from([ChannelId::ROOT]),
            channel_sequence: BTreeMap::new(),
            graph: DependencyGraph::new(),
            rebuilding: Vec::new(),
            pipelines: crate::pipeline::PipelineTable::new(),
            scheduler: Scheduler::new(),
            publisher: Publisher::new(),
            position: BTreeMap::new(),
            refusals: 0,
        }
    }

    #[must_use]
    pub const fn id(&self) -> SessionId {
        self.id
    }

    #[must_use]
    pub const fn generation(&self) -> SessionGeneration {
        self.generation
    }

    #[must_use]
    pub const fn refusals(&self) -> usize {
        self.refusals
    }

    /// The host device incarnation this session's native objects belong to.
    #[must_use]
    pub const fn epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    /// The pair every lease this session issues carries.
    #[must_use]
    pub const fn lifetime(&self) -> Lifetime {
        Lifetime::new(self.generation, self.epoch)
    }

    /// Close this semantic lifetime and open the next.
    ///
    /// New resolution stops; accepted work is not invalidated. The ordering
    /// plane is deliberately *not* cleared: a transaction accepted in the old
    /// generation still has to complete, still publishes its stamp, and still
    /// releases whatever waited on it. Dropping it here would be a reset that
    /// can lose a completion the host is still going to deliver.
    ///
    /// The device epoch does not move. A guest reset says nothing about the
    /// host device, which may be perfectly healthy — recreating it here would
    /// throw away work the host is still executing in order to answer a
    /// question the guest did not ask.
    /// Work already *running* against a pipeline holds it through the
    /// executor's own lease and is untouched. Work still *waiting* for one is
    /// not: its wait was on a name, and the name is what a reset ends, so
    /// those transactions come back as [`Reset::stranded`] for the caller to
    /// withdraw.
    pub fn reset(&mut self) -> Reset {
        let closed = self.generation;
        self.generation = self.generation.next();
        // Accepted work is untouched — that is the whole of what a reset is
        // not — but the *names* are gone, so nothing may reach these pipelines
        // again and their host objects are the caller's to destroy.
        let taken = self.pipelines.generation_closed(closed);
        // Every removed name, not only the destroyable ones: a `Declared`
        // pipeline has no host object and a draw can be parked on it just the
        // same.
        let mut stranded: Vec<IngressOrdinal> = taken
            .removed
            .iter()
            .flat_map(|id| self.scheduler.pipeline_refused(*id))
            .collect();
        // One transaction may have been waiting on several of them.
        stranded.sort_unstable();
        stranded.dedup();
        Reset {
            generation: self.generation,
            destroy: taken.destroy,
            stranded,
        }
    }

    /// Whether this session has a host device at all.
    #[must_use]
    pub const fn device_state(&self) -> DeviceState {
        self.device
    }

    /// End the host device incarnation.
    ///
    /// Every lease from this epoch becomes unusable at once: no timeline will
    /// advance to release it, because the thing that would advance it is what
    /// was lost. The semantic generation does not move — the guest has not
    /// reset, still names what it named, and is owed a typed terminal fact
    /// rather than a silent new lifetime.
    ///
    /// This does **not** open a replacement. Losing a device and having one
    /// again are two events, and folding them into one call would mean work
    /// submitted in between is admitted into an incarnation that does not
    /// exist. Until [`SessionModel::recreate_device`] runs, admission refuses.
    ///
    /// The pipelines whose builds it took are kept until a replacement device
    /// exists and come back from [`SessionModel::recreate_device`]; see
    /// [`crate::pipeline::PipelineTable::device_lost`] for why a build is an
    /// incarnation's and the object is not.
    ///
    /// Returns the epoch that died — the identity the retirement queue has to
    /// be told about — and the work that died with it.
    ///
    /// **Every transaction admitted into that epoch is stranded.** Nothing will
    /// complete them, because the thing that would is what was lost, and each
    /// one holds a position in the publication order, the dependency graph and
    /// the readiness service. Left there, the channel never publishes again and
    /// every later transaction sharing a backing with one of them waits forever
    /// — so they are withdrawn here rather than named and left for a caller to
    /// remember, which is also the only thing that *can* withdraw them: the
    /// positions are this model's and a caller cannot enumerate them.
    ///
    /// They still come back. The guest is owed a typed terminal fact for each
    /// packet it will never see completed, and this model has no failure
    /// channel to put one on.
    pub fn device_lost(&mut self) -> DeviceLoss {
        self.device = DeviceState::Lost;
        // Ingress order, so a report reads in the order the guest sent them.
        let stranded: Vec<IngressOrdinal> = self.position.keys().copied().collect();
        let mut released = Vec::new();
        for ingress in &stranded {
            released.extend(self.withdraw(*ingress));
        }
        // The host builds died with the incarnation that made them. Held until
        // a replacement exists rather than reported now: there is nothing to
        // build against until then, and a caller handed the list while the
        // device is lost has nowhere to take it.
        self.rebuilding = self.pipelines.device_lost();
        DeviceLoss {
            epoch: self.epoch,
            stranded,
            released,
        }
    }

    /// Open the next host device incarnation after a loss.
    ///
    /// # Errors
    ///
    /// If the device was never lost. A replacement created while the current
    /// incarnation is live would orphan every lease against a device that is
    /// still perfectly able to execute them.
    pub fn recreate_device(&mut self) -> Result<DeviceEpoch, Refusal> {
        if self.device == DeviceState::Live {
            return Err(Refusal::DeviceNotLost { epoch: self.epoch });
        }
        self.epoch = self.epoch.next();
        self.device = DeviceState::Live;
        Ok(self.epoch)
    }

    /// The pipelines a device loss sent back to the start of their build.
    ///
    /// Read rather than taken, because starting a build is not this plane's
    /// and a caller that fails to start one must still be able to see which it
    /// owes. Each entry leaves when the table takes it past
    /// [`crate::pipeline::PipelineState::Declared`] again — so a list that
    /// stops shrinking is builds nobody started, which is a transaction
    /// waiting forever and is exactly what the list exists to make visible.
    #[must_use]
    pub fn rebuilding(&mut self) -> Vec<ResourceId> {
        let table = &self.pipelines;
        self.rebuilding.retain(|id| {
            table
                .get(*id)
                .is_some_and(|p| p.state == crate::pipeline::PipelineState::Declared)
        });
        self.rebuilding.clone()
    }

    /// Turn a packet into a transaction, or refuse it.
    ///
    /// # A refused packet still owes its completion word, and the caller owes it
    ///
    /// Nothing is mutated on a refusal, which is what makes the orders readable
    /// — and it is also why **nothing in this model will ever publish a refused
    /// packet's stamp.** The publisher is not told about it, so it holds no
    /// position for it; [`Self::complete`] and [`Self::withdraw`] both name an
    /// ingress ordinal that was never issued.
    ///
    /// The guest does not know about admission. It wrote a completion word into
    /// the packet header and it waits on that word, so a packet this model
    /// declines is a packet whose caller must stamp
    /// [`Packet::completion`] itself — exactly as it would for a packet the
    /// model accepted and published. A caller that treats a refusal as "nothing
    /// happened" hangs the channel on the first unestablished opcode a real
    /// guest sends, and hangs it *silently*, because a fence that never advances
    /// produces no event.
    ///
    /// This is what makes an honest ledger row affordable. A command the model
    /// has no contract for is [`Refusal::UnestablishedContract`], the work does
    /// not happen, and the guest's fence still moves — so refusing costs the
    /// work and not the channel. It is the difference between a feature this
    /// device does not implement and a device that stops.
    ///
    /// # Errors
    ///
    /// Returns the one check that refused. Nothing is mutated on a refusal —
    /// no ordinal is consumed and no sequence advances — so a refused packet
    /// leaves no gap in either order for a reader to explain.
    pub fn admit(&mut self, packet: &Packet) -> Result<Admitted, Refusal> {
        if self.device == DeviceState::Lost {
            self.refusals += 1;
            return Err(Refusal::DeviceLost { epoch: self.epoch });
        }
        // Before the packet's own contract is judged, because this is not about
        // the packet: it is a lifetime question, and the objects the packet
        // names no longer exist whatever its opcode says.
        if packet.session != self.generation {
            self.refusals += 1;
            return Err(Refusal::GenerationClosed {
                named: packet.session,
                current: self.generation,
            });
        }
        let Some(judged) = find(packet.channel, packet.opcode) else {
            self.refusals += 1;
            return Err(Refusal::UnknownCommand {
                channel: packet.channel,
                opcode: packet.opcode,
            });
        };
        let Some(class) = classify(packet.channel, packet.opcode) else {
            debug_assert!(judged.closure.blocks_cutover());
            self.refusals += 1;
            return Err(Refusal::UnestablishedContract {
                channel: packet.channel,
                opcode: packet.opcode,
            });
        };
        // The opcode says which class the packet is and the payload says which
        // one it *became*. A decoder that resolved a delete into a present
        // would be resolving it against a namespace that does not own it, and
        // the transaction would then be ordered as the wrong kind of work.
        if packet.payload.class() != class {
            self.refusals += 1;
            return Err(Refusal::PayloadMismatch {
                channel: packet.channel,
                opcode: packet.opcode,
                declared: class,
                decoded: packet.payload.class(),
            });
        }

        // Shape before content, and before anything is charged: a domain no
        // channel definition opened is an envelope fact like the generation and
        // the payload class above it. It used to be checked *after* the
        // pipeline leases below, which took — and charged the census for —
        // leases for a packet that was then refused, so the number that says
        // whether compilation starts early enough grew with refused packets.
        if !self.open_channels.contains(&packet.domain) {
            self.refusals += 1;
            return Err(Refusal::ChannelNotOpen {
                channel: packet.domain,
            });
        }

        // The waits are the table's answer about this payload's own leases, not
        // a list the caller brought. A caller that could state them could state
        // ones the payload does not lease — parking the transaction for a
        // compilation it has no interest in, which the guest experiences as a
        // frame that never arrives — or omit ones it does, which runs a draw
        // against a pipeline that is still being built.
        //
        // Non-EXEC payloads lease nothing, so they wait for nothing: only GPU
        // work binds a pipeline.
        let generation = self.generation;
        let pipeline_waits = match packet.payload.exec() {
            Some(work) => self.pipelines.waits_for(&work.pipeline_leases, generation),
            None => Ok(Vec::new()),
        };
        let pipeline_waits = match pipeline_waits {
            Ok(waits) => waits,
            Err(refusal) => {
                self.refusals += 1;
                return Err(Refusal::PipelineUnusable(refusal));
            }
        };

        let ingress = self.next_ingress;
        self.next_ingress = ingress.next();
        let sequence = self
            .channel_sequence
            .entry(packet.domain)
            .or_default()
            .next();
        self.channel_sequence.insert(packet.domain, sequence);

        self.publisher.admit(packet.domain, sequence);
        self.position.insert(ingress, (packet.domain, sequence));

        let hazard_waits = self.graph.admit(ingress, packet.payload.accesses());
        let ready = self.scheduler.admit(
            ingress,
            &hazard_waits,
            &packet.stamp_waits,
            &pipeline_waits,
            packet.completion,
        );
        Ok(Admitted {
            transaction: DeviceTransaction {
                identity: TransactionIdentity {
                    session: self.generation,
                    domain: packet.domain,
                    domain_sequence: sequence,
                    ingress,
                },
                stamp_waits: packet.stamp_waits.clone(),
                completion: packet.completion,
                payload: packet.payload.clone(),
            },
            hazard_waits,
            ready,
        })
    }

    /// Complete a transaction: release its dependents, stop its accesses
    /// creating edges, and hand its channel's publication order whatever it
    /// now owes.
    ///
    /// The first two halves are one call because they are one fact. A
    /// completion that released dependents without retiring accesses would
    /// leave a finished transaction ordering later work forever.
    ///
    /// The third is deliberately *not* the same fact. A stamp becomes readable
    /// when its channel's publication order reaches it, which may be now or
    /// may be after an earlier position finishes, so what comes back is what
    /// the channel actually published — possibly this transaction's stamp,
    /// possibly a queue of them, possibly nothing. Whatever it published is
    /// also published to the readiness service, because a packet waiting on a
    /// completion word waits for the word the guest would read.
    ///
    /// # The incarnation is an argument for the reason it is one on `reached`
    ///
    /// A completion is an asynchronous fact: it was produced by a submission
    /// made under some host device incarnation and arrives some time later,
    /// by which point that incarnation may be dead. That is the same shape as
    /// a timeline point, and [`crate::retire::NativeRetirement::reached`] takes the
    /// epoch for the same reason — the incarnation is what makes the number
    /// mean anything.
    ///
    /// It matters here because [`Self::device_lost`] withdraws every
    /// transaction admitted into the lost epoch, including ones a host was
    /// already executing. Submission is not completion, so those can still
    /// report back. Without the epoch this call had no way to tell that
    /// completion from a caller inventing an ordinal, and answered both by
    /// panicking on a race the contract creates.
    ///
    /// # Errors
    ///
    /// If the completion was produced under an incarnation that has ended.
    ///
    /// # Panics
    ///
    /// If the ordinal holds no channel position under the *current*
    /// incarnation. That is not a race — nothing but a loss takes an accepted
    /// transaction out — so it is a caller completing something it never
    /// admitted, or completing it twice.
    #[must_use = "what the channel published is what the guest may now read"]
    pub fn complete(
        &mut self,
        epoch: DeviceEpoch,
        ingress: IngressOrdinal,
    ) -> Result<Vec<Release>, Refusal> {
        // `Lost` and not only a mismatch: the epoch does not advance until a
        // replacement is opened, so between the loss and `recreate_device` the
        // dead incarnation's number is still the current one.
        if epoch != self.epoch || self.device == DeviceState::Lost {
            return Err(Refusal::CompletionAfterLoss {
                submitted_under: epoch,
                current: self.epoch,
            });
        }
        let owed = self.scheduler.complete(ingress);
        self.graph.retire(ingress);
        let (domain, sequence) = self
            .position
            .remove(&ingress)
            .expect("completing a transaction that holds no channel position");
        let released = self.publisher.complete(domain, sequence, owed);
        for release in &released {
            if let Some(stamp) = release.stamp {
                self.scheduler.publish(stamp);
            }
        }
        Ok(released)
    }

    /// Remove a transaction that will never publish, releasing everything it
    /// was holding.
    ///
    /// A transaction that cannot finish holds a position in **three** planes,
    /// and taking it out of one is not taking it out. Its channel's publication
    /// head is the visible one. The other two are the ones that hang: its
    /// accesses stay live in the dependency graph, so every later transaction
    /// touching that memory takes a hazard wait on an ordinal nothing will ever
    /// complete; and it stays pending in the readiness service, which is the
    /// only thing that decrements a dependent's remaining hazards.
    ///
    /// This used to release the first and neither of the other two, so
    /// withdrawing a transaction to un-stall a channel stalled every later one
    /// that shared a backing with it — the exact failure the withdrawal exists
    /// to prevent, moved from one plane to another.
    ///
    /// Nothing here decides *that* it cannot finish; the caller does, and says
    /// so on its failure channel.
    ///
    /// **Its own completion word is not published.** The work never ran, and a
    /// stamp published for it is a value the guest acts on. What the guest is
    /// owed instead is the typed reason, which is why the caller names one.
    #[must_use = "what the channel published is what the guest may now read"]
    pub fn withdraw(&mut self, ingress: IngressOrdinal) -> Vec<Release> {
        // Releases this transaction's dependents and forgets what it was
        // waiting on. The stamp it owed comes back and is deliberately dropped:
        // publication is `complete`'s and this is not a completion.
        let _never_published = self.scheduler.complete(ingress);
        self.graph.retire(ingress);
        let (domain, sequence) = self
            .position
            .remove(&ingress)
            .expect("withdrawing a transaction that holds no channel position");
        let released = self.publisher.withdraw(domain, sequence);
        for release in &released {
            if let Some(stamp) = release.stamp {
                self.scheduler.publish(stamp);
            }
        }
        released
    }

    #[must_use]
    pub const fn publisher(&self) -> &Publisher {
        &self.publisher
    }

    /// Open a submission domain that no guest command asked for.
    ///
    /// A domain has to be opened before anything may be admitted to it.
    /// Creating it on first use instead would mean a packet naming a channel
    /// the guest never defined gets an ordering position and a completion
    /// obligation in a publication order nothing drains — and the guest waits
    /// on that word forever.
    ///
    /// **This is the bootstrap door, not the guest's.** The root ring exists
    /// before the guest can name anything, so whoever opened it opens its
    /// domain like any other; every domain the guest itself defines arrives as
    /// a `CmdDefineFifo` and goes through [`Self::apply_control`].
    ///
    /// # Errors
    ///
    /// If the domain is already open.
    pub fn open_channel(&mut self, domain: ChannelId) -> Result<(), Refusal> {
        if !self.open_channels.insert(domain) {
            self.refusals += 1;
            return Err(Refusal::ChannelAlreadyOpen { channel: domain });
        }
        Ok(())
    }

    /// Whether a domain is open.
    #[must_use]
    pub fn channel_open(&self, domain: ChannelId) -> bool {
        self.open_channels.contains(&domain)
    }

    /// End a channel's publication lifetime.
    ///
    /// The bootstrap door's other half — see [`Self::open_channel`]. A guest's
    /// `CmdFreeFifo` reaches it through [`Self::apply_control`].
    ///
    /// # Errors
    ///
    /// If no channel definition opened the domain — a driver bug, or a second
    /// free of one already freed. Checked first because it is the precondition
    /// the other answer assumes: a domain that was never opened holds no
    /// positions, so a publisher that found none would have called it drained.
    ///
    /// If the channel still holds unreleased positions. A free that dropped
    /// them would drop the completion words the guest is waiting on, so the
    /// caller drains first.
    pub fn retire_channel(&mut self, domain: ChannelId) -> Result<(), FreeRefusal> {
        if domain == ChannelId::ROOT {
            self.refusals += 1;
            return Err(FreeRefusal::IsRoot);
        }
        if !self.open_channels.remove(&domain) {
            self.refusals += 1;
            return Err(FreeRefusal::NotOpen(Refusal::ChannelNotOpen {
                channel: domain,
            }));
        }
        if let Err(owed) = self.publisher.retire(domain) {
            // Nothing was freed, so the domain is still open.
            self.open_channels.insert(domain);
            self.refusals += 1;
            return Err(FreeRefusal::Owed(owed));
        }
        self.channel_sequence.remove(&domain);
        Ok(())
    }

    /// Perform a control operation's effect on this session.
    ///
    /// **The join between a resolved control packet and the state it changes.**
    /// [`crate::control::resolve`] turned the guest's bytes into a
    /// [`ControlOp`] and this model held the channel set, and nothing carried
    /// one to the other — so a guest's `CmdDefineFifo` decoded into an
    /// operation whose entire effect, opening the domain its next packet names,
    /// nobody performed. Every packet on that channel is then refused
    /// [`Refusal::ChannelNotOpen`], which is a device that answers a correct
    /// guest with a wall.
    ///
    /// The two other operation shapes are `Ok(())` and that is the claim, not
    /// an omission: [`ControlOp::Inert`]'s payload does nothing and
    /// [`ControlOp::Display`]'s belongs to the layer that has a display.
    /// Neither touches ordering, which is what this model owns. Matching
    /// exhaustively is what makes a fourth shape a compile error here rather
    /// than a silently ignored command.
    ///
    /// The envelope is *not* this call's business. A control transaction takes
    /// its ordering position and publishes its completion word like every other
    /// class, whether its transition happened or not — which is why a refusal
    /// here is a value the caller reports and not a reason to withhold a stamp.
    ///
    /// # Errors
    ///
    /// [`ControlRefusal`], forwarding whichever owner refused: a definition
    /// naming a domain that is already open, or a free naming one that still
    /// owes publication.
    pub fn apply_control(&mut self, op: ControlOp) -> Result<(), ControlRefusal> {
        match op {
            ControlOp::Channel {
                transition: ChannelTransition::Open,
                domain,
            } => self.open_channel(domain).map_err(ControlRefusal::Open),
            ControlOp::Channel {
                transition: ChannelTransition::Free,
                domain,
            } => self.retire_channel(domain).map_err(ControlRefusal::Free),
            ControlOp::Display { .. } | ControlOp::Inert { .. } => Ok(()),
        }
    }

    /// The pipeline objects this session's work binds, for the layer that
    /// declares them and advances their compilation.
    ///
    /// Read and write, because declaring a pipeline and stepping it through
    /// translation are the compiling layer's and not this plane's. What this
    /// plane owns is the consequence, which is why the two steps that *have* a
    /// consequence — a pipeline becoming usable and a pipeline becoming
    /// impossible — are [`Self::pipeline_ready`] and [`Self::pipeline_refused`]
    /// rather than calls on the table. A pipeline the *guest* ends is the
    /// third: [`Self::pipeline_retired`].
    pub const fn pipelines(&mut self) -> &mut crate::pipeline::PipelineTable {
        &mut self.pipelines
    }

    /// A pipeline finished building: record it, and release what was held for
    /// it.
    ///
    /// **The two halves are one call because they are one fact.** The table
    /// knew a pipeline had become `Ready` and the scheduler knew which
    /// transactions were parked on it, and nothing carried one to the other —
    /// so a transaction admitted with a pipeline wait was admitted into a wait
    /// nothing could ever discharge. It holds its channel's publication head,
    /// and every completion word behind it stops arriving.
    ///
    /// Returns whether the step was legal and taken. An illegal one is real: a
    /// compile that finishes after the guest deleted the pipeline, which must
    /// not resurrect it — and must not release work either, since that work
    /// cannot be admitted against a retired pipeline in the first place.
    /// The pipeline table, read-only.
    ///
    /// The read half of [`Self::pipelines`], so a caller that only asks — the
    /// walk's usage source — does not need `&mut SessionModel` and cannot step
    /// a pipeline's state by accident.
    #[must_use]
    pub fn pipelines_ref(&self) -> &crate::pipeline::PipelineTable {
        &self.pipelines
    }

    pub fn pipeline_ready(&mut self, pipeline: ResourceId) -> bool {
        if !self
            .pipelines
            .advance(pipeline, crate::pipeline::PipelineState::Ready)
        {
            return false;
        }
        self.scheduler.pipeline_ready(pipeline);
        true
    }

    /// A pipeline will never build: record the reason, and name the
    /// transactions that can therefore never be ready.
    ///
    /// They come back rather than being made ready or dropped. Made ready they
    /// would execute against a pipeline that does not exist; dropped they would
    /// hold their channel's publication head forever. The caller withdraws each
    /// one — see [`Self::withdraw`] — and says why on its failure channel.
    ///
    /// [`Ended::took`] says whether the refusal was a legal step, for
    /// [`Self::pipeline_ready`]'s reason — an illegal one strands nothing and
    /// must not read as a legal one that happened to strand nothing.
    pub fn pipeline_refused(
        &mut self,
        pipeline: ResourceId,
        reason: crate::pipeline::RefusalReason,
    ) -> Ended {
        if !self.pipelines.refuse(pipeline, reason) {
            return Ended::default();
        }
        Ended {
            took: true,
            stranded: self.scheduler.pipeline_refused(pipeline),
        }
    }

    /// The guest deleted a pipeline: retire it, and name the transactions that
    /// were waiting for it.
    ///
    /// A guest deleting a pipeline mid-compile is ordinary —
    /// [`crate::pipeline::PipelineState::may_become`] says so — and nothing
    /// orders that delete behind the work that leased it. The compile then
    /// lands into a table with no usable entry, [`Self::pipeline_ready`]
    /// answers `false`, and a transaction parked on the pipeline is parked on
    /// a wait nothing can discharge: the same hang [`Self::reset`] used to
    /// leave behind, arriving through the other door that ends a pipeline's
    /// name.
    ///
    /// `pipeline_ready`'s "must not release work either, since that work
    /// cannot be admitted against a retired pipeline in the first place" is
    /// about work admitted *after* the retirement. The window opens at
    /// admission, while the pipeline was still pending, and this is that
    /// window's other end.
    ///
    /// They come back rather than being dropped, for
    /// [`Self::pipeline_refused`]'s reason, and the caller withdraws each.
    /// [`Ended::took`] says whether the retirement was a legal step — and a
    /// guest deleting a render pipeline this device never drew with makes that
    /// `false` on a real boot, which is ordinary and is why it is named.
    pub fn pipeline_retired(&mut self, pipeline: ResourceId) -> Ended {
        if !self.pipelines.retire(pipeline) {
            return Ended::default();
        }
        Ended {
            took: true,
            stranded: self.scheduler.pipeline_refused(pipeline),
        }
    }

    /// A completion word became readable: record the value the timeline now
    /// stands at, and release whatever was waiting for it.
    ///
    /// **The guest advances timelines this model does not own.** A device that
    /// published only from its own completions would hold packets against a
    /// value already written — see [`crate::ready::Scheduler::publish`] — so
    /// the publication has to arrive from whoever writes the word, which is
    /// the drain and not this plane.
    ///
    /// Nothing comes back. The released transactions join the ready list and
    /// leave it through [`Self::take_ready`], which is the one place work is
    /// taken; a door that returned them here would be a second one, and a
    /// caller using both would run the same transaction twice.
    ///
    /// The value is not necessarily the one that lands: a slot only ever moves
    /// *later* on its wrapping timeline, so a word written behind the slot is
    /// recorded as no movement at all. Ask [`Self::published_stamp`] for what
    /// the slot actually holds — the two disagreeing is a fence going
    /// backwards, which unsatisfies waits the guest has already been told are
    /// met.
    pub fn stamp_published(&mut self, stamp: CompletionStamp) {
        self.scheduler.publish(stamp);
    }

    /// What a slot's timeline stands at, or `None` if nothing ever published
    /// to it.
    #[must_use]
    pub fn published_stamp(
        &self,
        slot: crate::identity::StampSlot,
    ) -> Option<crate::identity::StampValue> {
        self.scheduler.published_value(slot)
    }

    /// Transactions that have become ready since the last call.
    #[must_use = "a transaction taken off the ready list and not run is one that never runs"]
    pub fn take_ready(&mut self) -> Vec<IngressOrdinal> {
        self.scheduler.take_ready()
    }

    #[must_use]
    pub const fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    #[must_use]
    pub const fn graph(&self) -> &DependencyGraph {
        &self.graph
    }

    /// Reclaim the space retired transactions were holding.
    pub fn compact(&mut self) {
        self.graph.compact();
    }

    /// Whether a payload class reaches an executor, for a caller deciding what
    /// to hand where.
    #[must_use]
    pub const fn executes(payload: PayloadClass) -> bool {
        crate::transaction::reaches_an_executor(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessIntent, AccessKey, AccessMode, BackingId, ResourceKey};
    use crate::identity::{ObjectListRef, SlotGeneration, StampSlot, StampValue};
    use crate::retire::Validity;

    /// Complete under the incarnation that is current, which is what a caller
    /// that has not lost its device does. The loss cases name the epoch
    /// themselves.
    fn done(s: &mut SessionModel, ingress: IngressOrdinal) -> Vec<Release> {
        let epoch = s.epoch();
        s.complete(epoch, ingress)
            .expect("the incarnation is the live one")
    }

    /// A session with the two submission domains the tests use already open,
    /// because opening them is a channel definition's job and not a thing under
    /// test here. `channel_lifetime` tests the opening itself.
    fn session() -> SessionModel {
        let mut s = SessionModel::new(SessionId(1));
        s.open_channel(ChannelId(2)).expect("fresh");
        s.open_channel(ChannelId(3)).expect("fresh");
        s
    }

    /// A packet whose payload is the one its opcode's class calls for, with
    /// nothing in it. What is under test on this plane is ordering, not decode,
    /// so the payload is the emptiest lawful member of the right class — and it
    /// has to be the *right* class, because `admit` refuses a payload that
    /// disagrees with the opcode.
    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn packet(opcode: u16) -> Packet {
        Packet {
            channel: Channel::Child,
            domain: ChannelId(2),
            // The tests here drive one lifetime; `reset` has its own.
            session: SessionGeneration::FIRST,
            opcode,
            stamp_waits: Vec::new(),
            completion: None,
            payload: empty_payload(Channel::Child, opcode),
        }
    }

    /// **A refused packet's completion word is never published by this model,
    /// so its caller owes it.**
    ///
    /// The counterpart to [`Self::admit`]'s "nothing is mutated on a refusal".
    /// Nothing mutated means the publisher was never told, which means no
    /// position exists to publish and no ingress ordinal exists to name in
    /// [`SessionModel::complete`] or [`SessionModel::withdraw`]. The stamp is
    /// unreachable from inside the model — not withheld, absent.
    ///
    /// The guest is not party to admission. It wrote a completion word into the
    /// packet header and it waits on that word, so a caller reading a refusal as
    /// "nothing happened" hangs the channel on the first unestablished opcode a
    /// real guest sends, and hangs it silently — a fence that never advances
    /// produces no event. This test is the claim in a form a cutover cannot
    /// quietly drop: two admitted packets on either side of a refused one, and
    /// what the channel publishes is exactly their two stamps.
    #[test]
    fn a_refused_packet_publishes_no_stamp_and_the_channel_skips_its_value() {
        let mut s = SessionModel::new(SessionId(1));
        s.open_channel(ChannelId(2)).expect("fresh");

        let stamped = |value: u32, opcode: u16| {
            let mut p = packet(opcode);
            p.completion = Some(CompletionStamp {
                slot: StampSlot(2),
                value: StampValue(value),
            });
            p
        };

        let first = s.admit(&stamped(1, 0x37)).expect("an established opcode");

        // The guest's next packet: an opcode whose contract the ledger has not
        // settled. It carries stamp 2, and the guest is waiting on 2.
        let unestablished = stamped(2, 0xffff);
        let refusal = s.admit(&unestablished).expect_err("no such command");
        assert!(
            matches!(
                refusal,
                Refusal::UnknownCommand { .. } | Refusal::UnestablishedContract { .. }
            ),
            "the refusal a ledger row that is not settled produces, got {refusal:?}"
        );
        assert_eq!(
            s.publisher().outstanding(ChannelId(2)),
            1,
            "the refused packet took no position, so there is none to publish"
        );

        let third = s.admit(&stamped(3, 0x37)).expect("an established opcode");

        let mut published = Vec::new();
        for admitted in [first, third] {
            for release in s
                .complete(DeviceEpoch::FIRST, admitted.transaction.identity.ingress)
                .expect("the live incarnation")
            {
                published.extend(release.stamp);
            }
        }
        assert_eq!(
            published
                .iter()
                .map(|stamp| stamp.value.0)
                .collect::<Vec<_>>(),
            vec![1, 3],
            "value 2 is nowhere in what the channel published, and the guest is \
             waiting on it — whoever holds the refusal is the only thing that \
             can move that word"
        );
    }

    /// **A refused admission takes nothing.**
    ///
    /// `admit` promises that nothing is mutated on a refusal, and the pipeline
    /// leases were the exception: they were taken before the channel-open check
    /// below them, so a packet naming a domain no channel definition opened
    /// charged the census for leases the transaction never held. That number is
    /// what says whether starting compilation at declaration is early enough,
    /// and one that grows with refused packets cannot answer it.
    #[test]
    fn a_refused_admission_takes_no_pipeline_lease() {
        let mut s = SessionModel::new(SessionId(1));
        s.open_channel(ChannelId(2)).expect("fresh");
        let pipe = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        s.pipelines().declare(pipe, SessionGeneration::FIRST);
        let before = s.pipelines().census();

        // A domain no definition opened.
        let mut closed = packet(0x37);
        closed.domain = ChannelId(7);
        if let Payload::Exec(work) = &mut closed.payload {
            work.pipeline_leases.push(pipe);
        }
        assert_eq!(
            s.admit(&closed),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(7)
            })
        );
        assert_eq!(s.pipelines().census(), before, "a refusal took a lease");

        // And a lease list that refuses part way charges nothing for the part
        // ahead of the refusal.
        let absent = ResourceId {
            slot: ObjectListRef(10),
            generation: SlotGeneration(1),
        };
        let mut partial = packet(0x37);
        if let Payload::Exec(work) = &mut partial.payload {
            work.pipeline_leases.push(pipe);
            work.pipeline_leases.push(absent);
        }
        assert!(matches!(
            s.admit(&partial),
            Err(Refusal::PipelineUnusable(_))
        ));
        assert_eq!(
            s.pipelines().census(),
            before,
            "the pipelines ahead of the refused one were charged"
        );

        // The lease the admitted packet does hold is counted.
        let mut good = packet(0x37);
        if let Payload::Exec(work) = &mut good.payload {
            work.pipeline_leases.push(pipe);
        }
        s.admit(&good).expect("open domain, declared pipeline");
        assert_eq!(
            s.pipelines().census().leases_pending,
            before.leases_pending + 1
        );
    }

    /// An access naming no memory: the vocabulary for a target that could not
    /// be resolved.
    fn domain_only(domain: ChannelId) -> AccessIntent {
        AccessIntent {
            domain,
            key: crate::access::AccessKey::DomainOnly,
            mode: crate::access::AccessMode::Read,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    fn empty_payload(channel: Channel, opcode: u16) -> Payload {
        match classify(channel, opcode) {
            Some(PayloadClass::Exec) => Payload::Exec(crate::exec::ExecWork::default()),
            Some(PayloadClass::ResourceLifecycle) => Payload::ResourceLifecycle(
                crate::transaction::LifecyclePayload::new(
                    crate::lifecycle::LifecycleOp::DeleteTask {
                        task: crate::identity::TaskId(1),
                    },
                    Vec::new(),
                )
                .expect("a task teardown names no resource"),
            ),
            Some(PayloadClass::Query) => Payload::Query(crate::transaction::QueryPayload::new(
                crate::query::QueryRequest {
                    kind: crate::query::QueryKind::of(channel, opcode).expect("a query"),
                    destination: crate::query::ReplyDestination {
                        backing: BackingId(1),
                        bytes: crate::access::ByteRange {
                            offset: 0,
                            length: 64,
                        },
                    },
                    reply: crate::query::ReplyShape::Fixed { bytes: 16 },
                },
                ChannelId(0),
                None,
            )),
            Some(PayloadClass::Present) => {
                let packet = crate::present::resolve(channel, opcode, &0u32.to_le_bytes())
                    .expect("a present with a trailer");
                // The target is unresolved in this fixture, and a present that
                // named nothing at all would be refused — see `PresentPayload`.
                Payload::Present(
                    crate::transaction::PresentPayload::new(
                        packet,
                        vec![(packet.mapping, domain_only(ChannelId(0)))],
                    )
                    .expect("one read of the packet's own target"),
                )
            }
            // A packet the model refuses never reaches its payload, so the
            // class it would have had is not a thing this can answer. `Nop` is
            // the emptiest payload there is, and the refusal happens first.
            Some(PayloadClass::Control) | None => {
                Payload::Control(crate::control::ControlOp::Inert {
                    kind: crate::control::ControlKind::of(channel, opcode)
                        .unwrap_or(crate::control::ControlKind::Nop),
                })
            }
        }
    }

    /// Give a packet's payload the accesses a test wants it to make.
    ///
    /// There is one list and the payload owns it, so this reaches into the
    /// payload rather than setting a field beside it.
    fn touching(mut packet: Packet, accesses: Vec<AccessIntent>) -> Packet {
        match &mut packet.payload {
            Payload::Exec(work) => work.accesses = accesses,
            // Its accesses must all be reads of the packet's own target, so
            // they are rebuilt through the payload rather than assigned.
            Payload::Present(present) => {
                let packet = *present.packet();
                *present = crate::transaction::PresentPayload::new(
                    packet,
                    accesses.into_iter().map(|a| (packet.mapping, a)).collect(),
                )
                .expect("reads of the packet's own target");
            }
            // The teardown `empty_payload` builds names no resource, so its
            // access list is unconstrained — but it is still the payload's, and
            // it is rebuilt rather than reached into.
            Payload::ResourceLifecycle(lifecycle) => {
                *lifecycle = crate::transaction::LifecyclePayload::new(
                    lifecycle.op().clone(),
                    accesses
                        .into_iter()
                        .map(|a| {
                            (
                                ResourceId {
                                    slot: ObjectListRef(0),
                                    generation: SlotGeneration(1),
                                },
                                a,
                            )
                        })
                        .collect(),
                )
                .expect("the fixture's operation names no resource");
            }
            // A query's access is its reply window and is not the test's to
            // choose — see `QueryPayload`.
            Payload::Query(query) => assert_eq!(
                accesses,
                vec![*query.access()],
                "a query touches its destination and nothing else"
            ),
            Payload::Control(_) => {
                assert!(accesses.is_empty(), "a control packet touches nothing");
            }
        }
        packet
    }

    fn whole(backing: u64, mode: AccessMode) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(2),
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

    /// An EXEC and a resource delete get the same envelope, and differ only in
    /// what they carry.
    #[test]
    fn every_accepted_packet_gets_the_same_envelope() {
        let mut s = session();
        let exec = s.admit(&packet(0x37)).expect("EXEC is accepted");
        let delete = s.admit(&packet(0x25)).expect("delete is accepted");
        assert_eq!(exec.transaction.class(), PayloadClass::Exec);
        assert_eq!(delete.transaction.class(), PayloadClass::ResourceLifecycle);
        assert_eq!(exec.transaction.identity.ingress, IngressOrdinal(1));
        assert_eq!(delete.transaction.identity.ingress, IngressOrdinal(2));
        assert_eq!(
            exec.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
        assert_eq!(
            delete.transaction.identity.domain_sequence,
            ChannelSequence(2)
        );
        assert!(SessionModel::executes(exec.transaction.class()));
        assert!(!SessionModel::executes(delete.transaction.class()));
    }

    /// The opcode declares a class and the payload arrives as one. If they can
    /// differ, they will: a decode that resolved a delete against the display's
    /// namespace would produce a `Present` under opcode `0x25`, and the
    /// transaction would then be ordered as a frame rather than a retirement.
    /// So `admit` compares them, and refuses rather than trusting either.
    #[test]
    fn a_payload_that_is_not_its_opcodes_class_is_refused() {
        let mut s = session();
        let mut wrong = packet(0x25);
        wrong.payload = Payload::Exec(crate::exec::ExecWork::default());
        let err = s.admit(&wrong).expect_err("a delete is not an EXEC");
        assert_eq!(
            err,
            Refusal::PayloadMismatch {
                channel: Channel::Child,
                opcode: 0x25,
                declared: PayloadClass::ResourceLifecycle,
                decoded: PayloadClass::Exec,
            }
        );
        assert_eq!(err.slug(), "ingress_payload_mismatch");
        // And it left no gap: the next packet takes position one.
        let next = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(next.transaction.identity.ingress, IngressOrdinal(1));
        assert_eq!(
            next.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
    }

    /// A transaction's pipeline waits are the table's answer about its own
    /// leases, and nothing else can state them.
    ///
    /// The two lists used to be built in different places — the leases as the
    /// records resolved, the waits from whatever the caller asked the table —
    /// and nothing tied them together, so a wait for a pipeline the packet
    /// never binds was representable, as was a packet that bound one and waited
    /// for nothing. The first parks the transaction for a compilation it has no
    /// interest in; the second runs a draw against a pipeline still being
    /// built.
    #[test]
    fn a_transactions_pipeline_waits_are_the_pipelines_it_binds() {
        let mut s = session();
        let pipeline = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let gen = s.generation();
        s.pipelines().declare(pipeline, gen);

        // Binding nothing waits for nothing, whatever the table holds.
        assert!(s.admit(&packet(0x37)).expect("accepted").ready);

        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.push(pipeline);
        let admitted = s.admit(&leased).expect("accepted");
        assert!(!admitted.ready, "a pipeline still compiling holds the work");
        assert_eq!(s.scheduler().waiting_on_pipelines(), 1);

        // A packet that is not GPU work leases nothing, so it waits for
        // nothing: only an EXEC binds a pipeline.
        assert!(s.admit(&packet(0x1e)).expect("accepted").ready);
        assert_eq!(s.scheduler().waiting_on_pipelines(), 1);
    }

    /// A pipeline finishing releases the work that was held for it.
    ///
    /// The table knew the pipeline had become ready and the scheduler knew who
    /// was parked on it, and nothing carried one to the other — so a
    /// transaction admitted with a pipeline wait was admitted into a wait
    /// nothing could discharge, holding its channel's publication head and
    /// every completion word behind it.
    #[test]
    fn a_pipeline_finishing_releases_the_work_that_was_held_for_it() {
        let mut s = session();
        let pipeline = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let gen = s.generation();
        s.pipelines().declare(pipeline, gen);
        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.push(pipeline);
        let admitted = s.admit(&leased).expect("accepted");
        assert!(!admitted.ready);
        assert!(s.take_ready().is_empty());

        // The intermediate steps are the compiling layer's and release nothing.
        for step in [
            crate::pipeline::PipelineState::Translating,
            crate::pipeline::PipelineState::Compiling,
        ] {
            s.pipelines().advance(pipeline, step);
        }
        assert!(
            s.take_ready().is_empty(),
            "a pipeline is not ready until it is"
        );

        assert!(s.pipeline_ready(pipeline));
        assert_eq!(s.take_ready(), vec![admitted.transaction.identity.ingress]);
        assert_eq!(s.scheduler().waiting_on_pipelines(), 0);

        // A second arrival of the same news is not a legal step and releases
        // nothing, which is what stops a late compile callback resurrecting a
        // pipeline the guest deleted.
        assert!(!s.pipeline_ready(pipeline));
    }

    /// The guest deleting a pipeline is the other door that ends its name, and
    /// it strands the same way.
    ///
    /// A delete mid-compile is ordinary and nothing orders it behind the work
    /// that leased the pipeline, so the draw already parked on it is parked on
    /// a wait the landing compile cannot discharge.
    #[test]
    fn a_deleted_pipeline_names_the_work_it_left_waiting() {
        let mut s = session();
        let pipeline = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let gen = s.generation();
        s.pipelines().declare(pipeline, gen);
        s.pipelines()
            .advance(pipeline, crate::pipeline::PipelineState::Translating);
        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.push(pipeline);
        let admitted = s.admit(&leased).expect("accepted");
        assert!(!admitted.ready);

        assert_eq!(
            s.pipeline_retired(pipeline),
            Ended {
                took: true,
                stranded: vec![admitted.transaction.identity.ingress]
            }
        );
        assert!(
            !s.pipeline_ready(pipeline),
            "the compile landing afterwards resurrects nothing"
        );
        assert!(s.take_ready().is_empty(), "and releases nobody");

        // A second delete is not a legal step and names nobody twice — and
        // says which of the two it is, rather than answering an empty list for
        // both.
        assert_eq!(s.pipeline_retired(pipeline), Ended::default());
    }

    /// A reset ends the pipeline's *name*, and a transaction waiting for one
    /// was waiting on the name.
    ///
    /// The compile lands afterwards into a table with no entry for it, so
    /// nothing releases the waiter: left unnamed it holds its channel's
    /// publication head and every completion word behind it stops arriving.
    /// `device_lost` has always answered this for its own lifetime; a reset is
    /// the other one.
    #[test]
    fn a_reset_names_the_work_it_left_waiting_for_a_pipeline() {
        let mut s = session();
        // One of each: a pipeline the host had started building, which the
        // reset also hands back to be destroyed, and one still `Declared`,
        // which it does not — the waiter is stranded either way, and scoping
        // the answer to the destroyable ones is what missed the second.
        let building = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let declared = ResourceId {
            slot: ObjectListRef(10),
            generation: SlotGeneration(1),
        };
        let gen = s.generation();
        s.pipelines().declare(building, gen);
        s.pipelines().declare(declared, gen);
        s.pipelines()
            .advance(building, crate::pipeline::PipelineState::Translating);

        let mut waiters = Vec::new();
        for pipeline in [building, declared] {
            let mut leased = packet(0x37);
            let Payload::Exec(work) = &mut leased.payload else {
                panic!("an EXEC");
            };
            work.pipeline_leases.push(pipeline);
            let admitted = s.admit(&leased).expect("accepted");
            assert!(!admitted.ready, "parked on {pipeline:?}");
            waiters.push(admitted.transaction.identity.ingress);
        }
        assert_eq!(s.scheduler().waiting_on_pipelines(), 2);

        let reset = s.reset();
        assert_eq!(
            reset.destroy,
            vec![building],
            "only the one a host object exists for is destroyed"
        );
        assert_eq!(
            reset.stranded, waiters,
            "and both waiters are named, in ingress order"
        );

        // The half that makes it a hang: the news the compiling layer is about
        // to deliver reaches nobody.
        assert!(
            !s.pipeline_ready(building),
            "a compile landing after the reset resurrects nothing"
        );
        assert!(s.take_ready().is_empty(), "and releases nobody");

        // So the caller withdrawing them is the only thing that frees the
        // channel, which is what naming them is for.
        for ingress in reset.stranded {
            let _ = s.withdraw(ingress);
        }
        assert_eq!(s.scheduler().waiting_on_pipelines(), 0);
    }

    /// One transaction waiting on two of the closed generation's pipelines is
    /// named once. A caller withdraws each name it is handed.
    #[test]
    fn a_transaction_stranded_by_two_pipelines_is_named_once() {
        let mut s = session();
        let gen = s.generation();
        let mut leases = Vec::new();
        for slot in [11, 12] {
            let id = ResourceId {
                slot: ObjectListRef(slot),
                generation: SlotGeneration(1),
            };
            s.pipelines().declare(id, gen);
            leases.push(id);
        }
        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.extend(leases);
        let admitted = s.admit(&leased).expect("accepted");

        assert_eq!(
            s.reset().stranded,
            vec![admitted.transaction.identity.ingress]
        );
    }

    /// A pipeline that will never build is refused at ingress once it is known,
    /// and the work already admitted for it is named rather than dropped.
    ///
    /// Named because the two outcomes a caller must not take are worse: made
    /// ready, the work executes against a pipeline that does not exist; dropped,
    /// it holds its channel's publication head forever.
    #[test]
    fn a_pipeline_that_will_never_build_names_the_work_it_stranded() {
        let mut s = session();
        let pipeline = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let gen = s.generation();
        s.pipelines().declare(pipeline, gen);
        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.push(pipeline);
        let admitted = s.admit(&leased).expect("accepted");

        let reason = crate::pipeline::RefusalReason::CompilationFailed("out of registers");
        assert_eq!(
            s.pipeline_refused(pipeline, reason),
            Ended {
                took: true,
                stranded: vec![admitted.transaction.identity.ingress]
            }
        );
        // And the next packet binding it is refused at ingress rather than
        // admitted into a wait that cannot resolve.
        let err = s.admit(&leased).expect_err("it can never run");
        assert_eq!(
            err,
            Refusal::PipelineUnusable(crate::pipeline::LeaseRefusal::Refused { pipeline, reason })
        );
        assert_eq!(err.slug(), "pipeline_compilation_failed");
    }

    /// Work binding a pipeline this session never declared is refused, and not
    /// as the same failure as one that could not be built.
    #[test]
    fn work_binding_an_undeclared_pipeline_is_refused_as_absent() {
        let mut s = session();
        let pipeline = ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(1),
        };
        let mut leased = packet(0x37);
        let Payload::Exec(work) = &mut leased.payload else {
            panic!("an EXEC");
        };
        work.pipeline_leases.push(pipeline);
        let err = s.admit(&leased).expect_err("nothing declared it");
        assert_eq!(
            err,
            Refusal::PipelineUnusable(crate::pipeline::LeaseRefusal::Absent {
                pipeline,
                because: crate::pipeline::AbsentBecause::Undeclared,
            })
        );
        assert_eq!(err.slug(), "pipeline_absent_undeclared");
        // The refusal consumed no ordinal, like every other one here.
        let gen = s.generation();
        s.pipelines().declare(pipeline, gen);
        for step in [
            crate::pipeline::PipelineState::Translating,
            crate::pipeline::PipelineState::Compiling,
            crate::pipeline::PipelineState::Ready,
        ] {
            s.pipelines().advance(pipeline, step);
        }
        let ok = s.admit(&leased).expect("declared and built");
        assert_eq!(ok.transaction.identity.ingress, IngressOrdinal(1));
        assert!(ok.ready, "a pipeline already built is nothing to wait for");
    }

    /// **A device loss takes the pipeline builds with it.**
    ///
    /// A `VkPipeline` is an object of one device incarnation. `Ready` is a
    /// statement that the *host* has built it, so after the incarnation that
    /// built it is gone, `Ready` is a live name over a dead handle — which is
    /// one of the two failures [`crate::retire`] separates the two lifetimes to
    /// prevent, and it is worse than the other because the lease is taken at
    /// admission: the transaction is admitted with no wait, given an ordering
    /// position and a completion obligation, and only then does an executor
    /// discover there is nothing to record with.
    ///
    /// The guest's object survives — it has not reset and still names what it
    /// named — so the build starts again rather than the pipeline becoming
    /// absent.
    #[test]
    fn a_replacement_device_rebuilds_the_pipelines_the_lost_one_had_built() {
        use crate::pipeline::{Lease, PipelineState};
        let mut s = session();
        let gen = s.generation();
        let built = res(80);
        let refused = res(81);
        let deleted = res(82);
        for p in [built, refused, deleted] {
            s.pipelines().declare(p, gen);
        }
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            s.pipelines().advance(built, step);
        }
        s.pipelines().refuse(
            refused,
            crate::pipeline::RefusalReason::CompilationFailed("x"),
        );
        s.pipelines().retire(deleted);
        assert_eq!(s.pipelines().lease(built, gen), Lease::Ready);

        let died = s.device_lost();
        let epoch = s.recreate_device().expect("lost, so replaceable");
        assert_ne!(epoch, died.epoch);
        assert_eq!(
            s.generation(),
            gen,
            "a device loss is not a reset; the guest still names its objects"
        );

        assert_eq!(
            s.pipelines().lease(built, gen),
            Lease::Pending,
            "the host build died with the incarnation that made it"
        );
        assert_eq!(
            s.rebuilding(),
            vec![built],
            "and the caller is told which builds to start, because nothing \
             else can enumerate them"
        );
        assert_eq!(
            s.pipelines().lease(refused, gen),
            Lease::Refused(crate::pipeline::RefusalReason::CompilationFailed("x")),
            "a pipeline this device cannot describe is not describable by the \
             next one either; refused stays terminal"
        );
        assert_eq!(
            s.pipelines().lease(deleted, gen),
            Lease::Absent(crate::pipeline::AbsentBecause::Retired),
            "and a deleted object is not resurrected by a new device"
        );

        // The rebuilt one goes through the whole lifetime again, and is then
        // usable exactly as before.
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            s.pipelines().advance(built, step);
        }
        assert_eq!(s.pipelines().lease(built, gen), Lease::Ready);
        assert!(s.rebuilding().is_empty());
    }

    /// The envelope has no access list of its own, so the accesses the
    /// dependency graph ordered against are the ones the payload is holding —
    /// there is no second list for a caller to have filled differently.
    #[test]
    fn the_accesses_a_transaction_is_ordered_by_are_its_payloads() {
        let mut s = session();
        let w = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Write)]))
            .expect("accepted");
        assert_eq!(w.transaction.accesses(), &[whole(1, AccessMode::Write)]);
        assert_eq!(
            w.transaction.payload.exec().expect("an EXEC").accesses,
            w.transaction.accesses(),
            "the envelope's answer is the payload's, not a copy of it"
        );
        // A reader of the same backing is ordered behind it, which is only
        // possible if the graph saw the payload's list.
        let r = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Read)]))
            .expect("accepted");
        assert_eq!(r.hazard_waits, vec![w.transaction.identity.ingress]);
    }

    /// The executor's view of an admitted EXEC is derived from the envelope,
    /// so its identity is the envelope's by construction rather than by
    /// agreement. Before the split these were two stampings of one fact and a
    /// resolver assigned one of them.
    #[test]
    fn the_executors_view_of_an_exec_carries_the_identity_admission_assigned() {
        let mut s = session();
        s.admit(&packet(0x37)).expect("accepted");
        let second = s.admit(&packet(0x37)).expect("accepted").transaction;
        let view = second.exec().expect("an EXEC");
        assert_eq!(view.identity, second.identity);
        assert_eq!(view.ingress(), IngressOrdinal(2));
        assert_eq!(view.domain_sequence(), ChannelSequence(2));
        assert_eq!(view.domain(), ChannelId(2));
        assert!(
            core::ptr::eq(view.work, second.payload.exec().expect("an EXEC")),
            "the view borrows the envelope's work rather than copying it"
        );
        // And nothing else offers one.
        assert!(s
            .admit(&packet(0x25))
            .expect("accepted")
            .transaction
            .exec()
            .is_none());
    }

    /// [`Payload::Control`] has no access list, and that is a contract claim:
    /// opening a channel, moving a cursor and doing nothing touch no guest
    /// resource. A control packet with an access would be a decode error
    /// upstream, and it is not representable here.
    #[test]
    fn control_transactions_touch_no_resource() {
        let mut s = session();
        let mut seen = 0;
        for p in reims_vgpu_protocol::packets::LEDGER {
            if classify(p.channel, p.opcode) != Some(PayloadClass::Control) {
                continue;
            }
            let payload = empty_payload(p.channel, p.opcode);
            assert_eq!(payload.class(), PayloadClass::Control);
            assert!(
                payload.accesses().is_empty(),
                "{} {:#04x} ({}) is control and names a resource",
                p.channel.name(),
                p.opcode,
                p.name
            );
            seen += 1;
        }
        assert_eq!(seen, 23, "the twenty-three control packets");
        // And the graph agrees: a control packet creates no hazard edge against
        // a writer of anything.
        let w = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Write)]))
            .expect("accepted");
        let nop = s.admit(&packet(0x1e)).expect("CmdNOP is accepted");
        assert!(
            nop.hazard_waits.is_empty(),
            "a control packet waits for nothing it does not touch"
        );
        assert!(w.transaction.accesses().len() == 1);
    }

    /// The refusal that keeps the rest of the model honest.
    #[test]
    fn a_command_with_no_established_contract_never_becomes_a_transaction() {
        let mut s = session();
        // CmdDelay: judged, unresolved.
        let err = s.admit(&packet(0x3d)).expect_err("unresolved is refused");
        assert!(matches!(err, Refusal::UnestablishedContract { .. }));
        assert_eq!(err.slug(), "ingress_unestablished_contract");
        // An opcode no dispatch table declares refuses differently, because the
        // two are different problems and only one is closed by writing a
        // handler.
        let err = s.admit(&packet(0x1d)).expect_err("undeclared is refused");
        assert!(matches!(err, Refusal::UnknownCommand { .. }));
        assert_eq!(s.refusals(), 2);
    }

    /// A refusal must leave no gap in either order, or a reader of the ordinals
    /// has to explain a hole that means nothing.
    #[test]
    fn a_refusal_consumes_no_ordinal_and_no_sequence() {
        let mut s = session();
        s.admit(&packet(0x37)).expect("accepted");
        s.admit(&packet(0x3d)).expect_err("refused");
        let next = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(next.transaction.identity.ingress, IngressOrdinal(2));
        assert_eq!(
            next.transaction.identity.domain_sequence,
            ChannelSequence(2)
        );
    }

    /// Channel sequences are per domain; the ingress ordinal is not.
    #[test]
    fn two_domains_keep_separate_sequences_in_one_arrival_order() {
        let mut s = session();
        let mut a = packet(0x37);
        a.domain = ChannelId(2);
        let mut b = packet(0x37);
        b.domain = ChannelId(3);
        let first = s.admit(&a).expect("accepted");
        let second = s.admit(&b).expect("accepted");
        let third = s.admit(&a).expect("accepted");
        assert_eq!(
            [
                first.transaction.identity.ingress,
                second.transaction.identity.ingress,
                third.transaction.identity.ingress
            ],
            [IngressOrdinal(1), IngressOrdinal(2), IngressOrdinal(3)]
        );
        assert_eq!(
            second.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
        assert_eq!(
            third.transaction.identity.domain_sequence,
            ChannelSequence(2)
        );
    }

    #[test]
    fn hazards_and_completion_travel_together() {
        let mut s = session();
        let mut writer = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        writer.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let reader = touching(packet(0x37), vec![whole(1, AccessMode::Read)]);

        let w = s.admit(&writer).expect("accepted");
        assert!(w.ready);
        let r = s.admit(&reader).expect("accepted");
        assert!(!r.ready);
        assert_eq!(r.hazard_waits, vec![w.transaction.identity.ingress]);
        assert_eq!(s.take_ready(), vec![w.transaction.identity.ingress]);

        let released = done(&mut s, w.transaction.identity.ingress);
        assert_eq!(s.take_ready(), vec![r.transaction.identity.ingress]);
        assert_eq!(
            released,
            vec![Release {
                sequence: ChannelSequence(1),
                stamp: writer.completion,
            }],
            "it was its channel's head, so its stamp published at once"
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            Some(StampValue(1)),
            "and a packet waiting on that word may now run"
        );
        // And the completed transaction stops ordering later work.
        let later = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        let l = s.admit(&later).expect("accepted");
        assert_eq!(l.hazard_waits, vec![r.transaction.identity.ingress]);
    }

    /// Withdrawing a transaction stops it ordering later work, not only its
    /// channel.
    ///
    /// A transaction holds a position in three planes and a withdrawal used to
    /// release one of them. The dependency graph kept its accesses live, so
    /// every later transaction touching that backing took a hazard wait on an
    /// ordinal nothing would ever complete; and the readiness service kept it
    /// pending, and that is the only thing that decrements a dependent's
    /// remaining hazards. So un-stalling a channel stalled every later
    /// transaction that shared a backing with the one taken out.
    #[test]
    fn a_withdrawn_transaction_stops_ordering_the_work_behind_it() {
        let mut s = session();
        let doomed = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        let after = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);

        let d = s.admit(&doomed).expect("accepted");
        let a = s.admit(&after).expect("accepted");
        assert!(!a.ready, "it waits on the one before it");
        assert_eq!(a.hazard_waits, vec![d.transaction.identity.ingress]);
        assert_eq!(s.take_ready(), vec![d.transaction.identity.ingress]);

        let _ = s.withdraw(d.transaction.identity.ingress);
        assert_eq!(
            s.take_ready(),
            vec![a.transaction.identity.ingress],
            "the hazard it held is released, not left on an ordinal nothing completes"
        );
        assert_eq!(s.scheduler().pending(), 1, "and it is no longer pending");

        // Its accesses stop ordering anything admitted later, too: a third
        // writer waits on the one still live and not on the one that left.
        let l = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Write)]))
            .expect("accepted");
        assert_eq!(l.hazard_waits, vec![a.transaction.identity.ingress]);
    }

    /// A withdrawn transaction publishes no completion word of its own, and
    /// still releases the ones queued behind it.
    ///
    /// The work never ran, so a stamp published for it is a value the guest
    /// acts on. What the guest is owed is the typed reason, which is the
    /// caller's to name.
    #[test]
    fn a_withdrawal_publishes_what_was_behind_it_and_not_its_own_word() {
        let mut s = session();
        let mut doomed = packet(0x37);
        doomed.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let mut behind = packet(0x37);
        behind.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let d = s.admit(&doomed).expect("accepted");
        let b = s.admit(&behind).expect("accepted");
        assert!(done(&mut s, b.transaction.identity.ingress).is_empty());

        assert_eq!(
            s.withdraw(d.transaction.identity.ingress)
                .into_iter()
                .map(|r| r.stamp)
                .collect::<Vec<_>>(),
            vec![behind.completion],
            "only the word behind it"
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            Some(StampValue(2))
        );
    }

    /// Out-of-order completion is ordinary; out-of-order publication is not.
    #[test]
    fn a_channel_publishes_in_its_own_order_however_the_work_finishes() {
        let mut s = session();
        let mut first = packet(0x37);
        first.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let mut second = packet(0x37);
        second.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let a = s.admit(&first).expect("accepted");
        let b = s.admit(&second).expect("accepted");

        assert!(
            done(&mut s, b.transaction.identity.ingress).is_empty(),
            "the second position finished first and published nothing"
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            None,
            "a guest polling that word must not see the later value first"
        );
        assert_eq!(
            s.publisher().blocked(),
            vec![(ChannelId(2), 1)],
            "and the cost of holding it is counted"
        );
        assert_eq!(
            done(&mut s, a.transaction.identity.ingress)
                .into_iter()
                .map(|r| r.stamp)
                .collect::<Vec<_>>(),
            vec![first.completion, second.completion],
            "the head finishing publishes both, in channel order"
        );
    }

    /// A position that cannot finish must leave, or its channel never publishes
    /// again.
    #[test]
    fn withdrawing_a_head_releases_what_was_queued_behind_it() {
        let mut s = session();
        let mut second = packet(0x37);
        second.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let a = s.admit(&packet(0x37)).expect("accepted");
        let b = s.admit(&second).expect("accepted");
        assert!(done(&mut s, b.transaction.identity.ingress).is_empty());
        assert_eq!(
            s.withdraw(a.transaction.identity.ingress)
                .into_iter()
                .map(|r| r.stamp)
                .collect::<Vec<_>>(),
            vec![second.completion]
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            Some(StampValue(2))
        );
    }

    /// Freeing a domain no definition opened is a refusal, not a success.
    ///
    /// The mirror of `open_channel`'s `ChannelAlreadyOpen`, and the same fact
    /// `admit` already refuses a packet for. Answered `Ok`, a driver bug and a
    /// double free both looked like clean runs while the FIFO the guest meant
    /// to free stayed open.
    #[test]
    fn freeing_a_domain_nothing_opened_is_refused_by_name() {
        let mut s = session();
        let before = s.refusals();
        assert_eq!(
            s.retire_channel(ChannelId(7)),
            Err(FreeRefusal::NotOpen(Refusal::ChannelNotOpen {
                channel: ChannelId(7)
            }))
        );
        assert_eq!(s.refusals(), before + 1, "and it is counted as one");

        // The double free of a domain that did exist reaches the same answer.
        assert_eq!(s.retire_channel(ChannelId(2)), Ok(()));
        assert_eq!(
            s.retire_channel(ChannelId(2)),
            Err(FreeRefusal::NotOpen(Refusal::ChannelNotOpen {
                channel: ChannelId(2)
            }))
        );
    }

    #[test]
    fn a_channel_with_unpublished_work_cannot_end_its_lifetime() {
        let mut s = session();
        let a = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(
            s.retire_channel(ChannelId(2)),
            Err(FreeRefusal::Owed(RetireRefusal::LivePositions {
                outstanding: 1
            }))
        );
        let _ = done(&mut s, a.transaction.identity.ingress);
        assert_eq!(s.retire_channel(ChannelId(2)), Ok(()));
        assert!(
            !s.channel_open(ChannelId(2)),
            "a freed channel stops being nameable"
        );
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(2)
            }),
            "and a packet naming it is refused rather than reopening it"
        );
        // A later definition of the channel starts at position one rather than
        // continuing the lifetime that just ended.
        s.open_channel(ChannelId(2)).expect("free again");
        let next = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(
            next.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
    }

    /// A guest's own channel commands reach the channel set, end to end.
    ///
    /// Bytes, resolve, apply, admit — because the defect this closes was
    /// exactly a missing link in that chain: the operation resolved, the model
    /// held the channels, and nothing joined them, so a correct guest that
    /// defined a FIFO and used it got `ChannelNotOpen` on every packet.
    ///
    /// The bootstrap door is deliberately not used here. The root domain is
    /// open because a session has one — the ring exists before the guest can
    /// name anything, and the command that opens every other domain arrives on
    /// it — and everything after that is the guest's own bytes.
    #[test]
    fn a_guests_channel_commands_open_and_end_the_domain_it_then_submits_on() {
        const DEFINE: u16 = 0x30;
        const FREE: u16 = 0x31;
        let mut s = SessionModel::new(SessionId(1));
        assert!(
            s.channel_open(ChannelId::ROOT),
            "a session with no root domain has nowhere for a channel definition to arrive"
        );
        assert_eq!(
            s.open_channel(ChannelId::ROOT),
            Err(Refusal::ChannelAlreadyOpen {
                channel: ChannelId::ROOT
            }),
            "and it is open once, not opened again by a bootstrap that ran twice"
        );
        assert_eq!(
            s.retire_channel(ChannelId::ROOT),
            Err(FreeRefusal::IsRoot),
            "the root FIFO's publication lifetime is the device's, not a \
             guest command's"
        );

        let domain = ChannelId(2).0.to_le_bytes();
        let define = crate::control::resolve(Channel::Root, DEFINE, &domain).expect("a definition");
        let free = crate::control::resolve(Channel::Root, FREE, &domain).expect("a free");

        // Before the definition is applied, the domain it names is not one.
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(2)
            })
        );
        assert_eq!(s.apply_control(define), Ok(()));
        assert!(s.channel_open(ChannelId(2)));
        let admitted = s.admit(&packet(0x37)).expect("the domain is open now");

        // The free is refused while that packet still owes publication, and the
        // domain stays open — a refused transition changes nothing.
        assert_eq!(
            s.apply_control(free),
            Err(ControlRefusal::Free(FreeRefusal::Owed(
                RetireRefusal::LivePositions { outstanding: 1 }
            )))
        );
        assert!(s.channel_open(ChannelId(2)));

        let _ = done(&mut s, admitted.transaction.identity.ingress);
        assert_eq!(s.apply_control(free), Ok(()));
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(2)
            }),
            "the lifetime the guest ended is over"
        );
    }

    /// Redefining a live domain is refused rather than resetting its
    /// publication order, and the refusal is the opening owner's own.
    #[test]
    fn a_second_definition_of_a_live_domain_is_refused() {
        let mut s = session();
        let domain = ChannelId(2).0.to_le_bytes();
        let define = crate::control::resolve(Channel::Root, 0x30, &domain).expect("a definition");
        assert_eq!(
            s.apply_control(define),
            Err(ControlRefusal::Open(Refusal::ChannelAlreadyOpen {
                channel: ChannelId(2)
            }))
        );
        assert_eq!(
            s.apply_control(define).unwrap_err().slug(),
            "ingress_channel_already_open"
        );
    }

    /// Every control operation that is not a channel transition leaves this
    /// model alone — which is the claim, not an omission.
    ///
    /// A display command's content belongs to the layer that has a display and
    /// an inert payload does nothing; neither touches ordering. The census is
    /// over the whole ledger so a control row that grows an ordering effect
    /// cannot be added without this failing.
    #[test]
    fn only_the_two_channel_commands_change_what_this_model_holds() {
        use reims_vgpu_protocol::packets::LEDGER;
        let mut applied = 0usize;
        for p in LEDGER {
            let Some(kind) = crate::control::ControlKind::of(p.channel, p.opcode) else {
                continue;
            };
            if kind.channel_transition().is_some() {
                continue;
            }
            let op =
                crate::control::resolve(p.channel, p.opcode, &[0u8; 8]).expect("a control packet");
            let mut s = session();
            let before = (s.channel_open(ChannelId(2)), s.channel_open(ChannelId(3)));
            assert_eq!(s.apply_control(op), Ok(()), "{}", kind.name());
            assert_eq!(
                (s.channel_open(ChannelId(2)), s.channel_open(ChannelId(3))),
                before,
                "{} moved a channel",
                kind.name()
            );
            applied += 1;
        }
        assert_eq!(
            applied, 21,
            "the ledger's control rows less the two channel commands"
        );
    }

    /// A packet naming a domain no definition opened is refused at ingress.
    /// Creating the domain on first use instead would give the packet an
    /// ordering position and a completion obligation in a publication order
    /// nothing drains, and the guest waits on that word forever.
    #[test]
    fn a_packet_on_an_undefined_channel_is_refused_and_consumes_nothing() {
        let mut s = SessionModel::new(SessionId(1));
        let before = s.refusals();
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::ChannelNotOpen {
                channel: ChannelId(2)
            })
        );
        assert_eq!(s.refusals(), before + 1);
        s.open_channel(ChannelId(2)).expect("fresh");
        let first = s.admit(&packet(0x37)).expect("accepted");
        assert_eq!(
            first.transaction.identity.ingress,
            IngressOrdinal::default().next(),
            "the refused packet consumed no ordinal"
        );
        assert_eq!(
            first.transaction.identity.domain_sequence,
            ChannelSequence(1)
        );
    }

    /// Reopening a live channel would reset a publication order that still has
    /// positions in it.
    #[test]
    fn a_channel_is_defined_once() {
        let mut s = session();
        assert_eq!(
            s.open_channel(ChannelId(2)),
            Err(Refusal::ChannelAlreadyOpen {
                channel: ChannelId(2)
            })
        );
    }

    /// The two transitions are independent, which is the whole reason they are
    /// two identities.
    #[test]
    fn a_guest_reset_and_a_device_loss_move_different_lifetimes() {
        let mut s = session();
        let start = s.lifetime();

        let _ = s.reset();
        assert_eq!(
            s.epoch(),
            start.epoch,
            "a guest reset says nothing about the host device, which may be \
             perfectly healthy"
        );
        assert_ne!(s.generation(), start.session);
        assert_eq!(s.device_state(), DeviceState::Live);

        let after_reset = s.lifetime();
        let died = s.device_lost();
        assert_eq!(
            died.epoch, after_reset.epoch,
            "the epoch that died is named"
        );
        assert!(died.stranded.is_empty(), "nothing was admitted");
        assert_eq!(
            s.generation(),
            after_reset.session,
            "the guest has not reset; it still names what it named"
        );
        assert_eq!(s.device_state(), DeviceState::Lost);

        let replacement = s.recreate_device().expect("lost, so replaceable");
        assert_ne!(replacement, died.epoch);
        assert_eq!(s.generation(), after_reset.session);
    }

    /// Submission is not completion, so a host handed work before a loss can
    /// report it back after one. `device_lost` withdrew that transaction — it
    /// is the only thing that takes an *accepted and submitted* transaction out
    /// — so there is no position left, and the caller could not have known:
    /// nothing public tells it whether an ordinal is still outstanding.
    ///
    /// The incarnation is what answers it, which is why `complete` takes one,
    /// for the reason [`crate::retire::NativeRetirement::reached`] takes one. Both
    /// windows are covered: the epoch does not advance until a replacement is
    /// opened, so between the loss and `recreate_device` the dead incarnation's
    /// number is still the current one and only `DeviceState` separates them.
    #[test]
    fn a_completion_from_the_lost_incarnation_is_refused_and_not_a_panic() {
        let mut s = session();
        let admitted = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Write)]))
            .expect("accepted");
        let ingress = admitted.transaction.identity.ingress;
        let submitted_under = s.epoch();
        // The executor takes it and submits it. Then the device dies.
        let loss = s.device_lost();
        assert_eq!(loss.stranded, vec![ingress]);

        // The host reports the submission it had already accepted. The epoch
        // still reads as the dead one, so the state is what separates them.
        assert_eq!(
            s.complete(submitted_under, ingress),
            Err(Refusal::CompletionAfterLoss {
                submitted_under,
                current: submitted_under,
            })
        );

        // And after a replacement, where the number separates them too.
        let replacement = s.recreate_device().expect("lost, so replaceable");
        assert_eq!(
            s.complete(submitted_under, ingress),
            Err(Refusal::CompletionAfterLoss {
                submitted_under,
                current: replacement,
            })
        );

        // The replacement device completes its own work normally.
        let after = s
            .admit(&touching(packet(0x37), vec![whole(1, AccessMode::Write)]))
            .expect("accepted");
        assert_eq!(
            s.complete(replacement, after.transaction.identity.ingress),
            Ok(vec![Release {
                sequence: ChannelSequence(2),
                stamp: None,
            }]),
            "the position published; it owed no stamp"
        );
    }

    /// A device loss takes every transaction admitted into it out of all three
    /// planes, and names them.
    ///
    /// Nothing will complete them: the thing that would is what was lost. Left
    /// in place, each one holds its channel's publication head forever and its
    /// accesses keep ordering work admitted after the replacement device
    /// arrives. They are withdrawn here rather than named and left for a caller
    /// to remember, which is also the only place they *can* be withdrawn — the
    /// positions are this model's and a caller cannot enumerate them.
    #[test]
    fn a_device_loss_strands_the_work_admitted_into_it_and_names_it() {
        let mut s = session();
        let mut first = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        first.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let mut second = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        second.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let a = s.admit(&first).expect("accepted");
        let b = s.admit(&second).expect("accepted");
        assert_eq!(s.scheduler().pending(), 2);

        let loss = s.device_lost();
        assert_eq!(loss.epoch, s.epoch());
        assert_eq!(
            loss.stranded,
            vec![
                a.transaction.identity.ingress,
                b.transaction.identity.ingress
            ],
            "in ingress order, so a report reads in the order the guest sent them"
        );
        assert!(
            loss.released.is_empty(),
            "neither had completed, so no word was owed behind them"
        );
        assert_eq!(s.scheduler().pending(), 0);
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            None,
            "work that never ran publishes no word the guest could act on"
        );

        // And the replacement device starts clean: a writer of the same backing
        // waits on nothing.
        s.recreate_device().expect("lost");
        let next = s.admit(&first).expect("accepted");
        assert!(
            next.hazard_waits.is_empty(),
            "the dead epoch's accesses no longer order anything"
        );
        assert!(next.ready);
    }

    /// A completion the host delivered before the device died is still owed to
    /// the guest, and a loss releases it.
    ///
    /// It was queued behind a position that is now stranded. Dropping it with
    /// the stranded work would lose a completion that really happened.
    #[test]
    fn a_loss_releases_a_word_that_was_waiting_behind_the_work_it_stranded() {
        let mut s = session();
        let mut head = packet(0x37);
        head.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(1),
        });
        let mut behind = packet(0x37);
        behind.completion = Some(CompletionStamp {
            slot: StampSlot(0),
            value: StampValue(2),
        });
        let h = s.admit(&head).expect("accepted");
        let b = s.admit(&behind).expect("accepted");
        // The second finished; the first is still running when the device dies.
        assert!(done(&mut s, b.transaction.identity.ingress).is_empty());

        let loss = s.device_lost();
        assert_eq!(loss.stranded, vec![h.transaction.identity.ingress]);
        assert_eq!(
            loss.released
                .into_iter()
                .map(|r| r.stamp)
                .collect::<Vec<_>>(),
            vec![behind.completion]
        );
        assert_eq!(
            s.scheduler().published_value(StampSlot(0)),
            Some(StampValue(2))
        );
    }

    /// Work submitted between a loss and its replacement has nothing to run on
    /// and must be told so, not admitted into an incarnation that is gone.
    #[test]
    fn a_lost_device_refuses_admission_until_it_is_replaced() {
        let mut s = session();
        let epoch = s.device_lost().epoch;
        assert_eq!(
            s.admit(&packet(0x37)),
            Err(Refusal::DeviceLost { epoch }),
            "ordering and completion cannot be promised on a device that is \
             not there"
        );
        assert_eq!(s.refusals(), 1);
        s.recreate_device().expect("lost");
        assert!(s.admit(&packet(0x37)).is_ok());
    }

    #[test]
    fn a_live_device_cannot_be_replaced() {
        let mut s = session();
        assert_eq!(
            s.recreate_device(),
            Err(Refusal::DeviceNotLost { epoch: s.epoch() }),
            "a replacement made while the device is live would orphan every \
             lease against a device still able to execute them"
        );
    }

    /// A lease issued before either transition is judged on both, separately.
    #[test]
    fn a_lease_from_before_both_transitions_is_judged_on_both() {
        let mut s = session();
        let lease = s.lifetime();
        assert_eq!(lease.against(s.generation(), s.epoch()), Validity::Live);

        let _ = s.reset();
        let after = lease.against(s.generation(), s.epoch());
        assert!(!after.admits_new_work(), "the guest may not name it");
        assert!(
            after.handles_usable(),
            "and the submission the host is still executing must finish"
        );

        assert!(
            s.device_lost().stranded.is_empty(),
            "this test admitted nothing"
        );
        s.recreate_device().expect("lost");
        assert_eq!(lease.against(s.generation(), s.epoch()), Validity::Gone);
    }

    /// Two attached devices are two values with nothing between them. The test
    /// exists so that adding a process-global anywhere in this plane fails
    /// here rather than in a guest.
    #[test]
    fn one_sessions_reset_or_loss_cannot_reach_another_session() {
        let mut a = session();
        let mut b = SessionModel::new(SessionId(2));
        b.open_channel(ChannelId(2)).expect("fresh");
        let untouched = b.lifetime();
        let admitted = b.admit(&packet(0x37)).expect("accepted");

        let _ = a.reset();
        assert!(
            a.device_lost().stranded.is_empty(),
            "the other session's work is not this one's to strand"
        );
        a.recreate_device().expect("lost");

        assert_eq!(b.lifetime(), untouched);
        assert_eq!(b.device_state(), DeviceState::Live);
        assert_eq!(untouched.against(b.generation(), b.epoch()), Validity::Live);
        assert!(b.admit(&packet(0x37)).is_ok());
        assert!(!done(&mut b, admitted.transaction.identity.ingress).is_empty());
    }

    /// A reset opens a new lifetime and does not throw away work that has not
    /// completed. Dropping it here would be a reset that loses a completion the
    /// host is still going to deliver.
    #[test]
    fn a_reset_opens_a_generation_without_abandoning_accepted_work() {
        let mut s = session();
        let writer = touching(packet(0x37), vec![whole(1, AccessMode::Write)]);
        let w = s.admit(&writer).expect("accepted");
        let before = s.generation();
        let after = s.reset().generation;
        assert!(after > before);
        assert_eq!(s.scheduler().pending(), 1);
        // Work accepted after the reset carries the new generation and still
        // orders against the old transaction, which has not completed.
        let mut reader = touching(packet(0x37), vec![whole(1, AccessMode::Read)]);
        reader.session = after;
        let r = s.admit(&reader).expect("accepted");
        assert_eq!(r.transaction.identity.session, after);
        assert_eq!(r.hazard_waits, vec![w.transaction.identity.ingress]);
    }

    /// **A closed generation's pipelines leave with it.**
    ///
    /// The generation check in `lease` already makes them unusable, so nothing
    /// misbehaved — but the entries stayed, so a guest resetting in a loop
    /// grew the table without bound, and the host objects behind them were
    /// handed to nobody. `PipelineState::Retired` names "its generation
    /// closed" as one of its two ways in and only the other had a path.
    ///
    /// Destroyed, not abandoned: a reset says nothing about the host device,
    /// so the handles are live and merely unnameable — which is exactly
    /// [`crate::retire::Validity::SemanticallyClosed`].
    #[test]
    fn a_reset_hands_back_the_pipelines_its_generation_could_name() {
        use crate::pipeline::{Lease, PipelineState};
        let mut s = session();
        let gen = s.generation();
        let (declared, translating, ready, refused) = (res(90), res(91), res(92), res(93));
        for p in [declared, translating, ready, refused] {
            s.pipelines().declare(p, gen);
        }
        s.pipelines()
            .advance(translating, PipelineState::Translating);
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            s.pipelines().advance(ready, step);
        }
        s.pipelines()
            .refuse(refused, crate::pipeline::RefusalReason::Undescribable("x"));
        assert_eq!(s.pipelines().len(), 4);

        let reset = s.reset();
        assert!(reset.generation > gen);
        assert_eq!(
            reset.destroy,
            vec![translating, ready],
            "only the ones the host has something for; a declaration never \
             reached it and a refusal is a build that did not happen"
        );
        assert_eq!(
            s.pipelines().len(),
            0,
            "and none of the four can ever be named again, so none of them stays"
        );
        for p in [declared, translating, ready, refused] {
            assert_eq!(
                s.pipelines().lease(p, gen),
                Lease::Absent(crate::pipeline::AbsentBecause::Undeclared)
            );
            assert_eq!(
                s.pipelines().lease(p, reset.generation),
                Lease::Absent(crate::pipeline::AbsentBecause::Undeclared)
            );
        }

        // The next generation declares its own, and a second reset takes only
        // those.
        let next = reset.generation;
        s.pipelines().declare(declared, next);
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            s.pipelines().advance(declared, step);
        }
        assert_eq!(s.reset().destroy, vec![declared]);
    }

    /// A reset closes names; a device loss takes handles. The two doors hand
    /// back different things for that reason.
    #[test]
    fn a_reset_destroys_pipelines_and_a_device_loss_rebuilds_them() {
        use crate::pipeline::PipelineState;
        let build = |s: &mut SessionModel, id, gen| {
            s.pipelines().declare(id, gen);
            for step in [
                PipelineState::Translating,
                PipelineState::Compiling,
                PipelineState::Ready,
            ] {
                s.pipelines().advance(id, step);
            }
        };

        let mut s = session();
        let p = res(94);
        let gen = s.generation();
        build(&mut s, p, gen);
        let _ = s.device_lost();
        s.recreate_device().expect("lost");
        assert_eq!(
            s.rebuilding(),
            vec![p],
            "the object survives a loss and its build starts again"
        );
        assert_eq!(s.pipelines().len(), 1);

        let mut s = session();
        let gen = s.generation();
        build(&mut s, p, gen);
        assert_eq!(
            s.reset().destroy,
            vec![p],
            "the object does not survive a reset, and its handle is live"
        );
        assert_eq!(s.pipelines().len(), 0);
        assert!(s.rebuilding().is_empty());
    }

    /// A packet read under a lifetime that has since closed is refused, and the
    /// refusal names both generations.
    ///
    /// A reset races the drain: a packet that left the ring before it and
    /// reaches ingress after names objects that no longer exist. Nothing else
    /// can tell — the guest's bytes carry no generation, and by the time this
    /// plane sees the packet its own generation has already moved — which is
    /// why the reader states the one it was holding.
    ///
    /// Not the same event as the reset itself: work already *admitted* is
    /// untouched, which is the test above.
    #[test]
    fn a_packet_read_before_a_reset_is_refused_after_it() {
        let mut s = session();
        let stale = packet(0x37);
        let closed = s.generation();
        let current = s.reset().generation;
        let err = s.admit(&stale).expect_err("its lifetime is over");
        assert_eq!(
            err,
            Refusal::GenerationClosed {
                named: closed,
                current,
            }
        );
        assert_eq!(err.slug(), "ingress_generation_closed");
        // And it consumed nothing, like every other refusal here.
        let mut fresh = packet(0x37);
        fresh.session = current;
        assert_eq!(
            s.admit(&fresh)
                .expect("accepted")
                .transaction
                .identity
                .ingress,
            IngressOrdinal(1)
        );
    }

    #[test]
    fn a_session_carries_the_identity_it_was_created_with() {
        let s = SessionModel::new(SessionId(4));
        assert_eq!(s.id(), SessionId(4));
        assert_eq!(s.generation(), SessionGeneration::FIRST);
    }

    /// Two of the retired slots and an acknowledged no-op are all `Control`,
    /// and all of them are still transactions: they retire stamps, and
    /// something has to order that.
    #[test]
    fn the_acknowledged_noops_are_transactions_like_any_other() {
        let mut s = session();
        for opcode in [0x1e, 0x03, 0x32] {
            let t = s.admit(&packet(opcode)).expect("accepted");
            assert_eq!(t.transaction.class(), PayloadClass::Control);
            assert!(t.ready);
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

        fn below(&mut self, bound: u64) -> u64 {
            if bound == 0 {
                return 0;
            }
            self.next() % bound
        }
    }

    /// Opcodes chosen so every admission outcome is reachable: three
    /// established classes, one the ledger leaves unresolved, and one no
    /// dispatch table names.
    const OPCODES: [u16; 5] = [0x37, 0x25, 0x1e, 0x09, 0xfe];
    const DOMAINS: [ChannelId; 3] = [ChannelId(2), ChannelId(3), ChannelId(4)];

    /// **Ingress is the one place a packet becomes a transaction, and a refused
    /// packet leaves no trace of having tried.**
    ///
    /// The shadow holds what ingress is *about* and nothing else: which
    /// domains are open, the next ordinal, each domain's next sequence, and the
    /// generation and device state. It has no publisher, no graph and no
    /// readiness service, so it can state the refusal precedence and the "no
    /// ordinal is consumed" rule without being able to express any of the
    /// machinery those rules protect.
    ///
    /// After every step the three planes are checked to agree with each other
    /// and with the shadow: a transaction holds a position in the publication
    /// order, the dependency graph and the readiness service, and completing,
    /// withdrawing or losing the device must take it out of all three.
    #[test]
    fn a_refused_packet_takes_no_position_and_an_admitted_one_takes_three() {
        let mut admitted = 0usize;
        let mut refused_unknown = 0usize;
        let mut refused_unestablished = 0usize;
        let mut refused_closed_channel = 0usize;
        let mut refused_generation = 0usize;
        let mut refused_device_lost = 0usize;
        let mut completions = 0usize;
        let mut withdrawals = 0usize;
        let mut resets = 0usize;
        let mut losses = 0usize;
        let mut stranded_total = 0usize;
        let mut channel_retires_refused = 0usize;
        let mut channel_frees_unopened = 0usize;

        for seed in 0..384u64 {
            let mut rng = Rng::new(seed);
            let mut s = SessionModel::new(SessionId(1));
            // Shadow.
            let mut open: BTreeSet<ChannelId> = BTreeSet::new();
            let mut sequence: BTreeMap<ChannelId, u64> = BTreeMap::new();
            // Positions each domain has taken, and positions it has released.
            // The publication order holds the difference, which is *not* the
            // outstanding transactions: a transaction that completed behind an
            // unfinished head has left the readiness service and still holds
            // its position, which is the whole point of `publish`.
            let mut taken: BTreeMap<ChannelId, usize> = BTreeMap::new();
            let mut released: BTreeMap<ChannelId, usize> = BTreeMap::new();
            let mut next_ingress = 1u64;
            let mut generation = SessionGeneration::FIRST;
            let mut live_device = true;
            // Ingress ordinals admitted and not yet completed or withdrawn.
            let mut outstanding: Vec<(IngressOrdinal, ChannelId)> = Vec::new();

            for _ in 0..40 {
                match rng.below(16) {
                    // Open a domain.
                    0..=1 => {
                        let d = DOMAINS[rng.below(3) as usize];
                        let already = open.contains(&d);
                        match s.open_channel(d) {
                            Ok(()) => {
                                assert!(!already, "seed {seed}: reopened a live domain");
                                open.insert(d);
                            }
                            Err(Refusal::ChannelAlreadyOpen { channel }) => {
                                assert!(already);
                                assert_eq!(channel, d);
                            }
                            Err(other) => panic!("seed {seed}: open refused as {other:?}"),
                        }
                    }
                    // End a domain's publication lifetime.
                    2 => {
                        let d = DOMAINS[rng.below(3) as usize];
                        let live = s.publisher().outstanding(d);
                        let was_open = open.contains(&d);
                        match s.retire_channel(d) {
                            Ok(()) => {
                                assert!(was_open, "seed {seed}: freed a domain nothing opened");
                                assert_eq!(live, 0, "seed {seed}: retired a live channel");
                                open.remove(&d);
                                // The next lifetime starts at position one.
                                sequence.remove(&d);
                                taken.remove(&d);
                                released.remove(&d);
                            }
                            Err(FreeRefusal::Owed(RetireRefusal::LivePositions {
                                outstanding: n,
                            })) => {
                                assert!(was_open);
                                assert_eq!(n, live);
                                assert!(n > 0);
                                channel_retires_refused += 1;
                            }
                            // A free of a domain no definition opened, which
                            // the sweep produces because it names domains
                            // rather than only open ones.
                            Err(FreeRefusal::NotOpen(refusal)) => {
                                assert!(!was_open, "seed {seed}: refused an open domain");
                                assert_eq!(refusal, Refusal::ChannelNotOpen { channel: d });
                                assert_eq!(live, 0);
                                channel_frees_unopened += 1;
                            }
                            // The sweep names only child domains, so the root
                            // refusal is unreachable from here — spelled out
                            // rather than defaulted so that a sweep that grew
                            // the root domain would fail here instead of
                            // counting a refusal as an unopened free.
                            Err(FreeRefusal::IsRoot) => {
                                panic!("seed {seed}: the sweep named the root domain")
                            }
                        }
                    }
                    // Admit a packet, which may be wrong in any of five ways.
                    3..=9 => {
                        let opcode = OPCODES[rng.below(OPCODES.len() as u64) as usize];
                        let domain = DOMAINS[rng.below(3) as usize];
                        let stale = rng.below(6) == 0;
                        let mut p = packet(opcode);
                        p.domain = domain;
                        p.session = if stale { generation.next() } else { generation };
                        p.payload = empty_payload(Channel::Child, opcode);

                        let expected = expected_refusal(live_device, stale, opcode, domain, &open);
                        let before_ingress = next_ingress;
                        match s.admit(&p) {
                            Ok(a) => {
                                assert!(
                                    expected.is_none(),
                                    "seed {seed}: admitted what should refuse as {expected:?}"
                                );
                                assert_eq!(
                                    a.transaction.identity.ingress,
                                    IngressOrdinal(next_ingress),
                                    "seed {seed}: ordinal"
                                );
                                next_ingress += 1;
                                let seq = sequence.entry(domain).or_insert(0);
                                *seq += 1;
                                assert_eq!(
                                    a.transaction.identity.domain_sequence,
                                    ChannelSequence(*seq),
                                    "seed {seed}: channel sequence"
                                );
                                assert_eq!(a.transaction.identity.session, generation);
                                outstanding.push((IngressOrdinal(next_ingress - 1), domain));
                                *taken.entry(domain).or_default() += 1;
                                admitted += 1;
                            }
                            Err(refusal) => {
                                let expected =
                                    expected.expect("a refusal the shadow did not predict");
                                assert_eq!(
                                    std::mem::discriminant(&refusal),
                                    std::mem::discriminant(&expected),
                                    "seed {seed}: refused as {refusal:?}, expected {expected:?}"
                                );
                                // Nothing was consumed.
                                assert_eq!(before_ingress, next_ingress);
                                count_refusal(
                                    refusal,
                                    &mut refused_unknown,
                                    &mut refused_unestablished,
                                    &mut refused_closed_channel,
                                    &mut refused_generation,
                                    &mut refused_device_lost,
                                );
                            }
                        }
                    }
                    // A transaction finished.
                    10..=12 => {
                        if outstanding.is_empty() {
                            continue;
                        }
                        let i = rng.below(outstanding.len() as u64) as usize;
                        let (ordinal, domain) = outstanding.swap_remove(i);
                        let n = done(&mut s, ordinal).len();
                        *released.entry(domain).or_default() += n;
                        completions += 1;
                    }
                    // A transaction will never finish.
                    13 => {
                        if outstanding.is_empty() {
                            continue;
                        }
                        let i = rng.below(outstanding.len() as u64) as usize;
                        let (ordinal, domain) = outstanding.swap_remove(i);
                        let n = s.withdraw(ordinal).len();
                        // A withdrawal releases what was queued behind it and
                        // does not publish its own position, so its own place
                        // leaves the order too.
                        *released.entry(domain).or_default() += n + 1;
                        withdrawals += 1;
                    }
                    // The guest reset. The ordering plane is untouched.
                    14 => {
                        let before = s.scheduler().pending();
                        generation = s.reset().generation;
                        assert_eq!(s.generation(), generation, "seed {seed}: reset generation");
                        assert_eq!(
                            s.scheduler().pending(),
                            before,
                            "seed {seed}: a reset dropped accepted work"
                        );
                        assert_eq!(
                            s.epoch(),
                            s.lifetime().epoch,
                            "seed {seed}: a reset moved the device epoch"
                        );
                        resets += 1;
                    }
                    // The device died, or was replaced.
                    _ => {
                        if !live_device {
                            let epoch = s.recreate_device().expect("it was lost");
                            assert_eq!(s.epoch(), epoch);
                            live_device = true;
                        } else if rng.below(3) == 0 {
                            let loss = s.device_lost();
                            let mut expected: Vec<IngressOrdinal> =
                                outstanding.iter().map(|(o, _)| *o).collect();
                            expected.sort_unstable();
                            let mut stranded = loss.stranded.clone();
                            stranded.sort_unstable();
                            assert_eq!(
                                stranded, expected,
                                "seed {seed}: the loss named the wrong work"
                            );
                            stranded_total += stranded.len();
                            // A loss withdraws every position it holds, and
                            // each withdrawal releases whatever was queued
                            // behind it, so nothing is left in any order.
                            for d in DOMAINS {
                                let n = taken.get(&d).copied().unwrap_or(0);
                                released.insert(d, n);
                            }
                            outstanding.clear();
                            live_device = false;
                            losses += 1;
                            assert_eq!(
                                s.scheduler().pending(),
                                0,
                                "seed {seed}: stranded work stayed in the readiness service"
                            );
                            assert_eq!(
                                s.graph().live_accesses(),
                                0,
                                "seed {seed}: stranded accesses stayed in the graph"
                            );
                        } else {
                            assert_eq!(
                                s.recreate_device(),
                                Err(Refusal::DeviceNotLost { epoch: s.epoch() }),
                                "seed {seed}: a live device was replaced"
                            );
                        }
                    }
                }

                // The three planes agree with each other and with the shadow.
                assert_eq!(
                    s.scheduler().pending(),
                    outstanding.len(),
                    "seed {seed}: readiness holds a different set"
                );
                for d in DOMAINS {
                    let expected = taken.get(&d).copied().unwrap_or(0)
                        - released.get(&d).copied().unwrap_or(0);
                    assert_eq!(
                        s.publisher().outstanding(d),
                        expected,
                        "seed {seed}: {d:?} publication order"
                    );
                }
                for d in DOMAINS {
                    assert_eq!(
                        s.channel_open(d),
                        open.contains(&d),
                        "seed {seed}: {d:?} open"
                    );
                }
                assert_eq!(
                    s.device_state() == DeviceState::Live,
                    live_device,
                    "seed {seed}: device state"
                );
            }

            // Everything comes out. A session that has drained holds nothing in
            // any of the three planes.
            for (ordinal, _) in std::mem::take(&mut outstanding) {
                let _ = s.withdraw(ordinal);
            }
            assert_eq!(s.scheduler().pending(), 0, "seed {seed}");
            assert_eq!(s.graph().live_accesses(), 0, "seed {seed}");
            for d in DOMAINS {
                assert_eq!(s.publisher().outstanding(d), 0, "seed {seed}: {d:?}");
                if s.channel_open(d) {
                    s.retire_channel(d).expect("drained");
                }
            }
        }

        // Non-vacuity: every shape an assertion above depends on reaching.
        assert!(admitted > 900, "packets admitted: {admitted}");
        assert!(completions > 400, "transactions completed: {completions}");
        assert!(withdrawals > 120, "transactions withdrawn: {withdrawals}");
        assert!(resets > 200, "guest resets: {resets}");
        assert!(losses > 200, "device losses: {losses}");
        assert!(
            stranded_total > 80,
            "work stranded by a loss: {stranded_total}"
        );
        assert!(
            channel_retires_refused > 60,
            "channel frees refused for live positions: {channel_retires_refused}"
        );
        assert!(
            channel_frees_unopened > 60,
            "channel frees refused for a domain nothing opened: {channel_frees_unopened}"
        );
        assert!(refused_unknown > 700, "unknown opcode: {refused_unknown}");
        assert!(
            refused_unestablished > 700,
            "unestablished contract: {refused_unestablished}"
        );
        assert!(
            refused_closed_channel > 1_000,
            "closed channel: {refused_closed_channel}"
        );
        assert!(
            refused_generation > 700,
            "closed generation: {refused_generation}"
        );
        assert!(
            refused_device_lost > 900,
            "device lost: {refused_device_lost}"
        );
    }

    /// The refusal `admit` owes this packet, in the order it decides them.
    ///
    /// Stated here as a precedence list rather than derived from the model, so
    /// a check that moved would show up as a disagreement about *which* reason
    /// rather than only about whether there was one.
    fn expected_refusal(
        live_device: bool,
        stale: bool,
        opcode: u16,
        domain: ChannelId,
        open: &BTreeSet<ChannelId>,
    ) -> Option<Refusal> {
        if !live_device {
            return Some(Refusal::DeviceLost {
                epoch: DeviceEpoch::FIRST,
            });
        }
        if stale {
            return Some(Refusal::GenerationClosed {
                named: SessionGeneration::FIRST,
                current: SessionGeneration::FIRST,
            });
        }
        if find(Channel::Child, opcode).is_none() {
            return Some(Refusal::UnknownCommand {
                channel: Channel::Child,
                opcode,
            });
        }
        if classify(Channel::Child, opcode).is_none() {
            return Some(Refusal::UnestablishedContract {
                channel: Channel::Child,
                opcode,
            });
        }
        if !open.contains(&domain) {
            return Some(Refusal::ChannelNotOpen { channel: domain });
        }
        None
    }

    fn count_refusal(
        refusal: Refusal,
        unknown: &mut usize,
        unestablished: &mut usize,
        closed_channel: &mut usize,
        generation: &mut usize,
        device_lost: &mut usize,
    ) {
        match refusal {
            Refusal::UnknownCommand { .. } => *unknown += 1,
            Refusal::UnestablishedContract { .. } => *unestablished += 1,
            Refusal::ChannelNotOpen { .. } => *closed_channel += 1,
            Refusal::GenerationClosed { .. } => *generation += 1,
            Refusal::DeviceLost { .. } => *device_lost += 1,
            other => panic!("unexpected refusal {other:?}"),
        }
    }
}
