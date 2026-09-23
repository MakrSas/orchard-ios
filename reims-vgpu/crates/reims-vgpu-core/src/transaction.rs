//! The transaction envelope: what every accepted packet is, before anything
//! knows how a host would execute it.
//!
//! # One shape for GPU and non-GPU work
//!
//! A draw is not the unit of scheduling here and neither is an EXEC. Every
//! accepted FIFO packet is an ordered device transaction with the same
//! envelope: channel identity, a channel-local sequence, an ingress ordinal,
//! explicit prerequisites, a completion obligation, and one typed payload. A
//! resource delete, a display present and an EXEC differ in their payload and
//! in nothing else, because ordering and publication are owed to all three
//! equally — and the architecture this replaces gave each of them its own
//! partial mechanism.
//!
//! # Five payloads, and no catch-all among them
//!
//! [`PayloadClass::Control`] is the one that could rot into a bucket, so the
//! rule is written into [`classify`] and checked: a command may be `Control`
//! only when its established contract is a real control operation or an
//! acknowledged no-op. A command whose contract is *unknown* has no class at
//! all — it is a typed refusal at ingress, not a `Control` that quietly does
//! nothing — and [`crate::identity`]'s ordering guarantees never apply to work
//! that was never accepted.
//!
//! That rule is not enforceable by reading this file, so it is enforced against
//! the closure ledger: every packet class the ledger has judged maps to exactly
//! one payload, and every class it has *not* judged maps to none.

use crate::access::{AccessIntent, AccessKey, AccessMode, ResourceKey};
use crate::control::ControlOp;
use crate::exec::ExecWork;
use crate::identity::{ChannelId, ResourceId};
use crate::identity::{CompletionStamp, StampWait, TransactionIdentity};
use crate::lifecycle::LifecycleOp;
use crate::present::PresentPacket;
use crate::query::QueryRequest;
use reims_vgpu_protocol::closure::Closure;
use reims_vgpu_protocol::packets::{find, Channel};

/// What kind of work a transaction carries.
///
/// The payload contents are each their own type; this is the discriminant, and
/// it exists separately because ingress has to know which payload a packet
/// becomes before it has decoded one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PayloadClass {
    /// The GPU-work transaction: a counted resource table and an ordered list
    /// of serialized command streams.
    Exec,
    /// Task, object, resource and heap lifetime: create, delete, map, unmap,
    /// replace-physical, invalidate, synchronize, discard.
    ResourceLifecycle,
    /// A question with a decoded reply destination. The guest blocks on the
    /// answer, so an unanswered one is a wrong answer rather than lost work —
    /// which is why a query is its own class and not a control command.
    Query,
    /// A display present, carrying the surface identity and the presentation
    /// contract.
    Present,
    /// Display, cursor and channel control, and the acknowledged no-ops.
    /// **Not** a bucket for commands whose contract is unknown; see the module
    /// docs.
    Control,
}

impl PayloadClass {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exec => "exec",
            Self::ResourceLifecycle => "resource_lifecycle",
            Self::Query => "query",
            Self::Present => "present",
            Self::Control => "control",
        }
    }
}

/// The payload class a packet becomes, or `None` when the model refuses it.
///
/// `None` is not "unhandled". It is the answer for a command whose contract has
/// not been established, and the caller's obligation is a typed refusal at
/// ingress — because a transaction the model cannot describe must not be given
/// ordering and completion guarantees it cannot honour.
#[must_use]
pub fn classify(channel: Channel, opcode: u16) -> Option<PayloadClass> {
    use Channel::{Child, Root};
    use PayloadClass::{Control, Exec, Present, Query, ResourceLifecycle};
    let judged = find(channel, opcode)?;
    if judged.closure.blocks_cutover() {
        return None;
    }
    Some(match (channel, opcode) {
        // The GPU-work packet, and the only one.
        (Child, 0x37) => Exec,

        // Task, object-list, resource and mapping lifetime. Root and child are
        // two views of one flat opcode space, so the shared numbers appear
        // twice on purpose rather than through a fallthrough.
        (Root, 0x20) | (Child, 0x20) => ResourceLifecycle,
        (Root, 0x33) | (Child, 0x33) => ResourceLifecycle,
        (Root, 0x38) | (Child, 0x38) => ResourceLifecycle,
        (Child, 0x22 | 0x25 | 0x28 | 0x34 | 0x35 | 0x36 | 0x39 | 0x3c | 0x3e | 0x3f) => {
            ResourceLifecycle
        }

        // Questions with reply destinations.
        (Root, 0x2d | 0x3a) => Query,
        (Child, 0x3b | 0x40) => Query,

        // The three present forms. Enumerated rather than written as the
        // range they happen to occupy: they are three named commands whose
        // trailers differ, and a range would quietly adopt a fourth number if
        // the dispatch table ever grew one between them.
        #[allow(clippy::manual_range_patterns)]
        (Child, 0x06 | 0x07 | 0x08) => Present,

        // Display registration, cursor, channel lifetime, and the fence with no
        // payload. Everything left that the ledger has judged.
        _ => Control,
    })
}

/// Whether a class is one this model executes as GPU work.
///
/// Separate from [`PayloadClass::Exec`] being the only such class today,
/// because the question a reader asks is "does this reach an executor", and
/// answering it by naming one variant is how a second one gets added without
/// the readers noticing.
#[must_use]
pub const fn reaches_an_executor(class: PayloadClass) -> bool {
    matches!(class, PayloadClass::Exec)
}

/// Whether the closure ledger records this class as doing nothing by contract.
///
/// A `Control` transaction is not automatically a no-op — a present is not, a
/// cursor move is not — so the answer comes from the ledger rather than from
/// the payload class.
#[must_use]
pub fn is_acknowledged_noop(channel: Channel, opcode: u16) -> bool {
    matches!(
        find(channel, opcode).map(|p| p.closure),
        Some(Closure::ProvenNoOp { .. })
    )
}

/// What a transaction carries, and what it touches.
///
/// # The class was a discriminant, and a discriminant executes nothing
///
/// [`PayloadClass`] answers "which kind of work is this" at ingress, before
/// anything is decoded. It is not the work. A `DeviceTransaction` that carried
/// only the class named a packet it could not describe: an executor holding one
/// had to go back to the bytes, and every access the packet made had to be
/// stated *beside* the class in a list nothing tied to it.
///
/// That "beside" is the defect. An envelope with its own `accesses` field and a
/// payload with its own contents are two descriptions of one packet that can
/// disagree — a delete whose envelope named a backing its op did not, an EXEC
/// whose envelope listed accesses its records never made. Both were
/// representable, and a hazard edge built from the wrong one is a race rather
/// than a slowdown.
///
/// So the payload owns what it touches, and [`Self::accesses`] is the only way
/// to ask. There is one list per transaction and the payload is holding it.
#[derive(Clone, Debug, PartialEq)]
pub enum Payload {
    /// The GPU-work transaction. Its accesses are its records', collected by
    /// [`crate::exec::ExecBuilder`] as they resolved; nothing else may add to
    /// them.
    Exec(ExecWork),
    /// One lifetime operation, and the resources it touches as the namespace
    /// that owns them resolved.
    ResourceLifecycle(LifecyclePayload),
    /// A question, and the write its answer will make.
    Query(QueryPayload),
    /// What the guest asked to show, and the frame reading it.
    Present(PresentPayload),
    /// A control operation. **No access list, and that is a contract claim
    /// rather than an omission**: opening a channel, moving a cursor, acking a
    /// display and doing nothing all touch no guest resource, so a control
    /// packet that appeared to have one would be a decode error somewhere
    /// upstream. Held to by `control_transactions_touch_no_resource`.
    Control(ControlOp),
}

/// A lifetime operation and the accesses that resolve it.
///
/// **The two have to describe the same work, and until this type they did not
/// have to.** `Payload`'s own doc says the payload owns what it touches so an
/// envelope and its operation cannot disagree; a `Synchronize` naming three
/// resources beside an access list naming two others was representable, and the
/// hazard edges built from the list would order the operation against memory it
/// does not use while leaving the memory it does use unordered.
///
/// [`LifecycleOp::resources`] is the operation's own statement. When it is
/// non-empty, [`Self::new`] requires the accesses to name exactly that set —
/// every one of them, and nothing else. When it is empty the operation makes no
/// per-resource statement (a task teardown, a map notice) and the access list is
/// the only one there is, so nothing here constrains it.
#[derive(Clone, Debug, PartialEq)]
pub struct LifecyclePayload {
    op: LifecycleOp,
    accesses: Vec<AccessIntent>,
}

/// Why an operation and the accesses offered for it do not describe the same
/// work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessMismatch {
    /// An access names a resource the operation does not. The edges it builds
    /// order this operation against memory it never touches.
    Unnamed { resource: ResourceId },
    /// A resource the operation names has no access. Nothing orders the
    /// operation against work still reading it, which for a delete or a discard
    /// is a use-after-free.
    Unaccessed { resource: ResourceId },
}

impl AccessMismatch {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unnamed { .. } => "lifecycle_access_names_unnamed_resource",
            Self::Unaccessed { .. } => "lifecycle_resource_has_no_access",
        }
    }
}

impl LifecyclePayload {
    /// Build the payload, holding the accesses to the operation's own resource
    /// list.
    ///
    /// `resolved` pairs each access with the resource it was resolved for,
    /// because the two live in different name spaces: an operation names
    /// [`ResourceId`]s and an access names a backing. Only the resolver knows
    /// which is which, so it says so here rather than leaving the join to be
    /// re-derived — or not made at all.
    ///
    /// # Errors
    ///
    /// [`AccessMismatch`] when the two sets differ in either direction.
    pub fn new(
        op: LifecycleOp,
        resolved: Vec<(ResourceId, AccessIntent)>,
    ) -> Result<Self, AccessMismatch> {
        if !op.resources().is_empty() {
            for (resource, _) in &resolved {
                if !op.resources().contains(resource) {
                    return Err(AccessMismatch::Unnamed {
                        resource: *resource,
                    });
                }
            }
            for resource in op.resources() {
                if !resolved.iter().any(|(named, _)| named == resource) {
                    return Err(AccessMismatch::Unaccessed {
                        resource: *resource,
                    });
                }
            }
        }
        Ok(Self {
            op,
            accesses: resolved.into_iter().map(|(_, access)| access).collect(),
        })
    }

    #[must_use]
    pub const fn op(&self) -> &LifecycleOp {
        &self.op
    }

    #[must_use]
    pub fn accesses(&self) -> &[AccessIntent] {
        &self.accesses
    }
}

/// A present and the reads it makes of what it shows.
///
/// Two claims the free `{ packet, accesses }` pair could not make.
///
/// **Every access is for the mapping the packet names.** A present shows one
/// thing — the guest's display pipe serializes plane 0's surface into a
/// fixed-size command and there is no plane list on the wire — so an access
/// resolved for a different mapping is an envelope describing a frame the
/// packet did not ask for. More than one access is still legitimate: a
/// biplanar surface is two planes of one mapping.
///
/// **No access writes.** A present reads what the guest already produced; the
/// writer is the EXEC the present waits for. Modelled as a writer it would
/// reserve a content version, beat the real writer's, and publish bytes nothing
/// produced.
///
/// The list may not be empty. A present that names nothing is ordered against
/// nothing and shows whatever happens to be at the surface — the stale-frame
/// class. A target that could not be resolved says so with
/// [`AccessKey::DomainOnly`], which is the vocabulary for exactly that and
/// which still buys submission-domain ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct PresentPayload {
    packet: PresentPacket,
    accesses: Vec<AccessIntent>,
}

/// Why a present's accesses do not describe the frame its packet asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentMismatch {
    /// An access was resolved for a mapping this packet does not show.
    NotTheTarget {
        shown: crate::identity::MappingId,
        named: crate::identity::MappingId,
    },
    /// An access writes. A present reads.
    Writes { mode: AccessMode },
    /// No access at all, so nothing orders the present against the work that
    /// draws the frame.
    NothingNamed,
}

impl PresentMismatch {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NotTheTarget { .. } => "present_access_names_another_mapping",
            Self::Writes { .. } => "present_access_writes",
            Self::NothingNamed => "present_names_nothing",
        }
    }
}

impl PresentPayload {
    /// Build the payload, holding the accesses to the packet's own target.
    ///
    /// `resolved` pairs each access with the mapping it was resolved for, for
    /// [`LifecyclePayload::new`]'s reason: a packet names a mapping and an
    /// access names a backing, and only the mapper knows which is which.
    ///
    /// # Errors
    ///
    /// [`PresentMismatch`] when an access names another mapping, writes, or
    /// when there are none.
    pub fn new(
        packet: PresentPacket,
        resolved: Vec<(crate::identity::MappingId, AccessIntent)>,
    ) -> Result<Self, PresentMismatch> {
        if resolved.is_empty() {
            return Err(PresentMismatch::NothingNamed);
        }
        for (named, access) in &resolved {
            if *named != packet.mapping {
                return Err(PresentMismatch::NotTheTarget {
                    shown: packet.mapping,
                    named: *named,
                });
            }
            if access.mode.writes() {
                return Err(PresentMismatch::Writes { mode: access.mode });
            }
        }
        Ok(Self {
            packet,
            accesses: resolved.into_iter().map(|(_, access)| access).collect(),
        })
    }

    #[must_use]
    pub const fn packet(&self) -> &PresentPacket {
        &self.packet
    }

    #[must_use]
    pub fn accesses(&self) -> &[AccessIntent] {
        &self.accesses
    }
}

/// A query and the one access it makes.
///
/// **The access is not a field beside the request; it is derived from it.** A
/// query touches exactly one thing — the window its reply is written into — and
/// that window is already named by [`QueryRequest::destination`]. Held as a
/// `Vec<AccessIntent>` beside the request, as the other classes hold theirs, it
/// was free to be empty or to name something else, and either is a real defect:
/// an answer the dependency graph does not know about is content a later
/// transfer may overwrite between the write and the guest's read, which is
/// exactly what [`crate::query::ReplyWrite`]'s `#[must_use]` says.
///
/// So the fields are private and [`Self::new`] is the only way in. The
/// alternative — a public pair and a test that they agree — is a rule someone
/// has to remember at each construction.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryPayload {
    request: QueryRequest,
    access: AccessIntent,
}

impl QueryPayload {
    /// Build the payload for a query, deriving its access from its destination.
    ///
    /// `output` is the content version the write reserves, `None` when the
    /// destination's content is not versioned. The mode is
    /// [`AccessMode::Write`] and never read-modify-write: a reply is written
    /// whole and this device reads nothing at the destination — a guest's
    /// previous contents there are what it would see if no answer landed, which
    /// is the failure this write exists to prevent, not an input to it.
    #[must_use]
    pub fn new(
        request: QueryRequest,
        domain: ChannelId,
        output: Option<crate::access::ContentVersion>,
    ) -> Self {
        let destination = request.destination;
        Self {
            request,
            access: AccessIntent {
                domain,
                key: AccessKey::Range(
                    ResourceKey {
                        backing: destination.backing,
                        // A reply destination is a guest buffer the request
                        // named, not a window of a heap this device placed.
                        heap: None,
                    },
                    destination.bytes,
                ),
                mode: AccessMode::Write,
                // A reply is written by the device, not by a pipeline stage.
                api_stages: 0,
                input_content_version: None,
                output_content_version: output,
            },
        }
    }

    #[must_use]
    pub const fn request(&self) -> &QueryRequest {
        &self.request
    }

    /// The write the answer will make. There is exactly one.
    #[must_use]
    pub const fn access(&self) -> &AccessIntent {
        &self.access
    }
}

impl Payload {
    /// Which class this is.
    #[must_use]
    pub const fn class(&self) -> PayloadClass {
        match self {
            Self::Exec(_) => PayloadClass::Exec,
            Self::ResourceLifecycle(_) => PayloadClass::ResourceLifecycle,
            Self::Query(_) => PayloadClass::Query,
            Self::Present(_) => PayloadClass::Present,
            Self::Control(_) => PayloadClass::Control,
        }
    }

    /// Everything this transaction touches, at the precision the contract
    /// supplied.
    ///
    /// Empty is a claim — that the transaction touches no resource — and not an
    /// absence of information; imprecision is
    /// [`crate::access::AccessKey::DomainOnly`].
    #[must_use]
    pub fn accesses(&self) -> &[AccessIntent] {
        match self {
            Self::Exec(work) => &work.accesses,
            Self::ResourceLifecycle(lifecycle) => lifecycle.accesses(),
            Self::Present(present) => present.accesses(),
            Self::Query(query) => std::slice::from_ref(query.access()),
            Self::Control(_) => &[],
        }
    }

    /// The EXEC work, for the executor that is the only reader entitled to it.
    #[must_use]
    pub const fn exec(&self) -> Option<&ExecWork> {
        match self {
            Self::Exec(work) => Some(work),
            _ => None,
        }
    }
}

/// One accepted packet, with everything the model needs and nothing a host
/// would.
///
/// The envelope is identical for an EXEC, a resource delete and a present.
/// That is the point: ordering, prerequisites and publication are owed to all
/// three equally, and the architecture this replaces gave each of them its own
/// partial mechanism. What differs between them is [`Self::payload`] and
/// [`Self::accesses`].
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceTransaction {
    /// Where this packet sits, in every order the device keeps. Assigned by
    /// [`crate::session::SessionModel::admit`], which is the only service that
    /// observes arrival — see [`TransactionIdentity`].
    pub identity: TransactionIdentity,
    /// Points that must be published before this may begin. Decoded at ingress
    /// and before any packet side effect, because a packet that acted and then
    /// discovered it had to wait has already happened.
    pub stamp_waits: Vec<StampWait>,
    /// What this publishes when its work has completed, if it publishes
    /// anything.
    pub completion: Option<CompletionStamp>,
    /// The work, and everything it touches. There is no access list beside it;
    /// see [`Payload`].
    pub payload: Payload,
}

impl DeviceTransaction {
    /// Everything this transaction touches.
    #[must_use]
    pub fn accesses(&self) -> &[AccessIntent] {
        self.payload.accesses()
    }

    /// Which class of work this is.
    #[must_use]
    pub const fn class(&self) -> PayloadClass {
        self.payload.class()
    }

    /// This transaction as the executor sees it, when it is GPU work.
    ///
    /// Derived, not stored. The identity is this envelope's and the work is
    /// this envelope's payload, so there is no copy to keep in step — which is
    /// the whole reason [`crate::exec::ExecTransaction`] borrows.
    #[must_use]
    pub const fn exec(&self) -> Option<crate::exec::ExecTransaction<'_>> {
        match self.payload.exec() {
            Some(work) => Some(work.stamp(self.identity)),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::packets::LEDGER;

    fn resource(slot: u32) -> ResourceId {
        ResourceId {
            slot: crate::identity::ObjectListRef(slot),
            generation: crate::identity::SlotGeneration(1),
        }
    }

    fn whole(backing: u64) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(1),
            key: AccessKey::Whole(ResourceKey {
                backing: crate::access::BackingId(backing),
                heap: None,
            }),
            mode: AccessMode::ReadWrite,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    /// An operation that names resources and an envelope that names others
    /// used to be representable, and the edges built from the envelope would
    /// have ordered the operation against memory it does not touch.
    #[test]
    fn a_lifecycle_envelope_must_name_the_operations_own_resources() {
        let op = crate::lifecycle::LifecycleOp::Synchronize {
            task: crate::identity::TaskId(1),
            resources: vec![resource(1), resource(2)],
        };
        assert!(LifecyclePayload::new(
            op.clone(),
            vec![(resource(1), whole(10)), (resource(2), whole(11))],
        )
        .is_ok());

        // A third resource nobody synchronised.
        assert_eq!(
            LifecyclePayload::new(
                op.clone(),
                vec![
                    (resource(1), whole(10)),
                    (resource(2), whole(11)),
                    (resource(3), whole(12)),
                ],
            ),
            Err(AccessMismatch::Unnamed {
                resource: resource(3)
            })
        );

        // And the direction that is a use-after-free: a resource the operation
        // acts on with nothing ordering it against work still reading it.
        assert_eq!(
            LifecyclePayload::new(op, vec![(resource(1), whole(10))]),
            Err(AccessMismatch::Unaccessed {
                resource: resource(2)
            })
        );
    }

    /// An operation that names no resource constrains nothing, because its
    /// access list is the only statement there is.
    ///
    /// A task teardown retires everything in the task and names none of it; a
    /// map notice names an address interval. Requiring an empty access list for
    /// either would say the transaction touches nothing, which is the opposite
    /// of true.
    #[test]
    fn an_operation_naming_no_resource_leaves_its_accesses_alone() {
        for op in [
            crate::lifecycle::LifecycleOp::DeleteTask {
                task: crate::identity::TaskId(1),
            },
            crate::lifecycle::LifecycleOp::UnmapMemory {
                task: crate::identity::TaskId(1),
                span: crate::access::GuestSpan {
                    base: 0x1000,
                    length: 0x1000,
                },
            },
        ] {
            assert!(op.resources().is_empty(), "{:?}", op.kind());
            let payload =
                LifecyclePayload::new(op, vec![(resource(7), whole(10)), (resource(8), whole(11))])
                    .expect("no per-resource statement to disagree with");
            assert_eq!(payload.accesses().len(), 2);
        }
    }

    /// Every kind's `resources()` is its own operation's list, and the two
    /// single-resource kinds are not "no resources".
    #[test]
    fn a_single_resource_operation_names_that_resource() {
        let one = crate::lifecycle::LifecycleOp::DeleteResource {
            task: crate::identity::TaskId(1),
            object_ref: 4,
            resource: Some(resource(4)),
        };
        assert_eq!(one.resources(), &[resource(4)]);
        assert_eq!(
            LifecyclePayload::new(one, Vec::new()),
            Err(AccessMismatch::Unaccessed {
                resource: resource(4)
            }),
            "a delete with nothing ordering it is the use-after-free case"
        );
    }

    fn present_packet(mapping: u32) -> crate::present::PresentPacket {
        crate::present::PresentPacket {
            form: crate::present::PresentForm::SwapMapping,
            mapping: crate::identity::MappingId(mapping),
            task: None,
        }
    }

    fn reads(backing: u64) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(1),
            key: AccessKey::Whole(ResourceKey {
                backing: crate::access::BackingId(backing),
                heap: None,
            }),
            mode: AccessMode::Read,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    /// A present shows one mapping, and every access it carries is for that
    /// one. More than one is still legitimate — a biplanar surface is two
    /// planes of one mapping.
    #[test]
    fn a_present_reads_the_mapping_it_shows_and_no_other() {
        let packet = present_packet(7);
        let two_planes = PresentPayload::new(
            packet,
            vec![(packet.mapping, reads(10)), (packet.mapping, reads(11))],
        )
        .expect("two planes of one surface");
        assert_eq!(two_planes.accesses().len(), 2);

        assert_eq!(
            PresentPayload::new(packet, vec![(crate::identity::MappingId(8), reads(10))]),
            Err(PresentMismatch::NotTheTarget {
                shown: crate::identity::MappingId(7),
                named: crate::identity::MappingId(8),
            }),
            "an envelope describing a frame the packet did not ask for"
        );
    }

    /// A present reads. The writer is the EXEC it waits for, and a present
    /// modelled as a writer reserves a version, beats the real writer's and
    /// publishes bytes nothing produced.
    #[test]
    fn a_present_never_writes_what_it_shows() {
        let packet = present_packet(7);
        for mode in [
            AccessMode::Write,
            AccessMode::ReadWrite,
            AccessMode::Unknown,
        ] {
            let mut access = reads(10);
            access.mode = mode;
            assert_eq!(
                PresentPayload::new(packet, vec![(packet.mapping, access)]),
                Err(PresentMismatch::Writes { mode }),
                "{mode:?}"
            );
        }
    }

    /// A present that names nothing is ordered against nothing. An unresolved
    /// target says so with `DomainOnly`, which still buys domain ordering.
    #[test]
    fn a_present_that_names_nothing_is_refused_rather_than_unordered() {
        let packet = present_packet(7);
        assert_eq!(
            PresentPayload::new(packet, Vec::new()),
            Err(PresentMismatch::NothingNamed)
        );
        let mut unresolved = reads(0);
        unresolved.key = AccessKey::DomainOnly;
        assert!(PresentPayload::new(packet, vec![(packet.mapping, unresolved)]).is_ok());
    }

    /// A query's access is exactly the window its reply goes to.
    ///
    /// The one thing this class cannot get wrong any more. An answer the
    /// dependency graph does not know about is content a later transfer may
    /// overwrite between the write and the guest's read — and the guest is
    /// blocked on the completion word, so it reads whatever is there the moment
    /// the word advances.
    #[test]
    fn a_query_touches_its_reply_window_and_nothing_else() {
        let destination = crate::query::ReplyDestination {
            backing: crate::access::BackingId(9),
            bytes: crate::access::ByteRange {
                offset: 0x200,
                length: 4096,
            },
        };
        let payload = Payload::Query(QueryPayload::new(
            crate::query::QueryRequest {
                kind: crate::query::QueryKind::DeviceInfo,
                destination,
                reply: crate::query::ReplyShape::Fixed { bytes: 16 },
            },
            ChannelId(2),
            Some(crate::access::ContentVersion(7)),
        ));
        let accesses = payload.accesses();
        assert_eq!(accesses.len(), 1, "one window, not a list");
        let access = accesses[0];
        assert_eq!(
            access.key,
            AccessKey::Range(
                ResourceKey {
                    backing: destination.backing,
                    heap: None,
                },
                destination.bytes,
            ),
            "the destination the request named, at byte precision"
        );
        assert_eq!(access.domain, ChannelId(2));
        assert_eq!(
            access.mode,
            AccessMode::Write,
            "a reply is written whole; the bytes already there are the failure \
             this write prevents, not an input to it"
        );
        assert_eq!(
            access.output_content_version,
            Some(crate::access::ContentVersion(7))
        );
        assert_eq!(access.input_content_version, None);
    }

    /// The claim the module docs make and cannot check by being read: the
    /// classification is total over everything the ledger has judged, and empty
    /// over everything it has not.
    #[test]
    fn every_judged_packet_class_becomes_exactly_one_payload() {
        for p in LEDGER {
            let class = classify(p.channel, p.opcode);
            if p.closure.blocks_cutover() {
                assert_eq!(
                    class,
                    None,
                    "{} {:#04x} has no established contract, so the model must \
                     refuse it at ingress rather than accept it as a payload \
                     that quietly does nothing",
                    p.channel.name(),
                    p.opcode
                );
            } else {
                assert!(
                    class.is_some(),
                    "{} {:#04x} is judged {} and reaches no payload class",
                    p.channel.name(),
                    p.opcode,
                    p.closure.name()
                );
            }
        }
    }

    /// A packet the ledger has never heard of is not a `Control`.
    #[test]
    fn an_unjudged_opcode_has_no_payload() {
        assert_eq!(classify(Channel::Root, 0x00), None);
        assert_eq!(classify(Channel::Child, 0x1d), None);
        assert_eq!(classify(Channel::Child, 0xffff), None);
    }

    /// The retired slots are the acknowledged no-ops, and they are the only
    /// `Control` transactions that are.
    #[test]
    fn the_acknowledged_noops_are_the_retired_slots() {
        let noops: Vec<_> = LEDGER
            .iter()
            .filter(|p| is_acknowledged_noop(p.channel, p.opcode))
            .map(|p| (p.channel, p.opcode))
            .collect();
        assert_eq!(noops.len(), 15, "the reference host's retired slots");
        for (ch, op) in noops {
            assert_eq!(classify(ch, op), Some(PayloadClass::Control));
        }
    }

    #[test]
    fn exactly_one_packet_class_reaches_an_executor() {
        let executed: Vec<_> = LEDGER
            .iter()
            .filter_map(|p| classify(p.channel, p.opcode).map(|c| (p, c)))
            .filter(|(_, c)| reaches_an_executor(*c))
            .map(|(p, _)| (p.channel, p.opcode, p.name))
            .collect();
        assert_eq!(
            executed,
            vec![(Channel::Child, 0x37, "CmdExecIndirect2")],
            "a second packet class reaching an executor is a real change and \
             not a table edit"
        );
    }

    /// Present is not lifecycle and lifecycle is not control. Spot-checked
    /// against the readings that are easiest to get backwards.
    #[test]
    fn the_classes_that_look_alike_are_told_apart() {
        assert_eq!(
            classify(Channel::Child, 0x25),
            Some(PayloadClass::ResourceLifecycle),
            "CmdDeleteResource retires an object-table entry"
        );
        assert_eq!(
            classify(Channel::Child, 0x08),
            Some(PayloadClass::Present),
            "CmdDisplaySwapMapping is a present, not display control"
        );
        assert_eq!(
            classify(Channel::Child, 0x40),
            Some(PayloadClass::Query),
            "the guest blocks on the heap-texture reply, so it is a query and \
             not a control command that happens to write memory"
        );
        assert_eq!(
            classify(Channel::Child, 0x1e),
            Some(PayloadClass::Control),
            "CmdNOP's whole obligation is retiring its stamps"
        );
    }
    /// The whole model's totality claim, in one place: a judged packet reaches
    /// exactly one payload class, and that class's own vocabulary then names
    /// exactly what the packet is. Each class checks its half in its own
    /// module; this is the join, and it is the test that fails when a class
    /// gains a member nobody gave a meaning to.
    ///
    /// `Exec` is the one class whose members are not enumerated here, because
    /// there is exactly one of them and `exactly_one_packet_class_reaches_an_executor`
    /// above says so.
    #[test]
    fn every_judged_packet_reaches_a_class_and_a_meaning_within_it() {
        use crate::control::ControlKind;
        use crate::lifecycle::LifecycleKind;
        use crate::query::QueryKind;
        let mut counts = [0usize; 5];
        for p in LEDGER {
            let Some(class) = classify(p.channel, p.opcode) else {
                continue;
            };
            let named = match class {
                PayloadClass::Exec => (p.channel, p.opcode) == (Channel::Child, 0x37),
                PayloadClass::ResourceLifecycle => LifecycleKind::of(p.channel, p.opcode).is_some(),
                PayloadClass::Query => QueryKind::of(p.channel, p.opcode).is_some(),
                PayloadClass::Control => ControlKind::of(p.channel, p.opcode).is_some(),
                PayloadClass::Present => {
                    crate::present::PresentForm::of(p.channel, p.opcode).is_some()
                }
            };
            assert!(
                named,
                "{} {:#04x} ({}) reaches {} and has no meaning inside it",
                p.channel.name(),
                p.opcode,
                p.name,
                class.name()
            );
            counts[class as usize] += 1;
        }
        assert_eq!(
            counts,
            [1, 16, 4, 3, 23],
            "one EXEC, sixteen lifecycle rows, four queries, three presents,              and twenty-three control packets. A change here is a change to              what the guest may send, not a table edit."
        );
    }
}
