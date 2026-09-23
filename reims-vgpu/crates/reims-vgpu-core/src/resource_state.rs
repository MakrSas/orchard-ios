//! The content-representation operations, and the one of them that has teeth.
//!
//! # Why these survive into the model at all
//!
//! All nine are proven no-ops in the ledger, and the cells say why: "guest
//! pages are written directly; there is no host-private texture layout", "guest
//! pages are the single copy of resource content". Both sentences describe the
//! *placement* the current executor chose. Neither is a property of the guest's
//! request.
//!
//! The plan gives placement to the executor — it is the only layer allowed to
//! turn host capabilities into placement and transfer policy — and
//! [`crate::content`] already models a device-owned replica precisely because
//! an executor may choose one. The moment it does, `synchronizeResource:` stops
//! being free: it becomes a readback the guest's next CPU read depends on. A
//! model that had dropped the record at decode would have nothing to schedule
//! that readback from.
//!
//! So these are operations, and the same reasoning `crate::sync` applies to
//! barriers applies here: a host fact belongs to the host.
//!
//! # Only one of the five has a definable effect
//!
//! [`ContentDirective::Synchronize`] means the guest's own pages must be
//! current for the named content. That is a statement about
//! [`crate::content::Replica`] and it is checkable, so
//! [`ResourceStateOp::publication_requirement`] returns it.
//!
//! The other four do not get invented effects. `optimizeContentsFor…` is a
//! preference about a representation the model does not describe;
//! `invalidateCompressedTexture:` says a compression metadata this model has no
//! concept of is stale. Mapping either onto a replica discard would be a guess
//! with a cost: discarding a device replica that holds the *authority* destroys
//! content the guest never wrote. So they are carried, reported, and left for
//! an executor that has a representation to act on.

use crate::access::{
    AccessMode, Participation, ParticipationExtent, Participations, SubresourceRange,
};
use crate::content::Replica;
use crate::identity::ResourceId;
pub use reims_vgpu_protocol::resource_state::{ContentDirective, ContentTarget};

/// A texture subresource named by a `slice:level:` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SliceLevel {
    pub slice: u16,
    pub level: u16,
}

impl SliceLevel {
    /// The one level and one slice this names.
    #[must_use]
    pub const fn subresource(self) -> SubresourceRange {
        SubresourceRange::one(self.slice as u32, self.level as u32)
    }
}

/// What a content-representation operation is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceStateTarget {
    /// A resource, and the subresource within it the record named.
    ///
    /// `subresource: None` is "the record carried a ref alone", not "slice 0,
    /// level 0". A whole-texture synchronise covers every level, and collapsing
    /// it to the top one would publish a fraction of what the guest asked for.
    Resource {
        resource: ResourceId,
        subresource: Option<SliceLevel>,
    },
    /// The encoder that issued it, because the record named nothing.
    ///
    /// A variant rather than a resource with a reserved id: "the record named
    /// nothing" and "the record named resource zero" have to stay different, or
    /// the flush orders against an unrelated object.
    Encoder,
}

/// One content-representation operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceStateOp {
    pub directive: ContentDirective,
    pub target: ResourceStateTarget,
}

impl ResourceStateOp {
    /// The target shape the record was sent at.
    #[must_use]
    pub const fn target_kind(&self) -> ContentTarget {
        match self.target {
            ResourceStateTarget::Resource {
                subresource: Some(_),
                ..
            } => ContentTarget::SliceLevel,
            ResourceStateTarget::Resource {
                subresource: None, ..
            } => ContentTarget::WholeResource,
            ResourceStateTarget::Encoder => ContentTarget::Encoder,
        }
    }

    /// The resource this operation is about, if it is about one.
    #[must_use]
    pub const fn resource(&self) -> Option<ResourceId> {
        match self.target {
            ResourceStateTarget::Resource { resource, .. } => Some(resource),
            ResourceStateTarget::Encoder => None,
        }
    }

    /// The replica that must be current when this operation completes.
    ///
    /// `Some(GuestPages)` for a synchronise and nothing for the rest. This is
    /// the model's whole semantic contribution here, and it is deliberately
    /// small: four of the five directives have no effect this layer can state
    /// without inventing one.
    #[must_use]
    pub const fn publication_requirement(&self) -> Option<Replica> {
        match self.directive {
            ContentDirective::Synchronize => Some(Replica::GuestPages),
            ContentDirective::OptimizeForCpu
            | ContentDirective::OptimizeForGpu
            | ContentDirective::InvalidateCompressed
            | ContentDirective::FlushCompressedReinterpretation => None,
        }
    }

    /// The memory this record names by itself.
    ///
    /// A **read** of the named content for a synchronise, and nothing for the
    /// other four.
    ///
    /// The read follows directly from [`Self::publication_requirement`]: a
    /// synchronise says the guest's own pages must be current for this
    /// content, and making them current means the content is read. So a
    /// synchronise must not be ordered before the writes it is publishing, and
    /// a read participation is exactly the edge that says so. Declaring it a
    /// write instead would be claiming this operation produces the resource's
    /// next content, which it does not — what it produces is the guest replica,
    /// and that is `publication_requirement`'s statement rather than a
    /// participation.
    ///
    /// The other four name nothing because the model has no representation for
    /// what they change; see the module doc for why inventing one has a cost.
    /// `ResourceStateTarget::Encoder` names nothing because the record did.
    #[must_use]
    pub fn participations(&self) -> Participations {
        if self.publication_requirement().is_none() {
            return Participations::NONE;
        }
        let ResourceStateTarget::Resource {
            resource,
            subresource,
        } = self.target
        else {
            return Participations::NONE;
        };
        Participations::one(Participation {
            resource,
            // A whole-resource synchronise covers every level, and the record
            // carried no level count — so the honest extent is the resource
            // and not its top level. Narrowing here is the hazard edge that
            // does not get built.
            extent: subresource.map_or(ParticipationExtent::Whole, |s| {
                ParticipationExtent::Subresource(s.subresource())
            }),
            mode: AccessMode::Read,
            // A content request names no shader stage.
            api_stages: 0,
        })
    }

    /// Whether an executor is free to do nothing at all for this operation
    /// *once it has stated where the content is*.
    ///
    /// Never answered here — the method returns what the model knows, which is
    /// that a synchronise depends on placement and the other four depend on a
    /// representation. It exists so a caller asking "can I skip this" has to
    /// pass through a name that says the answer is not the model's.
    #[must_use]
    pub const fn depends_on_placement(&self) -> bool {
        matches!(self.directive, ContentDirective::Synchronize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};
    use crate::operation::{classify, OperationClass, OperationHome};
    use reims_vgpu_protocol::closure::LEDGER;
    use reims_vgpu_protocol::resource_state::content_request;

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn op(directive: ContentDirective, subresource: Option<SliceLevel>) -> ResourceStateOp {
        ResourceStateOp {
            directive,
            target: ResourceStateTarget::Resource {
                resource: res(1),
                subresource,
            },
        }
    }

    /// Every content request the protocol names is classified as a
    /// resource-state operation, and every classified resource-state operation
    /// that carries an opcode is a content request.
    ///
    /// The second direction is the one that would fail first: the residency
    /// records also classify as `ResourceState`, and they are all unresolved —
    /// so if one of them is ever closed, this test says the payload vocabulary
    /// owes it a shape.
    #[test]
    fn the_content_requests_are_exactly_the_judged_resource_state_operations() {
        for op in LEDGER {
            let Some(opcode) = op.opcode else { continue };
            let classified =
                classify(op) == Some(OperationHome::Stream(OperationClass::ResourceState));
            let judged_request = content_request(op.rail, opcode).is_some();
            // Every content request is a classified resource-state operation.
            assert!(
                !judged_request || classified,
                "{:?} {opcode:#x} is a content request that is not classified as one",
                op.rail
            );
            // And every classified one that is *not* a request is a residency
            // record, which is unresolved and so cannot be classified at all.
            assert!(
                !classified || judged_request,
                "{:?} {opcode:#x} is a resource-state operation with no payload",
                op.rail
            );
        }
    }

    /// The residency records are the resource-state operations that are *not*
    /// content requests, and every one of them is unresolved. That is what
    /// keeps them out of the vocabulary rather than a special case.
    #[test]
    fn the_residency_records_are_unresolved_and_so_have_no_payload() {
        let mut residency = 0;
        for op in LEDGER {
            let Some(opcode) = op.opcode else { continue };
            if content_request(op.rail, opcode).is_some() {
                continue;
            }
            if !op.selector.starts_with("use") {
                continue;
            }
            residency += 1;
            assert!(
                op.closure.blocks_cutover(),
                "{:?} {opcode:#x} is a closed residency row and now needs a payload",
                op.rail
            );
            assert_eq!(classify(op), None);
        }
        assert!(residency >= 5, "the useResource/useHeap family");
    }

    /// A whole-resource record is not slice 0 level 0.
    #[test]
    fn a_whole_resource_request_is_not_the_top_subresource() {
        let whole = op(ContentDirective::Synchronize, None);
        let top = op(
            ContentDirective::Synchronize,
            Some(SliceLevel { slice: 0, level: 0 }),
        );
        assert_ne!(whole, top);
        assert_eq!(whole.target_kind(), ContentTarget::WholeResource);
        assert_eq!(top.target_kind(), ContentTarget::SliceLevel);
    }

    /// Synchronise publishes to the guest's pages; nothing else claims an
    /// effect.
    #[test]
    fn only_synchronise_states_a_publication_requirement() {
        assert_eq!(
            op(ContentDirective::Synchronize, None).publication_requirement(),
            Some(Replica::GuestPages)
        );
        for directive in [
            ContentDirective::OptimizeForCpu,
            ContentDirective::OptimizeForGpu,
            ContentDirective::InvalidateCompressed,
            ContentDirective::FlushCompressedReinterpretation,
        ] {
            assert_eq!(op(directive, None).publication_requirement(), None);
            assert!(!op(directive, None).depends_on_placement());
        }
        assert!(op(ContentDirective::Synchronize, None).depends_on_placement());
    }

    /// The encoder-scoped flush is about no resource, and that is readable
    /// from the operation rather than by inspecting a sentinel.
    #[test]
    fn the_encoder_scoped_flush_names_no_resource() {
        let flush = ResourceStateOp {
            directive: ContentDirective::FlushCompressedReinterpretation,
            target: ResourceStateTarget::Encoder,
        };
        assert_eq!(flush.resource(), None);
        assert_eq!(flush.target_kind(), ContentTarget::Encoder);
        assert_eq!(
            op(ContentDirective::Synchronize, None).resource(),
            Some(res(1))
        );
    }

    /// A slice/level record names exactly one subresource, so two synchronises
    /// of different levels are independent.
    #[test]
    fn two_levels_of_one_texture_are_separate_subresources() {
        let a = SliceLevel { slice: 0, level: 0 }.subresource();
        let b = SliceLevel { slice: 0, level: 1 }.subresource();
        assert!(!a.overlaps(b));
    }
}
