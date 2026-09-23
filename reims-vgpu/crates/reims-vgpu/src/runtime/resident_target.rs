//! A rail's own name for a resident render target, held by the layers above the
//! rail that must carry it without understanding it.
//!
//! # The thing that has to be carried
//!
//! A render Store leaves the pixels in the rail's own image and records that the
//! guest is owed a copy of them. Paying that debt later has to reach *the image
//! the draw wrote*, and the ledger cannot re-derive which one that was:
//! [`crate::runtime::writeback_debt::WritebackDebt::target`] records the defect
//! that proved it, where a fresh derivation read the mapping's generation as of
//! the payment and named a different image than the one the Store had filled.
//! So the ledger holds the rail's name for it, verbatim, from the arm to the
//! payment.
//!
//! # Why it is opaque
//!
//! That name is the rail's vocabulary. The Vulkan rail's is an engine
//! `TargetIdentity`; Metal's rail has no equivalent and arms no debt at all.
//! Spelling the Vulkan type in the ledger made a neutral runtime module name a
//! rail — it was a `pub type ResidentIdentity = …::TargetIdentity` on the arm
//! that compiled Vulkan and a zero-sized `NoResidentIdentity` on the arm that
//! did not, so the ledger had two shapes across a feature boundary and nothing
//! in the toolchain compared them.
//!
//! A [`ResidentTarget`] has one shape on every arm. It can be stored, cloned,
//! compared and handed back; it cannot be constructed, read or interpreted
//! except by a rail that can name its own [`RailTarget`].
//!
//! # What equality means
//!
//! Two targets are equal when the same rail issued them for the same image.
//! Targets from different rails are never equal, which is what
//! [`RailTarget::same_target`]'s downcast enforces — and which
//! [`crate::backend::select`]'s one-rail-per-process latch already makes
//! unreachable, so the check is a statement of the rule rather than a live
//! branch.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// One rail's name for one of its resident render targets.
///
/// `Eq` is by value, not by pointer: two clones of one identity name one image
/// and must compare equal, which is what the ledger's supersede check and the
/// blit rail's "is this the resident I armed" question both ask.
pub trait RailTarget: Any + Send + Sync + fmt::Debug {
    /// Whether `other` is this rail's name for the same image.
    ///
    /// Implement as a downcast to `Self` followed by the rail's own equality;
    /// a target another rail issued is not the same image and must answer
    /// `false` rather than panicking.
    fn same_target(&self, other: &dyn RailTarget) -> bool;

    /// This target as `Any`, so a rail can recover its own type from a handle
    /// it issued.
    ///
    /// Spelled out rather than left to a trait upcast so that the recovery is a
    /// visible part of the contract: a rail that issues a target is the only
    /// one that may read it back.
    fn as_any(&self) -> &dyn Any;
}

/// An opaque, cloneable handle to whatever the running rail calls a resident
/// render target. See the module doc.
///
/// `Arc` rather than `Box` because the ledger hands the same target to a
/// payment and to the census that reports it, and a rail identity is a value
/// with no destructor to run — sharing it costs one refcount and copying it
/// would cost a clone of the rail's own type through a vtable.
#[derive(Clone)]
pub struct ResidentTarget(Arc<dyn RailTarget>);

impl ResidentTarget {
    /// Wrap this rail's name for one of its resident targets.
    pub fn new<T: RailTarget>(target: T) -> Self {
        Self(Arc::new(target))
    }

    /// This rail's own view of the target, or `None` when another rail issued
    /// it.
    ///
    /// `None` is unreachable while [`crate::backend::select`] latches one rail
    /// per process; it is an answer rather than a panic because every caller is
    /// on a path that already has a lawful "this resident is not mine to read"
    /// reply, and a caught panic at a `reims_vgpu_qemu_*` entry point is a dead
    /// device.
    pub fn get<T: RailTarget>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref::<T>()
    }
}

impl PartialEq for ResidentTarget {
    fn eq(&self, other: &Self) -> bool {
        self.0.same_target(&*other.0)
    }
}

impl Eq for ResidentTarget {}

impl fmt::Debug for ResidentTarget {
    /// The rail's own rendering, unwrapped: a fail line that named the wrapper
    /// would say `ResidentTarget(Surface { … })` and cost every reader the
    /// same word.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct OneRail(u32);
    impl RailTarget for OneRail {
        fn same_target(&self, other: &dyn RailTarget) -> bool {
            other.as_any().downcast_ref::<Self>() == Some(self)
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct OtherRail(u32);
    impl RailTarget for OtherRail {
        fn same_target(&self, other: &dyn RailTarget) -> bool {
            other.as_any().downcast_ref::<Self>() == Some(self)
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn a_target_compares_by_what_the_rail_named_and_not_by_the_handle() {
        let armed = ResidentTarget::new(OneRail(7));
        assert_eq!(armed, armed.clone(), "a clone names the same image");
        assert_eq!(
            armed,
            ResidentTarget::new(OneRail(7)),
            "two handles built from one identity name the same image"
        );
        assert_ne!(armed, ResidentTarget::new(OneRail(8)));
    }

    #[test]
    fn two_rails_never_name_the_same_image() {
        assert_ne!(
            ResidentTarget::new(OneRail(7)),
            ResidentTarget::new(OtherRail(7)),
            "the integers match and the rails do not"
        );
        assert!(
            ResidentTarget::new(OtherRail(7)).get::<OneRail>().is_none(),
            "a rail must not read back a target it did not issue"
        );
    }

    #[test]
    fn a_rail_reads_back_the_identity_it_issued() {
        let handle = ResidentTarget::new(OneRail(7));
        assert_eq!(handle.get::<OneRail>(), Some(&OneRail(7)));
    }

    #[test]
    fn the_handle_renders_as_the_rails_own_identity() {
        assert_eq!(
            format!("{:?}", ResidentTarget::new(OneRail(7))),
            "OneRail(7)"
        );
    }
}
