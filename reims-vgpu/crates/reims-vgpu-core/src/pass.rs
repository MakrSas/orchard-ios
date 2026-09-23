//! The render pass: ten attachment slots, and the memory a pass touches before
//! any draw does.
//!
//! # The largest footprint in the model comes from a descriptor, not a draw
//!
//! A draw names a pipeline and reads whatever is bound. What it *writes* is the
//! pass's attachments, and those are named once, in the pass descriptor, before
//! the first draw runs. So a model that derived render-target participation
//! from draws would have to reconstruct it from state; deriving it here means
//! it is read out of the record that declared it.
//!
//! # Load and store are accesses, and both directions matter
//!
//! An attachment whose load action preserves its prior contents is **read** by
//! the pass — the pass composites onto what is there — and an attachment the
//! pass stores is **written**. `DontCare` on load preserves as legitimately as
//! `Load` does: undefined contents permit any contents, including the ones
//! already present, and the guest relies on that reading when it redraws only a
//! damage rect. Only `Clear` refuses the prior contents.
//!
//! That is [`reims_vgpu_protocol::pass_action::LoadAction::preserves_prior_contents`],
//! and the reason it is not restated here: it is the same fact the executors
//! already answer with, and a second copy is how one arm ends up clearing what
//! the other preserves.
//!
//! # A resolve target is a second attachment, not a decoration
//!
//! A multisample resolve writes a *different* texture from the one it reads.
//! Both are participations, at different subresources, and a model that carried
//! only the attachment would order a later read of the resolve target against
//! nothing.
//!
//! # The pass extent is not the texture's
//!
//! `renderTargetWidth`/`Height` are the pass's own, and a guest may bind a large
//! texture and render into a corner of it. They are carried as declared,
//! because the difference is what an executor needs to set a framebuffer extent
//! that is not the attachment's size.

use crate::access::{AccessMode, ByteRange, Participation, ParticipationExtent, SubresourceRange};
use crate::identity::ResourceId;
pub use reims_vgpu_protocol::pass_action::LoadAction;
pub use reims_vgpu_protocol::pass_action::{
    DepthResolveFilter, StencilResolveFilter, StoreActionOptions,
};

/// Colour attachment slots the record always carries, written or not.
///
/// Eight, always: the record is a fixed shape and an unattached slot is one
/// whose texture ref is zero, not one that is absent.
///
/// Taken from the layout rather than written again. [`pass_descriptor`] walks
/// the wire body's colour array and indexes this descriptor's by the same
/// number, so two constants that could disagree are a panic on guest data at
/// the moment they do — and the wire's is the one that decides, because it is
/// the array that exists.
///
/// [`pass_descriptor`]: crate::resolve::pass_descriptor
pub const COLOR_ATTACHMENTS: usize = reims_vgpu_protocol::decode::RENDER_PASS_COLOR_ATTACHMENTS;

// `AttachmentSlot::Color` names a slot in a `u8`, and `pass_descriptor` casts
// the array index into one. Bounded here rather than checked there: the cast
// is lossless for every count this constant can hold, and a wire that grew
// past it should stop the build rather than renumber slot 256 as slot 0.
const _: () = assert!(COLOR_ATTACHMENTS <= u8::MAX as usize + 1);

/// Which of the pass's fixed slots an attachment is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttachmentSlot {
    Color(u8),
    Depth,
    Stencil,
}

impl AttachmentSlot {
    #[must_use]
    pub const fn is_color(self) -> bool {
        matches!(self, Self::Color(_))
    }
}

/// What the pass does with an attachment's result.
///
/// The four ordinals the wire form represents. `Unknown` and
/// `CustomSampleDepthStore` are Metal values this record shape carries no state
/// for; they parse to `None` rather than folding onto a neighbour, because a
/// store action guessed wrong is either a discarded frame or a resolve that
/// never happens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StoreAction {
    DontCare,
    Store,
    MultisampleResolve,
    StoreAndMultisampleResolve,
}

impl StoreAction {
    /// Parse the attachment prefix's ordinal.
    #[must_use]
    pub const fn parse(raw: u16) -> Option<StoreAction> {
        use reims_vgpu_protocol::pass_action as ordinals;
        Some(match raw {
            ordinals::MTL_STORE_ACTION_DONT_CARE => Self::DontCare,
            ordinals::MTL_STORE_ACTION_STORE => Self::Store,
            ordinals::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE => Self::MultisampleResolve,
            ordinals::MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE => {
                Self::StoreAndMultisampleResolve
            }
            _ => return None,
        })
    }

    /// Whether the pass's result reaches the attachment itself.
    ///
    /// `DontCare` answers false, and that is a claim about *publication* rather
    /// than about the pass writing: the pass may well write the attachment
    /// during rendering, and the contents afterwards are undefined. What is
    /// false is that a later reader may rely on them.
    #[must_use]
    pub const fn stores_attachment(self) -> bool {
        matches!(self, Self::Store | Self::StoreAndMultisampleResolve)
    }

    /// Whether the pass resolves into the slot's resolve target.
    #[must_use]
    pub const fn resolves(self) -> bool {
        matches!(
            self,
            Self::MultisampleResolve | Self::StoreAndMultisampleResolve
        )
    }
}

/// One attachment slot, after resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attachment {
    pub slot: AttachmentSlot,
    /// `None` when the slot is unattached.
    pub texture: Option<ResourceId>,
    pub level: u16,
    pub slice: u16,
    pub depth_plane: u16,
    /// `None` when the slot has no resolve target.
    pub resolve_texture: Option<ResourceId>,
    pub resolve_level: u16,
    pub resolve_slice: u16,
    pub resolve_depth_plane: u16,
    pub load: LoadAction,
    pub store: StoreAction,
    /// What the store action is asked to do beyond storing.
    ///
    /// The wire prefix has carried this word since the layout was measured and
    /// nothing here read it, which made a pass asking for a resolve at
    /// programmable sample positions indistinguishable from one asking for the
    /// ordinary resolve. It is on the attachment rather than on the pass
    /// because the guest sets it per slot — the capture drives colour, depth
    /// and stencil separately and gets three independent words.
    ///
    /// Carried, not decided. Whether a host can resolve at custom sample
    /// positions is an executor's question, and this crate names none.
    pub store_options: StoreActionOptions,
    /// The value this slot is cleared to, as the guest's bits.
    ///
    /// Four words, interpreted by [`Self::slot`] rather than by a tag: a colour
    /// slot's clear is four `double` components, a depth slot's is one, and a
    /// stencil slot's is a `uint32`. Storing the tag beside the slot would make
    /// "a colour slot holding a depth clear" representable, and the slot is
    /// already the answer — so the accessors read it and there is nothing for a
    /// second field to disagree with.
    ///
    /// Bits rather than floats for the reason [`crate::render::FloatBits`]
    /// gives: a descriptor is compared, and float equality makes a NaN clear
    /// colour differ from itself.
    ///
    /// It is carried at all because a `LoadAction::Clear` without it is not
    /// executable. The load action says the attachment is cleared and this says
    /// to what; a model that dropped it would turn every guest clear into a
    /// clear to whatever the executor happened to pick.
    pub clear_bits: [u64; 4],
}

impl Attachment {
    /// An unattached slot, which every pass carries eight of before the guest
    /// fills any in.
    #[must_use]
    pub const fn unattached(slot: AttachmentSlot) -> Attachment {
        Attachment {
            slot,
            texture: None,
            level: 0,
            slice: 0,
            depth_plane: 0,
            resolve_texture: None,
            resolve_level: 0,
            resolve_slice: 0,
            resolve_depth_plane: 0,
            load: LoadAction::DontCare,
            store: StoreAction::DontCare,
            store_options: StoreActionOptions::None,
            clear_bits: [0; 4],
        }
    }

    /// The colour this slot is cleared to, if it is a colour slot.
    #[must_use]
    pub fn clear_color(&self) -> Option<[f64; 4]> {
        self.slot.is_color().then(|| {
            let mut out = [0.0f64; 4];
            for (component, bits) in out.iter_mut().zip(self.clear_bits) {
                *component = f64::from_bits(bits);
            }
            out
        })
    }

    /// The depth this slot is cleared to, if it is the depth slot.
    #[must_use]
    pub fn clear_depth(&self) -> Option<f64> {
        matches!(self.slot, AttachmentSlot::Depth).then(|| f64::from_bits(self.clear_bits[0]))
    }

    /// The stencil value this slot is cleared to, if it is the stencil slot.
    #[must_use]
    pub fn clear_stencil(&self) -> Option<u32> {
        matches!(self.slot, AttachmentSlot::Stencil).then(|| self.clear_bits[0] as u32)
    }

    /// The subresource this attachment renders into.
    ///
    /// The depth plane is a slice of a 3D texture in this model's coordinates:
    /// a 3D attachment names one plane, and it is the same axis an array slice
    /// names on a 2D array. Which of the two a texture has is the texture's
    /// property, and only one of the fields is non-zero for a given one.
    #[must_use]
    pub const fn subresource(&self) -> SubresourceRange {
        SubresourceRange::one(
            self.slice as u32 + self.depth_plane as u32,
            self.level as u32,
        )
    }

    /// The subresource the resolve target receives.
    #[must_use]
    pub const fn resolve_subresource(&self) -> SubresourceRange {
        SubresourceRange::one(
            self.resolve_slice as u32 + self.resolve_depth_plane as u32,
            self.resolve_level as u32,
        )
    }

    /// How the pass accesses the attached texture, if it is attached.
    ///
    /// A pass that preserves prior contents reads; a pass that renders writes.
    /// Both together are read-write, which is the ordinary case for a
    /// composited frame.
    #[must_use]
    pub fn access_mode(&self) -> Option<AccessMode> {
        self.texture?;
        let reads = self.load.preserves_prior_contents();
        // The pass writes the attachment whenever it renders into it, which is
        // whenever it is attached. `DontCare` on store says a later reader may
        // not rely on the result — it does not say the memory was left alone,
        // and a hazard edge that assumed it was would let a concurrent reader
        // see a half-drawn frame.
        Some(if reads {
            AccessMode::ReadWrite
        } else {
            AccessMode::Write
        })
    }
}

/// The buffer a pass writes occlusion counts into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisibilityResultBuffer {
    pub buffer: ResourceId,
}

/// The pass's own target extent, which is not the attachment's size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderTargetExtent {
    pub width: u64,
    pub height: u64,
    pub array_length: u64,
}

/// A resolved render pass descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassDescriptor {
    pub color: [Attachment; COLOR_ATTACHMENTS],
    pub depth: Attachment,
    pub stencil: Attachment,
    /// The buffer `setVisibilityResultMode:offset:` indexes.
    ///
    /// It lives only here: that record carries the mode and the offset and
    /// names no buffer, so a model that took the buffer from the draw record
    /// would find none.
    pub visibility_result_buffer: Option<VisibilityResultBuffer>,
    /// How the depth slot reduces its samples into its resolve target.
    ///
    /// On the descriptor and not on [`Attachment`] because the record carries
    /// it on the depth and stencil bodies only — a colour slot has no such
    /// word — and because the two slots' filters are two ordinal spaces. A
    /// field on the shared attachment type would be a colour slot able to hold
    /// a depth filter, and a single filter type would make `1` mean `Min` and
    /// `DepthResolvedSample` at once.
    ///
    /// It was decoded by the wire layer and read by nothing until now, which
    /// made a pass asking to resolve depth at the furthest sample
    /// indistinguishable from one asking for sample zero. Those pick different
    /// depths out of the same samples, and the guest reads the result back as
    /// geometry — so the symptom is wrong occlusion later rather than a wrong
    /// frame now.
    pub depth_resolve_filter: DepthResolveFilter,
    /// How the stencil slot reduces its samples into its resolve target.
    ///
    /// [`StencilResolveFilter::DepthResolvedSample`] takes the stencil of
    /// whichever sample the *depth* filter chose, so this word is also where
    /// the dependency between the two resolves is stated.
    pub stencil_resolve_filter: StencilResolveFilter,
    pub extent: RenderTargetExtent,
}

impl PassDescriptor {
    /// A pass with nothing attached.
    #[must_use]
    pub fn empty() -> PassDescriptor {
        let mut color = [Attachment::unattached(AttachmentSlot::Color(0)); COLOR_ATTACHMENTS];
        for (index, slot) in color.iter_mut().enumerate() {
            slot.slot = AttachmentSlot::Color(index as u8);
        }
        PassDescriptor {
            color,
            depth: Attachment::unattached(AttachmentSlot::Depth),
            stencil: Attachment::unattached(AttachmentSlot::Stencil),
            visibility_result_buffer: None,
            depth_resolve_filter: DepthResolveFilter::Sample0,
            stencil_resolve_filter: StencilResolveFilter::Sample0,
            extent: RenderTargetExtent::default(),
        }
    }

    /// Every slot, colour first, in the order the record carries them.
    pub fn attachments(&self) -> impl Iterator<Item = &Attachment> {
        self.color
            .iter()
            .chain(core::iter::once(&self.depth))
            .chain(core::iter::once(&self.stencil))
    }

    /// Every slot the guest attached something to.
    pub fn attached(&self) -> impl Iterator<Item = &Attachment> {
        self.attachments().filter(|a| a.texture.is_some())
    }

    /// The memory this pass touches by virtue of its descriptor alone.
    ///
    /// Two per resolving attachment — the attachment and its resolve target —
    /// one per ordinary attachment, and one for the visibility buffer. Nothing
    /// here comes from a draw: this is what the pass costs before any draw is
    /// recorded, which is what makes a pass with no draws still a write.
    #[must_use]
    pub fn participations(&self) -> Vec<Participation> {
        let mut out = Vec::with_capacity(COLOR_ATTACHMENTS + 4);
        self.extend_participations(&mut out);
        out
    }

    /// [`Self::participations`], appended to a buffer the caller owns.
    ///
    /// The shape [`crate::exec::ResolvedOperation::participations`] needs: a
    /// pass descriptor is reached once per `writeDescriptor` record, and a
    /// fresh `Vec` for each would be an allocation per pass. The owning method
    /// above stays for the readings that want a value.
    pub fn extend_participations(&self, out: &mut Vec<Participation>) {
        for attachment in self.attached() {
            let (Some(texture), Some(mode)) = (attachment.texture, attachment.access_mode()) else {
                continue;
            };
            out.push(Participation {
                resource: texture,
                extent: ParticipationExtent::Subresource(attachment.subresource()),
                mode,
                api_stages: NO_STAGES,
            });
            if let (true, Some(resolve)) = (attachment.store.resolves(), attachment.resolve_texture)
            {
                out.push(Participation {
                    resource: resolve,
                    extent: ParticipationExtent::Subresource(attachment.resolve_subresource()),
                    mode: AccessMode::Write,
                    api_stages: NO_STAGES,
                });
            }
        }
        if let Some(visibility) = self.visibility_result_buffer {
            // The offsets the draws write at are theirs, not the descriptor's,
            // so the descriptor's own claim is the whole buffer. A narrower one
            // would need every draw's visibility offset, which is a later
            // question and a different record.
            out.push(Participation {
                resource: visibility.buffer,
                extent: ParticipationExtent::Whole,
                mode: AccessMode::Write,
                api_stages: NO_STAGES,
            });
        }
    }

    /// The byte range one occlusion query occupies, at `offset`.
    ///
    /// `setVisibilityResultMode:offset:` names an offset into the buffer above
    /// and the count is a `uint64`, so a draw's exact contribution is eight
    /// bytes there. Stated here, beside the buffer, because the two records
    /// that make up an occlusion query are on opposite sides of the pass.
    #[must_use]
    pub const fn visibility_result_range(offset: u64) -> ByteRange {
        ByteRange {
            offset,
            length: VISIBILITY_RESULT_BYTES,
        }
    }
}

/// The bytes one occlusion count occupies.
pub const VISIBILITY_RESULT_BYTES: u64 = 8;

/// A pass declares no shader stage; the stages come from the pipeline.
const NO_STAGES: u32 = 0;

#[cfg(test)]
mod tests {
    /// The claim `COLOR_ATTACHMENTS` is derived for: an empty descriptor has
    /// exactly one slot per colour attachment the wire body carries, each
    /// named by its own index.
    ///
    /// `crate::resolve::pass_descriptor` walks the wire array and writes
    /// `descriptor.color[index]`, so a descriptor with fewer slots than the
    /// body has is an index out of range on a guest render pass --- a panic on
    /// guest data, which is the one failure this model may not have.
    #[test]
    fn a_descriptor_has_one_colour_slot_per_slot_the_wire_carries() {
        let descriptor = PassDescriptor::empty();
        assert_eq!(
            descriptor.color.len(),
            reims_vgpu_protocol::decode::RENDER_PASS_COLOR_ATTACHMENTS
        );
        // Not the same statement twice: the line above compares the descriptor
        // to the wire's count, and this one proves that count really is the
        // length of the array `pass_descriptor` walks.
        assert_eq!(
            core::mem::size_of_val(&descriptor.color) / core::mem::size_of::<Attachment>(),
            core::mem::size_of::<
                [reims_vgpu_protocol::decode::ColorAttachmentBody;
                    reims_vgpu_protocol::decode::RENDER_PASS_COLOR_ATTACHMENTS],
            >() / core::mem::size_of::<reims_vgpu_protocol::decode::ColorAttachmentBody>()
        );
        for (index, attachment) in descriptor.color.iter().enumerate() {
            assert_eq!(
                attachment.slot,
                AttachmentSlot::Color(u8::try_from(index).expect("bounded by the const assertion"))
            );
        }
    }

    /// A clear value is read through the slot that owns it, and every other
    /// reading is `None`. There is no tag to disagree with the slot, which is
    /// what makes "a colour slot holding a depth clear" unrepresentable rather
    /// than merely unlikely.
    #[test]
    fn a_clear_value_is_read_through_the_slot_that_owns_it() {
        let mut colour = Attachment::unattached(AttachmentSlot::Color(3));
        colour.clear_bits = [0.25f64, 0.5, 0.75, 1.0].map(f64::to_bits);
        assert_eq!(colour.clear_color(), Some([0.25, 0.5, 0.75, 1.0]));
        assert_eq!(colour.clear_depth(), None);
        assert_eq!(colour.clear_stencil(), None);

        let mut depth = Attachment::unattached(AttachmentSlot::Depth);
        depth.clear_bits[0] = 1.0f64.to_bits();
        assert_eq!(depth.clear_depth(), Some(1.0));
        assert_eq!(depth.clear_color(), None);

        let mut stencil = Attachment::unattached(AttachmentSlot::Stencil);
        stencil.clear_bits[0] = 0x1234;
        assert_eq!(stencil.clear_stencil(), Some(0x1234));
        assert_eq!(stencil.clear_depth(), None);
    }

    /// A NaN clear colour compares equal to itself, which is what a descriptor
    /// held in a state table needs and what `f64` equality would not give.
    #[test]
    fn a_nan_clear_colour_compares_equal_to_itself() {
        let mut a = Attachment::unattached(AttachmentSlot::Color(0));
        a.clear_bits = [f64::NAN.to_bits(); 4];
        let b = a;
        assert_eq!(a, b);
        assert!(a.clear_color().expect("colour")[0].is_nan());
    }
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn attached(slot: AttachmentSlot, load: LoadAction, store: StoreAction) -> Attachment {
        Attachment {
            texture: Some(res(1)),
            load,
            store,
            ..Attachment::unattached(slot)
        }
    }

    /// A pass with nothing attached touches nothing.
    #[test]
    fn an_empty_pass_participates_in_nothing() {
        assert!(PassDescriptor::empty().participations().is_empty());
        assert_eq!(PassDescriptor::empty().attached().count(), 0);
        assert_eq!(PassDescriptor::empty().attachments().count(), 10);
    }

    /// The eight colour slots know which of the eight they are, so a
    /// participation can be attributed to a slot rather than to a position in
    /// an array.
    #[test]
    fn the_colour_slots_are_numbered() {
        let pass = PassDescriptor::empty();
        for (index, attachment) in pass.color.iter().enumerate() {
            assert_eq!(attachment.slot, AttachmentSlot::Color(index as u8));
            assert!(attachment.slot.is_color());
        }
        assert!(!pass.depth.slot.is_color());
    }

    /// `DontCare` preserves as legitimately as `Load` does, so both read. Only
    /// `Clear` refuses the prior contents.
    #[test]
    fn only_clear_refuses_the_prior_contents() {
        for load in [LoadAction::Load, LoadAction::DontCare] {
            assert_eq!(
                attached(AttachmentSlot::Color(0), load, StoreAction::Store).access_mode(),
                Some(AccessMode::ReadWrite),
                "{load:?}"
            );
        }
        assert_eq!(
            attached(
                AttachmentSlot::Color(0),
                LoadAction::Clear,
                StoreAction::Store
            )
            .access_mode(),
            Some(AccessMode::Write)
        );
    }

    /// An attachment the pass will not publish is still written during the
    /// pass. A hazard edge that assumed otherwise would let a concurrent reader
    /// see a half-drawn frame.
    #[test]
    fn a_dont_care_store_is_still_a_write() {
        let attachment = attached(
            AttachmentSlot::Color(0),
            LoadAction::Clear,
            StoreAction::DontCare,
        );
        assert_eq!(attachment.access_mode(), Some(AccessMode::Write));
        assert!(!attachment.store.stores_attachment());
    }

    /// An unattached slot participates in nothing at all.
    #[test]
    fn an_unattached_slot_has_no_access() {
        assert_eq!(
            Attachment::unattached(AttachmentSlot::Depth).access_mode(),
            None
        );
    }

    /// A resolve writes a second texture, and the model says so.
    #[test]
    fn a_resolve_target_is_its_own_participation() {
        let mut pass = PassDescriptor::empty();
        pass.color[0] = Attachment {
            texture: Some(res(1)),
            resolve_texture: Some(res(2)),
            resolve_level: 3,
            load: LoadAction::Clear,
            store: StoreAction::MultisampleResolve,
            ..Attachment::unattached(AttachmentSlot::Color(0))
        };
        let parts = pass.participations();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].resource, res(1));
        assert_eq!(parts[0].mode, AccessMode::Write);
        assert_eq!(parts[1].resource, res(2));
        assert_eq!(parts[1].mode, AccessMode::Write);
        assert_eq!(
            parts[1].extent,
            ParticipationExtent::Subresource(SubresourceRange {
                base_level: 3,
                level_count: 1,
                base_slice: 0,
                slice_count: 1,
                plane: 0,
            })
        );
    }

    /// A resolve target attached to a slot that does not resolve is not
    /// written. The guest may leave the field set from a previous
    /// configuration, and writing it would be a write the pass never made.
    #[test]
    fn a_resolve_target_a_store_action_does_not_use_is_not_written() {
        let mut pass = PassDescriptor::empty();
        pass.color[0] = Attachment {
            texture: Some(res(1)),
            resolve_texture: Some(res(2)),
            load: LoadAction::Clear,
            store: StoreAction::Store,
            ..Attachment::unattached(AttachmentSlot::Color(0))
        };
        let parts = pass.participations();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].resource, res(1));
    }

    /// The store ordinals that this record shape carries a resolve for.
    #[test]
    fn exactly_two_store_actions_resolve_and_two_publish() {
        assert_eq!(
            [
                StoreAction::DontCare,
                StoreAction::Store,
                StoreAction::MultisampleResolve,
                StoreAction::StoreAndMultisampleResolve,
            ]
            .iter()
            .filter(|a| a.resolves())
            .count(),
            2
        );
        assert!(StoreAction::Store.stores_attachment());
        assert!(StoreAction::StoreAndMultisampleResolve.stores_attachment());
        assert!(!StoreAction::MultisampleResolve.stores_attachment());
    }

    /// An ordinal this record shape carries no state for is refused, not
    /// rounded to a neighbour.
    #[test]
    fn an_unrepresented_store_ordinal_parses_to_nothing() {
        for raw in 0u16..=3 {
            assert!(StoreAction::parse(raw).is_some(), "{raw}");
        }
        for raw in [4u16, 5, 64, u16::MAX] {
            assert_eq!(StoreAction::parse(raw), None, "{raw}");
        }
    }

    /// The visibility buffer lives on the descriptor and nowhere else, and its
    /// per-draw range is eight bytes.
    #[test]
    fn the_visibility_buffer_is_the_passes_and_a_query_is_eight_bytes() {
        let mut pass = PassDescriptor::empty();
        pass.visibility_result_buffer = Some(VisibilityResultBuffer { buffer: res(7) });
        let parts = pass.participations();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].resource, res(7));
        assert_eq!(parts[0].extent, ParticipationExtent::Whole);
        assert_eq!(parts[0].mode, AccessMode::Write);
        assert_eq!(
            PassDescriptor::visibility_result_range(0x40),
            ByteRange {
                offset: 0x40,
                length: 8
            }
        );
    }

    /// Two passes rendering into different levels of one texture do not
    /// conflict; two into the same level do.
    #[test]
    fn attachment_subresources_separate_levels() {
        let mut a = attached(
            AttachmentSlot::Color(0),
            LoadAction::Clear,
            StoreAction::Store,
        );
        let mut b = a;
        a.level = 0;
        b.level = 1;
        assert!(!a.subresource().overlaps(b.subresource()));
        b.level = 0;
        assert!(a.subresource().overlaps(b.subresource()));
    }
}
