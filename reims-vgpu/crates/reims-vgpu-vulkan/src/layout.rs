//! Image layouts and queue-family ownership, as explicitly planned transitions.
//!
//! # Why a tracker and not a guess at each use
//!
//! A `VkImage` is in exactly one layout per subresource, and using it in the
//! wrong one is undefined behavior the validation layers catch and the driver
//! sometimes does not. The only way to know the layout is to have tracked every
//! transition, so this tracks them — and the transition it produces names both
//! ends, because a barrier that guessed `oldLayout` would either be wrong or
//! would have to use `VK_IMAGE_LAYOUT_UNDEFINED` and throw the contents away.
//!
//! # Discarding contents is a decision, never an inference
//!
//! A transition out of `UNDEFINED` lets a driver skip decompressing or
//! resolving whatever was there, which is a real saving on a full-surface
//! overwrite. It also destroys the contents. So [`Contents`] is a parameter and
//! not something derived from the use: the caller knows whether the pass loads
//! or clears, and this cannot. An implementation that inferred it would be
//! choosing, silently, to lose bytes the guest wrote.
//!
//! # No transition is the steady state, and it is reported as such
//!
//! [`LayoutTracker::plan`] returns `None` when the subresource is already in
//! the layout the use needs. That is the ordinary case for a draw in a
//! steady-state frame and the reason the tracker earns its keep: the alternative
//! is a barrier per use, every frame, for a layout that never changed.
//!
//! `None` is *not* "no barrier is needed". A read after a write in one layout
//! needs an execution and memory dependency and no layout change, and that
//! dependency is [`crate::barrier`]'s and the dependency compiler's. This module
//! answers layout, and layout only.
//!
//! # Ownership is two halves and both are recorded
//!
//! Moving an image between queue families is a release on the old family and an
//! acquire on the new one, and skipping either makes the contents undefined.
//! [`LayoutTracker::plan_family`] returns both halves together so a caller
//! cannot record one; whether the image is `CONCURRENT` instead is the
//! creation-time decision above this module, and an image declared that way is
//! simply never given a family here.

use ash::vk;
use std::collections::HashMap;

/// A native image this rail owns.
///
/// Minted by the rail rather than derived from a semantic identity: one backing
/// may host several images, and an image may outlive the resource that named it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageId(pub u64);

/// One subresource of one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Subresource {
    pub level: u32,
    pub layer: u32,
    /// Multi-planar formats lay each plane out separately, so a plane is a
    /// separate subresource and not a coordinate inside one.
    pub plane: u32,
}

impl Subresource {
    #[must_use]
    pub const fn new(level: u32, layer: u32) -> Self {
        Self {
            level,
            layer,
            plane: 0,
        }
    }
}

/// What the image is about to be used for.
///
/// The set is closed on purpose: a use with no entry is a use whose layout this
/// rail has not decided, and adding a variant is where that decision gets made
/// rather than at a call site picking a `VkImageLayout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    ColorAttachment,
    DepthStencilAttachment,
    /// Read-only depth or stencil, for a pass that tests without writing.
    DepthStencilRead,
    SampledRead,
    /// A storage image. `GENERAL` because a storage image is written through a
    /// descriptor that has no other optimal layout.
    Storage,
    TransferSrc,
    TransferDst,
    Present,
}

impl Use {
    #[must_use]
    pub const fn layout(self) -> vk::ImageLayout {
        match self {
            Self::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Self::DepthStencilAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            Self::DepthStencilRead => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            Self::SampledRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            Self::Storage => vk::ImageLayout::GENERAL,
            Self::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            Self::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            Self::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    /// The stages that touch the image in this use.
    #[must_use]
    pub const fn stages(self) -> vk::PipelineStageFlags2 {
        match self {
            Self::ColorAttachment => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            // Both, because a depth or stencil access happens in either. Which
            // one is the pipeline's business --- a fragment shader that
            // discards or writes `gl_FragDepth` forces the test late --- and a
            // source mask naming only the early stage does not wait for the
            // write the late stage performed. Sampling the depth buffer that
            // was just rendered would then race it.
            Self::DepthStencilAttachment | Self::DepthStencilRead => {
                vk::PipelineStageFlags2::from_raw(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
                )
            }
            // Any shader stage may sample or store, and which ones is the
            // pipeline's business rather than the layout's. `ALL_COMMANDS` here
            // would be the lazy answer; `ALL_GRAPHICS` plus compute is the
            // honest one for a use that says nothing about the stage.
            Self::SampledRead | Self::Storage => vk::PipelineStageFlags2::from_raw(
                vk::PipelineStageFlags2::ALL_GRAPHICS.as_raw()
                    | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
            ),
            Self::TransferSrc | Self::TransferDst => vk::PipelineStageFlags2::ALL_TRANSFER,
            // Presentation is not a pipeline stage. The image is handed to the
            // presentation engine, and the dependency is a semaphore rather
            // than a stage mask.
            Self::Present => vk::PipelineStageFlags2::NONE,
        }
    }

    /// The access this use performs.
    #[must_use]
    pub const fn access(self) -> vk::AccessFlags2 {
        match self {
            Self::ColorAttachment => vk::AccessFlags2::from_raw(
                vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
                    | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw(),
            ),
            Self::DepthStencilAttachment => vk::AccessFlags2::from_raw(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
            ),
            Self::DepthStencilRead => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            Self::SampledRead => vk::AccessFlags2::SHADER_SAMPLED_READ,
            Self::Storage => vk::AccessFlags2::from_raw(
                vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
            ),
            Self::TransferSrc => vk::AccessFlags2::TRANSFER_READ,
            Self::TransferDst => vk::AccessFlags2::TRANSFER_WRITE,
            Self::Present => vk::AccessFlags2::NONE,
        }
    }
}

/// Whether the contents survive the transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contents {
    /// The transition names the current layout, and the driver preserves what
    /// is there.
    Keep,
    /// The transition comes out of `UNDEFINED`. Cheaper, and the previous
    /// contents are gone. Only for a use that overwrites or clears every byte
    /// the caller cares about.
    Discard,
}

/// One planned layout transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a transition nobody records leaves the image in a layout the next use will not expect"]
pub struct Transition {
    pub image: ImageId,
    pub subresource: Subresource,
    pub from: vk::ImageLayout,
    pub to: vk::ImageLayout,
    pub src_stages: vk::PipelineStageFlags2,
    pub dst_stages: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
    /// True when `from` is `UNDEFINED` because the caller said the contents
    /// were not needed. Carried so a census can price how much of a frame's
    /// bandwidth the discard path is saving.
    pub discarded_contents: bool,
}

/// Half of a queue-family ownership move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "recording one half of an ownership transfer makes the contents undefined"]
pub struct OwnershipTransfer {
    pub image: ImageId,
    pub subresource: Subresource,
    pub from_family: u32,
    pub to_family: u32,
    /// The layout on both sides. An ownership transfer may change layout too,
    /// but the two halves must agree, so it is one value rather than two.
    pub layout: vk::ImageLayout,
    /// What the releasing queue has to make available: the stages and accesses
    /// of the last use recorded on the family giving the image up.
    ///
    /// **A transfer is a memory dependency and not only a change of owner.**
    /// Vulkan ignores a release's destination masks and an acquire's source
    /// masks, which leaves exactly these two halves carrying the availability
    /// and visibility operations. Carried on the plan rather than left to the
    /// recorder because only the tracker knows what the previous owner did:
    /// with both halves empty the source queue's writes are never made
    /// available and nothing becomes visible to the destination, so the image
    /// holds whatever that queue happens to see — a race with no failing call
    /// in it.
    pub src_stages: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    /// What the acquiring queue needs visible: the use it is taking the image
    /// for.
    pub dst_stages: vk::PipelineStageFlags2,
    pub dst_access: vk::AccessFlags2,
}

/// Why a plan could not be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// The image was never declared, so its layout is not known and cannot be
    /// assumed. Assuming `UNDEFINED` would discard the contents of an image
    /// somebody else is tracking.
    UnknownImage { image: ImageId },
    /// The subresource is outside what the image was declared with.
    UnknownSubresource {
        image: ImageId,
        subresource: Subresource,
    },
    /// An ownership transfer was asked for on an image with no recorded family.
    /// Either it is `CONCURRENT` and needs none, or its first owner was never
    /// recorded — and guessing which would either emit a transfer that is
    /// invalid usage or skip one that is required.
    NoRecordedFamily { image: ImageId },
}

impl Decline {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownImage { .. } => "vk_layout_unknown_image",
            Self::UnknownSubresource { .. } => "vk_layout_unknown_subresource",
            Self::NoRecordedFamily { .. } => "vk_layout_no_recorded_family",
        }
    }
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::UnknownImage { image } | Self::NoRecordedFamily { image } => {
                write!(f, "{} image={}", self.slug(), image.0)
            }
            Self::UnknownSubresource { image, subresource } => write!(
                f,
                "{} image={} level={} layer={} plane={}",
                self.slug(),
                image.0,
                subresource.level,
                subresource.layer,
                subresource.plane
            ),
        }
    }
}

/// What one subresource is doing.
#[derive(Clone, Copy, Debug)]
struct State {
    layout: vk::ImageLayout,
    /// The queue family that owns it, when the image is `EXCLUSIVE`.
    family: Option<u32>,
    /// The stages and access of the last use, which are the source half of the
    /// next transition's dependency.
    last_stages: vk::PipelineStageFlags2,
    last_access: vk::AccessFlags2,
}

#[derive(Clone, Debug)]
struct Image {
    levels: u32,
    layers: u32,
    planes: u32,
    subresources: HashMap<Subresource, State>,
}

/// What the tracker has been asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Uses that needed no layout change. The steady state, and the number that
    /// says what the tracker is saving over a barrier per use.
    pub already_in_layout: usize,
    pub transitions: usize,
    /// Transitions the caller declared discardable.
    pub discards: usize,
    pub ownership_transfers: usize,
}

/// Every image's per-subresource layout and family.
#[derive(Debug, Default)]
pub struct LayoutTracker {
    images: HashMap<ImageId, Image>,
    census: Census,
}

impl LayoutTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Declare a freshly created image.
    ///
    /// Every subresource starts `UNDEFINED`, which is what `vkCreateImage`
    /// leaves them in. `family` is the queue family an `EXCLUSIVE` image is
    /// first owned by, or `None` for a `CONCURRENT` one.
    ///
    /// # Panics
    ///
    /// If any dimension is zero. An image with no levels, layers or planes is
    /// not a shallow image; it is one nothing can name a subresource of.
    pub fn declare(
        &mut self,
        image: ImageId,
        levels: u32,
        layers: u32,
        planes: u32,
        family: Option<u32>,
    ) {
        assert!(
            levels > 0 && layers > 0 && planes > 0,
            "an image with no subresources cannot be used"
        );
        let mut subresources = HashMap::new();
        for level in 0..levels {
            for layer in 0..layers {
                for plane in 0..planes {
                    subresources.insert(
                        Subresource {
                            level,
                            layer,
                            plane,
                        },
                        State {
                            layout: vk::ImageLayout::UNDEFINED,
                            family,
                            last_stages: vk::PipelineStageFlags2::NONE,
                            last_access: vk::AccessFlags2::NONE,
                        },
                    );
                }
            }
        }
        self.images.insert(
            image,
            Image {
                levels,
                layers,
                planes,
                subresources,
            },
        );
    }

    /// Forget an image whose native object is gone.
    pub fn forget(&mut self, image: ImageId) {
        self.images.remove(&image);
    }

    /// The layout a subresource is in.
    ///
    /// # Errors
    ///
    /// If the image or the subresource was never declared.
    pub fn layout(
        &self,
        image: ImageId,
        subresource: Subresource,
    ) -> Result<vk::ImageLayout, Decline> {
        Ok(self.state(image, subresource)?.layout)
    }

    /// The queue family that owns a subresource, if the image is `EXCLUSIVE`.
    ///
    /// # Errors
    ///
    /// If the image or the subresource was never declared.
    pub fn family(&self, image: ImageId, subresource: Subresource) -> Result<Option<u32>, Decline> {
        Ok(self.state(image, subresource)?.family)
    }

    /// Plan a use, and record it.
    ///
    /// `Ok(None)` means the subresource is already in the layout the use needs
    /// — the steady state. It does **not** mean no barrier is owed: a hazard in
    /// one layout is still a hazard, and that dependency belongs to
    /// [`crate::barrier`] and the dependency compiler.
    ///
    /// The use is recorded either way, so the next transition's source half is
    /// this use's stages and access rather than a guess.
    ///
    /// # Errors
    ///
    /// If the image or the subresource was never declared.
    pub fn plan(
        &mut self,
        image: ImageId,
        subresource: Subresource,
        use_: Use,
        contents: Contents,
    ) -> Result<Option<Transition>, Decline> {
        let current = *self.state(image, subresource)?;
        let target = use_.layout();
        let discarding = contents == Contents::Discard;
        let transition = if current.layout == target && !discarding {
            self.census.already_in_layout += 1;
            None
        } else {
            self.census.transitions += 1;
            if discarding {
                self.census.discards += 1;
            }
            Some(Transition {
                image,
                subresource,
                from: if discarding {
                    vk::ImageLayout::UNDEFINED
                } else {
                    current.layout
                },
                to: target,
                // Nothing has touched a fresh subresource, so an empty source
                // half is correct there and only there — and a discard is the
                // same statement about the bytes.
                src_stages: if discarding {
                    vk::PipelineStageFlags2::NONE
                } else {
                    current.last_stages
                },
                src_access: if discarding {
                    vk::AccessFlags2::NONE
                } else {
                    current.last_access
                },
                dst_stages: use_.stages(),
                dst_access: use_.access(),
                discarded_contents: discarding,
            })
        };
        let state = self
            .images
            .get_mut(&image)
            .and_then(|i| i.subresources.get_mut(&subresource))
            .expect("just resolved");
        state.layout = target;
        state.last_stages = use_.stages();
        state.last_access = use_.access();
        Ok(transition)
    }

    /// Plan the two halves of a queue-family ownership move, and record it.
    ///
    /// `Ok(None)` when the family already owns it. The two halves come back
    /// together because recording one of them makes the contents undefined.
    ///
    /// `use_` is what the acquiring family is taking the image for. It is
    /// required, not optional: it is the visibility half of the transfer's
    /// memory dependency, and a transfer planned without it hands the new
    /// owner an image whose contents were never made visible to it. See
    /// [`OwnershipTransfer::src_stages`].
    ///
    /// # Errors
    ///
    /// If the image or subresource was never declared, or the image has no
    /// recorded family — see [`Decline::NoRecordedFamily`].
    pub fn plan_family(
        &mut self,
        image: ImageId,
        subresource: Subresource,
        to_family: u32,
        use_: Use,
    ) -> Result<Option<OwnershipTransfer>, Decline> {
        let current = *self.state(image, subresource)?;
        let Some(from_family) = current.family else {
            return Err(Decline::NoRecordedFamily { image });
        };
        if from_family == to_family {
            return Ok(None);
        }
        self.census.ownership_transfers += 1;
        let state = self
            .images
            .get_mut(&image)
            .and_then(|i| i.subresources.get_mut(&subresource))
            .expect("just resolved");
        state.family = Some(to_family);
        Ok(Some(OwnershipTransfer {
            image,
            subresource,
            from_family,
            to_family,
            layout: current.layout,
            // The releasing family's last recorded use is what has to be made
            // available; the tracker is the only thing that knows it.
            src_stages: current.last_stages,
            src_access: current.last_access,
            dst_stages: use_.stages(),
            dst_access: use_.access(),
        }))
    }

    /// The extent check below is redundant with the lookup that follows it:
    /// `declare` inserts exactly the in-range keys, so a missing key and an
    /// out-of-range coordinate are one fact. It is kept because it is the half
    /// a reader can see, and because a sparse `declare` would make them differ.
    fn state(&self, image: ImageId, subresource: Subresource) -> Result<&State, Decline> {
        let img = self
            .images
            .get(&image)
            .ok_or(Decline::UnknownImage { image })?;
        if subresource.level >= img.levels
            || subresource.layer >= img.layers
            || subresource.plane >= img.planes
        {
            return Err(Decline::UnknownSubresource { image, subresource });
        }
        img.subresources
            .get(&subresource)
            .ok_or(Decline::UnknownSubresource { image, subresource })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: ImageId = ImageId(1);

    fn tracker() -> LayoutTracker {
        let mut t = LayoutTracker::new();
        t.declare(IMG, 3, 2, 1, Some(0));
        t
    }

    /// A fresh image is `UNDEFINED`, which is what `vkCreateImage` leaves it in
    /// — not the layout of its first use.
    #[test]
    fn a_fresh_subresource_is_undefined_and_its_first_use_transitions_from_there() {
        let mut t = tracker();
        assert_eq!(
            t.layout(IMG, Subresource::new(0, 0)),
            Ok(vk::ImageLayout::UNDEFINED)
        );
        let transition = t
            .plan(
                IMG,
                Subresource::new(0, 0),
                Use::ColorAttachment,
                Contents::Keep,
            )
            .expect("declared")
            .expect("a transition is needed");
        assert_eq!(transition.from, vk::ImageLayout::UNDEFINED);
        assert_eq!(transition.to, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(
            transition.src_stages,
            vk::PipelineStageFlags2::NONE,
            "nothing has touched it, so there is nothing to wait for"
        );
        assert!(!transition.discarded_contents, "the caller did not ask to");
    }

    /// The steady state, and the reason the tracker earns its keep.
    #[test]
    fn a_use_in_the_layout_it_is_already_in_needs_no_transition() {
        let mut t = tracker();
        let sub = Subresource::new(0, 0);
        assert!(t
            .plan(IMG, sub, Use::SampledRead, Contents::Keep)
            .expect("declared")
            .is_some_and(|t| t.to == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        for _ in 0..10 {
            assert_eq!(t.plan(IMG, sub, Use::SampledRead, Contents::Keep), Ok(None));
        }
        assert_eq!(t.census().transitions, 1);
        assert_eq!(t.census().already_in_layout, 10);
    }

    /// The next transition's source half is the last use, not a guess and not
    /// `ALL_COMMANDS`.
    #[test]
    fn a_transition_waits_for_the_use_that_came_before_it() {
        let mut t = tracker();
        let sub = Subresource::new(1, 0);
        t.plan(IMG, sub, Use::ColorAttachment, Contents::Keep)
            .expect("declared");
        let transition = t
            .plan(IMG, sub, Use::SampledRead, Contents::Keep)
            .expect("declared")
            .expect("colour attachment to shader read is a transition");
        assert_eq!(transition.from, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(transition.to, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(
            transition.src_stages,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
        );
        assert!(transition
            .src_access
            .contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE));
        assert!(transition
            .dst_access
            .contains(vk::AccessFlags2::SHADER_SAMPLED_READ));
    }

    /// A depth or stencil access happens in the early *and* the late fragment
    /// test stages, and which one a given draw used is a property of its
    /// pipeline rather than of the layout. So a transition sourced at a depth
    /// use has to wait for both: a fragment shader that discards or writes
    /// `gl_FragDepth` forces the test late, and a source mask naming only the
    /// early stage lets the sample below read the depth buffer while the late
    /// stage is still writing it.
    ///
    /// This is the same claim `barrier::ACCESS_STAGES` makes for the two
    /// depth-stencil accesses, and the reason `barrier::stages` turns one Metal
    /// fragment stage into four Vulkan ones.
    #[test]
    fn a_depth_use_names_both_fragment_test_stages() {
        const BOTH: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
        );
        assert_eq!(Use::DepthStencilAttachment.stages(), BOTH);
        assert_eq!(Use::DepthStencilRead.stages(), BOTH);

        let mut t = tracker();
        let sub = Subresource::new(1, 0);
        t.plan(IMG, sub, Use::DepthStencilAttachment, Contents::Keep)
            .expect("declared");
        let transition = t
            .plan(IMG, sub, Use::SampledRead, Contents::Keep)
            .expect("declared")
            .expect("depth attachment to shader read is a transition");
        assert_eq!(transition.src_stages, BOTH);
        assert!(transition
            .src_access
            .contains(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE));
    }

    /// Discarding is the caller's decision. The saving is real and so is the
    /// loss, which is why nothing here infers it from the use.
    #[test]
    fn a_discard_comes_out_of_undefined_and_waits_for_nothing() {
        let mut t = tracker();
        let sub = Subresource::new(0, 1);
        t.plan(IMG, sub, Use::SampledRead, Contents::Keep)
            .expect("declared");
        let transition = t
            .plan(IMG, sub, Use::ColorAttachment, Contents::Discard)
            .expect("declared")
            .expect("a discard is always a transition");
        assert_eq!(
            transition.from,
            vk::ImageLayout::UNDEFINED,
            "the previous contents are gone"
        );
        assert_eq!(transition.src_stages, vk::PipelineStageFlags2::NONE);
        assert_eq!(transition.src_access, vk::AccessFlags2::NONE);
        assert!(transition.discarded_contents);
        assert_eq!(t.census().discards, 1);
    }

    /// A discard into the layout the image is already in is still a transition:
    /// the point of it is telling the driver the contents are not needed, and
    /// "already in the right layout" would skip exactly that.
    #[test]
    fn a_discard_is_a_transition_even_in_the_same_layout() {
        let mut t = tracker();
        let sub = Subresource::new(2, 1);
        t.plan(IMG, sub, Use::ColorAttachment, Contents::Keep)
            .expect("declared");
        let again = t
            .plan(IMG, sub, Use::ColorAttachment, Contents::Discard)
            .expect("declared")
            .expect("the discard is the whole request");
        assert_eq!(again.from, vk::ImageLayout::UNDEFINED);
        assert_eq!(again.to, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    /// Subresources are tracked apart, because they are in separate layouts.
    #[test]
    fn subresources_have_their_own_layouts() {
        let mut t = tracker();
        t.plan(
            IMG,
            Subresource::new(0, 0),
            Use::TransferDst,
            Contents::Discard,
        )
        .expect("declared");
        assert_eq!(
            t.layout(IMG, Subresource::new(0, 0)),
            Ok(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        );
        assert_eq!(
            t.layout(IMG, Subresource::new(1, 0)),
            Ok(vk::ImageLayout::UNDEFINED),
            "a sibling level was not moved"
        );
        assert_eq!(
            t.layout(IMG, Subresource::new(0, 1)),
            Ok(vk::ImageLayout::UNDEFINED),
            "and neither was a sibling layer"
        );
    }

    /// Assuming `UNDEFINED` for an image nobody declared would discard the
    /// contents of an image somebody else is tracking.
    #[test]
    fn an_undeclared_image_or_subresource_is_refused() {
        let mut t = tracker();
        assert_eq!(
            t.plan(
                ImageId(9),
                Subresource::new(0, 0),
                Use::SampledRead,
                Contents::Keep
            ),
            Err(Decline::UnknownImage { image: ImageId(9) })
        );
        let outside = Subresource::new(3, 0);
        assert_eq!(
            t.plan(IMG, outside, Use::SampledRead, Contents::Keep),
            Err(Decline::UnknownSubresource {
                image: IMG,
                subresource: outside
            })
        );
        assert_eq!(
            t.plan(
                IMG,
                Subresource {
                    level: 0,
                    layer: 0,
                    plane: 1
                },
                Use::SampledRead,
                Contents::Keep
            ),
            Err(Decline::UnknownSubresource {
                image: IMG,
                subresource: Subresource {
                    level: 0,
                    layer: 0,
                    plane: 1
                }
            }),
            "a plane the image does not have is a separate layout that does not exist"
        );
    }

    /// Presentation is not a pipeline stage: the image is handed to the
    /// presentation engine and the dependency is a semaphore.
    #[test]
    fn a_present_transition_names_no_stage() {
        let mut t = tracker();
        let sub = Subresource::new(0, 0);
        t.plan(IMG, sub, Use::ColorAttachment, Contents::Discard)
            .expect("declared");
        let transition = t
            .plan(IMG, sub, Use::Present, Contents::Keep)
            .expect("declared")
            .expect("colour attachment to present is a transition");
        assert_eq!(transition.to, vk::ImageLayout::PRESENT_SRC_KHR);
        assert_eq!(transition.dst_stages, vk::PipelineStageFlags2::NONE);
        assert_eq!(transition.dst_access, vk::AccessFlags2::NONE);
        assert!(
            transition
                .src_access
                .contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
            "and the write it has to wait for is still named"
        );
    }

    /// Both halves together, because recording one makes the contents
    /// undefined.
    #[test]
    fn an_ownership_move_is_one_answer_with_both_families_in_it() {
        let mut t = tracker();
        let sub = Subresource::new(0, 0);
        t.plan(IMG, sub, Use::TransferDst, Contents::Discard)
            .expect("declared");
        assert_eq!(t.family(IMG, sub), Ok(Some(0)));
        let transfer = t
            .plan_family(IMG, sub, 1, Use::SampledRead)
            .expect("declared")
            .expect("family 0 to family 1");
        assert_eq!(transfer.from_family, 0);
        assert_eq!(transfer.to_family, 1);
        // The dependency the transfer carries: what the releasing family last
        // did, and what the acquiring family is taking it for. Empty on either
        // side is a move of ownership that orders no memory.
        assert_eq!(transfer.src_stages, Use::TransferDst.stages());
        assert_eq!(transfer.src_access, Use::TransferDst.access());
        assert_eq!(transfer.dst_stages, Use::SampledRead.stages());
        assert_eq!(transfer.dst_access, Use::SampledRead.access());
        assert!(
            !transfer.src_stages.is_empty() && !transfer.dst_stages.is_empty(),
            "a transfer with an empty half orders nothing across the queues"
        );
        assert_eq!(
            transfer.layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            "the two halves must agree on the layout, so it is one value"
        );
        assert_eq!(t.family(IMG, sub), Ok(Some(1)));
        assert_eq!(
            t.plan_family(IMG, sub, 1, Use::SampledRead),
            Ok(None),
            "the family already owns it"
        );
        assert_eq!(t.census().ownership_transfers, 1);
    }

    /// A `CONCURRENT` image needs no transfer, and one whose first owner was
    /// never recorded must not be guessed at either way.
    #[test]
    fn an_image_with_no_recorded_family_refuses_an_ownership_move() {
        let mut t = LayoutTracker::new();
        t.declare(ImageId(2), 1, 1, 1, None);
        let sub = Subresource::new(0, 0);
        assert_eq!(t.family(ImageId(2), sub), Ok(None));
        assert_eq!(
            t.plan_family(ImageId(2), sub, 1, Use::SampledRead),
            Err(Decline::NoRecordedFamily { image: ImageId(2) })
        );
    }

    #[test]
    fn a_forgotten_image_is_no_longer_tracked() {
        let mut t = tracker();
        t.forget(IMG);
        assert_eq!(
            t.layout(IMG, Subresource::new(0, 0)),
            Err(Decline::UnknownImage { image: IMG })
        );
    }

    #[test]
    #[should_panic(expected = "no subresources")]
    fn an_image_with_no_subresources_is_not_a_shallow_image() {
        LayoutTracker::new().declare(IMG, 1, 0, 1, None);
    }

    /// Every use resolves to a layout, and no two uses that need different
    /// treatment share one.
    #[test]
    fn the_uses_that_need_different_layouts_have_different_layouts() {
        let uses = [
            Use::ColorAttachment,
            Use::DepthStencilAttachment,
            Use::DepthStencilRead,
            Use::SampledRead,
            Use::Storage,
            Use::TransferSrc,
            Use::TransferDst,
            Use::Present,
        ];
        let mut layouts: Vec<vk::ImageLayout> = uses.iter().map(|u| u.layout()).collect();
        let count = layouts.len();
        layouts.sort_unstable_by_key(|l| l.as_raw());
        layouts.dedup();
        assert_eq!(layouts.len(), count, "two uses collapsed onto one layout");
        for u in uses {
            assert_ne!(
                u.layout(),
                vk::ImageLayout::UNDEFINED,
                "no use may resolve to the layout that means 'contents gone'"
            );
        }
    }

    #[test]
    fn every_decline_has_its_own_slug() {
        let all = [
            Decline::UnknownImage { image: IMG },
            Decline::UnknownSubresource {
                image: IMG,
                subresource: Subresource::new(0, 0),
            },
            Decline::NoRecordedFamily { image: IMG },
        ];
        let mut slugs: Vec<&str> = all.iter().map(|d| d.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count);
    }

    // ---- A driven sweep of the transition chain --------------------------
    //
    // The unit tests above each state one rule. What none of them states is the
    // property the whole module exists for: that across an arbitrary history of
    // declares, uses, discards, ownership moves and forgets, every transition
    // handed out starts where the previous one for that subresource ended.
    // A gap in that chain is a `oldLayout` the image is not in, which is
    // undefined behavior the validation layers may or may not be watching for.
    //
    // The shadow below is deliberately dumber than the tracker: it never sees a
    // `Transition`, and it derives the layout it expects from the *arguments*
    // of the calls that were accepted --- a use's layout is `Use::layout()` and
    // nothing else. So it cannot agree with the tracker by making the same
    // mistake; it can only agree by the tracker having chained correctly.

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

    const EVERY_USE: [Use; 8] = [
        Use::ColorAttachment,
        Use::DepthStencilAttachment,
        Use::DepthStencilRead,
        Use::SampledRead,
        Use::Storage,
        Use::TransferSrc,
        Use::TransferDst,
        Use::Present,
    ];

    /// One subresource, as the shadow holds it: the last use it was put to and
    /// the family that owns it. The layout is not stored --- it is *derived*
    /// from the last use, which is the point.
    #[derive(Clone, Copy)]
    struct ShadowSub {
        last: Option<Use>,
        family: Option<u32>,
    }

    impl ShadowSub {
        /// A subresource nothing has used is `UNDEFINED`, and every other one
        /// is in the layout its last use needs.
        fn layout(self) -> vk::ImageLayout {
            self.last.map_or(vk::ImageLayout::UNDEFINED, Use::layout)
        }

        fn src_stages(self) -> vk::PipelineStageFlags2 {
            self.last.map_or(vk::PipelineStageFlags2::NONE, Use::stages)
        }

        fn src_access(self) -> vk::AccessFlags2 {
            self.last.map_or(vk::AccessFlags2::NONE, Use::access)
        }
    }

    struct ShadowImage {
        levels: u32,
        layers: u32,
        planes: u32,
        subs: HashMap<Subresource, ShadowSub>,
    }

    #[derive(Default)]
    struct Shadow {
        images: HashMap<ImageId, ShadowImage>,
    }

    impl Shadow {
        fn declare(
            &mut self,
            image: ImageId,
            levels: u32,
            layers: u32,
            planes: u32,
            family: Option<u32>,
        ) {
            let mut subs = HashMap::new();
            for level in 0..levels {
                for layer in 0..layers {
                    for plane in 0..planes {
                        subs.insert(
                            Subresource {
                                level,
                                layer,
                                plane,
                            },
                            ShadowSub { last: None, family },
                        );
                    }
                }
            }
            self.images.insert(
                image,
                ShadowImage {
                    levels,
                    layers,
                    planes,
                    subs,
                },
            );
        }

        fn get(&self, image: ImageId, sub: Subresource) -> Result<ShadowSub, Decline> {
            let img = self
                .images
                .get(&image)
                .ok_or(Decline::UnknownImage { image })?;
            if sub.level >= img.levels || sub.layer >= img.layers || sub.plane >= img.planes {
                return Err(Decline::UnknownSubresource {
                    image,
                    subresource: sub,
                });
            }
            Ok(img.subs[&sub])
        }

        /// Every (image, subresource) the shadow believes is declared, with the
        /// layout and family it believes each is in. Compared against the
        /// tracker after every single call, so a write that lands on the wrong
        /// key is caught at the call that made it and not ten calls later.
        fn projection(&self) -> Vec<(ImageId, Subresource, vk::ImageLayout, Option<u32>)> {
            let mut out = Vec::new();
            for (&id, img) in &self.images {
                for (&sub, state) in &img.subs {
                    out.push((id, sub, state.layout(), state.family));
                }
            }
            out.sort_unstable_by_key(|&(id, sub, _, _)| (id, sub));
            out
        }
    }

    /// What the driver did, so a floor can say the sweep reached each path
    /// rather than that it merely ran.
    #[derive(Default)]
    struct Tally {
        transitions: usize,
        steady: usize,
        discards: usize,
        moves: usize,
        already_owned: usize,
        unknown_image: usize,
        unknown_subresource: usize,
        no_recorded_family: usize,
        forgets: usize,
    }

    #[test]
    fn a_driven_history_never_breaks_the_transition_chain() {
        let ids = [ImageId(1), ImageId(2), ImageId(3), ImageId(4)];
        let mut tally = Tally::default();

        for seed in 0..400_u64 {
            let mut rng = Rng::new(seed);
            let mut tracker = LayoutTracker::new();
            let mut shadow = Shadow::default();
            let (mut ok_plans, mut ok_discards, mut ok_moves) = (0_usize, 0_usize, 0_usize);

            for _ in 0..120 {
                // Steer at the declared images most of the time, so the sweep
                // spends its budget on the chain rather than on refusals ---
                // but not always, or an undeclared image is never asked about.
                let declared: Vec<ImageId> = ids
                    .iter()
                    .copied()
                    .filter(|i| shadow.images.contains_key(i))
                    .collect();
                let image = if !declared.is_empty() && rng.below(4) != 0 {
                    declared[rng.below(declared.len() as u64) as usize]
                } else {
                    ids[rng.below(ids.len() as u64) as usize]
                };
                // Mostly inside the declared extent, so the budget goes on
                // the chain; sometimes one past it, so the out-of-range refusal
                // is driven on a live image rather than only on an absent one.
                let sub = match shadow.images.get(&image) {
                    Some(img) if rng.below(4) != 0 => Subresource {
                        level: rng.below(u64::from(img.levels)) as u32,
                        layer: rng.below(u64::from(img.layers)) as u32,
                        plane: rng.below(u64::from(img.planes)) as u32,
                    },
                    _ => Subresource {
                        level: rng.below(4) as u32,
                        layer: rng.below(3) as u32,
                        plane: rng.below(2) as u32,
                    },
                };

                match rng.below(40) {
                    0..=2 => {
                        let levels = 1 + rng.below(3) as u32;
                        let layers = 1 + rng.below(2) as u32;
                        let planes = 1 + rng.below(2) as u32;
                        let family = if rng.below(4) == 0 {
                            None
                        } else {
                            Some(rng.below(3) as u32)
                        };
                        tracker.declare(image, levels, layers, planes, family);
                        shadow.declare(image, levels, layers, planes, family);
                    }
                    3 => {
                        tracker.forget(image);
                        shadow.images.remove(&image);
                        tally.forgets += 1;
                    }
                    4..=6 => {
                        let to = rng.below(3) as u32;
                        let before = shadow.get(image, sub);
                        match tracker.plan_family(image, sub, to, Use::SampledRead) {
                            Ok(moved) => {
                                let before = before.expect("the tracker accepted it");
                                let from = before.family.expect("an accepted move had an owner");
                                assert_eq!(
                                    moved.is_some(),
                                    from != to,
                                    "a move to the family that already owns it is not a move"
                                );
                                if let Some(transfer) = moved {
                                    assert_eq!(transfer.image, image);
                                    assert_eq!(transfer.subresource, sub);
                                    assert_eq!(transfer.from_family, from);
                                    assert_eq!(transfer.to_family, to);
                                    assert_eq!(
                                        transfer.layout,
                                        before.layout(),
                                        "an ownership move names the layout both halves are in"
                                    );
                                    ok_moves += 1;
                                    tally.moves += 1;
                                } else {
                                    tally.already_owned += 1;
                                }
                                shadow
                                    .images
                                    .get_mut(&image)
                                    .and_then(|i| i.subs.get_mut(&sub))
                                    .expect("declared")
                                    .family = Some(to);
                            }
                            Err(declined) => {
                                match before {
                                    // An owner the shadow has is one the
                                    // tracker must have too, so the only
                                    // refusal left on a live subresource is the
                                    // concurrent image.
                                    Ok(state) => {
                                        assert_eq!(
                                            state.family, None,
                                            "{declined} on an owned subresource"
                                        );
                                        assert_eq!(declined, Decline::NoRecordedFamily { image });
                                        tally.no_recorded_family += 1;
                                    }
                                    Err(expected) => {
                                        assert_eq!(declined, expected);
                                        count_decline(&mut tally, declined);
                                    }
                                }
                            }
                        }
                    }
                    7 => {
                        assert_eq!(
                            tracker.layout(image, sub),
                            shadow.get(image, sub).map(ShadowSub::layout)
                        );
                        assert_eq!(
                            tracker.family(image, sub),
                            shadow.get(image, sub).map(|s| s.family)
                        );
                    }
                    _ => {
                        let before = shadow.get(image, sub);
                        // Repeat the subresource's own last use often. A frame
                        // in its steady state is mostly that, and a driver that
                        // never repeats never reaches the no-transition path
                        // this module exists for.
                        let use_ = match before {
                            Ok(ShadowSub {
                                last: Some(last), ..
                            }) if rng.below(5) < 2 => last,
                            _ => EVERY_USE[rng.below(EVERY_USE.len() as u64) as usize],
                        };
                        // Discards are the interesting minority: they are the
                        // only way `from` may leave the chain, and the sweep is
                        // largely about proving that is the only way.
                        let contents = if rng.below(4) == 0 {
                            Contents::Discard
                        } else {
                            Contents::Keep
                        };
                        let discarding = contents == Contents::Discard;
                        match tracker.plan(image, sub, use_, contents) {
                            Ok(planned) => {
                                let before = before.expect("the tracker accepted it");
                                let owed = before.layout() != use_.layout() || discarding;
                                assert_eq!(
                                    planned.is_some(),
                                    owed,
                                    "a use in the layout it is in owes no transition, \
                                     and a discard always does"
                                );
                                if let Some(t) = planned {
                                    assert_eq!(t.image, image);
                                    assert_eq!(t.subresource, sub);
                                    assert_eq!(t.to, use_.layout());
                                    assert_eq!(t.dst_stages, use_.stages());
                                    assert_eq!(t.dst_access, use_.access());
                                    assert_eq!(t.discarded_contents, discarding);
                                    if discarding {
                                        assert_eq!(t.from, vk::ImageLayout::UNDEFINED);
                                        assert_eq!(t.src_stages, vk::PipelineStageFlags2::NONE);
                                        assert_eq!(t.src_access, vk::AccessFlags2::NONE);
                                        ok_discards += 1;
                                        tally.discards += 1;
                                    } else {
                                        // The chain: this transition starts
                                        // where the last accepted use for this
                                        // exact subresource left it.
                                        assert_eq!(
                                            t.from,
                                            before.layout(),
                                            "the chain broke at {image:?} {sub:?}"
                                        );
                                        assert_eq!(t.src_stages, before.src_stages());
                                        assert_eq!(t.src_access, before.src_access());
                                        tally.transitions += 1;
                                    }
                                } else {
                                    tally.steady += 1;
                                }
                                ok_plans += 1;
                                shadow
                                    .images
                                    .get_mut(&image)
                                    .and_then(|i| i.subs.get_mut(&sub))
                                    .expect("declared")
                                    .last = Some(use_);
                            }
                            Err(declined) => {
                                assert_eq!(Err(declined), before.map(|_| unreachable!()));
                                count_decline(&mut tally, declined);
                            }
                        }
                    }
                }

                // After every call, including every refused one: the tracker's
                // own answers for every live subresource, against the shadow's.
                // A refusal that mutated something, or a write that landed on a
                // neighbouring key, fails here at the call that did it.
                let mut seen = Vec::new();
                for (&id, img) in &shadow.images {
                    for &sub in img.subs.keys() {
                        seen.push((
                            id,
                            sub,
                            tracker
                                .layout(id, sub)
                                .expect("the shadow says it is declared"),
                            tracker
                                .family(id, sub)
                                .expect("the shadow says it is declared"),
                        ));
                    }
                }
                seen.sort_unstable_by_key(|&(id, sub, _, _)| (id, sub));
                assert_eq!(seen, shadow.projection());
            }

            // The census counts what the tracker was *asked*, over the whole
            // history --- so it survives every forget in it.
            let census = tracker.census();
            assert_eq!(census.already_in_layout + census.transitions, ok_plans);
            assert_eq!(census.discards, ok_discards);
            assert!(census.discards <= census.transitions);
            assert_eq!(census.ownership_transfers, ok_moves);
        }

        // Floors on each path separately. One aggregate "some refusals
        // happened" would let a path go undriven and still read as covered.
        assert!(tally.transitions > 5_000, "{}", tally.transitions);
        assert!(tally.steady > 2_000, "{}", tally.steady);
        assert!(tally.discards > 2_000, "{}", tally.discards);
        assert!(tally.moves > 500, "{}", tally.moves);
        assert!(tally.already_owned > 200, "{}", tally.already_owned);
        assert!(tally.unknown_image > 5_000, "{}", tally.unknown_image);
        assert!(
            tally.unknown_subresource > 2_000,
            "{}",
            tally.unknown_subresource
        );
        assert!(
            tally.no_recorded_family > 200,
            "{}",
            tally.no_recorded_family
        );
        assert!(tally.forgets > 500, "{}", tally.forgets);
    }

    fn count_decline(tally: &mut Tally, declined: Decline) {
        match declined {
            Decline::UnknownImage { .. } => tally.unknown_image += 1,
            Decline::UnknownSubresource { .. } => tally.unknown_subresource += 1,
            Decline::NoRecordedFamily { .. } => tally.no_recorded_family += 1,
        }
    }
}
