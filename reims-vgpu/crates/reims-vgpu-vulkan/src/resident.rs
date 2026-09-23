//! Which native object a guest resource name resolves to, and what happens to
//! the previous one when the guest reuses the name.
//!
//! # A name is a slot and a generation, and the two failures are different
//!
//! The guest names resources by object-list slot. It deletes and recreates in
//! the same slot constantly, so a slot number alone is not an identity — work
//! still holding the old number would resolve to the new object, read the
//! wrong texture, and produce a frame nobody can explain. [`ResourceId`]
//! carries the generation for that reason, and this table resolves on both.
//!
//! A slot that was never filled and a slot filled by a *later* generation are
//! kept as separate refusals. They are different defects: the first is decode
//! or ordering, the second is a lifetime that let accepted work outlive its
//! resource. One refusal covering both would make the more serious one
//! invisible inside the noisier one.
//!
//! # Publishing over a live entry does not free it
//!
//! When the guest recreates a slot, the previous object may still be named by
//! a submission the GPU has not reached. So a replacement retires the old
//! entry through [`NativeRetirement`] against the point of its last use — it
//! leaves the table immediately, because nothing may resolve to it any more,
//! and it leaves the device only when the timeline says so. Destroying it at
//! replacement is the use-after-free that looks like a driver bug.
//!
//! Publishing the *same* generation twice is refused instead of retiring
//! anything: two natives for one guest object is not a lifetime event, it is a
//! defect, and the second native is handed straight back rather than leaked.
//!
//! A publication naming an *older* generation than the slot holds is refused
//! for a stronger reason. Generations only ever advance, so an older one is a
//! publication that arrived after the declaration replacing it. Taking it
//! would retire the newer native --- destroying the object the guest's current
//! name means --- and leave that name resolving to [`Miss::Stale`] against a
//! generation from the past. That is the same defect [`Residency::delete`]
//! already refuses, where a delete naming a stale generation must not take the
//! live object with it; publication is the other door into it, and the two
//! must answer alike.
//!
//! # This table holds handles and calls nothing
//!
//! Retirement produces the objects to destroy; the caller destroys them
//! through the device that made them. So every rule above is tested with no
//! GPU.

// A refused publication returns the native it would not take, so the caller
// destroys what it just made instead of leaking it. That makes the `Err`
// variant as large as a native object by construction; boxing it would put a
// heap allocation on the failure path in exchange for nothing.
#![allow(clippy::result_large_err)]

use ash::vk;
use reims_vgpu_core::identity::{
    DeviceEpoch, ObjectListRef, ResourceId, SessionGeneration, SlotGeneration, TimelinePoint,
};
use reims_vgpu_core::retire::{Lifetime, NativeRetirement};
use reims_vgpu_core::texture_shape::Texture;
use std::collections::BTreeMap;

use crate::buffer::BufferPlan;
use crate::image::ImagePlan;

/// A guest buffer, allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub plan: BufferPlan,
}

/// A guest texture, allocated, with every view it is addressable through.
///
/// The views are here rather than in a table beside this one because their
/// lifetime is exactly the image's — see [`crate::view`]. A second table would
/// be a second thing to retire in the same order and a second thing to get
/// wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeImage {
    /// The checked declaration this image was made from.
    ///
    /// Kept rather than re-derived from [`Self::plan`]: a transfer needs the
    /// guest's own format code to convert a byte pitch into texels and to name
    /// an aspect, and a `VkFormat` cannot answer either — several guest formats
    /// map to one native one, and the mapping is not invertible.
    pub texture: Texture,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub plan: ImagePlan,
    /// The whole-texture views a sampled binding uses, by the guest format the
    /// bind named.
    ///
    /// A map and not one view, because one image genuinely has several, and
    /// two independent decisions of this rail each produce a second one:
    ///
    /// - [`crate::pixel::sample_view_format`] answers a sampled bind with the
    ///   bind's own spelling of the stored texel, so one surface serves both
    ///   `BGRA8Unorm` and `BGRA8Unorm_sRGB` through one image and two views.
    ///   That is why [`crate::image::plan`] sets `MUTABLE_FORMAT` for a format
    ///   with a second spelling whatever the guest declared.
    /// - [`crate::view::sampled_aspect`] reads which plane a bind means off
    ///   the guest view format, because `X32_Stencil8` and
    ///   `Depth32Float_Stencil8` translate to one `VkFormat` and are two views
    ///   of one combined texture.
    ///
    /// The guest view format is the key because it is the input both of those
    /// take and the only thing that separates the cases — a `VkFormat` cannot,
    /// which is the second point's whole content. One view here would make the
    /// state this rail already decided on unrepresentable, and a caller
    /// reaching for "the" sampled view would bind the depth plane to a shader
    /// reading stencil, or the linear spelling to a shader that decodes sRGB.
    ///
    /// Their lifetime is exactly the image's, which is why they live here; see
    /// [`Self::views`] for the destruction order that follows from it.
    pub sampled: BTreeMap<u16, vk::ImageView>,
    /// One per attachable slice, in [`crate::view::attachments`] order.
    pub attachments: Vec<vk::ImageView>,
}

impl NativeImage {
    /// The sampled view a bind naming `guest_view_format` reads through, if
    /// one has been made for it.
    ///
    /// `None` is "not made yet" and not "not possible": the set grows as the
    /// guest names spellings, and every one of them is legal because the image
    /// carries `MUTABLE_FORMAT` whenever a second spelling exists.
    #[must_use]
    pub fn sampled_view(&self, guest_view_format: u16) -> Option<vk::ImageView> {
        self.sampled.get(&guest_view_format).copied()
    }

    /// Every `VkImageView` this image owns, sampled and attachment alike.
    ///
    /// One list because they are destroyed together with the image, and a
    /// caller walking only one of the two fields would leak the other — which
    /// is the failure the fields being here rather than in a second table
    /// exists to prevent, and which two fields reintroduce unless something
    /// names them together.
    pub fn views(&self) -> impl Iterator<Item = vk::ImageView> + '_ {
        self.sampled
            .values()
            .copied()
            .chain(self.attachments.iter().copied())
    }
}

/// What a guest resource name resolves to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Native {
    Buffer(NativeBuffer),
    Image(NativeImage),
}

impl Native {
    #[must_use]
    pub const fn kind(&self) -> Kind {
        match self {
            Self::Buffer(_) => Kind::Buffer,
            Self::Image(_) => Kind::Image,
        }
    }
}

/// Which of the two a name is bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Buffer,
    Image,
}

impl Kind {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Image => "image",
        }
    }
}

/// Why a resource name resolved to nothing usable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Miss {
    /// Nothing was ever published in this slot.
    Unknown { slot: ObjectListRef },
    /// The slot holds a different generation than the name asks for.
    ///
    /// `held` newer than `named` is accepted work outliving its resource;
    /// `held` older is a name from ahead of what this table has been told.
    /// Both are here, with both numbers, because they are opposite defects.
    Stale {
        slot: ObjectListRef,
        held: SlotGeneration,
        named: SlotGeneration,
    },
    /// The name resolves, but to the other kind of object.
    WrongKind {
        slot: ObjectListRef,
        held: Kind,
        wanted: Kind,
    },
}

impl Miss {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unknown { .. } => "vk_resident_unknown_slot",
            Self::Stale { .. } => "vk_resident_stale_generation",
            Self::WrongKind { .. } => "vk_resident_wrong_kind",
        }
    }
}

impl std::fmt::Display for Miss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { slot } => write!(f, "{} slot={}", self.slug(), slot.0),
            Self::Stale { slot, held, named } => write!(
                f,
                "{} slot={} held={} named={}",
                self.slug(),
                slot.0,
                held.0,
                named.0
            ),
            Self::WrongKind { slot, held, wanted } => write!(
                f,
                "{} slot={} held={} wanted={}",
                self.slug(),
                slot.0,
                held.name(),
                wanted.name()
            ),
        }
    }
}

/// Why a native object could not be bound to a guest name.
///
/// Both variants are a publication this slot's history does not admit, and
/// both hand the native back; they are separate because they are opposite
/// defects, the same way [`Miss::Unknown`] and [`Miss::Stale`] are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A second native for a guest object that already has one.
    Occupied {
        slot: ObjectListRef,
        generation: SlotGeneration,
    },
    /// A publication naming a generation this slot has already moved past.
    ///
    /// Carries both numbers for the reason [`Miss::Stale`] does: the distance
    /// says whether one declaration was reordered or a whole run of them was.
    Rewound {
        slot: ObjectListRef,
        held: SlotGeneration,
        named: SlotGeneration,
    },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Occupied { .. } => "vk_resident_already_published",
            Self::Rewound { .. } => "vk_resident_generation_rewound",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Occupied { slot, generation } => write!(
                f,
                "{} slot={} generation={}",
                self.slug(),
                slot.0,
                generation.0
            ),
            Self::Rewound { slot, held, named } => write!(
                f,
                "{} slot={} held={} named={}",
                self.slug(),
                slot.0,
                held.0,
                named.0
            ),
        }
    }
}

/// One entry, and the lifetimes its handles are under.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    generation: SlotGeneration,
    lifetime: Lifetime,
    /// The latest point a submission naming this object will signal. Zero
    /// until something uses it, which is correct: a resource nothing has
    /// submitted against is retired the moment it is deleted.
    last_use: TimelinePoint,
    native: Native,
}

/// The native objects one device epoch holds for one session generation.
///
/// Keyed by slot, not by `ResourceId`: a slot holds at most one object, and a
/// map keyed by the pair would silently accept two generations of one slot
/// being live at once — which is exactly the state that lets old work resolve.
#[derive(Debug, Default)]
pub struct Residency {
    slots: BTreeMap<u32, Entry>,
    published: usize,
    replaced: usize,
}

impl Residency {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn population(&self) -> usize {
        self.slots.len()
    }

    /// How many objects have been published, and how many of those displaced a
    /// live predecessor. A replacement count that tracks the publication count
    /// is a guest churning one slot, which is a real thing to see.
    #[must_use]
    pub const fn census(&self) -> (usize, usize) {
        (self.published, self.replaced)
    }

    /// Bind a native object to a guest name.
    ///
    /// A previous generation in the same slot is retired against its last use;
    /// see the module doc.
    ///
    /// # Errors
    ///
    /// [`Refusal`] when this slot's generation is not moving forward, with the
    /// native handed back so the caller can destroy what it just made.
    pub fn publish(
        &mut self,
        id: ResourceId,
        lifetime: Lifetime,
        native: Native,
        retire: &mut NativeRetirement<Native>,
    ) -> Result<(), (Native, Refusal)> {
        if let Some(previous) = self.slots.get(&id.slot.0) {
            if previous.generation == id.generation {
                return Err((
                    native,
                    Refusal::Occupied {
                        slot: id.slot,
                        generation: id.generation,
                    },
                ));
            }
            if previous.generation > id.generation {
                return Err((
                    native,
                    Refusal::Rewound {
                        slot: id.slot,
                        held: previous.generation,
                        named: id.generation,
                    },
                ));
            }
        }
        let displaced = self.slots.insert(
            id.slot.0,
            Entry {
                generation: id.generation,
                lifetime,
                last_use: TimelinePoint(0),
                native,
            },
        );
        self.published += 1;
        if let Some(previous) = displaced {
            self.replaced += 1;
            retire.queue(previous.lifetime, previous.last_use, previous.native);
        }
        Ok(())
    }

    /// What a name resolves to.
    ///
    /// # Errors
    ///
    /// [`Miss`] naming which of the two failures happened.
    pub fn resolve(&self, id: ResourceId) -> Result<&Native, Miss> {
        Ok(&self.entry(id)?.native)
    }

    fn entry(&self, id: ResourceId) -> Result<&Entry, Miss> {
        let entry = self
            .slots
            .get(&id.slot.0)
            .ok_or(Miss::Unknown { slot: id.slot })?;
        if entry.generation != id.generation {
            return Err(Miss::Stale {
                slot: id.slot,
                held: entry.generation,
                named: id.generation,
            });
        }
        Ok(entry)
    }

    /// The buffer a name resolves to.
    ///
    /// # Errors
    ///
    /// [`Miss`], including [`Miss::WrongKind`] when the name is a texture. A
    /// caller that resolved the enum itself and matched on it would either
    /// have to repeat this refusal or fall through silently.
    pub fn buffer(&self, id: ResourceId) -> Result<&NativeBuffer, Miss> {
        match self.resolve(id)? {
            Native::Buffer(buffer) => Ok(buffer),
            other => Err(Miss::WrongKind {
                slot: id.slot,
                held: other.kind(),
                wanted: Kind::Buffer,
            }),
        }
    }

    /// The image a name resolves to.
    ///
    /// # Errors
    ///
    /// [`Miss`], including [`Miss::WrongKind`] when the name is a buffer.
    pub fn image(&self, id: ResourceId) -> Result<&NativeImage, Miss> {
        match self.resolve(id)? {
            Native::Image(image) => Ok(image),
            other => Err(Miss::WrongKind {
                slot: id.slot,
                held: other.kind(),
                wanted: Kind::Image,
            }),
        }
    }

    /// Record that a submission signalling `at` names this resource.
    ///
    /// Moves forward and never back: two submissions may name one resource,
    /// and the one that frees it is the later. Keeping the earlier point is
    /// how a destroy races a submission that is still running.
    ///
    /// # Errors
    ///
    /// [`Miss`] when the name does not resolve, so a use of a stale name
    /// cannot silently extend the live object's lifetime.
    pub fn used(&mut self, id: ResourceId, at: TimelinePoint) -> Result<(), Miss> {
        // Resolved first, so a stale name changes nothing.
        self.entry(id)?;
        let entry = self
            .slots
            .get_mut(&id.slot.0)
            .expect("the entry resolved a line ago");
        if at.0 > entry.last_use.0 {
            entry.last_use = at;
        }
        Ok(())
    }

    /// The guest deleted this resource.
    ///
    /// Leaves the table at once — nothing may resolve to it again — and leaves
    /// the device when the timeline passes its last use.
    ///
    /// # Errors
    ///
    /// [`Miss`] when the name does not resolve. Nothing is removed: a delete
    /// naming a stale generation must not take the live object with it, which
    /// is precisely the bug a slot-only key produces.
    pub fn delete(
        &mut self,
        id: ResourceId,
        retire: &mut NativeRetirement<Native>,
    ) -> Result<(), Miss> {
        self.entry(id)?;
        let entry = self
            .slots
            .remove(&id.slot.0)
            .expect("the entry resolved a line ago");
        retire.queue(entry.lifetime, entry.last_use, entry.native);
        Ok(())
    }

    /// The session generation closed, or the device epoch ended: everything
    /// here stops being nameable.
    ///
    /// Queues every entry rather than destroying any, because a closed
    /// generation says nothing about whether the GPU is finished — that is
    /// still the timeline's answer, and [`NativeRetirement`] is where the two
    /// lifetimes are compared.
    pub fn drain(&mut self, retire: &mut NativeRetirement<Native>) -> usize {
        let drained = self.slots.len();
        for (_, entry) in std::mem::take(&mut self.slots) {
            retire.queue(entry.lifetime, entry.last_use, entry.native);
        }
        drained
    }

    /// What is left of a name's lease against the current lifetimes.
    ///
    /// # Errors
    ///
    /// [`Miss`] when the name does not resolve at all.
    pub fn validity(
        &self,
        id: ResourceId,
        session: SessionGeneration,
        epoch: DeviceEpoch,
    ) -> Result<reims_vgpu_core::retire::Validity, Miss> {
        Ok(self.entry(id)?.lifetime.against(session, epoch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::retire::Validity;
    use std::collections::BTreeSet;

    fn id(slot: u32, generation: u64) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(generation),
        }
    }

    fn lifetime() -> Lifetime {
        Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST)
    }

    fn native_buffer(handle: u64) -> Native {
        Native::Buffer(NativeBuffer {
            buffer: vk::Buffer::from_raw(handle),
            memory: vk::DeviceMemory::from_raw(handle),
            plan: BufferPlan {
                size: 64,
                usage: crate::buffer::EVERY_CLASS,
                aliased: false,
            },
        })
    }

    fn native_image(handle: u64) -> Native {
        Native::Image(NativeImage {
            texture: reims_vgpu_core::texture_shape::TextureShape {
                kind: reims_vgpu_core::texture_shape::TextureKind::D2.ordinal(),
                width: 4,
                height: 4,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 1,
                pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                usage: reims_vgpu_core::texture_shape::TextureUsage::SHADER_READ,
            }
            .checked()
            .expect("a valid declaration"),
            image: vk::Image::from_raw(handle),
            memory: vk::DeviceMemory::from_raw(handle),
            plan: ImagePlan {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
            sampled: BTreeMap::from([(
                reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                vk::ImageView::from_raw(handle),
            )]),
            attachments: Vec::new(),
        })
    }

    fn handle(native: &Native) -> u64 {
        match native {
            Native::Buffer(b) => b.buffer.as_raw(),
            Native::Image(i) => i.image.as_raw(),
        }
    }

    /// One image, several sampled views. Two of this rail's own decisions each
    /// produce a second one — a bind's sRGB spelling of a stored texel, and
    /// the plane a combined depth-stencil bind names — and a record holding
    /// one view could express neither.
    #[test]
    fn an_image_holds_one_sampled_view_per_spelling_the_guest_names() {
        use reims_vgpu_core::pixel_format::{
            MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA8_UNORM,
        };
        let Native::Image(mut image) = native_image(7) else {
            panic!("an image");
        };
        image.sampled = BTreeMap::from([
            (MTL_FORMAT_BGRA8_UNORM, vk::ImageView::from_raw(1)),
            (MTL_FORMAT_BGRA8_UNORM_SRGB, vk::ImageView::from_raw(2)),
        ]);
        image.attachments = vec![vk::ImageView::from_raw(3)];

        // The two spellings are one image and two views, and the bind's own
        // spelling is what selects between them.
        assert_eq!(
            image.sampled_view(MTL_FORMAT_BGRA8_UNORM),
            Some(vk::ImageView::from_raw(1))
        );
        assert_eq!(
            image.sampled_view(MTL_FORMAT_BGRA8_UNORM_SRGB),
            Some(vk::ImageView::from_raw(2))
        );
        // Not made yet is not impossible: the set grows as the guest names
        // spellings.
        assert_eq!(image.sampled_view(MTL_FORMAT_RGBA8_UNORM), None);

        // And everything destroyed with the image is reachable from one walk,
        // so a caller cannot free the attachment views and leak the sampled
        // ones.
        let views: Vec<vk::ImageView> = image.views().collect();
        assert_eq!(
            views,
            vec![
                vk::ImageView::from_raw(1),
                vk::ImageView::from_raw(2),
                vk::ImageView::from_raw(3),
            ]
        );
    }

    #[test]
    fn a_published_name_resolves_to_what_was_published() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(3, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        assert_eq!(residency.population(), 1);
        assert_eq!(
            residency
                .buffer(id(3, 1))
                .expect("a buffer")
                .buffer
                .as_raw(),
            0xB1
        );
        assert_eq!(retire.outstanding(), 0);
    }

    #[test]
    fn an_unfilled_slot_and_a_stale_generation_are_different_refusals() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        assert_eq!(
            residency.resolve(id(3, 1)).err(),
            Some(Miss::Unknown {
                slot: ObjectListRef(3)
            })
        );

        residency
            .publish(id(3, 2), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        // A name from before the recreation, which is accepted work outliving
        // its resource — and must not resolve to the new object.
        assert_eq!(
            residency.resolve(id(3, 1)).err(),
            Some(Miss::Stale {
                slot: ObjectListRef(3),
                held: SlotGeneration(2),
                named: SlotGeneration(1),
            })
        );
        // And a name from ahead of it, which is the opposite defect.
        assert_eq!(
            residency.resolve(id(3, 3)).err(),
            Some(Miss::Stale {
                slot: ObjectListRef(3),
                held: SlotGeneration(2),
                named: SlotGeneration(3),
            })
        );
    }

    #[test]
    fn recreating_a_slot_retires_the_previous_object_rather_than_dropping_it() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(3, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        residency
            .used(id(3, 1), TimelinePoint(9))
            .expect("a live name");
        residency
            .publish(id(3, 2), lifetime(), native_buffer(0xB2), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        // Out of the table immediately: nothing may resolve to it.
        assert_eq!(residency.population(), 1);
        assert_eq!(
            residency
                .buffer(id(3, 2))
                .expect("the new one")
                .buffer
                .as_raw(),
            0xB2
        );
        // Still on the device until the GPU is past the submission naming it.
        assert_eq!(retire.outstanding(), 1);
        assert!(retire
            .reached(DeviceEpoch::FIRST, TimelinePoint(8))
            .is_empty());
        let retired = retire.reached(DeviceEpoch::FIRST, TimelinePoint(9));
        assert_eq!(retired.len(), 1);
        assert_eq!(handle(&retired[0].object), 0xB1);
        assert_eq!(residency.census(), (2, 1));
    }

    #[test]
    fn publishing_the_same_generation_twice_hands_the_second_native_back() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(3, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        let (returned, occupied) = residency
            .publish(id(3, 1), lifetime(), native_buffer(0xB2), &mut retire)
            .expect_err("one guest object has one native");

        assert_eq!(
            occupied,
            Refusal::Occupied {
                slot: ObjectListRef(3),
                generation: SlotGeneration(1),
            }
        );
        // Handed back rather than leaked, and nothing was retired: the caller
        // destroys what it just made.
        assert_eq!(handle(&returned), 0xB2);
        assert_eq!(retire.outstanding(), 0);
        assert_eq!(
            residency
                .buffer(id(3, 1))
                .expect("the first")
                .buffer
                .as_raw(),
            0xB1
        );
        assert_eq!(residency.census(), (1, 0));
    }

    #[test]
    fn publishing_a_generation_the_slot_has_moved_past_does_not_retire_the_live_one() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(3, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        residency
            .publish(id(3, 4), lifetime(), native_buffer(0xB4), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        assert_eq!(residency.census(), (2, 1));

        // A publication of generation 2 arriving now is one the guest has
        // already declared past. Taking it would destroy 0xB4 --- the object
        // the guest's current name means --- and leave that name resolving
        // against a generation from the past.
        let (returned, refusal) = residency
            .publish(id(3, 2), lifetime(), native_buffer(0xB2), &mut retire)
            .expect_err("a slot's generation only advances");

        assert_eq!(
            refusal,
            Refusal::Rewound {
                slot: ObjectListRef(3),
                held: SlotGeneration(4),
                named: SlotGeneration(2),
            }
        );
        assert_eq!(handle(&returned), 0xB2);
        // Nothing retired beyond the one lawful replacement, and the live name
        // still answers.
        assert_eq!(retire.outstanding(), 1);
        assert_eq!(
            residency
                .buffer(id(3, 4))
                .expect("the live one")
                .buffer
                .as_raw(),
            0xB4
        );
        assert_eq!(residency.census(), (2, 1));
    }

    #[test]
    fn a_use_moves_the_last_point_forward_and_never_back() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(1, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        residency.used(id(1, 1), TimelinePoint(7)).expect("live");
        // An earlier submission admitted after a later one — which happens,
        // because admission order is not submission order.
        residency.used(id(1, 1), TimelinePoint(3)).expect("live");
        residency.delete(id(1, 1), &mut retire).expect("live");

        // Freed at seven and not at three: the later submission is the one
        // still reading it.
        assert!(retire
            .reached(DeviceEpoch::FIRST, TimelinePoint(3))
            .is_empty());
        assert_eq!(
            retire.reached(DeviceEpoch::FIRST, TimelinePoint(7)).len(),
            1
        );
    }

    #[test]
    fn a_use_of_a_stale_name_extends_nothing() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(1, 2), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        assert!(residency.used(id(1, 1), TimelinePoint(99)).is_err());
        residency.delete(id(1, 2), &mut retire).expect("live");
        // Not held to 99 by a name that resolves to nothing.
        assert_eq!(
            retire.reached(DeviceEpoch::FIRST, TimelinePoint(0)).len(),
            1
        );
    }

    #[test]
    fn deleting_a_stale_name_does_not_take_the_live_object_with_it() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(4, 2), lifetime(), native_buffer(0xB2), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        assert!(matches!(
            residency.delete(id(4, 1), &mut retire),
            Err(Miss::Stale { .. })
        ));
        assert_eq!(residency.population(), 1);
        assert_eq!(retire.outstanding(), 0);
        assert!(residency.buffer(id(4, 2)).is_ok());
    }

    #[test]
    fn resolving_a_texture_as_a_buffer_refuses_by_name() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(2, 1), lifetime(), native_image(0x1A), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        assert_eq!(
            residency.buffer(id(2, 1)).err(),
            Some(Miss::WrongKind {
                slot: ObjectListRef(2),
                held: Kind::Image,
                wanted: Kind::Buffer,
            })
        );
        assert!(residency.image(id(2, 1)).is_ok());
        // And the mirror, so neither accessor is the only one that checks.
        residency
            .publish(id(5, 1), lifetime(), native_buffer(0xB5), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        assert_eq!(
            residency.image(id(5, 1)).err(),
            Some(Miss::WrongKind {
                slot: ObjectListRef(5),
                held: Kind::Buffer,
                wanted: Kind::Image,
            })
        );
    }

    #[test]
    fn a_closed_generation_leaves_the_table_and_waits_for_the_timeline() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        for slot in 0..3 {
            residency
                .publish(
                    id(slot, 1),
                    lifetime(),
                    native_buffer(u64::from(slot) + 1),
                    &mut retire,
                )
                .unwrap_or_else(|(_, e)| panic!("{e}"));
            residency
                .used(id(slot, 1), TimelinePoint(u64::from(slot) + 1))
                .expect("live");
        }

        assert_eq!(residency.drain(&mut retire), 3);
        assert_eq!(residency.population(), 0);
        assert_eq!(retire.outstanding(), 3);
        // Each at its own point: a generation closing says nothing about what
        // the GPU has finished.
        assert_eq!(
            retire.reached(DeviceEpoch::FIRST, TimelinePoint(1)).len(),
            1
        );
        assert_eq!(
            retire.reached(DeviceEpoch::FIRST, TimelinePoint(3)).len(),
            2
        );
    }

    #[test]
    fn a_name_reports_which_of_its_two_lifetimes_ended() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(id(1, 1), lifetime(), native_buffer(0xB1), &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        let session = SessionGeneration::FIRST;
        let epoch = DeviceEpoch::FIRST;
        assert_eq!(
            residency.validity(id(1, 1), session, epoch),
            Ok(Validity::Live)
        );
        assert_eq!(
            residency.validity(id(1, 1), session.next(), epoch),
            Ok(Validity::SemanticallyClosed)
        );
        assert_eq!(
            residency.validity(id(1, 1), session, epoch.next()),
            Ok(Validity::HandlesUnusable)
        );
        assert_eq!(
            residency.validity(id(1, 1), session.next(), epoch.next()),
            Ok(Validity::Gone)
        );
        assert!(residency.validity(id(1, 2), session, epoch).is_err());
    }

    #[test]
    fn every_refusal_names_itself() {
        let misses = [
            Miss::Unknown {
                slot: ObjectListRef(1),
            },
            Miss::Stale {
                slot: ObjectListRef(1),
                held: SlotGeneration(2),
                named: SlotGeneration(1),
            },
            Miss::WrongKind {
                slot: ObjectListRef(1),
                held: Kind::Buffer,
                wanted: Kind::Image,
            },
        ];
        let slugs: BTreeSet<&str> = misses.iter().map(|m| m.slug()).collect();
        assert_eq!(slugs.len(), misses.len());
        for miss in misses {
            assert!(miss.to_string().starts_with(miss.slug()));
            assert!(miss.slug().starts_with("vk_resident_"));
        }
        let refusals = [
            Refusal::Occupied {
                slot: ObjectListRef(1),
                generation: SlotGeneration(1),
            },
            Refusal::Rewound {
                slot: ObjectListRef(1),
                held: SlotGeneration(2),
                named: SlotGeneration(1),
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_resident_"));
        }
    }

    // ---- A driven history of the table's lifetime bookkeeping ------------
    //
    // Every rule this module states is about *conservation*: a native that
    // leaves the table reaches retirement exactly once, against the last point
    // a submission named it, and nothing else. A native queued twice is a
    // double destroy; a native queued against too early a point is a
    // use-after-free; a native that leaves the table and is never queued is a
    // leak. None of the three is visible in a single call.
    //
    // The shadow below never asks the table anything. It follows the calls'
    // *arguments* --- which slot, which generation, which point --- and every
    // claim is checked against that.

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

    /// One slot, as the shadow holds it.
    #[derive(Clone, Copy)]
    struct Held {
        generation: u64,
        last_use: u64,
        /// The `vkBuffer`/`vkImage` raw handle, which is unique per publish and
        /// is therefore the identity a conservation claim is written against.
        handle: u64,
        is_image: bool,
    }

    #[derive(Default)]
    struct Tally {
        published: usize,
        occupied: usize,
        rewound: usize,
        replaced: usize,
        deleted: usize,
        used: usize,
        used_stale: usize,
        drained: usize,
        unknown: usize,
        stale: usize,
        wrong_kind: usize,
    }

    #[test]
    fn every_native_that_leaves_the_table_is_retired_once_against_its_last_use() {
        let mut tally = Tally::default();
        for seed in 0..600_u64 {
            let mut rng = Rng::new(seed);
            let mut table = Residency::new();
            let mut retire = NativeRetirement::new();
            let mut shadow: BTreeMap<u32, Held> = BTreeMap::new();
            // Every handle ever minted, and where the shadow says it went.
            let mut minted: Vec<u64> = Vec::new();
            // handle -> the last-use point the shadow says it left with. The
            // number a destroy must wait for, and the one a use-after-free
            // gets wrong.
            let mut departed: BTreeMap<u64, u64> = BTreeMap::new();
            let mut handle = 0_u64;
            let (mut publishes, mut replacements) = (0_usize, 0_usize);

            for _ in 0..80 {
                let slot = rng.below(5) as u32;
                // Mostly the generation the shadow holds, so the resolving
                // paths are driven; sometimes a neighbour on either side, so
                // both directions of `Stale` are, and so a publication naming
                // a generation the slot has already moved past is.
                let generation = match shadow.get(&slot) {
                    Some(held) if rng.below(4) != 0 => held.generation,
                    Some(held) if rng.below(2) == 0 => held.generation + 1 + rng.below(2),
                    Some(held) => held.generation.saturating_sub(1 + rng.below(2)).max(1),
                    None => 1 + rng.below(3),
                };
                let id = id(slot, generation);

                match rng.below(16) {
                    0..=4 => {
                        handle += 1;
                        let is_image = rng.below(2) == 0;
                        let native = if is_image {
                            native_image(handle)
                        } else {
                            native_buffer(handle)
                        };
                        let displaced = shadow.get(&slot).copied();
                        match table.publish(id, lifetime(), native, &mut retire) {
                            Ok(()) => {
                                assert!(
                                    displaced.is_none_or(|d| d.generation < generation),
                                    "a slot's generation only ever advances"
                                );
                                minted.push(handle);
                                publishes += 1;
                                tally.published += 1;
                                if let Some(gone) = displaced {
                                    replacements += 1;
                                    tally.replaced += 1;
                                    departed.insert(gone.handle, gone.last_use);
                                }
                                shadow.insert(
                                    slot,
                                    Held {
                                        generation,
                                        last_use: 0,
                                        handle,
                                        is_image,
                                    },
                                );
                            }
                            Err((back, refusal)) => {
                                let held = displaced.expect("something refused it");
                                // Which refusal, derived from the history and
                                // never from the table.
                                if held.generation == generation {
                                    assert_eq!(
                                        refusal,
                                        Refusal::Occupied {
                                            slot: id.slot,
                                            generation: id.generation,
                                        }
                                    );
                                    tally.occupied += 1;
                                } else {
                                    assert!(
                                        held.generation > generation,
                                        "a publication that advances the slot is lawful"
                                    );
                                    assert_eq!(
                                        refusal,
                                        Refusal::Rewound {
                                            slot: id.slot,
                                            held: SlotGeneration(held.generation),
                                            named: id.generation,
                                        }
                                    );
                                    tally.rewound += 1;
                                }
                                // Handed back rather than leaked, and it is the
                                // one the caller passed in.
                                assert_eq!(raw_of(&back), handle);
                                handle -= 1;
                            }
                        }
                    }
                    5..=7 => {
                        let at = TimelinePoint(1 + rng.below(20));
                        let before = shadow.get(&slot).copied();
                        match table.used(id, at) {
                            Ok(()) => {
                                let held = before.expect("it resolved");
                                assert_eq!(held.generation, generation);
                                // Forward only: an earlier submission naming a
                                // resource does not shorten a later one's hold.
                                shadow.get_mut(&slot).expect("held").last_use =
                                    held.last_use.max(at.0);
                                tally.used += 1;
                            }
                            Err(miss) => {
                                assert_eq!(miss, expected_miss(before, slot, generation, None));
                                tally.used_stale += 1;
                                count_miss(&mut tally, miss);
                            }
                        }
                    }
                    8 | 9 => {
                        let before = shadow.get(&slot).copied();
                        match table.delete(id, &mut retire) {
                            Ok(()) => {
                                let held = before.expect("it resolved");
                                assert_eq!(held.generation, generation);
                                departed.insert(held.handle, held.last_use);
                                shadow.remove(&slot);
                                tally.deleted += 1;
                            }
                            Err(miss) => {
                                assert_eq!(miss, expected_miss(before, slot, generation, None));
                                count_miss(&mut tally, miss);
                            }
                        }
                    }
                    10 => {
                        let expected = shadow.len();
                        assert_eq!(table.drain(&mut retire), expected);
                        for gone in shadow.values() {
                            departed.insert(gone.handle, gone.last_use);
                        }
                        shadow.clear();
                        tally.drained += 1;
                    }
                    _ => {
                        // The three resolving doors, against the shadow.
                        let before = shadow.get(&slot).copied();
                        let want_image = rng.below(2) == 0;
                        let resolved = if want_image {
                            table.image(id).map(|i| i.image.as_raw())
                        } else {
                            table.buffer(id).map(|b| b.buffer.as_raw())
                        };
                        match resolved {
                            Ok(raw) => {
                                let held = before.expect("it resolved");
                                assert_eq!(held.generation, generation);
                                assert_eq!(held.is_image, want_image);
                                assert_eq!(raw, held.handle);
                            }
                            Err(miss) => {
                                assert_eq!(
                                    miss,
                                    expected_miss(before, slot, generation, Some(want_image))
                                );
                                count_miss(&mut tally, miss);
                            }
                        }
                    }
                }

                assert_eq!(table.population(), shadow.len());
                assert_eq!(table.census(), (publishes, replacements));
            }

            // Conservation, over the whole history. Everything ever published
            // is either still nameable or waiting to be destroyed --- never
            // both, never neither, never twice.
            let still_held: BTreeMap<u64, u64> =
                shadow.values().map(|h| (h.handle, h.last_use)).collect();
            let mut queued: BTreeMap<u64, u64> = BTreeMap::new();
            for retired in retire.reached(DeviceEpoch::FIRST, TimelinePoint(u64::MAX)) {
                let raw = raw_of(&retired.object);
                assert!(
                    queued.insert(raw, retired.last_use.0).is_none(),
                    "handle {raw} was queued for destruction twice"
                );
                assert!(
                    !still_held.contains_key(&raw),
                    "handle {raw} is queued for destruction and still nameable"
                );
                // The point a destroy waits for. Too early is a use-after-free
                // and too late is a leak, and only the history knows which one
                // this is.
                assert_eq!(
                    Some(&retired.last_use.0),
                    departed.get(&raw),
                    "handle {raw} was retired against the wrong last use"
                );
            }
            assert_eq!(retire.outstanding(), 0);
            for handle in &minted {
                let where_it_went = still_held.contains_key(handle) || queued.contains_key(handle);
                assert!(where_it_went, "handle {handle} was published and then lost");
            }
            assert_eq!(minted.len(), still_held.len() + queued.len());
        }

        // Floors per path. One aggregate would let a path go undriven and
        // still read as covered.
        assert!(tally.published > 5_000, "{}", tally.published);
        assert!(tally.occupied > 2_000, "{}", tally.occupied);
        assert!(tally.rewound > 100, "{}", tally.rewound);
        assert!(tally.replaced > 700, "{}", tally.replaced);
        assert!(tally.deleted > 800, "{}", tally.deleted);
        assert!(tally.used > 1_000, "{}", tally.used);
        assert!(tally.used_stale > 3_000, "{}", tally.used_stale);
        assert!(tally.drained > 1_500, "{}", tally.drained);
        assert!(tally.unknown > 8_000, "{}", tally.unknown);
        assert!(tally.stale > 1_000, "{}", tally.stale);
        assert!(tally.wrong_kind > 1_000, "{}", tally.wrong_kind);
    }

    /// The raw handle, which is the same number for either kind because the
    /// fixtures mint them from one counter.
    fn raw_of(native: &Native) -> u64 {
        match native {
            Native::Buffer(buffer) => buffer.buffer.as_raw(),
            Native::Image(image) => image.image.as_raw(),
        }
    }

    /// What the shadow says a name should miss with, derived from the calls'
    /// arguments and never from the table.
    fn expected_miss(
        held: Option<Held>,
        slot: u32,
        generation: u64,
        want_image: Option<bool>,
    ) -> Miss {
        let Some(held) = held else {
            return Miss::Unknown {
                slot: ObjectListRef(slot),
            };
        };
        if held.generation != generation {
            return Miss::Stale {
                slot: ObjectListRef(slot),
                held: SlotGeneration(held.generation),
                named: SlotGeneration(generation),
            };
        }
        let wanted = want_image.expect("a kind question is the only miss left");
        Miss::WrongKind {
            slot: ObjectListRef(slot),
            held: if held.is_image {
                Kind::Image
            } else {
                Kind::Buffer
            },
            wanted: if wanted { Kind::Image } else { Kind::Buffer },
        }
    }

    fn count_miss(tally: &mut Tally, miss: Miss) {
        match miss {
            Miss::Unknown { .. } => tally.unknown += 1,
            Miss::Stale { .. } => tally.stale += 1,
            Miss::WrongKind { .. } => tally.wrong_kind += 1,
        }
    }
}
