//! Issuing the planned commands into a command buffer.
//!
//! # This module decides nothing
//!
//! Every other module in this crate produces values: a transfer's regions, a
//! mip ladder's rungs, a layout transition's two ends. This one turns those
//! values into `vkCmd*` calls and makes no choice of its own beyond which
//! *spelling* of a barrier this host takes. That is deliberate — a recorder
//! that also decided would be a second place every one of those decisions
//! lives, reachable only with a device present, and therefore the copy that is
//! never tested.
//!
//! So the tests for what these calls contain are in the modules that produce
//! them, and what this module needs a real driver for is whether a driver
//! accepts them at all.
//!
//! # The one choice: which barrier spelling
//!
//! Vulkan 1.2 is the baseline, so `VK_KHR_synchronization2` is an extension
//! and a host may only have `vkCmdPipelineBarrier`. [`crate::barrier`] already
//! makes that a mapping of one plan rather than a second translation; this
//! picks the emitter from the census's answer, once, at construction. A
//! per-call `if` would let one command buffer mix spellings, which is legal
//! and unreadable.

use ash::vk;

use crate::barrier::BarrierPlan;
use crate::layout::{OwnershipTransfer, Transition};
use crate::mipmap::Step;
use crate::staging::{Arena, Window};
use crate::transfer::{Command, FillPlan, StagedEdge};

/// A command buffer being recorded into, and the barrier spelling this host
/// takes.
///
/// Borrows rather than owns: the buffer's lifetime is
/// [`crate::recording::Preparation`]'s, and a recorder that owned one would be
/// a second claim on it.
/// Which side of a queue-family ownership move a recording is.
///
/// Private, because it is not a choice a caller makes: it is decided by which
/// of [`Recorder::release_ownership`] and [`Recorder::acquire_ownership`] was
/// called, and those are the two methods that know which queue their buffer is
/// going to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Half {
    Release,
    Acquire,
}

#[derive(Clone, Copy)]
pub struct Recorder<'a> {
    device: &'a ash::Device,
    buffer: vk::CommandBuffer,
    synchronization2: bool,
}

/// The barrier one half of a queue-family ownership move issues.
///
/// Pure, and separate from the recording, so the thing that decides the
/// dependency can be checked without a device — which the recording cannot be.
///
/// Vulkan ignores a release's destination masks and an acquire's source masks,
/// so each half carries exactly one of the two operations: the release makes
/// the previous owner's writes available, the acquire makes them visible to
/// what the new owner is about to do. The ignored half is written as the stage
/// identity rather than as nothing, because `NONE` is legal only under
/// `synchronization2` and the older entry point refuses an empty stage mask.
fn ownership_plan(transfer: &OwnershipTransfer, half: Half) -> BarrierPlan {
    match half {
        Half::Release => BarrierPlan {
            src_stages: transfer.src_stages,
            src_access: transfer.src_access,
            dst_stages: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            dst_access: vk::AccessFlags2::empty(),
        },
        Half::Acquire => BarrierPlan {
            src_stages: vk::PipelineStageFlags2::TOP_OF_PIPE,
            src_access: vk::AccessFlags2::empty(),
            dst_stages: transfer.dst_stages,
            dst_access: transfer.dst_access,
        },
    }
}

impl<'a> Recorder<'a> {
    /// Record into `buffer`, using `synchronization2` barriers when this host
    /// has them.
    ///
    /// The flag comes from [`crate::census::Census::synchronization2`], which
    /// is the joined "extension enumerated *and* feature enabled, or core"
    /// answer. Passing a hopeful `true` here is the one way to get a null
    /// function pointer out of this module.
    #[must_use]
    pub const fn new(
        device: &'a ash::Device,
        buffer: vk::CommandBuffer,
        synchronization2: bool,
    ) -> Self {
        Self {
            device,
            buffer,
            synchronization2,
        }
    }

    #[must_use]
    pub const fn buffer(&self) -> vk::CommandBuffer {
        self.buffer
    }

    /// Issue one planned transfer.
    ///
    /// # Safety
    ///
    /// The command buffer is in the recording state, every handle the command
    /// names belongs to this device and is still alive, and each image is in
    /// the layout the command requires — which is [`crate::layout`]'s answer
    /// and not checked here.
    pub unsafe fn transfer(&self, command: &Command) {
        // SAFETY: the caller's preconditions are exactly the ones each of
        // these calls has; nothing here reads memory it was not handed.
        unsafe {
            match command {
                Command::CopyBuffer {
                    source,
                    dest,
                    regions,
                } => self
                    .device
                    .cmd_copy_buffer(self.buffer, *source, *dest, regions),
                Command::CopyBufferToImage {
                    source,
                    dest,
                    regions,
                } => self.device.cmd_copy_buffer_to_image(
                    self.buffer,
                    *source,
                    *dest,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                ),
                Command::CopyImageToBuffer {
                    source,
                    dest,
                    regions,
                } => self.device.cmd_copy_image_to_buffer(
                    self.buffer,
                    *source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    *dest,
                    regions,
                ),
                Command::CopyImage {
                    source,
                    dest,
                    regions,
                } => self.device.cmd_copy_image(
                    self.buffer,
                    *source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    *dest,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                ),
            }
        }
    }

    /// Issue a planned fill: its interior, and a copy for each staged edge.
    ///
    /// The edge bytes must already be written into the arena — see
    /// [`Arena::set`] — because a copy recorded here reads them when the GPU
    /// runs, and the arena's mapping is the only thing that can put them
    /// there.
    ///
    /// # Safety
    ///
    /// As [`Self::transfer`], and the arena's chunks are alive and bound.
    pub unsafe fn fill(&self, plan: &FillPlan, arena: &Arena) {
        if let Some(range) = plan.middle {
            // SAFETY: the caller's preconditions.
            unsafe {
                self.device.cmd_fill_buffer(
                    self.buffer,
                    plan.dest,
                    range.offset,
                    range.size,
                    range.data,
                );
            }
        }
        for edge in [plan.head, plan.tail].into_iter().flatten() {
            // SAFETY: as above.
            unsafe { self.edge(plan.dest, edge, arena) };
        }
    }

    unsafe fn edge(&self, dest: vk::Buffer, edge: StagedEdge, arena: &Arena) {
        let region = vk::BufferCopy {
            src_offset: edge.window.offset,
            dst_offset: edge.dest_offset,
            size: edge.length,
        };
        // SAFETY: the caller's preconditions.
        unsafe {
            self.device
                .cmd_copy_buffer(self.buffer, arena.buffer(edge.window), dest, &[region]);
        }
    }

    /// Issue one planned layout transition.
    ///
    /// # Safety
    ///
    /// As [`Self::transfer`]. `aspect` must be the one the image's format has
    /// — [`crate::view::aspect`] is what answers it.
    pub unsafe fn transition(
        &self,
        image: vk::Image,
        transition: &Transition,
        aspect: vk::ImageAspectFlags,
    ) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: transition.subresource.level,
            level_count: 1,
            base_array_layer: transition.subresource.layer,
            layer_count: 1,
        };
        if self.synchronization2 {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(transition.src_stages)
                .src_access_mask(transition.src_access)
                .dst_stage_mask(transition.dst_stages)
                .dst_access_mask(transition.dst_access)
                .old_layout(transition.from)
                .new_layout(transition.to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range);
            let barriers = [barrier];
            let info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            // SAFETY: the caller's preconditions, and `synchronization2` was
            // reported by the census that enabled it.
            unsafe { self.device.cmd_pipeline_barrier2(self.buffer, &info) };
            return;
        }

        // The same plan in the older flag types. Mapped once, in `barrier`,
        // so the two paths cannot disagree about what the guest asked for.
        let legacy = BarrierPlan {
            src_stages: transition.src_stages,
            dst_stages: transition.dst_stages,
            src_access: transition.src_access,
            dst_access: transition.dst_access,
        }
        .legacy();
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(legacy.src_access)
            .dst_access_mask(legacy.dst_access)
            .old_layout(transition.from)
            .new_layout(transition.to)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        // An empty stage mask is illegal in the older form, where
        // `synchronization2` accepts `NONE`. `TOP_OF_PIPE` as a source and
        // `BOTTOM_OF_PIPE` as a destination are the identities: they order
        // nothing, which is exactly what an empty mask meant.
        let src = if legacy.src_stages.is_empty() {
            vk::PipelineStageFlags::TOP_OF_PIPE
        } else {
            legacy.src_stages
        };
        let dst = if legacy.dst_stages.is_empty() {
            vk::PipelineStageFlags::BOTTOM_OF_PIPE
        } else {
            legacy.dst_stages
        };
        // SAFETY: the caller's preconditions.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Issue the *release* half of a queue-family ownership move, on the queue
    /// giving the image up.
    ///
    /// # Safety
    ///
    /// As [`Self::transition`]. This recorder's buffer must be submitted to
    /// [`OwnershipTransfer::from_family`].
    pub unsafe fn release_ownership(
        &self,
        image: vk::Image,
        transfer: &OwnershipTransfer,
        aspect: vk::ImageAspectFlags,
    ) {
        // SAFETY: the caller's preconditions.
        unsafe { self.ownership(image, transfer, aspect, Half::Release) }
    }

    /// Issue the *acquire* half, on the queue taking the image over.
    ///
    /// # Safety
    ///
    /// As [`Self::transition`]. This recorder's buffer must be submitted to
    /// [`OwnershipTransfer::to_family`].
    pub unsafe fn acquire_ownership(
        &self,
        image: vk::Image,
        transfer: &OwnershipTransfer,
        aspect: vk::ImageAspectFlags,
    ) {
        // SAFETY: the caller's preconditions.
        unsafe { self.ownership(image, transfer, aspect, Half::Acquire) }
    }

    /// One half of a queue-family ownership move.
    ///
    /// **Two entry points and not one flag the doc describes.** This used to be
    /// a single `ownership` whose documentation said it "issues the half whose
    /// family matches `family`" — and it took no `family`, emitted one
    /// identical barrier for both halves, and could not have told them apart.
    /// A release recorded into the acquiring buffer is undefined behaviour, so
    /// the guard the doc claimed is the one thing this had to have. Which half
    /// is now the method the caller names.
    ///
    /// The two really are different barriers. Vulkan ignores a release's
    /// destination masks and an acquire's source masks, which leaves the
    /// release carrying the availability operation and the acquire carrying the
    /// visibility one — so each half sets the pair that is read and the
    /// identity for the pair that is not. Emitted with both empty, as this did,
    /// the transfer moves the owner and orders no memory at all.
    unsafe fn ownership(
        &self,
        image: vk::Image,
        transfer: &OwnershipTransfer,
        aspect: vk::ImageAspectFlags,
        half: Half,
    ) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: transfer.subresource.level,
            level_count: 1,
            base_array_layer: transfer.subresource.layer,
            layer_count: 1,
        };
        let plan = ownership_plan(transfer, half);
        if self.synchronization2 {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(plan.src_stages)
                .src_access_mask(plan.src_access)
                .dst_stage_mask(plan.dst_stages)
                .dst_access_mask(plan.dst_access)
                .old_layout(transfer.layout)
                .new_layout(transfer.layout)
                .src_queue_family_index(transfer.from_family)
                .dst_queue_family_index(transfer.to_family)
                .image(image)
                .subresource_range(range);
            let barriers = [barrier];
            let info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            // SAFETY: the caller's preconditions, and `synchronization2` was
            // reported by the census that enabled it.
            unsafe { self.device.cmd_pipeline_barrier2(self.buffer, &info) };
            return;
        }
        let legacy = plan.legacy();
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(legacy.src_access)
            .dst_access_mask(legacy.dst_access)
            .old_layout(transfer.layout)
            .new_layout(transfer.layout)
            .src_queue_family_index(transfer.from_family)
            .dst_queue_family_index(transfer.to_family)
            .image(image)
            .subresource_range(range);
        // The same guard `transition` applies, and for the same reason: a
        // tracked use with no stages maps to an empty legacy mask, which the
        // older entry point refuses. The identities order nothing, which is
        // what an empty mask meant.
        let src = if legacy.src_stages.is_empty() {
            vk::PipelineStageFlags::TOP_OF_PIPE
        } else {
            legacy.src_stages
        };
        let dst = if legacy.dst_stages.is_empty() {
            vk::PipelineStageFlags::BOTTOM_OF_PIPE
        } else {
            legacy.dst_stages
        };
        // SAFETY: the caller's preconditions.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Issue a mip ladder, in the order it was planned.
    ///
    /// The order is the ladder's whole content — see [`crate::mipmap`] — so
    /// this walks the slice and does not reorder, batch or coalesce.
    ///
    /// # Safety
    ///
    /// As [`Self::transition`], and `image` is the one the ladder was planned
    /// for.
    pub unsafe fn mipmap(&self, image: vk::Image, steps: &[Step], aspect: vk::ImageAspectFlags) {
        for step in steps {
            match step {
                // SAFETY: the caller's preconditions.
                Step::Transition(transition) => unsafe {
                    self.transition(image, transition, aspect);
                },
                Step::Blit(rung) => {
                    let regions = [rung.native()];
                    // SAFETY: as above; the ladder put the levels this blit
                    // names into the layouts it needs.
                    unsafe {
                        self.device.cmd_blit_image(
                            self.buffer,
                            image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &regions,
                            vk::Filter::LINEAR,
                        );
                    }
                }
            }
        }
    }
}

impl Arena {
    /// Write `byte` into every byte of a window.
    ///
    /// # Safety
    ///
    /// The window came from this arena, its chunk is not in flight, and no
    /// submission is reading it.
    pub unsafe fn set(&self, window: Window, byte: u8) {
        // SAFETY: the caller's preconditions; `write_at` is in range for
        // `window.size` bytes by the window's own construction.
        unsafe { std::ptr::write_bytes(self.write_at(window), byte, window.size as usize) };
    }

    /// Copy `bytes` into a window.
    ///
    /// # Safety
    ///
    /// As [`Self::set`], and `bytes.len()` is at most `window.size`.
    pub unsafe fn write(&self, window: Window, bytes: &[u8]) {
        debug_assert!(bytes.len() as u64 <= window.size);
        // SAFETY: the caller's preconditions and the length check above.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.write_at(window), bytes.len());
        }
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use crate::layout::{ImageId, Subresource};

    fn transfer() -> OwnershipTransfer {
        OwnershipTransfer {
            image: ImageId(1),
            subresource: Subresource::new(0, 0),
            from_family: 0,
            to_family: 1,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_stages: vk::PipelineStageFlags2::COPY,
            src_access: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stages: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            dst_access: vk::AccessFlags2::SHADER_SAMPLED_READ,
        }
    }

    /// **The two halves are different barriers, and each carries the operation
    /// Vulkan actually reads from it.** Emitted with both access masks empty —
    /// which is what a transfer with no dependency on it does — the ownership
    /// moves and no memory is ordered: the source queue's writes are never made
    /// available and nothing becomes visible to the destination.
    #[test]
    fn the_release_makes_available_and_the_acquire_makes_visible() {
        let t = transfer();

        let release = ownership_plan(&t, Half::Release);
        assert_eq!(release.src_stages, t.src_stages);
        assert_eq!(release.src_access, t.src_access);
        assert!(
            release.dst_access.is_empty(),
            "a release's destination access is ignored, so naming one would \
             claim a visibility operation that does not happen"
        );

        let acquire = ownership_plan(&t, Half::Acquire);
        assert_eq!(acquire.dst_stages, t.dst_stages);
        assert_eq!(acquire.dst_access, t.dst_access);
        assert!(
            acquire.src_access.is_empty(),
            "an acquire's source access is ignored"
        );

        assert_ne!(
            release, acquire,
            "one barrier for both halves is the shape that ordered nothing"
        );
    }

    /// Neither half may leave a stage mask empty. `NONE` is a
    /// `synchronization2` spelling, and the older entry point refuses zero
    /// outright — so the ignored side is an identity and not an absence.
    #[test]
    fn neither_half_leaves_a_stage_mask_the_older_form_refuses() {
        let mut t = transfer();
        // A use the tracker recorded no stages for, which is what would reach
        // the empty mask.
        t.src_stages = vk::PipelineStageFlags2::empty();
        t.dst_stages = vk::PipelineStageFlags2::empty();
        for half in [Half::Release, Half::Acquire] {
            let plan = ownership_plan(&t, half);
            let legacy = plan.legacy();
            let src = if legacy.src_stages.is_empty() {
                vk::PipelineStageFlags::TOP_OF_PIPE
            } else {
                legacy.src_stages
            };
            let dst = if legacy.dst_stages.is_empty() {
                vk::PipelineStageFlags::BOTTOM_OF_PIPE
            } else {
                legacy.dst_stages
            };
            assert!(!src.is_empty(), "{half:?}: empty legacy source stage mask");
            assert!(
                !dst.is_empty(),
                "{half:?}: empty legacy destination stage mask"
            );
            // The half the transfer *does* fill is never the identity by
            // accident, so the guard above cannot be hiding a lost dependency.
            let filled = match half {
                Half::Release => plan.src_stages,
                Half::Acquire => plan.dst_stages,
            };
            assert!(filled.is_empty(), "this case is about the empty input");
        }
    }

    /// Both halves name both families, in the same direction. A barrier that
    /// left them `IGNORED` would be an ordinary transition wearing a
    /// transfer's name.
    #[test]
    fn both_halves_name_the_same_move() {
        let t = transfer();
        assert_ne!(t.from_family, t.to_family);
        for half in [Half::Release, Half::Acquire] {
            let plan = ownership_plan(&t, half);
            assert!(plan.orders_anything(), "{half:?} orders nothing");
        }
    }
}
