//! The rail's half of the architecture plan's structural zeros: what a warm
//! draw costs the heap allocator, and what an unchanged one costs at all.
//!
//! # Why this is measured rather than reasoned about
//!
//! Two of the plan's required zeros land here — "heap allocations per
//! steady-state draw" and "full binding-table rebuilds on an unchanged draw" —
//! and both fail the same way: not as a visible regression but as a percent or
//! two of drain duty spread evenly across a profile, because a helper on a
//! per-draw path started returning a `Vec`. No single line gets slower, so
//! nobody bisects to it.
//!
//! The instrument is [`reims_vgpu_testkit::allocations`], shared with the
//! model's own suite. Its counter is per thread and off unless a measurement
//! asks for it, so these run alongside everything else.

use ash::vk;
use ash::vk::Handle;
use reims_vgpu_core::bind::{BufferBinding, ObjectBinding};
use reims_vgpu_core::blit::{BlitOp, ImagePitch, Origin3, Size3, SpanOrigin, TexturePoint};
use reims_vgpu_core::identity::{
    DeviceEpoch, ObjectListRef, ResourceId, SessionGeneration, SlotGeneration, TimelinePoint,
};
use reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM;
use reims_vgpu_core::retire::{Lifetime, NativeRetirement};
use reims_vgpu_core::texture_shape::{TextureKind, TextureShape, TextureUsage};
use reims_vgpu_testkit::allocations::{measure, Counting};
use reims_vgpu_vulkan::bindings::BindingTable;
use reims_vgpu_vulkan::buffer::{BufferPlan, EVERY_CLASS};
use reims_vgpu_vulkan::descriptor::SetRing;
use reims_vgpu_vulkan::image::ImagePlan;
use reims_vgpu_vulkan::resident::{Native, NativeBuffer, NativeImage, Residency};
use reims_vgpu_vulkan::transfer;

#[global_allocator]
static ALLOCATOR: Counting = Counting::new();

fn id(slot: u32) -> ResourceId {
    ResourceId {
        slot: ObjectListRef(slot),
        generation: SlotGeneration(1),
    }
}

fn buffer(slot: u32) -> Option<BufferBinding> {
    Some(BufferBinding {
        buffer: Some(id(slot)),
        offset: 0,
        stride: None,
    })
}

fn texture(slot: u32) -> Option<ObjectBinding> {
    Some(ObjectBinding {
        object: Some(id(slot)),
        lod_clamps: None,
    })
}

/// The draw the plan's zero is actually about: nothing was rebound since the
/// last one.
///
/// A guest that draws the same material twice in a row rebinds nothing between
/// the two, and the second draw must cost nothing at all — not a rebuilt table,
/// not a descriptor set, not a trip into the allocator. The table answers that
/// from its dirty set, so the whole of the second draw's binding work is
/// [`BindingTable::is_clean`].
#[test]
fn an_unchanged_draw_touches_neither_the_table_nor_the_allocator() {
    let mut table = BindingTable::new(31, 31, 16);
    for slot in 0..8 {
        table.bind_buffer(slot as usize, buffer(slot));
        table.bind_texture(slot as usize, texture(100 + slot));
    }
    let first = table.take_dirty();
    assert_eq!(first.len(), 16, "the first draw writes what was bound");

    let (clean, allocations) = measure(|| table.is_clean());
    assert!(clean, "nothing was rebound, so nothing is owed");
    assert_eq!(
        allocations, 0,
        "asking whether a draw owes any descriptor writes builds nothing"
    );

    // And the bindings are still readable without cost, which is what makes
    // "unchanged" mean the contents survived rather than that they were
    // dropped.
    let (held, none) = measure(|| table.buffer(3));
    assert_eq!(held, buffer(3));
    assert_eq!(none, 0);
}

/// A draw that did rebind pays for the slots it changed, and the cost does not
/// scale with the size of the table.
///
/// The distinction is the whole value of the dirty set: a table with 31 buffer
/// slots of which two moved must cost what two cost, and a table with 512 of
/// which two moved must cost the same. A rebuild would make both cost the
/// table.
#[test]
fn a_rebinding_draw_pays_for_its_slots_and_not_for_the_table() {
    let emit_cost = |slots: u32| -> usize {
        let mut table = BindingTable::new(slots as usize, slots as usize, 16);
        for slot in 0..slots {
            table.bind_buffer(slot as usize, buffer(slot));
        }
        let _ = table.take_dirty();
        let (dirty, allocations) = measure(|| {
            table.bind_buffer(1, buffer(9_000));
            table.bind_buffer(2, buffer(9_001));
            table.take_dirty()
        });
        assert_eq!(dirty.len(), 2, "two slots moved");
        allocations
    };

    let narrow = emit_cost(31);
    let wide = emit_cost(512);
    assert_eq!(
        narrow, wide,
        "a two-slot rebind costs {narrow} on a 31-slot table and {wide} on a \
         512-slot one; the cost is following the table rather than the change"
    );
    // The three are `take_dirty`'s replacement masks, one per binding kind,
    // sized to the table so that the binds after it never resize them. They
    // are per *emitting* draw and not per slot, which is why the number does
    // not move between a 31-slot table and a 512-slot one --- and why an
    // unchanged draw, which never calls this, pays none of them.
    assert_eq!(
        narrow, 3,
        "one replacement mask per binding kind and nothing else"
    );
}

/// The descriptor ring's steady state: a set taken, submitted, and recycled,
/// frame after frame, over a fixed ring.
///
/// Nothing here may allocate. The ring's whole reason for existing is that the
/// sets are made once and reused, so an allocation on this path would mean a
/// per-frame heap cost hiding behind a pool that was supposed to remove one.
#[test]
fn taking_and_recycling_a_descriptor_set_allocates_nothing() {
    let mut ring = SetRing::new(4);
    // Warm the ring: every set has been through the cycle at least once, so
    // whatever it holds per set is already held.
    for frame in 1..=16u64 {
        let emission = ring.emit().expect("a free set");
        assert!(emission.whole() || frame > 1);
        ring.submitted(TimelinePoint(frame));
        ring.recycle(TimelinePoint(frame));
    }

    let ((), allocations) = measure(|| {
        for frame in 17..=32u64 {
            let emission = ring.emit().expect("a free set");
            let _ = emission.set();
            ring.submitted(TimelinePoint(frame));
            ring.recycle(TimelinePoint(frame));
        }
    });
    assert_eq!(
        allocations, 0,
        "sixteen frames of take-submit-recycle over a fixed ring"
    );

    // The clean draw's half of the same steady state. A draw with nothing
    // dirty writes no descriptors and binds the holder, and it goes through
    // the ring rather than around it --- see `SetRing::bind` --- so it is on
    // the hottest path there is and has to cost nothing.
    let (bound, allocations) = measure(|| {
        let mut last = None;
        for _ in 0..64 {
            last = ring.bind();
        }
        last
    });
    assert!(
        bound.is_some(),
        "something has been emitted, so one is held"
    );
    assert_eq!(
        allocations, 0,
        "sixty-four clean draws binding the holder they already have"
    );
}

/// Walking the slots an emission owes costs nothing.
///
/// This is the door the descriptor emitter walks, once per binding kind, on
/// every draw that rebound anything --- and the walk carries no state past a
/// word index and the residue of that word, so there is nothing for it to
/// build. It returned a `Vec` because it was only ever driven from tests, and
/// the emitter would have inherited three allocations per rebinding draw the
/// moment it was written. Measured before the emitter exists rather than after,
/// which is the only time the number is free to fix.
#[test]
fn walking_the_slots_an_emission_owes_allocates_nothing() {
    let mut table = BindingTable::new(512, 512, 16);
    for slot in 0..512u32 {
        table.bind_buffer(slot as usize, buffer(slot));
        table.bind_texture(slot as usize, texture(10_000 + slot));
    }
    for slot in 0..16u32 {
        table.bind_sampler(slot as usize, texture(20_000 + slot));
    }
    // Take the dirty set outside the measurement: its three replacement masks
    // are a separate, already-measured cost, and what is under test is the
    // walk over what it reported.
    let dirty = table.take_dirty();
    assert_eq!(dirty.len(), 512 + 512 + 16, "everything was just bound");

    let (visited, allocations) = measure(|| {
        let mut visited = 0usize;
        for slot in dirty.buffers.slots() {
            visited += slot;
        }
        for slot in dirty.textures.slots() {
            visited += slot;
        }
        for slot in dirty.samplers.slots() {
            visited += slot;
        }
        visited
    });
    // Two full 0..512 runs and one 0..16, so the walk really did visit every
    // slot --- an empty iterator does not allocate either, and a measurement
    // that did not check this would pass with the `Vec` restored and the mask
    // drained.
    assert_eq!(visited, 2 * (511 * 512 / 2) + 15 * 16 / 2);
    assert_eq!(
        allocations, 0,
        "the walk over a dirty set builds nothing of its own"
    );
}

/// Planning a transfer costs the allocator nothing.
///
/// Four of the five transfer shapes produce exactly one region, and a frame
/// that streams — a texture upload, a readback, a buffer copy — issues them by
/// the dozen. A `Vec` to carry a value the plan already had is one trip into
/// the allocator per record on that path, which is the per-record heap cost
/// the plan's structural zeros rule out.
///
/// A slice span of more than one level is the one shape that genuinely has
/// several regions, and it is measured too so the number is the level count
/// and not the level count plus a surprise.
#[test]
fn planning_a_transfer_allocates_nothing_per_single_region_record() {
    let mut residency = Residency::new();
    let mut retire = NativeRetirement::new();
    let lifetime = Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST);
    let shape = TextureShape {
        kind: TextureKind::D2Array.ordinal(),
        width: 64,
        height: 32,
        depth: 1,
        mipmap_level_count: 4,
        sample_count: 1,
        array_length: 3,
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        usage: TextureUsage::SHADER_READ,
    }
    .checked()
    .expect("a valid declaration");
    for (slot, native) in [
        (
            1u32,
            Native::Buffer(NativeBuffer {
                buffer: vk::Buffer::from_raw(0xB1),
                memory: vk::DeviceMemory::from_raw(0xB1),
                plan: BufferPlan {
                    size: 1 << 20,
                    usage: EVERY_CLASS,
                    aliased: false,
                },
            }),
        ),
        (
            2,
            Native::Image(NativeImage {
                texture: shape,
                image: vk::Image::from_raw(0x1A),
                memory: vk::DeviceMemory::from_raw(0x1A),
                plan: ImagePlan {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent: vk::Extent3D {
                        width: 64,
                        height: 32,
                        depth: 1,
                    },
                    mip_levels: 4,
                    array_layers: 3,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::SAMPLED,
                    flags: vk::ImageCreateFlags::empty(),
                },
                sampled: Default::default(),
                attachments: Vec::new(),
            }),
        ),
        (
            3,
            Native::Image(NativeImage {
                texture: shape,
                image: vk::Image::from_raw(0x1B),
                memory: vk::DeviceMemory::from_raw(0x1B),
                plan: ImagePlan {
                    image_type: vk::ImageType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent: vk::Extent3D {
                        width: 64,
                        height: 32,
                        depth: 1,
                    },
                    mip_levels: 4,
                    array_layers: 3,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::SAMPLED,
                    flags: vk::ImageCreateFlags::empty(),
                },
                sampled: Default::default(),
                attachments: Vec::new(),
            }),
        ),
    ] {
        residency
            .publish(id(slot), lifetime, native, &mut retire)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
    }

    let point = |slot: u32| TexturePoint {
        texture: id(slot),
        slice: 0,
        level: 0,
        origin: Origin3 { x: 0, y: 0, z: 0 },
    };
    let flat = ImagePitch {
        bytes_per_row: 0,
        bytes_per_image: 0,
    };
    let size = Size3 {
        width: 8,
        height: 8,
        depth: 1,
    };
    let from_top = SpanOrigin {
        texture: id(2),
        base_slice: 0,
        base_level: 0,
    };
    for op in [
        BlitOp::BufferToBuffer {
            source: id(1),
            source_offset: 0,
            dest: id(1),
            dest_offset: 4096,
            size: 256,
        },
        BlitOp::BufferToTexture {
            source: id(1),
            source_offset: 0,
            source_pitch: flat,
            size,
            dest: point(2),
            options: Default::default(),
        },
        BlitOp::TextureToBuffer {
            source: point(2),
            size,
            dest: id(1),
            dest_offset: 0,
            dest_pitch: flat,
            options: Default::default(),
        },
        BlitOp::TextureRegion {
            source: point(2),
            dest: point(3),
            size,
            options: Default::default(),
        },
        BlitOp::TextureSlices {
            source: from_top,
            dest: SpanOrigin {
                texture: id(3),
                ..from_top
            },
            slice_count: 3,
            level_count: 1,
        },
    ] {
        let kind = op.kind();
        let (planned, allocations) = measure(|| transfer::plan(&op, &residency));
        let planned = planned
            .unwrap_or_else(|e| panic!("{kind:?}: {e}"))
            .expect("native work to record");
        assert_eq!(planned.region_count(), 1, "{kind:?} names one region");
        assert_eq!(allocations, 0, "{kind:?} planned through the allocator");
    }

    // The one shape that really does have several. Its cost is the one `Vec`
    // holding them, and nothing beside it.
    let op = BlitOp::TextureSlices {
        source: from_top,
        dest: SpanOrigin {
            texture: id(3),
            ..from_top
        },
        slice_count: 3,
        level_count: 4,
    };
    let (planned, allocations) = measure(|| transfer::plan(&op, &residency));
    assert_eq!(
        planned
            .expect("plannable")
            .expect("native work to record")
            .region_count(),
        4
    );
    assert_eq!(allocations, 1, "one allocation for the four regions");
}
