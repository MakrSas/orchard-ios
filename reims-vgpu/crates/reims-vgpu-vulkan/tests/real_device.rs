//! The rail's modules, composed on whatever GPU this machine has.
//!
//! Every module in this crate deliberately splits its fallible bookkeeping from
//! the Vulkan handles, so each one is unit-tested with no device present. That
//! leaves exactly one thing untested: whether the pieces are true *about a
//! driver*. A ring that recycles a slot when its timeline point is reached is
//! correct arithmetic; whether the point it was given is the one the GPU
//! actually signals is a fact about a submission.
//!
//! So this test does the smallest thing that can be wrong end to end. It fills
//! a buffer on the GPU, waits on the timeline the submission signalled, and
//! reads the bytes back. A rail that got the timeline value, the queue family,
//! the memory type or the coherence rule wrong fails here and passes every
//! unit test in the crate.
//!
//! A machine with no Vulkan prints that it had none and asserts nothing
//! further. It says which of the two happened, so a silently skipped run is
//! distinguishable from a passing one in the output.

use ash::vk;
use reims_vgpu_core::blend::ColorAttachmentState;
use reims_vgpu_core::blit::{BufferSpan, FillPattern};
use reims_vgpu_core::identity::DeviceEpoch as EpochId;
use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SessionGeneration, SlotGeneration};
use reims_vgpu_core::pass::{LoadAction, PassDescriptor, RenderTargetExtent, StoreAction};
use reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM;
use reims_vgpu_core::retire::{Lifetime, NativeRetirement};
use reims_vgpu_core::texture_shape::{TextureKind, TextureShape, TextureUsage};
use reims_vgpu_core::topology::PrimitiveType;
use reims_vgpu_core::vertex_format::VertexFormat;
use reims_vgpu_vulkan::blend;
use reims_vgpu_vulkan::buffer;
use reims_vgpu_vulkan::device::DeviceEpoch;
use reims_vgpu_vulkan::host::VulkanHost;
use reims_vgpu_vulkan::image;
use reims_vgpu_vulkan::layout;
use reims_vgpu_vulkan::memory::{select_memory_type, MappedMemoryKind, MemoryClass};
use reims_vgpu_vulkan::mipmap;
use reims_vgpu_vulkan::pass;
use reims_vgpu_vulkan::pipeline;
use reims_vgpu_vulkan::placement::Route;
use reims_vgpu_vulkan::pools::WorkerPool;
use reims_vgpu_vulkan::raster;
use reims_vgpu_vulkan::record;
use reims_vgpu_vulkan::renderpass;
use reims_vgpu_vulkan::resident;
use reims_vgpu_vulkan::staging;
use reims_vgpu_vulkan::timeline::Timeline;
use reims_vgpu_vulkan::topology;
use reims_vgpu_vulkan::transfer;
use reims_vgpu_vulkan::vertex;
use reims_vgpu_vulkan::view;

/// What the GPU writes, and what the CPU has to read back.
const PATTERN: u32 = 0xABCD_EF01;
const BYTES: u64 = 256;
const DEPTH: usize = 2;
/// Generous: this is a `vkCmdFillBuffer` of 256 bytes. A timeout here is a
/// failure and not a slow machine.
const TIMEOUT_NS: u64 = 5_000_000_000;

#[test]
fn a_filled_buffer_reads_back_after_the_timeline_this_rail_reserved() {
    let Ok(host) = VulkanHost::open("reims-vgpu-vulkan integration") else {
        println!("no real device: nothing to compose");
        return;
    };
    println!("composing on: {}", host.report_line());

    let census = host.census();
    let mut epoch = DeviceEpoch::create(
        host.instance(),
        host.physical_device(),
        census,
        EpochId::FIRST,
    )
    .expect("the driver refused a set its own census admitted");

    // One owner for one `VkQueue`, taken from this epoch's ledger.
    let family = census.queues().universal().index;
    let owner = epoch
        .queues()
        .claim_in(family, 0)
        .expect("the chosen family has queue zero");
    let device = epoch.device().clone();
    // SAFETY: the family and index came from an owner this epoch handed out.
    let queue = unsafe { device.get_device_queue(owner.family(), owner.index()) };

    // The timeline the whole test is about. Zero initial value: the rail never
    // hands out zero as a point, so a retirement queued against a fresh
    // timeline cannot be collected before its work was recorded.
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore = unsafe {
        device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
            None,
        )
    }
    .expect("timeline semaphores are the support floor");
    let mut timeline = Timeline::adopt(semaphore);

    // One pool per worker, on the family the owner names — a command buffer
    // submitted to any other family is invalid usage, which is why the pool
    // carries it.
    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(owner.family())
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
    }
    .expect("a command pool on the chosen family");
    let buffers = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(DEPTH as u32),
        )
    }
    .expect("command buffers");
    let mut worker = WorkerPool::adopt(pool, owner.family(), buffers);
    assert_eq!(worker.ring().depth(), DEPTH);

    // The buffer the GPU writes and the CPU reads. `Readback` is the class
    // whose whole point is that coherence is a preference and not a
    // requirement, so which type this lands on is a real host answer rather
    // than a constant — the run prints it. On the Intel ARL iGPU this was
    // developed against it is a `HOST_CACHED` non-coherent type, so the
    // invalidate below is the arm that executes there.
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(BYTES)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .expect("a transfer destination");
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let properties = unsafe {
        host.instance()
            .get_physical_device_memory_properties(host.physical_device())
    };

    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    unsafe {
        host.instance()
            .get_physical_device_properties2(host.physical_device(), &mut properties2);
    }

    let request = census.memory().topology.request(MemoryClass::Readback);
    let pick = select_memory_type(
        &properties,
        requirements.memory_type_bits,
        &request,
        requirements.size,
        maintenance3.max_memory_allocation_size,
    )
    .expect("a readback type exists on every Vulkan device");
    println!(
        "readback type={} heap={} coherent={}",
        pick.index,
        pick.heap_index,
        MappedMemoryKind::of(&properties, pick.index).coherent
    );

    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(pick.index),
            None,
        )
    }
    .expect("the selected type allocates");
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.expect("bind");

    // Record and submit. The point comes from the rail's cursor, and the whole
    // question is whether the GPU signals that value.
    let lease = worker.ring_mut().begin().expect("a free slot");
    let command = worker.buffers()[lease.slot()];
    unsafe {
        device
            .begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin");
        device.cmd_fill_buffer(command, buffer, 0, vk::WHOLE_SIZE, PATTERN);
        device.end_command_buffer(command).expect("end");
    }

    let point = timeline.reserve();
    assert_ne!(point.0, 0, "zero is never handed out");
    let commands = [command];
    let signals = [semaphore];
    let values = [point.0];
    let mut timeline_info =
        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .command_buffers(&commands)
        .signal_semaphores(&signals)
        .push_next(&mut timeline_info);
    unsafe { device.queue_submit(queue, &[submit], vk::Fence::null()) }.expect("submit");
    worker.ring_mut().submitted(lease, point);

    // Exhaustion refuses rather than blocking or recycling a slot the GPU may
    // still be reading. One slot is in flight and one is taken; the third
    // request has nowhere to go.
    let held = worker.ring_mut().begin().expect("the second slot");
    let refused = worker
        .ring_mut()
        .begin()
        .expect_err("every slot is spoken for");
    assert_eq!(refused.depth, DEPTH);
    assert_eq!(refused.in_flight, 1, "the held lease is not in flight");
    worker.ring_mut().abandon(held);

    // Wait on the point the rail reserved, then ask the driver where the
    // timeline actually got to.
    // SAFETY: `device` is the device the semaphore was created on and it has
    // not been destroyed.
    unsafe { timeline.wait(&device, point, TIMEOUT_NS) }.expect("the GPU signalled the point");
    // SAFETY: as above.
    let reached = unsafe { timeline.poll(&device) }.expect("a forwards reading");
    assert!(
        reached.reached(point),
        "the driver signalled {reached:?}, below the reserved {point:?}"
    );

    // Which is what makes the slot recordable again.
    assert_eq!(worker.ring_mut().recycle(reached), 1);
    assert_eq!(worker.ring().in_flight(), 0);
    assert!(worker.ring().resettable());
    let after = worker.ring_mut().begin().expect("the recycled slot");
    worker.ring_mut().abandon(after);

    // And the bytes. A non-coherent mapping owes an invalidate first; the
    // memory module is what says whether this one does.
    let kind = MappedMemoryKind::of(&properties, pick.index);
    let mapped =
        unsafe { device.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty()) }
            .expect("host visible");
    if !kind.coherent {
        let range = vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { device.invalidate_mapped_memory_ranges(&[range]) }.expect("invalidate");
    }
    // SAFETY: the mapping covers `requirements.size >= BYTES` bytes, the GPU is
    // finished with them, and `u32` has no alignment requirement the driver's
    // mapping does not already meet.
    let words = unsafe { std::slice::from_raw_parts(mapped.cast::<u32>(), (BYTES / 4) as usize) };
    assert!(
        words.iter().all(|w| *w == PATTERN),
        "the GPU's fill did not reach the CPU: {:#010x} at {:?}",
        words[0],
        words.iter().position(|w| *w != PATTERN)
    );
    unsafe { device.unmap_memory(memory) };

    println!(
        "filled and read back {} words at {PATTERN:#010x}",
        words.len()
    );

    // Teardown in creation order's reverse. The epoch's own `Drop` idles the
    // device, but these are this test's objects and it gives them back itself.
    unsafe {
        device.device_wait_idle().expect("idle before teardown");
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
        device.destroy_command_pool(worker.pool(), None);
        device.destroy_semaphore(timeline.semaphore(), None);
    }
    epoch.queues().release(owner);
    // `epoch`'s own `Drop` idles and destroys the device. `device` is a clone
    // of its loader table and owns nothing, so it needs no teardown of its own.
    drop(epoch);
}

/// The image half: a decoded texture declaration reaching a real `VkImage`
/// through the query that admitted it.
///
/// The plan is pure and the admission is pure, so both are unit-tested here
/// with invented `VkImageFormatProperties`. What no unit test can check is
/// whether the tuple this rail builds is one a driver answers at all — a
/// usage bit a format does not support turns
/// `vkGetPhysicalDeviceImageFormatProperties` into
/// `ERROR_FORMAT_NOT_SUPPORTED`, and every downstream number this rail then
/// validates against would be zero.
///
/// So this asks the driver about exactly the tuple the plan names, admits the
/// plan against exactly that answer, and creates the image from exactly the
/// admitted plan. A refusal anywhere is this rail's, with both numbers in it.
#[test]
fn a_decoded_texture_becomes_an_image_the_driver_admitted() {
    let Ok(host) = VulkanHost::open("reims-vgpu-vulkan image integration") else {
        println!("no real device: nothing to allocate");
        return;
    };
    let census = host.census();
    let epoch = DeviceEpoch::create(
        host.instance(),
        host.physical_device(),
        census,
        EpochId::FIRST,
    )
    .expect("the driver refused a set its own census admitted");
    let device = epoch.device().clone();

    // A cube array with a mip chain: the shape whose layer count is the one a
    // caller deriving it itself gets wrong, and whose mip count a device is
    // entitled to cap below the pyramid.
    let declaration = TextureShape {
        kind: TextureKind::CubeArray.ordinal(),
        width: 64,
        height: 64,
        depth: 1,
        mipmap_level_count: 7,
        sample_count: 1,
        array_length: 2,
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        usage: TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
    };
    let texture = declaration
        .checked()
        .expect("a declaration the guest API admits");
    assert_eq!(texture.layers(), 12);

    let plan = image::plan(
        texture,
        vk::Format::R8G8B8A8_UNORM,
        Route::HostStaging {
            working: MemoryClass::DeviceLocal,
        },
    )
    .expect("a plannable texture");
    let query = plan.query();

    // The one thing that needs a device: what this exact combination's limits
    // are. Not the general `maxImageDimension2D`, which is the ceiling over
    // every usage and would admit tuples this one does not.
    let reported = unsafe {
        host.instance().get_physical_device_image_format_properties(
            host.physical_device(),
            query.format,
            query.image_type,
            query.tiling,
            query.usage,
            query.flags,
        )
    }
    .expect("RGBA8 sampled and color-attachable is universal on Vulkan 1.2");
    println!(
        "image tuple extent={}x{}x{} mips={} layers={} samples=0x{:x}",
        reported.max_extent.width,
        reported.max_extent.height,
        reported.max_extent.depth,
        reported.max_mip_levels,
        reported.max_array_layers,
        reported.sample_counts.as_raw(),
    );

    let admitted = plan
        .admitted(reported)
        .unwrap_or_else(|refusal| panic!("{refusal}"));
    let info = admitted.create_info();
    assert_eq!(info.array_layers, 12);
    assert_eq!(info.mip_levels, 7);
    assert!(info.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE));

    let image = unsafe { device.create_image(&info, None) }
        .expect("the driver takes what its own reported properties admitted");

    // And it is allocatable: an image plan that cannot be backed is a plan
    // that passed every check and produces nothing.
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let properties = unsafe {
        host.instance()
            .get_physical_device_memory_properties(host.physical_device())
    };
    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    unsafe {
        host.instance()
            .get_physical_device_properties2(host.physical_device(), &mut properties2);
    }
    let pick = select_memory_type(
        &properties,
        requirements.memory_type_bits,
        &census.memory().topology.request(MemoryClass::DeviceLocal),
        requirements.size,
        maintenance3.max_memory_allocation_size,
    )
    .expect("a device-local type exists for a sampled colour image");
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(pick.index),
            None,
        )
    }
    .expect("the selected type allocates");
    unsafe { device.bind_image_memory(image, memory, 0) }.expect("bind");
    println!(
        "image bytes={} type={} for {} subresources",
        requirements.size,
        pick.index,
        texture.subresources()
    );

    // A device that caps this tuple below the declaration refuses with both
    // numbers rather than clamping. Asked here against the driver's own answer
    // narrowed by one, so the assertion is about this rail's comparison and not
    // about any particular host's limits.
    let one_layer_short = vk::ImageFormatProperties {
        max_array_layers: 11,
        ..reported
    };
    assert_eq!(
        plan.admitted(one_layer_short),
        Err(image::Refusal::LayersBeyondDevice {
            declared: 12,
            max: 11,
        })
    );

    // Every view this texture is addressable through, created for real. The
    // expansion is pure arithmetic and unit-tested as such; what a driver has
    // to agree with is that eighty-four single-slice views over a
    // cube-compatible image are legal, which no arithmetic can establish.
    let whole =
        view::whole(texture, MTL_FORMAT_RGBA8_UNORM, vk::Format::R8G8B8A8_UNORM).create_info(image);
    let sampled = unsafe { device.create_image_view(&whole, None) }
        .expect("a cube-array view over a cube-compatible image");
    let expansion = view::attachments(texture, vk::Format::R8G8B8A8_UNORM);
    assert_eq!(expansion.len() as u32, texture.subresources());
    let attachments: Vec<vk::ImageView> = expansion
        .iter()
        .map(|attachment| {
            unsafe { device.create_image_view(&attachment.plan.create_info(image), None) }
                .unwrap_or_else(|e| {
                    panic!(
                        "level {} slice {} refused: {e}",
                        attachment.level, attachment.slice
                    )
                })
        })
        .collect();
    println!("views whole=1 attachments={}", attachments.len());

    // The buffer half, through the same census. `maxBufferSize` is core in 1.3
    // and an extension below it, so whether this host reported one at all is a
    // real answer and the run prints it.
    let limits = census.buffers();
    println!("buffer max={:?}", limits.max_buffer_size);
    let buffer_plan = buffer::plan(
        1 << 16,
        Route::HostStaging {
            working: MemoryClass::DeviceLocal,
        },
        limits,
    )
    .unwrap_or_else(|refusal| panic!("{refusal}"));
    // Every operation class at once. A driver that rejected the combination
    // would fail here, which is the only thing about the wide usage set that a
    // unit test cannot check.
    let wide = unsafe { device.create_buffer(&buffer_plan.create_info(), None) }
        .expect("a buffer bindable as every class");
    unsafe { device.destroy_buffer(wide, None) };

    unsafe {
        device.device_wait_idle().expect("idle before teardown");
        for attachment in attachments {
            device.destroy_image_view(attachment, None);
        }
        device.destroy_image_view(sampled, None);
        device.destroy_image(image, None);
        device.free_memory(memory, None);
    }
    drop(epoch);
}

/// The ragged fill, end to end on the GPU.
///
/// The split into a staged head, a native interior and a staged tail is pure
/// arithmetic and unit-tested as such. What only a device can settle is
/// whether those three commands, run together, write exactly the bytes the
/// guest named — no byte outside the range, and none inside it left out. So
/// this zeroes a buffer on the GPU, runs a fill over a range that starts and
/// ends off a four-byte boundary, and reads every byte back.
#[test]
fn a_ragged_fill_writes_exactly_the_range_the_guest_named() {
    let Ok(host) = VulkanHost::open("reims-vgpu-vulkan fill integration") else {
        println!("no real device: nothing to fill");
        return;
    };
    let census = host.census();
    let mut epoch = DeviceEpoch::create(
        host.instance(),
        host.physical_device(),
        census,
        EpochId::FIRST,
    )
    .expect("the driver refused a set its own census admitted");
    let device = epoch.device().clone();
    let family = census.queues().universal().index;
    let owner = epoch
        .queues()
        .claim_in(family, 0)
        .expect("the chosen family has queue zero");
    // SAFETY: the family and index came from an owner this epoch handed out.
    let queue = unsafe { device.get_device_queue(owner.family(), owner.index()) };

    let properties = unsafe {
        host.instance()
            .get_physical_device_memory_properties(host.physical_device())
    };
    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    unsafe {
        host.instance()
            .get_physical_device_properties2(host.physical_device(), &mut properties2);
    }
    let limits = unsafe {
        host.instance()
            .get_physical_device_properties(host.physical_device())
    }
    .limits;

    // One host-visible buffer per role, each allocated through the rail's own
    // memory selection so the class it lands on is a real host answer.
    let make = |size: u64, class: MemoryClass| {
        let buffer = unsafe {
            device.create_buffer(
                &buffer::plan(
                    size,
                    Route::HostStaging { working: class },
                    census.buffers(),
                )
                .unwrap_or_else(|refusal| panic!("{refusal}"))
                .create_info(),
                None,
            )
        }
        .expect("a buffer bindable as every class");
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let pick = select_memory_type(
            &properties,
            requirements.memory_type_bits,
            &census.memory().topology.request(class),
            requirements.size,
            maintenance3.max_memory_allocation_size,
        )
        .expect("a host-visible type exists on every Vulkan device");
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(pick.index),
                None,
            )
        }
        .expect("the selected type allocates");
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }.expect("bind");
        (buffer, memory, requirements.size, pick.index)
    };

    const BYTES: u64 = 256;
    const START: u64 = 1;
    const LENGTH: u64 = 253;
    const VALUE: u8 = 0xAB;

    let (dest, dest_memory, dest_size, dest_type) = make(BYTES, MemoryClass::Readback);
    let (scratch, scratch_memory, scratch_size, _) = make(64, MemoryClass::Upload);
    let mapped =
        unsafe { device.map_memory(scratch_memory, 0, scratch_size, vk::MemoryMapFlags::empty()) }
            .expect("upload memory is host visible");
    let mut arena = staging::Arena::adopt(
        64,
        limits.non_coherent_atom_size,
        vec![scratch],
        vec![scratch_memory],
        vec![mapped.cast::<u8>()],
    );

    // The guest's name for the destination, resolved the way every transfer
    // resolves one.
    let mut residency = resident::Residency::new();
    let mut retirement = NativeRetirement::new();
    let name = ResourceId {
        slot: ObjectListRef(1),
        generation: SlotGeneration(1),
    };
    residency
        .publish(
            name,
            Lifetime::new(SessionGeneration::FIRST, EpochId::FIRST),
            resident::Native::Buffer(resident::NativeBuffer {
                buffer: dest,
                memory: dest_memory,
                plan: buffer::plan(
                    BYTES,
                    Route::HostStaging {
                        working: MemoryClass::Readback,
                    },
                    census.buffers(),
                )
                .expect("plannable"),
            }),
            &mut retirement,
        )
        .unwrap_or_else(|(_, e)| panic!("{e}"));

    let plan = transfer::plan_fill(
        BufferSpan {
            buffer: name,
            offset: START,
            length: LENGTH,
        },
        FillPattern::Byte(VALUE),
        &residency,
        &mut arena,
    )
    .unwrap_or_else(|refusal| panic!("{refusal}"))
    .expect("bytes to fill");
    let head = plan
        .head
        .expect("a head, because 1 is not a multiple of four");
    let tail = plan.tail.expect("a tail, because 254 is not either");
    println!(
        "fill head={}@{} middle={}@{} tail={}@{}",
        head.length,
        head.dest_offset,
        plan.middle.expect("an interior").size,
        plan.middle.expect("an interior").offset,
        tail.length,
        tail.dest_offset,
    );

    // The CPU half: the edge bytes into the arena, flushed over the range the
    // arena says covers them. An edge carries the bytes rather than one
    // repeated byte, because a four-byte pattern ending raggedly has a tail of
    // its own leading bytes.
    for edge in [head, tail] {
        // SAFETY: the window came from this arena and nothing has been
        // submitted against it, and the slice is the window's own length.
        unsafe { arena.write(edge.window, &edge.bytes[..edge.length as usize]) };
    }
    let scratch_kind = MappedMemoryKind::of(&properties, {
        let requirements = unsafe { device.get_buffer_memory_requirements(scratch) };
        select_memory_type(
            &properties,
            requirements.memory_type_bits,
            &census.memory().topology.request(MemoryClass::Upload),
            requirements.size,
            maintenance3.max_memory_allocation_size,
        )
        .expect("an upload type")
        .index
    });
    if !scratch_kind.coherent {
        let (offset, size) = arena.flush_range(head.window);
        let (tail_offset, tail_size) = arena.flush_range(tail.window);
        let ranges = [
            vk::MappedMemoryRange::default()
                .memory(scratch_memory)
                .offset(offset)
                .size(size),
            vk::MappedMemoryRange::default()
                .memory(scratch_memory)
                .offset(tail_offset)
                .size(tail_size),
        ];
        unsafe { device.flush_mapped_memory_ranges(&ranges) }.expect("flush");
    }

    // Record: zero the whole buffer, barrier, then the fill.
    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(owner.family()),
            None,
        )
    }
    .expect("a command pool");
    let command = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .expect("a command buffer")[0];
    let recorder = record::Recorder::new(&device, command, census.synchronization2());
    unsafe {
        device
            .begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin");
        device.cmd_fill_buffer(command, dest, 0, BYTES, 0);
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ);
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        recorder.fill(&plan, &arena);
        device.end_command_buffer(command).expect("end");
    }

    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore = unsafe {
        device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
            None,
        )
    }
    .expect("timeline semaphores are the support floor");
    let mut timeline = Timeline::adopt(semaphore);
    let point = timeline.reserve();
    let commands = [command];
    let signals = [semaphore];
    let values = [point.0];
    let mut timeline_info =
        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .command_buffers(&commands)
        .signal_semaphores(&signals)
        .push_next(&mut timeline_info);
    unsafe { device.queue_submit(queue, &[submit], vk::Fence::null()) }.expect("submit");
    arena.submitted(point);
    // SAFETY: the semaphore belongs to this device and is alive.
    unsafe { timeline.wait(&device, point, TIMEOUT_NS) }.expect("the GPU signalled the point");
    // SAFETY: as above.
    let reached = unsafe { timeline.poll(&device) }.expect("a forwards reading");
    assert_eq!(arena.recycle(reached), 1, "the timeline returns the chunk");

    // Every byte, checked against the range the guest named.
    let kind = MappedMemoryKind::of(&properties, dest_type);
    let read = unsafe { device.map_memory(dest_memory, 0, dest_size, vk::MemoryMapFlags::empty()) }
        .expect("host visible");
    if !kind.coherent {
        let range = vk::MappedMemoryRange::default()
            .memory(dest_memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { device.invalidate_mapped_memory_ranges(&[range]) }.expect("invalidate");
    }
    // SAFETY: the mapping covers `dest_size >= BYTES` bytes and the GPU is done.
    let bytes = unsafe { std::slice::from_raw_parts(read.cast::<u8>(), BYTES as usize) };
    for (index, byte) in bytes.iter().enumerate() {
        let index = index as u64;
        let inside = (START..START + LENGTH).contains(&index);
        assert_eq!(
            *byte,
            if inside { VALUE } else { 0 },
            "byte {index} is {byte:#04x}, and it is {} the range",
            if inside { "inside" } else { "outside" }
        );
    }
    println!("filled bytes {START}..{} and nothing else", START + LENGTH);
    unsafe { device.unmap_memory(dest_memory) };

    unsafe {
        device.device_wait_idle().expect("idle before teardown");
        device.unmap_memory(scratch_memory);
        device.destroy_command_pool(pool, None);
        device.destroy_semaphore(semaphore, None);
        device.destroy_buffer(dest, None);
        device.free_memory(dest_memory, None);
        device.destroy_buffer(scratch, None);
        device.free_memory(scratch_memory, None);
    }
    epoch.queues().release(owner);
    drop(epoch);
}

/// The mip ladder, run on the GPU and read back.
///
/// A filtered reduction of a constant image is that constant at every level.
/// So this fills level zero with one colour, runs the ladder this rail plans,
/// and reads the single texel of the top level. A ladder whose barriers were
/// in the wrong order, whose extents were level zero's, or whose blit named
/// the wrong layouts fails here and passes every unit test in the crate.
#[test]
fn a_mip_ladder_reduces_a_constant_image_to_that_constant() {
    let Ok(host) = VulkanHost::open("reims-vgpu-vulkan mipmap integration") else {
        println!("no real device: nothing to reduce");
        return;
    };
    let census = host.census();
    let mut epoch = DeviceEpoch::create(
        host.instance(),
        host.physical_device(),
        census,
        EpochId::FIRST,
    )
    .expect("the driver refused a set its own census admitted");
    let device = epoch.device().clone();
    let family = census.queues().universal().index;
    let owner = epoch
        .queues()
        .claim_in(family, 0)
        .expect("the chosen family has queue zero");
    // SAFETY: the family and index came from an owner this epoch handed out.
    let queue = unsafe { device.get_device_queue(owner.family(), owner.index()) };

    const SIDE: u32 = 4;
    const LEVELS: u32 = 3;
    const TEXEL: [u8; 4] = [0x10, 0x20, 0x30, 0x40];
    let format = vk::Format::R8G8B8A8_UNORM;

    // Measured, not assumed: all three bits `vkCmdBlitImage` with a linear
    // filter demands, read off this device for this format.
    let format_properties = unsafe {
        host.instance()
            .get_physical_device_format_properties(host.physical_device(), format)
    };
    let features = format_properties.optimal_tiling_features;
    let support = mipmap::FilterSupport {
        blit_source: features.contains(vk::FormatFeatureFlags::BLIT_SRC),
        blit_dest: features.contains(vk::FormatFeatureFlags::BLIT_DST),
        linear_blit_source: features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
    };
    println!(
        "blit src={} dst={} linear={}",
        support.blit_source, support.blit_dest, support.linear_blit_source
    );

    let declaration = TextureShape {
        kind: TextureKind::D2.ordinal(),
        width: SIDE,
        height: SIDE,
        depth: 1,
        mipmap_level_count: LEVELS,
        sample_count: 1,
        array_length: 1,
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        usage: TextureUsage::SHADER_READ,
    };
    let texture = declaration.checked().expect("a valid declaration");
    let plan = image::plan(
        texture,
        format,
        Route::HostStaging {
            working: MemoryClass::DeviceLocal,
        },
    )
    .expect("plannable");
    let query = plan.query();
    let reported = unsafe {
        host.instance().get_physical_device_image_format_properties(
            host.physical_device(),
            query.format,
            query.image_type,
            query.tiling,
            query.usage,
            query.flags,
        )
    }
    .expect("RGBA8 sampled is universal");
    let admitted = plan
        .admitted(reported)
        .unwrap_or_else(|refusal| panic!("{refusal}"));
    let image = unsafe { device.create_image(&admitted.create_info(), None) }.expect("an image");

    let properties = unsafe {
        host.instance()
            .get_physical_device_memory_properties(host.physical_device())
    };
    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    unsafe {
        host.instance()
            .get_physical_device_properties2(host.physical_device(), &mut properties2);
    }
    let bind = |requirements: vk::MemoryRequirements, class: MemoryClass| {
        let pick = select_memory_type(
            &properties,
            requirements.memory_type_bits,
            &census.memory().topology.request(class),
            requirements.size,
            maintenance3.max_memory_allocation_size,
        )
        .expect("a memory type for this class");
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(pick.index),
                None,
            )
        }
        .expect("the selected type allocates");
        (memory, pick.index, requirements.size)
    };
    let (image_memory, _, _) = bind(
        unsafe { device.get_image_memory_requirements(image) },
        MemoryClass::DeviceLocal,
    );
    unsafe { device.bind_image_memory(image, image_memory, 0) }.expect("bind");

    // One host-visible buffer carrying level zero's texels up and the top
    // level's one texel back.
    let staging_bytes = u64::from(SIDE * SIDE * 4);
    let staging = unsafe {
        device.create_buffer(
            &buffer::plan(
                staging_bytes,
                Route::HostStaging {
                    working: MemoryClass::Readback,
                },
                census.buffers(),
            )
            .expect("plannable")
            .create_info(),
            None,
        )
    }
    .expect("a staging buffer");
    let requirements = unsafe { device.get_buffer_memory_requirements(staging) };
    let (staging_memory, staging_type, staging_size) = bind(requirements, MemoryClass::Readback);
    unsafe { device.bind_buffer_memory(staging, staging_memory, 0) }.expect("bind");

    let kind = MappedMemoryKind::of(&properties, staging_type);
    let mapped =
        unsafe { device.map_memory(staging_memory, 0, staging_size, vk::MemoryMapFlags::empty()) }
            .expect("host visible");
    // SAFETY: the mapping covers at least `staging_bytes`, and nothing is
    // submitted against it yet.
    unsafe {
        let bytes = std::slice::from_raw_parts_mut(mapped.cast::<u8>(), staging_bytes as usize);
        for texel in bytes.chunks_exact_mut(4) {
            texel.copy_from_slice(&TEXEL);
        }
    }
    if !kind.coherent {
        let range = vk::MappedMemoryRange::default()
            .memory(staging_memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { device.flush_mapped_memory_ranges(&[range]) }.expect("flush");
    }

    // The layout tracker owns every transition below, including the two the
    // ladder does not plan: getting level zero written, and getting the top
    // level readable afterwards.
    let tracked = layout::ImageId(1);
    let mut tracker = layout::LayoutTracker::new();
    tracker.declare(tracked, LEVELS, 1, 1, Some(owner.family()));
    let aspect = view::aspect(MTL_FORMAT_RGBA8_UNORM);

    let pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(owner.family()),
            None,
        )
    }
    .expect("a command pool");
    let command = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .expect("a command buffer")[0];
    let recorder = record::Recorder::new(&device, command, census.synchronization2());
    unsafe {
        device
            .begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin");
    }

    // Level zero, from the staging buffer. Its contents are about to be
    // entirely overwritten, so the transition discards.
    let upload = tracker
        .plan(
            tracked,
            layout::Subresource::new(0, 0),
            layout::Use::TransferDst,
            layout::Contents::Discard,
        )
        .expect("a declared subresource")
        .expect("UNDEFINED is not TRANSFER_DST");
    // SAFETY: the command buffer is recording and every handle is this
    // device's.
    unsafe { recorder.transition(image, &upload, aspect) };
    let region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D {
            width: SIDE,
            height: SIDE,
            depth: 1,
        },
    };
    // SAFETY: as above; level zero is in TRANSFER_DST by the transition just
    // recorded.
    unsafe {
        recorder.transfer(&transfer::Command::CopyBufferToImage {
            source: staging,
            dest: image,
            regions: transfer::Regions::One(region),
        });
    }

    let ladder = mipmap::plan(tracked, texture, support, &mut tracker)
        .unwrap_or_else(|refusal| panic!("{refusal}"));
    assert_eq!(
        ladder
            .iter()
            .filter(|s| matches!(s, mipmap::Step::Blit(_)))
            .count(),
        (LEVELS - 1) as usize
    );
    // SAFETY: as above; the ladder's own transitions put each level into the
    // layout its blit needs.
    unsafe { recorder.mipmap(image, &ladder, aspect) };

    // And the top level out. The ladder left it a transfer destination.
    let readback = tracker
        .plan(
            tracked,
            layout::Subresource::new(LEVELS - 1, 0),
            layout::Use::TransferSrc,
            layout::Contents::Keep,
        )
        .expect("a declared subresource")
        .expect("TRANSFER_DST is not TRANSFER_SRC");
    // SAFETY: as above.
    unsafe { recorder.transition(image, &readback, aspect) };
    // SAFETY: as above.
    unsafe {
        recorder.transfer(&transfer::Command::CopyImageToBuffer {
            source: image,
            dest: staging,
            regions: transfer::Regions::One(vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: aspect,
                    mip_level: LEVELS - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            }),
        });
    }
    unsafe { device.end_command_buffer(command) }.expect("end");

    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore = unsafe {
        device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
            None,
        )
    }
    .expect("timeline semaphores are the support floor");
    let mut timeline = Timeline::adopt(semaphore);
    let point = timeline.reserve();
    let commands = [command];
    let signals = [semaphore];
    let values = [point.0];
    let mut timeline_info =
        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .command_buffers(&commands)
        .signal_semaphores(&signals)
        .push_next(&mut timeline_info);
    unsafe { device.queue_submit(queue, &[submit], vk::Fence::null()) }.expect("submit");
    // SAFETY: the semaphore is this device's and is alive.
    unsafe { timeline.wait(&device, point, TIMEOUT_NS) }.expect("the GPU signalled the point");

    if !kind.coherent {
        let range = vk::MappedMemoryRange::default()
            .memory(staging_memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe { device.invalidate_mapped_memory_ranges(&[range]) }.expect("invalidate");
    }
    // SAFETY: the mapping is at least four bytes and the GPU is done.
    let top = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), 4) };
    assert_eq!(
        top, TEXEL,
        "the top level of a constant image is not that constant"
    );
    println!("reduced {SIDE}x{SIDE} to {top:?} through {LEVELS} levels");

    unsafe {
        device.device_wait_idle().expect("idle before teardown");
        device.unmap_memory(staging_memory);
        device.destroy_command_pool(pool, None);
        device.destroy_semaphore(semaphore, None);
        device.destroy_buffer(staging, None);
        device.free_memory(staging_memory, None);
        device.destroy_image(image, None);
        device.free_memory(image_memory, None);
    }
    epoch.queues().release(owner);
    drop(epoch);
}

/// What this driver actually declines as a vertex attribute, and whether the
/// substitute the rail would reach for is there.
///
/// The unit tests exercise the widening path against a synthetic cell. This
/// asks a real driver which formats it refuses — Vulkan mandates only a subset
/// and every three-channel 8- and 16-bit format is outside it — and then holds
/// the invariant the widening rests on: a declined format either has a
/// supported wider sibling to substitute, or it has no wider sibling at all,
/// in which case the rail refuses by name rather than binding something wrong.
///
/// It asserts nothing about *which* formats a host declines, because that is
/// the fact being measured. What it prints is that list, so a host where the
/// widening path is never reached is distinguishable from one where it is.
#[test]
fn a_declined_vertex_format_has_a_substitute_or_no_sibling_at_all() {
    let Ok(host) = VulkanHost::open("reims-vgpu-vulkan integration") else {
        println!("no real device: nothing to compose");
        return;
    };
    let cell = host.census().vertex();

    let declined: Vec<VertexFormat> = VertexFormat::ALL
        .into_iter()
        .filter(|f| !cell.formats.has(*f))
        .collect();
    println!(
        "vertex formats: {} of {} supported; declined: {:?}",
        cell.formats.count(),
        VertexFormat::ALL.len(),
        declined.iter().map(|f| f.name()).collect::<Vec<_>>()
    );

    for guest in declined {
        let planned = vertex::attribute(0, 0, guest, 0, 64, cell.formats, || {
            vertex::ShaderInput::Channels(guest.components())
        });
        match guest.widened() {
            Some(wider) if cell.formats.has(wider) => {
                let plan = planned.expect("a supported wider sibling is the substitute");
                assert_eq!(plan.widened_from, Some(guest));
                assert_eq!(plan.format, vertex::format(wider));
            }
            // No sibling, or a sibling this driver also declined: refused by
            // name, and never bound as something else.
            _ => assert_eq!(
                planned,
                Err(vertex::Refusal::NoFormat { guest }),
                "{} was declined and not refused",
                guest.name()
            ),
        }
    }

    // Whatever this host declines, every four-channel 32-bit format is
    // mandatory in Vulkan, so the substitutes the rail relies on are there.
    for mandatory in [
        VertexFormat::Float4,
        VertexFormat::UInt4,
        VertexFormat::UChar4,
        VertexFormat::UShort4,
    ] {
        assert!(
            cell.formats.has(mandatory),
            "{} is mandatory and this driver declined it",
            mandatory.name()
        );
    }
}

/// Both rungs of the pass carrier, against a driver.
///
/// The unit tests build the create infos and read them back. That leaves the
/// one thing they cannot say: whether a driver accepts them. So this creates a
/// real image, binds it, makes a real attachment view, and then a real
/// `VkRenderPass` and `VkFramebuffer` from one build — and asserts the
/// pipeline structure the dynamic rung would use names the same format, since
/// it is the same compatibility key.
///
/// A driver that refuses either object fails here and passes every unit test
/// in the crate, which is the whole reason it exists.
#[test]
fn a_planned_pass_becomes_a_render_pass_and_a_framebuffer_this_driver_accepts() {
    let Ok(host) = VulkanHost::open("reims-vgpu-vulkan pass integration") else {
        println!("no real device: nothing to compose");
        return;
    };
    let census = host.census();
    let epoch = DeviceEpoch::create(
        host.instance(),
        host.physical_device(),
        census,
        EpochId::FIRST,
    )
    .expect("the driver refused a set its own census admitted");
    let device = epoch.device().clone();

    // One 64x32 colour target, the shape a pass attaches.
    let texture = TextureShape {
        kind: TextureKind::D2.ordinal(),
        width: 64,
        height: 32,
        depth: 1,
        mipmap_level_count: 1,
        sample_count: 1,
        array_length: 1,
        pixel_format: MTL_FORMAT_RGBA8_UNORM,
        usage: TextureUsage::RENDER_TARGET,
    }
    .checked()
    .expect("a declaration the guest API admits");

    let planned = image::plan(
        texture,
        vk::Format::R8G8B8A8_UNORM,
        Route::HostStaging {
            working: MemoryClass::DeviceLocal,
        },
    )
    .expect("a plannable texture");
    let query = planned.query();
    let reported = unsafe {
        host.instance().get_physical_device_image_format_properties(
            host.physical_device(),
            query.format,
            query.image_type,
            query.tiling,
            query.usage,
            query.flags,
        )
    }
    .expect("RGBA8 color-attachable is universal on Vulkan 1.2");
    let admitted = planned
        .admitted(reported)
        .unwrap_or_else(|refusal| panic!("{refusal}"));
    let image = unsafe { device.create_image(&admitted.create_info(), None) }.expect("an image");

    // A framebuffer names a bound image, so this one has to be backed.
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let properties = unsafe {
        host.instance()
            .get_physical_device_memory_properties(host.physical_device())
    };
    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    unsafe {
        host.instance()
            .get_physical_device_properties2(host.physical_device(), &mut properties2);
    }
    let pick = select_memory_type(
        &properties,
        requirements.memory_type_bits,
        &census.memory().topology.request(MemoryClass::DeviceLocal),
        requirements.size,
        maintenance3.max_memory_allocation_size,
    )
    .expect("a device-local type exists for a colour target");
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(pick.index),
            None,
        )
    }
    .expect("the selected type allocates");
    unsafe { device.bind_image_memory(image, memory, 0) }.expect("bind");

    let expansion = view::attachments(texture, vk::Format::R8G8B8A8_UNORM);
    let attachment = expansion.first().expect("one level, one layer");
    let image_view = unsafe { device.create_image_view(&attachment.plan.create_info(image), None) }
        .expect("an attachment view");

    // The plan, from a descriptor the guest could have sent.
    let mut descriptor = PassDescriptor::empty();
    descriptor.extent = RenderTargetExtent {
        width: 64,
        height: 32,
        array_length: 1,
    };
    descriptor.color[0].texture = Some(ResourceId {
        slot: ObjectListRef(1),
        generation: SlotGeneration(1),
    });
    descriptor.color[0].load = LoadAction::Clear;
    descriptor.color[0].store = StoreAction::Store;
    let pass_plan = pass::plan(&descriptor, |_| MTL_FORMAT_RGBA8_UNORM).expect("a legal pass");

    let built = renderpass::build(
        &pass_plan,
        &[renderpass::Bound {
            format: vk::Format::R8G8B8A8_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            view: image_view,
            // The view is over the whole of a 64x32 image, which is the render
            // area the descriptor above declared.
            coverage: renderpass::Coverage {
                extent: vk::Extent2D {
                    width: 64,
                    height: 32,
                },
                layers: 1,
            },
            resolve: None,
        }],
        None,
    )
    .expect("one colour attachment, one image");

    let render_pass = built
        .with_render_pass_create_info(|info| unsafe { device.create_render_pass(info, None) })
        .expect("a render pass this driver accepts");
    let framebuffer = built
        .with_framebuffer_create_info(render_pass, |info| unsafe {
            device.create_framebuffer(info, None)
        })
        .expect("a framebuffer this driver accepts");

    // The dynamic rung's pipeline structure names the same format, whether or
    // not this host is on that rung — it is the same compatibility key.
    let compatibility = built.compatibility();
    let rendering = compatibility.rendering_info();
    assert_eq!(rendering.color_attachment_count, 1);
    // SAFETY: the array is `compatibility.color`, which outlives `rendering`.
    let formats = unsafe { std::slice::from_raw_parts(rendering.p_color_attachment_formats, 1) };
    assert_eq!(formats[0], vk::Format::R8G8B8A8_UNORM);
    assert_eq!(built.attachments()[0].format, vk::Format::R8G8B8A8_UNORM);
    assert_eq!(built.attachments()[0].load_op, vk::AttachmentLoadOp::CLEAR);
    println!(
        "pass carrier={} attachments={}",
        pass::select(census.passes(), pass::Narrowing::from_env())
            .carrier
            .name(),
        built.attachments().len()
    );

    // SAFETY: nothing was submitted, so nothing names any of these.
    unsafe {
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_image_view(image_view, None);
        device.destroy_image(image, None);
        device.free_memory(memory, None);
    }
}

/// The two halves of the pipeline-assembly probe: `tests/shaders/
/// pipeline_probe.vert` and `.frag`, compiled with `glslc -O`.
///
/// Embedded rather than compiled at test time, for the reason this repository
/// already embeds its scatter kernel: a shader toolchain is not a requirement
/// for running the tests. The risk that buys is source and words drifting
/// apart, which
/// `the_embedded_probe_spirv_matches_its_source` catches where `glslc` exists
/// and reports as skipped where it does not.
mod probe_spirv {
    pub const PROBE_VERT_SPIRV: [u32; 129] = [
        0x07230203, 0x00010000, 0x000d000b, 0x00000015, 0x00000000, 0x00020011, 0x00000001,
        0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
        0x00000000, 0x00000001, 0x0007000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000,
        0x0000000d, 0x00000011, 0x00030047, 0x0000000b, 0x00000002, 0x00050048, 0x0000000b,
        0x00000000, 0x0000000b, 0x00000000, 0x00050048, 0x0000000b, 0x00000001, 0x0000000b,
        0x00000001, 0x00050048, 0x0000000b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048,
        0x0000000b, 0x00000003, 0x0000000b, 0x00000004, 0x00040047, 0x00000011, 0x0000001e,
        0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
        0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040015,
        0x00000008, 0x00000020, 0x00000000, 0x0004002b, 0x00000008, 0x00000009, 0x00000001,
        0x0004001c, 0x0000000a, 0x00000006, 0x00000009, 0x0006001e, 0x0000000b, 0x00000007,
        0x00000006, 0x0000000a, 0x0000000a, 0x00040020, 0x0000000c, 0x00000003, 0x0000000b,
        0x0004003b, 0x0000000c, 0x0000000d, 0x00000003, 0x00040015, 0x0000000e, 0x00000020,
        0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040020, 0x00000010,
        0x00000001, 0x00000007, 0x0004003b, 0x00000010, 0x00000011, 0x00000001, 0x00040020,
        0x00000013, 0x00000003, 0x00000007, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
        0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000007, 0x00000012, 0x00000011,
        0x00050041, 0x00000013, 0x00000014, 0x0000000d, 0x0000000f, 0x0003003e, 0x00000014,
        0x00000012, 0x000100fd, 0x00010038,
    ];

    pub const PROBE_FRAG_SPIRV: [u32; 76] = [
        0x07230203, 0x00010000, 0x000d000b, 0x0000000d, 0x00000000, 0x00020011, 0x00000001,
        0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
        0x00000000, 0x00000001, 0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000,
        0x00000009, 0x00030010, 0x00000004, 0x00000007, 0x00040047, 0x00000009, 0x0000001e,
        0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016,
        0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020,
        0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003,
        0x0004002b, 0x00000006, 0x0000000a, 0x3f800000, 0x0004002b, 0x00000006, 0x0000000b,
        0x00000000, 0x0007002c, 0x00000007, 0x0000000c, 0x0000000a, 0x0000000b, 0x0000000b,
        0x0000000a, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
        0x00000005, 0x0003003e, 0x00000009, 0x0000000c, 0x000100fd, 0x00010038,
    ];
}

/// The embedded probe words are what the sources compile to.
///
/// Skips rather than fails without `glslc`, so a checkout with no shader
/// toolchain reports "not checked" instead of claiming it checked.
#[test]
fn the_embedded_probe_spirv_matches_its_source() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/shaders");
    let mut checked = 0;
    for (name, embedded) in [
        ("pipeline_probe.vert", &probe_spirv::PROBE_VERT_SPIRV[..]),
        ("pipeline_probe.frag", &probe_spirv::PROBE_FRAG_SPIRV[..]),
    ] {
        let out = std::env::temp_dir().join(format!("reims-vgpu-{name}.spv"));
        let Ok(status) = std::process::Command::new("glslc")
            .arg("-O")
            .arg(dir.join(name))
            .arg("-o")
            .arg(&out)
            .status()
        else {
            println!("no glslc: {name} not checked against its source");
            continue;
        };
        assert!(status.success(), "glslc refused {name}");
        let bytes = std::fs::read(&out).expect("glslc wrote no output");
        let recompiled: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            recompiled, embedded,
            "{name} no longer compiles to the embedded words - recompile and update them"
        );
        let _ = std::fs::remove_file(&out);
        checked += 1;
    }
    println!("probe spirv checked={checked}");
}

/// A planned pipeline is one this driver creates.
///
/// Everything the assembly composes is real here: two shader modules, a
/// pipeline layout, a vertex input the vertex shader consumes, and the pass
/// this host's own carrier chose. Nothing is drawn — the claim under test is
/// that the create info is one Vulkan accepts, which is the only claim the
/// unit tests cannot make on their own.
#[test]
fn an_assembled_key_becomes_a_pipeline_this_driver_accepts() {
    let Ok(host) = VulkanHost::open("reims-vgpu-vulkan pipeline integration") else {
        println!("no real device: nothing to assemble");
        return;
    };
    let census = host.census();
    let epoch = DeviceEpoch::create(
        host.instance(),
        host.physical_device(),
        census,
        EpochId::FIRST,
    )
    .expect("the driver refused a set its own census admitted");
    let device = epoch.device().clone();

    let module = |words: &[u32]| {
        let info = vk::ShaderModuleCreateInfo::default().code(words);
        unsafe { device.create_shader_module(&info, None) }.expect("our own SPIR-V")
    };
    let vertex = module(&probe_spirv::PROBE_VERT_SPIRV);
    let fragment = module(&probe_spirv::PROBE_FRAG_SPIRV);
    let layout =
        unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None) }
            .expect("an empty layout is always creatable");

    // The rasterizer state, planned against this host's real cell — so a host
    // with the dynamic-state features exercises the dynamic arm and one
    // without exercises the baked arm, rather than the test choosing.
    let raster_cell = census.raster();
    let raster_plan = raster::plan(
        raster::GuestRasterState {
            cull_mode: raster::MTL_CULL_MODE_BACK,
            winding: raster::MTL_WINDING_COUNTER_CLOCKWISE,
            ..raster::GuestRasterState::DEFAULT
        },
        raster_cell,
    )
    .expect("the two states this asks for need no optional feature");

    let key = pipeline::GraphicsKey {
        stages: vec![
            pipeline::StageKey {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vertex,
                entry: "main".into(),
            },
            pipeline::StageKey {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: fragment,
                entry: "main".into(),
            },
        ],
        layout,
        bindings: vec![vertex::BindingPlan {
            binding: 0,
            stride: 16,
            input_rate: vk::VertexInputRate::VERTEX,
            divisor: 1,
        }],
        attributes: vec![vertex::AttributePlan {
            location: 0,
            binding: 0,
            format: vertex::format(VertexFormat::Float4),
            offset: 0,
            widened_from: None,
        }],
        topology: topology::key(PrimitiveType::Triangle, census.topology()),
        raster: raster_plan.state,
        multisample: pipeline::MultisamplePlan::default(),
        depth_stencil: None,
        blend: vec![blend::plan(&ColorAttachmentState::OPAQUE, census.blend())
            .expect("a default attachment needs no feature")],
        compatibility: renderpass::Compatibility {
            color: vec![vk::Format::B8G8R8A8_UNORM],
            resolve: vec![false],
            depth_stencil: None,
            depth: false,
            stencil: false,
            samples: vk::SampleCountFlags::TYPE_1,
        },
        viewports: reims_vgpu_vulkan::raster::ViewportSlots::ONE,
    };
    let built = pipeline::build(key).unwrap_or_else(|refusal| panic!("{refusal}"));

    // The carrier this host actually chose, so whichever rung it is on is the
    // rung the driver is asked to accept.
    let carrier = pass::select(census.passes(), pass::Narrowing::from_env()).carrier;
    let render_pass = match carrier {
        pass::Carrier::DynamicRendering => vk::RenderPass::null(),
        pass::Carrier::RenderPassObject => {
            let attachment = vk::AttachmentDescription::default()
                .format(vk::Format::B8G8R8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let reference = [vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
            let subpass = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&reference)];
            let attachments = [attachment];
            let info = vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpass);
            unsafe { device.create_render_pass(&info, None) }.expect("one colour attachment")
        }
    };

    let pipelines = built.with_create_info(carrier, render_pass, |info| unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(info),
            None,
        )
    });
    let pipelines =
        pipelines.unwrap_or_else(|(_, e)| panic!("the driver refused the pipeline: {e}"));
    let pipeline = pipelines[0];
    assert_ne!(pipeline, vk::Pipeline::null());

    println!(
        "vk_pipeline carrier={} dynamic={} topology={:?} cull_dynamic={}",
        carrier.name(),
        built.dynamic_states().len(),
        built.key().topology,
        raster_plan.dynamic.cull_mode.is_some(),
    );

    // SAFETY: nothing was submitted, so nothing names any of these.
    unsafe {
        device.destroy_pipeline(pipeline, None);
        if render_pass != vk::RenderPass::null() {
            device.destroy_render_pass(render_pass, None);
        }
        device.destroy_pipeline_layout(layout, None);
        device.destroy_shader_module(fragment, None);
        device.destroy_shader_module(vertex, None);
    }
}
