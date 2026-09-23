//! A guest-chosen slot number must not become host memory.
//!
//! # The claim
//!
//! [`reims_vgpu_core::encoder`]'s law is that nothing refuses a slot above the
//! argument table's size: a high slot is a slot, not an error, because a cap
//! standing in for the serializer's truncation points has twice dropped whole
//! binds. That law is about *meaning*, and reading it as a claim about layout
//! is what made one plural bind record naming `first = 0xF000_0000` materialise
//! four billion empty entries — an abort on guest data, and below the abort a
//! guest could buy hundreds of megabytes per record and keep them.
//!
//! So the law stands and the storage does not follow the slot number. That is a
//! claim about *bytes* rather than about trips into the allocator: growing one
//! `Vec` to a guest-chosen index is a single trip and can be every byte the
//! host has, so [`reims_vgpu_testkit::allocations::measure`]'s count cannot see
//! it and [`reims_vgpu_testkit::allocations::measure_cost`] is what this suite
//! asks.
//!
//! An integration test because `reims-vgpu-core` forbids `unsafe` and a
//! `GlobalAlloc` cannot be written without it — the same arrangement, and the
//! same instrument, as `a_warm_path_stops_allocating`.

use reims_vgpu_core::bind::BufferBinding;
use reims_vgpu_core::encoder::SlotTable;
use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
use reims_vgpu_testkit::allocations::{measure_cost, Counting};

#[global_allocator]
static ALLOCATOR: Counting = Counting::new();

fn buffer(id: u32) -> BufferBinding {
    BufferBinding {
        buffer: Some(ResourceId {
            slot: ObjectListRef(id),
            generation: SlotGeneration(1),
        }),
        offset: 0,
        stride: None,
    }
}

/// What one binding is allowed to cost, generously: a `Vec` that has to grow
/// doubles, and the hint reserves the argument table's usual size once. Four
/// kilobytes is far above every one of those and far below the megabytes a
/// slot-indexed layout charges for the same record.
const ONE_BINDING: usize = 4096;

/// The instrument first: a measurement that cannot see a known allocation is
/// not evidence about one it does not expect.
#[test]
fn the_instrument_sees_the_bytes_a_growing_table_asks_for() {
    let (_, cost) = measure_cost(|| {
        let mut table: SlotTable<BufferBinding> = SlotTable::default();
        for slot in 0..512 {
            table.set(slot, Some(buffer(slot)));
        }
        table
    });
    assert!(cost.trips > 0, "512 bindings entered the allocator");
    assert!(
        cost.bytes >= 512 * size_of::<BufferBinding>(),
        "and asked for at least what it stored: {cost:?}"
    );
}

/// The rule: a bind at a slot the guest chose costs a binding, not the gap
/// below it.
#[test]
fn a_bind_at_a_far_slot_costs_one_binding() {
    // A megabyte of slots: enough that a slot-indexed layout is tens of
    // megabytes and unmistakable, and small enough that a run under the old
    // layout fails this assertion rather than the host.
    let far = 1 << 20;
    let (table, cost) = measure_cost(|| {
        let mut table: SlotTable<BufferBinding> = SlotTable::default();
        table.set(far, Some(buffer(1)));
        table
    });
    assert_eq!(table.get(far), Some(buffer(1)), "the slot is bound");
    assert_eq!(table.bound().count(), 1, "and it is the only one");
    assert!(
        cost.bytes <= ONE_BINDING,
        "a slot number is not a footprint: {cost:?}"
    );
}

/// The extreme the wire can actually carry, which is the one that aborted.
#[test]
fn the_highest_slot_a_bind_record_can_name_is_still_one_binding() {
    let (table, cost) = measure_cost(|| {
        let mut table: SlotTable<BufferBinding> = SlotTable::default();
        table.set(u32::MAX, Some(buffer(1)));
        table
    });
    assert_eq!(table.get(u32::MAX), Some(buffer(1)));
    assert!(
        cost.bytes <= ONE_BINDING,
        "the whole u32 range is still one binding: {cost:?}"
    );
}

/// And unbinding a slot nothing bound stores nothing at all — the gap was
/// already the absence of an entry.
#[test]
fn unbinding_a_far_slot_that_was_never_bound_allocates_nothing() {
    let (table, cost) = measure_cost(|| {
        let mut table: SlotTable<BufferBinding> = SlotTable::default();
        table.set(u32::MAX, None);
        table
    });
    assert_eq!(table.bound().count(), 0);
    assert_eq!(cost.trips, 0, "nothing to store: {cost:?}");
}
