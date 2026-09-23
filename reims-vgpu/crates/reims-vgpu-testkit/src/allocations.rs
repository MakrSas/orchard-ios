//! Counting the trips a path makes into the heap allocator.
//!
//! # Why this is a shared instrument
//!
//! The architecture plan lists "heap allocations per steady-state draw" among
//! its structural zeros — a required value, not a target. A structural zero
//! that nothing measures is a claim, and the way it stops being true is not a
//! visible regression: a helper that returns a `Vec` on a per-access or
//! per-draw path costs one `malloc` per call and shows up as a percent or two
//! of drain duty spread evenly across a profile. No single line got slower, so
//! nobody bisects to it.
//!
//! The semantic model and the executor rail both have paths a warm frame takes
//! many times, so both need the same instrument, and a second copy of a
//! `GlobalAlloc` is a second thing to get subtly wrong.
//!
//! # Using it
//!
//! A test crate installs the allocator and measures around the path:
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOCATOR: reims_vgpu_testkit::allocations::Counting =
//!     reims_vgpu_testkit::allocations::Counting::new();
//!
//! let (answer, trips) = reims_vgpu_testkit::allocations::measure(|| warm_path());
//! ```
//!
//! # A trip is not a size
//!
//! [`measure`] counts *trips*, which is the right number for "this per-draw
//! path must not enter the allocator at all". It is the wrong number for "this
//! path must not turn a guest-chosen index into host memory": one `Vec` grown
//! to an index the guest wrote is a single trip and can be every byte the host
//! has. [`measure_cost`] answers both, and a suite asks for the one its claim
//! is about.
//!
//! It has to be an integration test rather than a unit test wherever the crate
//! under measurement forbids `unsafe`, which `reims-vgpu-core` does — a claim
//! about the semantic model worth more than the convenience of measuring from
//! inside it.
//!
//! # The counter is per thread and off by default
//!
//! `#[global_allocator]` is program-wide and libtest runs tests in parallel, so
//! a process-wide counter would count whatever else happened to be running. The
//! count lives in thread-local storage, initialised at compile time so that
//! reading it cannot itself allocate and recurse. Only [`measure`] turns it on,
//! and only for its own thread; every other test pays one relaxed thread-local
//! read per allocation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// Trips into the allocator on this thread since counting began. `const`
    /// initialisation matters: a lazily initialised thread-local allocates on
    /// first use, from inside the allocator.
    static COUNT: Cell<usize> = const { Cell::new(0) };
    /// Bytes this thread asked the allocator to hold since counting began.
    /// A trip is not a size: one `Vec` growing to a slot number the guest
    /// chose is a single trip and can be every byte the host has, so a claim
    /// about *how much* a path may hold needs its own number.
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static ON: Cell<bool> = const { Cell::new(false) };
}

/// A `System` allocator that counts, for the thread and the region a
/// [`measure`] asks about.
pub struct Counting;

impl Counting {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for Counting {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every method forwards to `System`, which is a correct allocator. The
// bookkeeping around it allocates nothing — `Cell<usize>` and `Cell<bool>` are
// const-initialised and have no destructor, so no thread-local registration
// happens on first use. `try_with` rather than `with`, because a thread tearing
// down may already have destroyed its storage, and a panic inside the allocator
// aborts the process.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump(layout.size());
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A `Vec` growing is an allocation by the measure that matters here:
        // it is a trip into the allocator and the bytes may move. The size
        // charged is the new one, because that is what the caller asked the
        // host to hold.
        bump(new_size);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump(layout.size());
        System.alloc_zeroed(layout)
    }
}

fn bump(size: usize) {
    if ON.try_with(Cell::get).unwrap_or(false) {
        let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get().saturating_add(size)));
    }
}

/// Run `body` and return how many times it entered the allocator.
///
/// Not nestable, and it does not need to be: the counter is a single depth,
/// and a measurement inside a measurement would mean the inner path is not the
/// thing being measured.
pub fn measure<T>(body: impl FnOnce() -> T) -> (T, usize) {
    let (out, cost) = measure_cost(body);
    (out, cost.trips)
}

/// What one measured region cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    /// Trips into the allocator.
    pub trips: usize,
    /// Bytes asked for across them.
    pub bytes: usize,
}

/// Run `body` and return both what it cost.
///
/// The two numbers answer different questions and neither implies the other. A
/// per-draw path that must not enter the allocator at all is a claim about
/// `trips`; a path handed a guest-chosen index that must not turn it into a
/// footprint is a claim about `bytes`, and that one is a single trip.
///
/// Not nestable, for [`measure`]'s reason.
pub fn measure_cost<T>(body: impl FnOnce() -> T) -> (T, Cost) {
    COUNT.with(|c| c.set(0));
    BYTES.with(|c| c.set(0));
    ON.with(|c| c.set(true));
    let out = body();
    ON.with(|c| c.set(false));
    (
        out,
        Cost {
            trips: COUNT.with(Cell::get),
            bytes: BYTES.with(Cell::get),
        },
    )
}
