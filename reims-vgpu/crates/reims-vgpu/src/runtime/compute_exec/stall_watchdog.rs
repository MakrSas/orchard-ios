//! One watchdog thread for every compute dispatch, rather than one per dispatch.
//!
//! The measurement this serves is unchanged: a backend call that cannot be
//! bounded by a Vulkan fence timeout (pipeline creation, some driver submits)
//! gets a deadline, and a deadline that passes while the call has not returned
//! writes `compute_engine_stall` plus the private request inputs under `/tmp`,
//! so the stall reproduces without another VM boot.
//!
//! What changed is the cost of *arming* it. Arming used to spawn a thread that
//! slept for the threshold and then, in the overwhelmingly common case, found
//! the call already returned and exited having done nothing — one thread
//! created per dispatch, each holding a private copy of the kernel's SPIR-V
//! alive for the whole threshold. Two of the plan's structural zeros say that
//! is wrong: threads created per EXEC is zero, and heap allocations per
//! steady-state dispatch is zero.
//!
//! So the deadline moved into a registry and the sleeping moved into one
//! thread. A slot is a fixed row in a table allocated once; arming refills that
//! row's buffers in place, which allocates on the first dispatch through a slot
//! and never again. The single thread waits on the nearest armed deadline and
//! is woken when a nearer one is armed.
//!
//! The registry is deliberately bounded. A dispatch that finds every slot armed
//! is *not* watched, and says so on the failure channel — an unbounded table
//! would trade the thread-per-dispatch for an allocation-per-dispatch, and a
//! silent unwatched dispatch would read as "no stall" when it means "no
//! watchdog".

use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::backend::vulkan::engine::ComputeRequest;

/// Concurrent dispatches the table can watch at once.
///
/// The engine serializes compute through one process-global lock, so the
/// steady-state occupancy is one. The headroom is for the arm/disarm overlap
/// and for a fired slot that has not been reclaimed by its guard yet.
const SLOTS: usize = 16;

/// A dispatch that could not be watched because every slot was armed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StallWatchdogDecline {
    NoFreeSlot { pipeline_ref: u32, slots: usize },
}

impl crate::observe::Decline for StallWatchdogDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoFreeSlot { .. } => "compute_stall_watchdog_no_free_slot",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoFreeSlot {
                pipeline_ref,
                slots,
            } => vec![
                ("pipe", pipeline_ref.to_string()),
                ("slots", slots.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(StallWatchdogDecline);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotState {
    /// Nothing is being watched here; the buffers keep their capacity.
    Free,
    /// A call is in flight and its deadline has not passed.
    Armed,
    /// The deadline passed and the report was written. The slot stays out of
    /// service until its guard drops, so one stalled call reports once.
    Reported,
}

struct Slot {
    state: SlotState,
    deadline: Option<Instant>,
    /// The threshold this call was armed with, kept so the report names the
    /// promise that was broken rather than the wake-up jitter around it.
    threshold_ms: u128,
    pipeline_ref: u32,
    grid: [u32; 3],
    buffers: usize,
    images: usize,
    /// `(binding, width, height)` per storage image. Refilled in place.
    image_geometry: Vec<(u32, u32, u32)>,
    /// The module the stalled call was given. Refilled in place.
    spirv: Vec<u32>,
}

impl Slot {
    fn free() -> Self {
        Self {
            state: SlotState::Free,
            deadline: None,
            threshold_ms: 0,
            pipeline_ref: 0,
            grid: [0; 3],
            buffers: 0,
            images: 0,
            image_geometry: Vec::new(),
            spirv: Vec::new(),
        }
    }
}

/// What a fired slot hands to the reporting code, which runs with the table
/// unlocked. Allocating here is fine: it happens only when a call really did
/// not return inside its threshold.
struct Report {
    pipeline_ref: u32,
    elapsed_ms: u128,
    grid: [u32; 3],
    buffers: usize,
    images: usize,
    image_geometry: Vec<(u32, u32, u32)>,
    spirv: Vec<u32>,
}

struct Registry {
    table: Mutex<Vec<Slot>>,
    armed: Condvar,
}

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry = Registry {
            table: Mutex::new((0..SLOTS).map(|_| Slot::free()).collect()),
            armed: Condvar::new(),
        };
        // Started once, for the life of the process. It is the only thread this
        // subsystem ever creates.
        std::thread::spawn(watch_forever);
        registry
    })
}

/// A watched call. Dropping it returns the slot to service; there is no way to
/// arm without holding one, so a call cannot leak a slot by forgetting to
/// disarm.
pub struct StallWatch {
    slot: Option<usize>,
}

impl Drop for StallWatch {
    fn drop(&mut self) {
        let Some(index) = self.slot else { return };
        let registry = registry();
        let mut table = registry
            .table
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let slot = &mut table[index];
        slot.state = SlotState::Free;
        slot.deadline = None;
    }
}

/// Give the backend call `threshold` to return. See the module docs.
pub fn arm_compute_engine_stall_watchdog(
    pipeline_ref: u32,
    req: &ComputeRequest,
    threshold: Duration,
) -> StallWatch {
    let registry = registry();
    let deadline = Instant::now() + threshold;
    let mut table = registry
        .table
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let Some(index) = table.iter().position(|slot| slot.state == SlotState::Free) else {
        drop(table);
        crate::observe::Emit::decline(
            "compute_stall_watchdog",
            &StallWatchdogDecline::NoFreeSlot {
                pipeline_ref,
                slots: SLOTS,
            },
        )
        .fail_once(u64::from(pipeline_ref));
        return StallWatch { slot: None };
    };
    let slot = &mut table[index];
    slot.state = SlotState::Armed;
    slot.deadline = Some(deadline);
    slot.threshold_ms = threshold.as_millis();
    slot.pipeline_ref = pipeline_ref;
    slot.grid = req.dispatch.threadgroups_per_grid();
    slot.buffers = req.storage_buffers.len();
    slot.images = req.storage_images.len();
    // Refill, never rebuild: the capacity a slot reached on its first dispatch
    // is the capacity every later dispatch through that slot reuses.
    slot.image_geometry.clear();
    slot.image_geometry.extend(
        req.storage_images
            .iter()
            .map(|image| (image.binding, image.width, image.height)),
    );
    slot.spirv.clear();
    slot.spirv.extend_from_slice(&req.spirv);
    drop(table);
    // Only the thread's *next* wake time can be wrong, and only if this
    // deadline is nearer than the one it is already waiting on.
    registry.armed.notify_one();
    StallWatch { slot: Some(index) }
}

fn watch_forever() {
    let registry = registry();
    let mut table = registry
        .table
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    loop {
        let now = Instant::now();
        let mut due: Vec<Report> = Vec::new();
        for slot in table.iter_mut() {
            if slot.state != SlotState::Armed {
                continue;
            }
            let Some(deadline) = slot.deadline else {
                continue;
            };
            if deadline > now {
                continue;
            }
            slot.state = SlotState::Reported;
            due.push(Report {
                pipeline_ref: slot.pipeline_ref,
                elapsed_ms: slot.threshold_ms,
                grid: slot.grid,
                buffers: slot.buffers,
                images: slot.images,
                image_geometry: slot.image_geometry.clone(),
                spirv: slot.spirv.clone(),
            });
        }
        if !due.is_empty() {
            // Writing the dump touches the filesystem. The table is not held
            // across it, so an unrelated dispatch can still arm.
            drop(table);
            for report in due {
                report.write();
            }
            table = registry
                .table
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            continue;
        }
        let nearest = table
            .iter()
            .filter(|slot| slot.state == SlotState::Armed)
            .filter_map(|slot| slot.deadline)
            .min();
        table = match nearest {
            Some(deadline) => {
                registry
                    .armed
                    .wait_timeout(table, deadline.saturating_duration_since(now))
                    .unwrap_or_else(|poison| poison.into_inner())
                    .0
            }
            None => registry
                .armed
                .wait(table)
                .unwrap_or_else(|poison| poison.into_inner()),
        };
    }
}

impl Report {
    fn write(self) {
        // `elapsed_ms` is the threshold, not the overshoot: the reader wants
        // "this call had not returned after N ms", and the watchdog's wake-up
        // jitter is not part of that claim.
        let Report {
            pipeline_ref,
            elapsed_ms,
            grid,
            buffers,
            images,
            image_geometry,
            spirv,
        } = self;
        crate::observe::fail(format!(
            "compute_engine_stall reason=backend_call_unreturned pipe={pipeline_ref} \
             elapsed_ms={elapsed_ms} grid={grid:?} nbuf={buffers} nimg={images} \
             image_geom={image_geometry:?}"
        ));
        let base = format!("/tmp/reims-vgpu-compute-stall-pipe-{pipeline_ref}");
        let mut bytes = Vec::with_capacity(spirv.len().saturating_mul(4));
        for word in spirv {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        if let Err(e) = std::fs::write(format!("{base}.spv"), &bytes) {
            crate::observe::fail(format!(
                "compute_engine_stall reason=spv_dump_failed pipe={pipeline_ref} err={e}"
            ));
        }
        let meta = format!(
            "pipe={pipeline_ref}\nelapsed_ms={elapsed_ms}\ngrid={grid:?}\nnbuf={buffers}\n\
             nimg={images}\nimage_geom={image_geometry:?}\n"
        );
        if let Err(e) = std::fs::write(format!("{base}.txt"), meta) {
            crate::observe::fail(format!(
                "compute_engine_stall reason=metadata_dump_failed pipe={pipeline_ref} err={e}"
            ));
        }
    }
}

/// The threshold every production dispatch arms with.
pub const COMPUTE_ENGINE_STALL_PROXY_MS: u64 = 2_000;
