//! Command pools and command buffers, owned per worker and recycled against
//! the timeline.
//!
//! # A pool belongs to one worker because the driver says so
//!
//! `VkCommandPool` is externally synchronized, and so is every command buffer
//! allocated from it. Two workers recording from one pool is a driver data race
//! whether or not they touch the same buffer — allocation, reset and free all
//! go through the pool. The architecture already assigns work to an exact
//! worker ([`reims_vgpu_core::executor::FixedExecutor`]); this is the native
//! half of the same claim, and [`WorkerPools::for_worker`] hands out `&mut` so
//! two workers cannot hold one pool at all.
//!
//! # A buffer is free when the timeline says so, and not before
//!
//! Re-recording a command buffer the GPU is still reading is the
//! use-after-submit that produces corruption a long way from its cause. So a
//! slot leaves [`SlotState::Submitted`] only when the timeline point it was
//! submitted at has been reached — the same exactness
//! [`reims_vgpu_core::retire`] applies to native objects, applied to the
//! buffers themselves.
//!
//! [`Lease`] is how that is enforced rather than remembered: it is not `Clone`,
//! there is no way to make one except by taking a free slot, and
//! [`BufferRing::submitted`] consumes it. A buffer therefore cannot be
//! submitted twice, and a slot cannot be recorded into while a lease for it is
//! outstanding.
//!
//! # Exhaustion is a refusal, never a wait
//!
//! When every slot is in flight, [`BufferRing::begin`] refuses. It does not
//! block on the timeline and it does not recycle the oldest slot — blocking
//! inside a recording path is how one worker's depth becomes every worker's
//! latency, and recycling the oldest is the use-after-submit above with extra
//! steps. The caller polls the timeline, recycles, and retries or parks; which
//! of those is right is a scheduling decision and not this ring's.
//!
//! # The bookkeeping is separate from the handles
//!
//! [`BufferRing`] holds no Vulkan object, so every rule above is tested on a
//! machine with no GPU. [`WorkerPool`] is the ring plus the handles it names.

use ash::vk;
use reims_vgpu_core::executor::WorkerId;
use reims_vgpu_core::identity::TimelinePoint;

/// What one command-buffer slot is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// Nothing holds it; it may be recorded into.
    Free,
    /// A [`Lease`] is outstanding for it.
    Recording,
    /// Submitted, and readable by the GPU until the timeline reaches this
    /// point.
    Submitted(TimelinePoint),
}

/// The exclusive right to record into one command-buffer slot.
///
/// Not `Clone`, and constructible only by [`BufferRing::begin`]. Consumed by
/// [`BufferRing::submitted`] or [`BufferRing::abandon`], so a buffer cannot be
/// submitted twice and a slot cannot be handed out while a lease for it exists.
#[derive(Debug, PartialEq, Eq)]
pub struct Lease {
    slot: usize,
}

impl Lease {
    /// Which slot this lease is for, so the caller can index its own
    /// `VkCommandBuffer` array.
    #[must_use]
    pub const fn slot(&self) -> usize {
        self.slot
    }
}

/// Every slot is in flight.
///
/// Not a wait and not a licence to recycle the oldest: the caller polls the
/// timeline and retries, or parks the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exhausted {
    pub depth: usize,
    /// Slots waiting on the timeline. Equal to `depth` when nothing is
    /// recording, and smaller when the caller is holding leases it has not
    /// submitted.
    pub in_flight: usize,
}

impl Exhausted {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        "vk_pool_no_free_command_buffer"
    }
}

impl std::fmt::Display for Exhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} depth={} in_flight={}",
            self.slug(),
            self.depth,
            self.in_flight
        )
    }
}

/// One worker's command-buffer slots, and which of them may be recorded into.
///
/// No Vulkan object: this is the part that can be wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferRing {
    slots: Vec<SlotState>,
    recycled: usize,
    refused: usize,
}

impl BufferRing {
    /// A ring of `depth` slots, all free.
    ///
    /// # Panics
    ///
    /// If `depth` is zero. A ring with no slots refuses every recording, which
    /// is a worker that can never do anything rather than a shallow one.
    #[must_use]
    pub fn new(depth: usize) -> Self {
        assert!(depth > 0, "a worker with no command buffers cannot record");
        Self {
            slots: vec![SlotState::Free; depth],
            recycled: 0,
            refused: 0,
        }
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn state(&self, slot: usize) -> Option<SlotState> {
        self.slots.get(slot).copied()
    }

    /// Take a free slot to record into.
    ///
    /// # Errors
    ///
    /// [`Exhausted`] when every slot is recording or in flight.
    pub fn begin(&mut self) -> Result<Lease, Exhausted> {
        match self.slots.iter().position(|s| *s == SlotState::Free) {
            Some(slot) => {
                self.slots[slot] = SlotState::Recording;
                Ok(Lease { slot })
            }
            None => {
                self.refused += 1;
                Err(Exhausted {
                    depth: self.slots.len(),
                    in_flight: self.in_flight(),
                })
            }
        }
    }

    /// The slot was submitted, and the GPU may read it until `at`.
    ///
    /// Consumes the lease, which is what makes a second submission of the same
    /// buffer unwritable rather than merely wrong.
    pub fn submitted(&mut self, lease: Lease, at: TimelinePoint) {
        self.slots[lease.slot] = SlotState::Submitted(at);
    }

    /// The recording was given up and never submitted, so the slot is free at
    /// once — there is no GPU work to wait for.
    pub fn abandon(&mut self, lease: Lease) {
        self.slots[lease.slot] = SlotState::Free;
    }

    /// Free every slot the timeline has passed. Returns how many.
    ///
    /// The one place a slot becomes recordable again, and it consults the
    /// timeline rather than an age or a count.
    pub fn recycle(&mut self, reached: TimelinePoint) -> usize {
        let mut freed = 0;
        for slot in &mut self.slots {
            if let SlotState::Submitted(at) = *slot {
                if reached.reached(at) {
                    *slot = SlotState::Free;
                    freed += 1;
                }
            }
        }
        self.recycled += freed;
        freed
    }

    /// Slots waiting on the timeline.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, SlotState::Submitted(_)))
            .count()
    }

    /// Slots with an outstanding lease.
    #[must_use]
    pub fn recording(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| **s == SlotState::Recording)
            .count()
    }

    #[must_use]
    pub fn free(&self) -> usize {
        self.slots.iter().filter(|s| **s == SlotState::Free).count()
    }

    /// Whether `vkResetCommandPool` is legal: nothing is recording and nothing
    /// is in flight.
    ///
    /// A pool reset frees every buffer at once, so it is only ever correct when
    /// the ring says the GPU is finished with all of them. Asking the ring is
    /// the same question as asking the timeline, and cheaper.
    #[must_use]
    pub fn resettable(&self) -> bool {
        self.slots.iter().all(|s| *s == SlotState::Free)
    }

    /// Recordings recycled, and recordings refused for want of a slot.
    ///
    /// The second number is what says whether the depth is the bottleneck. A
    /// depth that never refuses is one nothing is waiting on.
    #[must_use]
    pub const fn census(&self) -> (usize, usize) {
        (self.recycled, self.refused)
    }
}

/// One worker's pool and the buffers allocated from it.
///
/// Owns handles and destroys nothing: this crate allocates no Vulkan object it
/// does not also hand back, so [`WorkerPool::pool`] and
/// [`WorkerPool::buffer`] are what a teardown call takes.
#[derive(Debug)]
pub struct WorkerPool {
    pool: vk::CommandPool,
    family: u32,
    buffers: Vec<vk::CommandBuffer>,
    ring: BufferRing,
}

impl WorkerPool {
    /// Adopt a pool and the buffers allocated from it.
    ///
    /// # Panics
    ///
    /// If no buffers were allocated. See [`BufferRing::new`].
    #[must_use]
    pub fn adopt(pool: vk::CommandPool, family: u32, buffers: Vec<vk::CommandBuffer>) -> Self {
        let ring = BufferRing::new(buffers.len());
        Self {
            pool,
            family,
            buffers,
            ring,
        }
    }

    #[must_use]
    pub const fn pool(&self) -> vk::CommandPool {
        self.pool
    }

    /// The queue family this pool's buffers may be submitted to. A submission
    /// to any other family is invalid usage, so the pool carries it rather than
    /// the caller remembering.
    #[must_use]
    pub const fn family(&self) -> u32 {
        self.family
    }

    #[must_use]
    pub fn buffers(&self) -> &[vk::CommandBuffer] {
        &self.buffers
    }

    /// The buffer a lease names.
    #[must_use]
    pub fn buffer(&self, lease: &Lease) -> vk::CommandBuffer {
        self.buffers[lease.slot()]
    }

    #[must_use]
    pub const fn ring(&self) -> &BufferRing {
        &self.ring
    }

    pub const fn ring_mut(&mut self) -> &mut BufferRing {
        &mut self.ring
    }
}

/// One pool per worker, and no way to hold two at once.
#[derive(Debug, Default)]
pub struct WorkerPools {
    pools: Vec<WorkerPool>,
}

impl WorkerPools {
    #[must_use]
    pub const fn new() -> Self {
        Self { pools: Vec::new() }
    }

    /// Add the next worker's pool. Workers are numbered by insertion order,
    /// matching the fixed population the executor was built with.
    pub fn push(&mut self, pool: WorkerPool) -> WorkerId {
        let id = u16::try_from(self.pools.len()).unwrap_or(u16::MAX);
        self.pools.push(pool);
        WorkerId(id)
    }

    #[must_use]
    pub fn population(&self) -> usize {
        self.pools.len()
    }

    /// The pool belonging to one worker, exclusively.
    ///
    /// `&mut` is the enforcement: `VkCommandPool` is externally synchronized,
    /// and two workers that could hold one pool at the same time would be a
    /// driver data race no amount of care at the call site fixes.
    pub fn for_worker(&mut self, worker: WorkerId) -> Option<&mut WorkerPool> {
        self.pools.get_mut(usize::from(worker.0))
    }

    #[must_use]
    pub fn of_worker(&self, worker: WorkerId) -> Option<&WorkerPool> {
        self.pools.get(usize::from(worker.0))
    }

    #[must_use]
    pub fn all(&self) -> &[WorkerPool] {
        &self.pools
    }

    /// Recycle every worker's finished buffers against one timeline reading.
    /// Returns how many slots became free.
    pub fn recycle(&mut self, reached: TimelinePoint) -> usize {
        self.pools.iter_mut().map(|p| p.ring.recycle(reached)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(n: u64) -> TimelinePoint {
        TimelinePoint(n)
    }

    /// The invariant the whole module exists for: a submitted slot is not
    /// recordable until the timeline passes it.
    #[test]
    fn a_submitted_slot_is_not_free_until_the_timeline_passes_it() {
        let mut r = BufferRing::new(2);
        let first = r.begin().expect("all free");
        assert_eq!(r.state(first.slot()), Some(SlotState::Recording));
        r.submitted(first, at(7));
        assert_eq!(r.state(0), Some(SlotState::Submitted(at(7))));
        assert_eq!(r.in_flight(), 1);

        assert_eq!(r.recycle(at(6)), 0, "the host has not got there");
        assert_eq!(r.recycle(at(7)), 1, "and now it has");
        assert_eq!(r.state(0), Some(SlotState::Free));
        assert_eq!(r.census(), (1, 0));
    }

    /// Exhaustion refuses. It does not block, and it does not recycle the
    /// oldest slot — which would be the use-after-submit with extra steps.
    #[test]
    fn a_full_ring_refuses_rather_than_recycling_the_oldest() {
        let mut r = BufferRing::new(2);
        let a = r.begin().expect("free");
        let b = r.begin().expect("free");
        r.submitted(a, at(1));
        r.submitted(b, at(2));
        assert_eq!(
            r.begin(),
            Err(Exhausted {
                depth: 2,
                in_flight: 2
            })
        );
        assert_eq!(
            r.state(0),
            Some(SlotState::Submitted(at(1))),
            "the oldest slot was not taken back"
        );
        assert_eq!(r.census(), (0, 1), "and the refusal is counted");
        // Polling the timeline is the caller's answer.
        assert_eq!(r.recycle(at(1)), 1);
        assert!(r.begin().is_ok());
    }

    /// A lease is exclusive, so a slot cannot be handed out twice.
    #[test]
    fn an_outstanding_lease_holds_its_slot() {
        let mut r = BufferRing::new(1);
        let held = r.begin().expect("free");
        assert_eq!(
            r.begin(),
            Err(Exhausted {
                depth: 1,
                in_flight: 0
            }),
            "in_flight is zero: the slot is held by a lease, not by the GPU"
        );
        assert_eq!(r.recording(), 1);
        assert_eq!(
            r.recycle(at(u64::MAX)),
            0,
            "and no timeline reading frees a slot nothing submitted"
        );
        r.abandon(held);
        assert_eq!(r.free(), 1);
    }

    /// A recording that was given up never reached the GPU, so there is
    /// nothing to wait for.
    #[test]
    fn an_abandoned_recording_frees_its_slot_at_once() {
        let mut r = BufferRing::new(1);
        let lease = r.begin().expect("free");
        r.abandon(lease);
        assert!(r.resettable());
        assert!(r.begin().is_ok());
    }

    /// A pool reset frees every buffer at once, so it is legal only when the
    /// GPU is finished with all of them.
    #[test]
    fn a_pool_is_resettable_only_when_nothing_is_recording_or_in_flight() {
        let mut r = BufferRing::new(2);
        assert!(r.resettable());
        let lease = r.begin().expect("free");
        assert!(!r.resettable(), "a recording is not finished with");
        r.submitted(lease, at(3));
        assert!(!r.resettable(), "and neither is a submission");
        r.recycle(at(3));
        assert!(r.resettable());
    }

    #[test]
    fn slots_are_reused_lowest_first_so_a_ring_of_one_works() {
        let mut r = BufferRing::new(3);
        for expected in 0..3 {
            let lease = r.begin().expect("free");
            assert_eq!(lease.slot(), expected);
            r.submitted(lease, at(expected as u64 + 1));
        }
        r.recycle(at(2));
        let reused = r.begin().expect("two came back");
        assert_eq!(reused.slot(), 0);
    }

    #[test]
    #[should_panic(expected = "cannot record")]
    fn a_ring_of_no_slots_is_not_a_shallow_worker() {
        let _ = BufferRing::new(0);
    }

    /// Two workers cannot hold one pool, because `for_worker` hands out `&mut`
    /// and a `VkCommandPool` is externally synchronized.
    #[test]
    fn each_worker_has_its_own_pool_and_its_own_ring() {
        let mut pools = WorkerPools::new();
        let a = pools.push(WorkerPool::adopt(
            vk::CommandPool::null(),
            0,
            vec![vk::CommandBuffer::null(); 2],
        ));
        let b = pools.push(WorkerPool::adopt(
            vk::CommandPool::null(),
            0,
            vec![vk::CommandBuffer::null(); 1],
        ));
        assert_eq!((a, b), (WorkerId(0), WorkerId(1)));
        assert_eq!(pools.population(), 2);

        // Fill worker b's single slot; worker a is untouched by it.
        let lease = pools
            .for_worker(b)
            .expect("worker b")
            .ring_mut()
            .begin()
            .expect("free");
        pools
            .for_worker(b)
            .expect("worker b")
            .ring_mut()
            .submitted(lease, at(5));
        assert!(pools
            .for_worker(b)
            .expect("worker b")
            .ring_mut()
            .begin()
            .is_err());
        assert_eq!(pools.of_worker(a).expect("worker a").ring().free(), 2);

        // One timeline reading recycles across every worker.
        assert_eq!(pools.recycle(at(5)), 1);
        assert_eq!(pools.of_worker(b).expect("worker b").ring().free(), 1);
        assert!(pools.for_worker(WorkerId(9)).is_none());
    }

    /// A submission to a family other than the pool's is invalid usage, so the
    /// pool carries the family rather than the caller remembering it.
    #[test]
    fn a_pool_names_the_family_its_buffers_may_be_submitted_to() {
        let pool = WorkerPool::adopt(vk::CommandPool::null(), 3, vec![vk::CommandBuffer::null()]);
        assert_eq!(pool.family(), 3);
        assert_eq!(pool.buffers().len(), 1);
        assert_eq!(pool.ring().depth(), 1);
    }
}
