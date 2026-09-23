//! Which queue family this rail submits to, and the type that makes a
//! `VkQueue` have one owner.
//!
//! # One queue, one owner, and why that is a type
//!
//! `vkQueueSubmit` requires external synchronization on the queue. Two parts of
//! a device that both hold a `VkQueue` and both submit to it are a data race in
//! the driver and, historically here, a deadlock: two submission orders on one
//! queue can each be waiting for something the other has not submitted yet. The
//! architecture's answer is that the queue has exactly one owner, and the
//! reliable way to say that is a value that cannot be copied.
//!
//! [`QueueOwner`] is that value. It is not `Clone`, it has no public
//! constructor, and [`QueuePlan::claim`] hands out at most one per
//! `(family, index)` pair — the second claim gets `None`. So "the queue has one
//! owner" is not a convention a new call site can miss; there is nothing for a
//! second submitter to be constructed from.
//!
//! # One family carries everything, and that is a measurement
//!
//! A dedicated transfer family exists on discrete parts and this rail
//! deliberately does not submit to it. That is not an omission: the copy engine
//! was built on the x86/Vulkan pathway against a discrete host and measured, and
//! the block is *conserved*. Moving the scatter to a second queue collapses the
//! drain worker's wait for a ring slot by more than thirty times and buys no
//! frames, because the wait reappears as a wait for a transfer command buffer,
//! and then as the graphics submission's own write-after-read wait for the
//! scratch buffer. Three arrangements, three counters, one number — the wall is
//! the bus, and a copy engine moves the same bytes over the same link.
//!
//! So [`QueuePlan`] records a transfer-only family as a census fact and routes
//! nothing to it. The route to more frames is fewer bytes crossing, and a
//! second queue is worth building only after the byte volume comes down. What a
//! plan must never do is *silently* discard the fact that the family was there,
//! which is why it is a field and not a dropped `Option`.
//!
//! # Gated on flags, never on a device
//!
//! Everything here reads `VkQueueFlags` and a count. No vendor id, no driver
//! name, no API version, no portability-subset check. `GRAPHICS` and `COMPUTE`
//! both imply transfer support, so a family carrying either is one this rail
//! already submits to — which is the whole test for "dedicated".

use ash::vk;
use std::collections::BTreeSet;

/// One physical device queue family, as the plan needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Family {
    pub index: u32,
    pub flags: vk::QueueFlags,
    /// `VkQueueFamilyProperties::queueCount`. At least one, or the family would
    /// not be reported.
    pub count: u32,
}

impl Family {
    #[must_use]
    pub fn graphics(self) -> bool {
        self.flags.contains(vk::QueueFlags::GRAPHICS)
    }

    #[must_use]
    pub fn compute(self) -> bool {
        self.flags.contains(vk::QueueFlags::COMPUTE)
    }

    /// Whether this family transfers and shares no engine with graphics or
    /// compute.
    ///
    /// Sparse binding, video decode/encode and optical flow do **not**
    /// disqualify it: they say what else that hardware block can do, not that it
    /// shares the graphics engine.
    #[must_use]
    pub fn transfer_only(self) -> bool {
        self.flags.contains(vk::QueueFlags::TRANSFER)
            && !self
                .flags
                .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
    }
}

/// Read the families off a physical device's reported properties.
#[must_use]
pub fn families(props: &[vk::QueueFamilyProperties]) -> Vec<Family> {
    props
        .iter()
        .enumerate()
        .map(|(index, p)| Family {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            flags: p.queue_flags,
            count: p.queue_count,
        })
        .collect()
}

/// Why no plan could be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// No family carries `GRAPHICS`. Not a degraded host: this rail draws, and
    /// a device that cannot is one it cannot execute on at all.
    NoGraphicsFamily { families: usize },
}

impl Decline {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoGraphicsFamily { .. } => "vk_queues_no_graphics_family",
        }
    }
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::NoGraphicsFamily { families } => {
                write!(f, "{} families={families}", self.slug())
            }
        }
    }
}

/// The exclusive right to submit to one `VkQueue`.
///
/// Not `Clone` and not constructible outside this module: two of these are two
/// submission orders on a queue the driver requires external synchronization
/// for. Hold it in whatever owns submission for that queue, and pass it by
/// `&mut` — a function that wants to submit takes the owner rather than a
/// handle, so the borrow checker is what serialises submission.
#[derive(Debug, PartialEq, Eq)]
pub struct QueueOwner {
    family: u32,
    index: u32,
}

impl QueueOwner {
    /// The family this queue belongs to, for a `VkCommandPool` that has to
    /// match it.
    #[must_use]
    pub const fn family(&self) -> u32 {
        self.family
    }

    /// Which queue within the family, for `vkGetDeviceQueue`.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }
}

/// Which family this rail submits to, and what else the host offered.
///
/// Split from [`QueuePlan`] because the two have different lifetimes and
/// different rules. This is an immutable measurement of the device and may be
/// held in the capability catalog ([`crate::census::Census`]) alongside every
/// other reported fact; the plan adds the mutable ledger of which `VkQueue`
/// values have an owner, which belongs to a device epoch and must not be
/// process-global. Folding them together would put per-epoch ownership state
/// into a catalog the architecture requires to be immutable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueChoice {
    universal: Family,
    dedicated_transfer: Option<Family>,
    family_count: usize,
}

impl QueueChoice {
    /// Choose a family from what the device reports.
    ///
    /// Prefers a family that carries both `GRAPHICS` and `COMPUTE`, so a draw
    /// and a dispatch that depend on each other are one submission order and
    /// need no cross-family semaphore or ownership transfer. Falls back to
    /// graphics alone, which is a host where this rail's compute work has to be
    /// refused rather than silently routed elsewhere — see
    /// [`QueuePlan::compute`].
    ///
    /// # Errors
    ///
    /// If no family draws.
    pub fn from_families(families: &[Family]) -> Result<Self, Decline> {
        let universal = families
            .iter()
            .find(|f| f.graphics() && f.compute())
            .or_else(|| families.iter().find(|f| f.graphics()))
            .copied()
            .ok_or(Decline::NoGraphicsFamily {
                families: families.len(),
            })?;
        Ok(Self {
            universal,
            // Recorded, and deliberately not submitted to; see the module docs
            // for the measurement that decided it.
            dedicated_transfer: families.iter().copied().find(|f| f.transfer_only()),
            family_count: families.len(),
        })
    }

    #[must_use]
    pub const fn universal(self) -> Family {
        self.universal
    }

    #[must_use]
    pub fn compute(self) -> bool {
        self.universal.compute()
    }

    #[must_use]
    pub const fn dedicated_transfer(self) -> Option<Family> {
        self.dedicated_transfer
    }

    #[must_use]
    pub const fn family_count(self) -> usize {
        self.family_count
    }
}

/// The chosen family plus the ledger of which of its queues have an owner.
///
/// One per device epoch. See [`QueueChoice`] for why the ledger is not in the
/// catalog.
#[derive(Debug, PartialEq, Eq)]
pub struct QueuePlan {
    universal: Family,
    dedicated_transfer: Option<Family>,
    family_count: usize,
    claimed: BTreeSet<(u32, u32)>,
}

impl QueuePlan {
    /// Choose a family from what the device reports, and start an empty ledger.
    ///
    /// # Errors
    ///
    /// If no family draws.
    pub fn choose(families: &[Family]) -> Result<Self, Decline> {
        Ok(Self::adopt(QueueChoice::from_families(families)?))
    }

    /// Start a ledger against a choice the catalog already made.
    #[must_use]
    pub fn adopt(choice: QueueChoice) -> Self {
        Self {
            universal: choice.universal,
            dedicated_transfer: choice.dedicated_transfer,
            family_count: choice.family_count,
            claimed: BTreeSet::new(),
        }
    }

    /// The measurement this plan was started from, with no ownership state.
    #[must_use]
    pub const fn choice(&self) -> QueueChoice {
        QueueChoice {
            universal: self.universal,
            dedicated_transfer: self.dedicated_transfer,
            family_count: self.family_count,
        }
    }

    /// The family every submission goes to.
    #[must_use]
    pub const fn universal(&self) -> Family {
        self.universal
    }

    /// Whether the chosen family can also dispatch.
    ///
    /// False is a real host arrangement and not a bug, and the caller's answer
    /// to it is a typed refusal of compute work. Routing the dispatch to
    /// another family instead would put it in a second submission order, where
    /// its dependency on a draw stops being expressible as a pipeline barrier.
    #[must_use]
    pub fn compute(&self) -> bool {
        self.universal.compute()
    }

    /// A transfer-only family, if the host has one — a copy engine that runs
    /// beside the graphics one rather than through it.
    ///
    /// Recorded for the report; nothing is routed to it, and the rest of this
    /// doc is why.
    ///
    /// `None` where the host has no such family, which is most integrated
    /// parts. That is the arrangement rather than a degraded one, and every
    /// copy stays on the chosen family.
    ///
    /// # What a boot found, and what it is worth
    ///
    /// This was added without ever being read on a live device. It has been now, on
    /// the x86/Vulkan pathway against an RTX 5080 Laptop:
    ///
    /// ```text
    /// vk_queues families=6 graphics_family=0 compute_capable=true transfer_family=1
    /// ```
    ///
    /// So the copy engine is there, and every byte this device moves is still going
    /// to family 0 with the draws. The size of that is measurable rather than
    /// arguable, because the guest-page writeback carries its own GPU timestamps: a
    /// driven Safari-drag second reports `gpu_us=167437` over `gpu=836` copies —
    /// **167 ms of GPU time per second at ~200 us a copy**, which for a 3.33 MB
    /// copy is a healthy ~16 GB/s and not a slow rail. Scaling the buffer gather by
    /// its share of the bytes (2.74 GB/s against the writeback's 5.19) puts total
    /// copy occupancy near 255 ms/s.
    ///
    /// In the same second `draw_phase`'s `slot_us` is 245 ms/s — the drain worker
    /// blocked in `begin_entry` on a ring slot whose fence the GPU has not signalled.
    /// Those two numbers being within 5 % of each other is the reason to look here:
    /// the CPU's wait for the GPU is about the size of the GPU's copy work, and that
    /// work is serialised against the rendering only because it shares a queue.
    ///
    /// # An ablation says the whole ceiling is this
    ///
    /// The correspondence above is not a proof, so it was tested directly: a probe
    /// boot recorded the writeback's barriers, batch flush, stamp and every CPU-side
    /// bookkeeping step exactly as normal, and skipped only the
    /// `cmd_copy_image_to_buffer`/scatter commands themselves — the GPU work, and
    /// nothing else. The guest loses its frames that way, so it is an ablation and
    /// never a shipping arm; it is recorded here because of what it measured.
    ///
    /// | | shipping | writeback GPU work removed |
    /// |---|---|---|
    /// | `present_hz` median | 72.7-76.4 | **104.0** |
    /// | seconds below 100 Hz | 24/24 | **4/25** |
    /// | `slot_us` | 245 750 us/s | **3 986 us/s** |
    /// | `drain_duty` `duty` | 0.81 | 0.59 |
    /// | `draw_us/draw` | 132-139 us | 78 us |
    /// | draws | 4 383-4 800/s | 5 916-6 407/s |
    ///
    /// `slot_us` falls by a factor of 62. It was not ring depth, not submission
    /// overhead and not jitter: it was this device's own copies sitting in the queue
    /// ahead of the draws whose slots it was waiting for. Every earlier attempt on
    /// `slot_us` moved a number that was downstream of this one, which is why
    /// halving the submissions once bought no frames at all.
    ///
    /// # The prize is not here, and a built rail measured that
    ///
    /// It reads from the table above as if moving the copies off this queue were
    /// worth the gap between 76 Hz and 104 Hz. It is not, and the way to find that
    /// out was to build it: a second queue, a ring of transfer command buffers, two
    /// timeline semaphores, and the writeback's scatter submitted to the copy
    /// engine instead of appended to the draw batch. It ran, on the x86/Vulkan
    /// pathway against the same host, with `vk_queues transfer_family=1`.
    ///
    /// The split was at the **scratch buffer**, not at the image, and that part of
    /// the design was right and stays recorded because it is the cheap answer to the
    /// ownership problem below. The detile (`vkCmdCopyImageToBuffer` into the
    /// device-local scratch) stayed on `gq`, so the render target never left its
    /// family and never gave up its lossless framebuffer compression. Only the
    /// scatter — `vkCmdCopyBuffer` out of the scratch into imported guest pages —
    /// crossed, and the only resources both queues saw were buffers, which are free
    /// to share `CONCURRENT`.
    ///
    /// What four driven Safari-drag boots measured, against a 67.8 Hz baseline taken
    /// on the same tree and machine that hour:
    ///
    /// | arrangement | `present_hz` med | `slot_us` | CPU wait on the copy engine |
    /// |---|---|---|---|
    /// | shipping — every copy on `gq` | 67.8 | 265 000 us/s | — |
    /// | scatter on the copy engine, 4-deep ring | 67.4 | **8 000 us/s** | 240 000 us/s |
    /// | same, 16-deep | 69.8 | 290 000 us/s | ~0 |
    /// | same, 64-deep | 69.0 | 230 000 us/s | ~0 |
    ///
    /// `slot_us` really does collapse — by 33x, close to what the ablation
    /// predicted. And it buys nothing, because **the block is conserved**. At depth
    /// 4 the drain worker stops waiting for a ring slot and starts waiting for a
    /// transfer command buffer instead, for the same 240 ms a second. Deepening the
    /// ring removes that wait and the block reappears a third time, as the graphics
    /// submission's own write-after-read wait for the scratch buffer it is about to
    /// overwrite. Three arrangements, three different counters, one number.
    ///
    /// # Because the wall is the bus, and every queue shares it
    ///
    /// A narrower ablation says so directly. Skipping only the image read, with the
    /// scatter still running and still moving its bytes, gives **72.9 Hz** — four
    /// Hertz, not thirty. The earlier ablation reached 104 Hz because it removed the
    /// scatter too, and with it the bus traffic. A copy engine moves those same
    /// bytes over that same link.
    ///
    /// The traffic is the finding: ~1 500 guest-page writebacks a second at ~3.34 MB
    /// each is **~5.0 GB/s into guest RAM**, sustained, at ~70 displayed frames a
    /// second. That is about **21 full-surface writebacks per frame the user sees**,
    /// spread over roughly six surfaces — and it is split across two rails, the
    /// render Store at ~613/s (`readback_split`'s `vouch`) and the GVA Store making
    /// up the rest of `guest_write_linear`.
    ///
    /// So the route to 120 Hz is fewer bytes crossing, and nothing about which
    /// engine carries them. Do not rebuild this rail to chase frames. It is worth
    /// rebuilding only *after* the byte volume comes down, when a decoupled
    /// `slot_us` would have something left to convert into frames — and the shape it
    /// should take is the one above.
    ///
    /// A copy engine is still not free, and anything built here has to answer three
    /// costs the shared queue does not pay: a cross-queue dependency needs a
    /// semaphore rather than a pipeline barrier, an image written by one family and
    /// read by another needs an ownership transfer or `CONCURRENT` sharing, and
    /// splitting a copy out of the batch it is currently appended to restores the
    /// second submission that appending it removed. Splitting at the scratch buffer
    /// answers the middle one for nothing, which is why that is where it belongs.
    #[must_use]
    pub const fn dedicated_transfer(&self) -> Option<Family> {
        self.dedicated_transfer
    }

    /// Take the sole owner of one queue in the chosen family.
    ///
    /// `None` when `index` is past the family's queue count, or when this queue
    /// already has an owner. Both are the same answer to the caller — there is
    /// no queue here for you to submit to — and neither is a device error.
    pub fn claim(&mut self, index: u32) -> Option<QueueOwner> {
        self.claim_in(self.universal.index, index)
    }

    /// Take the sole owner of one queue in a named family.
    ///
    /// For a caller that legitimately needs a queue outside the chosen family —
    /// a presentation-only family, say. The single-owner rule is the same one,
    /// and it is enforced across families because the pair is the key.
    pub fn claim_in(&mut self, family: u32, index: u32) -> Option<QueueOwner> {
        if family == self.universal.index && index >= self.universal.count {
            return None;
        }
        if !self.claimed.insert((family, index)) {
            return None;
        }
        Some(QueueOwner { family, index })
    }

    /// Give a queue's ownership back, so something else may take it.
    ///
    /// Takes the owner by value, which is what makes the release honest: the
    /// caller cannot keep submitting through an owner it has handed back.
    pub fn release(&mut self, owner: QueueOwner) {
        self.claimed.remove(&(owner.family, owner.index));
    }

    /// Queues currently owned.
    #[must_use]
    pub fn claimed(&self) -> usize {
        self.claimed.len()
    }

    /// One line for the always-on report: what the host offered and what was
    /// chosen from it.
    #[must_use]
    pub fn census(&self) -> String {
        format!(
            "vk_queues families={} universal_family={} queues={} compute={} transfer_only_family={}",
            self.family_count,
            self.universal.index,
            self.universal.count,
            u8::from(self.compute()),
            self.dedicated_transfer
                .map_or_else(|| "none".to_string(), |f| f.index.to_string()),
        )
    }
}

/// Synthetic queue-family tables for the arrangements in the support matrix.
///
/// Public for the same reason [`crate::memory::fixtures`] is: the consumer that
/// needs them is in another crate, and a forked fixture is a second answer to
/// "what does a discrete part's queue layout look like".
pub mod fixtures {
    use super::Family;
    use ash::vk::QueueFlags as F;

    fn family(index: u32, flags: F, count: u32) -> Family {
        Family {
            index,
            flags,
            count,
        }
    }

    /// A discrete part: a universal family, a transfer-only copy engine, and a
    /// compute-only family.
    #[must_use]
    pub fn discrete() -> Vec<Family> {
        vec![
            family(
                0,
                F::GRAPHICS | F::COMPUTE | F::TRANSFER | F::SPARSE_BINDING,
                16,
            ),
            family(1, F::TRANSFER | F::SPARSE_BINDING, 2),
            family(2, F::COMPUTE | F::TRANSFER | F::SPARSE_BINDING, 8),
        ]
    }

    /// An integrated part: one family that does everything, and no copy engine.
    #[must_use]
    pub fn integrated() -> Vec<Family> {
        vec![family(
            0,
            F::GRAPHICS | F::COMPUTE | F::TRANSFER | F::SPARSE_BINDING,
            1,
        )]
    }

    /// A host whose only drawing family cannot dispatch.
    #[must_use]
    pub fn graphics_without_compute() -> Vec<Family> {
        vec![
            family(0, F::GRAPHICS | F::TRANSFER, 1),
            family(1, F::TRANSFER, 1),
        ]
    }

    /// A video-only family beside the universal one. `VIDEO_DECODE` is not a
    /// reason to call a family a copy engine.
    #[must_use]
    pub fn universal_and_video() -> Vec<Family> {
        vec![
            family(0, F::GRAPHICS | F::COMPUTE | F::TRANSFER, 4),
            family(1, F::VIDEO_DECODE_KHR, 1),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(flags: vk::QueueFlags) -> Family {
        Family {
            index: 0,
            flags,
            count: 1,
        }
    }

    /// The invariant the type exists for: the second claim on one queue gets
    /// nothing, so there is no way to construct two submission orders on it.
    #[test]
    fn a_queue_has_one_owner_and_the_second_claim_gets_none() {
        let mut plan = QueuePlan::choose(&fixtures::discrete()).expect("draws");
        let first = plan.claim(0).expect("the family has sixteen queues");
        assert_eq!(first.family(), 0);
        assert_eq!(first.index(), 0);
        assert_eq!(plan.claim(0), None, "already owned");
        assert!(
            plan.claim(1).is_some(),
            "a different queue is a different one"
        );
        assert_eq!(plan.claimed(), 2);
        // And handing it back makes it available, by value so the caller cannot
        // keep submitting through it.
        plan.release(first);
        assert_eq!(plan.claimed(), 1);
        assert!(plan.claim(0).is_some());
    }

    #[test]
    fn a_queue_past_the_familys_count_cannot_be_claimed() {
        let mut plan = QueuePlan::choose(&fixtures::integrated()).expect("draws");
        assert!(plan.claim(0).is_some(), "the one queue it has");
        assert_eq!(plan.claim(1), None, "and there is no second");
        assert_eq!(plan.claimed(), 1, "a refused claim owns nothing");
    }

    /// A draw and a dispatch that depend on each other should be one
    /// submission order, so a family that does both wins even when a
    /// compute-only family exists.
    #[test]
    fn the_chosen_family_draws_and_dispatches() {
        let plan = QueuePlan::choose(&fixtures::discrete()).expect("draws");
        assert_eq!(plan.universal().index, 0);
        assert!(plan.compute());
    }

    /// A host whose drawing family cannot dispatch is a real arrangement. The
    /// plan says so rather than routing the dispatch into a second submission
    /// order where its dependency on a draw stops being a pipeline barrier.
    #[test]
    fn a_graphics_only_host_is_chosen_and_reports_no_compute() {
        let plan = QueuePlan::choose(&fixtures::graphics_without_compute()).expect("draws");
        assert_eq!(plan.universal().index, 0);
        assert!(!plan.compute());
    }

    #[test]
    fn a_device_that_cannot_draw_is_declined_by_name() {
        let compute_only = vec![Family {
            index: 0,
            flags: vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            count: 4,
        }];
        assert_eq!(
            QueuePlan::choose(&compute_only),
            Err(Decline::NoGraphicsFamily { families: 1 })
        );
        assert_eq!(
            QueuePlan::choose(&[]),
            Err(Decline::NoGraphicsFamily { families: 0 })
        );
    }

    /// The copy engine is found and reported, and nothing is routed to it. The
    /// measurement behind that is in the module docs; what is testable here is
    /// that the fact is not silently dropped.
    #[test]
    fn a_transfer_only_family_is_recorded_and_not_submitted_to() {
        let plan = QueuePlan::choose(&fixtures::discrete()).expect("draws");
        assert_eq!(plan.dedicated_transfer().map(|f| f.index), Some(1));
        assert_ne!(
            plan.universal().index,
            1,
            "recorded is not the same as chosen"
        );
        assert!(plan.census().contains("transfer_only_family=1"));

        let integrated = QueuePlan::choose(&fixtures::integrated()).expect("draws");
        assert_eq!(integrated.dedicated_transfer(), None);
        assert!(integrated.census().contains("transfer_only_family=none"));
    }

    /// `GRAPHICS` and `COMPUTE` both imply transfer, so a family carrying
    /// either is one this rail already submits to. Video and sparse do not
    /// disqualify.
    #[test]
    fn dedicated_means_no_graphics_and_no_compute_and_nothing_else() {
        let f = |flags| Family {
            index: 9,
            flags,
            count: 1,
        };
        assert!(f(vk::QueueFlags::TRANSFER).transfer_only());
        assert!(f(vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING).transfer_only());
        assert!(
            f(vk::QueueFlags::TRANSFER | vk::QueueFlags::VIDEO_DECODE_KHR).transfer_only(),
            "what else the block can do is not whether it shares the graphics engine"
        );
        assert!(!f(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER).transfer_only());
        assert!(!f(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER).transfer_only());
        assert!(
            !f(vk::QueueFlags::SPARSE_BINDING).transfer_only(),
            "a family that does not transfer is not a copy engine"
        );
        let video = QueuePlan::choose(&fixtures::universal_and_video()).expect("draws");
        assert_eq!(
            video.dedicated_transfer(),
            None,
            "a video family carries no TRANSFER bit here"
        );
    }

    /// The single-owner rule is keyed on the pair, so a queue outside the
    /// chosen family is subject to it too.
    #[test]
    fn ownership_is_exclusive_across_families() {
        let mut plan = QueuePlan::choose(&fixtures::discrete()).expect("draws");
        let owner = plan.claim_in(2, 0).expect("a compute family queue");
        assert_eq!(owner.family(), 2);
        assert_eq!(plan.claim_in(2, 0), None);
        assert!(
            plan.claim(0).is_some(),
            "and the chosen family's queue 0 is a different pair"
        );
    }

    #[test]
    fn families_are_read_off_the_reported_properties_in_order() {
        let props = [
            vk::QueueFamilyProperties::default()
                .queue_flags(vk::QueueFlags::GRAPHICS)
                .queue_count(2),
            vk::QueueFamilyProperties::default()
                .queue_flags(vk::QueueFlags::TRANSFER)
                .queue_count(1),
        ];
        let read = families(&props);
        assert_eq!(
            read,
            vec![
                Family {
                    index: 0,
                    flags: vk::QueueFlags::GRAPHICS,
                    count: 2
                },
                Family {
                    index: 1,
                    flags: vk::QueueFlags::TRANSFER,
                    count: 1
                },
            ]
        );
    }
    /// A family that also draws or dispatches is one this rail already submits
    /// to, so picking it would move nothing off the graphics engine while
    /// adding an ownership transfer to every copy. Both bits disqualify, and
    /// `TRANSFER` is often not even spelled on a graphics family — the spec
    /// makes it implicit — so the sweep covers families that name it and
    /// families that do not.
    #[test]
    fn a_family_that_also_draws_or_dispatches_is_not_a_copy_engine() {
        use vk::QueueFlags as F;
        for flags in [
            F::GRAPHICS,
            F::COMPUTE,
            F::GRAPHICS | F::COMPUTE,
            F::GRAPHICS | F::TRANSFER,
            F::COMPUTE | F::TRANSFER,
            F::GRAPHICS | F::COMPUTE | F::TRANSFER | F::SPARSE_BINDING,
        ] {
            assert_eq!(
                QueuePlan::choose(&[one(flags)])
                    .ok()
                    .and_then(|p| p.dedicated_transfer()),
                None,
                "{flags:?} shares the engine this rail already submits to"
            );
        }
    }

    /// The bits that say what *else* a copy engine can do must not disqualify
    /// it. A discrete part commonly exposes several transfer-only families that
    /// differ exactly in these, and refusing them would leave a host with a copy
    /// engine reading as a host without one.
    #[test]
    fn the_other_bits_on_a_transfer_only_family_do_not_disqualify_it() {
        use vk::QueueFlags as F;
        for extra in [
            F::empty(),
            F::SPARSE_BINDING,
            F::VIDEO_DECODE_KHR,
            F::VIDEO_ENCODE_KHR,
            F::OPTICAL_FLOW_NV,
        ] {
            let plan = QueuePlan::choose(&[
                one(F::GRAPHICS | F::COMPUTE | F::TRANSFER),
                Family {
                    index: 1,
                    flags: F::TRANSFER | extra,
                    count: 1,
                },
            ])
            .expect("draws");
            assert_eq!(
                plan.dedicated_transfer().map(|f| f.index),
                Some(1),
                "TRANSFER | {extra:?} is still a copy engine"
            );
        }
    }

    /// A family with no transfer bit at all is not one, and a host that offers
    /// none answers `None` rather than falling to index zero — which would
    /// submit copies to the graphics family under a name that says otherwise.
    #[test]
    fn a_host_with_no_copy_engine_answers_none_rather_than_the_first_family() {
        use vk::QueueFlags as F;
        let plan = QueuePlan::choose(&[one(F::GRAPHICS | F::COMPUTE | F::TRANSFER)])
            .expect("the single-family host every integrated part presents");
        assert_eq!(plan.dedicated_transfer(), None);
        assert_eq!(
            QueuePlan::choose(&[one(F::GRAPHICS | F::SPARSE_BINDING)])
                .expect("draws")
                .dedicated_transfer(),
            None,
            "a family with no transfer bit is not a copy engine"
        );
    }
}
