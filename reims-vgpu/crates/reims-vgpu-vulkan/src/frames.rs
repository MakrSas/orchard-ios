//! Swapchain frame slots, and the binary semaphores that cannot simply be
//! reused when something goes wrong.
//!
//! # What this owns that [`reims_vgpu_core::present`] does not
//!
//! The core stream owns the semantic side of presentation: which frames are in
//! flight, in what order, what parks when no image is free, and what the old
//! swapchain is still owed. None of that names a Vulkan object.
//!
//! This owns the native side: which swapchain image a frame reserved, and the
//! two binary semaphores that carry it — one the presentation engine signals
//! when the image is ready to render into, one the queue signals when the
//! rendering is done and presentation may proceed.
//!
//! # A binary semaphore is not reusable just because the frame failed
//!
//! This is the whole reason the module exists. A timeline semaphore has a
//! value, so a failed operation leaves it observably where it was. A binary
//! semaphore has a *pending signal*, and a failure can leave one outstanding
//! with nothing waiting on it. Reusing that semaphore in the next frame is
//! invalid usage, and the way it fails is a hang or a corrupted frame several
//! presents later.
//!
//! The three ways a frame can fail are not the same:
//!
//! - **Acquire refused.** `vkAcquireNextImageKHR` returning an error does not
//!   signal the semaphore, so both semaphores stay usable and the slot goes
//!   straight back. (`VK_SUBOPTIMAL_KHR` is *not* this case: it acquires an
//!   image and signals, so it is a success this ring is told about as one.)
//! - **Submission refused.** The acquire succeeded, so the presentation engine
//!   will signal the acquire semaphore — and the submission that was going to
//!   wait on it never happened. That signal is now outstanding forever, and the
//!   acquire semaphore [`MustRecreate`].
//! - **Present refused.** The submission happened and signalled the render
//!   semaphore; `vkQueuePresentKHR` failing may or may not have consumed that
//!   wait. The render semaphore [`MustRecreate`].
//!
//! [`MustRecreate`]: Usability::MustRecreate
//!
//! # A quarantined semaphore is not destroyed, it is outlived
//!
//! The corollary the same hazard forces: a semaphore with a signal that may
//! still arrive cannot be destroyed either. `vkDestroySemaphore` requires that
//! no operation on it is outstanding, and the whole problem is not knowing.
//! So [`FrameRing::quarantined`] hands back the slots whose old semaphores must
//! be kept alive, and the caller retires them at swapchain teardown — after the
//! device is idle and the presentation engine can no longer signal anything.
//!
//! # Only a receipt advances a slot
//!
//! [`Reservation`] is not `Clone` and has no public constructor, and every exit
//! from it consumes it. So a slot cannot be presented twice, cannot be
//! abandoned and submitted, and cannot advance because somebody believed the
//! submission had gone through — it advances because the queue owner handed
//! back a receipt.

use reims_vgpu_core::identity::TimelinePoint;

/// Whether a slot's semaphore may be used again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Usability {
    #[default]
    Usable,
    /// A signal may be outstanding on it. It cannot be waited on, cannot be
    /// signalled again, and cannot be destroyed until the device is idle.
    MustRecreate,
}

impl Usability {
    #[must_use]
    pub const fn usable(self) -> bool {
        matches!(self, Self::Usable)
    }
}

/// The two semaphores one frame slot carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Health {
    /// Signalled by the presentation engine when the image may be rendered
    /// into.
    pub acquire: Usability,
    /// Signalled by the queue when rendering is done and the present may go.
    pub render: Usability,
}

impl Health {
    #[must_use]
    pub const fn ready(self) -> bool {
        self.acquire.usable() && self.render.usable()
    }
}

/// What one frame slot is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// Nothing holds it.
    Free,
    /// An image has been acquired into it and nothing has been submitted yet.
    Reserved { image: u32 },
    /// Submitted and presented; the GPU is done with it when the timeline
    /// reaches this point.
    Submitted { image: u32, at: TimelinePoint },
}

/// The exclusive right to render one acquired swapchain image.
///
/// Not `Clone`, constructible only by [`FrameRing::acquired`], and consumed by
/// every exit. A frame therefore cannot be presented twice, and cannot both be
/// abandoned and submitted.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "an acquired image that is neither submitted nor abandoned strands its slot"]
pub struct Reservation {
    slot: usize,
    image: u32,
}

impl Reservation {
    /// Which slot, so the caller can index its own semaphore arrays.
    #[must_use]
    pub const fn slot(&self) -> usize {
        self.slot
    }

    /// The swapchain image index the presentation engine handed back.
    #[must_use]
    pub const fn image(&self) -> u32 {
        self.image
    }
}

/// Why a reserved frame is being given up.
///
/// Not one cause, because the three leave different semaphores in different
/// states. See the module doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abandon {
    /// The submission was refused before reaching the driver, or the driver
    /// refused it. The acquire semaphore's signal is outstanding with nothing
    /// to wait on it.
    SubmitRefused,
    /// `vkQueuePresentKHR` refused after a successful submit. The render
    /// semaphore's wait may or may not have executed.
    PresentRefused,
    /// The frame was given up before anything was submitted and before the
    /// presentation engine could signal — a swapchain being recreated under a
    /// reservation that had not yet been used. Both semaphores are suspect,
    /// because the acquire's signal is still coming.
    SwapchainReplaced,
}

impl Abandon {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::SubmitRefused => "vk_frame_submit_refused",
            Self::PresentRefused => "vk_frame_present_refused",
            Self::SwapchainReplaced => "vk_frame_swapchain_replaced",
        }
    }

    /// What this cause does to the slot's semaphores.
    #[must_use]
    pub const fn leaves(self) -> Health {
        match self {
            Self::SubmitRefused => Health {
                acquire: Usability::MustRecreate,
                render: Usability::Usable,
            },
            Self::PresentRefused => Health {
                acquire: Usability::Usable,
                render: Usability::MustRecreate,
            },
            Self::SwapchainReplaced => Health {
                acquire: Usability::MustRecreate,
                render: Usability::MustRecreate,
            },
        }
    }
}

/// No slot can take an acquire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoSlot {
    pub depth: usize,
    pub in_flight: usize,
    /// Slots that are otherwise free but whose semaphores must be recreated
    /// first. Non-zero here means the caller's next move is
    /// [`FrameRing::recreated`] and not a wait — a distinction a bare "no
    /// slots" would lose.
    pub awaiting_semaphores: usize,
}

impl NoSlot {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        "vk_frame_no_slot"
    }
}

impl std::fmt::Display for NoSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} depth={} in_flight={} awaiting_semaphores={}",
            self.slug(),
            self.depth,
            self.in_flight,
            self.awaiting_semaphores
        )
    }
}

/// What the ring has done.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub acquired: usize,
    pub presented: usize,
    pub abandoned: usize,
    pub recycled: usize,
    /// Acquires refused for want of a slot.
    pub refused: usize,
    /// Semaphores that had to be recreated after a failure.
    pub recreated: usize,
}

/// One swapchain generation's frame slots.
///
/// No Vulkan object: this is the part that can be wrong. The caller holds the
/// `VkSemaphore` pairs and indexes them by [`Reservation::slot`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameRing {
    slots: Vec<SlotState>,
    health: Vec<Health>,
    census: Census,
}

impl FrameRing {
    /// A ring of `depth` slots, all free with fresh semaphores.
    ///
    /// `depth` is the number of images the host **returned**, never the number
    /// requested — see [`reims_vgpu_core::present`], which owns the same rule
    /// for the semantic side.
    ///
    /// # Panics
    ///
    /// If `depth` is zero. A swapchain with no images is not a shallow one.
    #[must_use]
    pub fn new(depth: usize) -> Self {
        assert!(depth > 0, "a swapchain with no images cannot present");
        Self {
            slots: vec![SlotState::Free; depth],
            health: vec![Health::default(); depth],
            census: Census::default(),
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

    #[must_use]
    pub fn health(&self, slot: usize) -> Option<Health> {
        self.health.get(slot).copied()
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// The slot to pass semaphores from for the next acquire.
    ///
    /// Answers before the acquire call, because `vkAcquireNextImageKHR` takes
    /// the semaphore as an argument: the caller needs a usable slot in hand to
    /// make the call at all, and only learns the image index from its result.
    ///
    /// # Errors
    ///
    /// [`NoSlot`] when every slot is in flight or awaiting recreated
    /// semaphores. Never a wait: a device that blocked here would let a slow
    /// display stop everything else, which is the rule
    /// [`reims_vgpu_core::present`] states for the semantic half.
    pub fn next_slot(&mut self) -> Result<usize, NoSlot> {
        match (0..self.slots.len())
            .find(|&i| self.slots[i] == SlotState::Free && self.health[i].ready())
        {
            Some(slot) => Ok(slot),
            None => {
                self.census.refused += 1;
                Err(NoSlot {
                    depth: self.slots.len(),
                    in_flight: self.in_flight(),
                    awaiting_semaphores: self.awaiting_semaphores(),
                })
            }
        }
    }

    /// The acquire succeeded and handed back `image`.
    ///
    /// `VK_SUBOPTIMAL_KHR` comes here too: it acquires an image and signals the
    /// semaphore, so it is a success with a recreation owed afterwards — and
    /// treating it as a failure strands the signal this module exists to track.
    ///
    /// # Panics
    ///
    /// If the slot is not the free, healthy one [`Self::next_slot`] handed out.
    pub fn acquired(&mut self, slot: usize, image: u32) -> Reservation {
        assert_eq!(
            self.slots[slot],
            SlotState::Free,
            "acquired into a slot that was not free"
        );
        assert!(
            self.health[slot].ready(),
            "acquired with a semaphore that must be recreated"
        );
        self.slots[slot] = SlotState::Reserved { image };
        self.census.acquired += 1;
        Reservation { slot, image }
    }

    /// The acquire call itself failed, so nothing was signalled.
    ///
    /// Takes the slot rather than a [`Reservation`], because there is no
    /// reservation: no image was acquired. Both semaphores stay usable, which
    /// is the one failure that costs nothing.
    pub fn acquire_refused(&mut self, slot: usize) {
        debug_assert_eq!(self.slots[slot], SlotState::Free);
        self.census.abandoned += 1;
    }

    /// The queue owner accepted the submission and the present went through.
    ///
    /// The only way a slot becomes [`SlotState::Submitted`], and it consumes
    /// the reservation — a receipt, not a belief.
    pub fn presented(&mut self, reservation: Reservation, at: TimelinePoint) {
        self.slots[reservation.slot] = SlotState::Submitted {
            image: reservation.image,
            at,
        };
        self.census.presented += 1;
    }

    /// The frame is being given up, and the cause says what that costs.
    ///
    /// Returns the slot's health afterwards, so the caller knows at once which
    /// of its two semaphores it now has to replace.
    pub fn abandoned(&mut self, reservation: Reservation, cause: Abandon) -> Health {
        let slot = reservation.slot;
        // Assignment rather than accumulation, and that is an argument rather
        // than a shortcut: a reservation can only exist for a slot
        // [`Self::acquired`] found healthy, so the health being overwritten is
        // always the default. A slot cannot collect a second cause on top of a
        // first, because it is not acquirable again until
        // [`Self::recreated`].
        self.health[slot] = cause.leaves();
        self.slots[slot] = SlotState::Free;
        self.census.abandoned += 1;
        self.health[slot]
    }

    /// The caller has created fresh semaphores for a slot.
    ///
    /// The old ones are **not** destroyed here and must not be destroyed by the
    /// caller yet either; see [`Self::quarantined`].
    pub fn recreated(&mut self, slot: usize) {
        if !self.health[slot].ready() {
            self.census.recreated += 1;
        }
        self.health[slot] = Health::default();
    }

    /// Slots whose replaced semaphores must be kept alive until the device is
    /// idle.
    ///
    /// A semaphore with a signal that may still arrive cannot be destroyed:
    /// `vkDestroySemaphore` requires nothing outstanding on it, and not knowing
    /// is the entire problem. The caller keeps these and retires them with the
    /// swapchain.
    #[must_use]
    pub fn quarantined(&self) -> Vec<usize> {
        (0..self.health.len())
            .filter(|&i| !self.health[i].ready())
            .collect()
    }

    /// Free every slot the timeline has passed. Returns how many.
    ///
    /// A presented slot's image is the presentation engine's until the
    /// rendering that wrote it is done, and the timeline is the only thing that
    /// knows.
    pub fn recycle(&mut self, reached: TimelinePoint) -> usize {
        let mut freed = 0;
        for slot in &mut self.slots {
            if let SlotState::Submitted { at, .. } = *slot {
                if reached.reached(at) {
                    *slot = SlotState::Free;
                    freed += 1;
                }
            }
        }
        self.census.recycled += freed;
        freed
    }

    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, SlotState::Submitted { .. }))
            .count()
    }

    #[must_use]
    pub fn reserved(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, SlotState::Reserved { .. }))
            .count()
    }

    /// Free slots that cannot be acquired into until their semaphores are
    /// replaced.
    #[must_use]
    pub fn awaiting_semaphores(&self) -> usize {
        (0..self.slots.len())
            .filter(|&i| self.slots[i] == SlotState::Free && !self.health[i].ready())
            .count()
    }

    /// Whether the swapchain may be destroyed: nothing reserved and nothing in
    /// flight.
    ///
    /// Quarantined semaphores do not block it — destroying the swapchain is
    /// exactly what makes them destroyable, since the presentation engine has
    /// nothing left to signal through.
    #[must_use]
    pub fn retirable(&self) -> bool {
        self.slots.iter().all(|s| *s == SlotState::Free)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(n: u64) -> TimelinePoint {
        TimelinePoint(n)
    }

    #[test]
    fn a_frame_goes_round_and_the_timeline_frees_it() {
        let mut ring = FrameRing::new(3);
        let slot = ring.next_slot().expect("a free slot");
        let reservation = ring.acquired(slot, 2);
        assert_eq!(reservation.image(), 2);
        assert_eq!(ring.state(slot), Some(SlotState::Reserved { image: 2 }));
        assert_eq!(ring.reserved(), 1);

        ring.presented(reservation, at(7));
        assert_eq!(
            ring.state(slot),
            Some(SlotState::Submitted {
                image: 2,
                at: at(7)
            })
        );
        assert_eq!(ring.in_flight(), 1);
        assert!(!ring.retirable());

        assert_eq!(ring.recycle(at(6)), 0);
        assert_eq!(ring.recycle(at(7)), 1);
        assert_eq!(ring.state(slot), Some(SlotState::Free));
        assert!(ring.retirable());
        assert_eq!(ring.census().presented, 1);
    }

    /// The claim the module exists for, one cause at a time.
    #[test]
    fn each_failure_suspects_exactly_the_semaphore_it_could_have_stranded() {
        for (cause, acquire, render) in [
            (
                Abandon::SubmitRefused,
                Usability::MustRecreate,
                Usability::Usable,
            ),
            (
                Abandon::PresentRefused,
                Usability::Usable,
                Usability::MustRecreate,
            ),
            (
                Abandon::SwapchainReplaced,
                Usability::MustRecreate,
                Usability::MustRecreate,
            ),
        ] {
            let mut ring = FrameRing::new(2);
            let slot = ring.next_slot().expect("free");
            let reservation = ring.acquired(slot, 0);
            let health = ring.abandoned(reservation, cause);
            assert_eq!(health.acquire, acquire, "{}", cause.slug());
            assert_eq!(health.render, render, "{}", cause.slug());
            assert_eq!(ring.state(slot), Some(SlotState::Free));
            // Free, and still not acquirable.
            assert_eq!(ring.awaiting_semaphores(), 1);
            assert_eq!(ring.quarantined(), vec![slot]);
        }
    }

    /// A refused acquire signalled nothing, which is the one failure that costs
    /// no semaphore.
    #[test]
    fn a_refused_acquire_leaves_both_semaphores_usable() {
        let mut ring = FrameRing::new(2);
        let slot = ring.next_slot().expect("free");
        ring.acquire_refused(slot);
        assert_eq!(ring.health(slot), Some(Health::default()));
        assert!(ring.quarantined().is_empty());
        // And the very same slot is handed straight back.
        assert_eq!(ring.next_slot(), Ok(slot));
    }

    /// A free slot with a suspect semaphore is not a slot, and the refusal says
    /// which of the two situations it is.
    #[test]
    fn a_suspect_slot_is_refused_and_the_refusal_says_what_to_do_about_it() {
        let mut ring = FrameRing::new(1);
        let slot = ring.next_slot().expect("free");
        let reservation = ring.acquired(slot, 0);
        ring.abandoned(reservation, Abandon::SubmitRefused);

        let refused = ring.next_slot().expect_err("nothing acquirable");
        assert_eq!(refused.depth, 1);
        assert_eq!(refused.in_flight, 0, "nothing was ever submitted");
        assert_eq!(
            refused.awaiting_semaphores, 1,
            "the caller's next move is recreation, not a wait"
        );
        assert!(refused.to_string().contains("awaiting_semaphores=1"));

        ring.recreated(slot);
        assert_eq!(ring.next_slot(), Ok(slot));
        assert_eq!(ring.census().recreated, 1);
        assert!(ring.quarantined().is_empty());
    }

    /// A slot cannot collect two causes: it stops being acquirable the moment
    /// the first one lands. That is what makes `abandoned` an assignment rather
    /// than an accumulation — and the two failures below therefore land on two
    /// different slots and owe two different semaphores.
    #[test]
    fn a_suspect_slot_cannot_take_a_second_cause_on_top_of_the_first() {
        let mut ring = FrameRing::new(2);
        let first_slot = ring.next_slot().expect("free");
        let first = ring.acquired(first_slot, 0);
        ring.abandoned(first, Abandon::SubmitRefused);
        assert_eq!(
            ring.health(first_slot).unwrap().acquire,
            Usability::MustRecreate
        );

        let second_slot = ring.next_slot().expect("the other slot");
        assert_ne!(
            second_slot, first_slot,
            "a suspect slot is not handed out again"
        );
        let second = ring.acquired(second_slot, 1);
        ring.abandoned(second, Abandon::PresentRefused);

        // One owes its acquire semaphore and one its render semaphore; neither
        // has been given the other's damage.
        assert_eq!(ring.health(first_slot).unwrap().render, Usability::Usable);
        assert_eq!(ring.health(second_slot).unwrap().acquire, Usability::Usable);
        assert_eq!(ring.quarantined(), vec![0, 1]);
        assert_eq!(ring.next_slot().unwrap_err().awaiting_semaphores, 2);

        // And recreating one brings back exactly that one.
        ring.recreated(first_slot);
        assert_eq!(ring.next_slot(), Ok(first_slot));
        assert_eq!(ring.quarantined(), vec![second_slot]);
    }

    #[test]
    fn a_recreated_healthy_slot_is_not_counted_as_a_recreation() {
        let mut ring = FrameRing::new(1);
        ring.recreated(0);
        assert_eq!(
            ring.census().recreated,
            0,
            "nothing was owed, so nothing was replaced"
        );
    }

    #[test]
    fn a_full_ring_refuses_rather_than_waiting() {
        let mut ring = FrameRing::new(2);
        for (image, point) in [(0u32, 1u64), (1, 2)] {
            let slot = ring.next_slot().expect("free");
            let reservation = ring.acquired(slot, image);
            ring.presented(reservation, at(point));
        }
        let refused = ring.next_slot().expect_err("every image is in flight");
        assert_eq!(refused.in_flight, 2);
        assert_eq!(refused.awaiting_semaphores, 0);
        assert_eq!(ring.census().refused, 1);
        // Nothing was taken from the presentation engine to serve the refusal.
        assert_eq!(ring.in_flight(), 2);

        assert_eq!(ring.recycle(at(1)), 1);
        assert!(ring.next_slot().is_ok());
    }

    /// A reservation that is neither presented nor abandoned strands its slot,
    /// which is why it is `#[must_use]` and consumed by both exits.
    #[test]
    fn a_reservation_has_exactly_two_exits() {
        let mut ring = FrameRing::new(1);
        let slot = ring.next_slot().expect("free");
        let reservation = ring.acquired(slot, 0);
        // Reserved is not free, so the one slot cannot be taken again.
        assert!(ring.next_slot().is_err());
        assert!(!ring.retirable(), "a reserved image is still ours");
        ring.presented(reservation, at(1));
        assert_eq!(ring.in_flight(), 1);
    }

    #[test]
    #[should_panic(expected = "not free")]
    fn acquiring_into_a_reserved_slot_is_a_construction_error() {
        let mut ring = FrameRing::new(1);
        let slot = ring.next_slot().expect("free");
        let held = ring.acquired(slot, 0);
        let _second = ring.acquired(slot, 1);
        drop(held);
    }

    #[test]
    #[should_panic(expected = "must be recreated")]
    fn acquiring_with_a_suspect_semaphore_is_a_construction_error() {
        let mut ring = FrameRing::new(1);
        let slot = ring.next_slot().expect("free");
        let reservation = ring.acquired(slot, 0);
        ring.abandoned(reservation, Abandon::SubmitRefused);
        let _ = ring.acquired(slot, 0);
    }

    #[test]
    #[should_panic(expected = "cannot present")]
    fn a_swapchain_with_no_images_is_a_construction_error() {
        let _ = FrameRing::new(0);
    }

    #[test]
    fn quarantine_does_not_block_swapchain_teardown() {
        let mut ring = FrameRing::new(2);
        let slot = ring.next_slot().expect("free");
        let reservation = ring.acquired(slot, 0);
        ring.abandoned(reservation, Abandon::SwapchainReplaced);
        assert_eq!(ring.quarantined(), vec![slot]);
        assert!(
            ring.retirable(),
            "destroying the swapchain is what makes them destroyable"
        );
    }

    #[test]
    fn every_cause_names_itself_and_its_effect() {
        for cause in [
            Abandon::SubmitRefused,
            Abandon::PresentRefused,
            Abandon::SwapchainReplaced,
        ] {
            assert!(cause.slug().starts_with("vk_frame_"));
            assert!(
                !cause.leaves().ready(),
                "a cause that costs nothing is not one"
            );
        }
    }
}
