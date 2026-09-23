//! One guest surface's presentation stream: what is in flight, in what order,
//! and what the old swapchain is still owed.
//!
//! # The returned image count is authoritative and the requested depth is not
//!
//! A device asks a host for a buffering depth and the host returns however many
//! images it decided to give. Every bound that matters — how many frames may be
//! in flight, when acquire has to wait — is the *returned* count. Treating the
//! request as the bound is the bug where a host that returned four images is
//! driven as though it had three, or worse as though it had five, and the
//! symptom is a stall or a validation error a long way from here. So the
//! requested depth is recorded as policy and never compared against anything.
//!
//! # Backpressure stops presentation and nothing else
//!
//! [`PresentStream::acquire`] never blocks. When every returned image is in
//! flight it refuses, and the refusal is this stream's alone: no draw, no
//! resource lifecycle and no other surface is held by it. A device that waited
//! for an image inside a general drain would let a slow display stop compute.
//!
//! # A present that cannot acquire is parked here, and nowhere else
//!
//! Refusing an acquire is not the whole answer, because the packet that asked
//! for it is a transaction the guest is waiting on: dropping it is a lost frame
//! and a completion word that never arrives. So a present with no image to take
//! parks in *this surface's* queue, in arrival order, and
//! [`PresentStream::wake`] hands it back when an image frees. Nothing outside
//! the surface is held: another surface has its own stream, and a channel with
//! no present on it never touches this queue at all. That is what
//! "backpressure does not head-of-line block" means structurally rather than by
//! convention — there is no shared queue for a slow display to fill.
//!
//! # Order is FIFO unless the contract says a pending present may be replaced
//!
//! Presenting in submission order is the default and the only safe assumption.
//! Some presentation contracts allow a newer frame to supersede one that has
//! not been shown yet, which is a real latency win and a real correctness
//! hazard: superseding a frame the guest was told was presented is a dropped
//! frame the guest cannot account for. [`Order`] is therefore a value this
//! stream is constructed with, chosen from a structural capability by the layer
//! that has one, and never inferred here.
//!
//! # A replaced swapchain is retired, not destroyed
//!
//! Resizing or reconfiguring produces a new swapchain generation while the old
//! one's images are still being read by submitted work. The old generation goes
//! to deferred retirement against the timeline point of its last use — the same
//! exactness [`crate::retire`] applies to every other native object — and the
//! caller gets it back when that point is reached, never before.

use crate::identity::{CompletionStamp, IngressOrdinal, MappingId, TaskId, TimelinePoint};
use reims_vgpu_protocol::packets::Channel;
pub use reims_vgpu_protocol::present::PresentForm;
use std::collections::VecDeque;
use std::num::NonZeroU64;

/// One present packet, decoded.
///
/// What the guest asked to show, and — for the two forms that name one — the
/// task that owns it. The pipe or display index at word zero is deliberately
/// absent: no path in this device reads it, and a field nothing reads is one
/// that quietly acquires a wrong meaning. It stays available on
/// [`reims_vgpu_protocol::present::Trailer`] for the layer that has a display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentPacket {
    pub form: PresentForm,
    /// What to show. See [`MappingId`] for why this is not an object-list ref
    /// even though both arrive as `u32`.
    pub mapping: MappingId,
    /// The task that owns what is being shown, for the two forms whose trailer
    /// names one. `None` for [`PresentForm::SwapMapping`], whose word at that
    /// position is unidentified — reading it as a task would report whatever it
    /// happens to hold.
    pub task: Option<TaskId>,
}

/// Why a present packet's bytes did not become one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveRefusal {
    /// The packet is not a present.
    NotPresent { channel: Channel, opcode: u16 },
    /// The payload is shorter than the emitting command's own trailer, so
    /// there is nothing to show. Refused rather than clamped: presenting
    /// mapping zero and completing the packet in silence is a frame the guest
    /// believes it showed.
    Payload(reims_vgpu_protocol::present::Refusal),
}

impl ResolveRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NotPresent { .. } => "present_not_a_present_packet",
            Self::Payload(inner) => inner.slug(),
        }
    }
}

/// Turn one present packet into what it asks to show.
///
/// **The join between the three trailers and the model.** The wire could read
/// a present's words and this crate could name which of the three forms a
/// packet is, and nothing carried a packet from one to the other — so
/// [`crate::transaction::Payload::Present`] named a form and lost the target.
///
/// Where the frame goes and what has to happen for it to arrive are
/// [`PresentStream`]'s and the display layer's. This says what the guest asked
/// for.
///
/// # Errors
///
/// [`ResolveRefusal`]: a packet that is not a present, or one too short to
/// carry its own trailer.
pub fn resolve(
    channel: Channel,
    opcode: u16,
    payload: &[u8],
) -> Result<PresentPacket, ResolveRefusal> {
    let Some(form) = PresentForm::of(channel, opcode) else {
        return Err(ResolveRefusal::NotPresent { channel, opcode });
    };
    let trailer =
        reims_vgpu_protocol::present::trailer(form, payload).map_err(ResolveRefusal::Payload)?;
    Ok(PresentPacket {
        form,
        mapping: MappingId(trailer.target),
        task: trailer.task.map(TaskId),
    })
}

/// One configuration of a surface's swapchain.
///
/// Changes on every reconfiguration, so a ticket from before a resize cannot be
/// completed against the swapchain that replaced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SwapchainGeneration(NonZeroU64);

impl SwapchainGeneration {
    pub const FIRST: SwapchainGeneration = SwapchainGeneration(NonZeroU64::MIN);

    #[must_use]
    pub const fn next(self) -> SwapchainGeneration {
        match NonZeroU64::new(self.0.get().saturating_add(1)) {
            Some(n) => SwapchainGeneration(n),
            None => SwapchainGeneration::FIRST,
        }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Whether a pending present may be replaced by a newer one.
///
/// Not a guess and not a device name: the layer that knows the presentation
/// contract passes it in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Every acquired frame is presented, in the order it was acquired.
    Fifo,
    /// A frame that has not been shown yet may be superseded by a newer one.
    Superseding,
}

/// A frame in flight, and where it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// An image has been reserved and the frame is not yet drawn.
    AcquirePending,
    /// The frame is drawn and may be queued.
    Ready,
    /// Handed to the host's presentation queue.
    Queued,
}

/// One frame's place in the stream.
///
/// Not `Clone`: two copies of a ticket are two claims on one image, and the
/// second one to be completed would release an image the first still holds.
#[derive(Debug, PartialEq, Eq)]
pub struct Ticket {
    pub generation: SwapchainGeneration,
    /// Position in this generation's presentation order.
    pub sequence: u64,
    /// Which of the returned images this frame occupies.
    pub image: usize,
}

/// Why the stream would not do something.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Every returned image is in flight. Presentation waits; nothing else
    /// does.
    NoFreeImage { images: usize },
    /// The ticket belongs to a swapchain that has been replaced.
    StaleGeneration {
        named: SwapchainGeneration,
        current: SwapchainGeneration,
    },
    /// The frame is not in the phase this transition starts from.
    WrongPhase { at: Phase, expected: Phase },
    /// A frame was asked to be presented ahead of an earlier one under an
    /// order that does not permit it.
    OutOfOrder { head: u64, named: u64 },
    /// A frame was superseded by a newer one, under an order that permits it.
    /// Not an error: the caller owes the guest this fact and owes the host
    /// nothing.
    Superseded { by: u64 },
    /// A newer frame was queued ahead of an earlier one that has not been
    /// drawn yet.
    ///
    /// Superseding relaxes *presentation* order; it says nothing about who owns
    /// an image. A frame still in [`Phase::AcquirePending`] holds a claim this
    /// stream handed out and the caller has not returned — it is being written
    /// into right now — so dropping it would free the image under its writer and
    /// hand the same one to the next acquire. That is the second claim on one
    /// image that [`Ticket`] not being `Clone` exists to make unspellable, and
    /// this stream must not create it either.
    UndrawnFrameAhead { sequence: u64 },
    /// No swapchain has been configured, so there is nothing to acquire from.
    NotConfigured,
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoFreeImage { .. } => "present_no_free_image",
            Self::StaleGeneration { .. } => "present_stale_generation",
            Self::WrongPhase { .. } => "present_wrong_phase",
            Self::OutOfOrder { .. } => "present_out_of_order",
            Self::Superseded { .. } => "present_superseded",
            Self::UndrawnFrameAhead { .. } => "present_undrawn_frame_ahead",
            Self::NotConfigured => "present_not_configured",
        }
    }
}

/// A swapchain whose images are still being read by submitted work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a retired swapchain nothing retires is a leak"]
pub struct Retired {
    pub generation: SwapchainGeneration,
    /// The point after which nothing reads its images.
    pub last_use: TimelinePoint,
}

/// What replacing a swapchain left behind.
///
/// Two obligations, not one. The swapchain is the caller's to destroy once
/// [`PresentStream::reached`] gives it back, and the frames are the caller's
/// to answer to the guest now: each was an admitted present with a completion
/// word owed, and the generation they were acquired from no longer exists, so
/// they will never be shown. Handing them back by name is the same discipline
/// [`PresentStream::queue`] applies to a superseded frame and
/// [`PresentStream::abandon_parked`] to a parked one — there is no path here
/// that discards an admitted present without giving it to somebody.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "a retired swapchain nothing retires is a leak, and a dropped frame nobody answers for is a completion word nothing publishes"]
pub struct Replaced {
    pub retired: Retired,
    /// The sequences that were in flight on the old generation, in order.
    pub dropped: Vec<u64>,
}

/// Everything a presentation stream was holding when its device incarnation
/// ended.
///
/// Three obligations, and the third is the one that could not be met any other
/// way. A swapchain normally leaves through [`PresentStream::reached`], at the
/// timeline point after which nothing reads its images — and a lost device
/// will never reach any point, because the thing that would advance the
/// timeline is what was lost. Waiting for it is how a device loss becomes a
/// leak and a hang instead of a transition, which is the argument
/// [`crate::retire::NativeRetirement::epoch_lost`] makes for the same shape.
///
/// The swapchains come back as bare generations rather than as [`Retired`],
/// for the reason `retire` returns [`crate::retire::Abandoned`] rather than
/// [`crate::retire::Retired`] there: a `Retired` carries the point its images
/// stop being read, and offering one for a swapchain whose device is gone
/// would be offering a caller a wait it must not take.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "a lost stream's frames are completion words nothing publishes and its swapchains are objects nothing tears down"]
pub struct Lost {
    /// Frames that held an image of the current swapchain, oldest first. Each
    /// was an admitted present the guest is still waiting on.
    pub dropped: Vec<u64>,
    /// Presents that never got an image, in arrival order.
    pub parked: Vec<PresentRequest>,
    /// Every swapchain the stream held — the current one and the ones already
    /// deferred — oldest first. None of them may be waited for.
    pub swapchains: Vec<SwapchainGeneration>,
}

/// A present packet that has not got an image yet.
///
/// Carries the completion word because the packet's obligation travels with it:
/// a parked present that lost its stamp is a guest waiting on a word nothing
/// can publish any more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentRequest {
    pub ingress: IngressOrdinal,
    pub stamp: Option<CompletionStamp>,
}

/// What admitting a present packet did.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a present that is neither acquired nor parked is a frame nothing will ever show"]
pub enum Admission {
    /// An image was free and the frame may be drawn.
    Acquired {
        request: PresentRequest,
        ticket: Ticket,
    },
    /// Every returned image is in flight. The request is held in this surface's
    /// own queue behind `ahead` others, and nothing outside this surface is
    /// held with it.
    Parked { ahead: usize },
    /// No swapchain has been configured. Parking would be waiting for an image
    /// from a swapchain that does not exist, so this is the one present the
    /// caller has to answer for itself.
    NotConfigured,
}

#[derive(Debug)]
struct InFlight {
    sequence: u64,
    image: usize,
    phase: Phase,
}

/// One guest surface's presentation stream.
#[derive(Debug)]
pub struct PresentStream {
    order: Order,
    generation: SwapchainGeneration,
    /// What the host actually returned. The only bound anything here uses.
    images: Option<usize>,
    /// What was asked for. Policy, recorded so a report can say what the host
    /// did with it, and never compared against anything.
    requested_depth: Option<usize>,
    next_sequence: u64,
    in_flight: VecDeque<InFlight>,
    /// Present packets waiting for an image, oldest first.
    parked: VecDeque<PresentRequest>,
    retiring: Vec<Retired>,
    presented: usize,
    superseded: usize,
    backpressured: usize,
}

impl PresentStream {
    #[must_use]
    pub fn new(order: Order) -> Self {
        Self {
            order,
            generation: SwapchainGeneration::FIRST,
            images: None,
            requested_depth: None,
            next_sequence: 0,
            in_flight: VecDeque::new(),
            parked: VecDeque::new(),
            retiring: Vec::new(),
            presented: 0,
            superseded: 0,
            backpressured: 0,
        }
    }

    #[must_use]
    pub const fn order(&self) -> Order {
        self.order
    }

    #[must_use]
    pub const fn generation(&self) -> SwapchainGeneration {
        self.generation
    }

    /// The image count the host returned, which is the only bound this stream
    /// uses.
    #[must_use]
    pub const fn images(&self) -> Option<usize> {
        self.images
    }

    /// The depth that was requested. Policy; nothing here compares against it.
    #[must_use]
    pub const fn requested_depth(&self) -> Option<usize> {
        self.requested_depth
    }

    /// Configure the first swapchain.
    ///
    /// The first one of a stream, and the first one after
    /// [`Self::device_lost`] left it unconfigured. A stream that already has a
    /// swapchain is reconfigured through [`Self::replace`], which is the door
    /// that retires the old one.
    ///
    /// # Panics
    ///
    /// If the host returned no images. A swapchain of zero images cannot
    /// present, and admitting one would turn every acquire into backpressure
    /// that never lifts.
    ///
    /// If a swapchain already exists. The module's rule is that "a replaced
    /// swapchain is retired, not destroyed", and overwriting the image count
    /// here retires nothing: the old swapchain never reaches `retiring`, so no
    /// [`Self::reached`] ever hands it back and nothing destroys it. Worse
    /// than the leak, the generation does not advance, so the frames still in
    /// flight on the old swapchain keep tickets that the generation check
    /// accepts — including ones naming an image index the new swapchain does
    /// not have.
    pub fn configure(&mut self, requested_depth: usize, returned_images: usize) {
        assert!(
            self.images.is_none(),
            "a stream with a swapchain is reconfigured through `replace`, which retires the old one"
        );
        self.install(requested_depth, returned_images);
    }

    /// Take a swapchain's shape, whichever door asked.
    fn install(&mut self, requested_depth: usize, returned_images: usize) {
        assert!(
            returned_images > 0,
            "a swapchain of no images cannot present"
        );
        self.requested_depth = Some(requested_depth);
        self.images = Some(returned_images);
    }

    /// Replace the swapchain, handing the old generation to deferred
    /// retirement.
    ///
    /// Frames in flight on the old generation are dropped from this stream's
    /// order — they belong to a swapchain that no longer exists — and returned
    /// by name in [`Replaced::dropped`], because each is a present the guest is
    /// still waiting on. The swapchain itself is not destroyed: `last_use` is
    /// when its images stop being read, and the caller gets it back from
    /// [`Self::reached`] then.
    ///
    /// Parked presents are not touched. They never got an image, so they never
    /// belonged to the old generation, and the new one may have room for them:
    /// see [`Self::wake`].
    ///
    /// # Panics
    ///
    /// If the host returned no images, as [`Self::configure`].
    ///
    /// If there is no swapchain to replace. The retirement this returns is a
    /// promise that a swapchain exists and stops being read at `last_use`, and
    /// a stream that never configured one — or whose [`Self::device_lost`]
    /// took it — has none to promise about. Manufacturing the retirement
    /// anyway hands the caller a generation to destroy that it never created.
    /// It is the same question [`Self::device_lost`] already asks before
    /// listing the current generation among the swapchains it returns.
    /// [`Self::configure`] is the door to the first one.
    pub fn replace(
        &mut self,
        requested_depth: usize,
        returned_images: usize,
        last_use: TimelinePoint,
    ) -> Replaced {
        assert!(
            self.images.is_some(),
            "there is no swapchain to replace; `configure` is the door to the first one"
        );
        let retired = Retired {
            generation: self.generation,
            last_use,
        };
        self.retiring.push(retired);
        self.generation = self.generation.next();
        let dropped = self.in_flight.drain(..).map(|f| f.sequence).collect();
        self.next_sequence = 0;
        self.install(requested_depth, returned_images);
        Replaced { retired, dropped }
    }

    /// The host device incarnation ended: take everything the stream held.
    ///
    /// No timeline is consulted and nothing is left behind — see [`Lost`] for
    /// why waiting is not available here. The stream keeps its [`Order`],
    /// which is a property of the presentation contract rather than of the
    /// device, and is left unconfigured: there is no swapchain, so
    /// [`Self::acquire`] answers [`Refusal::NotConfigured`] until one is made
    /// against the replacement device.
    ///
    /// The generation advances, so a ticket that outlived the loss names a
    /// swapchain that is gone and is refused as [`Refusal::StaleGeneration`]
    /// rather than matching the first swapchain of the next device.
    pub fn device_lost(&mut self) -> Lost {
        let dropped = self.in_flight.drain(..).map(|f| f.sequence).collect();
        let parked = self.parked.drain(..).collect();
        let mut swapchains: Vec<SwapchainGeneration> =
            self.retiring.drain(..).map(|r| r.generation).collect();
        // The current one is the newest, and it is held only while a swapchain
        // exists at all: an unconfigured stream has none.
        if self.images.is_some() {
            swapchains.push(self.generation);
        }
        swapchains.sort_unstable();
        self.generation = self.generation.next();
        self.next_sequence = 0;
        self.images = None;
        self.requested_depth = None;
        Lost {
            dropped,
            parked,
            swapchains,
        }
    }

    /// The timeline reached `at`: take the swapchains nothing reads any more.
    ///
    /// `#[must_use]` on the method, because the lint does not look inside a
    /// `Vec` and the one on [`Retired`] therefore says nothing about a call
    /// whose result is dropped — which is the swapchain leak it names.
    #[must_use = "a retired swapchain nothing retires is a leak"]
    pub fn reached(&mut self, at: TimelinePoint) -> Vec<Retired> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.retiring.len() {
            if at.reached(self.retiring[i].last_use) {
                out.push(self.retiring.swap_remove(i));
            } else {
                i += 1;
            }
        }
        out.sort_unstable_by_key(|r| r.generation);
        out
    }

    /// Reserve an image for the next frame.
    ///
    /// Never waits. When every returned image is in flight this refuses, and
    /// the refusal holds up presentation and nothing else.
    ///
    /// # Errors
    ///
    /// [`Refusal::NotConfigured`] before a swapchain exists, and
    /// [`Refusal::NoFreeImage`] when all of them are in flight.
    pub fn acquire(&mut self) -> Result<Ticket, Refusal> {
        let outcome = self.take_image();
        if matches!(outcome, Err(Refusal::NoFreeImage { .. })) {
            self.backpressured += 1;
        }
        outcome
    }

    /// Reserve an image without charging the backpressure census.
    ///
    /// Separate because [`Self::wake`] asks whether an image is free on every
    /// image release, and counting those as backpressure would make the number
    /// that says whether the returned count is the bottleneck grow with how
    /// often the stream is polled rather than with how often a frame waited.
    fn take_image(&mut self) -> Result<Ticket, Refusal> {
        let images = self.images.ok_or(Refusal::NotConfigured)?;
        if self.in_flight.len() >= images {
            return Err(Refusal::NoFreeImage { images });
        }
        // Scanned rather than collected: the in-flight list is at most the
        // returned image count long — three or four — so building a `Vec` of
        // the used indexes costs an allocation per acquired frame to save
        // nothing.
        let image = (0..images)
            .find(|i| !self.in_flight.iter().any(|f| f.image == *i))
            .expect("fewer frames in flight than images");
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.in_flight.push_back(InFlight {
            sequence,
            image,
            phase: Phase::AcquirePending,
        });
        Ok(Ticket {
            generation: self.generation,
            sequence,
            image,
        })
    }

    /// Admit a present packet: give it an image, or park it for one.
    ///
    /// A request parks behind anything already parked even when an image is
    /// free, because the frame ahead of it asked first and presenting them out
    /// of arrival order is the reordering [`Order`] exists to decide about —
    /// not something a queue drains its way into.
    pub fn submit(&mut self, request: PresentRequest) -> Admission {
        if self.parked.is_empty() {
            match self.take_image() {
                Ok(ticket) => return Admission::Acquired { request, ticket },
                Err(Refusal::NotConfigured) => return Admission::NotConfigured,
                Err(_) => {}
            }
        }
        // Charged where the frame waits and not where the acquire failed. A
        // present that parks *behind* a parked one never asks for an image at
        // all, and charging only the acquire made a guest pushing a hundred
        // frames at a two-image swapchain report one — the exact wrong answer
        // to the question the number exists for.
        self.backpressured += 1;
        let ahead = self.parked.len();
        self.parked.push_back(request);
        Admission::Parked { ahead }
    }

    /// The parked presents that can acquire now, oldest first.
    ///
    /// Call after an image is released — [`Self::complete`] — or after a
    /// swapchain is replaced with more images. Returns nothing when nothing is
    /// waiting, which is the ordinary case and is not a refusal.
    #[must_use = "a woken present nobody takes is a completion word nothing publishes and an image nothing releases"]
    pub fn wake(&mut self) -> Vec<(PresentRequest, Ticket)> {
        let mut woken = Vec::new();
        while !self.parked.is_empty() {
            let Ok(ticket) = self.take_image() else { break };
            let request = self.parked.pop_front().expect("not empty");
            woken.push((request, ticket));
        }
        woken
    }

    /// Presents waiting for an image.
    #[must_use]
    pub fn parked(&self) -> usize {
        self.parked.len()
    }

    /// Give back every parked present without acquiring for it.
    ///
    /// For a teardown or a reset: the frames will not be shown, and their
    /// completion words are still owed. Dropping them silently is the hang this
    /// returns them to prevent, so there is no path that discards a parked
    /// present without handing it to somebody.
    #[must_use = "a parked present nobody takes is a completion word nothing publishes"]
    pub fn abandon_parked(&mut self) -> Vec<PresentRequest> {
        self.parked.drain(..).collect()
    }

    /// Where a ticket's frame is.
    #[must_use]
    pub fn phase(&self, ticket: &Ticket) -> Option<Phase> {
        if ticket.generation != self.generation {
            return None;
        }
        self.in_flight
            .iter()
            .find(|f| f.sequence == ticket.sequence)
            .map(|f| f.phase)
    }

    /// The frame is drawn.
    ///
    /// # Errors
    ///
    /// If the ticket is from a replaced swapchain, or the frame is not
    /// awaiting its content.
    pub fn ready(&mut self, ticket: &Ticket) -> Result<(), Refusal> {
        self.advance(ticket, Phase::AcquirePending, Phase::Ready)
    }

    /// The frame has been handed to the host's presentation queue.
    ///
    /// Under [`Order::Fifo`] a frame may only be queued when it is at the head
    /// of the order: presenting out of submission order is a reordered frame
    /// the guest cannot account for.
    ///
    /// Under [`Order::Superseding`] a newer frame may go first, and every
    /// earlier one still in flight is dropped and reported — the caller owes
    /// the guest that fact. Every earlier one must already be drawn:
    /// see [`Refusal::UndrawnFrameAhead`].
    ///
    /// # Errors
    ///
    /// As [`Self::ready`], plus [`Refusal::OutOfOrder`] under FIFO and
    /// [`Refusal::UndrawnFrameAhead`] under superseding.
    pub fn queue(&mut self, ticket: &Ticket) -> Result<Vec<u64>, Refusal> {
        self.check_generation(ticket)?;
        // **Everything that can refuse is decided before anything is dropped.**
        // The supersede used to run first and the queuing frame's own phase was
        // checked after it, so a `queue` of a frame that was not drawn yet
        // dropped every earlier frame, freed their images, and then returned
        // `WrongPhase` — discarding the `dropped` list. The caller was told
        // only that its own phase was wrong, so it never reported those frames
        // to the guest and never stopped drawing into images this stream had
        // already handed to the next acquire.
        let Some(at) = self
            .in_flight
            .iter()
            .position(|f| f.sequence == ticket.sequence)
        else {
            return Err(Refusal::Superseded {
                by: self.next_sequence,
            });
        };
        if self.in_flight[at].phase != Phase::Ready {
            return Err(Refusal::WrongPhase {
                at: self.in_flight[at].phase,
                expected: Phase::Ready,
            });
        }
        // **A queued frame is the host's, not this stream's.** Both orders
        // reason about the earlier frames this stream still owns — the ones in
        // `AcquirePending` or `Ready`. A frame in `Phase::Queued` has been
        // handed to the host's presentation queue and its image is being read
        // there, so neither order may pass judgement on it: FIFO must not
        // refuse behind it, because waiting for it to complete would cap the
        // host at one outstanding present however many images it returned, and
        // superseding must not drop it, because freeing its image to the next
        // acquire while the presentation engine reads it is the same second
        // claim on one image that [`Refusal::UndrawnFrameAhead`] names one
        // phase earlier.
        let mut dropped = Vec::new();
        let earlier_unqueued =
            |f: &InFlight| f.sequence < ticket.sequence && f.phase != Phase::Queued;
        match self.order {
            Order::Fifo => {
                if let Some(behind) = self.in_flight.iter().find(|f| earlier_unqueued(f)) {
                    return Err(Refusal::OutOfOrder {
                        head: behind.sequence,
                        named: ticket.sequence,
                    });
                }
            }
            Order::Superseding => {
                if let Some(undrawn) = self
                    .in_flight
                    .iter()
                    .find(|f| f.sequence < ticket.sequence && f.phase == Phase::AcquirePending)
                {
                    return Err(Refusal::UndrawnFrameAhead {
                        sequence: undrawn.sequence,
                    });
                }
                // Nothing can refuse from here, so the drops are final.
                let mut i = 0;
                while i < self.in_flight.len() {
                    if earlier_unqueued(&self.in_flight[i]) {
                        dropped.push(self.in_flight[i].sequence);
                        self.in_flight.remove(i);
                        self.superseded += 1;
                    } else {
                        i += 1;
                    }
                }
            }
        }
        self.in_flight
            .iter_mut()
            .find(|f| f.sequence == ticket.sequence)
            .expect("found above and never dropped: it is not earlier than itself")
            .phase = Phase::Queued;
        Ok(dropped)
    }

    /// The host has shown the frame; its image is free again.
    ///
    /// # Errors
    ///
    /// As [`Self::ready`].
    pub fn complete(&mut self, ticket: &Ticket) -> Result<(), Refusal> {
        self.check_generation(ticket)?;
        let Some(at) = self
            .in_flight
            .iter()
            .position(|f| f.sequence == ticket.sequence)
        else {
            return Err(Refusal::Superseded {
                by: self.next_sequence,
            });
        };
        if self.in_flight[at].phase != Phase::Queued {
            return Err(Refusal::WrongPhase {
                at: self.in_flight[at].phase,
                expected: Phase::Queued,
            });
        }
        self.in_flight.remove(at);
        self.presented += 1;
        Ok(())
    }

    fn check_generation(&self, ticket: &Ticket) -> Result<(), Refusal> {
        if ticket.generation == self.generation {
            return Ok(());
        }
        Err(Refusal::StaleGeneration {
            named: ticket.generation,
            current: self.generation,
        })
    }

    fn advance(&mut self, ticket: &Ticket, from: Phase, to: Phase) -> Result<(), Refusal> {
        self.check_generation(ticket)?;
        let Some(frame) = self
            .in_flight
            .iter_mut()
            .find(|f| f.sequence == ticket.sequence)
        else {
            return Err(Refusal::Superseded {
                by: self.next_sequence,
            });
        };
        if frame.phase != from {
            return Err(Refusal::WrongPhase {
                at: frame.phase,
                expected: from,
            });
        }
        frame.phase = to;
        Ok(())
    }

    /// Frames occupying an image.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// Swapchains handed to deferred retirement and not yet reached.
    #[must_use]
    pub fn awaiting_retirement(&self) -> usize {
        self.retiring.len()
    }

    /// Frames presented, frames superseded, and frames that waited for an
    /// image.
    ///
    /// The third number is what says whether the host's returned count is the
    /// bottleneck, which is not something a requested depth could answer — so
    /// it counts every frame that waited, whether it waited because no image
    /// was free or because another frame was already waiting for one.
    #[must_use]
    pub const fn census(&self) -> (usize, usize, usize) {
        (self.presented, self.superseded, self.backpressured)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::{classify, PayloadClass};
    use reims_vgpu_protocol::packets::LEDGER;

    /// A present payload whose target word sits where `form` puts it, with a
    /// different value in every other word — so a decoder reading the wrong
    /// slot cannot accidentally read the right number.
    fn present_bytes(form: PresentForm, mapping: u32, task: u32) -> Vec<u8> {
        let mut out = vec![0u8; form.trailer_len()];
        out[0..4].copy_from_slice(&0xDEADu32.to_le_bytes());
        let t = form.target_offset();
        out[t..t + 4].copy_from_slice(&mapping.to_le_bytes());
        if let Some(k) = form.task_offset() {
            out[k..k + 4].copy_from_slice(&task.to_le_bytes());
        }
        out
    }

    /// The join: a guest's present bytes become what it asked to show, on all
    /// three forms, with each form reading its own slots.
    #[test]
    fn a_present_payload_becomes_what_the_guest_asked_to_show() {
        for (channel, opcode, form) in [
            (Channel::Child, 0x06u16, PresentForm::Transaction2),
            (Channel::Child, 0x07, PresentForm::Transaction3),
            (Channel::Child, 0x08, PresentForm::SwapMapping),
        ] {
            let bytes = present_bytes(form, 0x5e, 0x77);
            assert_eq!(
                resolve(channel, opcode, &bytes),
                Ok(PresentPacket {
                    form,
                    mapping: MappingId(0x5e),
                    task: form.names_a_task().then_some(TaskId(0x77)),
                }),
                "{}",
                form.name()
            );
        }
    }

    /// The number that says whether the returned image count is the
    /// bottleneck counts every frame that waited.
    ///
    /// A present that parks behind a parked one never asks for an image, so
    /// charging the failed acquire counted the first frame of a queue and none
    /// of the rest — a guest pushing frames at a two-image swapchain reported
    /// one, whatever the depth of the queue behind it.
    #[test]
    fn every_frame_that_waited_for_an_image_is_counted() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let mut waited = 0;
        for n in 0..10 {
            match s.submit(request(n)) {
                Admission::Acquired { .. } => {}
                Admission::Parked { .. } => waited += 1,
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(waited, 8, "two images, ten frames");
        assert_eq!(
            s.census().2,
            waited,
            "and the census says so, not just about the first one"
        );
    }

    /// The two swapchain doors are total between them: `configure` is the only
    /// way in from unconfigured and `replace` the only way from configured.
    ///
    /// Configuring over a live swapchain retired nothing — so nothing ever
    /// destroyed it — and left the frames in flight on it holding tickets a
    /// generation that never advanced still accepts, one of which can name an
    /// image index the new swapchain does not have.
    #[test]
    #[should_panic(expected = "reconfigured through `replace`")]
    fn configuring_over_a_live_swapchain_is_not_a_door() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(3, 3);
        s.configure(2, 2);
    }

    /// And the mirror: the retirement `replace` returns promises a swapchain
    /// exists and stops being read, which an unconfigured stream cannot
    /// promise. `device_lost` already asks this question before listing the
    /// current generation among the swapchains it hands back.
    #[test]
    #[should_panic(expected = "no swapchain to replace")]
    fn replacing_a_swapchain_that_was_never_made_is_not_a_door() {
        let mut s = PresentStream::new(Order::Fifo);
        let _ = s.replace(2, 2, at(1));
    }

    /// The door a device loss leaves open is `configure`, because the loss
    /// took the swapchain with it.
    #[test]
    fn a_stream_that_lost_its_device_configures_its_next_swapchain() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(3, 3);
        let lost = s.device_lost();
        assert_eq!(lost.swapchains.len(), 1, "the one that existed");
        assert_eq!(s.images(), None);
        s.configure(2, 2);
        assert_eq!(s.images(), Some(2));
    }

    /// The swap form's second word is unidentified, so nothing may report it as
    /// a task — and the two transaction forms keep theirs in *different* slots,
    /// so one read at the other's would be a plausible wrong answer.
    #[test]
    fn no_form_reads_another_forms_slots() {
        // op6's twelve bytes, padded to op7's thirty-six so the short refusal
        // is not what this test measures.
        let mut two = present_bytes(PresentForm::Transaction2, 0x5e, 0x77);
        two.resize(PresentForm::Transaction3.trailer_len(), 0);
        assert_eq!(
            resolve(Channel::Child, 0x07, &two).expect("long enough"),
            PresentPacket {
                form: PresentForm::Transaction3,
                mapping: MappingId(0x77),
                task: Some(TaskId(0x5e)),
            },
            "reading op6's payload as op7 must swap the two, not agree with it"
        );
        let swap = present_bytes(PresentForm::SwapMapping, 0x5e, 0);
        assert_eq!(
            resolve(Channel::Child, 0x08, &swap)
                .expect("long enough")
                .task,
            None
        );
    }

    /// Zero is a value the guest sends and it means nothing to show. A present
    /// carrying it is well formed, and its completion is owed in full.
    #[test]
    fn a_present_of_nothing_is_a_well_formed_present() {
        let bytes = present_bytes(PresentForm::SwapMapping, 0, 0);
        assert_eq!(
            resolve(Channel::Child, 0x08, &bytes)
                .expect("well formed")
                .mapping,
            MappingId(0)
        );
    }

    /// A payload too short to carry its own trailer is refused, not clamped:
    /// presenting mapping zero and completing in silence is a frame the guest
    /// believes it showed.
    #[test]
    fn a_present_too_short_for_its_trailer_is_refused() {
        for (opcode, form) in [
            (0x06u16, PresentForm::Transaction2),
            (0x07, PresentForm::Transaction3),
            (0x08, PresentForm::SwapMapping),
        ] {
            let need = form.trailer_len();
            let bytes = present_bytes(form, 0x5e, 0x77);
            let refusal = resolve(Channel::Child, opcode, &bytes[..need - 1])
                .expect_err("one byte under its trailer");
            assert_eq!(
                refusal,
                ResolveRefusal::Payload(reims_vgpu_protocol::present::Refusal::Short {
                    have: need - 1,
                    need,
                }),
                "{}",
                form.name()
            );
            assert_eq!(refusal.slug(), "display_present_short");
        }
    }

    /// And a packet that is not a present does not become one, whatever its
    /// payload holds.
    #[test]
    fn a_packet_that_is_not_a_present_does_not_resolve_to_one() {
        for p in LEDGER {
            if PresentForm::of(p.channel, p.opcode).is_some() {
                continue;
            }
            assert_eq!(
                resolve(p.channel, p.opcode, &[0u8; 64]),
                Err(ResolveRefusal::NotPresent {
                    channel: p.channel,
                    opcode: p.opcode
                }),
                "{} {:#04x} ({})",
                p.channel.name(),
                p.opcode,
                p.name
            );
        }
    }

    /// The claim every other payload class already makes about itself: the
    /// class's vocabulary is exhaustive over what the ledger judged into it,
    /// and empty over everything else. Present was the one class whose
    /// `every_judged_packet_reaches_a_class_and_a_meaning_within_it` arm was a
    /// hardcoded `true`, which is not a check.
    #[test]
    fn every_present_packet_has_exactly_one_form() {
        let mut seen: Vec<PresentForm> = Vec::new();
        for p in LEDGER {
            let form = PresentForm::of(p.channel, p.opcode);
            let is_present = classify(p.channel, p.opcode) == Some(PayloadClass::Present);
            assert_eq!(
                form.is_some(),
                is_present,
                "{} {:#04x} ({}) is classified {:?} and resolves to {:?}",
                p.channel.name(),
                p.opcode,
                p.name,
                classify(p.channel, p.opcode),
                form
            );
            if let Some(form) = form {
                assert!(
                    !seen.contains(&form),
                    "{} is two packets' form",
                    form.name()
                );
                seen.push(form);
            }
        }
        assert_eq!(seen.len(), 3, "the three present forms");
    }

    fn at(n: u64) -> TimelinePoint {
        TimelinePoint(n)
    }

    fn present(stream: &mut PresentStream) -> Ticket {
        let ticket = stream.acquire().expect("an image");
        stream.ready(&ticket).expect("acquired");
        stream.queue(&ticket).expect("at the head");
        ticket
    }

    /// The count that bounds anything is the one the host returned.
    #[test]
    fn the_returned_image_count_bounds_the_stream_and_the_request_does_not() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(3, 2);
        assert_eq!(s.requested_depth(), Some(3));
        assert_eq!(s.images(), Some(2));

        let first = s.acquire().expect("first of two");
        let second = s.acquire().expect("second of two");
        assert_ne!(first.image, second.image, "two frames, two images");
        assert_eq!(
            s.acquire(),
            Err(Refusal::NoFreeImage { images: 2 }),
            "three were asked for and two were given; two is the bound"
        );
        assert_eq!(s.census().2, 1);
    }

    #[test]
    fn an_unconfigured_stream_has_nothing_to_acquire_from() {
        let mut s = PresentStream::new(Order::Fifo);
        assert_eq!(s.acquire(), Err(Refusal::NotConfigured));
    }

    /// Completing a frame frees its image, and the freed image is reused.
    #[test]
    fn a_presented_frame_returns_its_image_to_the_stream() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let first = present(&mut s);
        let image = first.image;
        s.complete(&first).expect("queued");
        assert_eq!(s.in_flight(), 0, "the frame is shown and holds nothing");

        let a = s.acquire().expect("free");
        let b = s.acquire().expect("free");
        assert!([a.image, b.image].contains(&image), "the image came back");
        assert_eq!(s.census().0, 1);
    }

    /// The phases are a machine, not a set of flags.
    #[test]
    fn a_frame_cannot_be_queued_before_it_is_drawn() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let ticket = s.acquire().expect("an image");
        assert_eq!(s.phase(&ticket), Some(Phase::AcquirePending));
        assert_eq!(
            s.queue(&ticket),
            Err(Refusal::WrongPhase {
                at: Phase::AcquirePending,
                expected: Phase::Ready
            })
        );
        s.ready(&ticket).expect("acquired");
        assert_eq!(
            s.ready(&ticket),
            Err(Refusal::WrongPhase {
                at: Phase::Ready,
                expected: Phase::AcquirePending
            }),
            "and it cannot be drawn twice"
        );
        s.queue(&ticket).expect("ready");
        assert_eq!(s.phase(&ticket), Some(Phase::Queued));
    }

    #[test]
    fn a_frame_cannot_be_completed_before_it_is_queued() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let ticket = s.acquire().expect("an image");
        assert_eq!(
            s.complete(&ticket),
            Err(Refusal::WrongPhase {
                at: Phase::AcquirePending,
                expected: Phase::Queued
            })
        );
    }

    /// FIFO means what it says: the head goes first.
    #[test]
    fn fifo_refuses_to_present_a_later_frame_first() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let first = s.acquire().expect("an image");
        let second = s.acquire().expect("an image");
        s.ready(&second).expect("acquired");
        assert_eq!(
            s.queue(&second),
            Err(Refusal::OutOfOrder {
                head: first.sequence,
                named: second.sequence
            }),
            "a reordered frame is one the guest cannot account for"
        );
    }

    /// And a superseding contract drops the frames it passed, by name.
    #[test]
    fn a_superseding_stream_reports_every_frame_it_dropped() {
        let mut s = PresentStream::new(Order::Superseding);
        s.configure(3, 3);
        let first = s.acquire().expect("an image");
        let second = s.acquire().expect("an image");
        let third = s.acquire().expect("an image");
        s.ready(&third).expect("acquired");
        assert_eq!(
            s.queue(&third),
            Err(Refusal::UndrawnFrameAhead {
                sequence: first.sequence
            }),
            "an earlier frame is still being drawn into its image"
        );
        s.ready(&first).expect("acquired");
        s.ready(&second).expect("acquired");
        assert_eq!(
            s.queue(&third).expect("superseding"),
            vec![first.sequence, second.sequence],
            "the caller owes the guest these two"
        );
        assert_eq!(s.census().1, 2);
        assert_eq!(
            s.complete(&first),
            Err(Refusal::Superseded { by: 3 }),
            "a dropped frame does not complete"
        );
        s.complete(&third).expect("queued");
        assert_eq!(s.in_flight(), 0);
    }

    /// **A refused queue drops nothing.**
    ///
    /// The supersede used to run before the queuing frame's own phase was
    /// checked, so a `queue` of a frame that was not drawn yet dropped every
    /// earlier frame and freed their images — and then returned `WrongPhase`,
    /// discarding the list of what it had dropped. The caller learned only that
    /// its own phase was wrong, so it neither reported those frames to the
    /// guest nor stopped drawing into images the next acquire could hand out.
    #[test]
    fn a_queue_that_refuses_supersedes_nothing() {
        let mut s = PresentStream::new(Order::Superseding);
        s.configure(3, 3);
        let first = s.acquire().expect("an image");
        let second = s.acquire().expect("an image");
        let third = s.acquire().expect("an image");
        s.ready(&first).expect("acquired");
        s.ready(&second).expect("acquired");
        // `third` is not drawn, so this cannot be queued at all.
        assert_eq!(
            s.queue(&third),
            Err(Refusal::WrongPhase {
                at: Phase::AcquirePending,
                expected: Phase::Ready,
            })
        );
        assert_eq!(s.in_flight(), 3, "nothing was dropped");
        assert_eq!(s.census().1, 0, "and nothing was counted as superseded");
        assert_eq!(s.phase(&first), Some(Phase::Ready));
        assert_eq!(s.phase(&second), Some(Phase::Ready));
        // The frames are still theirs to present, in order.
        assert_eq!(s.queue(&first).expect("head"), Vec::<u64>::new());
        s.complete(&first).expect("queued");
    }

    /// **FIFO orders presentation; it does not serialize it.**
    ///
    /// The returned image count is the bound on frames in flight, and a stream
    /// that will hold only one frame at the host at a time cannot use it: with
    /// three images it still shows a frame every present-completion latency.
    /// So the law is that no *earlier* frame may still be unqueued, not that
    /// this frame is at the head of everything still in flight.
    #[test]
    fn fifo_lets_every_returned_image_be_at_the_host_at_once() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(3, 3);
        let first = s.acquire().expect("an image");
        let second = s.acquire().expect("an image");
        let third = s.acquire().expect("an image");
        for t in [&first, &second, &third] {
            s.ready(t).expect("acquired");
        }
        assert_eq!(s.queue(&first).expect("nothing ahead"), Vec::<u64>::new());
        assert_eq!(
            s.queue(&third),
            Err(Refusal::OutOfOrder {
                head: second.sequence,
                named: third.sequence
            }),
            "the earliest frame not yet handed over is what names the refusal"
        );
        assert_eq!(
            s.queue(&second).expect("first is queued"),
            Vec::<u64>::new()
        );
        assert_eq!(s.queue(&third).expect("both are queued"), Vec::<u64>::new());
        assert_eq!(s.in_flight(), 3, "three images, three presents at the host");
        // And they come back in the order the host shows them.
        for t in [&first, &second, &third] {
            s.complete(t).expect("queued");
        }
        assert_eq!(s.census().0, 3);
    }

    /// **A frame already handed to the host cannot be superseded.**
    ///
    /// Superseding replaces a frame *that has not been shown yet* — one this
    /// stream still owns. A queued frame is one the host owns: its image is
    /// being read by the presentation engine, and dropping it here frees that
    /// image to the next acquire while the host still holds it. That is the
    /// same second claim on one image that [`Refusal::UndrawnFrameAhead`]
    /// names one phase earlier, and it ends with a `complete` for a frame the
    /// host really did show being answered `Superseded`.
    #[test]
    fn superseding_passes_over_a_frame_the_host_already_has() {
        let mut s = PresentStream::new(Order::Superseding);
        s.configure(3, 3);
        let first = s.acquire().expect("an image");
        let second = s.acquire().expect("an image");
        let third = s.acquire().expect("an image");
        for t in [&first, &second, &third] {
            s.ready(t).expect("acquired");
        }
        s.queue(&first).expect("nothing ahead");
        // `second` is still this stream's to drop; `first` is not.
        assert_eq!(
            s.queue(&third).expect("superseding"),
            vec![second.sequence],
            "only the frame the host has not got"
        );
        assert_eq!(s.census().1, 1);
        assert_eq!(
            s.phase(&first),
            Some(Phase::Queued),
            "the host still has it and its image is still claimed"
        );
        assert_eq!(s.in_flight(), 2);
        // Which is why the host's completion for it is still an answer.
        s.complete(&first).expect("the host showed it");
        s.complete(&third).expect("queued");
        assert_eq!(s.census(), (2, 1, 0));
    }

    /// A replaced swapchain is retired against a timeline point, not destroyed.
    #[test]
    fn replacing_a_swapchain_defers_it_to_its_last_use() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let old = present(&mut s);

        let Replaced { retired, dropped } = s.replace(2, 3, at(40));
        assert_eq!(retired.generation, SwapchainGeneration::FIRST);
        assert_eq!(
            dropped,
            vec![old.sequence],
            "the frame is gone and the guest is still owed its word"
        );
        assert_eq!(s.generation(), SwapchainGeneration::FIRST.next());
        assert_eq!(s.images(), Some(3));
        assert_eq!(s.in_flight(), 0, "frames of a gone swapchain are gone");

        assert_eq!(
            s.complete(&old),
            Err(Refusal::StaleGeneration {
                named: SwapchainGeneration::FIRST,
                current: SwapchainGeneration::FIRST.next(),
            }),
            "a ticket from before the resize names a swapchain that is gone"
        );

        assert!(
            s.reached(at(39)).is_empty(),
            "submitted work is still reading its images"
        );
        assert_eq!(s.awaiting_retirement(), 1);
        assert_eq!(s.reached(at(40)), vec![retired]);
        assert_eq!(s.awaiting_retirement(), 0);
    }

    /// **A lost device leaves nothing to wait for.**
    ///
    /// Every other way out of this stream consults something the lost device
    /// was going to provide: a swapchain leaves at a timeline point, a frame
    /// leaves when the host says it showed it, a parked present leaves when an
    /// image frees. After the loss none of those will ever happen, so all
    /// three come back at once — and the swapchains come back as generations
    /// rather than as `Retired`, because a `Retired` is an offer to wait.
    #[test]
    fn a_lost_device_hands_back_everything_the_stream_was_holding() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        // One swapchain already deferred, one current with a frame on it, and
        // a present that never got an image.
        let old = present(&mut s);
        let replaced = s.replace(2, 1, at(40));
        assert_eq!(replaced.dropped, vec![old.sequence]);
        let Admission::Acquired { ticket, .. } = s.submit(request(1)) else {
            panic!("the one image was free");
        };
        assert_eq!(s.submit(request(2)), Admission::Parked { ahead: 0 });
        assert_eq!(s.awaiting_retirement(), 1);

        let current = s.generation();
        let lost = s.device_lost();
        assert_eq!(lost.dropped, vec![ticket.sequence]);
        assert_eq!(lost.parked, vec![request(2)]);
        assert_eq!(
            lost.swapchains,
            vec![replaced.retired.generation, current],
            "the deferred one and the live one, and neither with a point"
        );

        assert_eq!(s.in_flight(), 0);
        assert_eq!(s.parked(), 0);
        assert_eq!(s.awaiting_retirement(), 0, "nothing is left waiting");
        assert!(
            s.reached(at(u64::MAX)).is_empty(),
            "and no timeline can produce one afterwards"
        );
        assert_eq!(s.images(), None);
        assert_eq!(s.acquire(), Err(Refusal::NotConfigured));

        // A ticket that outlived the loss names a swapchain that is gone, and
        // is not mistaken for the first one of the next device.
        s.configure(2, 2);
        assert_eq!(
            s.complete(&ticket),
            Err(Refusal::StaleGeneration {
                named: current,
                current: current.next(),
            })
        );
        assert_eq!(s.order(), Order::Fifo, "the contract is not the device's");
        let fresh = s.acquire().expect("the replacement swapchain");
        assert_eq!(fresh.sequence, 0, "a new order starts at its own beginning");
    }

    /// Two replacements in flight retire independently and in generation order.
    #[test]
    fn several_retired_swapchains_come_back_in_order() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let first = s.replace(2, 2, at(10)).retired;
        let second = s.replace(2, 2, at(5)).retired;
        assert_eq!(s.reached(at(7)), vec![second]);
        assert_eq!(s.reached(at(10)), vec![first]);
    }

    /// Backpressure is this stream's and nothing else's: the refusal names the
    /// bound and nothing here can hold another service.
    #[test]
    fn backpressure_is_a_refusal_and_never_a_wait() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let held = s.acquire().expect("an image");
        for _ in 0..8 {
            assert_eq!(s.acquire(), Err(Refusal::NoFreeImage { images: 1 }));
        }
        assert_eq!(s.census().2, 8, "eight refusals and not one wait");
        s.ready(&held).expect("acquired");
        s.queue(&held).expect("at the head");
        s.complete(&held).expect("queued");
        s.acquire().expect("the image came back");
    }

    #[test]
    #[should_panic(expected = "cannot present")]
    fn a_swapchain_of_no_images_is_a_contract_violation() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 0);
    }
    fn request(ingress: u64) -> PresentRequest {
        PresentRequest {
            ingress: IngressOrdinal(ingress),
            stamp: Some(CompletionStamp {
                slot: crate::identity::StampSlot(1),
                value: crate::identity::StampValue(ingress as u32),
            }),
        }
    }

    /// A present with no image left does not vanish and does not wait: it
    /// parks in this surface's own queue, and the image release is what hands
    /// it back.
    #[test]
    fn a_present_with_no_image_parks_and_wakes_in_arrival_order() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(2, 2);
        let mut tickets = Vec::new();
        for i in 0..2 {
            let Admission::Acquired { ticket, .. } = s.submit(request(i)) else {
                panic!("an image was free");
            };
            tickets.push(ticket);
        }
        assert_eq!(s.submit(request(2)), Admission::Parked { ahead: 0 });
        assert_eq!(s.submit(request(3)), Admission::Parked { ahead: 1 });
        assert_eq!(s.parked(), 2);
        assert!(s.wake().is_empty(), "no image has been released yet");

        s.ready(&tickets[0]).expect("drawn");
        s.queue(&tickets[0]).expect("head of the order");
        s.complete(&tickets[0]).expect("shown");
        let woken = s.wake();
        assert_eq!(woken.len(), 1, "one image freed, one present woken");
        assert_eq!(woken[0].0, request(2), "the one that asked first");
        assert_eq!(s.parked(), 1);
        s.ready(&tickets[1]).expect("drawn");
        s.queue(&tickets[1]).expect("head of the order now");
        s.complete(&tickets[1]).expect("shown");
        assert_eq!(s.wake()[0].0, request(3));
        assert_eq!(s.parked(), 0);
    }

    /// A request parks behind anything already parked even when an image is
    /// free, or a queue drains its way into presenting frames out of order.
    #[test]
    fn a_present_never_overtakes_one_already_parked() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let Admission::Acquired { ticket, .. } = s.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(s.submit(request(1)), Admission::Parked { ahead: 0 });
        s.ready(&ticket).expect("drawn");
        s.queue(&ticket).expect("head");
        s.complete(&ticket).expect("shown");
        // An image is free now, and the newcomer still goes behind.
        assert_eq!(s.submit(request(2)), Admission::Parked { ahead: 1 });
        let woken = s.wake();
        assert_eq!(woken.len(), 1);
        assert_eq!(woken[0].0, request(1), "the older request took the image");
    }

    /// The structural half of "backpressure does not head-of-line block":
    /// there is no queue for a slow display to fill that another surface
    /// reaches.
    #[test]
    fn a_full_surface_does_not_hold_another_ones_presents() {
        let mut slow = PresentStream::new(Order::Fifo);
        slow.configure(1, 1);
        let mut fast = PresentStream::new(Order::Fifo);
        fast.configure(2, 2);
        let Admission::Acquired { .. } = slow.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(slow.submit(request(1)), Admission::Parked { ahead: 0 });
        assert!(matches!(
            fast.submit(request(2)),
            Admission::Acquired { .. }
        ));
        assert_eq!(fast.parked(), 0);
    }

    /// Resizing while presents are parked: they belong to the surface, not to
    /// the swapchain, so a larger replacement is what lets them through.
    #[test]
    fn parked_presents_survive_a_swapchain_replacement() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let Admission::Acquired { ticket, .. } = s.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(s.submit(request(1)), Admission::Parked { ahead: 0 });
        let Replaced { retired, dropped } = s.replace(3, 3, at(7));
        assert_eq!(retired.generation, SwapchainGeneration::FIRST);
        assert_eq!(
            dropped,
            vec![ticket.sequence],
            "the frame that had an image is owed; the parked one never had one"
        );
        assert_eq!(
            s.phase(&ticket),
            None,
            "the in-flight frame belonged to a swapchain that is gone"
        );
        let woken = s.wake();
        assert_eq!(woken.len(), 1, "the parked present takes a new image");
        assert_eq!(woken[0].0, request(1));
        assert_eq!(woken[0].1.generation, SwapchainGeneration::FIRST.next());
    }

    #[test]
    fn a_present_with_no_swapchain_is_not_parked() {
        let mut s = PresentStream::new(Order::Fifo);
        assert_eq!(s.submit(request(0)), Admission::NotConfigured);
        assert_eq!(
            s.parked(),
            0,
            "parking would be waiting for a swapchain that does not exist"
        );
    }

    #[test]
    fn abandoning_parked_presents_hands_them_back() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let Admission::Acquired { .. } = s.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(s.submit(request(1)), Admission::Parked { ahead: 0 });
        assert_eq!(s.submit(request(2)), Admission::Parked { ahead: 1 });
        assert_eq!(s.abandon_parked(), vec![request(1), request(2)]);
        assert_eq!(s.parked(), 0);
    }

    /// The backpressure census counts frames that waited, not polls that found
    /// nothing.
    #[test]
    fn waking_an_empty_queue_is_not_backpressure() {
        let mut s = PresentStream::new(Order::Fifo);
        s.configure(1, 1);
        let Admission::Acquired { ticket, .. } = s.submit(request(0)) else {
            panic!("the first image was free");
        };
        assert_eq!(s.submit(request(1)), Admission::Parked { ahead: 0 });
        assert_eq!(s.census().2, 1, "one frame waited");
        for _ in 0..10 {
            assert!(s.wake().is_empty());
        }
        assert_eq!(s.census().2, 1, "and ten polls did not make it ten");
        s.ready(&ticket).expect("drawn");
        s.queue(&ticket).expect("head");
        s.complete(&ticket).expect("shown");
        assert_eq!(s.wake().len(), 1);
        assert_eq!(s.census().2, 1);
    }

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

    /// **One image has one claim, and a superseded frame never keeps one.**
    ///
    /// The shadow is a per-image owner slot — `Option<sequence>` for each of
    /// the returned images — which is the whole invariant and nothing else. It
    /// has no queue, no phase machine and no order; a stream that handed an
    /// image out while its slot was occupied, or freed one it had not handed
    /// out, cannot agree with it.
    ///
    /// Driven over both orders, because the two differ in exactly the place
    /// this can go wrong: a superseding queue releases images the caller
    /// acquired and did not complete.
    #[test]
    fn one_image_has_one_claim_across_acquire_supersede_and_replace() {
        let mut presented = 0usize;
        let mut superseded = 0usize;
        let mut at_the_host = 0usize;
        let mut dropped_by_replace = 0usize;
        let mut parked_admissions = 0usize;
        let mut woken = 0usize;
        let mut undrawn_refusals = 0usize;
        let mut replaces = 0usize;
        let mut abandoned = 0usize;
        let mut lost_devices = 0usize;

        for seed in 0..512u64 {
            let mut rng = Rng::new(seed);
            let order = if seed % 2 == 0 {
                Order::Fifo
            } else {
                Order::Superseding
            };
            let mut s = PresentStream::new(order);
            let mut images = (rng.below(3) + 1) as usize;
            s.configure(images + 1, images);
            // Shadow: who owns each image, and every ticket still held.
            let mut owner: Vec<Option<u64>> = vec![None; images];
            let mut tickets: Vec<Ticket> = Vec::new();
            let mut ingress = 0u64;
            // This stream's own tally; `presented` is the whole sweep's.
            let mut presented_here = 0usize;
            // Sequences this stream has handed to the host and not yet had
            // back: the frames neither order may pass judgement on.
            let mut queued: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
            // Every swapchain generation this stream has handed back, by
            // whichever door: a deferred retirement reaching its point, or a
            // device loss taking it with no point at all. One list, because
            // handing one back twice is the failure however it left.
            let mut generations: Vec<u64> = Vec::new();

            for _ in 0..64 {
                match rng.below(48) {
                    // Acquire directly.
                    0..=9 => match s.acquire() {
                        Ok(t) => {
                            assert!(
                                owner[t.image].is_none(),
                                "seed {seed}: image {} handed out twice",
                                t.image
                            );
                            owner[t.image] = Some(t.sequence);
                            tickets.push(t);
                        }
                        Err(Refusal::NoFreeImage { images: n }) => {
                            assert_eq!(n, images);
                            assert_eq!(
                                owner.iter().filter(|o| o.is_some()).count(),
                                images,
                                "seed {seed}: refused with a free image"
                            );
                        }
                        Err(other) => panic!("seed {seed}: acquire refused as {other:?}"),
                    },
                    // Admit a present packet, which parks when nothing is free.
                    10..=15 => {
                        ingress += 1;
                        let request = PresentRequest {
                            ingress: IngressOrdinal(ingress),
                            stamp: None,
                        };
                        match s.submit(request) {
                            Admission::Acquired { ticket, .. } => {
                                assert!(owner[ticket.image].is_none(), "seed {seed}");
                                owner[ticket.image] = Some(ticket.sequence);
                                tickets.push(ticket);
                            }
                            Admission::Parked { .. } => parked_admissions += 1,
                            Admission::NotConfigured => panic!("seed {seed}: configured"),
                        }
                    }
                    // Draw a frame.
                    16..=25 => {
                        if !tickets.is_empty() {
                            let i = rng.below(tickets.len() as u64) as usize;
                            let _ = s.ready(&tickets[i]);
                        }
                    }
                    // Hand a frame to the presentation queue.
                    26..=35 => {
                        if tickets.is_empty() {
                            continue;
                        }
                        let i = rng.below(tickets.len() as u64) as usize;
                        let named = tickets[i].sequence;
                        match s.queue(&tickets[i]) {
                            Ok(dropped) => {
                                for d in &dropped {
                                    assert!(
                                        !queued.contains(d),
                                        "seed {seed}: dropped {d}, which the host already has"
                                    );
                                    let slot = owner
                                        .iter_mut()
                                        .find(|o| **o == Some(*d))
                                        .expect("a dropped frame held an image");
                                    *slot = None;
                                    queued.remove(d);
                                }
                                superseded += dropped.len();
                                queued.insert(named);
                                at_the_host = at_the_host.max(queued.len());
                            }
                            Err(Refusal::UndrawnFrameAhead { .. }) => undrawn_refusals += 1,
                            Err(
                                Refusal::OutOfOrder { .. }
                                | Refusal::WrongPhase { .. }
                                | Refusal::Superseded { .. }
                                | Refusal::StaleGeneration { .. },
                            ) => {}
                            Err(other) => panic!("seed {seed}: queue refused as {other:?}"),
                        }
                    }
                    // The host showed a frame.
                    36..=45 => {
                        if tickets.is_empty() {
                            continue;
                        }
                        let i = rng.below(tickets.len() as u64) as usize;
                        match s.complete(&tickets[i]) {
                            Ok(()) => {
                                let seq = tickets[i].sequence;
                                let slot = owner
                                    .iter_mut()
                                    .find(|o| **o == Some(seq))
                                    .expect("a completed frame held an image");
                                *slot = None;
                                queued.remove(&seq);
                                tickets.remove(i);
                                presented += 1;
                                presented_here += 1;
                            }
                            Err(
                                Refusal::WrongPhase { .. }
                                | Refusal::Superseded { .. }
                                | Refusal::StaleGeneration { .. },
                            ) => {}
                            Err(other) => panic!("seed {seed}: complete refused as {other:?}"),
                        }
                    }
                    // The device incarnation ended: everything comes back at
                    // once, and nothing is left waiting for a timeline that
                    // will never advance.
                    46 => {
                        let held = owner.iter().filter(|o| o.is_some()).count();
                        let waiting = s.parked();
                        let deferred = s.awaiting_retirement();
                        let live = s.images().is_some();
                        let lost = s.device_lost();
                        assert_eq!(
                            lost.dropped.len(),
                            held,
                            "seed {seed}: a frame went unnamed"
                        );
                        assert!(
                            lost.dropped.windows(2).all(|w| w[0] < w[1]),
                            "seed {seed}: frames handed back out of order"
                        );
                        assert_eq!(lost.parked.len(), waiting, "seed {seed}");
                        assert_eq!(
                            lost.swapchains.len(),
                            deferred + usize::from(live),
                            "seed {seed}: a swapchain went unnamed"
                        );
                        for g in &lost.swapchains {
                            assert!(
                                !generations.contains(&g.get()),
                                "seed {seed}: {g:?} handed back twice"
                            );
                            generations.push(g.get());
                        }
                        assert_eq!(s.awaiting_retirement(), 0, "seed {seed}");
                        assert!(s.reached(at(u64::MAX)).is_empty(), "seed {seed}");
                        lost_devices += 1;
                        abandoned += lost.parked.len();
                        // No swapchain until one is made against the next
                        // device.
                        images = (rng.below(3) + 1) as usize;
                        s.configure(images + 1, images);
                        owner = vec![None; images];
                        tickets.clear();
                        queued.clear();
                    }
                    // Resize: a whole new generation, and the old one deferred.
                    _ => {
                        let before = s.awaiting_retirement();
                        images = (rng.below(3) + 1) as usize;
                        let replaced = s.replace(images + 1, images, at(rng.below(8) + 1));
                        assert_eq!(
                            replaced.dropped.len(),
                            owner.iter().filter(|o| o.is_some()).count(),
                            "seed {seed}: a frame holding an image was dropped unnamed"
                        );
                        assert!(
                            replaced.dropped.windows(2).all(|w| w[0] < w[1]),
                            "seed {seed}: frames handed back out of order"
                        );
                        dropped_by_replace += replaced.dropped.len();
                        assert_eq!(s.awaiting_retirement(), before + 1, "seed {seed}");
                        replaces += 1;
                        owner = vec![None; images];
                        tickets.clear();
                        queued.clear();
                    }
                }

                // Waking never hands out an occupied image either.
                for (request, ticket) in s.wake() {
                    assert!(
                        owner[ticket.image].is_none(),
                        "seed {seed}: wake handed out image {}",
                        ticket.image
                    );
                    assert!(request.ingress.0 > 0);
                    owner[ticket.image] = Some(ticket.sequence);
                    tickets.push(ticket);
                    woken += 1;
                }

                // The observers agree with the shadow after every step.
                assert_eq!(
                    s.in_flight(),
                    owner.iter().filter(|o| o.is_some()).count(),
                    "seed {seed}: in_flight"
                );
                assert!(
                    s.in_flight() <= images,
                    "seed {seed}: more frames in flight than images"
                );
                assert_eq!(s.images(), Some(images), "seed {seed}: images");
                assert_eq!(s.census().0, presented_here, "seed {seed}: presented");
            }

            // Nothing may be dropped without being handed to somebody.
            abandoned += s.abandon_parked().len();
            assert_eq!(s.parked(), 0);
            // The deferred swapchains all come back, and only once.
            for r in s.reached(at(u64::MAX)) {
                assert!(
                    !generations.contains(&r.generation.get()),
                    "seed {seed}: {:?} retired twice",
                    r.generation
                );
                generations.push(r.generation.get());
            }
            assert!(
                generations.windows(2).all(|w| w[0] < w[1]),
                "seed {seed}: retirements are not in generation order"
            );
            assert_eq!(
                s.awaiting_retirement(),
                0,
                "seed {seed}: a swapchain leaked"
            );
        }

        // Non-vacuity: every shape an assertion above depends on reaching.
        assert!(presented > 900, "frames presented: {presented}");
        assert!(superseded > 90, "frames superseded: {superseded}");
        assert!(
            at_the_host > 1,
            "frames at the host at once: {at_the_host} --- a stream that holds              one present at a time cannot use the images it was given"
        );
        assert!(
            undrawn_refusals > 100,
            "supersedes refused for an undrawn frame ahead: {undrawn_refusals}"
        );
        assert!(
            parked_admissions > 2_000,
            "presents parked for an image: {parked_admissions}"
        );
        assert!(woken > 1_000, "parked presents woken: {woken}");
        assert!(replaces > 500, "swapchain replacements: {replaces}");
        assert!(
            lost_devices > 500,
            "device incarnations lost: {lost_devices}"
        );
        assert!(
            dropped_by_replace > 800,
            "frames a replacement handed back: {dropped_by_replace}"
        );
        assert!(abandoned > 800, "parked presents handed back: {abandoned}");
    }
}
