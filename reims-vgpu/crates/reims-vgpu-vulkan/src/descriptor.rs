//! Which mechanism carries a draw's descriptors on this host, and what one
//! emission is therefore allowed to write.
//!
//! # Three tiers, and the order is a capability ladder
//!
//! 1. **Descriptor buffers**, where the driver reports the extension *and* a
//!    qualification probe on this host passed.
//! 2. **Push descriptors**, where the extension is present and the layout fits
//!    the device's reported push limit.
//! 3. **Per-worker pools and reusable sets** otherwise.
//!
//! The ladder is ordered rather than a table because each rung is a strictly
//! narrower requirement than the one below: the pooled rung is Vulkan 1.0 and
//! is always reachable, so [`select`] is total and never refuses. There is no
//! "no descriptor mechanism" state to represent.
//!
//! # Reported support is not qualification
//!
//! `VK_EXT_descriptor_buffer` being in the extension list says the entry points
//! resolve, not that this driver's implementation of them is correct or fast.
//! The architecture asks for descriptor buffers *where the driver reports
//! support and validation and performance tests pass*, which is two facts and
//! not one, so [`DescriptorCell`] carries them separately. A host with the
//! extension and no qualification result takes push descriptors, which is why
//! an M-series host — whose documented feature set lists push descriptors and
//! dynamic rendering and not descriptor buffers — needs no name-based check to
//! land on the right rung.
//!
//! # An operator switch may only walk *down* the ladder
//!
//! [`Narrowing`] can take a rung away. It cannot add one:
//! [`reims_vgpu_config::PUSH_DESCRIPTORS`] set on where the device has no
//! extension still yields the pooled rung, because the cell is consulted first
//! and the switch is only ever a veto. That is the narrow-never-widen rule
//! expressed as a data shape rather than as a rule callers remember.
//!
//! # The tier changes what "emit only what changed" costs
//!
//! Dirty-only emission is the point of [`crate::bindings`], but it is not
//! uniformly free:
//!
//! - Pushing writes into command-buffer state, so pushing only the dirty slots
//!   is exactly right and allocates nothing.
//! - A descriptor buffer's region is memory this rail owns, so the same holds
//!   as long as the region is not one the GPU is still reading.
//! - **An allocated set a recorded draw will read must not be updated.** A
//!   descriptor set is read when the draw that bound it *executes*, not when it
//!   is bound, so the whole span from the bind to the end of that submission is
//!   closed to writes. Writing inside it is undefined behaviour, and it is
//!   undefined quietly. So on the pooled rung a dirty-only update is legal only
//!   into a set no draw has bound since it was last written; otherwise the
//!   emission takes a *fresh* set, and a fresh set is empty, so that emission
//!   must write every bound slot and not merely the changed ones.
//!
//! Submission is the *end* of that span and not its beginning, which is the
//! distinction this module gets its shape from. Two dirty draws recorded into
//! one command buffer are the case a submission-scoped rule misses entirely:
//! the second write lands in the set the first draw already bound, nothing is
//! pending yet so no rule is violated on paper, and the first draw executes
//! reading the second draw's descriptors. There is no validation error and no
//! crash --- one draw silently samples another draw's textures.
//!
//! [`SetRing`] is where that span lives, so a caller cannot express the illegal
//! update: [`SetRing::emit`] answers with the set *and* with whether the write
//! is partial or whole, and it closes the set it hands out. A draw that emits
//! nothing still closes the set it binds, through [`SetRing::bind`].
//!
//! # Exhaustion is a refusal, never a wait
//!
//! Identical to [`crate::pools`], for the same reason: blocking in a recording
//! path turns one worker's depth into every worker's latency, and recycling a
//! set the GPU may still read is the corruption this module exists to prevent.

use crate::bindings::{BindingTable, Dirty};
use reims_vgpu_config::Switch;
use reims_vgpu_core::identity::TimelinePoint;

/// What this device offers for carrying descriptors, as capabilities rather
/// than names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DescriptorCell {
    /// `VK_KHR_push_descriptor` (or the Vulkan 1.4 promotion) is present.
    pub push_descriptor: bool,
    /// `maxPushDescriptors`. A layout with more descriptors than this cannot be
    /// pushed at all, so it is part of the capability and not a tuning number.
    pub max_push_descriptors: u32,
    /// `VK_EXT_descriptor_buffer` is present.
    pub descriptor_buffer: bool,
    /// A qualification probe for descriptor buffers ran on this host and
    /// passed.
    ///
    /// Separate from [`Self::descriptor_buffer`] deliberately; see the module
    /// doc. A caller with no probe result supplies `false`, which is the
    /// conservative direction — the pooled and push rungs are the ones every
    /// supported host is already known to execute correctly.
    pub descriptor_buffer_qualified: bool,
}

/// An operator's veto over a rung.
///
/// Every field removes a rung and none adds one. Read from the configuration
/// owner by [`Narrowing::from_env`]; taken as a value everywhere else so the
/// ladder is testable without touching the environment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Narrowing {
    /// Keep descriptor state off the push path even where the device offers it
    /// and the layout fits.
    pub no_push_descriptors: bool,
    /// Keep descriptor state off the descriptor-buffer path even where the
    /// device offers it and it qualified.
    pub no_descriptor_buffers: bool,
}

impl Narrowing {
    /// The vetoes an operator set.
    ///
    /// [`reims_vgpu_config::PUSH_DESCRIPTORS`] is a narrowing-only A/B control:
    /// only [`Switch::Off`] does anything, because `On` would have to widen to
    /// mean anything and there is nothing above the rung it names.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            no_push_descriptors: reims_vgpu_config::switch(reims_vgpu_config::PUSH_DESCRIPTORS)
                == Switch::Off,
            // No switch enables descriptor buffers, and that is the point: the
            // gate above them is a *measurement*, and an environment variable
            // that could stand in for one would be a way to widen this rail
            // onto an unqualified driver from a shell.
            no_descriptor_buffers: false,
        }
    }
}

/// Which mechanism carries descriptors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Descriptors are written into rail-owned memory and the shader reads them
    /// from there.
    DescriptorBuffer,
    /// Descriptors are pushed into command-buffer state. `max` is the device's
    /// reported limit, carried so a layout can be checked against it without
    /// consulting the cell again.
    PushDescriptor { max: u32 },
    /// Descriptors live in sets allocated from a per-worker pool.
    PooledSets,
}

impl Tier {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DescriptorBuffer => "descriptor_buffer",
            Self::PushDescriptor { .. } => "push_descriptor",
            Self::PooledSets => "pooled_sets",
        }
    }

    /// Whether an emission on this tier can be a partial write without first
    /// finding a region the GPU is not reading.
    ///
    /// True for pushing, where the write goes into command-buffer state that no
    /// submission is reading yet. False for both memory-backed tiers, where the
    /// answer depends on the timeline and is [`SetRing::emit`]'s to give.
    #[must_use]
    pub const fn partial_is_always_legal(self) -> bool {
        matches!(self, Self::PushDescriptor { .. })
    }
}

/// Why a rung was not taken. Reported so a slow host says which fact cost it
/// the faster mechanism, rather than reporting a bare tier name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Declined {
    /// The device does not offer it.
    Unsupported,
    /// The device offers it, but no qualification probe has passed here.
    Unqualified,
    /// An operator narrowed it away.
    NarrowedByOperator,
    /// The device reports a push limit of zero, so nothing can be pushed. A
    /// distinct answer from [`Self::Unsupported`]: the extension is there and
    /// the limit is what refuses.
    LimitIsZero,
}

impl Declined {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Unqualified => "unqualified",
            Self::NarrowedByOperator => "narrowed_by_operator",
            Self::LimitIsZero => "limit_is_zero",
        }
    }
}

/// The chosen rung and what the ones above it were declined for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub tier: Tier,
    /// `None` when descriptor buffers were chosen.
    pub descriptor_buffer: Option<Declined>,
    /// `None` when the ladder never reached the push rung, or took it.
    pub push_descriptor: Option<Declined>,
}

/// Walk the ladder once.
///
/// Total: the pooled rung requires nothing beyond Vulkan 1.0, so there is no
/// failure to return.
#[must_use]
pub fn select(cell: DescriptorCell, narrowing: Narrowing) -> Selection {
    let descriptor_buffer = if !cell.descriptor_buffer {
        Declined::Unsupported
    } else if narrowing.no_descriptor_buffers {
        Declined::NarrowedByOperator
    } else if !cell.descriptor_buffer_qualified {
        Declined::Unqualified
    } else {
        return Selection {
            tier: Tier::DescriptorBuffer,
            descriptor_buffer: None,
            push_descriptor: None,
        };
    };

    let push_descriptor = if !cell.push_descriptor {
        Declined::Unsupported
    } else if narrowing.no_push_descriptors {
        Declined::NarrowedByOperator
    } else if cell.max_push_descriptors == 0 {
        Declined::LimitIsZero
    } else {
        return Selection {
            tier: Tier::PushDescriptor {
                max: cell.max_push_descriptors,
            },
            descriptor_buffer: Some(descriptor_buffer),
            push_descriptor: None,
        };
    };

    Selection {
        tier: Tier::PooledSets,
        descriptor_buffer: Some(descriptor_buffer),
        push_descriptor: Some(push_descriptor),
    }
}

/// How one shader-visible layout is carried, once its descriptor count is
/// known.
///
/// The push rung is chosen per device but *applies* per layout: a layout with
/// more descriptors than `maxPushDescriptors` cannot be pushed at all, and the
/// only lawful answer for it is an allocated set. So the device-level
/// [`Selection`] narrows once more here, and a device on the push rung still
/// runs some layouts through the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutCarrier {
    DescriptorBuffer,
    Push,
    /// The layout is carried by an allocated set. `over_push_limit` says
    /// whether that is because the layout was too wide for the push rung this
    /// device is otherwise on — the number that says raising nothing local
    /// would help.
    Set {
        over_push_limit: bool,
    },
}

impl Selection {
    /// How a layout with `descriptors` bindings is carried.
    #[must_use]
    pub const fn carrier(self, descriptors: u32) -> LayoutCarrier {
        match self.tier {
            Tier::DescriptorBuffer => LayoutCarrier::DescriptorBuffer,
            Tier::PushDescriptor { max } => {
                if descriptors <= max {
                    LayoutCarrier::Push
                } else {
                    LayoutCarrier::Set {
                        over_push_limit: true,
                    }
                }
            }
            Tier::PooledSets => LayoutCarrier::Set {
                over_push_limit: false,
            },
        }
    }
}

/// What one allocated descriptor set is doing.
///
/// Orthogonal to *which* set describes the current bindings: that is the ring's
/// [`SetRing::holder`], and a holder can be in either of the last two states.
/// Keeping them apart is what lets a submitted set stay bindable — see
/// [`SetRing`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetState {
    /// Nothing worth keeping is in it, and nothing names it.
    Free,
    /// Written, and no draw has bound it since, so it may be updated in place.
    ///
    /// Reachable only before a set's first bind and again once
    /// [`SetRing::recycle`] has proved the submission that read it is over.
    Live,
    /// A recorded draw binds it and no submission carries that draw yet.
    ///
    /// Not writable: the draw reads it when it executes, so the contents are
    /// already promised. Not freeable by the timeline either, because there is
    /// no point to wait for --- only [`SetRing::abandoned`], which says the
    /// recording that bound it will never run, releases one.
    Bound,
    /// Named by a submission and readable by the GPU until this point. Writing
    /// into it before then is undefined behaviour.
    Submitted(TimelinePoint),
    /// Both at once: a submission may read it until this point, *and* a
    /// recorded draw that has not been submitted binds it.
    ///
    /// # Why one state cannot say both
    ///
    /// [`SetRing::bind`] used to leave a submitted set alone, on the reading
    /// that binding changes no contents — true, and not the whole rule.
    /// [`SetRing::recycle`] returns a submitted *holder* to [`Self::Live`],
    /// which is writable, and a draw that bound it in the meantime is then a
    /// draw whose descriptors a later dirty draw in the same recording updates
    /// underneath it. That is the exact failure
    /// `a_second_draw_in_one_recording_never_writes_the_set_the_first_one_bound`
    /// names, reached the long way round, and it is on the documented
    /// exhaustion path: a dirty draw that finds no set polls the timeline and
    /// calls `recycle` before retrying.
    ///
    /// So the timeline passing the point returns this to [`Self::Bound`] and
    /// not to `Live`: the GPU is finished with it and the recording is not.
    SubmittedAndBound(TimelinePoint),
}

/// Where an emission writes, and how much of the binding table it has to write.
///
/// A *receipt*, not a description: [`SetRing::emit`] is the only thing that
/// makes one, and handing one to [`SetRing::abandoned`] frees the set it names.
/// So it is neither `Copy` nor `Clone`, and its fields are private. A caller
/// that could copy one could abandon the same set twice — freeing a set a later
/// emission had since taken and a submission is reading — and a caller that
/// could build one from an integer could free a set it never took. Both are
/// exactly the illegal update this module claims cannot be expressed, so the
/// type stops expressing them. [`crate::frames::Reservation`] is the same
/// shape for the same reason.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "an emission that is planned and not written leaves the set stale"]
pub struct SetEmission {
    /// Index into the caller's own `VkDescriptorSet` array.
    set: usize,
    /// `true` when the set already holds the previous contents, so only the
    /// dirty slots need writing. `false` when the set is fresh and empty, so
    /// every bound slot must be written — see [`Self::whole`].
    partial: bool,
}

impl SetEmission {
    /// Index into the caller's own `VkDescriptorSet` array.
    #[must_use]
    pub const fn set(&self) -> usize {
        self.set
    }

    /// Whether the set already held the previous contents, so only the dirty
    /// slots need writing.
    #[must_use]
    pub const fn partial(&self) -> bool {
        self.partial
    }

    /// Whether the caller must write every bound slot rather than only the
    /// dirty ones.
    ///
    /// The counterpart to [`crate::bindings::BindingTable::disturb_all`]: a
    /// fresh set is precisely the case where everything the driver held is
    /// gone, so the caller disturbs the table and takes the resulting full
    /// dirty set.
    #[must_use]
    pub const fn whole(&self) -> bool {
        !self.partial
    }

    /// A receipt for a set the ring never handed out, so that tests can drive
    /// the paths that only a wrong caller could reach. Not reachable from
    /// outside the crate, which is the whole point of the type.
    #[cfg(test)]
    pub(crate) const fn forged(set: usize, partial: bool) -> Self {
        Self { set, partial }
    }
}

/// Every set is either live to somebody else or in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exhausted {
    pub depth: usize,
    pub in_flight: usize,
}

impl Exhausted {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        "vk_descriptor_no_free_set"
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

/// What the ring has been asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Emissions that updated a set in place, writing only what changed.
    pub partial: usize,
    /// Emissions that had to take a fresh set and write the whole table,
    /// because the previous set was in flight. The number that says whether the
    /// depth is buying anything.
    pub whole: usize,
    /// Sets returned by the timeline.
    pub recycled: usize,
    /// Emissions refused for want of a set.
    pub refused: usize,
    /// Emissions planned and then given up, because the recording that owned
    /// them failed before its write completed.
    pub abandoned: usize,
}

/// One worker's descriptor sets for one layout: which of them describes the
/// current bindings, and which of them may be written.
///
/// No Vulkan object: this is the part that can be wrong. The caller holds the
/// `VkDescriptorPool` and the `VkDescriptorSet` array and indexes them by
/// [`SetEmission::set`] and [`SetRing::holder`].
///
/// Per worker for the same reason [`crate::pools::WorkerPool`] is: a
/// `VkDescriptorPool` is externally synchronized, so allocation from it by two
/// workers is a driver data race whether or not they want the same set.
///
/// # Submitting a set does not invalidate its contents
///
/// The rule a descriptor set is under is that it must not be *written* while a
/// submission may read it. Binding it again is fine, and its contents still
/// describe exactly what they described before. So this ring tracks two facts
/// rather than one: `holder` is the set that matches the binding table, and
/// [`SetState`] is whether writing into it is legal yet.
///
/// Collapsing those two into "the live set" is the shape that looks right and
/// costs a frame: a steady-state draw would find no live set after every
/// submission, take a fresh one, and rewrite the entire table each frame while
/// changing nothing. Here a draw with nothing dirty calls no emission at all
/// and binds [`Self::holder`], and a draw with one changed slot after a
/// submission is the only case that pays for a whole table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetRing {
    sets: Vec<SetState>,
    /// The set whose contents match the binding table, when one does. At most
    /// one: two would be two answers to what is bound.
    holder: Option<usize>,
    census: Census,
}

impl SetRing {
    /// A ring of `depth` sets, all free.
    ///
    /// # Panics
    ///
    /// If `depth` is zero. A ring with no sets refuses every emission, which is
    /// a worker that can never draw rather than a shallow one.
    #[must_use]
    pub fn new(depth: usize) -> Self {
        assert!(depth > 0, "a worker with no descriptor sets cannot draw");
        Self {
            sets: vec![SetState::Free; depth],
            holder: None,
            census: Census::default(),
        }
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        self.sets.len()
    }

    #[must_use]
    pub fn state(&self, set: usize) -> Option<SetState> {
        self.sets.get(set).copied()
    }

    /// The set whose contents match the binding table, as an observation.
    ///
    /// `None` before anything has ever been emitted. Valid whether or not that
    /// set is in flight, which is the point. A draw calls [`Self::bind`]
    /// instead: it answers the same question and closes the set to the writes
    /// the draw's own execution forbids.
    #[must_use]
    pub const fn holder(&self) -> Option<usize> {
        self.holder
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Plan the write for an emission. Called only when something is dirty; a
    /// clean draw calls [`Self::bind`] and writes nothing.
    ///
    /// Answers with a set that is legal to write *now*: the holder while no
    /// draw has bound it since it was last written, and otherwise a free one.
    /// It never returns a set some already-recorded draw will read, and it
    /// never waits for one — see the module doc.
    ///
    /// The set it hands back becomes [`SetState::Bound`] here rather than at
    /// the caller's `vkCmdBindDescriptorSets`, because an emission *is* a write
    /// followed by a bind — that is what [`Emission::Set`] means — and a ring
    /// that waited to be told would be trusting the caller with the one rule it
    /// exists to keep. A recording that gives up says so with
    /// [`Self::abandoned`], which is the only way out that does not go through
    /// a submission.
    ///
    /// # Errors
    ///
    /// [`Exhausted`] when nothing is writable and no set is free. The caller
    /// polls the timeline, calls [`Self::recycle`], and retries or parks.
    pub fn emit(&mut self) -> Result<SetEmission, Exhausted> {
        if let Some(set) = self.holder {
            if self.sets[set] == SetState::Live {
                self.sets[set] = SetState::Bound;
                self.census.partial += 1;
                return Ok(SetEmission { set, partial: true });
            }
        }
        match self.sets.iter().position(|s| *s == SetState::Free) {
            Some(set) => {
                self.sets[set] = SetState::Bound;
                self.holder = Some(set);
                self.census.whole += 1;
                Ok(SetEmission {
                    set,
                    partial: false,
                })
            }
            None => {
                self.census.refused += 1;
                Err(Exhausted {
                    depth: self.sets.len(),
                    in_flight: self.in_flight(),
                })
            }
        }
    }

    /// A draw binds the holder without writing anything, and the set is closed
    /// to writes from here on.
    ///
    /// The clean-draw half of [`Self::emit`]. A draw with nothing dirty still
    /// reads the holder's descriptors when it executes, so a later dirty draw
    /// in the same recording must not update them in place — which is exactly
    /// what would happen if the ring only heard about draws that wrote.
    ///
    /// Returns `None` before anything has ever been emitted, in which case the
    /// draw binds no set.
    ///
    /// Idempotent, and it does not leave a submitted set alone. Binding changes
    /// no contents, which is why that looked safe; what it changes is who is
    /// entitled to the contents next. See [`SetState::SubmittedAndBound`] for
    /// the sequence that made a submitted set writable again with a recorded
    /// draw already holding it.
    #[must_use]
    pub fn bind(&mut self) -> Option<usize> {
        let set = self.holder?;
        self.sets[set] = match self.sets[set] {
            SetState::Live => SetState::Bound,
            SetState::Submitted(at) => SetState::SubmittedAndBound(at),
            unchanged => unchanged,
        };
        Some(set)
    }

    /// The command buffer was submitted, and the GPU may read what it bound
    /// until `at`.
    ///
    /// The end of the span [`SetState::Bound`] opened: from here the timeline,
    /// and nothing else, says when a set may be written again. No set's
    /// contents change, so no set changes holder.
    ///
    /// # Every bound set, and not only the holder
    ///
    /// [`Self::abandoned`]'s doc states the law this keeps: a bound set is not
    /// a leak because "the recording that bound it either reaches the queue or
    /// says here that it never will". This is the reaching-the-queue half, and
    /// it used to stamp the holder alone — which is one set, while a recording
    /// can leave several bound.
    ///
    /// Two dirty draws in one command buffer are exactly that: the first
    /// [`Self::emit`]s a set and binds it, the second finds that set
    /// [`SetState::Bound`] and takes a fresh one, which becomes the holder. The
    /// first set is then bound by a recorded draw, named by no receipt the
    /// caller can return, and reachable by nothing — `recycle` acts on
    /// `Submitted` alone and [`Self::reset`] refuses while anything is bound.
    /// At ring depth *N* that lost a set per such command buffer until `emit`
    /// answered `Exhausted { in_flight: 0 }` for the rest of the process and
    /// the pool could never be reset.
    ///
    /// Every bound set in a per-worker ring was bound by a draw in the command
    /// buffer being submitted, because a set bound by an *earlier* one is
    /// already `Submitted`. So stamping all of them is exact rather than
    /// conservative.
    ///
    /// A holder still [`SetState::Live`] is stamped too. It should not happen
    /// --- a submission that names no draw that bound the holder has no reason
    /// to reach here --- but stamping it costs one set's reuse until the
    /// timeline passes, while not stamping it would leave a set writable that
    /// a submission is reading. Between a conservative answer and a wrong one
    /// this is the conservative one.
    ///
    /// A holder already submitted has its point moved forward rather than kept:
    /// a second command buffer binding the same unchanged set means the GPU may
    /// read it until the later of the two, and keeping the earlier point would
    /// recycle it while the second submission is still running. A *non*-holder
    /// submitted set is untouched: nothing in this recording could name it, so
    /// its own point is still the last read of it.
    ///
    /// Does nothing when nothing has been emitted, which is the case where the
    /// caller submitted a command buffer that bound no descriptors.
    pub fn submitted(&mut self, at: TimelinePoint) {
        let holder = self.holder;
        for (index, state) in self.sets.iter_mut().enumerate() {
            *state = match *state {
                SetState::Bound => SetState::Submitted(at),
                SetState::Live if holder == Some(index) => SetState::Submitted(at),
                SetState::Submitted(previous) if holder == Some(index) && at > previous => {
                    SetState::Submitted(at)
                }
                // The recording reached the queue, so the bound half is
                // discharged. The read window is the later of the two
                // submissions whatever the set's role: a draw in *this*
                // command buffer bound it, which is what put it here, and the
                // earlier submission's point may still be the later one.
                SetState::SubmittedAndBound(previous) => {
                    SetState::Submitted(if at > previous { at } else { previous })
                }
                unchanged => unchanged,
            };
        }
    }

    /// Release every set the timeline has passed. Returns how many.
    ///
    /// The one place a set becomes writable again, and it consults the timeline
    /// rather than an age or a count. The holder returns to [`SetState::Live`]
    /// rather than to [`SetState::Free`]: nothing erased its contents, so the
    /// next changed slot is a partial write into it and not a fresh table.
    pub fn recycle(&mut self, reached: TimelinePoint) -> usize {
        let holder = self.holder;
        let mut freed = 0;
        for (index, set) in self.sets.iter_mut().enumerate() {
            match *set {
                SetState::Submitted(at) if reached.reached(at) => {
                    *set = if holder == Some(index) {
                        SetState::Live
                    } else {
                        SetState::Free
                    };
                    freed += 1;
                }
                // The GPU is finished with it and the recording that bound it
                // is not, so it becomes bound and nothing more. Counted as
                // freed all the same: what this returns is how many sets the
                // timeline released, and this one is no longer in flight.
                SetState::SubmittedAndBound(at) if reached.reached(at) => {
                    *set = SetState::Bound;
                    freed += 1;
                }
                _ => {}
            }
        }
        self.census.recycled += freed;
        freed
    }

    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.sets
            .iter()
            .filter(|s| matches!(s, SetState::Submitted(_) | SetState::SubmittedAndBound(_)))
            .count()
    }

    #[must_use]
    pub fn free(&self) -> usize {
        self.sets.iter().filter(|s| **s == SetState::Free).count()
    }

    /// How many sets a recorded draw binds and no submission carries yet.
    ///
    /// [`SetState::SubmittedAndBound`] is *not* one of them, and deliberately:
    /// a submission does carry it, so it is counted by [`Self::in_flight`].
    /// These four counts partition the ring — `free + live + bound + in_flight`
    /// is the depth — and a set that answered to two of them would break the
    /// one identity that says no set was lost.
    #[must_use]
    pub fn bound(&self) -> usize {
        self.sets.iter().filter(|s| **s == SetState::Bound).count()
    }

    /// Whether `vkResetDescriptorPool` is legal: nothing in flight and nothing
    /// bound.
    ///
    /// A [`SetState::Live`] holder does not block a reset — nothing reads it —
    /// but the reset destroys its contents, which is why [`Self::reset`] and
    /// not the caller is what performs the bookkeeping.
    ///
    /// A [`SetState::Bound`] set does block one, for the reason it exists: a
    /// recorded draw will read it, and a reset frees every set in the pool, so
    /// that draw would bind a set the pool has given away. That the submission
    /// has not happened yet is what makes the case easy to miss and does not
    /// make it legal — the recording has to be abandoned first, which is what
    /// [`Self::abandoned`] is for.
    #[must_use]
    pub fn resettable(&self) -> bool {
        self.in_flight() == 0 && self.bound() == 0
    }

    /// The recording that planned this emission gave it up, so whatever the
    /// emission was going to write was never written.
    ///
    /// Clears [`Self::holder`] when it names this set, which is the whole
    /// point: after a half-finished write, the set does not describe the
    /// binding table, and a later clean draw that bound it would bind
    /// descriptors that were never emitted. The next emission takes a fresh
    /// set and writes the whole table, which is the correct price for a failed
    /// one.
    ///
    /// The set itself returns to [`SetState::Free`] *unless* a submission
    /// already named it. A partial emission reuses a live holder, and another
    /// recording may have submitted that same holder in between; freeing it
    /// then would hand a set the GPU is reading to the next writer. So the
    /// in-flight case gives up the holder and nothing else, and the timeline
    /// stays the only thing that frees a submitted set.
    ///
    /// This is the only way out of [`SetState::Bound`] that is not a
    /// submission, and it is why a bound set is not a leak: the recording that
    /// bound it either reaches the queue or says here that it never will.
    pub fn abandoned(&mut self, emission: SetEmission) {
        let Some(state) = self.sets.get_mut(emission.set) else {
            return;
        };
        *state = match *state {
            // Still readable by the submission that named it; only the
            // recording's hold is given up.
            SetState::Submitted(at) | SetState::SubmittedAndBound(at) => SetState::Submitted(at),
            _ => SetState::Free,
        };
        if self.holder == Some(emission.set) {
            self.holder = None;
        }
        self.census.abandoned += 1;
    }

    /// The pool was reset, so every set is gone and nothing describes the
    /// bindings any more.
    ///
    /// # Panics
    ///
    /// If [`Self::resettable`] is false. `vkResetDescriptorPool` on a pool
    /// whose sets a submission or a recorded draw may read is undefined
    /// behaviour, and a ring that quietly tolerated it here would be recording
    /// a lie about state the caller already destroyed.
    pub fn reset(&mut self) {
        assert!(
            self.resettable(),
            "a descriptor pool with sets in flight or bound by a recording cannot be reset"
        );
        self.sets.iter_mut().for_each(|s| *s = SetState::Free);
        self.holder = None;
    }
}

/// What one emission writes, on whichever tier this device is on.
///
/// Produced by [`plan`], which is the one place that knows a push needs no set,
/// a clean table needs no emission, and a fresh set needs the whole table. A
/// caller that goes through it cannot get one of those three right and another
/// wrong.
///
/// Not `Clone`: two of the three variants carry a [`SetEmission`], and cloning
/// one would reproduce the receipt it exists to keep unique.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a planned emission that is not written leaves descriptors stale"]
pub enum Emission {
    /// Push the slots into command-buffer state. No allocation, and always
    /// partial.
    Push { slots: Dirty },
    /// Write the slots into an allocated set, then bind it.
    Set { emission: SetEmission, slots: Dirty },
    /// Write the slots into a descriptor-buffer region. Carried through the
    /// same ring for the same reason: a region a submission may be reading is
    /// no more writable than a set is.
    Buffer { emission: SetEmission, slots: Dirty },
}

impl Emission {
    /// The slots this emission writes.
    pub const fn slots(&self) -> &Dirty {
        match self {
            Self::Push { slots } | Self::Set { slots, .. } | Self::Buffer { slots, .. } => slots,
        }
    }

    /// Whether the caller had to write the whole bound table rather than only
    /// what changed.
    #[must_use]
    pub const fn whole(&self) -> bool {
        match self {
            Self::Push { .. } => false,
            Self::Set { emission, .. } | Self::Buffer { emission, .. } => emission.whole(),
        }
    }

    /// Which set or region the write lands in, when it lands in one.
    #[must_use]
    pub const fn set(&self) -> Option<usize> {
        match self {
            Self::Push { .. } => None,
            Self::Set { emission, .. } | Self::Buffer { emission, .. } => Some(emission.set()),
        }
    }
}

/// Plan what a draw has to write before it is recorded.
///
/// `Ok(None)` is the steady state and the common answer: the table is clean, so
/// there is nothing to write and the draw binds [`SetRing::holder`] unchanged.
///
/// The three tier-specific rules are applied here and nowhere else:
///
/// - Pushing goes into command-buffer state, so the dirty slots are exactly
///   what is written and the ring is never consulted.
/// - A set or region write needs somewhere the GPU is not reading, which is the
///   ring's answer.
/// - When that answer is a *fresh* set, the set is empty, so the table is
///   disturbed first and the emission carries every bound slot. That is the one
///   case a caller reading only `take_dirty` would get wrong, and its symptom
///   is a shader reading a descriptor nobody wrote.
///
/// # Errors
///
/// [`Exhausted`] when a set is needed and none can be written. Nothing has been
/// taken from the table in that case — the dirty set is untouched, so the
/// caller may park the draw and plan it again after recycling.
pub fn plan(
    carrier: LayoutCarrier,
    ring: &mut SetRing,
    table: &mut BindingTable,
) -> Result<Option<Emission>, Exhausted> {
    if table.is_clean() {
        return Ok(None);
    }
    match carrier {
        LayoutCarrier::Push => Ok(Some(Emission::Push {
            slots: table.take_dirty(),
        })),
        LayoutCarrier::Set { .. } | LayoutCarrier::DescriptorBuffer => {
            let emission = ring.emit()?;
            if emission.whole() {
                table.disturb_all();
            }
            let slots = table.take_dirty();
            Ok(Some(
                if matches!(carrier, LayoutCarrier::DescriptorBuffer) {
                    Emission::Buffer { emission, slots }
                } else {
                    Emission::Set { emission, slots }
                },
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::BindingTable;
    use reims_vgpu_core::bind::{BufferBinding, ObjectBinding};
    use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration, TimelinePoint};

    fn at(n: u64) -> TimelinePoint {
        TimelinePoint(n)
    }

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn qualified() -> DescriptorCell {
        DescriptorCell {
            push_descriptor: true,
            max_push_descriptors: 32,
            descriptor_buffer: true,
            descriptor_buffer_qualified: true,
        }
    }

    /// The M-series shape: push descriptors documented, descriptor buffers not.
    fn push_only() -> DescriptorCell {
        DescriptorCell {
            push_descriptor: true,
            max_push_descriptors: 32,
            descriptor_buffer: false,
            descriptor_buffer_qualified: false,
        }
    }

    #[test]
    fn the_ladder_is_total_and_the_bottom_rung_needs_nothing() {
        let bare = DescriptorCell::default();
        let chosen = select(bare, Narrowing::default());
        assert_eq!(chosen.tier, Tier::PooledSets);
        assert_eq!(chosen.descriptor_buffer, Some(Declined::Unsupported));
        assert_eq!(chosen.push_descriptor, Some(Declined::Unsupported));
    }

    #[test]
    fn a_qualified_host_takes_descriptor_buffers() {
        let chosen = select(qualified(), Narrowing::default());
        assert_eq!(chosen.tier, Tier::DescriptorBuffer);
        assert_eq!(chosen.descriptor_buffer, None);
    }

    #[test]
    fn reported_support_without_qualification_falls_to_push() {
        let cell = DescriptorCell {
            descriptor_buffer_qualified: false,
            ..qualified()
        };
        let chosen = select(cell, Narrowing::default());
        assert_eq!(chosen.tier, Tier::PushDescriptor { max: 32 });
        assert_eq!(chosen.descriptor_buffer, Some(Declined::Unqualified));
        assert_eq!(chosen.push_descriptor, None);
    }

    #[test]
    fn the_m_series_shape_lands_on_push_without_naming_a_device() {
        assert_eq!(
            select(push_only(), Narrowing::default()).tier,
            Tier::PushDescriptor { max: 32 }
        );
    }

    #[test]
    fn a_zero_push_limit_is_its_own_answer() {
        let cell = DescriptorCell {
            max_push_descriptors: 0,
            ..push_only()
        };
        let chosen = select(cell, Narrowing::default());
        assert_eq!(chosen.tier, Tier::PooledSets);
        assert_eq!(chosen.push_descriptor, Some(Declined::LimitIsZero));
    }

    #[test]
    fn a_veto_walks_down_one_rung_at_a_time() {
        let no_buffers = Narrowing {
            no_descriptor_buffers: true,
            ..Narrowing::default()
        };
        let chosen = select(qualified(), no_buffers);
        assert_eq!(chosen.tier, Tier::PushDescriptor { max: 32 });
        assert_eq!(chosen.descriptor_buffer, Some(Declined::NarrowedByOperator));

        let neither = Narrowing {
            no_descriptor_buffers: true,
            no_push_descriptors: true,
        };
        let chosen = select(qualified(), neither);
        assert_eq!(chosen.tier, Tier::PooledSets);
        assert_eq!(chosen.push_descriptor, Some(Declined::NarrowedByOperator));
    }

    /// The narrow-never-widen rule as a property: for every cell, every
    /// narrowing may only move the choice *down* the ladder, never up.
    #[test]
    fn no_narrowing_can_widen_any_cell() {
        fn rung(t: Tier) -> u8 {
            match t {
                Tier::DescriptorBuffer => 2,
                Tier::PushDescriptor { .. } => 1,
                Tier::PooledSets => 0,
            }
        }
        let mut pairs = 0;
        for push in [false, true] {
            for max in [0u32, 1, 32] {
                for buffer in [false, true] {
                    for qual in [false, true] {
                        let cell = DescriptorCell {
                            push_descriptor: push,
                            max_push_descriptors: max,
                            descriptor_buffer: buffer,
                            descriptor_buffer_qualified: qual,
                        };
                        let open = rung(select(cell, Narrowing::default()).tier);
                        for a in [false, true] {
                            for b in [false, true] {
                                let narrowed = rung(
                                    select(
                                        cell,
                                        Narrowing {
                                            no_push_descriptors: a,
                                            no_descriptor_buffers: b,
                                        },
                                    )
                                    .tier,
                                );
                                assert!(narrowed <= open, "{cell:?} {a} {b}");
                                pairs += 1;
                            }
                        }
                    }
                }
            }
        }
        // Non-vacuity: the sweep has to have contained cases that actually
        // moved, or "never widens" is satisfied by a constant.
        assert_eq!(pairs, 96);
        let moved = rung(select(qualified(), Narrowing::default()).tier)
            > rung(
                select(
                    qualified(),
                    Narrowing {
                        no_push_descriptors: true,
                        no_descriptor_buffers: true,
                    },
                )
                .tier,
            );
        assert!(moved);
    }

    #[test]
    fn a_layout_wider_than_the_push_limit_takes_a_set_on_a_push_device() {
        let chosen = select(push_only(), Narrowing::default());
        assert_eq!(chosen.carrier(32), LayoutCarrier::Push);
        assert_eq!(
            chosen.carrier(33),
            LayoutCarrier::Set {
                over_push_limit: true
            }
        );
        // A pooled device's sets are not "over the limit"; there is no limit
        // that would have helped.
        let pooled = select(DescriptorCell::default(), Narrowing::default());
        assert_eq!(
            pooled.carrier(1),
            LayoutCarrier::Set {
                over_push_limit: false
            }
        );
    }

    #[test]
    fn a_second_draw_in_one_recording_never_writes_the_set_the_first_one_bound() {
        let mut ring = SetRing::new(3);
        let first = ring.emit().expect("a free set");
        assert!(first.whole());
        assert_eq!(
            ring.state(first.set),
            Some(SetState::Bound),
            "the draw that took it reads it when it executes"
        );

        // Nothing has been submitted, so a submission-scoped rule would call
        // this set writable. The draw already recorded against it would then
        // execute reading this emission's descriptors instead of its own.
        let second = ring.emit().expect("a free set");
        assert_ne!(second.set, first.set, "the bound set was rewritten");
        assert!(second.whole(), "a fresh set holds nothing");
        assert_eq!(ring.census().partial, 0);
    }

    /// The same rule, reached the long way round: through a submission, a clean
    /// draw, and the exhaustion path the caller is documented to take.
    ///
    /// A depth-1 ring is the whole point — it forces the retry to land on the
    /// one set the recorded draw is holding, which is what a deeper ring would
    /// let the free list hide.
    #[test]
    fn recycling_under_a_clean_draw_never_reopens_the_set_that_draw_bound() {
        let mut ring = SetRing::new(1);
        let dirty = ring.emit().expect("the only set");
        ring.submitted(TimelinePoint(7));
        assert_eq!(
            ring.state(dirty.set),
            Some(SetState::Submitted(TimelinePoint(7)))
        );

        // A clean draw in a *later* recording binds the holder without writing
        // it. Nothing about the contents changed, which is why leaving the
        // state alone looked correct.
        assert_eq!(ring.bind(), Some(dirty.set));

        // A dirty draw in that same recording finds nothing writable and takes
        // the documented exhaustion path: poll the timeline, recycle, retry.
        ring.emit().expect_err("the only set is in flight");
        assert_eq!(ring.recycle(TimelinePoint(7)), 1, "the GPU is done with it");

        // The GPU is done with it and the recording is not. A retry that got
        // this set would write descriptors under the clean draw already
        // recorded against them.
        ring.emit()
            .expect_err("the only set is bound by a recorded draw");
        assert_eq!(ring.state(dirty.set), Some(SetState::Bound));
        assert_eq!(ring.bound(), 1);
        assert_eq!(ring.in_flight(), 0);
        assert!(!ring.resettable(), "a recorded draw will read it");

        // And the recording's own two exits both return it: the queue, or
        // giving up.
        ring.submitted(TimelinePoint(9));
        assert_eq!(
            ring.state(dirty.set),
            Some(SetState::Submitted(TimelinePoint(9)))
        );
    }

    #[test]
    fn a_recycled_set_is_writable_in_place_again_and_only_then() {
        let mut ring = SetRing::new(3);
        let first = ring.emit().expect("a free set");
        ring.submitted(at(4));
        assert_eq!(ring.recycle(at(4)), 1);
        assert_eq!(ring.state(first.set), Some(SetState::Live));

        let second = ring.emit().expect("the live set");
        assert_eq!(second.set, first.set);
        assert!(second.partial, "nothing erased the contents");
        assert_eq!(ring.state(first.set), Some(SetState::Bound));
    }

    #[test]
    fn a_clean_draw_closes_the_set_it_binds_to_the_next_draws_write() {
        let mut ring = SetRing::new(3);
        let first = ring.emit().expect("a free set");
        ring.submitted(at(4));
        assert_eq!(ring.recycle(at(4)), 1);

        // A draw with nothing dirty writes nothing and binds the holder. It
        // still reads those descriptors when it executes.
        assert_eq!(ring.bind(), Some(first.set));
        assert_eq!(ring.state(first.set), Some(SetState::Bound));

        let second = ring.emit().expect("a free set");
        assert_ne!(second.set, first.set, "the clean draw's set was rewritten");
        assert!(second.whole());
    }

    #[test]
    fn a_submitted_set_is_never_written_again_before_the_timeline() {
        let mut ring = SetRing::new(3);
        let first = ring.emit().expect("a free set");
        ring.submitted(at(7));
        assert_eq!(ring.state(first.set), Some(SetState::Submitted(at(7))));

        let second = ring.emit().expect("a second free set");
        assert_ne!(second.set, first.set, "the submitted set was rewritten");
        assert!(
            second.whole(),
            "a fresh set is empty, so the emission owes the whole table"
        );

        // And it stays out of reach until the timeline says otherwise.
        assert_eq!(ring.recycle(at(6)), 0);
        assert_eq!(ring.recycle(at(7)), 1);
        assert_eq!(ring.state(first.set), Some(SetState::Free));
    }

    #[test]
    fn exhaustion_refuses_rather_than_recycling_the_oldest() {
        let mut ring = SetRing::new(2);
        for point in [1u64, 2] {
            assert!(ring.emit().expect("a free set").whole());
            ring.submitted(at(point));
        }
        let refused = ring.emit().expect_err("every set is in flight");
        assert_eq!(refused.depth, 2);
        assert_eq!(refused.in_flight, 2);
        assert_eq!(refused.slug(), "vk_descriptor_no_free_set");
        // Nothing was taken from the GPU to serve the refusal.
        assert_eq!(ring.in_flight(), 2);
        assert_eq!(ring.census().refused, 1);
        assert!(!ring.resettable());
    }

    #[test]
    fn submitting_with_nothing_live_is_not_a_state_change() {
        let mut ring = SetRing::new(2);
        ring.submitted(at(4));
        assert!(ring.resettable());
        assert_eq!(ring.free(), 2);
    }

    /// The whole point, end to end: a steady-state draw on the push rung writes
    /// nothing, and the same steady state on the pooled rung writes nothing
    /// too — the fresh-set cost is paid only when a submission intervened.
    #[test]
    fn a_steady_state_draw_writes_no_descriptors_on_either_rung() {
        let mut table = BindingTable::new(4, 4, 4);
        let binding = BufferBinding {
            buffer: Some(res(9)),
            offset: 0,
            stride: None,
        };
        assert!(table.bind_buffer(0, Some(binding)));

        // First draw: the slot moved, so it is emitted.
        let mut ring = SetRing::new(2);
        let slots = table.take_dirty();
        assert_eq!(slots.len(), 1);
        let emission = ring.emit().expect("a free set");
        assert!(emission.whole());

        // Second draw with no rebind: nothing dirty, so no emission happens at
        // all and the ring is never consulted.
        assert!(table.is_clean());
        assert_eq!(ring.census().whole, 1);
        assert_eq!(ring.census().partial, 0);

        // A rebind of the same value is still nothing.
        assert!(!table.bind_buffer(0, Some(binding)));
        assert!(table.is_clean());
    }

    /// After a submission, a rebind must land in a set the GPU is not reading,
    /// and that set owes the whole bound table rather than the one slot that
    /// changed.
    #[test]
    fn a_fresh_set_after_a_submission_carries_every_bound_slot() {
        let mut table = BindingTable::new(4, 4, 4);
        for slot in 0..3 {
            table.bind_texture(
                slot,
                Some(ObjectBinding {
                    object: Some(res(slot as u32 + 1)),
                    lod_clamps: None,
                }),
            );
        }
        let mut ring = SetRing::new(3);
        let first = ring.emit().expect("a free set");
        assert!(first.whole());
        let written = table.take_dirty();
        assert_eq!(written.textures.len(), 3);
        ring.submitted(at(5));

        // One slot changes.
        assert!(table.bind_texture(
            1,
            Some(ObjectBinding {
                object: Some(res(42)),
                lod_clamps: None,
            })
        ));
        let next = ring.emit().expect("a second set");
        assert!(next.whole(), "the previous set is in flight");
        // So the caller disturbs the table: everything the fresh set does not
        // hold has to be written into it.
        table.disturb_all();
        let owed = table.take_dirty();
        assert_eq!(owed.textures.len(), 3, "one changed, three owed");
        assert_eq!(ring.census().whole, 2);
    }

    /// The claim the holder exists for: across frames that change nothing, no
    /// descriptor is ever written a second time.
    #[test]
    fn an_unchanged_table_across_submissions_never_rewrites_a_set() {
        let mut ring = SetRing::new(2);
        let first = ring.emit().expect("a free set");
        assert!(first.whole());

        for frame in 1..=8u64 {
            // Nothing dirty, so `emit` is never called; the draw binds the
            // holder, which is valid whether or not it is in flight.
            assert_eq!(ring.holder(), Some(first.set));
            ring.submitted(at(frame));
        }

        assert_eq!(
            ring.census().whole,
            1,
            "the table was rewritten after a submission"
        );
        assert_eq!(ring.census().partial, 0);
        assert_eq!(ring.free(), 1, "no second set was ever taken");
    }

    /// The command buffer this ring is written around: two dirty draws, two
    /// sets, one submission. Both sets are the GPU's until the timeline says
    /// otherwise, and both come back.
    ///
    /// Stamping the holder alone left the first set `Bound` with nothing
    /// reachable to free it — `recycle` acts on submitted sets and `reset`
    /// refuses while anything is bound — so a ring of depth two was down to
    /// one set after one such command buffer and to none after two.
    #[test]
    fn a_submission_ends_the_span_of_every_set_the_command_buffer_bound() {
        let mut ring = SetRing::new(2);
        let first = ring.emit().expect("a free set");
        // The second dirty draw finds the first set bound by a recorded draw
        // and takes another, which becomes the holder.
        let second = ring.emit().expect("a second free set");
        assert_ne!(first.set, second.set);
        assert_eq!(ring.bound(), 2);

        ring.submitted(at(7));
        assert_eq!(ring.bound(), 0, "the command buffer bound both of them");
        assert_eq!(ring.in_flight(), 2);
        assert_eq!(ring.state(first.set), Some(SetState::Submitted(at(7))));

        assert_eq!(ring.recycle(at(7)), 2, "and the timeline frees both");
        assert_eq!(
            ring.state(first.set),
            Some(SetState::Free),
            "the set no draw holds is reusable"
        );
        assert_eq!(
            ring.state(second.set),
            Some(SetState::Live),
            "and the holder keeps its contents"
        );
        assert!(
            ring.resettable(),
            "nothing is left that only a receipt frees"
        );
    }

    #[test]
    fn a_recycled_holder_keeps_its_contents_so_the_next_change_is_partial() {
        let mut ring = SetRing::new(2);
        let first = ring.emit().expect("a free set");
        ring.submitted(at(5));
        assert_eq!(ring.recycle(at(5)), 1);
        assert_eq!(
            ring.state(first.set),
            Some(SetState::Live),
            "a recycled holder is writable again and still holds its contents"
        );

        let next = ring.emit().expect("the holder");
        assert_eq!(next.set, first.set);
        assert!(next.partial, "nothing erased the contents");
        assert_eq!(ring.census().whole, 1);
        assert_eq!(ring.census().partial, 1);
    }

    #[test]
    fn a_second_submission_of_one_set_extends_its_point_and_never_shortens_it() {
        let mut ring = SetRing::new(2);
        let first = ring.emit().expect("a free set");
        ring.submitted(at(9));
        // A later command buffer binds the same unchanged set.
        ring.submitted(at(12));
        assert_eq!(ring.state(first.set), Some(SetState::Submitted(at(12))));
        assert_eq!(
            ring.recycle(at(11)),
            0,
            "the later submission is still running"
        );

        // An out-of-order report cannot walk the point backwards.
        ring.submitted(at(4));
        assert_eq!(ring.state(first.set), Some(SetState::Submitted(at(12))));
        assert_eq!(ring.recycle(at(12)), 1);
    }

    #[test]
    fn a_pool_reset_leaves_nothing_describing_the_bindings() {
        let mut ring = SetRing::new(2);
        let taken = ring.emit().expect("a free set");
        assert!(taken.whole());
        ring.submitted(at(3));
        assert!(!ring.resettable());
        assert_eq!(ring.recycle(at(3)), 1);

        assert!(ring.resettable(), "a live holder is not in flight");
        ring.reset();
        assert_eq!(ring.holder(), None);
        assert_eq!(ring.free(), 2);
        // The next emission owes the whole table, because the reset destroyed
        // what the holder had.
        assert!(ring.emit().expect("a free set").whole());
    }

    #[test]
    #[should_panic(expected = "cannot be reset")]
    fn resetting_a_pool_with_sets_in_flight_is_refused() {
        let mut ring = SetRing::new(2);
        assert!(ring.emit().expect("a free set").whole());
        ring.submitted(at(3));
        ring.reset();
    }

    /// `plan` end to end on each rung: what it takes, what it owes, and the
    /// three answers that differ.
    #[test]
    fn plan_applies_the_tier_rule_the_caller_would_get_wrong() {
        let mut table = BindingTable::new(4, 4, 4);
        let mut ring = SetRing::new(2);

        // Clean: no emission on any rung, and the ring is not touched.
        assert_eq!(plan(LayoutCarrier::Push, &mut ring, &mut table), Ok(None));
        assert_eq!(ring.holder(), None);

        for slot in 0..3 {
            assert!(table.bind_texture(
                slot,
                Some(ObjectBinding {
                    object: Some(res(slot as u32 + 1)),
                    lod_clamps: None,
                })
            ));
        }

        // Push: the dirty slots, no set, always partial.
        let pushed = plan(LayoutCarrier::Push, &mut ring, &mut table)
            .expect("pushing needs no set")
            .expect("three slots moved");
        assert_eq!(pushed.slots().textures.len(), 3);
        assert_eq!(pushed.set(), None);
        assert!(!pushed.whole());
        assert_eq!(ring.holder(), None, "pushing allocates nothing");

        // A set, fresh: the whole bound table even though one slot moved.
        assert!(table.bind_texture(
            0,
            Some(ObjectBinding {
                object: Some(res(99)),
                lod_clamps: None,
            })
        ));
        let carrier = LayoutCarrier::Set {
            over_push_limit: false,
        };
        let written = plan(carrier, &mut ring, &mut table)
            .expect("a free set")
            .expect("one slot moved");
        assert!(written.whole());
        assert_eq!(written.slots().textures.len(), 3, "a fresh set is empty");
        assert_eq!(written.set(), ring.holder());

        // Once the submission that read it is over, the holder still describes
        // the table, so the next change is written in place.
        ring.submitted(at(3));
        assert_eq!(ring.recycle(at(3)), 1);
        assert!(table.bind_texture(
            1,
            Some(ObjectBinding {
                object: Some(res(100)),
                lod_clamps: None,
            })
        ));
        let again = plan(carrier, &mut ring, &mut table)
            .expect("the holder is live")
            .expect("one slot moved");
        assert!(!again.whole());
        assert_eq!(again.slots().textures.len(), 1);
        assert_eq!(again.set(), written.set());
    }

    /// A refused plan leaves the dirty set intact, so the caller can park the
    /// draw and plan it again once the timeline has released a set.
    #[test]
    fn a_refused_plan_takes_nothing_from_the_table() {
        let mut table = BindingTable::new(4, 4, 4);
        let mut ring = SetRing::new(1);
        let carrier = LayoutCarrier::Set {
            over_push_limit: false,
        };

        assert!(table.bind_buffer(
            0,
            Some(BufferBinding {
                buffer: Some(res(1)),
                offset: 0,
                stride: None,
            })
        ));
        assert!(plan(carrier, &mut ring, &mut table)
            .expect("a free set")
            .is_some());
        ring.submitted(at(4));

        assert!(table.bind_buffer(
            1,
            Some(BufferBinding {
                buffer: Some(res(2)),
                offset: 0,
                stride: None,
            })
        ));
        let refused = plan(carrier, &mut ring, &mut table).expect_err("the only set is in flight");
        assert_eq!(refused.depth, 1);
        assert!(!table.is_clean(), "the refused draw still owes its slots");

        assert_eq!(ring.recycle(at(4)), 1);
        let retried = plan(carrier, &mut ring, &mut table)
            .expect("the holder came back")
            .expect("the slots were still owed");
        assert!(!retried.whole(), "the holder kept its contents");
        assert_eq!(retried.slots().buffers.len(), 1);
    }

    /// The descriptor-buffer rung goes through the same ring, and says so.
    #[test]
    fn a_descriptor_buffer_region_is_no_more_writable_than_a_set() {
        let mut table = BindingTable::new(4, 4, 4);
        let mut ring = SetRing::new(2);
        assert!(table.bind_buffer(
            0,
            Some(BufferBinding {
                buffer: Some(res(1)),
                offset: 0,
                stride: None,
            })
        ));
        let written = plan(LayoutCarrier::DescriptorBuffer, &mut ring, &mut table)
            .expect("a free region")
            .expect("one slot moved");
        assert!(matches!(written, Emission::Buffer { .. }));
        assert!(written.whole());
    }

    #[test]
    fn only_pushing_is_unconditionally_partial() {
        assert!(Tier::PushDescriptor { max: 8 }.partial_is_always_legal());
        assert!(!Tier::PooledSets.partial_is_always_legal());
        assert!(!Tier::DescriptorBuffer.partial_is_always_legal());
    }

    #[test]
    #[should_panic(expected = "cannot draw")]
    fn a_ring_with_no_sets_is_a_construction_error() {
        let _ = SetRing::new(0);
    }

    #[test]
    fn every_declined_and_tier_names_itself() {
        for d in [
            Declined::Unsupported,
            Declined::Unqualified,
            Declined::NarrowedByOperator,
            Declined::LimitIsZero,
        ] {
            assert!(!d.name().is_empty());
        }
        for t in [
            Tier::DescriptorBuffer,
            Tier::PushDescriptor { max: 1 },
            Tier::PooledSets,
        ] {
            assert!(!t.name().is_empty());
        }
    }

    /// What the shadow says one set is, derived only from the calls made.
    ///
    /// A mirror of [`SetState`] deliberately, because the sweep's job is to
    /// hold the ring to a state machine written a second time from the outside
    /// — but the two claims it makes about an emission are *not* mirrored, and
    /// those are the ones that matter: an emitted set was neither stamped by a
    /// submission nor bound by a draw. Those are physical facts about what the
    /// hardware may be reading, and the shadow arrives at them from the
    /// history rather than from the ring.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Shadow {
        Free,
        Live,
        Bound,
        Submitted(u64),
        /// The mirror of [`SetState::SubmittedAndBound`]. A shadow without it
        /// could only disagree with the ring, never with the rule: it is the
        /// independent [`Reads`] record, not this arm, that catches a `bind`
        /// which leaves a submitted set alone.
        SubmittedAndBound(u64),
    }

    /// What the *history* says about one set, arrived at without restating the
    /// ring's state machine.
    ///
    /// [`Shadow`] is a deliberate mirror, and a mirror cannot fail where the
    /// thing it mirrors fails. One arm of `submitted` is not covered by the
    /// sweep's other, independent claims: a holder already submitted has its
    /// point moved *forward*, and a version that kept the earlier point would
    /// leave the mirror agreeing with it exactly — both would recycle the set
    /// while the later submission is still reading it, and nothing would
    /// disagree. That is the failure this record exists for, and it is the
    /// worst one a descriptor ring has.
    ///
    /// So this is not a state machine at all. It is two physical facts —
    /// whether a recorded draw is holding the set, and the latest point a
    /// submission may still read it at — and the second is a running maximum
    /// rather than a transition, which is what makes it a different statement
    /// from the code it checks.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct Reads {
        /// A recorded draw bound it and that command buffer has not been
        /// submitted or given up.
        bound: bool,
        /// The latest submission point that may still be reading it.
        until: Option<u64>,
    }

    /// A driven history of the ring against a shadow built only from what the
    /// calls said.
    ///
    /// The shadow is written from five places and no others — an `emit` or a
    /// `bind` closes the set it answers with, a `submitted(at)` stamps the
    /// holder, a `recycle(p)` releases every stamp `p` has passed, an
    /// `abandoned` gives up the holder, and a `reset` empties everything. It
    /// never reads a [`SetState`], so it cannot be wrong in the same way the
    /// ring is.
    ///
    /// The law it holds is the one this module exists for: **a set handed to an
    /// emission is never one anything may still read.** That is two spans and
    /// not one — from a bind to the end of the submission carrying it — and the
    /// unsubmitted half is the failure no test of a single sequence finds,
    /// because it needs two dirty draws in one recording and produces no
    /// validation error at all.
    ///
    /// Four more claims fall out of the same shadow: the `partial` flag is true
    /// exactly when the holder survived unread, `recycle` frees exactly the
    /// stamps the point passed, `Exhausted` reports the real depth and
    /// in-flight count rather than a plausible pair, and the ring's own state
    /// is set for set the shadow's.
    ///
    /// The points are deliberately not monotone. A monotone clock cannot drive
    /// `submitted`'s keep-the-later-of-two rule at all — the later point is
    /// always the new one — and it cannot produce the recycle that passes one
    /// submission and not another, which is the shape the safety law is about.
    #[test]
    fn a_driven_history_never_hands_out_a_set_anything_may_be_reading() {
        let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        let (mut whole, mut partial_emits, mut exhausted) = (0u64, 0u64, 0u64);
        let (mut submissions, mut recycled, mut abandons, mut resets) = (0u64, 0u64, 0u64, 0u64);
        let (mut binds, mut binds_that_closed) = (0u64, 0u64);
        // The safety law's own windows: an emission taken while some other set
        // was still in flight, and one taken while some other set was bound by
        // a draw that had not been submitted. Without them the sweep would
        // prove only that an idle ring hands out free sets.
        let (mut emit_under_flight, mut emit_under_bound) = (0u64, 0u64);

        for depth in [1usize, 2, 3, 5] {
            // Seven hundred and fifty and not six hundred: a submission now
            // ends every bound set's span rather than the holder's alone, so
            // a ring stops filling up with sets nothing could free and the
            // rarest arms — exhaustion, and an emission taken while another
            // set is bound — arrive less often per history. The floors below
            // are what they were; the number of histories is what moved.
            for _ in 0..750 {
                let mut ring = SetRing::new(depth);
                let mut shadow: Vec<Shadow> = vec![Shadow::Free; depth];
                let mut reads: Vec<Reads> = vec![Reads::default(); depth];
                let mut holder: Option<usize> = None;
                let mut clock = 1u64;

                for _ in 0..50 {
                    match next() % 16 {
                        // Emissions dominate: the ring exists for the draw
                        // path, and a history spent recycling drives none of it.
                        0..=4 => {
                            let outstanding = shadow
                                .iter()
                                .filter(|s| {
                                    matches!(s, Shadow::Submitted(_) | Shadow::SubmittedAndBound(_))
                                })
                                .count();
                            let bound = shadow
                                .iter()
                                .filter(|s| {
                                    matches!(s, Shadow::Bound | Shadow::SubmittedAndBound(_))
                                })
                                .count();
                            // Expected before the call, from the shadow alone.
                            let expect_partial = holder.is_some_and(|h| shadow[h] == Shadow::Live);
                            match ring.emit() {
                                Ok(emission) => {
                                    let set = emission.set();
                                    assert!(
                                        !matches!(
                                            shadow[set],
                                            Shadow::Submitted(_) | Shadow::SubmittedAndBound(_)
                                        ),
                                        "emitted set {set} is named by a submission: {:?}",
                                        shadow[set]
                                    );
                                    assert!(
                                        !matches!(
                                            shadow[set],
                                            Shadow::Bound | Shadow::SubmittedAndBound(_)
                                        ),
                                        "emitted set {set} is bound by a recorded draw"
                                    );
                                    assert_eq!(
                                        emission.partial(),
                                        expect_partial,
                                        "the partial flag disagrees with the holder's history"
                                    );
                                    if expect_partial {
                                        assert_eq!(Some(set), holder);
                                        partial_emits += 1;
                                    } else {
                                        assert_eq!(shadow[set], Shadow::Free);
                                        whole += 1;
                                    }
                                    if outstanding > 0 {
                                        emit_under_flight += 1;
                                    }
                                    if bound > 0 {
                                        emit_under_bound += 1;
                                    }
                                    // The same two claims, made from the
                                    // history instead of from the mirror.
                                    assert!(
                                        !reads[set].bound,
                                        "set {set} was emitted while a recorded draw holds it"
                                    );
                                    assert_eq!(
                                        reads[set].until, None,
                                        "set {set} was emitted while a submission may read it"
                                    );
                                    shadow[set] = Shadow::Bound;
                                    reads[set].bound = true;
                                    holder = Some(set);
                                }
                                Err(refusal) => {
                                    exhausted += 1;
                                    assert_eq!(refusal.depth, depth);
                                    assert_eq!(refusal.in_flight, outstanding);
                                }
                            }
                        }
                        // A clean draw: it writes nothing and binds the holder.
                        5..=6 => {
                            assert_eq!(ring.bind(), holder);
                            if let Some(h) = holder {
                                binds += 1;
                                // A draw bound it whether or not the mirror
                                // moved: binding is what puts a set in a
                                // command buffer.
                                reads[h].bound = true;
                                match shadow[h] {
                                    Shadow::Live => {
                                        binds_that_closed += 1;
                                        shadow[h] = Shadow::Bound;
                                    }
                                    Shadow::Submitted(at) => {
                                        binds_that_closed += 1;
                                        shadow[h] = Shadow::SubmittedAndBound(at);
                                    }
                                    _ => {}
                                }
                            }
                        }
                        7..=8 => {
                            // Usually a new point and sometimes an older one,
                            // so the keep-the-later rule is driven both ways.
                            let at = if next() % 3 == 0 {
                                TimelinePoint(clock.saturating_sub(next() % 4))
                            } else {
                                clock += 1;
                                TimelinePoint(clock)
                            };
                            ring.submitted(at);
                            if holder.is_some() {
                                submissions += 1;
                            }
                            // A submission ends the span every bound set
                            // opened, not the holder's alone: the command
                            // buffer being submitted is the one whose draws
                            // bound them.
                            for (set, slot) in shadow.iter_mut().enumerate() {
                                *slot = match *slot {
                                    Shadow::Bound => Shadow::Submitted(at.0),
                                    Shadow::Live if holder == Some(set) => Shadow::Submitted(at.0),
                                    Shadow::Submitted(previous)
                                        if holder == Some(set) && at.0 > previous =>
                                    {
                                        Shadow::Submitted(at.0)
                                    }
                                    Shadow::SubmittedAndBound(previous) => {
                                        Shadow::Submitted(previous.max(at.0))
                                    }
                                    unchanged => unchanged,
                                };
                            }
                            // The history's own account: this command buffer
                            // reads every set a draw of its bound, and the
                            // holder whether or not one did. `max`, because two
                            // command buffers may name one unchanged set and
                            // the GPU reads it until the later of them.
                            for (set, r) in reads.iter_mut().enumerate() {
                                if r.bound || holder == Some(set) {
                                    r.until = Some(r.until.map_or(at.0, |p| p.max(at.0)));
                                    r.bound = false;
                                }
                            }
                            // The claim in one line: after a submission there
                            // is nothing left that only a returned receipt
                            // could free.
                            assert_eq!(
                                ring.bound(),
                                0,
                                "a set bound by the submitted command buffer stayed bound"
                            );
                        }
                        9..=13 => {
                            let reached = TimelinePoint(clock.saturating_sub(next() % 5));
                            let freed = ring.recycle(reached);
                            let mut expected = 0;
                            for (set, slot) in shadow.iter_mut().enumerate() {
                                match *slot {
                                    Shadow::Submitted(at) if reached.reached(TimelinePoint(at)) => {
                                        *slot = if holder == Some(set) {
                                            Shadow::Live
                                        } else {
                                            Shadow::Free
                                        };
                                        expected += 1;
                                    }
                                    // The recording that bound it is still
                                    // open, so the timeline releases it only as
                                    // far as bound — never to a writable state.
                                    Shadow::SubmittedAndBound(at)
                                        if reached.reached(TimelinePoint(at)) =>
                                    {
                                        *slot = Shadow::Bound;
                                        expected += 1;
                                    }
                                    _ => {}
                                }
                            }
                            assert_eq!(freed, expected, "recycle freed the wrong count");
                            // The same count from the history: a set stops
                            // being readable exactly when the point it was
                            // last submitted at has been passed.
                            let mut released = 0;
                            for r in &mut reads {
                                if r.until.is_some_and(|p| reached.reached(TimelinePoint(p))) {
                                    r.until = None;
                                    released += 1;
                                }
                            }
                            assert_eq!(
                                freed, released,
                                "the ring recycled a different set of sets than the history says \
                                 the timeline has passed"
                            );
                            recycled += freed as u64;
                        }
                        14 => {
                            if let Some(h) = holder {
                                // The receipt a recording that gave up would
                                // hand back. Forged because the sweep does not
                                // keep them; what it drives is the ring's
                                // answer, not the receipt's own discipline,
                                // which the type enforces at compile time.
                                ring.abandoned(SetEmission::forged(h, true));
                                shadow[h] = match shadow[h] {
                                    Shadow::Submitted(at) | Shadow::SubmittedAndBound(at) => {
                                        Shadow::Submitted(at)
                                    }
                                    _ => Shadow::Free,
                                };
                                // The recording never reached the queue, so no
                                // draw of its is holding the set. What an
                                // *earlier* submission may still be reading is
                                // untouched — that is the timeline's to release.
                                reads[h].bound = false;
                                holder = None;
                                abandons += 1;
                            }
                        }
                        _ => {
                            // Legal exactly when the shadow says nothing is in
                            // flight and nothing is bound, which is also what
                            // `resettable` claims.
                            let idle = shadow
                                .iter()
                                .all(|s| matches!(s, Shadow::Free | Shadow::Live));
                            assert_eq!(ring.resettable(), idle);
                            if idle {
                                ring.reset();
                                resets += 1;
                                shadow.iter_mut().for_each(|s| *s = Shadow::Free);
                                reads.iter_mut().for_each(|r| *r = Reads::default());
                                holder = None;
                            }
                        }
                    }

                    // The ring's bookkeeping, set for set against the shadow's,
                    // after every operation rather than at the end.
                    for (set, expected) in shadow.iter().enumerate() {
                        let state = ring.state(set).expect("a set inside the ring");
                        let same = match (state, expected) {
                            (SetState::Free, Shadow::Free) => true,
                            (SetState::Live, Shadow::Live) => true,
                            (SetState::Bound, Shadow::Bound) => true,
                            (SetState::Submitted(at), Shadow::Submitted(want)) => at.0 == *want,
                            (SetState::SubmittedAndBound(at), Shadow::SubmittedAndBound(want)) => {
                                at.0 == *want
                            }
                            _ => false,
                        };
                        assert!(
                            same,
                            "set {set} is {state:?} and the shadow says {expected:?}"
                        );
                    }
                    assert_eq!(
                        ring.in_flight(),
                        shadow
                            .iter()
                            .filter(|s| {
                                matches!(s, Shadow::Submitted(_) | Shadow::SubmittedAndBound(_))
                            })
                            .count()
                    );
                    assert_eq!(
                        ring.bound(),
                        shadow.iter().filter(|s| **s == Shadow::Bound).count()
                    );

                    // Conservation, and the structural claim underneath it:
                    // every set is in exactly one state, at most one is `Live`,
                    // and that one is the holder. A `Live` set that nobody
                    // holds is a slot lost for the epoch — it is neither
                    // reusable nor recyclable — and no count alone would say so.
                    let live: Vec<usize> = (0..depth)
                        .filter(|s| ring.state(*s) == Some(SetState::Live))
                        .collect();
                    assert_eq!(
                        ring.free() + live.len() + ring.bound() + ring.in_flight(),
                        depth
                    );
                    assert!(live.len() <= 1, "two sets describe the bindings: {live:?}");
                    if let Some(only) = live.first() {
                        assert_eq!(ring.holder(), Some(*only), "a live set nobody holds");
                    }
                }
            }
        }

        assert!(whole > 10_000, "whole={whole}");
        assert!(partial_emits > 1_000, "partial={partial_emits}");
        assert!(exhausted > 20_000, "exhausted={exhausted}");
        assert!(submissions > 9_000, "submissions={submissions}");
        assert!(recycled > 4_000, "recycled={recycled}");
        assert!(abandons > 4_000, "abandons={abandons}");
        assert!(resets > 600, "resets={resets}");
        assert!(binds > 9_000, "binds={binds}");
        // Clean draws that found the holder writable and closed it. The whole
        // reason `bind` takes `&mut self`, and the rarest arm here.
        assert!(
            binds_that_closed > 400,
            "binds_that_closed={binds_that_closed}"
        );
        assert!(
            emit_under_flight > 2_000,
            "emit_under_flight={emit_under_flight}"
        );
        assert!(
            emit_under_bound > 8_000,
            "emit_under_bound={emit_under_bound}"
        );
    }

    #[test]
    fn an_abandoned_fresh_emission_frees_its_set_and_gives_up_the_holder() {
        let mut ring = SetRing::new(2);
        let first = ring.emit().expect("a free set");
        let set = first.set();
        assert_eq!(ring.holder(), Some(set));

        ring.abandoned(first);

        // Freed rather than leaked: nothing else can free a set that was never
        // submitted, so an abandoned one would take a slot out of the ring for
        // the life of the epoch.
        assert_eq!(ring.state(set), Some(SetState::Free));
        assert_eq!(ring.free(), 2);
        assert_eq!(ring.holder(), None);
        assert_eq!(ring.census().abandoned, 1);
    }

    #[test]
    fn the_emission_after_an_abandoned_one_writes_the_whole_table() {
        let mut ring = SetRing::new(2);
        let first = ring.emit().expect("a free set");
        ring.abandoned(first);

        let second = ring.emit().expect("a free set");
        // Not partial: the abandoned write may have landed in part, so nothing
        // in the ring describes the binding table any more.
        assert!(second.whole());
    }

    #[test]
    fn abandoning_a_submitted_holder_gives_up_the_holder_and_frees_nothing() {
        let mut ring = SetRing::new(2);
        let first = ring.emit().expect("a free set");
        let set = first.set();
        // The first recording is submitted and the GPU passes it, so the
        // holder is writable in place again. A second recording takes it, and
        // is itself submitted while still holding the receipt.
        ring.submitted(at(3));
        assert_eq!(ring.recycle(at(3)), 1);
        let partial = ring.emit().expect("the live holder");
        assert_eq!(partial.set(), set);
        ring.submitted(at(7));

        ring.abandoned(partial);

        // The GPU may still be reading it, so the timeline stays the only
        // thing that frees it.
        assert_eq!(ring.state(set), Some(SetState::Submitted(at(7))));
        assert_eq!(ring.holder(), None);
        assert_eq!(ring.recycle(at(7)), 1);
        assert_eq!(ring.state(set), Some(SetState::Free));
    }

    #[test]
    fn abandoning_an_emission_for_a_set_outside_the_ring_changes_nothing() {
        let mut ring = SetRing::new(1);
        let before = ring.clone();
        ring.abandoned(SetEmission::forged(9, false));
        assert_eq!(ring, before);
    }
}
