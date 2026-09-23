//! Host-visible scratch memory, sub-allocated linearly and recycled against
//! the timeline.
//!
//! # Why a chunk is reset whole and never freed piecemeal
//!
//! Every staging allocation has the same lifetime: it is written by the CPU,
//! read by one submission, and dead the moment that submission's timeline
//! point is reached. A general allocator with a free list would be paying for
//! fragmentation, coalescing and a per-allocation header to express a lifetime
//! that is already known, and it would still have to consult the timeline
//! before reusing anything.
//!
//! So a chunk is a bump pointer, and the only free operation is resetting a
//! whole chunk once the GPU is past every submission that named it. The cost
//! is that a chunk stays occupied by its longest-lived allocation; the benefit
//! is that allocation is one add and a bound check on the hottest upload path
//! in the device.
//!
//! # In flight is not free, and it is not a wait either
//!
//! A chunk a submission may still be reading is not writable, exactly as a
//! command buffer is not re-recordable and a descriptor set is not rewritable
//! — see [`crate::pools`] and [`crate::descriptor`]. Exhaustion refuses rather
//! than blocking or resetting the oldest chunk: blocking inside an upload path
//! makes one worker's depth every worker's latency, and resetting the oldest
//! is a use-after-submit with extra steps.
//!
//! # A flush range is rounded outward, and that is why it is computed here
//!
//! `vkFlushMappedMemoryRanges` requires offsets and sizes that are multiples
//! of `nonCoherentAtomSize`, or a size reaching the end of the *allocation*.
//! A caller flushing exactly its own sub-allocation would be passing an
//! unaligned range on every host whose atom is larger than the alignment it
//! asked for. Rounding *outward* is safe — a flush makes more writes visible
//! than asked, never fewer — and rounding inward silently drops the edge bytes
//! of every upload. [`Arena::flush_range`] does the rounding so no call site
//! has to know the rule.
//!
//! The escape clause is the one this arena cannot use. "Reaching the end of
//! the allocation" is a fact about the `VkDeviceMemory`, whose size is
//! `VkMemoryRequirements::size` and is routinely larger than the buffer the
//! chunk was sized from — so a range clamped to the *chunk* is neither a
//! multiple of the atom nor the end of anything, which is invalid use with no
//! validation message on a coherent host and a partial flush on an incoherent
//! one. So the chunk size is required to be a multiple of the atom instead:
//! every window lies inside a chunk, so both rounded ends are then multiples
//! of the atom by construction, the clamp has nothing left to do, and the
//! range is valid whatever the allocation turns out to be.
//!
//! # Bookkeeping without handles
//!
//! [`Chunk`] holds offsets and a timeline point and no Vulkan object, so every
//! rule above is tested on a machine with no GPU. [`Arena`] is the chunks plus
//! the handles they name.
//!
//! # And it hands every one of them back
//!
//! [`Arena::adopt`] takes the buffers, the allocations and the mappings by
//! value, so from that moment the arena is the only thing that knows them. The
//! rule this crate is under is that it allocates no Vulkan object it does not
//! also hand back --- [`crate::pools::WorkerPool`] says so of its command
//! pool, [`crate::resident::Residency::drain`] of its natives --- and an arena
//! with no way out would make an epoch's staging memory unreclaimable *and*
//! unnameable. A caller keeping its own copy of the three vectors to destroy
//! them later would be two sources for one fact, which is the shape the arena
//! exists to remove.
//!
//! So [`Arena::into_chunks`] is the door, and it asks the timeline first: a
//! `vkDestroyBuffer` on a chunk a submission is reading is the use-after-submit
//! this module spends its whole state machine preventing, arriving at teardown
//! instead of at allocation. [`Arena::device_lost`] is the second door, and it
//! asks nothing --- a device that is gone executes nothing, so no chunk is
//! being read, and a lost arena that waited for a timeline reading that will
//! never come would strand every allocation it holds.

use ash::vk;
use reims_vgpu_core::identity::TimelinePoint;

/// What one chunk is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkState {
    /// Not yet named by any submission. May be written and sub-allocated from.
    Open,
    /// Named by a submission and readable by the GPU until this point.
    Submitted(TimelinePoint),
}

/// A window of one chunk, handed to one caller.
///
/// Carries the chunk it came from so a flush can be computed and so a caller
/// cannot pair an offset from one chunk with the buffer of another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a staging window that is not written and copied is capacity spent on nothing"]
pub struct Window {
    pub chunk: usize,
    /// Byte offset within the chunk's buffer and its mapping.
    pub offset: u64,
    pub size: u64,
}

impl Window {
    /// The end of the window, which is where the next allocation may start.
    #[must_use]
    pub const fn end(self) -> u64 {
        self.offset + self.size
    }
}

/// Why an allocation could not be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Larger than a chunk, so no arrangement of chunks would fit it. The
    /// caller allocates a dedicated buffer; this is not a capacity problem
    /// that retrying solves.
    TooLarge { requested: u64, chunk: u64 },
    /// No chunk has room and none is free. Poll the timeline, recycle, retry
    /// or park — see the module doc.
    Exhausted { chunks: usize, in_flight: usize },
    /// An alignment that is not a power of two. Every Vulkan alignment is, and
    /// a caller passing zero or three has computed one rather than read it.
    BadAlignment { alignment: u64 },
    /// A request for no bytes. There is no window that names them: a zero
    /// window's flush range has size zero, which `vkFlushMappedMemoryRanges`
    /// refuses, and the copy it would feed is a region `vkCmdCopyBuffer`
    /// refuses too. A caller with nothing to stage skips staging.
    Empty,
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::TooLarge { .. } => "vk_staging_larger_than_chunk",
            Self::Exhausted { .. } => "vk_staging_exhausted",
            Self::BadAlignment { .. } => "vk_staging_bad_alignment",
            Self::Empty => "vk_staging_empty",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { requested, chunk } => {
                write!(f, "{} requested={requested} chunk={chunk}", self.slug())
            }
            Self::Exhausted { chunks, in_flight } => {
                write!(f, "{} chunks={chunks} in_flight={in_flight}", self.slug())
            }
            Self::BadAlignment { alignment } => {
                write!(f, "{} alignment={alignment}", self.slug())
            }
            Self::Empty => f.write_str(self.slug()),
        }
    }
}

/// How the arena has been used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub allocated: usize,
    /// Allocations that fitted the chunk already open.
    pub in_place: usize,
    /// Allocations that had to move to another chunk. A number that tracks
    /// `allocated` means the chunk size is too small for the frame.
    pub rolled: usize,
    pub recycled: usize,
    pub refused: usize,
    /// Bytes lost to alignment padding, which is what says whether the
    /// alignment a caller asks for is costing anything.
    pub padding: u64,
}

/// One chunk's bookkeeping. No Vulkan object: this is the part that can be
/// wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    size: u64,
    used: u64,
    state: ChunkState,
}

impl Chunk {
    #[must_use]
    pub const fn state(&self) -> ChunkState {
        self.state
    }

    #[must_use]
    pub const fn used(&self) -> u64 {
        self.used
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }
}

/// One chunk's Vulkan objects, as a teardown takes them.
///
/// The three together, because they are one chunk: a buffer destroyed without
/// its allocation freed is a leak, and an allocation freed while another chunk
/// still names it is the mismatch [`Arena::adopt`] asserts against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a chunk's buffer and allocation are the device's until something destroys them"]
pub struct ChunkHandles {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// The mapping. `vkFreeMemory` releases it, so a caller that frees the
    /// allocation need not unmap; one that unmaps explicitly needs this.
    pub mapped: *mut u8,
}

/// The arena cannot be taken apart yet: a submission may still be reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StillReading {
    pub chunks: usize,
    pub in_flight: usize,
}

impl StillReading {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        "vk_staging_still_reading"
    }
}

impl std::fmt::Display for StillReading {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} chunks={} in_flight={}",
            self.slug(),
            self.chunks,
            self.in_flight
        )
    }
}

/// The chunks, and the handles they name.
///
/// One arena per worker, for the reason a command pool is per worker: the
/// mapping is written by the CPU with no synchronization of its own.
#[derive(Debug)]
pub struct Arena {
    chunks: Vec<Chunk>,
    buffers: Vec<vk::Buffer>,
    memory: Vec<vk::DeviceMemory>,
    mapped: Vec<*mut u8>,
    /// The chunk allocations are being taken from. Moves only when one cannot
    /// fit, so a frame's uploads stay in one chunk and one flush covers them.
    open: usize,
    atom: u64,
    census: Census,
}

// SAFETY: the mapped pointers are the only non-`Send` field, and an `Arena` is
// owned by exactly one worker for the same reason a `VkCommandPool` is — see
// the module doc. Moving that ownership between threads is what `Send` says,
// and it is sound here because nothing else holds a pointer into a chunk: a
// [`Window`] is an offset, not a reference.
unsafe impl Send for Arena {}

impl Arena {
    /// Adopt `chunks` buffers of `size` bytes, each already allocated, bound
    /// and mapped.
    ///
    /// `atom` is `VkPhysicalDeviceLimits::nonCoherentAtomSize`, which
    /// [`Self::flush_range`] rounds to. Pass it even on a coherent allocation:
    /// the arena does not decide whether a flush is needed, only what range
    /// one would cover.
    ///
    /// # Panics
    ///
    /// If the three arrays disagree in length, if any is empty, if `size` is
    /// zero or not a multiple of `atom`, or if `atom` is not a power of two.
    /// Each is a caller that built the arena from mismatched pieces, and every
    /// one of them produces a pointer into the wrong allocation later — or, in
    /// the case of the chunk size, a flush range Vulkan does not accept. See
    /// the module doc.
    #[must_use]
    pub fn adopt(
        size: u64,
        atom: u64,
        buffers: Vec<vk::Buffer>,
        memory: Vec<vk::DeviceMemory>,
        mapped: Vec<*mut u8>,
    ) -> Self {
        assert!(
            !buffers.is_empty(),
            "an arena with no chunks stages nothing"
        );
        assert!(
            buffers.len() == memory.len() && memory.len() == mapped.len(),
            "a chunk is a buffer, its memory and its mapping together"
        );
        assert!(size > 0, "a chunk of no bytes holds nothing");
        assert!(
            atom.is_power_of_two(),
            "nonCoherentAtomSize is a power of two"
        );
        assert!(
            size.is_multiple_of(atom),
            "a chunk not a whole number of atoms has a tail no flush range can name"
        );
        let chunks = vec![
            Chunk {
                size,
                used: 0,
                state: ChunkState::Open,
            };
            buffers.len()
        ];
        Self {
            chunks,
            buffers,
            memory,
            mapped,
            open: 0,
            atom,
            census: Census::default(),
        }
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    #[must_use]
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    #[must_use]
    pub const fn open(&self) -> usize {
        self.open
    }

    /// The buffer a window's bytes are copied from.
    #[must_use]
    pub fn buffer(&self, window: Window) -> vk::Buffer {
        self.buffers[window.chunk]
    }

    #[must_use]
    pub fn memory(&self, window: Window) -> vk::DeviceMemory {
        self.memory[window.chunk]
    }

    /// A pointer to the window's first byte.
    ///
    /// # Safety
    ///
    /// The caller may write `window.size` bytes there, and only while it holds
    /// the window. The chunk is not readable by the GPU until
    /// [`Self::submitted`], which is what makes the write safe rather than
    /// this function.
    #[must_use]
    pub unsafe fn write_at(&self, window: Window) -> *mut u8 {
        // SAFETY: the window came from this arena, so its chunk index is in
        // range and `offset + size` is within that chunk's mapping.
        unsafe { self.mapped[window.chunk].add(window.offset as usize) }
    }

    /// How many chunks a submission may still be reading.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| matches!(c.state, ChunkState::Submitted(_)))
            .count()
    }

    /// Take `size` bytes aligned to `alignment`.
    ///
    /// # Errors
    ///
    /// [`Refusal`]. Nothing is taken on any of the three.
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Result<Window, Refusal> {
        if !alignment.is_power_of_two() {
            self.census.refused += 1;
            return Err(Refusal::BadAlignment { alignment });
        }
        // Before the room question, because no arrangement of chunks answers
        // it: a window of no bytes is one whose flush range is empty. See
        // [`Refusal::Empty`].
        if size == 0 {
            self.census.refused += 1;
            return Err(Refusal::Empty);
        }
        let chunk_size = self.chunks[0].size;
        if size > chunk_size {
            self.census.refused += 1;
            return Err(Refusal::TooLarge {
                requested: size,
                chunk: chunk_size,
            });
        }

        if let Some(window) = self.take_from(self.open, size, alignment) {
            self.census.allocated += 1;
            self.census.in_place += 1;
            return Ok(window);
        }
        // The open chunk is full. Roll to an open one with room — not to a
        // submitted one, however old: the timeline is the only thing that
        // frees a chunk.
        for index in 0..self.chunks.len() {
            if index == self.open {
                continue;
            }
            // Redundant with `take_from`'s own state check, which is the one
            // that decides. Kept because it is the half a reader of this loop
            // can see, and because it skips the alignment arithmetic for a
            // chunk that cannot answer.
            if self.chunks[index].state != ChunkState::Open {
                continue;
            }
            if let Some(window) = self.take_from(index, size, alignment) {
                self.open = index;
                self.census.allocated += 1;
                self.census.rolled += 1;
                return Ok(window);
            }
        }
        self.census.refused += 1;
        Err(Refusal::Exhausted {
            chunks: self.chunks.len(),
            in_flight: self.in_flight(),
        })
    }

    fn take_from(&mut self, index: usize, size: u64, alignment: u64) -> Option<Window> {
        let chunk = self.chunks.get_mut(index)?;
        if chunk.state != ChunkState::Open {
            return None;
        }
        let start = chunk.used.next_multiple_of(alignment);
        let end = start.checked_add(size)?;
        if end > chunk.size {
            return None;
        }
        self.census.padding += start - chunk.used;
        chunk.used = end;
        Some(Window {
            chunk: index,
            offset: start,
            size,
        })
    }

    /// Every chunk with anything in it is now named by a submission that will
    /// signal `at`.
    ///
    /// Chunks with nothing in them stay open: a submission that staged nothing
    /// must not cost the arena its capacity. A chunk already submitted has its
    /// point moved forward rather than kept, for the reason
    /// [`crate::descriptor::SetRing::submitted`] does: two submissions may
    /// read one chunk, and the later one is what frees it.
    pub fn submitted(&mut self, at: TimelinePoint) {
        for chunk in &mut self.chunks {
            if chunk.used == 0 {
                continue;
            }
            chunk.state = match chunk.state {
                ChunkState::Submitted(previous) if previous.0 > at.0 => {
                    ChunkState::Submitted(previous)
                }
                _ => ChunkState::Submitted(at),
            };
        }
    }

    /// Reset every chunk the timeline has passed. Returns how many.
    ///
    /// The one place capacity comes back, and it consults the timeline rather
    /// than an age or a count.
    pub fn recycle(&mut self, reached: TimelinePoint) -> usize {
        let mut freed = 0;
        for chunk in &mut self.chunks {
            if let ChunkState::Submitted(at) = chunk.state {
                if reached.0 >= at.0 {
                    chunk.state = ChunkState::Open;
                    chunk.used = 0;
                    freed += 1;
                }
            }
        }
        self.census.recycled += freed;
        freed
    }

    /// Whether the arena may be taken apart: no chunk is named by a
    /// submission the timeline has not passed.
    ///
    /// The same question [`crate::pools::BufferRing::resettable`] asks about
    /// command buffers, about the memory they copy from.
    #[must_use]
    pub fn retirable(&self) -> bool {
        self.in_flight() == 0
    }

    /// Take the arena apart into the handles it was adopted with.
    ///
    /// # Errors
    ///
    /// [`StillReading`] with the arena returned whole and nothing destroyed,
    /// when a submission may still be reading a chunk. The caller polls the
    /// timeline, calls [`Self::recycle`], and asks again.
    // A refused teardown returns the arena itself, which is the refusal's whole
    // content: the caller has nothing else that names these handles. That makes
    // the `Err` variant as large as the arena by construction, and boxing it
    // would put an allocation on the refusal path of an operation that happens
    // once per epoch, in exchange for nothing.
    #[allow(clippy::result_large_err)]
    pub fn into_chunks(self) -> Result<Vec<ChunkHandles>, (Self, StillReading)> {
        if !self.retirable() {
            let refusal = StillReading {
                chunks: self.chunks.len(),
                in_flight: self.in_flight(),
            };
            return Err((self, refusal));
        }
        Ok(self.take())
    }

    /// The device is gone, so take the arena apart whatever the timeline says.
    ///
    /// A lost device executes nothing further, so no chunk is being read and
    /// the timeline that would have proved it will never advance again. The
    /// handles still have to be destroyed: losing a device does not free its
    /// memory, and an arena that insisted on [`Self::into_chunks`] here would
    /// hold every staging allocation of the lost epoch for the life of the
    /// process.
    #[must_use = "a lost arena's chunks are buffers and device memory the device never gets back"]
    pub fn device_lost(self) -> Vec<ChunkHandles> {
        self.take()
    }

    /// Every chunk's handles, in adoption order, with nothing left behind.
    ///
    /// `#[must_use]` on the method as well as on [`ChunkHandles`]: the lint
    /// does not look inside a `Vec`, and this one consumes the arena — a
    /// caller that drops the result has lost the only thing that named these
    /// handles.
    #[must_use = "the chunks are buffers and device memory the device never gets back"]
    fn take(self) -> Vec<ChunkHandles> {
        self.buffers
            .into_iter()
            .zip(self.memory)
            .zip(self.mapped)
            .map(|((buffer, memory), mapped)| ChunkHandles {
                buffer,
                memory,
                mapped,
            })
            .collect()
    }

    /// The `(offset, size)` to flush or invalidate for a window.
    ///
    /// Rounded outward to `nonCoherentAtomSize`, so both ends are multiples of
    /// it and the range is always one Vulkan accepts. A caller passing its own
    /// offsets would be passing an unaligned range on every host whose atom is
    /// larger than the alignment it asked for, and rounding inward instead
    /// would silently drop the edge bytes of every upload.
    ///
    /// No clamp to the chunk. The rounded end cannot pass it: a window lies
    /// inside its chunk and [`Self::adopt`] requires the chunk to be a whole
    /// number of atoms, so rounding up reaches the chunk's end at the furthest.
    /// A clamp would be the thing that *broke* the range rather than saved it —
    /// see the module doc for why the "end of the allocation" escape clause is
    /// not one this arena can invoke.
    #[must_use]
    pub fn flush_range(&self, window: Window) -> (u64, u64) {
        let start = window.offset & !(self.atom - 1);
        let end = window.end().next_multiple_of(self.atom);
        debug_assert!(
            end <= self.chunks[window.chunk].size,
            "a window's rounded end left its chunk"
        );
        (start, end - start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use std::collections::BTreeSet;
    use std::ptr;

    const CHUNK: u64 = 1024;

    fn arena(chunks: usize, atom: u64) -> Arena {
        Arena::adopt(
            CHUNK,
            atom,
            (0..chunks)
                .map(|i| vk::Buffer::from_raw(i as u64 + 1))
                .collect(),
            (0..chunks)
                .map(|i| vk::DeviceMemory::from_raw(i as u64 + 1))
                .collect(),
            vec![ptr::null_mut(); chunks],
        )
    }

    /// The rule the crate states of every other handle-owning ring, applied
    /// here: everything adopted comes back, once, in order.
    #[test]
    fn an_idle_arena_hands_back_every_chunk_it_was_adopted_with() {
        let mut a = arena(3, 64);
        let _staged = a.allocate(16, 16).expect("room");
        a.submitted(at(5));
        assert_eq!(a.recycle(at(5)), 1);
        assert!(a.retirable());

        let chunks = a.into_chunks().unwrap_or_else(|(_, e)| panic!("{e}"));

        assert_eq!(chunks.len(), 3);
        for (index, chunk) in chunks.iter().enumerate() {
            let expected = index as u64 + 1;
            assert_eq!(chunk.buffer.as_raw(), expected);
            assert_eq!(chunk.memory.as_raw(), expected, "a chunk's own allocation");
        }
    }

    #[test]
    fn an_arena_a_submission_is_reading_is_returned_whole_rather_than_destroyed() {
        let mut a = arena(2, 64);
        let _staged = a.allocate(16, 16).expect("room");
        a.submitted(at(9));
        assert!(!a.retirable());

        let (a, refusal) = a.into_chunks().expect_err("the GPU may be reading it");

        assert_eq!(
            refusal,
            StillReading {
                chunks: 2,
                in_flight: 1,
            }
        );
        assert!(refusal.to_string().starts_with(refusal.slug()));
        // Nothing was destroyed, so the retry after a recycle has everything.
        let mut a = a;
        assert_eq!(a.recycle(at(9)), 1);
        assert_eq!(
            a.into_chunks().unwrap_or_else(|(_, e)| panic!("{e}")).len(),
            2
        );
    }

    /// A lost device advances no timeline, so the orderly door never opens
    /// again — and the memory is still allocated.
    #[test]
    fn a_lost_device_hands_back_the_chunks_a_timeline_would_never_release() {
        let mut a = arena(2, 64);
        let _staged = a.allocate(16, 16).expect("room");
        a.submitted(at(9));
        assert!(!a.retirable(), "and it never will be again");

        let chunks = a.device_lost();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].buffer.as_raw(), 1);
        assert_eq!(chunks[1].memory.as_raw(), 2);
    }

    fn at(n: u64) -> TimelinePoint {
        TimelinePoint(n)
    }

    #[test]
    fn allocations_bump_within_one_chunk_and_stay_there() {
        let mut arena = arena(2, 64);
        let first = arena.allocate(100, 4).expect("room");
        let second = arena.allocate(100, 4).expect("room");
        assert_eq!(first.chunk, 0);
        assert_eq!(second.chunk, 0);
        assert_eq!(first.offset, 0);
        assert_eq!(second.offset, 100);
        assert_eq!(arena.census().in_place, 2);
        assert_eq!(arena.census().rolled, 0);
        // The second chunk is untouched: rolling early would halve the useful
        // capacity of every frame.
        assert_eq!(arena.chunks()[1].used(), 0);
    }

    #[test]
    fn an_allocation_is_aligned_and_the_padding_is_counted() {
        let mut arena = arena(1, 64);
        let first = arena.allocate(10, 4).expect("room");
        assert_eq!(first.offset, 0);
        let second = arena.allocate(10, 256).expect("room");
        assert_eq!(second.offset, 256);
        // 246 bytes of padding, which is what says whether a caller's
        // alignment is costing anything.
        assert_eq!(arena.census().padding, 246);
    }

    #[test]
    fn an_alignment_that_is_not_a_power_of_two_is_refused() {
        let mut arena = arena(1, 64);
        for alignment in [0u64, 3, 6, 100] {
            assert_eq!(
                arena.allocate(4, alignment),
                Err(Refusal::BadAlignment { alignment })
            );
        }
        assert_eq!(arena.chunks()[0].used(), 0);
    }

    #[test]
    fn a_full_chunk_rolls_to_the_next_open_one_rather_than_refusing() {
        let mut arena = arena(2, 64);
        let _ = arena.allocate(1000, 4).expect("room");
        let rolled = arena.allocate(100, 4).expect("the other chunk");
        assert_eq!(rolled.chunk, 1);
        assert_eq!(rolled.offset, 0);
        assert_eq!(arena.open(), 1);
        assert_eq!(arena.census().rolled, 1);
        // And it stays on the new chunk rather than going back to look.
        assert_eq!(arena.allocate(4, 4).expect("room").chunk, 1);
    }

    #[test]
    fn a_request_larger_than_a_chunk_refuses_as_that_and_not_as_exhaustion() {
        let mut arena = arena(4, 64);
        // Four chunks with everything free, so this is not a capacity problem
        // and retrying would never help. The caller allocates a dedicated
        // buffer instead, and the refusal is what tells it so.
        assert_eq!(
            arena.allocate(CHUNK + 1, 4),
            Err(Refusal::TooLarge {
                requested: CHUNK + 1,
                chunk: CHUNK,
            })
        );
    }

    #[test]
    fn exhaustion_refuses_rather_than_resetting_the_oldest_chunk() {
        let mut arena = arena(2, 64);
        let _ = arena.allocate(CHUNK, 4).expect("room");
        arena.submitted(at(5));
        let _ = arena.allocate(CHUNK, 4).expect("the second chunk");
        arena.submitted(at(9));

        assert_eq!(
            arena.allocate(4, 4),
            Err(Refusal::Exhausted {
                chunks: 2,
                in_flight: 2,
            })
        );
        // Neither chunk was reset. The GPU may be reading both.
        assert_eq!(arena.in_flight(), 2);
        assert_eq!(arena.census().refused, 1);
    }

    #[test]
    fn only_the_timeline_returns_a_chunk_and_it_returns_the_whole_chunk() {
        let mut arena = arena(2, 64);
        let _ = arena.allocate(600, 4).expect("room");
        let _ = arena.allocate(300, 4).expect("room");
        arena.submitted(at(5));
        assert_eq!(arena.recycle(at(4)), 0);
        assert_eq!(arena.in_flight(), 1);

        assert_eq!(arena.recycle(at(5)), 1);
        assert_eq!(arena.chunks()[0].state(), ChunkState::Open);
        // The whole chunk, not the last allocation: two allocations went in
        // and one reset takes both back.
        assert_eq!(arena.chunks()[0].used(), 0);
        assert_eq!(arena.allocate(CHUNK, 4).expect("room").offset, 0);
    }

    #[test]
    fn a_chunk_named_twice_is_freed_by_the_later_submission() {
        let mut arena = arena(1, 64);
        let _ = arena.allocate(4, 4).expect("room");
        arena.submitted(at(9));
        arena.submitted(at(3));
        // Keeping the earlier point would free the chunk while the second
        // submission is still reading it.
        assert_eq!(arena.chunks()[0].state(), ChunkState::Submitted(at(9)));
        assert_eq!(arena.recycle(at(3)), 0);
        assert_eq!(arena.recycle(at(9)), 1);
    }

    #[test]
    fn a_submission_that_staged_nothing_costs_no_capacity() {
        let mut arena = arena(2, 64);
        let _ = arena.allocate(4, 4).expect("room");
        arena.submitted(at(7));
        // The second chunk was never written, so it must not be waiting on a
        // point that has nothing to do with it.
        assert_eq!(arena.chunks()[1].state(), ChunkState::Open);
        assert_eq!(arena.in_flight(), 1);
        assert_eq!(arena.allocate(CHUNK, 4).expect("the open chunk").chunk, 1);
    }

    #[test]
    fn a_flush_range_is_rounded_outward_to_the_atom() {
        let arena = arena(1, 64);
        let window = Window {
            chunk: 0,
            offset: 100,
            size: 10,
        };
        // 100..110 becomes 64..128: outward, because a flush that made fewer
        // writes visible than asked would drop the edge bytes of the upload.
        assert_eq!(arena.flush_range(window), (64, 64));

        let aligned = Window {
            chunk: 0,
            offset: 128,
            size: 128,
        };
        assert_eq!(arena.flush_range(aligned), (128, 128));
    }

    /// The last window in a chunk rounds up to the chunk's end and no further,
    /// and the range is a whole number of atoms like every other.
    ///
    /// The regression: the end was *clamped* to the chunk, which for a chunk
    /// that was not a whole number of atoms produced a size that is neither a
    /// multiple of `nonCoherentAtomSize` nor the end of the allocation ---
    /// invalid use of `vkFlushMappedMemoryRanges`, since a chunk's
    /// `VkDeviceMemory` is `VkMemoryRequirements::size` and is routinely
    /// larger than the buffer. `adopt` now requires the chunk to be a whole
    /// number of atoms, which makes the clamp unreachable and the range valid
    /// whatever the allocation turns out to be.
    #[test]
    fn a_flush_at_the_end_of_a_chunk_reaches_it_and_stops() {
        let arena = arena(1, 64);
        let tail = Window {
            chunk: 0,
            offset: CHUNK - 10,
            size: 10,
        };
        let (offset, size) = arena.flush_range(tail);
        assert_eq!(offset + size, CHUNK);
        assert_eq!(offset, CHUNK - 64);
        assert_eq!(size % 64, 0, "and it is a whole number of atoms");
    }

    /// A chunk with a tail no flush range can name is refused at adoption
    /// rather than producing one later.
    #[test]
    #[should_panic(expected = "whole number of atoms")]
    fn a_chunk_that_is_not_a_whole_number_of_atoms_is_refused() {
        let _ = Arena::adopt(
            1000,
            64,
            vec![vk::Buffer::from_raw(1)],
            vec![vk::DeviceMemory::from_raw(1)],
            vec![ptr::null_mut()],
        );
    }

    #[test]
    fn an_atom_of_one_leaves_every_range_exactly_as_asked() {
        let arena = arena(1, 1);
        let window = Window {
            chunk: 0,
            offset: 7,
            size: 5,
        };
        assert_eq!(arena.flush_range(window), (7, 5));
    }

    #[test]
    fn a_window_names_the_buffer_of_its_own_chunk() {
        let mut arena = arena(2, 64);
        let _ = arena.allocate(CHUNK, 4).expect("room");
        let second = arena.allocate(4, 4).expect("the other chunk");
        assert_eq!(second.chunk, 1);
        assert_eq!(arena.buffer(second).as_raw(), 2);
        assert_eq!(arena.memory(second).as_raw(), 2);
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::TooLarge {
                requested: 2,
                chunk: 1,
            },
            Refusal::Exhausted {
                chunks: 1,
                in_flight: 1,
            },
            Refusal::BadAlignment { alignment: 3 },
            Refusal::Empty,
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_staging_"));
        }
    }

    #[test]
    fn a_request_for_no_bytes_is_refused_rather_than_handed_an_unflushable_window() {
        let mut arena = arena(2, 64);
        assert_eq!(arena.allocate(0, 4), Err(Refusal::Empty));
        // Refused before the room question, so an alignment that is not a
        // power of two is still the answer a caller gets first, and the arena
        // is untouched apart from the count.
        assert_eq!(
            arena.allocate(0, 3),
            Err(Refusal::BadAlignment { alignment: 3 })
        );
        assert_eq!(arena.census().allocated, 0);
        assert_eq!(arena.census().refused, 2);
        // And the whole chunk is still there to hand out.
        assert_eq!(arena.allocate(CHUNK, 4).expect("room").offset, 0);
    }

    #[test]
    #[should_panic(expected = "an arena with no chunks stages nothing")]
    fn an_arena_with_no_chunks_is_refused() {
        let _ = Arena::adopt(CHUNK, 64, Vec::new(), Vec::new(), Vec::new());
    }

    #[test]
    #[should_panic(expected = "a chunk is a buffer, its memory and its mapping together")]
    fn an_arena_built_from_mismatched_pieces_is_refused() {
        let _ = Arena::adopt(
            CHUNK,
            64,
            vec![vk::Buffer::null(), vk::Buffer::null()],
            vec![vk::DeviceMemory::null()],
            vec![ptr::null_mut()],
        );
    }

    #[test]
    #[should_panic(expected = "nonCoherentAtomSize is a power of two")]
    fn an_atom_that_is_not_a_power_of_two_is_refused() {
        let _ = arena(1, 96);
    }

    // ---- A driven history of the arena's capacity and its lifetime -------
    //
    // Two claims, and neither is visible in one call. The first is that two
    // live windows never overlap --- an overlap is one upload writing over
    // another's bytes between the `memcpy` and the copy command, which
    // produces a wrong frame and no error anywhere. The second is that a
    // chunk's bytes are only reused after the timeline says the submission
    // that read them has completed; reusing them earlier is a write into
    // memory the GPU is still reading.
    //
    // The shadow follows the calls' arguments. It never asks the arena which
    // chunk is open, what a chunk's state is, or how much is used.

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

    /// One chunk, as the shadow holds it: what submission owns it, and every
    /// window handed out of it since it was last reset.
    #[derive(Default, Clone)]
    struct ShadowChunk {
        /// `None` is open. `Some(point)` is named by a submission signalling
        /// that point.
        submitted: Option<u64>,
        /// Whether anything has been taken from it since the last reset ---
        /// the shadow's own answer to "does a submission cost this chunk".
        dirty: bool,
        live: Vec<(u64, u64)>,
    }

    #[derive(Default)]
    struct Tally {
        allocated: usize,
        out_of_order: usize,
        rolled: usize,
        bad_alignment: usize,
        too_large: usize,
        exhausted: usize,
        empty: usize,
        submitted: usize,
        recycled: usize,
        flushes: usize,
    }

    #[test]
    fn a_driven_history_never_hands_out_bytes_that_are_still_in_flight() {
        const CHUNKS: usize = 3;
        let mut tally = Tally::default();

        for seed in 0..500_u64 {
            let mut rng = Rng::new(seed);
            let atom = 1 << (1 + rng.below(7));
            let mut arena = arena(CHUNKS, atom);
            let mut shadow = vec![ShadowChunk::default(); CHUNKS];
            let mut clock = 0_u64;

            for _ in 0..60 {
                match rng.below(16) {
                    0..=9 => {
                        // Sizes that mostly fit and sometimes do not, and one
                        // alignment in eight that is not a power of two.
                        let size = match rng.below(16) {
                            0 | 1 => CHUNK + 1 + rng.below(64),
                            2 => 0,
                            _ => 1 + rng.below(CHUNK / 3),
                        };
                        let alignment = match rng.below(8) {
                            0 => 3 + rng.below(5) * 2,
                            n => 1 << (n - 1),
                        };
                        // A bump pointer per chunk, which is what the module
                        // says it is --- and the padding an alignment costs is
                        // part of the room question, not a detail beside it.
                        let open_with_room = alignment.is_power_of_two()
                            && size > 0
                            && size <= CHUNK
                            && shadow.iter().any(|c| {
                                let used = c.live.last().map_or(0, |&(_, end)| end);
                                c.submitted.is_none()
                                    && used.next_multiple_of(alignment) + size <= CHUNK
                            });
                        match arena.allocate(size, alignment) {
                            Ok(window) => {
                                assert!(alignment.is_power_of_two());
                                assert!(size <= CHUNK);
                                assert!(window.chunk < CHUNKS);
                                assert_eq!(window.size, size);
                                assert_eq!(
                                    window.offset % alignment,
                                    0,
                                    "a window that is not aligned is one a driver rejects"
                                );
                                assert!(window.end() <= CHUNK, "past the end of the chunk");
                                let chunk = &mut shadow[window.chunk];
                                assert!(
                                    chunk.submitted.is_none(),
                                    "handed out bytes of a chunk a submission is still reading"
                                );
                                for &(start, end) in &chunk.live {
                                    assert!(
                                        window.end() <= start || window.offset >= end,
                                        "window {}..{} overlaps a live {start}..{end}",
                                        window.offset,
                                        window.end()
                                    );
                                }
                                chunk.live.push((window.offset, window.end()));
                                chunk.dirty = true;
                                tally.allocated += 1;

                                // Every window is flushable, and the range a
                                // flush covers must be one Vulkan accepts.
                                let (start, length) = arena.flush_range(window);
                                assert_eq!(start % atom, 0, "an unaligned flush offset");
                                assert!(start <= window.offset);
                                assert!(start + length >= window.end(), "the flush misses bytes");
                                assert!(start + length <= CHUNK);
                                assert!(
                                    length.is_multiple_of(atom),
                                    "a flush size Vulkan refuses: {length} against atom {atom}"
                                );
                                tally.flushes += 1;
                            }
                            Err(Refusal::BadAlignment { alignment: named }) => {
                                assert_eq!(named, alignment);
                                assert!(!alignment.is_power_of_two());
                                tally.bad_alignment += 1;
                            }
                            Err(Refusal::TooLarge { requested, chunk }) => {
                                assert_eq!((requested, chunk), (size, CHUNK));
                                assert!(size > CHUNK);
                                tally.too_large += 1;
                            }
                            Err(Refusal::Empty) => {
                                assert_eq!(size, 0);
                                tally.empty += 1;
                            }
                            Err(Refusal::Exhausted { chunks, .. }) => {
                                assert_eq!(chunks, CHUNKS);
                                assert!(
                                    !open_with_room,
                                    "refused an allocation an open chunk had room for"
                                );
                                tally.exhausted += 1;
                            }
                        }
                    }
                    10..=12 => {
                        clock += 1;
                        // Mostly the newest point, and sometimes an older one.
                        // A monotone driver can never tell whether the arena
                        // keeps the later of two points, because with a
                        // monotone clock the later one is always the new one
                        // --- so the rule the module states about a chunk named
                        // twice would go undriven.
                        let at = if rng.below(3) == 0 {
                            clock.saturating_sub(1 + rng.below(3))
                        } else {
                            clock
                        };
                        arena.submitted(TimelinePoint(at));
                        for chunk in &mut shadow {
                            if !chunk.dirty {
                                continue;
                            }
                            // Forward only: two submissions may read one chunk
                            // and the later one is what frees it.
                            chunk.submitted =
                                Some(chunk.submitted.map_or(at, |prior| prior.max(at)));
                        }
                        if shadow.iter().any(|c| c.dirty) {
                            tally.submitted += 1;
                            if at < clock {
                                tally.out_of_order += 1;
                            }
                        }
                    }
                    _ => {
                        // Sometimes short of the newest point, so a chunk that
                        // is still in flight stays that way.
                        let reached = clock.saturating_sub(rng.below(3));
                        let expected = shadow
                            .iter()
                            .filter(|c| c.submitted.is_some_and(|at| reached >= at))
                            .count();
                        assert_eq!(arena.recycle(TimelinePoint(reached)), expected);
                        for chunk in &mut shadow {
                            if chunk.submitted.is_some_and(|at| reached >= at) {
                                *chunk = ShadowChunk::default();
                            }
                        }
                        tally.recycled += 1;
                    }
                }

                // Whatever the arena believes about its own chunks, the shadow
                // believes the same --- derived from the calls and nothing else.
                for (index, chunk) in arena.chunks().iter().enumerate() {
                    let expected = match shadow[index].submitted {
                        None => ChunkState::Open,
                        Some(at) => ChunkState::Submitted(TimelinePoint(at)),
                    };
                    assert_eq!(chunk.state(), expected, "chunk {index}");
                    assert_eq!(
                        chunk.used(),
                        shadow[index].live.last().map_or(0, |&(_, end)| end),
                        "chunk {index} used"
                    );
                }
                assert_eq!(
                    arena.in_flight(),
                    shadow.iter().filter(|c| c.submitted.is_some()).count()
                );
            }
            tally.rolled += arena.census().rolled;
        }

        assert!(tally.allocated > 5_000, "{}", tally.allocated);
        assert!(tally.rolled > 500, "{}", tally.rolled);
        assert!(tally.bad_alignment > 500, "{}", tally.bad_alignment);
        assert!(tally.too_large > 500, "{}", tally.too_large);
        assert!(tally.exhausted > 500, "{}", tally.exhausted);
        assert!(tally.empty > 500, "{}", tally.empty);
        assert!(tally.submitted > 2_000, "{}", tally.submitted);
        assert!(tally.out_of_order > 500, "{}", tally.out_of_order);
        assert!(tally.recycled > 2_000, "{}", tally.recycled);
        assert!(tally.flushes > 5_000, "{}", tally.flushes);
    }
}
