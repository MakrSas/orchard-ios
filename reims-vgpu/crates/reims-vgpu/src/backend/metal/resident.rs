//! Colour render targets this rail keeps alive across draws.
//!
//! # What it removes, and why the target was ephemeral to begin with
//!
//! Every colour attachment used to be a fresh `MTLTexture` per draw: allocate,
//! upload the whole attachment's prior content into it with `replace_region`,
//! render, `getBytes` the whole attachment back out, drop the texture. On a
//! driven macos-13 boot the upload alone was `metal_rt_seed_us` = 1 670 us a
//! draw, 31 % of this rail's `engine_us` and the largest single item in it;
//! the readback was another 19 % and the allocation 1 %.
//!
//! The attachment was ephemeral because nothing in this rail retained a
//! host-side render target at all — not because the guest's stream asks for
//! one. A compositing layer is drawn into over and over, and every draw of it
//! was paying to move 8 MB in and 8 MB out.
//!
//! # The safety argument is borrowed, not new
//!
//! A retained target may only be *loaded from* when its pixels are this pass's
//! prior content, and getting that wrong is a silent wrong frame: the pass
//! composites onto the stale pixels and its Store publishes the composite back
//! over the guest's own pages, so the next frame loads what this one stored.
//!
//! This module does not make a new claim about content. It makes exactly one:
//!
//! > When [`crate::runtime::surface_cache`] would have served bytes `B` as this
//! > attachment's LOAD seed, and the retained texture is known to hold `B`, do
//! > not copy `B` into it again.
//!
//! "Would have served" is [`crate::runtime::draw`]'s door, already gated on
//! [`crate::runtime::surface_currency::CurrencyStandard::WatchedAndUnwritten`].
//! "Is known to hold `B`" is [`Resident::content_gen`]: the surface cache's own
//! `host_gen` at the moment this rail's Store published the texture's pixels
//! into it. Every writer of `host_surfaces` takes a fresh
//! `DeviceState::next_sampled_content_generation` in the same breath as it
//! changes the bytes, and `surface_cache::forget` removes the entry outright —
//! so a generation that still matches is a statement that nothing has replaced
//! the frame this texture was published as.
//!
//! The consequence worth stating: this rail can never be *more* wrong than the
//! seed door it rides on. If that door is correct, so is this; if it is not,
//! this changes only how expensive the wrong answer was to produce.
//!
//! # Eviction is lawful here
//!
//! `AGENTS.md` forbids silently evicting resource state that represents guest
//! work. A resident here represents none: its pixels are also in the guest's
//! own pages and in the surface cache, which is *why* the currency test can be
//! asked at all. Dropping one costs the next draw an upload and loses nothing,
//! so the bound below is a cache bound in the sense that is allowed. Every
//! eviction is still counted, because a rail that spends its whole budget
//! thrashing is indistinguishable from one that is off.

use crate::backend::metal::raw_metal;
use metal::{
    DeviceRef, MTLPixelFormat, MTLResourceOptions, MTLStorageMode, MTLTextureType, MTLTextureUsage,
    Texture, TextureDescriptor,
};
use parking_lot::Mutex;

/// How many colour attachments this rail keeps textures for at once.
///
/// A desktop compositor draws into a handful of large layers plus a tail of
/// small ones. The bound is on count and on bytes together because either alone
/// admits the other's worst case: sixteen 4x4 targets are nothing, and four
/// 1920x1080 ones are 33 MB.
const MAX_RESIDENTS: usize = 24;

/// Total bytes of retained colour targets.
///
/// 96 MB is twelve 1920x1080 BGRA8 attachments. Sized against what a driven
/// macos-13 boot actually asks for rather than against the host's memory:
/// `metal_resident_evicted` is the counter that says whether it is too small.
const MAX_RESIDENT_BYTES: u64 = 96 << 20;

/// What identifies one retained colour target.
///
/// The mapping is the identity a mapper-ref-texture attachment has; geometry
/// and format are in the key rather than checked beside it because a texture of
/// the wrong shape is not the same target, and a key that omitted them would
/// hand a 4x4 texture to a 1920x1080 pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResidentColorKey {
    /// The guest surface the attachment renders into. Never 0 — an attachment
    /// with no mapping has no stable identity and is not retained.
    pub mapping_id: u32,
    pub width: u32,
    pub height: u32,
    /// The `pixel_format` ordinal as [`super::render::ColorRt`] carries it,
    /// before this rail's `0 = RGBA8Unorm` default is applied. Keyed on the
    /// caller's word rather than the resolved `MTLPixelFormat` so that two
    /// spellings of one format cannot become two residents.
    pub pixel_format: u32,
}

impl ResidentColorKey {
    /// The key this rail retains a mapper-ref-texture surface's colour render
    /// target under.
    ///
    /// # Why this is a constructor and not four field writes
    ///
    /// Three call sites name this key for one texture: the draw that takes it,
    /// the sampled bind that reads it, and the present capture that reads it.
    /// [`Self::pixel_format`] is the rail's own word for the target's format and
    /// not the guest's declaration, so a site that spelled it from the guest's
    /// `format` would key a *different* resident than the one the draw filled —
    /// and the failure is silent, because a key that matches nothing simply
    /// misses and the reader falls through to a slower source that still
    /// answers. Naming it once means the day this rail renders a target in the
    /// format the guest declared, every site moves together or none does.
    pub fn for_surface(mapping_id: u32, width: u32, height: u32) -> Self {
        Self {
            mapping_id,
            width,
            height,
            // The word this rail hands the backend for every colour target: 0,
            // its `RGBA8Unorm` writeback format. See
            // `crate::runtime::draw::note_store_narrowing` for what that costs a
            // guest that declared a wider one.
            pixel_format: 0,
        }
    }

    fn bytes(&self, bpp: usize) -> u64 {
        u64::from(self.width)
            .saturating_mul(u64::from(self.height))
            .saturating_mul(bpp as u64)
    }
}

/// One retained target: this rail's payload plus the bookkeeping that decides
/// whether it may be loaded from and when it is dropped.
///
/// Generic in the payload for one reason: the rules below — the generation
/// test, the two bounds, the eviction order — are the half that decides
/// correctness, and an `MTLTexture` cannot be built without a device. With the
/// payload abstract they are exercised directly instead of only through a
/// pathway no test can reach.
struct Slot<T> {
    key: ResidentColorKey,
    payload: T,
    bytes: u64,
    /// The surface cache's `host_gen` for the frame this payload's pixels *are*.
    ///
    /// `0` means "no published frame corresponds to these pixels" and never
    /// matches an ask, because `DeviceState::next_sampled_content_generation`
    /// does not issue 0. A target is registered at 0 and only leaves that state
    /// when a Store has published its readback.
    content_gen: u64,
    /// Use order, for eviction. Not a timestamp: a counter cannot go backwards
    /// and needs no clock.
    used: u64,
}

struct Registry<T> {
    entries: Vec<Slot<T>>,
    bytes: u64,
    clock: u64,
}

impl<T: Clone> Registry<T> {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            clock: 0,
        }
    }

    fn position(&self, key: &ResidentColorKey) -> Option<usize> {
        self.entries.iter().position(|e| e.key == *key)
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    /// The payload for `key`, and whether its pixels are the cache's frame at
    /// `content_gen`.
    ///
    /// Handing it out always retires the published claim. See [`take`].
    fn take(&mut self, key: &ResidentColorKey, content_gen: u64) -> Option<(T, bool)> {
        let now = self.tick();
        let idx = self.position(key)?;
        let entry = &mut self.entries[idx];
        entry.used = now;
        let holds_prior = content_gen != 0 && entry.content_gen == content_gen;
        entry.content_gen = 0;
        Some((entry.payload.clone(), holds_prior))
    }

    /// The payload for `key` **only if** its pixels are the frame published at
    /// `content_gen`, leaving the claim in place.
    ///
    /// See [`borrow_published`] for why this one does not retire.
    fn borrow_published(&mut self, key: &ResidentColorKey, content_gen: u64) -> Option<T> {
        if content_gen == 0 {
            return None;
        }
        let now = self.tick();
        let idx = self.position(key)?;
        let entry = &mut self.entries[idx];
        if entry.content_gen != content_gen {
            return None;
        }
        entry.used = now;
        Some(entry.payload.clone())
    }

    fn admit(&mut self, key: ResidentColorKey, payload: T, bytes: u64) {
        let now = self.tick();
        if let Some(idx) = self.position(&key) {
            // A key whose target was evicted between a caller's lookup and
            // here, or a geometry this rail re-created. Replace rather than
            // duplicate: two entries under one key would make `current` answer
            // from whichever the scan reached first.
            let old = self.entries.swap_remove(idx);
            self.bytes = self.bytes.saturating_sub(old.bytes);
        }
        self.entries.push(Slot {
            key,
            payload,
            bytes,
            content_gen: 0,
            used: now,
        });
        self.bytes = self.bytes.saturating_add(bytes);
        self.trim();
    }

    fn publish(&mut self, key: &ResidentColorKey, content_gen: u64) {
        if content_gen == 0 {
            return;
        }
        if let Some(idx) = self.position(key) {
            self.entries[idx].content_gen = content_gen;
        }
    }

    fn forget_mapping(&mut self, mapping_id: u32) {
        let mut freed = 0u64;
        self.entries.retain(|e| {
            let keep = e.key.mapping_id != mapping_id;
            if !keep {
                freed = freed.saturating_add(e.bytes);
            }
            keep
        });
        self.bytes = self.bytes.saturating_sub(freed);
    }

    /// Drop least-recently-used entries until both bounds hold.
    ///
    /// Runs after an insert, and never empties the table. Both facts serve one
    /// case: an attachment larger than the whole byte budget. Trimming before
    /// the insert, or trimming down to nothing, drops it on the way in — and
    /// then this rail allocates a texture per draw, registers it, evicts it,
    /// and pays the eviction on top of everything it used to pay. A single
    /// oversized target is over budget and kept; the budget bounds the tail,
    /// not the one attachment the guest is actually drawing into.
    fn trim(&mut self) {
        while self.entries.len() > MAX_RESIDENTS || self.bytes > MAX_RESIDENT_BYTES {
            if self.entries.len() <= 1 {
                return;
            }
            let Some(victim) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i)
            else {
                return;
            };
            let gone = self.entries.swap_remove(victim);
            self.bytes = self.bytes.saturating_sub(gone.bytes);
            crate::runtime::drain::note_store_route("metal_resident_evicted");
        }
    }

    fn levels(&self) -> (usize, u64, usize) {
        (
            self.entries.len(),
            self.bytes,
            self.entries.iter().filter(|e| e.content_gen != 0).count(),
        )
    }
}

static REGISTRY: Mutex<Registry<Texture>> = Mutex::new(Registry::new());

/// The retained texture for `key`, if this rail still holds one, and whether
/// its pixels are already the surface cache's frame at `content_gen`.
///
/// `false` is not a failure: it is a texture whose *allocation* is reusable and
/// whose *content* is not, which is the pass that uploads a seed into a target
/// it did not have to allocate. Spending one answer for the other is the whole
/// hazard, so they arrive together and neither can be read without the other.
///
/// `content_gen` must be a generation the caller read from the surface cache
/// *and* found current under the seed door's evidence standard. A caller that
/// passes a generation it did not check is asking this module to vouch for
/// something it cannot see; `0` is the honest spelling of "I have no entry to
/// compare against" and never matches.
///
/// # Taking it retires the claim
///
/// The caller is about to render into this texture, so from the moment it is
/// handed over its pixels are no longer the frame the cache holds — whatever
/// the answer was. Only [`published`] puts the claim back, and only a Store
/// that actually refreshed the cache may call it. That is why there is one
/// function and not a lookup beside an invalidate: a rail that could read the
/// texture without retiring the claim would, on the draw whose Store is skipped
/// (a multi-draw chain's intermediate record), leave a target holding
/// intermediate pixels while still claiming to be the published frame.
pub fn take(key: &ResidentColorKey, content_gen: u64) -> Option<(Texture, bool)> {
    REGISTRY.lock().take(key, content_gen)
}

/// The retained texture for `key` **only if** its pixels are still the frame
/// published at `content_gen`, without retiring that claim.
///
/// # The counterpart to `take`, and why the difference is not an oversight
///
/// [`take`] retires because its caller is about to *render into* the texture,
/// so from the moment it is handed over the pixels stop being the published
/// frame. This caller only reads them out and puts nothing back, so the frame
/// the texture holds after the read is the same frame it held before — and
/// retiring here would cost the next draw a whole-attachment upload for a claim
/// that was never actually invalidated.
///
/// The generation test is the same one [`take`] applies and carries the same
/// meaning: `content_gen` must be the surface cache's own generation for this
/// mapping, read under the seed door's currency standard. A caller that passes a
/// generation it did not check is asking this module to vouch for what it cannot
/// see, and `0` — which
/// `DeviceState::next_sampled_content_generation` never issues — is the honest
/// spelling of "I have no published frame to compare against".
pub fn borrow_published(key: &ResidentColorKey, content_gen: u64) -> Option<Texture> {
    REGISTRY.lock().borrow_published(key, content_gen)
}

/// This mapping's published frame, read out of the resident colour target as
/// tight RGBA8.
///
/// The one source that answers when [`crate::runtime::surface_cache`] has ceded
/// a mapping's frame to this rail. Both of the cache's byte readers — the
/// present capture and the sampled bind of a surface — take it from here rather
/// than each reaching into the registry with their own idea of the key, the
/// texel order, and the row stride.
///
/// `None` when no resident holds the frame at `content_gen`, which is a lawful
/// steady-state answer (a cold mapping, a target the byte budget evicted, a
/// frame some other writer has replaced) and never a failure the caller has to
/// report as one.
///
/// # RGBA8, because that is what the texture holds
///
/// The resident is `RGBA8Unorm` — see [`ResidentColorKey::for_surface`] — so
/// this is the readback verbatim, with no exchange. It is spelled in the name
/// rather than left to the caller to infer, because the caller that wants BGRA8
/// (the present capture) and the caller that wants RGBA8 (the sampled bind)
/// would otherwise both have to know this rail's private choice of render-target
/// format, and a wrong guess is a channel swap that renders as an orange sky
/// rather than as a failure.
pub fn read_published_rgba8(key: &ResidentColorKey, content_gen: u64) -> Option<Vec<u8>> {
    use crate::protocol::pixel_format::RGBA8_BPP;
    // Only the format this rail actually renders colour targets in reads back as
    // RGBA8. A key naming any other one is a future this function has not been
    // taught, and `None` sends the caller to the source it already falls through
    // to rather than to a frame with the wrong texels in it.
    if key.pixel_format != 0 || key.width == 0 || key.height == 0 {
        return None;
    }
    let stride = (key.width as usize).checked_mul(RGBA8_BPP as usize)?;
    let need = stride.checked_mul(key.height as usize)?;
    let texture = borrow_published(key, content_gen)?;
    // Priced, because `getBytes:` on a tiled texture is not a memcpy — it
    // linearizes — and this readback replaced a `copy_from_slice` of a host
    // `Vec` for two callers that run about once a frame each. A rail that spends
    // more here than the host copy it removed has to be able to say so.
    let _span = crate::runtime::chain_phase::CostSpan::new("metal_resident_read_us");
    let mut rgba = vec![0u8; need];
    // SAFETY: every render into a resident colour target completes before its
    // Store publishes it (`render_core_mrt` waits), and `borrow_published`
    // succeeded, so the claim this reads under is one a completed Store made.
    // The slice does not outlive `texture`.
    if let Some(linear) =
        unsafe { raw_metal::linear_pixels(&texture, stride as u64, u64::from(key.height)) }
    {
        crate::runtime::drain::note_store_route("metal_resident_read_linear");
        rgba.copy_from_slice(linear);
    } else {
        crate::runtime::drain::note_store_route("metal_resident_read_tiled");
        let region = metal::MTLRegion {
            origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
            size: metal::MTLSize {
                width: u64::from(key.width),
                height: u64::from(key.height),
                depth: 1,
            },
        };
        texture.get_bytes(rgba.as_mut_ptr() as *mut _, stride as u64, region, 0);
    }
    // Count and bytes together: the rate is what says whether a reader is asking
    // for whole frames it does not need, and neither number can say it alone.
    crate::runtime::drain::note_store_route("metal_resident_reads");
    crate::runtime::drain::note_store_route_n("metal_resident_read_bytes", need as u64);
    Some(rgba)
}

/// Build a colour target for `key` and retain it.
///
/// Returns `None` when Metal would not allocate the texture; the caller refuses
/// the draw rather than rendering into nothing.
pub fn create(
    device: &DeviceRef,
    key: &ResidentColorKey,
    format: MTLPixelFormat,
    bpp: usize,
) -> Option<Texture> {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(format);
    descriptor.set_width(u64::from(key.width));
    descriptor.set_height(u64::from(key.height));
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let texture = linear_target(device, key, format, bpp, &descriptor)
        .or_else(|| raw_metal::new_texture(device, &descriptor))?;
    REGISTRY.lock().admit(*key, texture.clone(), key.bytes(bpp));
    Some(texture)
}

/// A colour target whose texels are a CPU-visible buffer's bytes, so that
/// reading them back is a pointer rather than a de-tiling copy. See
/// [`raw_metal::new_linear_texture`].
///
/// `None` sends [`create`] to the ordinary tiled texture, and that is the
/// capability gate: a host whose GPU will not render into a linear texture
/// answers nil at `newTextureWithDescriptor:offset:bytesPerRow:` and this rail
/// keeps the behaviour it had, one counter poorer. Nothing downstream has to
/// know which it got — [`read_published_rgba8`] and the draw's own readback both
/// ask the *texture* where its pixels are.
///
/// Row pitch is the tight row, rounded up to what the device requires. Padding
/// it would be lawful and is not done: every reader of these pixels wants a
/// tight frame, so a padded pitch would trade the copy this removes for a
/// per-row one. `RGBA8Unorm` needs 16-byte alignment on this host and a 1920-wide
/// tight row is 7680, so the rounding is a no-op at the geometry that matters
/// and a correctness requirement at the ones that do not.
fn linear_target(
    device: &DeviceRef,
    key: &ResidentColorKey,
    format: MTLPixelFormat,
    bpp: usize,
    descriptor: &metal::TextureDescriptorRef,
) -> Option<Texture> {
    let alignment = device.minimum_linear_texture_alignment_for_pixel_format(format);
    let tight = u64::from(key.width).checked_mul(bpp as u64)?;
    let bytes_per_row = if alignment == 0 {
        tight
    } else {
        tight.div_ceil(alignment).checked_mul(alignment)?
    };
    let length = bytes_per_row.checked_mul(u64::from(key.height))?;
    let buffer = raw_metal::new_buffer(device, length, MTLResourceOptions::StorageModeShared)?;
    let texture = raw_metal::new_linear_texture(&buffer, descriptor, 0, bytes_per_row);
    crate::runtime::drain::note_store_route(if texture.is_some() {
        "metal_resident_linear"
    } else {
        // Counted rather than logged per target: a host that cannot render into
        // a linear texture answers this for every target it ever creates, and
        // the census is where a rail-wide capability belongs.
        "metal_resident_tiled"
    });
    texture
}

/// Record that the retained texture for `key` now holds exactly the surface
/// cache's frame at `content_gen`.
///
/// Called by the Store, after the writeback has published the readback — never
/// before. Publishing a generation for pixels that have not reached the cache
/// would let the next draw skip an upload of bytes the cache does not hold.
pub fn published(key: &ResidentColorKey, content_gen: u64) {
    REGISTRY.lock().publish(key, content_gen);
}

/// Drop every retained target for `mapping_id`.
///
/// For a surface the guest has unmapped or this device has retired: the pixels
/// are gone and the memory should follow them. Not needed for correctness — a
/// stale entry can never be *served*, because its generation cannot match a
/// cache entry that no longer exists — which is why this is about memory and is
/// stated as such.
pub fn forget(mapping_id: u32) {
    REGISTRY.lock().forget_mapping(mapping_id);
}

/// `(count, bytes, live)` — retained targets, their total size, and how many of
/// them hold a published frame a load could actually be served from.
///
/// The third number is the one that says whether the rail is working. A count
/// that climbs while `live` stays at zero is a rail retaining textures and
/// re-uploading into all of them, which reads as a win on memory and is a loss
/// on everything.
pub fn levels() -> (usize, u64, usize) {
    REGISTRY.lock().levels()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(mapping_id: u32) -> ResidentColorKey {
        ResidentColorKey {
            mapping_id,
            width: 4,
            height: 4,
            pixel_format: 0,
        }
    }

    /// A reader borrows the published frame without retiring the claim, and a
    /// renderer taking the same target still does.
    ///
    /// The two are one hazard read from both sides. `take` must retire because
    /// its caller overwrites the pixels; `borrow_published` must not, because
    /// its caller only reads them — and a `borrow_published` that retired would
    /// make every present capture and every sampled bind of a ceded surface cost
    /// the *next* draw a whole-attachment upload, which reads as the rail being
    /// slow and never as a failure.
    #[test]
    fn a_borrow_reads_the_published_frame_and_a_take_retires_it() {
        let mut reg: Registry<u32> = Registry::new();
        reg.admit(key(1), 11, 64);
        reg.publish(&key(1), 7);

        assert_eq!(reg.borrow_published(&key(1), 7), Some(11));
        assert_eq!(
            reg.borrow_published(&key(1), 7),
            Some(11),
            "reading the frame does not consume it"
        );
        assert_eq!(
            reg.levels().2,
            1,
            "the target still holds a frame a load could be served from"
        );

        assert_eq!(
            reg.borrow_published(&key(1), 8),
            None,
            "a frame the surface has been republished since is not this one"
        );
        assert_eq!(
            reg.borrow_published(&key(1), 0),
            None,
            "0 is the caller having no frame to compare against, not a match"
        );
        assert_eq!(reg.borrow_published(&key(2), 7), None, "another mapping");

        assert_eq!(reg.take(&key(1), 7), Some((11, true)));
        assert_eq!(
            reg.borrow_published(&key(1), 7),
            None,
            "the renderer took it, so the pixels are no longer the published frame"
        );
    }

    /// A retained target may be loaded from only under the generation it was
    /// published at, and every other answer sends the caller to the upload.
    ///
    /// The three refusals are the whole safety rule. A freshly created target
    /// holds pixels no published frame corresponds to; a target whose surface
    /// has been republished since holds the previous frame; and generation 0 is
    /// the caller saying it has no cache entry to compare against, which is a
    /// question this module cannot answer rather than a match.
    #[test]
    fn a_retained_target_loads_only_under_the_generation_it_was_published_at() {
        let mut reg: Registry<u32> = Registry::new();
        reg.admit(key(1), 7, 64);
        assert_eq!(
            reg.take(&key(1), 5),
            Some((7, false)),
            "a target that has never been published holds no frame"
        );
        reg.publish(&key(1), 5);
        assert_eq!(reg.take(&key(1), 5), Some((7, true)));
        reg.publish(&key(1), 5);
        assert_eq!(
            reg.take(&key(1), 6),
            Some((7, false)),
            "the surface was republished, so these pixels are the previous frame"
        );
        reg.publish(&key(1), 5);
        assert_eq!(
            reg.take(&key(1), 0),
            Some((7, false)),
            "generation 0 is the absence of a cache entry, not a match"
        );
    }

    /// Taking the texture retires the published claim even when the answer was
    /// "yes, it holds the frame" — because the caller is about to render into
    /// it, and a Store that does not publish (a multi-draw chain's intermediate
    /// record) would otherwise leave intermediate pixels claiming to be the
    /// frame the cache holds.
    #[test]
    fn taking_a_target_retires_its_published_frame_whatever_the_answer_was() {
        let mut reg: Registry<u32> = Registry::new();
        reg.admit(key(1), 7, 64);
        reg.publish(&key(1), 5);
        assert_eq!(reg.take(&key(1), 5), Some((7, true)));
        assert_eq!(
            reg.take(&key(1), 5),
            Some((7, false)),
            "the pass that took it rendered into it; these pixels are not frame 5"
        );
    }

    /// One key holds one target. A geometry this rail re-created must replace
    /// the entry rather than sit beside it, because two entries under one key
    /// make every lookup answer from whichever the scan reaches first.
    #[test]
    fn re_admitting_a_key_replaces_its_target_rather_than_duplicating_it() {
        let mut reg: Registry<u32> = Registry::new();
        reg.admit(key(1), 7, 64);
        reg.publish(&key(1), 5);
        reg.admit(key(1), 9, 128);
        assert_eq!(reg.entries.len(), 1);
        assert_eq!(
            reg.bytes, 128,
            "the replaced target's bytes must be released"
        );
        assert_eq!(
            reg.take(&key(1), 5),
            Some((9, false)),
            "a fresh target holds no published frame, whatever the old one held"
        );
    }

    /// Both bounds, and the entry just admitted is never the one evicted.
    ///
    /// The count bound and the byte bound admit each other's worst case —
    /// sixteen 4x4 targets are nothing and four 1920x1080 ones are 33 MB — so
    /// each is asserted on a population the other would not have trimmed.
    #[test]
    fn the_trim_drops_least_recently_used_until_both_bounds_hold() {
        let mut reg: Registry<u32> = Registry::new();
        for i in 0..(MAX_RESIDENTS as u32 + 3) {
            reg.admit(key(i + 1), i, 64);
        }
        assert_eq!(reg.entries.len(), MAX_RESIDENTS);
        assert!(
            reg.position(&key(MAX_RESIDENTS as u32 + 3)).is_some(),
            "the most recently admitted target is the last thing that may be evicted"
        );
        assert!(
            reg.position(&key(1)).is_none(),
            "the least recently used target is the first"
        );

        let mut big: Registry<u32> = Registry::new();
        let half = MAX_RESIDENT_BYTES / 2 + 1;
        big.admit(key(1), 1, half);
        big.admit(key(2), 2, half);
        assert_eq!(big.entries.len(), 1, "two of these do not fit");
        assert!(big.position(&key(2)).is_some());

        // A single target larger than the whole budget is still admitted and
        // still usable for the draw that asked for it: dropping it on the way
        // in leaves the rail allocating a texture per draw, registering it and
        // evicting it — the old cost plus the eviction.
        let mut huge: Registry<u32> = Registry::new();
        huge.admit(key(1), 1, MAX_RESIDENT_BYTES * 4);
        assert_eq!(huge.entries.len(), 1);
        // And the next one still replaces it rather than joining it, so an
        // over-budget table cannot grow.
        huge.admit(key(2), 2, MAX_RESIDENT_BYTES * 4);
        assert_eq!(huge.entries.len(), 1);
        assert!(huge.position(&key(2)).is_some());
    }

    /// Retiring a surface releases every target it held, and its bytes.
    #[test]
    fn forgetting_a_mapping_releases_its_targets_and_their_bytes() {
        let mut reg: Registry<u32> = Registry::new();
        reg.admit(key(1), 1, 64);
        reg.admit(
            ResidentColorKey {
                mapping_id: 1,
                width: 8,
                height: 8,
                pixel_format: 0,
            },
            2,
            256,
        );
        reg.admit(key(2), 3, 64);
        assert_eq!(reg.bytes, 384);
        reg.forget_mapping(1);
        assert_eq!(reg.entries.len(), 1);
        assert_eq!(reg.bytes, 64);
        assert!(reg.position(&key(2)).is_some());
    }

    /// The census separates "retained" from "loadable", because a rail that
    /// retains everything and publishes nothing reads as a win on memory and is
    /// a loss on everything else.
    #[test]
    fn the_census_counts_published_targets_apart_from_retained_ones() {
        let mut reg: Registry<u32> = Registry::new();
        reg.admit(key(1), 1, 64);
        reg.admit(key(2), 2, 64);
        assert_eq!(reg.levels(), (2, 128, 0));
        reg.publish(&key(1), 5);
        assert_eq!(reg.levels(), (2, 128, 1));
    }
}
