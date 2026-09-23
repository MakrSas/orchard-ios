//! Object-list lookup, mapper-ref-texture registration, and x86 backing records.
//!
//! Live layout (reims-vgpu-resource-format): entry `ref` is at
//! `(object_list_pfn << PAGE_SHIFT) + ref * 12` in the task GVA space —
//! `[type|desc_len packed u32][desc_gva u64]`.
//!
//! **x86 backing present path (Ventura 13.7 RE):**
//! `AppleParavirtResource::allocateBackingHandle` calls
//! `ResourceHeap::addObject(type=4, objectId=IOSurface::getSurfaceID(), …)` so
//! the object-list index for a surface-backed resource **is** the present
//! `surface_id`. The descriptor's own layout is
//! [`reims_vgpu_wire::device_desc`], which states it once with `offset_of!`
//! rather than as the eight literal offsets that used to sit in this module.

use crate::model::{DeviceState, MappingEntry, TaskResource, TaskTable};
use crate::protocol::endian::{ld32, st16, st32, st64};
use crate::protocol::iosurface_pages::{
    entry_gpa_shift, page_size_of, DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BASE_OFFSET,
    DEVICE_DESC_BPE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS, DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT,
    DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT, DEVICE_PLANE_BPE, DEVICE_PLANE_BPR,
    DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS, DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE,
    PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
};
use crate::runtime::decode::resource::{
    decode_list_object_entry, list_object_entry_offset, ListObjectEntry, OBJECT_LIST_ENTRY_LEN,
    OBJECT_TYPE_BUFFER, OBJECT_TYPE_FUNCTION, OBJECT_TYPE_MAPPER_REF_TEXTURE,
    OBJECT_TYPE_SERIALIZER_OBJECT, OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS,
    OBJECT_TYPE_TEXTURE_VIEW,
};
use crate::runtime::gva_mem;
use crate::runtime::host::HostMemory;
use crate::runtime::texture;
use std::sync::Arc;

pub mod slot_recheck;

/// Fail-visible, de-duplicated per `(task_id, ref)`, for the mapper-ref-texture resolve
/// blind spot: an object ref that IS a mapper-ref-texture IOSurface texture but whose
/// descriptor cannot be read, cannot register a Metal/Vulkan texture, or carries
/// `mapping_id==0` used to collapse into a bare `None` → a coarse
/// `MissingTexture` at the draw site with no reason. `resolve_mapper_ref_texture` runs
/// per-draw per-ref (very hot), so a bare fail line would flood; the latch logs
/// each `(task,ref,reason)` once and is cleared when the ref resolves
/// ([`clear_mapper_ref_texture_fail`]). Only genuine failures for a *confirmed IOSurface*
/// ref are routed here — the legitimate "ref is a different object type" and
/// unbound-slot returns stay silent. Runs on the drain worker (off the QEMU main
/// core).
type MapperRefTextureFailure = (u32, u32, &'static str);
type MapperRefTextureFailureSet = std::collections::HashSet<MapperRefTextureFailure>;

fn mapper_ref_texture_fail_latch() -> &'static std::sync::Mutex<MapperRefTextureFailureSet> {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<MapperRefTextureFailureSet>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(MapperRefTextureFailureSet::new()))
}

fn note_mapper_ref_texture_fail(task_id: u32, ref_: u32, reason: &'static str, detail: String) {
    let mut guard = mapper_ref_texture_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((task_id, ref_, reason)) {
        crate::observe::fail(detail);
    }
}

/// Re-arm the fail latch for a ref that just resolved, so a later genuine
/// failure on the same ref is logged again (catches flapping).
fn clear_mapper_ref_texture_fail(task_id: u32, ref_: u32) {
    let mut guard = mapper_ref_texture_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.retain(|(t, r, _)| !(*t == task_id && *r == ref_));
}

/// Fail-visible, de-duplicated per `(surface_id, reason)`, for the backing
/// backing blind spot: a surface whose object-list descriptor decoded fine (an
/// active task, a valid `BackingRecord`) but whose page-backing construction then
/// failed — every downstream present/Store for that surface paints **stale or
/// black** with no reason. `apply_backing` is reached from the per-present
/// scanout path (`ensure_surface_for_present`, ~48/s under scroll), so a persistent
/// backing failure would flood; the latch logs each `(surface_id, reason)` once
/// and re-arms when the surface next resolves cleanly ([`clear_backing_fail`]), so a
/// flapping backing is re-logged. Only genuine backing candidate failures are
/// routed here — the caller's speculative per-task `continue`s (surface absent
/// from this task or a non-surface object type) stay silent. Runs on the drain
/// worker (off the QEMU main core).
///
/// # What the latch remembers, and why it is not just a set
///
/// The backing this device asked for, and when it asked. A refusal here is
/// frequently **transient** — the guest had not finished mapping the surface
/// when the per-present path first walked it, and the next attach resolves the
/// same pages — and the log gave no way to tell that from a surface this device
/// never managed to back. See [`clear_backing_fail`] for what the pair enables and
/// what it measured.
///
/// `gva` is the backing base the refusal named, or `None` for the refusals
/// raised before one can be computed (`sid_zero`, `page_size_zero`). It is the
/// backing and not `surface_id` that identifies a recovery, because surface ids
/// recycle within a boot and across geometries — the same caveat
/// `apply_backing`'s census line carries `gva0` for.
#[derive(Clone, Copy)]
struct ReportedBackingFail {
    gva: Option<u64>,
    /// When this refusal was first raised. **Never refreshed**, so its age is
    /// how long the surface has been unbacked.
    first_at_ms: u64,
    /// When the device last asked for this backing and was refused again. Its
    /// age is how long since anything asked at all.
    last_at_ms: u64,
    /// How many times the device has asked and been refused, the first
    /// included. The whole point of the pair: see
    /// [`backing_outstanding_census`].
    attempts: u32,
}

type BackingFailLatch = std::collections::HashMap<(u32, &'static str), ReportedBackingFail>;

fn backing_fail_latch() -> &'static std::sync::Mutex<BackingFailLatch> {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<BackingFailLatch>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(BackingFailLatch::new()))
}

fn note_backing_fail(surface_id: u32, reason: &'static str, gva: Option<u64>, detail: String) {
    let at_ms = crate::observe::elapsed_ms() as u64;
    let mut guard = backing_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    match guard.entry((surface_id, reason)) {
        std::collections::hash_map::Entry::Occupied(mut slot) => {
            // A repeat is the retry the design asks for, and it stays quiet on
            // the fail channel — `apply_backing` is the per-present path
            // and one line per frame would flood. It is counted instead, which
            // is what makes the silence readable.
            let held = slot.get_mut();
            held.gva = gva;
            held.last_at_ms = at_ms;
            held.attempts = held.attempts.saturating_add(1);
        }
        std::collections::hash_map::Entry::Vacant(slot) => {
            slot.insert(ReportedBackingFail {
                gva,
                first_at_ms: at_ms,
                last_at_ms: at_ms,
                attempts: 1,
            });
            crate::observe::fail(detail);
        }
    }
}

/// The first probe failure of the search in progress, per surface.
///
/// A surface lives in exactly one task's object list, so a search that walks
/// tasks in order meets non-owners before it meets the owner. Those misses are
/// the search working, not a backing failure, and reporting them as one is what
/// put ~95 `backing_fail reason=translate` lines on a driven boot's
/// always-on channel for surfaces that then backed perfectly — the resolve
/// succeeded on a later task and the line stayed behind to be read as a defect.
///
/// So a probe records its reason here and nothing is emitted until the search
/// runs out of tasks. The first reason is kept rather than the last: it is the
/// most specific one available, and the tail of a search is dominated by tasks
/// that simply do not list the surface.
struct PendingBackingFail {
    reason: &'static str,
    gva: Option<u64>,
    detail: String,
}

type BackingPendingLatch = std::collections::HashMap<u32, PendingBackingFail>;

fn backing_pending_latch() -> &'static std::sync::Mutex<BackingPendingLatch> {
    use std::sync::{Mutex, OnceLock};
    static PENDING: OnceLock<Mutex<BackingPendingLatch>> = OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(BackingPendingLatch::new()))
}

/// Record why one task's probe refused, to be reported only if none succeeds.
///
/// `gva` is the backing base the probe was walking, threaded through so a later
/// clean attach on the *same* backing can be recognised as a recovery rather
/// than guessed at from `surface_id`. `None` where the refusal precedes any
/// computable address.
fn defer_backing_fail(surface_id: u32, reason: &'static str, gva: Option<u64>, detail: String) {
    let mut guard = backing_pending_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.entry(surface_id).or_insert(PendingBackingFail {
        reason,
        gva,
        detail,
    });
}

/// The search found no task that could back this surface: report the first
/// probe's reason through the flood latch.
fn flush_backing_fail(surface_id: u32) {
    let pending = {
        let mut guard = backing_pending_latch()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.remove(&surface_id)
    };
    if let Some(pending) = pending {
        note_backing_fail(surface_id, pending.reason, pending.gva, pending.detail);
    }
}

/// Re-arm the backing fail latch for a surface that just backed cleanly, drop the
/// probe reasons the successful search left behind, and — when the backing that
/// landed is the one a refusal named — say so.
///
/// # A refusal here is usually a retry that then worked, and the log did not say
///
/// `backing_fail reason=translate` reads as lost guest work: the surface
/// could not be backed, so every present for it paints stale or black. It is
/// frequently nothing of the sort. `st=zero-pfn pte=0x0` means the guest had not
/// filled the leaf PTE when the per-present path walked it, and the refusal is
/// what makes the device ask again next frame instead of substituting a guess.
/// The retry is the design; a *silent* retry is what made the class unreadable.
///
/// Measured, driven x86/PCI, `web-content-probe -n 10 --churn 1` — six
/// refusals, and every one recovered on the same backing:
///
/// ```text
///   sid   refused at   backed at   delta
///    11        23652       23673    21 ms
///    72        49139       49146     7 ms
///    73        49165       49171     6 ms
///    48        51770       51771     1 ms
/// ```
///
/// Each was confirmed the *same* backing by matching the refusal's `gva=`
/// against the later attach's `gva0=`, not by surface id — ids recycle within a
/// boot and across geometries, so "the same sid resolved a frame later" can be a
/// different surface wearing the same number. That is why `gva` is threaded
/// through the latch rather than reconstructed here, and why a mismatch claims
/// nothing.
///
/// A prior session read this class as "the best-defined live failure class left"
/// and queued work against it. It was six successful retries. The recovery line
/// is what stops that from happening again, and a refusal that stays unpaired is
/// now the signal that something really did not back.
///
/// # The emitter confirms it on its own output
///
/// The table above was reconstructed by hand from a log that did not carry the
/// pairing. A second driven boot with this line in place produces it directly —
/// three refusals, three recoveries, **no unpaired refusal**, matched on the
/// backing:
///
/// ```text
///   backing_recovered sid=25 reason=translate gva=0x42ab000 after_ms=5
///   backing_recovered sid=25 reason=translate gva=0x4282000 after_ms=7
///   backing_recovered sid=46 reason=translate gva=0x93b1000 after_ms=20
/// ```
///
/// Note the two `sid=25` at different backings: the id really does get reused
/// inside one boot, and matching on it would have paired the second refusal with
/// the first attach. `backing_translate_refused` and `backing_recovered` sum
/// to 3 and 3 across the census windows, so the counters agree with the lines.
fn clear_backing_fail(surface_id: u32, backed_gva: u64) {
    backing_pending_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&surface_id);
    let mut superseded = 0usize;
    let recovered: Vec<(&'static str, u64)> = {
        let mut guard = backing_fail_latch()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Every entry for this surface is dropped, exactly as before: the latch
        // must re-arm or a later genuine failure on a recycled id goes unlogged.
        // Only the ones whose backing matches are *claimed* as recoveries.
        let mut out = Vec::new();
        guard.retain(|(s, reason), reported| {
            if *s != surface_id {
                return true;
            }
            if reported.gva == Some(backed_gva) {
                out.push((*reason, reported.first_at_ms));
            } else {
                // Dropped without being claimed: this surface backed, but at a
                // different address than the refusal named — the guest
                // re-pointed it, so the refusal is moot rather than repaired.
                // Counted because it used to be the *third* way a refusal could
                // leave the latch and the only one that said nothing, which made
                // an unpaired `backing_fail` unreadable: a reader diffing
                // the two line counts could not tell a refusal the guest walked
                // away from apart from one that never came back.
                superseded += 1;
            }
            false
        });
        out
    };
    if superseded > 0 {
        crate::runtime::drain::note_store_route_n("backing_superseded", superseded as u64);
    }
    // After the early return, not before it: this runs on every clean attach
    // (the per-present scanout path) and almost every one of those has nothing
    // latched, so the clock read stays off the hot path.
    if recovered.is_empty() {
        return;
    }
    let now = crate::observe::elapsed_ms() as u64;
    for (reason, at_ms) in recovered {
        crate::runtime::drain::note_store_route("backing_recovered");
        crate::observe::fail(format!(
            "backing_recovered sid={surface_id} reason={reason} gva={backed_gva:#x} \
             after_ms={} (the earlier refusal for this backing was a retry; the guest \
             finished mapping and it landed)",
            now.saturating_sub(at_ms)
        ));
    }
}

/// The backing refusals still latched, for the census — or `None` if there are
/// none.
///
/// # The reading this exists to make possible
///
/// A refusal leaves [`backing_fail_latch`] exactly three ways, and until this
/// existed only one of them said so. It is *recovered* when the surface backs at
/// the address the refusal named, which emits `backing_recovered`. It is
/// *superseded* when the surface backs somewhere else — the guest re-pointed it,
/// so the refusal is moot — which [`clear_backing_fail`] now counts. Or it is
/// still here, which is the only one that can be lost guest work, and it was the
/// silence the other two were mistaken for.
///
/// The three add up: `backing_fail` lines equal
/// `backing_recovered + backing_superseded + n` from the last
/// census window. That identity is the point — a reader who finds fewer
/// recoveries than refusals now has a line to check instead of a hand-matched
/// diff of backing GVAs, which is how this hole was found.
///
/// # Reading the three numbers, because two of them used to say the opposite
///
/// `oldest_ms` is the age of the longest-outstanding refusal's **first** raise,
/// `since_last_ms` the age of the most recent one, and `attempts` how many times
/// the device has asked. Together they separate the two states a bare count
/// cannot:
///
/// - `attempts` climbing, `since_last_ms` near zero — the device is asking every
///   frame and being refused every frame. **This is the one that is lost guest
///   work**, and every present for that surface is painting stale or black.
/// - `attempts=1`, `since_last_ms` tracking `oldest_ms` — the device asked once,
///   was refused, and **nothing has asked since**. `apply_backing` is
///   reached from the per-present path, so nothing asking means nothing is
///   presenting that surface: the guest is done with it. Nothing is lost.
///
/// This carried one number, `oldest_ms`, and its doc read it exactly backwards —
/// "`oldest_ms=12` is a refusal caught mid-retry; `oldest_ms=83000` is a surface
/// this device never backed". [`note_backing_fail`] used a plain `insert`, so a
/// retry *overwrote* the timestamp: an actively-retried refusal pins that age
/// near zero forever, and a large one means the retries stopped. The sentence
/// had it the wrong way round for both states, which is why the reading below
/// went unmade across three boots.
///
/// Silent when the latch is empty, because an empty latch is the expected state
/// and this rides a one-second cadence.
///
/// # What a driven boot reads
///
/// x86/PCI, `web-content-probe -n 10 --churn 1`, 12 951 log lines, probe green.
/// The identity closes exactly:
///
/// ```text
///   backing_fail        8
///   backing_recovered   7
///   backing_superseded  0
///   outstanding n             1      8 = 7 + 0 + 1
/// ```
///
/// Fifteen census lines for the boot, so the cadence costs nothing. The first
/// reads `oldest_ms=294` and the last `oldest_ms=14317` — which is the argument
/// for carrying the age at all, because one of those `n=1`s is a retry still in
/// flight and the other is a surface that was never backed, and the count alone
/// cannot tell them apart.
///
/// # The standing `sid=27` reading, and what it turned out to be
///
/// Three driven boots each ended with exactly one outstanding refusal, every
/// time `sid=27`, `reason=translate st=zero-pfn pte=0x0`, a 2-3 page surface, at
/// a different backing each boot. It was carried as an open question across two
/// sessions: abandoned by the guest, or a retry that never came?
///
/// **Abandoned.** It is answerable from the third boot's own census series
/// without any new signal, once `insert`'s refresh is accounted for. The 23
/// `backing_outstanding` lines run `oldest_ms` 204 → 22242 while `t` runs
/// 401037 → 423075: the two deltas are **equal at every sample**, 22038 apiece.
/// A timestamp that tracks wall clock exactly is a timestamp nothing refreshed,
/// so the device asked once, at t=400833, and never again.
///
/// The surface agrees. `sid=27` is 93×21 and then 99×29 — a tooltip — set up at
/// t≈26 s and untouched for the following six minutes. `zero-pfn` says the
/// guest's own leaf PTE is empty: it took the backing away. Nothing presented it
/// afterwards, which is why nothing asked again.
///
/// `attempts` is in the line so this costs a glance rather than an afternoon,
/// and so the *other* state — a surface retried every frame and refused every
/// frame — is not mistaken for it. That one is real lost work and this boot does
/// not contain one.
pub(crate) fn backing_outstanding_census() -> Option<String> {
    let now = crate::observe::elapsed_ms() as u64;
    let guard = backing_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // The oldest entry is the one worth naming: it is the one least likely to
    // be a retry still in flight, and a single line cannot carry them all.
    let oldest = guard
        .iter()
        .min_by_key(|(_, reported)| reported.first_at_ms)
        .map(|((sid, reason), reported)| (*sid, *reason, *reported))?;
    let (sid, reason, held) = oldest;
    Some(format!(
        "backing_outstanding n={} oldest_ms={} since_last_ms={} attempts={} \
         sid={sid} reason={reason} gva={}",
        guard.len(),
        now.saturating_sub(held.first_at_ms),
        now.saturating_sub(held.last_at_ms),
        held.attempts,
        gva_text(held.gva),
    ))
}

fn gva_text(gva: Option<u64>) -> String {
    gva.map_or_else(|| "none".to_string(), |g| format!("{g:#x}"))
}

/// Wire object type for surface / IOSurface backing (x86 Tahoe/Ventura).
pub const OBJECT_TYPE_BACKING: u8 = 4;
/// RefTextureHandle: surfaceID@0 + cookie@4 + guest blob@8 (texture-ref 28-06-26).
pub const OBJECT_TYPE_REF_TEXTURE: u8 = 5;
/// Ref-texture RefTexture descriptor (RE `allocateRefTextureHandle` + Metal
/// `initWithDevice:descriptor:iosurface:plane:field:`):
/// - `surfaceID@0` = `IOSurface::getSurfaceID()` = backing heap object id / mid
/// - `ownerTask@4` = the task whose object list holds that surface
/// - `args@8..` = **serialized texture args** length `desc_len-8` (MTLTextureDescriptor
///   stream for the **plane** view; plane is applied guest-side before serialize)
///
/// See [[reims-vgpu-resource-paging]] ref-texture section.
/// Ref-texture descriptor geometry, from the wire crate's Tier-2 view of it.
///
/// The ten offsets these used to be are `offset_of!` on
/// [`reims_vgpu_wire::device_desc::RefTextureHeader`], `RefTextureArgsHeader` and
/// `RefTextureRecord`, asserted there against the numbers this module used to
/// state. The two record tags come with them, since a tag is part of the
/// layout's identity and not of this device's policy.
pub(crate) use reims_vgpu_wire::device_desc::TYPE5_ARGS;
// Only `TYPE5_ARGS` is still named by a product path — the decoders read the
// rest through the wire views. These five are named only by the tests that
// assert the layout, so they are gated with those tests rather than left
// reachable from the staticlib. `TYPE5_RECORD_TAG_PLANE` is not among them:
// its one caller builds the descriptor with `wire::device_desc::RefTextureBuilder`
// on the line above and now names the tag from the same module.
#[cfg(test)]
pub(crate) use reims_vgpu_wire::device_desc::{
    TYPE5_ARG_RECORD, TYPE5_OWNER_TASK, TYPE5_RECORD_PLANE, TYPE5_RECORD_TAG_COLOR_VIEW,
    TYPE5_SURFACE_ID,
};

/// Shortest ref-texture descriptor: the header, with no args blob behind it.
pub const TYPE5_MIN_LEN: usize = TYPE5_ARGS;

/// Texture view named by a ref-texture descriptor's serialized args record.
///
/// This is not limited to IOSurface planes. The live desktop also uses
/// row-byte-equivalent reinterpretations such as a 480-wide RGBA32Uint view
/// over a 1920-wide BGRA8 surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefTextureView {
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// IOSurface plane the serialized view binds (record `+0x20`); 0 when the
    /// record is too short to carry the field (pre-plane blobs, tests).
    pub plane_index: u32,
}

/// Report the owner task a ref-texture descriptor names, once per distinct value.
///
/// Every backing record this device has resolved lived in task 0 — measured
/// (`backing_claimants`: `claims=1 winner=0` on every surface id of two driven
/// boots) and structural, since the guest registers IOSurface backings in the
/// accelerator's kernel task whose id is a hardcoded 0.
/// [`reims_vgpu_wire::device_desc::TYPE5_OWNER_TASK`] is
/// the guest saying the same thing on the wire, so this reads 0 forever and
/// stays on the quiet channel.
///
/// A non-zero value is the one reading that matters, and it is a failure line
/// because two things would follow from it at once: the backing search's "task 0
/// first" probe order is no longer the guest's answer, and the field's decoded
/// meaning is wrong. `first_sight` is keyed on the value alone, so the whole
/// boot costs one line whichever way it goes.
fn note_ref_texture_owner_task(desc: &[u8]) {
    let Ok(h) = reims_vgpu_wire::device_desc::ref_texture_header(desc) else {
        return;
    };
    let task = h.owner_task.get();
    if !crate::observe::first_sight("ref_texture_owner_task", task as u64) {
        return;
    }
    let line = format!("ref_texture_owner_task task={task}");
    if task == 0 {
        crate::observe::off(line);
    } else {
        crate::observe::fail(format!(
            "{line} (a ref-texture view names a surface owner other than the kernel task; \
             the backing search probes task 0 first on the reading that this is always 0)"
        ));
    }
}

/// Decode the serialized texture-view record from a full ref-texture descriptor.
///
/// Fail-closed: `None` unless the record tag matches and geometry is sane
/// (2D, nonzero). The record names the exact Metal view (format + geometry)
/// over the IOSurface bytes; callers must not replace it with base mapping
/// geometry merely because the surface itself is otherwise stageable.
pub fn decode_ref_texture_view(desc: &[u8]) -> Option<RefTextureView> {
    note_ref_texture_owner_task(desc);
    let rec = reims_vgpu_wire::device_desc::ref_texture_record(desc).ok()?;
    // Both the biplanar-plane tag and the full-colour view variant share the
    // field layout; any other tag stays unknown → fail closed (no invented
    // geometry).
    if !rec.tag_is_known() {
        return None;
    }
    let pixel_format = rec.pixel_format.get();
    let width = rec.width.get();
    let height = rec.height.get();
    let depth = rec.depth.get();
    if pixel_format == 0 || width == 0 || height == 0 || depth != 1 {
        return None;
    }
    Some(RefTextureView {
        pixel_format,
        width,
        height,
        depth,
        // `None` is a blob that stops before the field — pre-plane descriptors
        // and fixtures — which this rail reads as plane 0.
        plane_index: reims_vgpu_wire::device_desc::ref_texture_record_plane_index(desc)
            .unwrap_or(0),
    })
}

/// The stride is named only by the tests that walk a plane table by hand.
#[cfg(test)]
pub(crate) use reims_vgpu_wire::device_desc::TYPE4_PLANE_STRIDE;
/// Backing descriptor geometry, from the wire crate's Tier-2 view of it.
///
/// The eight offsets these used to be are now `offset_of!` on
/// [`reims_vgpu_wire::device_desc::BackingHeader`] and
/// [`reims_vgpu_wire::device_desc::BackingPlaneRecord`], asserted there. Only the
/// four the device still computes with are re-exported.
pub(crate) use reims_vgpu_wire::device_desc::{
    backing_len_for, TYPE4_MIN_LEN, TYPE4_PLANES, TYPE4_PLANE_CAP,
};

/// CoreVideo / IOSurface biplanar 420 full-range (`'420f'`).
pub const IOSURFACE_FOURCC_420F: u32 = 0x3432_3066;
/// CoreVideo / IOSurface biplanar 420 video-range (`'420v'`).
pub const IOSURFACE_FOURCC_420V: u32 = 0x3432_3076;

/// One backing plane record (stride 0x10 @ +0x14): offset, w, h, bpr|bpe<<24.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackingPlane {
    pub offset: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    /// From packed high 8 bits (`getPlaneBytesPerElement`); 0 if wire left it 0.
    pub bytes_per_element: u8,
}

/// Decoded backing descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackingRecord {
    pub length: u64,
    pub backing_pfn: u32,
    /// Wire `pixelFormat@0xc` — OSType FourCC or small MTL ordinal.
    pub pixel_format: u32,
    pub plane_count: u8,
    pub planes: [BackingPlane; TYPE4_PLANE_CAP],
    /// Plane0 convenience (present / single-plane geom).
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
}

/// CoreVideo biplanar 8-bit 420 family — **not** a single `MTLPixelFormat`.
///
/// Metal binds planes via `newTextureWithDescriptor:iosurface:plane:` as R8 (Y)
/// and RG8 (UV). Product must not invent BGRA.
#[inline]
pub fn iosurface_fourcc_is_biplanar(pixel_format: u32) -> bool {
    matches!(pixel_format, IOSURFACE_FOURCC_420F | IOSURFACE_FOURCC_420V)
}

/// True when backing / mapping cannot be staged as one linear color texture.
#[inline]
pub fn backing_is_multiplanar(surf: &BackingRecord) -> bool {
    surf.plane_count > 1 || iosurface_fourcc_is_biplanar(surf.pixel_format)
}

/// Mapping has multi-plane device geometry (plane_count≥2) or biplanar FourCC.
pub fn mapping_is_multiplanar(m: &MappingEntry) -> bool {
    use crate::protocol::iosurface_pages::decode_device_surface;
    if let Some(s) = decode_device_surface(&m.device_desc) {
        if s.plane_count > 1 {
            return true;
        }
        if iosurface_fourcc_is_biplanar(s.pixel_format) {
            return true;
        }
    }
    false
}

/// The device descriptor's `pixelFormat` word as a Metal format.
///
/// That field carries **two** encodings and always has. On x86 this device
/// synthesizes the descriptor and [`synthesize_device_desc_from_backing`] writes
/// the MTL ordinal for a known single-plane surface and the raw OSType FourCC
/// otherwise. On arm64 the descriptor is the guest's own and the field holds
/// whatever `getPixelFormat()` returned, which is a FourCC for media surfaces.
///
/// The arm64 mapper used to read it as `raw as u16` — a silent narrowing, and
/// wrong by the rule [`iosurface_pixel_format_to_mtl`] states about this exact
/// operation: `'BGRA'` truncates to `0x5241`, which is not a Metal format, so
/// `bytes_per_pixel` refuses it, every sample window refuses, and every render
/// target on that mapping resolves to nothing. The x86 arm meanwhile read the
/// same conceptual field as a FourCC. Two consumers, one field, two encodings
/// assumed — and the truncation is the arm that loses guest work silently.
///
/// The two encodings are disjoint, and the test between them is not a
/// plausibility one. An MTLPixelFormat is an enum ordinal, and the descriptor's
/// own per-plane format fields are 16 bits wide, so an ordinal fits in 16 bits by
/// construction. An OSType is four character bytes, none of them zero, so it
/// cannot. A value that does not fit therefore *cannot* be an ordinal and goes
/// through the FourCC table; a value that does fit is the ordinal it is.
///
/// Unknown FourCCs and multi-plane OSTypes come back 0 — the same fail-closed
/// refusal the backing path latches, never an invented BGRA8.
pub fn device_desc_format_to_mtl(raw: u32) -> u16 {
    if raw <= u16::MAX as u32 {
        return raw as u16;
    }
    iosurface_pixel_format_to_mtl(raw)
}

/// Map IOSurface OSType FourCC (or MTL raw) to a **single-plane** MTL pixel format.
///
/// Live x86 backing carries IOSurface `pixelFormat` as a FourCC (e.g. `'BGRA'` =
/// `0x42475241`). Truncating to u16 yields `0x5241` which is not a Metal format.
///
/// Returns **0** when:
/// - format is multi-plane (e.g. `'420f'` / `'420v'`) — no single MTLPixelFormat
/// - format is unknown — fail closed; **do not** invent BGRA8
///
/// Unknown formats fail closed.
pub fn iosurface_pixel_format_to_mtl(pixel_format: u32) -> u16 {
    use crate::protocol::pixel_format::{
        MTL_FORMAT_BGR10A2_UNORM, MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R8_UNORM,
        MTL_FORMAT_RG8_UNORM, MTL_FORMAT_RGBA16_FLOAT, MTL_FORMAT_RGBA8_UNORM,
    };
    if pixel_format == 0 {
        return 0;
    }
    // Multi-plane OSTypes are not MTLPixelFormats (Metal plane: API).
    if iosurface_fourcc_is_biplanar(pixel_format) {
        return 0;
    }
    // No pass-through for small values. This used to return `pixel_format as
    // u16` for anything at or below 0x200, on the reading that such a value was
    // "already an MTLPixelFormat ordinal". That decided which *encoding* a field
    // was in from the field's magnitude, and the caller already knows: every
    // caller here passes a backing `pixelFormat` (+0x0c), which is an IOSurface
    // OSType — a four-character code, so never below 0x20202020. The mapper-ref-texture and
    // ref-texture rails carry their MTL ordinal in a `u16` field of their own and do
    // not route through this function.
    //
    // The magnitude test was also wrong at its own boundary: MTLPixelFormat
    // BGRA10_XR is 552 (0x228) and its three siblings are 553-555, so a 10-bit
    // XR surface passed 0x200 and fell into the FourCC match below regardless.
    match pixel_format {
        // 'BGRA' / 'ARGB' (kb: ARGB fourcc → BGRA8Unorm 0x50 for render targets)
        0x4247_5241 | 0x4152_4742 => MTL_FORMAT_BGRA8_UNORM,
        // 'RGBA'
        0x5247_4241 => MTL_FORMAT_RGBA8_UNORM,
        // 'RGhA' / half-float variants seen as AhGR in notes
        0x5247_6841 | 0x4168_4752 => MTL_FORMAT_RGBA16_FLOAT,
        // 'l10r' — `kCVPixelFormatType_ARGB2101010LEPacked`. One single-plane
        // 32-bit little-endian word per texel: two bits of alpha in the high
        // bits, then ten each of red, green and blue, with blue in the low
        // bits. That is `MTLPixelFormatBGR10A2Unorm`'s word exactly, which is
        // also `VK_FORMAT_A2R10G10B10_UNORM_PACK32`'s — see
        // `protocol::pixel_format::MTL_FORMAT_BGR10A2_UNORM` and
        // `backend::vulkan::translate::pixel`, which already paired those two.
        //
        // Two independent sources agree on it, which is why it is stated rather
        // than proposed. A driven macos-13 x86/Vulkan boot running Asphalt 8
        // declares its 1280x720 render surface with this FourCC and then binds a
        // **ref-texture view** over the same allocation whose own
        // `pixel_format` field reads `0x5e` — the guest naming `BGR10A2Unorm`
        // itself, reported as `rt_ref_texture_view_differs sid=117 view=1280x720
        // fmt=0x5e ... base fmt=0x0`. The surface geometry closes it: the backing
        // record's `bytes_per_row` is 5120 over a width of 1280, which is four
        // bytes a texel, and its `length` 0x384000 is that stride times 720.
        //
        // Before this arm the surface reached `draw::render_target`'s
        // `rt_backing_base_format` as a zero — that resolve's typed refusal for a
        // multi-plane or unknown-FourCC surface — so **every** draw of the frame
        // failed and the game's window was black. One 100 s capture of it held
        // 20 822 `draw_fail_clear_fallback` records and 0 successful draws.
        0x6c31_3072 => MTL_FORMAT_BGR10A2_UNORM,
        // Single-plane R8 / RG8 OSTypes used as plane textures (not biplanar media fourcc).
        // 'L008' / common R8 fourccs are rare on backing; MTL ordinals already handled above.
        // 'R8  ' / 'RG08' if ever seen as OSType:
        0x5238_2020 => MTL_FORMAT_R8_UNORM,
        0x5247_3038 => MTL_FORMAT_RG8_UNORM,
        // Unknown FourCC: 0 — callers fail closed (no BGRA invent).
        _ => 0,
    }
}

/// Decode one backing plane record.
fn decode_backing_plane(desc: &[u8], plane_index: usize) -> Option<BackingPlane> {
    let r = reims_vgpu_wire::device_desc::backing_plane(desc, plane_index).ok()?;
    Some(BackingPlane {
        offset: r.offset.get(),
        width: r.width.get(),
        height: r.height.get(),
        bytes_per_row: r.bytes_per_row(),
        bytes_per_element: r.bytes_per_element(),
    })
}

/// The bytes of a backing record descriptor that [`decode_backing`] does
/// **not** read: `+0x11..0x14` and everything past the plane records it
/// consumed (`TYPE4_PLANES + plane_count * TYPE4_PLANE_STRIDE ..`).
///
/// Decoded today: `length` (+0x00), `backing_pfn` (+0x08), `pixel_format`
/// (+0x0c), `plane_count` (+0x10), and each plane's offset/width/height/packed
/// bpr. That is everything we know about a surface when the guest creates it —
/// and it is not enough to tell a desktop swapchain buffer from a same-geometry
/// offscreen render target, because a WebKit content tile is also 1920x1080
/// 'BGRA'. Membership is therefore reconstructed downstream by compositor-output
/// edges, full-frame-publish detection, output groups, presented-ness, and the
/// a/b seed.
///
/// **Measured: the guest is not telling us here.** Across one 1766 s x86/Vulkan
/// session with a real GUI login (boot `20260728-163046`), the probe below
/// emitted exactly two shapes for ≥5983 decodes over 453 distinct surface ids
/// and 154 distinct geometries — desktop swapchain buffers and never-displayed
/// content tiles alike:
///
/// ```text
/// backing_desc_shape distinct=1 1920x1080 fmt=0x42475241 planes=1 len=36 undecoded_len=3
/// undecoded_nz=0 backing_desc_shape distinct=2 320x320 fmt=0x34323066 planes=2 len=52
/// undecoded_len=3 undecoded_nz=0
/// ```
///
/// `len` is `TYPE4_PLANES + plane_count * TYPE4_PLANE_STRIDE` exactly, and it is
/// the *guest's* number — [`read_descriptor`] honours `descriptor_length` with no
/// clamp. The record ends where the plane array ends; the only bytes we skip are
/// the three at `+0x11`, and they were zero every time. There is nowhere in this
/// descriptor for a usage, bind, scanout or role hint to be, so no rule over
/// surface identity can classify a brand-new buffer before its first draw.
///
/// Narrow: this is the backing record on the x86 PCI pathway. It says nothing
/// about mapper-ref-texture (`decode_iosurface_texture_descriptor`, which does not run
/// here and whose 0x38/0x58 blobs are still read only to 0x20), and a
/// create-time record we never read at all would be invisible to it.
///
/// A `plane_count` above [`TYPE4_PLANE_CAP`] is clamped by the decoder, so the
/// records past the clamp fall into this span too — which is correct: they are
/// bytes we did not read.
///
/// Public so the probe's notion of "undecoded" is pinned by a test rather than
/// restated in a log format string.
pub fn undecoded_backing_bytes(desc: &[u8]) -> Vec<u8> {
    if desc.len() < TYPE4_MIN_LEN {
        return Vec::new();
    }
    let Ok(h) = reims_vgpu_wire::device_desc::backing_header(desc) else {
        return Vec::new();
    };
    let plane_count = (h.plane_count as usize).min(TYPE4_PLANE_CAP);
    let planes_end = backing_len_for(plane_count);
    let mut out = Vec::new();
    // The header's undecoded interior, named by the field rather than by the
    // literal it used to be: a field added before it moves this with it.
    let reserved = core::mem::offset_of!(reims_vgpu_wire::device_desc::BackingHeader, reserved);
    out.extend_from_slice(&desc[reserved..TYPE4_PLANES]);
    if planes_end < desc.len() {
        out.extend_from_slice(&desc[planes_end..]);
    }
    out
}

/// One always-on line per distinct `(len, undecoded span)`, capped.
///
/// Keyed on the **content** of the undecoded bytes, never on the record length.
/// The `display_txn_payload` probe keyed its budget on `(opcode, payload_len)`,
/// the length never varied, and it exhausted itself inside the first 400 ms —
/// it answered one question and then went blind for the rest of the session. A
/// new *value* is the interesting event here, so that is the key.
///
/// Runs before the decoder's own validity checks, so a record that fails to
/// decode still reports. An earlier version of this probe on the mapper-ref-texture
/// descriptor sat after its length check and emitted nothing at all on a live
/// boot; "the decoder never ran" and "the tail is constant" produced the same
/// silence, which is the reading the probe exists to rule out.
///
/// Hitting the cap is reported once. A silent truncation would read like "we
/// saw everything", which is the same class of error as a probe reporting a
/// confident constant.
fn note_backing_shape(desc: &[u8]) {
    const MAX_SHAPES: usize = 24;
    const HEX_MAX: usize = 128;
    use std::sync::Mutex;
    type ShapeKey = (usize, Vec<u8>);
    static SEEN: Mutex<Option<std::collections::BTreeSet<ShapeKey>>> = Mutex::new(None);

    let undecoded = undecoded_backing_bytes(desc);
    let (fresh, distinct) = {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        let seen = guard.get_or_insert_with(Default::default);
        if seen.len() > MAX_SHAPES {
            return;
        }
        (seen.insert((desc.len(), undecoded.clone())), seen.len())
    };
    if !fresh {
        return;
    }
    if distinct > MAX_SHAPES {
        crate::observe::fail(format!(
            "backing_desc_shape outcome=cap_reached distinct={distinct} \
             (the undecoded span varies per surface; it is not a constant tail)"
        ));
        return;
    }
    // Plane 0's geometry through the plane decoder, not through `TYPE4_PLANES
    // + 4` and `+ 8`. Those two literals were the plane record's interior
    // restated at a second site, so the stride was declared once and its
    // contents twice.
    let (w, h, fmt, pc) = match (
        reims_vgpu_wire::device_desc::backing_header(desc),
        decode_backing_plane(desc, 0),
    ) {
        (Ok(h), Some(p0)) => (p0.width, p0.height, h.pixel_format.get(), h.plane_count),
        _ => (0, 0, 0, 0),
    };
    let hex: String = desc
        .iter()
        .take(HEX_MAX)
        .map(|b| format!("{b:02x}"))
        .collect();
    crate::observe::fail(format!(
        "backing_desc_shape distinct={distinct} {w}x{h} fmt={fmt:#x} planes={pc} len={} \
         undecoded_len={} undecoded_nz={} hex={hex}{}",
        desc.len(),
        undecoded.len(),
        undecoded.iter().filter(|&&b| b != 0).count(),
        if desc.len() > HEX_MAX { "…" } else { "" },
    ));
}

/// Report, once per reason, that the backing decoder dropped something the guest
/// declared.
///
/// Deduped rather than sampled: each reason names a distinct shape of blob, and
/// a surface stream re-decodes the same descriptor thousands of times a boot, so
/// an undeduped line would flood while adding nothing. The first occurrence is
/// what a reader needs — after it, `backing_desc_shape` carries the geometry.
fn backing_decode_drop_latch() -> &'static std::sync::Mutex<std::collections::HashSet<&'static str>>
{
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<&'static str>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

fn note_backing_decode_drop(reason: &'static str, detail: String) {
    // The latch is flood protection, so the magnitude has to live somewhere it
    // survives: without this, a second malformed descriptor and a thousand of
    // them read identically — one line, and nothing to ask.
    crate::runtime::drain::census::note_store_route("backing_desc_refused");
    let fresh = {
        let mut guard = backing_decode_drop_latch()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        guard.insert(reason)
    };
    if fresh {
        crate::observe::fail(detail);
    }
}

/// Forget which reasons have been reported, so a test observes the first
/// occurrence rather than whatever an earlier test in the same process left
/// behind.
#[cfg(test)]
fn reset_backing_decode_drops() {
    backing_decode_drop_latch()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clear();
}

/// Decode a backing record descriptor blob.
pub fn decode_backing(desc: &[u8]) -> Option<BackingRecord> {
    note_backing_shape(desc);
    if desc.len() < TYPE4_MIN_LEN {
        return None;
    }
    let h = reims_vgpu_wire::device_desc::backing_header(desc).ok()?;
    let length = h.length.get();
    let backing_pfn = h.backing_pfn.get();
    let pixel_format = h.pixel_format.get();
    let plane_count_raw = h.plane_count;
    if backing_pfn == 0 || length == 0 {
        return None;
    }
    // Both of the checks below refuse the descriptor rather than publish a
    // partial one, and they refuse for the same reason the two above them do:
    // what came back would not be the surface the guest described.
    //
    // The alternative was tried and is what these replace. Truncating to the cap
    // published a surface that simply *has* eight planes, and defaulting an
    // unreachable plane record published a 0x0 plane at pitch 0 in a slot the
    // guest declared. Neither is a decode failure downstream — every later
    // reader sees a well-formed surface — so the loss surfaces as a layer that
    // samples blank, which is indistinguishable from content that is genuinely
    // empty. `None` reaches the caller's existing
    // `backing_fail reason=desc_decode`, so the refusal is attributable to
    // the surface id that could not be decoded.
    if plane_count_raw as usize > TYPE4_PLANE_CAP {
        // The bound itself is right — IOSurface's own `getPlaneCount` caps at
        // eight — so a descriptor declaring more is malformed rather than
        // merely large, and there is no correct prefix of it to keep.
        note_backing_decode_drop(
            "plane_count_over_cap",
            format!(
                "backing_decode_drop reason=plane_count_over_cap declared={plane_count_raw} \
                 cap={TYPE4_PLANE_CAP} fmt={pixel_format:#x}"
            ),
        );
        return None;
    }
    let plane_count = plane_count_raw;
    let mut planes = [BackingPlane::default(); TYPE4_PLANE_CAP];
    for (i, plane) in planes.iter_mut().enumerate().take(plane_count as usize) {
        let Some(p) = decode_backing_plane(desc, i) else {
            note_backing_decode_drop(
                "plane_record_short",
                format!(
                    "backing_decode_drop reason=plane_record_short plane={i} \
                     planes={plane_count} desc_len={} fmt={pixel_format:#x}",
                    desc.len()
                ),
            );
            return None;
        };
        *plane = p;
    }
    let (width, height, bpr) = if plane_count > 0 {
        let p0 = planes[0];
        (p0.width, p0.height, p0.bytes_per_row)
    } else {
        (0, 0, 0)
    };
    Some(BackingRecord {
        length,
        backing_pfn,
        pixel_format,
        plane_count,
        planes,
        width,
        height,
        bytes_per_row: bpr,
    })
}

/// Build `sIOSurfaceDeviceDescriptor` geometry from backing wire (no invent).
///
/// Multi-plane: plane records from backing planes; sample path selects by
/// geometry. Single-plane: surface-level fields only
/// (`plane_count==0` path in `sample_window_from_device_desc`).
fn synthesize_device_desc_from_backing(surf: &BackingRecord) -> Vec<u8> {
    let mut device_desc = vec![0u8; DEVICE_DESC_LEN];
    let multi = backing_is_multiplanar(surf);
    let mtl = iosurface_pixel_format_to_mtl(surf.pixel_format);
    // Device desc pixelFormat field: guest stores getPixelFormat() (FourCC for
    // biplanar media). Single-plane product sample uses MTL ordinal when known.
    let fmt_word = if multi {
        surf.pixel_format
    } else if mtl != 0 {
        mtl as u32
    } else {
        surf.pixel_format
    };
    st32(&mut device_desc[DEVICE_DESC_PIXEL_FORMAT..], fmt_word);
    // `allocSize` is a u32 field in the device descriptor and `length` is u64 on
    // the wire, so a surface above 4 GiB cannot be published faithfully. Saying
    // `u32::MAX` is the least wrong answer available — it is the largest size the
    // field can hold, so a reader sizing a mapping from it under-reads rather
    // than walking past the end — but it is still a size the guest did not ask
    // for, and it must not be published as though it were.
    let alloc = if surf.length > u32::MAX as u64 {
        note_backing_decode_drop(
            "alloc_size_over_u32",
            format!(
                "backing_decode_drop reason=alloc_size_over_u32 length={} \
                 published={} (device-descriptor allocSize is 32-bit)",
                surf.length,
                u32::MAX
            ),
        );
        u32::MAX
    } else {
        surf.length as u32
    };
    st32(&mut device_desc[DEVICE_DESC_ALLOC_SIZE..], alloc);
    // Surface-level dims/bpr from plane0 (same as backing plane0 convenience).
    let dims = ((surf.width as u64) << 8) | ((surf.height as u64) << 40);
    st64(&mut device_desc[DEVICE_DESC_DIMS..], dims);
    if surf.bytes_per_row > 0 {
        st32(&mut device_desc[DEVICE_DESC_BPR..], surf.bytes_per_row);
    }
    if multi && surf.plane_count > 0 {
        // Multi-plane: publish plane records; sample_window_from_device_desc
        // matches mapper-ref-texture R8/RG8 binds by (w,h,bpe), and declines when two
        // planes share all three. Do not invent bases from format alone.
        let n = (surf.plane_count as usize).min(TYPE4_PLANE_CAP);
        device_desc[DEVICE_DESC_PLANE_COUNT] = n as u8;
        // Surface-level bpe: plane0 element size when wire provides it.
        let bpe0 = surf.planes[0].bytes_per_element;
        if bpe0 != 0 {
            st16(&mut device_desc[DEVICE_DESC_BPE..], bpe0 as u16);
        }
        for i in 0..n {
            let p = &surf.planes[i];
            let base = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
            st32(&mut device_desc[base + DEVICE_PLANE_OFFSET..], p.offset);
            // plane_size: 0 = skip size check in sample_window_from_device_plane
            // (backing wire has offset/w/h/bpr, not a separate size field).
            st32(&mut device_desc[base + DEVICE_PLANE_SIZE..], 0);
            let pdims = ((p.width as u64) << 8) | ((p.height as u64) << 40);
            st64(&mut device_desc[base + DEVICE_PLANE_DIMS..], pdims);
            st32(&mut device_desc[base + DEVICE_PLANE_BPR..], p.bytes_per_row);
            if p.bytes_per_element != 0 {
                st16(
                    &mut device_desc[base + DEVICE_PLANE_BPE..],
                    p.bytes_per_element as u16,
                );
            } else if iosurface_fourcc_is_biplanar(surf.pixel_format) {
                // Contract: 420 Y bpe=1, UV bpe=2 when wire high-byte is 0.
                // Only fill when FourCC is known biplanar — not a free invent for
                // arbitrary multi-plane. Matches Metal R8/RG8 plane bind bpp.
                let bpe = if i == 0 { 1u16 } else { 2u16 };
                st16(&mut device_desc[base + DEVICE_PLANE_BPE..], bpe);
            }
        }
    } else {
        // Single-plane surface-level sample path (plane_count 0).
        device_desc[DEVICE_DESC_PLANE_COUNT] = 0;
        // Plane 0's offset, which this arm used to decode and then drop.
        //
        // `decode_backing_plane` reads four fields per plane; the surface-level
        // convenience copies took three of them (width, height, bytes-per-row)
        // and left the offset behind, so a single-plane surface whose pixels
        // start past the base of its allocation was read and written at 0. The
        // multi-plane arm above publishes every plane's offset, and the
        // consumers are symmetric: `sample_window_from_device_surface` returns
        // `base_offset` as the window offset and folds it into `span_end`, which
        // is exactly what `sample_window_from_device_plane` does with a plane's.
        // On the arm64 mapper path this same field is read straight out of the
        // guest's descriptor rather than synthesized, so dropping it here also
        // made the two pathways describe one surface differently.
        //
        // Zero is the ordinary value and stays silent; a non-zero one is the
        // population that was being misread, and `backing_base_offset_nonzero`
        // counts how large that is. Read on a driven x86/Vulkan boot: **0**, so
        // no single-plane surface on that workload starts past its base and this
        // is contract fidelity rather than a live repair. It is also why the
        // change was safe to make without a rate: every window it could move is
        // one the counter would have named.
        let base_offset = surf.planes[0].offset;
        if base_offset != 0 {
            crate::runtime::drain::note_store_route("backing_base_offset_nonzero");
            st32(&mut device_desc[DEVICE_DESC_BASE_OFFSET..], base_offset);
        }
        if mtl != 0 {
            if let Some(bpp) = crate::protocol::pixel_format::bytes_per_pixel(mtl) {
                st16(&mut device_desc[DEVICE_DESC_BPE..], bpp as u16);
            }
        }
    }
    device_desc
}

/// Name the object-list geometry behind a refused entry read.
///
/// `gva_mem::read_task_gva_by_id` already reports the refusal and which of the
/// walk's checks produced it, but it is generic over every caller and can only
/// name the address. For this caller the address is *derived* — `(pfn <<
/// page_shift) + ref * entry_len` — and the address alone cannot say which of
/// its three inputs is the surprising one.
///
/// That distinction was the question, and this line answered it. A driven x86
/// boot emits `gva_zero_pfn` from this site for tasks 2..=9, and the geometry
/// behind every one is `list_pfn=1 list_count=1048576` — the guest's own
/// `SetObjectList` values, taken straight off the wire by
/// `drain::apply_set_object_list`, which invents nothing. So the guest reserves
/// a million-slot object list at task-virtual page 1 and maps its pages lazily,
/// and a lookup that lands on an unpopulated page is the resolver saying "the
/// guest has not said yet". That is expected control flow, `None` is the right
/// answer, and no guest work is lost.
///
/// Worth knowing because the address alone reads like the opposite. `pfn = 1`
/// with a million slots is precisely the pair `TaskEntry::define` used to
/// *fabricate* for tasks with no list — `model::state` documents removing it,
/// after it caused this device to walk a neighbouring task's page table and
/// decode its entry as this task's. Seeing those two numbers again invites the
/// conclusion that the fabrication is back. It is not: they are what the guest
/// publishes, which is presumably where the fabricated defaults were copied from.
///
/// Off-channel and latched per task: expected control flow stays quiet, and the
/// measurement above is what establishes it is expected. One line per task per
/// boot (eight on that run), alongside the one the walker already emits per
/// (reason, task, site).
fn note_list_entry_unreadable(
    task_id: u32,
    ref_: u32,
    task: &crate::model::TaskEntry,
    entry_gva: u64,
) {
    if !crate::observe::first_sight("object_list_entry_unreadable", u64::from(task_id)) {
        return;
    }
    crate::observe::off(list_entry_unreadable_detail(task_id, ref_, task, entry_gva));
}

/// The line [`note_list_entry_unreadable`] emits, built separately so a test can
/// assert the geometry reaches it without going through the always-on sink.
fn list_entry_unreadable_detail(
    task_id: u32,
    ref_: u32,
    task: &crate::model::TaskEntry,
    entry_gva: u64,
) -> String {
    format!(
        "object_list_entry_unreadable task={task_id} ref={ref_} gva={entry_gva:#x} \
         list_pfn={} list_count={} entry_len={OBJECT_LIST_ENTRY_LEN} \
         (the walker's own refusal names the check; this names the three inputs \
          the address was built from, which it cannot see)",
        task.object_list_pfn, task.object_list_count
    )
}

/// What a missing object-list entry means to the caller that asked.
///
/// The read underneath is identical; only whether a miss is reportable differs,
/// and only the caller knows. A driven boot measured 18 `gva_read_refused
/// reason=gva_zero_pfn` lines on the fail channel — two per task for tasks 2
/// through 10 — and every one of them was [`Self::Probe`] working exactly as
/// designed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListLookup {
    /// The guest's own command named this ref against this task, so an entry
    /// the device cannot read is guest work it cannot execute. Reportable.
    Named,
    /// This device is asking *whether* the task owns the ref. Every task that
    /// does not own it misses, which is how the search finds the one that does —
    /// so a miss here is the answer, not a failure, and reporting it would put a
    /// line on the fail channel for each task the search correctly stepped over.
    Probe,
}

/// Which of the eight checks in an object-list lookup came back empty.
///
/// They used to be eight `None`s behind one `reason=no_list_entry`, and only the
/// unreadable arm said anything at all. That is why a macos-26 boot losing ~40
/// draws to this refusal could not say which of two opposite things had
/// happened: the device cleared the task's list under the guest
/// ([`Self::NoObjectList`] — `define_task` resets `object_list_pfn`/`count` and
/// macOS 26 re-issues `define_task` for a live tid with a new page-table root,
/// which macOS 13 never does), or the guest had simply not published the object
/// into a list this device read perfectly well ([`Self::SlotEmpty`]). The first
/// is a device defect and the second is a wait; they share a reason string and
/// nothing else.
///
/// The routes are counted only for a ref the guest **named** — see
/// [`ListLookup`], whose `Probe` arm misses by design on every task that does
/// not own the ref.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ListMiss {
    /// No task under this id at all.
    NoTask,
    /// The task exists and has been torn down or not activated.
    TaskInactive,
    /// The task has no object list registered. Either none was ever set, or one
    /// was set and a later `define_task` for the same id reset it.
    NoObjectList,
    /// The ref is past the count the guest declared for the list.
    RefBeyondList,
    /// `pfn << page_shift` plus the slot offset does not fit a `u64`.
    AddressOverflow,
    /// The slot's guest address did not read, carrying **which** of the walk's
    /// checks refused.
    ///
    /// The payload is the same distinction [`crate::runtime::host::MemError`]
    /// draws for every other guest read, and the reason it is here is
    /// [`slot_recheck`]: a slot that read and decoded cleanly at miss time
    /// cannot later be genuinely unmapped, rooted at a zero PFN and outside the
    /// address space all at once, so on a re-read the check that refused *is*
    /// the finding. It was the whole of a driven macos-26 boot's terminal
    /// verdicts while it was still one bare value.
    ///
    /// `object_list_entry_unreadable` stays alongside it, because that line
    /// names the three inputs the address was built from, which the walk's own
    /// refusal cannot see.
    Unreadable(crate::runtime::host::MemError),
    /// The sixteen bytes read and are not an object-list entry.
    Undecodable,
    /// The slot read and is zero. Not a device failure.
    ///
    /// Object refs are reusable indices: deletion clears the indexed slot and
    /// allocation may assign that same index to a later object. The packet does
    /// not carry the index generation, so neither an earlier entry nor a later
    /// one can answer for a zero slot. The lookup must miss until the current
    /// tenant is present in the list.
    SlotEmpty,
}

impl ListMiss {
    /// Every variant, for the distinctness check below.
    ///
    /// [`Self::route`]'s match is exhaustive, so a new variant cannot skip
    /// having a route — but it can skip this list. Add it here too, or the
    /// distinctness test stops covering the population it names.
    #[cfg(test)]
    const ALL: [Self; 8] = [
        Self::NoTask,
        Self::TaskInactive,
        Self::NoObjectList,
        Self::RefBeyondList,
        Self::AddressOverflow,
        Self::Unreadable(crate::runtime::host::MemError::Unmapped),
        Self::Undecodable,
        Self::SlotEmpty,
    ];

    pub fn route(self) -> &'static str {
        match self {
            Self::NoTask => "list_miss_no_task",
            Self::TaskInactive => "list_miss_task_inactive",
            Self::NoObjectList => "list_miss_no_object_list",
            Self::RefBeyondList => "list_miss_ref_beyond_list",
            Self::AddressOverflow => "list_miss_address_overflow",
            Self::Unreadable(_) => "list_miss_unreadable",
            Self::Undecodable => "list_miss_undecodable",
            Self::SlotEmpty => "list_miss_slot_empty",
        }
    }

    /// The same eight checks, seen a tranche later by
    /// [`slot_recheck`](self::slot_recheck).
    ///
    /// A second table rather than a prefix swap on [`Self::route`], because both
    /// are matched exhaustively and a new variant therefore cannot be added
    /// without giving it a name on both cadences. It sits here rather than in
    /// `slot_recheck` for the same reason the first one does: the two spellings
    /// of one check belong next to each other, where a divergence is visible.
    ///
    /// The recheck's first version collapsed four of these into one
    /// `slot_recheck_unreadable`, and the first driven boot put **20** readings
    /// in it — the whole of that boot's terminal verdicts, and unreadable in
    /// exactly the sense that it could not say which check refused. That is the
    /// failure [`ListMiss`] itself exists to have fixed once.
    pub fn recheck_route(self) -> &'static str {
        match self {
            Self::NoTask => "slot_recheck_no_task",
            Self::TaskInactive => "slot_recheck_task_inactive",
            Self::NoObjectList => "slot_recheck_no_object_list",
            Self::RefBeyondList => "slot_recheck_ref_beyond_list",
            Self::AddressOverflow => "slot_recheck_address_overflow",
            Self::Unreadable(_) => "slot_recheck_unreadable",
            Self::Undecodable => "slot_recheck_undecodable",
            // Not terminal: the watch survives to be asked again. Named anyway
            // so the table is total and the residue has a spelling if a caller
            // ever wants to report it.
            Self::SlotEmpty => "slot_recheck_still_empty",
        }
    }
}

/// Lookup one object-list slot for `task_id` / `ref_`, reporting a miss.
///
/// For a ref the guest named. A speculative caller wants [`probe_list_entry`];
/// see [`ListLookup`] for why the distinction is the caller's to make.
pub fn lookup_list_entry<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<ListObjectEntry> {
    list_entry(state, host, task_id, ref_, ListLookup::Named)
}

/// [`lookup_list_entry`] for a caller asking whether this task owns `ref_`.
///
/// Quiet on a miss. The object list really is where the guest said — a driven
/// boot reads `set_object_list … pfn=0x1 count=1048576 plen=12` for **every**
/// task including the one that then resolves 11 768 surfaces — so a task
/// missing at this slot is a task without that object, which is what the search
/// is asking.
pub fn probe_list_entry<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<ListObjectEntry> {
    list_entry(state, host, task_id, ref_, ListLookup::Probe)
}

fn list_entry<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
    lookup: ListLookup,
) -> Option<ListObjectEntry> {
    let found = list_entry_or_miss(state, host, task_id, ref_, lookup);
    match found {
        Ok(entry) => {
            // Only a ref the guest named: a probe's success is the search
            // finding an owner, which says nothing about what this task's own
            // list once held.
            if lookup == ListLookup::Named {
                slot_recheck::note_ref_resolved(task_id, ref_);
                // The control for the banding below, and the reason it is worth
                // reading: a miss skewing late says nothing unless the hits do
                // not. See `census::note_list_lookup_age`.
                crate::runtime::drain::note_list_lookup_age(
                    true,
                    crate::runtime::drain::tranche_elapsed_us(),
                );
            }
            Some(entry)
        }
        Err(miss) => {
            // Only for a ref the guest named. A probe misses on every task that
            // does not own the ref, which is how it finds the one that does —
            // counting those would bury the named misses under the search.
            if lookup == ListLookup::Named {
                crate::runtime::drain::note_store_route(miss.route());
                // How late in its tranche this lookup happened. The guest clears
                // a slot by writing its own memory, so a slot found cleared
                // should be one read late — if these band like the hits do, that
                // story is wrong however good the totals look.
                crate::runtime::drain::note_list_lookup_age(
                    false,
                    crate::runtime::drain::tranche_elapsed_us(),
                );
                if miss == ListMiss::SlotEmpty {
                    // Both instruments measure the slot, not the packet's
                    // outcome, so they run before this miss is returned.
                    note_slot_empty_claimants(state, host, task_id, ref_);
                    // The unconfounded half of the same question — see
                    // `slot_recheck` for why the claimant search above cannot
                    // settle it and this can.
                    slot_recheck::note_slot_empty(state, host, task_id, ref_);
                }
            }
            None
        }
    }
}

/// For a named ref whose own task's slot is empty: how many *other* live tasks
/// hold a real object at that slot, against how many there are?
///
/// [`ListMiss::SlotEmpty`] is the only miss a macos-26 boot produces and the
/// whole of that rail's lost draws. "Does anyone else have it" looked like the
/// question, and a first boot answered *every* miss with yes — which is when the
/// confound became obvious. **Every task registers its object list at the same
/// `pfn = 1`**, and refs are small and dense, so "another task has something at
/// slot 3" is close to a tautology on a busy guest and says nothing about
/// ownership.
///
/// So the reading is the *fraction*, and it is emitted banded against the live
/// task count rather than as a yes/no:
///
/// - **nowhere** — nobody has published it. The guest named a ref before writing
///   the slot, and the answer is for the packet to wait rather than for the draw
///   to be dropped.
/// - **one** — exactly one other task has it. That is a real ownership signal:
///   the object exists in a list this device did not look in.
/// - **many / all** — the slot index is simply populated across the guest's
///   tasks, and this search cannot tell ownership from coincidence. Anything
///   built on it would be built on the confound.
///
/// Costs one probe read per live task, on a miss only.
fn note_slot_empty_claimants<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
) {
    let live = state.tasks.live_count();
    let claimants: Vec<(u32, ListObjectEntry)> = state
        .tasks
        .live_ids()
        .filter(|&other| other != task_id)
        .filter_map(|other| probe_list_entry(state, host, other, ref_).map(|e| (other, e)))
        .collect();
    crate::runtime::drain::note_store_route(slot_empty_claim_route(claimants.len(), live));
    // The band was built when these lists were believed dense, where naming the
    // claimants would have been naming most of the guest. They are not: a driven
    // boot reads 4 to 18 occupied slots in a 341-entry first page, so a claim is
    // ~2 % likely by coincidence and *which* tasks claim is now a reading rather
    // than noise. Latched per `(task, ref)` — the band above is per miss and this
    // is per slot, which is also why the two counts differ.
    if !crate::observe::first_sight(
        "slot_empty_claimants",
        (u64::from(task_id) << 32) | u64::from(ref_),
    ) {
        return;
    }
    // Occupancy first, because it disqualifies most claims outright: a task
    // holding 316 of 341 slots claims every ref there is. A driven boot found
    // task 1 doing exactly that.
    //
    // Occupancy alone does not qualify the survivors, though. A sparse list
    // grows from index 0, so low refs are more likely occupied in *any* task,
    // and the missing refs here are 3, 4, 9, 10, 14 — the low end. `type=` is
    // the reading that position cannot fake: if the claimant's slot holds an
    // object of a kind the guest's command could not have meant, the claim is
    // coincidence however sparse the claimant is.
    let detail: Vec<String> = claimants
        .iter()
        .map(|&(other, entry)| {
            let held = slot_recheck::first_page_population(state, host, other)
                .map_or(-1i64, |p| p.populated as i64);
            format!("{other}:holds={held}:type={}", entry.object_type)
        })
        .collect();
    crate::observe::off(format!(
        "slot_empty_claimants task={task_id} ref={ref_} live={live} claimants=[{}] \
         (other live tasks holding a real object at this ref, each with how many objects \
          it holds in all and the object type it has here)",
        detail.join(" ")
    ));
}

/// Band a claimant count against the live task count.
///
/// Split out from the walk so the banding is testable without a guest: the walk
/// is a page-table read per task and the band is the only part that can be
/// wrong in a way that changes what the next session believes.
fn slot_empty_claim_route(claimants: usize, live_tasks: usize) -> &'static str {
    // `live_tasks` counts the asking task too, so the most that can claim is
    // one fewer. Comparing against that rather than against `live_tasks` is what
    // makes "all" mean all of them.
    let others = live_tasks.saturating_sub(1);
    match claimants {
        0 => "list_miss_slot_empty_claimed_nowhere",
        1 => "list_miss_slot_empty_claimed_by_one",
        n if others > 0 && n >= others => "list_miss_slot_empty_claimed_by_all",
        _ => "list_miss_slot_empty_claimed_by_many",
    }
}

/// The lookup proper, naming the check that refused.
///
/// Split from [`list_entry`] so the eight ways a slot comes back empty are eight
/// values rather than eight `None`s. See [`ListMiss`] for why that mattered.
fn list_entry_or_miss<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
    lookup: ListLookup,
) -> Result<ListObjectEntry, ListMiss> {
    let task = state.tasks.get(task_id).ok_or(ListMiss::NoTask)?;
    if !task.active {
        return Err(ListMiss::TaskInactive);
    }
    if task.object_list_count == 0 {
        return Err(ListMiss::NoObjectList);
    }
    let off =
        list_object_entry_offset(ref_, task.object_list_count).ok_or(ListMiss::RefBeyondList)?;
    let entry_gva = ((task.object_list_pfn as u64) << state.page_shift)
        .checked_add(off)
        .ok_or(ListMiss::AddressOverflow)?;
    let mut raw = [0u8; OBJECT_LIST_ENTRY_LEN];
    let read = match lookup {
        ListLookup::Named => gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            entry_gva,
            &mut raw,
            state.page_shift,
        ),
        ListLookup::Probe => gva_mem::try_read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            entry_gva,
            &mut raw,
            state.page_shift,
        ),
    };
    if let Err(why) = read {
        if lookup == ListLookup::Named {
            note_list_entry_unreadable(task_id, ref_, task, entry_gva);
        }
        return Err(ListMiss::Unreadable(why));
    }
    let e = decode_list_object_entry(&raw).map_err(|_| ListMiss::Undecodable)?;
    if e.descriptor_length == 0 || e.descriptor_gva == 0 {
        return Err(ListMiss::SlotEmpty);
    }
    Ok(e)
}

/// Read the descriptor blob for a list entry.
pub fn read_descriptor<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    entry: &ListObjectEntry,
) -> Option<Vec<u8>> {
    // Guest descriptor_length is authoritative — no product 4 KiB read clamp.
    let len =
        crate::runtime::draw::host_alloc_len(entry.descriptor_length as u64).filter(|&n| n > 0)?;
    let mut buf = vec![0u8; len];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        entry.descriptor_gva,
        &mut buf,
        state.page_shift,
    )
    .ok()?;
    Some(buf)
}

/// Which rung of the object-list ladder refused, as a value rather than a
/// spelling.
///
/// [`crate::observe::ladder_slug`] made the four rungs share one *vocabulary*;
/// this makes the first three share one *implementation*. A rail that matches on
/// this cannot skip the type check, cannot ask the three questions out of order,
/// and cannot invent a fourth condition between them — all three of which a
/// hand-written ladder can do silently.
///
/// The decode rung is deliberately absent. Its decoder differs per object type
/// and returns that decoder's own error, so folding it in here would mean either
/// one resolver per type or a decoder trait, and neither buys anything the call
/// site does not already have.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LadderRung {
    /// The guest has put nothing under this ref. Ordinary while a task's list is
    /// still being populated, which is why several rails answer it quietly.
    NoListEntry,
    /// Something is under the ref and it is not a type this caller accepts.
    ///
    /// Carries the tag it found, because every rail that reports this rung was
    /// formatting `ot={}` by hand from the entry it no longer has by then.
    WrongType { got: u8 },
    /// The entry names descriptor bytes that could not be read — the ref is
    /// live, and its descriptor GVA is not mapped right now.
    ///
    /// Carries the length the entry declared, for the same reason
    /// [`Self::WrongType`] carries the tag: it is the datum that explains this
    /// rung — "the entry said this many bytes at that GVA and they could not be
    /// read" — and the entry it came from is gone by the time a caller reports.
    DescRead { declared_len: u32 },
    /// The task's address space does not exist, so the reference is a slot in
    /// no namespace.
    ///
    /// **The rung below [`Self::NoListEntry`]**, and it is a different
    /// statement: an empty slot is a live task that has published nothing
    /// there, and this is a reference into a task that was never defined or has
    /// been deleted. The two used to be one answer because this device created a
    /// task's namespace on demand; `reims_vgpu_core::lifecycle::Lifecycle`
    /// refuses instead, and a name minted into a space with no page directory is
    /// a name whose every address resolves through nothing.
    ///
    /// Unreachable through the two callers that can produce it — both ask
    /// `lookup_list_entry` first, which requires `DeviceState::tasks` to hold an
    /// active task, and that table and the owner's are written by the same pair
    /// of functions. Named rather than assumed away, because the coupling is a
    /// property of two functions and a reader of either one cannot see it.
    NoTaskSpace,
}

/// Why construction of a retained sampler object did not complete.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SamplerResolveError {
    Rung(LadderRung),
    Decode {
        status: crate::runtime::decode::resource::DecodeStatus,
        descriptor_len: usize,
        tag: Option<u32>,
        declared_len: Option<u32>,
    },
}

/// Retrieve or construct the sampler named by `task_id` / `sampler_ref`.
///
/// A sampler is an immutable object in its own task-local reference space.
/// Successful construction snapshots and decodes its descriptor once; failed
/// construction remains retryable because nothing is registered until every
/// rung, including decode, succeeds. Its explicit sampler-delete command and
/// task teardown are the only events that retire this entry.
pub fn resolve_sampler_state<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    sampler_ref: u32,
) -> Result<Arc<crate::model::TaskSamplerState>, SamplerResolveError> {
    use crate::runtime::decode::resource::{
        decode_sampler_descriptor, OBJECT_TYPE_SERIALIZER_OBJECT,
    };

    if let Some(sampler) = state.task_sampler_states.get(task_id, sampler_ref) {
        return Ok(sampler);
    }

    let entry = lookup_list_entry(state, host, task_id, sampler_ref)
        .ok_or(SamplerResolveError::Rung(LadderRung::NoListEntry))?;
    if entry.object_type != OBJECT_TYPE_SERIALIZER_OBJECT {
        return Err(SamplerResolveError::Rung(LadderRung::WrongType {
            got: entry.object_type,
        }));
    }
    let bytes = read_descriptor(state, host, task_id, &entry).ok_or(SamplerResolveError::Rung(
        LadderRung::DescRead {
            declared_len: entry.descriptor_length,
        },
    ))?;
    let descriptor_len = bytes.len();
    let tag = bytes.get(..4).map(ld32);
    let declared_len = bytes.get(4..8).map(ld32);
    let descriptor =
        decode_sampler_descriptor(&bytes).map_err(|status| SamplerResolveError::Decode {
            status,
            descriptor_len,
            tag,
            declared_len,
        })?;
    // The model's name for this slot, taken at the one moment it can be.
    //
    // A sampler is constructed out of the guest's object list once and then
    // answered from `task_sampler_states` forever, so nothing asks the list
    // about it again — and the guest clears the list slot before it sends the
    // destroy. A driven macos-15 boot measured what that cost: of 1984 destroys
    // this device retires something for, **the semantic model had a name for
    // zero of them** (`delete_object_retained_ref_model_unnamed`), because the
    // only paths that declare a name are the ones this construction bypasses.
    // A retirement the model cannot name is a retirement it cannot be told
    // about, which is the whole of why `CmdDeleteObject` could not cross the
    // ingress bridge.
    //
    // `name_resource` is that door and not a second declaration site: it
    // answers from `DeviceState::object_name` when a name exists and declares
    // through `declare_object_name` when one does not, exactly as the
    // production `RefResolver` does. Its answer is deliberately dropped — this
    // function returns a sampler, and whether the slot could also be *named* is
    // a separate fact whose absence must not fail a construction that
    // succeeded.
    let _ = name_resource(state, host, task_id, sampler_ref);
    let sampler = state.task_sampler_states.register(
        task_id,
        sampler_ref,
        Arc::new(crate::model::TaskSamplerState { descriptor }),
    );
    crate::runtime::drain::note_store_route("sampler_state_constructed");
    Ok(sampler)
}

/// Object tags constructed through the task's resource registry.
///
/// Bit `n` names `APVObjectType` `n + 1`. The resource constructor accepts exactly 1, 2, 3, 4, 5,
/// 8, 9, 11, 12, 13, 14, and 15 — every tag either rail's guest driver assigns to something with
/// guest-owned storage: buffer, the two normal-texture tags, backing and ref texture on x86, the
/// child/serializer texture, and on arm the mapper-ref texture, the dual-plane texture, the heap
/// pair and the mapper-ref buffer. Function (6) and serializer objects (7) have separate registries
/// and lifetimes, so retaining their descriptors until `DeleteResource` would conflate distinct
/// APIs. An immutable render pipeline constructed from a serializer-object descriptor is retained
/// separately in `DeviceState::task_render_pipeline_states`; the serializer bytes themselves remain
/// outside this resource registry.
const RESOURCE_CONSTRUCTOR_TYPE_MASK: u16 = 0x7d9f;

fn object_type_is_resource(object_type: u8) -> bool {
    object_type
        .checked_sub(1)
        .filter(|&bit| bit < u16::BITS as u8)
        .is_some_and(|bit| RESOURCE_CONSTRUCTOR_TYPE_MASK & (1u16 << bit) != 0)
}

/// Whether a cached resolution still describes the entry the guest's own object
/// list holds for that reference.
///
/// The cache is keyed by `(task, ref)` and dropped on a new page-table root, a
/// new object list, a deleted object, a deleted task, and a replaced physical
/// page. The guest writes the list itself, in its own memory: none of those
/// events accompanies a slot being *overwritten in place*, and a compositor
/// recycles slots. A stale hit binds the bytes of whatever object used to live
/// at that reference, which is a wrong texture rather than a missing one -- and
/// on this rail one reference resolves to a small texture early in a boot and to
/// a 3840x2160 video plane later.
///
/// Read-only, and it decides nothing: it reads the list entry the guest holds
/// now and reports a disagreement. First sight per `(task, ref, kind of
/// disagreement)`, so a steady bind is silent and a recycled slot names itself
/// once.
fn note_stale_task_resource<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    obj_ref: u32,
    cached: &TaskResource,
) {
    let Some(entry) = lookup_list_entry(state, host, task_id, obj_ref) else {
        // The guest's list no longer answers for this reference at all. Counted
        // apart from the comparisons: it is not a disagreement about what the
        // slot holds, it is the list having moved or gone, and folding it into
        // the denominator would make the agreement rate a function of task
        // teardown.
        crate::runtime::drain::note_store_route("task_resource_unlisted");
        return;
    };
    // The entry is half the snapshot. The other half is the descriptor bytes it
    // points at, and a serializer that rewrites a descriptor in place leaves the
    // entry byte-identical -- same type, same length, same address -- while the
    // object it describes becomes a different surface, plane or extent. A
    // witness that compared only the entry would call that agreement.
    let live_descriptor = read_descriptor(state, host, task_id, &entry);
    let descriptor_moved = live_descriptor
        .as_ref()
        .is_some_and(|bytes| bytes.as_slice() != &*cached.descriptor);
    // The denominator, and it decides something. The `task_resource_stale` line
    // below is `first_sight`-gated, so a boot with no line is a boot where the
    // guest never overwrote a slot **or** a boot where this witness never
    // compared anything -- and those are opposite facts about the same silence.
    //
    // What it decides is the cutover's declaration discipline. This device
    // declares an object when it first constructs it, and on this interface the
    // guest replaces an object by writing over its own object-list record with
    // no packet at all. `reims_vgpu_core::namespace::Namespace::declare` says a
    // live slot may be redeclared and gives the new occupant a new generation;
    // if that happens on a real guest, the replacement model needs the device to
    // notice it here and redeclare, or every later reference resolves to the
    // previous occupant's bytes. If it never happens, it does not.
    //
    // Read, on a driven macos-15 boot to the desktop:
    //
    //     task_resource_reexamined            68965
    //     task_resource_slot_repointed            0
    //     task_resource_descriptor_rewritten      0
    //     task_resource_unlisted                  0
    //
    // And again, on the boot where the namespace is the naming authority and a
    // repoint would have to be answered rather than only counted:
    //
    //     task_resource_reexamined           137041
    //     task_resource_slot_repointed            0
    //     task_resource_descriptor_rewritten      0
    //     task_resource_unlisted                  0
    //
    // Twice the comparisons, the same zero. Sixty-nine thousand comparisons and
    // not one disagreement, then a hundred and thirty-seven thousand. On this rail the
    // guest never repoints a live slot and never rewrites a descriptor under
    // one, so declaring at first construction is sound here and the
    // redeclaration path is unexercised rather than unneeded — the model handles
    // the overwrite either way, and this is the number that says how often it
    // would have to.
    crate::runtime::drain::note_store_route("task_resource_reexamined");
    if entry == cached.entry && !descriptor_moved {
        return;
    }
    // Counted on every sighting rather than the first, and split, because the
    // two disagreements are different events: the slot pointing somewhere else
    // is the guest replacing the object, and the same bytes at the same address
    // having changed is the serializer rewriting a descriptor in place.
    crate::runtime::drain::note_store_route(if entry == cached.entry {
        "task_resource_descriptor_rewritten"
    } else {
        "task_resource_slot_repointed"
    });
    let disc = crate::backend::hash::hash_u64(
        u64::from(task_id) << 32 | u64::from(obj_ref),
        entry.descriptor_gva
            ^ (u64::from(entry.object_type) << 56)
            ^ (u64::from(descriptor_moved) << 63),
    );
    if !crate::observe::first_sight("task_resource_stale", disc) {
        return;
    }
    crate::observe::fail(format!(
        "task_resource_stale task={task_id} ref={obj_ref} \
         cached=[type={} len={} gva={:#x}] list=[type={} len={} gva={:#x}] \
         desc_moved={descriptor_moved} \
         (the guest overwrote the slot and the cached resolution outlived it)",
        cached.entry.object_type,
        cached.entry.descriptor_length,
        cached.entry.descriptor_gva,
        entry.object_type,
        entry.descriptor_length,
        entry.descriptor_gva,
    ));
}

/// The guest-VA allocation an object's own descriptor names, from bytes.
///
/// For the callers that hold descriptor *bytes* rather than a constructed
/// resource — a re-read of the guest's own list, where the whole point is not
/// to trust this device's cache. It decodes and then asks
/// [`Descriptor::backing_window`], which is the one implementation of the
/// arithmetic; a caller holding a `TaskResource` should use
/// [`cached_window`] and pay for no decode at all.
fn backing_window(
    page_shift: u32,
    entry: &ListObjectEntry,
    descriptor: &[u8],
) -> Option<(u64, u64)> {
    if crate::runtime::decode::resource::descriptor_is_heap_placement(descriptor) {
        return None;
    }
    crate::runtime::decode::resource::decode_descriptor(entry.object_type, descriptor)
        .ok()?
        .backing_window(page_shift)
}

/// Whether the guest's object list still says `holder` occupies `base`.
///
/// Read out of guest memory rather than out of this device's construction
/// cache, because the cache cannot see a free: see
/// [`note_backing_window_alias`] for what that costs and why this is the only
/// test that separates an alias from a reuse.
fn holder_still_names<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    holder: u32,
    base: u64,
) -> bool {
    let Some(entry) = probe_list_entry(state, host, task_id, holder) else {
        return false;
    };
    let Some(descriptor) = read_descriptor(state, host, task_id, &entry) else {
        return false;
    };
    backing_window(state.page_shift, &entry, &descriptor).is_some_and(|(held, _)| held == base)
}

/// Whether a second live reference in this task already names this window.
///
/// # The question this settles, and why it blocks a canonical backing identity
///
/// `reims_vgpu_core::access::BackingId` is the identity two resources sharing
/// backing must share, and the derivation this device is committed to is the
/// window plus an incarnation. The incarnation was counted *per reference*,
/// because a re-point packet names a reference and nothing else.
///
/// That is canonical exactly as long as one window has one live name. If two
/// references in one task name one window, then re-pointing through one of them
/// advances that reference's count and leaves the other where it was — and the
/// same bytes carry two ids. Two ids for one backing is a hazard edge the
/// dependency compiler never draws, which is a data race, and it is the failure
/// direction `BackingId`'s own doc calls false distinctness.
///
/// So this was never a diagnostic about tidiness: a single sighting falsifies
/// the derivation. **It sighted one.** A driven macos-15 boot examined 74
/// collisions and one was a genuine alias — two live references over a single
/// 8 294 400-byte window, 1920×1080×4, the compositor's own scanout allocation
/// and so the most hazard-critical backing on the device. The count moved from
/// the reference to the window, which is where `DeviceState::storage_incarnation`
/// keeps it now.
///
/// That is not the end of this instrument's job. It still adjudicates every
/// collision, and it is what fills the aliased-reference set the per-reference
/// state that *remains* is checked against — see
/// [`note_reference_shares_storage`]. A second sighting on a window nothing has
/// re-keyed is still a finding.
///
/// # The confound this has to exclude, and how
///
/// One name *after* another over one allocation is not two names at once, and
/// it is the common case: the guest frees an object by writing over its own
/// object-list record, with no packet, and then reuses the pages. This device
/// is not told, so [`crate::model::TaskResources`] goes on holding the freed
/// reference's construction and a liveness test against it would call a
/// sequence an alias.
///
/// So liveness is asked of the guest's own list, not of this device's cache:
/// the holder's *current* object-list entry is re-read and its window
/// re-derived, and the sighting stands only if that still names the very same
/// window. A holder whose slot has been overwritten, cleared, or re-pointed
/// answers with a different window or none, and the claim quietly moves to the
/// new reference. That is the same technique — and the same reason —
/// [`note_stale_task_resource`] uses to tell a recycled slot from a steady one.
///
/// One `first_sight` per distinct window, so a compositor recycling one
/// allocation reports once and not once per frame.
///
/// The aliasing reference is returned as well as reported, so the reading can
/// be asserted where a boot is not available — the same reason
/// [`note_heap_reference`] returns one.
fn note_backing_window_alias<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    obj_ref: u32,
    base: u64,
    size: u64,
) -> Option<u32> {
    let holder = state.claim_backing_window(task_id, obj_ref, base)?;
    // Every collision is counted, whichever way it is adjudicated, because a
    // zero on the sighting alone is not interpretable: a boot where no window
    // was ever claimed twice and a boot where the guest-list re-read never
    // succeeded read the same. `backing_window_reclaimed` is the denominator
    // that tells them apart, and a null reading only means "no aliases" when
    // it is non-zero.
    crate::runtime::drain::note_store_route("backing_window_collision");
    if !holder_still_names(state, host, task_id, holder, base) {
        crate::runtime::drain::note_store_route("backing_window_reclaimed");
        state.take_backing_window(task_id, obj_ref, base);
        return None;
    }
    // Recorded before the reporting gate, and on every sighting rather than the
    // first. The hot paths that ask `DeviceState::reference_is_aliased` are
    // only as sharp as this set is early and complete, and construction is the
    // earliest moment either name exists — a payment for a reference the set
    // has not heard of yet is a payment the alarm cannot examine.
    state.note_aliased_reference(task_id, obj_ref);
    state.note_aliased_reference(task_id, holder);
    if !crate::observe::first_sight("backing_window_alias", (u64::from(task_id) << 40) ^ base) {
        return Some(holder);
    }
    crate::observe::fail(format!(
        "backing_window_alias task={task_id} base={base:#x} size={size} \
         refs=[{holder},{obj_ref}] (two live references in one task name one \
         guest-VA window; the storage incarnation is keyed on the window so both \
         names move together, and this line is what any per-reference state \
         still standing for storage has to be checked against)"
    ));
    Some(holder)
}

/// Why an object this device constructed cannot be declared into the semantic
/// model.
///
/// Both arms are gaps rather than guest errors, and both are counted, because
/// the number that matters before the cutover is what fraction of a real
/// guest's object list the model can describe at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageRefusal {
    /// The storage has no canonical identity. Carries the identity's own
    /// refusal, so the reason is the one [`backing_id`] names rather than a
    /// second vocabulary for the same facts.
    Backing(BackingIdRefusal),
    /// The identity is known and the object's own byte window inside it is not.
    ///
    /// The mapper-ref texture is the case: its storage is a mapping's surface,
    /// which has an identity, and which *plane* of that surface the texture is
    /// comes from the surface descriptor rather than from this record. An
    /// extent guessed at would declare content authority over a neighbouring
    /// plane's bytes.
    ExtentUnrecovered { object_type: u8 },
}

impl StorageRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Backing(inner) => inner.slug(),
            Self::ExtentUnrecovered { .. } => "storage_extent_unrecovered",
        }
    }
}

/// What the semantic model would be told this object's storage is.
///
/// The object-list walk's per-object translation, and a pure function of the
/// entry, the descriptor and the identity interning:
/// `reims_vgpu_core::lifecycle::LifecycleOp::CreateResource` carries exactly
/// this value, and it is the last thing between a constructed guest object and
/// a declaration the model admits.
///
/// # The three answers, and why the third is not a fourth
///
/// An object that owns an allocation is `Dedicated`, with the *identity* of the
/// allocation and this object's own byte window inside it — two different
/// questions, answered by two different methods, because two textures at
/// different offsets in one allocation are one storage with two extents.
///
/// An object that owns no bytes at all is `NoBytes`, which is most of a list:
/// samplers, functions, pipeline states, depth-stencil states, views over
/// another object's bytes. That is a description and not a refusal — the model
/// holds the name, the generation and the deletion, and holds no memory because
/// there is none.
///
/// A heap placement would be `Placed`, and is not produced here. The identity
/// half is settled but a heap's *extent* is not, so a placement cannot be
/// bounded and two placements sharing bytes would come out distinct; it arrives
/// as [`BackingIdRefusal::HeapPlaced`] and is refused by name. No driven boot
/// has yet placed a heap texture, which is why the term is still open.
///
/// # Errors
///
/// [`StorageRefusal`], one arm per half of the answer that is missing.
pub fn declared_storage(
    state: &DeviceState,
    task_id: u32,
    obj_ref: u32,
    entry: &ListObjectEntry,
    descriptor: &[u8],
) -> Result<reims_vgpu_core::lifecycle::Storage, StorageRefusal> {
    use reims_vgpu_core::access::ByteRange;
    use reims_vgpu_core::lifecycle::Storage;
    let backing = match backing_id_of(state, task_id, obj_ref, entry, descriptor) {
        Ok(backing) => backing,
        // The object names no storage, which is the whole answer rather than a
        // failure to find one.
        Err(BackingIdRefusal::NamesNoStorage { .. }) => return Ok(Storage::NoBytes),
        Err(other) => return Err(StorageRefusal::Backing(other)),
    };
    // Two derivations, because the two kinds of storage measure their extent in
    // different spaces. An address-named object's window is an offset into the
    // allocation its own descriptor names, and that descriptor is where it comes
    // from. A mapper-ref texture's is an offset into a *surface* it shares with
    // the other planes of one IOSurface, and which plane it is comes from the
    // mapping's published device descriptor.
    let extent = if entry.object_type == OBJECT_TYPE_MAPPER_REF_TEXTURE {
        mapper_ref_surface_extent(state, descriptor)
    } else if entry.object_type == OBJECT_TYPE_BACKING {
        // A surface-backing object *is* the whole surface, and its own record
        // says how long that is. There is no plane to pick out and no
        // allocation to be a window of: the mapper-ref textures over this
        // mapping are the ones with offsets, and they take them from the
        // published device descriptor.
        reims_vgpu_wire::device_desc::backing_header(descriptor)
            .ok()
            .map(|header| (0u64, header.length.get()))
    } else {
        crate::runtime::decode::resource::decode_descriptor(entry.object_type, descriptor)
            .ok()
            .and_then(|decoded| decoded.allocation_extent())
    };
    let (offset, length) = extent.ok_or(StorageRefusal::ExtentUnrecovered {
        object_type: entry.object_type,
    })?;
    note_extent_against_allocation(state, task_id, obj_ref, entry, descriptor, offset, length);
    Ok(Storage::Dedicated {
        backing,
        extent: ByteRange { offset, length },
    })
}

/// Say when a declared extent is a strict sub-window of the allocation it sits
/// in, and how short.
///
/// **The census the exec window refusals need.** `Lifecycle::access` checks a
/// record's `Range` participation against the resource's *extent* — the union of
/// its level records for a texture, the whole allocation for a buffer — and a
/// driven macos-15 boot refuses seven exec packets a boot on
/// `OutOfPlacement`, one of them `ref=28` asking `offset=65536 length=196608`
/// of a resource whose resident range is `{offset: 0, length: 196608}`. The
/// window's end is 262 144, which is exactly 256 KiB, and the extent is exactly
/// 192 KiB: a record naming the allocation where the model holds a sub-window
/// of it would look precisely like that.
///
/// So this measures the gap directly, and the gap is not rare. A driven
/// macos-15 boot:
///
/// ```text
/// declared_extent_is_the_allocation   6348
/// declared_extent_is_a_sub_window     4117
/// declared_extent_over_no_window       204
/// ```
///
/// **Two declarations in five put an extent strictly inside its allocation**,
/// and every one of the 492 distinct `(task, ref)` pairs named is a texture —
/// 175 `OBJECT_TYPE_TEXTURE` and 317 `OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS`,
/// none of any other type. That is `Descriptor::allocation_extent`'s union of
/// level records doing exactly what its doc says, and it is what makes the
/// window check a live surface rather than a corner: the refusals are seven a
/// boot today against four thousand declarations that could produce one.
///
/// It also shows the neighbours. `ref=27` holds `{0, 65536}` of a 196 608-byte
/// allocation and `ref=28` holds the whole 196 608 of one — and it is `ref=28`
/// that a record asks for `offset=65536 length=196608` of, overrunning by
/// exactly `ref=27`'s length. What is still unrecovered is which record that
/// is: the refusal names its `StreamSite` and not its kind, and the kind is
/// what says whether the offset belongs to the object or to the allocation.
fn note_extent_against_allocation(
    state: &DeviceState,
    task_id: u32,
    obj_ref: u32,
    entry: &ListObjectEntry,
    descriptor: &[u8],
    offset: u64,
    length: u64,
) {
    let Some((_, allocation)) = backing_window(state.page_shift, entry, descriptor) else {
        crate::runtime::drain::note_store_route("declared_extent_over_no_window");
        return;
    };
    if offset.saturating_add(length) >= allocation {
        crate::runtime::drain::note_store_route("declared_extent_is_the_allocation");
        return;
    }
    crate::runtime::drain::note_store_route("declared_extent_is_a_sub_window");
    if crate::observe::first_sight(
        "declared_extent_is_a_sub_window",
        (u64::from(task_id) << 32) | u64::from(obj_ref),
    ) {
        crate::observe::fail(format!(
            "declared_extent_is_a_sub_window task={task_id} ref={obj_ref} \
             object_type={} extent_offset={offset} extent_length={length} \
             allocation={allocation} (the model will check every record's window against \
             the extent, so a record naming the allocation refuses here)",
            entry.object_type,
        ));
    }
}

/// The device answering the semantic model's mapping-namespace resolver.
///
/// # The first of the model's two resolvers this device satisfies
///
/// `reims_vgpu_core::resolve` splits resolution in two because it splits two
/// namespaces: a mapping id and an object-list ref arrive as `u32`s that overlap
/// numerically and name unrelated things. This is the mapping half, and the
/// device can answer it in full today — [`mapping_backing_id`] mints the
/// identity of a mapping's surface storage, and that is the whole of what this
/// trait asks.
///
/// The object half is answered by [`TaskRefResolver`], and what unblocked it was
/// the device taking a namespace: `RefResolver` answers a `ResourceId`, which
/// carries a slot *generation*, and generations are
/// `reims_vgpu_core::namespace::Namespace`'s to issue. While this device had
/// only a construction cache keyed by `(task, reference)` with no generation in
/// it, implementing the trait over that cache would have minted generations from
/// nothing — the second lifetime model the replacement plan forbids. It now has
/// the namespace, so the answer is the namespace's and is not minted here.
///
/// # A refusal is `None`, and that is the trait's own shape
///
/// The trait answers "the backing a mapping's surface is resolved to, or `None`
/// when the mapping names no live surface". Both of [`MappingBackingRefusal`]'s
/// arms are exactly that — a mapping id nothing is listed for, and an entry the
/// guest has never mapped a surface into — so nothing is lost by flattening
/// them here. The typed arms stay available to callers that want to say which,
/// and `reims_vgpu_core::query`'s present path wants only whether.
impl reims_vgpu_core::resolve::MappingResolver for DeviceState {
    fn backing(
        &self,
        mapping: reims_vgpu_core::identity::MappingId,
    ) -> Option<reims_vgpu_core::access::BackingId> {
        mapping_backing_id(self, mapping.0).ok()
    }
}

/// Census whether the refs a resource-lifetime packet names already have names
/// in the task's object namespace.
///
/// # The question the cutover cannot ask afterwards
///
/// `reims_vgpu_core::lifecycle::resource_list` resolves every ref in a counted
/// list or refuses the whole packet — deliberately, since a partial
/// `Invalidate` claims the resources that happened to resolve are the only ones
/// that went stale. So the four list commands cross to the model exactly when
/// every ref they name resolves, and this device's namespace is populated
/// **lazily**: [`resolve_resource`] declares an object the first time something
/// reads it, not when the guest publishes its object list.
///
/// Those two facts have a collision in them. A guest may legally synchronize,
/// invalidate or discard a resource this device has never had a reason to
/// construct — nothing in the interface says a resource must be drawn before it
/// is named — and every such packet would refuse at the bridge while being
/// perfectly well formed. Whether that is a corner or the common case is not
/// something the code says; it is a property of a real driver's ordering, and it
/// decides whether the lazy declaration can stand or whether the namespace has
/// to be populated from the guest's table at `SetObjectList` time.
///
/// So the reading is taken before the switch is cut, at the two arms that
/// already decode these lists, against an unmodified guest.
/// `lifetime_ref_asked` is the denominator — without it a boot where every ref
/// resolved and a boot where no list packet arrived read the same.
///
/// # It reads twice, because the first reading answered the wrong half
///
/// The first driven reading of this census was **49 unnamed of 1339**, which
/// said lazy declaration leaves a residue and said nothing about what closes it.
/// Read early in that same boot it was 16 of 17 — which is why a census is
/// summed over a whole run and never sampled: the cold start had the ratio
/// inverted, and stopping there would have concluded that lazy declaration was
/// useless rather than nearly sufficient. So the
/// census now asks the second question too, at the same site: for a ref
/// construction has not named, can [`name_resource`] name it from the guest's
/// own object list? That is what the bridge will do, and
/// `lifetime_ref_named_on_demand` is the count of refs it works for.
/// `lifetime_ref_unnameable` is the residue — an empty slot, an unreadable
/// descriptor — which no namespace could have answered for either and which the
/// cutover would still refuse.
///
/// Asking is not free of consequence and is not meant to be: `name_resource`
/// declares. That is the behaviour the reading is *of*, not an observation
/// perturbing it — the census is the first caller of the door the bridge will
/// use, and the numbers are what that door does on a real guest.
///
/// # The second reading, on a driven macos-15 boot to the desktop
///
/// ```text
/// lifetime_ref_asked                  1232
/// lifetime_ref_already_named          1213
/// lifetime_ref_named_on_demand          19
/// lifetime_ref_unnameable                0
/// lifetime_ref_list_would_refuse         0
/// object_named_before_constructed        0
/// object_declared_over_a_live_name       0
/// ```
///
/// 1213 + 19 = 1232 exactly, and **the residue is zero**: every ref a real
/// guest's lifetime lists named was answerable, and not one packet would have
/// been refused for want of a name. The 19 are the ones the previous boot
/// counted as losses (49 there, same order of magnitude on a workload of the
/// same size), and they are now answered from the guest's own table rather than
/// waiting for a draw that never comes.
///
/// The two zeros beside them are what says the door is safe as well as
/// sufficient. `object_declared_over_a_live_name` stayed at zero, so naming
/// ahead of construction displaced nothing; and `object_named_before_constructed`
/// — the arm [`resolve_resource`] takes when it finds a name it did not issue —
/// also read zero, which is a reading and not a gap: on this rail the 19 refs
/// named by a lifetime packet were never subsequently drawn. It is the counter
/// that would say so if they were, and the path it guards is the one that keeps
/// a later construction from minting a second generation over a live object.
///
/// # What it deliberately does not cover
///
/// The two single-ref commands (`DeleteResource`, `ReplacePhysical`) and the
/// EXEC resource table are not counted here. Each decodes elsewhere, and a
/// census spread across four decoders would be four chances to count one packet
/// twice or none. These two arms are where the model's own list join reads, and
/// they are the ones whose all-or-nothing rule makes an unnamed ref cost the
/// whole packet.
pub fn note_lifetime_refs_named<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    refs: &[u32],
) {
    // Counted per packet as well as per ref, because the refusal is per packet:
    // one unnamed ref in a list of forty costs all forty.
    crate::runtime::drain::note_store_route("lifetime_ref_list_asked");
    let mut unnamed = 0usize;
    for &object_ref in refs {
        crate::runtime::drain::note_store_route("lifetime_ref_asked");
        if state.object_name(task_id, object_ref).is_some() {
            crate::runtime::drain::note_store_route("lifetime_ref_already_named");
            continue;
        }
        // The second half of the reading, and the one that decides. The first
        // half said construction had not named this ref; this asks whether the
        // guest's own object list can, which is what the bridge will do. A ref
        // this cannot name is one no populated namespace would have either — an
        // empty slot or an unreadable descriptor — and is the residue the
        // cutover would still have to refuse.
        if name_resource(state, host, task_id, object_ref).is_some() {
            crate::runtime::drain::note_store_route("lifetime_ref_named_on_demand");
        } else {
            crate::runtime::drain::note_store_route("lifetime_ref_unnameable");
            unnamed += 1;
        }
    }
    if unnamed == 0 {
        return;
    }
    // The packet-level count is what the bridge's all-or-nothing rule prices.
    crate::runtime::drain::note_store_route("lifetime_ref_list_would_refuse");
    if crate::observe::first_sight(
        "lifetime_ref_unnamed",
        (u64::from(task_id) << 32) | refs.first().copied().map_or(0, u64::from),
    ) {
        crate::observe::fail(format!(
            "lifetime_ref_unnamed task={task_id} unnamed={unnamed} of={} first_ref={:?}              (the guest named a resource in a lifetime list that this device has never              constructed, so the namespace has no name for it; the model's list join              resolves every ref or refuses the packet, which makes this the count that              decides whether lazy declaration can stand)",
            refs.len(),
            refs.first(),
        ));
    }
}

/// The device answering the semantic model's object-namespace resolver, for one
/// task.
///
/// # Why this is a borrow of the device and not the device
///
/// `reims_vgpu_core::resolve::RefResolver::resource` takes an `object_ref` and
/// nothing else, because in the model a resolver *is* one namespace: the
/// interpreter that holds one is already inside the task whose refs it is
/// resolving. This device holds every task's namespace at once, keyed by
/// `task_id`, so it cannot implement the trait itself without inventing an
/// answer to "which task" — and the two plausible inventions, a current-task
/// field or a default, are both a place for the wrong namespace to answer a
/// well-formed question.
///
/// So the task is bound at construction and the trait is implemented on the
/// binding. A caller that has not said which task cannot ask, which is the
/// property the trait's shape assumes and this device would otherwise not have.
///
/// # It resolves and does not lease
///
/// [`DeviceState::object_name`] is the whole implementation, and it asks the
/// namespace's non-consuming door. A resolver that took a lease would be a claim
/// on a resource nobody acquired, made by a lookup — the dependency compiler
/// resolves refs to draw edges, not to hold things.
#[derive(Clone, Copy)]
pub struct TaskRefResolver<'a> {
    state: &'a DeviceState,
    task_id: u32,
}

impl<'a> TaskRefResolver<'a> {
    /// Bind this device's resolution to one task's namespace.
    #[must_use]
    pub const fn new(state: &'a DeviceState, task_id: u32) -> Self {
        Self { state, task_id }
    }

    /// The task whose namespace this answers from.
    #[must_use]
    pub const fn task_id(self) -> u32 {
        self.task_id
    }
}

impl reims_vgpu_core::resolve::RefResolver for TaskRefResolver<'_> {
    fn resource(&self, object_ref: u32) -> Option<reims_vgpu_core::identity::ResourceId> {
        self.state.object_name(self.task_id, object_ref)
    }
}

/// The device as the source of every task's object namespace.
///
/// # Why the device implements this and not [`TaskRefResolver`]'s trait
///
/// A byte-to-operation join in `reims_vgpu_core::lifecycle` learns which task a
/// packet is about by decoding the packet, so it cannot be handed a namespace
/// that is already bound — the binding is the answer it is computing. It is
/// handed this instead, and binds the task itself. That is the whole difference
/// between the two traits and it is not stylistic: a ref is an index into the
/// naming task's own object list, so another task's namespace does not refuse
/// it, it resolves it, to another task's resource.
///
/// [`TaskRefResolver`] stays for the callers that legitimately are inside one
/// task already — a command-stream walk, which was given its task by the header
/// it is walking under.
///
/// # Why it carries guest memory, and why there is not a cheaper one beside it
///
/// This resolves through [`name_resource`], which reads the guest's object list
/// when a reference has no name yet. A version answering only from names already
/// issued would need no host and would be wrong 49 times a boot — see
/// `name_resource`'s own measurement — and it would be wrong *silently*, because
/// the model's list join turns an unresolved ref into a refused packet rather
/// than an error anybody sees.
///
/// So there is one door and not two. Two implementations of one trait, one of
/// which declares and one of which does not, is the shape where a call site
/// picks the cheap one and loses a packet class; the trait answers what a
/// reference names, and on this interface that answer is in the guest's table
/// whether or not this device has read it yet.
pub struct TaskNames<'a, M> {
    state: &'a DeviceState,
    host: &'a M,
}

impl<'a, M: HostMemory> TaskNames<'a, M> {
    /// The device's object namespaces, backed by the guest memory their
    /// object lists live in.
    pub const fn new(state: &'a DeviceState, host: &'a M) -> Self {
        Self { state, host }
    }
}

impl<M: HostMemory> reims_vgpu_core::resolve::TaskNamespaces for TaskNames<'_, M> {
    fn resource(
        &self,
        task: reims_vgpu_core::identity::TaskId,
        object_ref: u32,
    ) -> Option<reims_vgpu_core::identity::ResourceId> {
        name_resource(self.state, self.host, task.0, object_ref)
    }
}

/// The device answering where a named resource's bytes are.
///
/// # Why the same holder answers both traits
///
/// [`TaskNames`] already carries the two things this needs — the namespaces that
/// issued the name, and the guest memory the object list lives in — and the two
/// questions are asked one after the other about one resource. A second holder
/// would be a second `(state, host)` pair that a call site could pair with the
/// wrong task.
///
/// # The generation is checked, and that is the whole of the guard
///
/// A [`reims_vgpu_core::identity::ResourceId`] is a slot *and* a generation, and
/// the slot alone is what [`backing_id`] reads the guest's list with. So a name
/// whose generation is no longer the slot's current one would resolve to the
/// bytes of whatever occupies the slot **now** — the wrong storage, keyed onto
/// an access the guest asked for over the old occupant. Asking
/// `DeviceState::object_name` first makes that a `None` instead: no key, an
/// access at the coarse rung, and no edge drawn against memory the packet did
/// not name.
///
/// # `heap` is `None` because a heap placement never gets this far
///
/// [`ResourceKey`](reims_vgpu_core::access::ResourceKey) carries the heap so a
/// heap-use record and a member access have something to compare. This device
/// has no heap term to carry: a heap's extent is unrecovered, so [`backing_id`]
/// refuses a placement by name as [`BackingIdRefusal::HeapPlaced`] rather than
/// minting an identity that would make two placements over one heap look
/// distinct. Every backing that reaches this line is therefore a dedicated
/// allocation, for which `None` is the fact and not an omission — and the day
/// the heap extent lands, the refusal is where the term arrives.
impl<M: HostMemory> reims_vgpu_core::resolve::ResourceStorage for TaskNames<'_, M> {
    fn storage(
        &self,
        task: reims_vgpu_core::identity::TaskId,
        resource: reims_vgpu_core::identity::ResourceId,
    ) -> Option<reims_vgpu_core::access::ResourceKey> {
        let slot = resource.slot.0;
        if self.state.object_name(task.0, slot) != Some(resource) {
            crate::runtime::drain::note_store_route("lifecycle_storage_name_not_current");
            return None;
        }
        match backing_id(self.state, self.host, task.0, slot) {
            Ok(backing) => Some(reims_vgpu_core::access::ResourceKey {
                backing,
                heap: None,
            }),
            Err(refusal) => {
                crate::runtime::drain::note_store_route(refusal.slug());
                None
            }
        }
    }
}

/// Whether a query's reply buffer lands inside an allocation this device
/// already has an identity for.
///
/// # The question, and why it is asked before anything needs the answer
///
/// `reims_vgpu_core::query::ReplyDestination` is a backing and a window of it,
/// not an address: a query writes into guest memory and the model orders that
/// write like any other. So the reply buffer needs a `BackingId`, and there are
/// only two ways to give it one — resolve it to the allocation it lies in, or
/// mint it one of its own.
///
/// Those are not interchangeable. If a reply buffer is a page of an object's
/// allocation and it gets an identity of its own, the reply write and every
/// access to that object come out over different backings and the hazard edge
/// between them is never drawn. If it lies in no allocation and is resolved to
/// one anyway, the write is ordered against memory it does not touch. Which
/// error is available is a question about a real guest, and this is the
/// instrument that answers it.
///
/// Only the two queries whose destination is a **task GVA** can be asked at all:
/// a device-info reply names a page frame, which is in no task's address space,
/// so no window in this table could contain it. `Gap::ReplyDestination` records
/// that split.
///
/// # What it looks at
///
/// Every constructed resource in the task, from the construction cache, which
/// is the same set `cached_window_peer` scans and is affordable here for the
/// reason it is not everywhere: a query is a handful of packets a boot, not a
/// per-draw path.
///
/// `query_reply_scanned` is the denominator. Without it a boot where no reply
/// overlapped anything and a boot where no query resolved a task read the same.
///
/// # What two driven boots answered
///
/// **Twenty-eight replies scanned, twenty-eight outside every allocation, none
/// inside.** macos-15 reads `query_reply_scanned = 24` and
/// `query_reply_outside_every_allocation = 24`; macos-26 reads 4 and 4. Neither
/// rail produced a `query_reply_inside_an_allocation` line.
///
/// So of the two errors the cutover has to choose between, only one has a
/// population here: giving a task-GVA reply destination a backing of its own is
/// the construction these guests support, and resolving it into an allocation
/// would be ordering the write against memory it does not touch. Twenty-eight
/// is a small denominator and this is a rail-and-workload reading, not a
/// theorem — the route stays so a boot that finds the other case says so.
pub fn note_query_reply_destination(
    state: &DeviceState,
    task_id: u32,
    reply_gva: u64,
    reply_len: u64,
) -> Option<u32> {
    crate::runtime::drain::note_store_route("query_reply_scanned");
    let end = reply_gva.saturating_add(reply_len.max(1));
    let containing =
        state
            .task_resources
            .in_task(task_id)
            .into_iter()
            .find_map(|(name, resource)| {
                let (base, size) = resource.backing_window(state.page_shift)?;
                (base < end && reply_gva < base.saturating_add(size)).then_some((
                    name.slot.0,
                    base,
                    size,
                ))
            });
    let Some((ref_, base, size)) = containing else {
        crate::runtime::drain::note_store_route("query_reply_outside_every_allocation");
        return None;
    };
    crate::runtime::drain::note_store_route("query_reply_inside_an_allocation");
    if crate::observe::first_sight(
        "query_reply_inside_an_allocation",
        (u64::from(task_id) << 40) ^ base,
    ) {
        crate::observe::fail(format!(
            "query_reply_inside_an_allocation task={task_id} gva={reply_gva:#x} len={reply_len} \
             holder={ref_} base={base:#x} size={size} (the reply buffer is part of an \
             allocation this device already identifies, so a reply destination given a \
             backing of its own would leave the write unordered against every access to \
             that object)"
        ));
    }
    Some(ref_)
}

/// The device answering the model's reply-destination resolver.
///
/// # Two destination spaces, one identity space
///
/// `reims_vgpu_core::query::ReplyDestination` is one `BackingId` and a window,
/// and the four questions do not share an address space to derive it from.
/// `CmdGetComputeInfo` and the heap-texture query carry a **task GVA**;
/// `CmdGetDeviceInfo` carries a **guest page frame** in no task's address space
/// at all. Each is resolved in the space it belongs to and both arrive at one
/// identity space, because every route this device interns on draws from one
/// monotone counter — see `crate::model::state::BackingWindowRefs`.
///
/// * A GVA inside an allocation this device already identifies takes **that
///   allocation's** identity, at the offset the reply lies at within it. This
///   is the case that must not be given an identity of its own: the reply write
///   and every access to that object would then come out over different
///   backings and the hazard edge between them would never be drawn. No driven
///   boot has produced one — `query_reply_inside_an_allocation` is 0 across
///   every boot measured — and it is answered correctly rather than left to the
///   case that has been observed.
/// * A GVA outside every allocation is storage all the same, and it is
///   identified as the window it is: `(task, page-aligned base)`, through the
///   same door an object's allocation goes through. A later object constructed
///   at that base arrives at the *same* number, which is right — it is the same
///   bytes.
/// * A page frame is identified as a frame, in the third key space, with no
///   incarnation because a frame is storage rather than a name for storage.
///
/// # The window is the request's, not the answer's
///
/// Each request states how much room it set aside — a pair capacity, a reply
/// length — and that is what the destination's [`ByteRange`] carries. The
/// answer's own length is `ReplyShape`'s, and `reims_vgpu_core::query::Stall::
/// ReplyTooLarge` is the comparison between the two. A destination carrying the
/// answer's length instead would make that comparison trivially true and the
/// overflow it exists to catch unrepresentable.
impl<M: HostMemory> reims_vgpu_core::query::Destinations for TaskNames<'_, M> {
    fn destination(
        &self,
        kind: reims_vgpu_core::query::QueryKind,
        payload: &[u8],
    ) -> Option<reims_vgpu_core::query::ReplyDestination> {
        use reims_vgpu_core::access::ByteRange;
        use reims_vgpu_core::query::{QueryKind, ReplyDestination};
        let state = self.state;
        if let Some(form) = kind.device_info_form() {
            let request = crate::protocol::fifo::decode_device_info(form, payload).ok()?;
            // A page frame in no task, so neither the window nor the mapping key
            // space can name it. The window is the guest's own capacity in
            // pairs, which is the reply buffer's allocation size — the same
            // number `reply_device_info` bounds its write by.
            return Some(ReplyDestination {
                backing: state.frame_backing_identity(request.reply_pfn),
                bytes: ByteRange {
                    offset: 0,
                    length: u64::from(request.pair_capacity)
                        * crate::protocol::info_reply::PAIR_LEN as u64,
                },
            });
        }
        let (raw_task, gva, length) = match kind {
            QueryKind::ComputeInfo => {
                let request = crate::protocol::fifo::decode_compute_info(payload).ok()?;
                (
                    request.raw_task,
                    request.reply_gva,
                    u64::from(request.pair_capacity) * crate::protocol::info_reply::PAIR_LEN as u64,
                )
            }
            QueryKind::HeapTextureSizeAndAlign => {
                let request = crate::protocol::fifo::decode_heap_texture_query(payload).ok()?;
                (request.raw_task, request.reply_gva, request.reply_len)
            }
            // Both answered `Some` above and returned. Named rather than
            // wildcarded so a fifth question is a compile error here.
            QueryKind::DeviceInfo | QueryKind::DeviceInfoLegacy => return None,
        };
        let task = crate::runtime::task_slot::resolve_task_word(
            &state.tasks,
            crate::runtime::task_slot::TaskWordSite::ComputeInfo,
            raw_task,
        )?;
        // The same question the census asks, asked once: whether these bytes are
        // already part of an allocation with an identity.
        match note_query_reply_destination(state, task, gva, length) {
            Some(holder) => {
                let name = state.object_name(task, holder)?;
                let resource = state.task_resources.get(task, name)?;
                let (base, _) = resource.backing_window(state.page_shift)?;
                Some(ReplyDestination {
                    backing: state.backing_identity(task, base),
                    bytes: ByteRange {
                        offset: gva.checked_sub(base)?,
                        length,
                    },
                })
            }
            None => {
                // Storage all the same, identified as the window it is. The
                // base is page aligned so two replies in one page are one
                // backing at two offsets rather than two backings.
                let page = 1u64 << state.page_shift;
                let base = gva & !(page - 1);
                Some(ReplyDestination {
                    backing: state.backing_identity(task, base),
                    bytes: ByteRange {
                        offset: gva - base,
                        length,
                    },
                })
            }
        }
    }
}

/// Whether a device-info reply's page frame could collide with storage this
/// device already identifies, asked at the moment the reply is written.
///
/// # The other half of the reply-destination question
///
/// [`note_query_reply_destination`] asks it of the two queries whose reply
/// buffer is a **task GVA**: does the buffer lie inside an allocation this
/// device has a `BackingId` for, so that a destination given an identity of its
/// own would leave the reply write unordered against every access to that
/// object. That instrument cannot be pointed at `CmdGetDeviceInfo`, whose reply
/// is a bare guest **page frame** in no task's address space — there is no
/// window in that table it could be inside of.
///
/// So the same danger has to be ruled out a different way, and the sound way is
/// the *identity space* rather than geometry. `reims_vgpu_core::access::
/// BackingId` is minted from `(task, allocation base, incarnation)`; a page
/// frame has neither of the first two, so a device-info destination can only
/// ever be given an identity of its own. That is correct exactly when the
/// identity it is given can equal nothing else's — and
/// [`DeviceState::backing_identities_minted`] is the whole of that question,
/// because one monotone counter feeds every key space this device interns on.
/// **Zero minted is the closing answer**: a fresh number cannot equal a number
/// nobody holds, whatever page frame the guest named.
///
/// Counting identities rather than tasks or resources is the correction a
/// driven boot forced. The first reading of this counted both populations and
/// found `tasks=1 resources=0`, which reads as the open case and is not one: a
/// task is a namespace, not storage, and a task that has constructed nothing
/// has caused nothing to be interned. The number that decides is the one the
/// dependency compiler compares.
///
/// The reply path's own documentation states the contract that makes this hold
/// at all: the guest asks **once**, when its accelerator starts, and frees the
/// reply buffer immediately after the single parse. What it does not say is
/// whether "when its accelerator starts" precedes this device's first mint, and
/// no reading settles that. It is what this counts.
///
/// # What a boot's counters mean
///
/// `device_info_reply_scanned` is the denominator — without it, a boot where
/// every reply preceded every mint and a boot where the guest never asked read
/// the same. `device_info_reply_before_any_identity` is the closing answer;
/// `device_info_reply_after_an_identity` is the one that keeps the term open,
/// and it carries the counts on the failure channel because a bare route would
/// not say how much storage the mint would have to be compared against.
///
/// # The mint count alone cannot close it, and two driven boots are why
///
/// macos-15 and macos-26 both read `device_info_reply_scanned = 1` and
/// `device_info_reply_after_an_identity = 1`: the guest asks after this device
/// has minted, every time, so the closing arm above never runs on a real rail
/// and the term would stay open forever on the counter alone.
///
/// The counter was also answering the wrong question. A monotone never-reused
/// [`crate::model::BackingWindowRefs`] counter feeds all three key spaces, so a
/// fresh number *cannot* equal one already handed out — that part is closed by
/// construction and needs no boot. The danger the plan names is the opposite
/// one and is about **storage, not numbers**: if the reply page is also part of
/// storage some other identity already names, then the reply write and every
/// access to that storage come out over two backings and the hazard edge
/// between them is never drawn.
///
/// So the question is asked of geometry, in the one key space a page frame can
/// collide with. A task-GVA window is reached through a task's page table and
/// is [`note_query_reply_destination`]'s half; what is left is a mapping's
/// surface, whose storage *is* a list of guest page frames
/// ([`crate::model::MappingEntry::page_entries`]). Scanning them is affordable
/// here for the same reason the other half's scan is: the guest asks once a
/// boot.
///
/// `device_info_reply_frame_in_no_mapping` is the closing answer and
/// `device_info_reply_frame_in_a_mapping` is the open one — and unlike the
/// mint count, both are reachable on a rail that has already minted.
pub fn note_device_info_reply_destination(state: &DeviceState, reply_pfn: u32) {
    crate::runtime::drain::note_store_route("device_info_reply_scanned");
    let minted = state.backing_identities_minted();
    if minted == 0 {
        crate::runtime::drain::note_store_route("device_info_reply_before_any_identity");
        return;
    }
    crate::runtime::drain::note_store_route("device_info_reply_after_an_identity");
    // Which mapping, if any, already holds this page as part of its surface.
    // `mapped` and a non-empty page list together are what make an entry live
    // storage: an unmapped mapping's stale list names pages it no longer holds,
    // and reporting one would be a finding about nothing.
    let reply_gpa = state.pfn_gpa(reply_pfn);
    let page_shift = state.page_shift;
    let holder = state.mappings.iter().find_map(|(&id, mapping)| {
        if !mapping.mapped || mapping.page_entries.is_empty() {
            return None;
        }
        mapping
            .page_entries
            .iter()
            .any(|&entry| {
                crate::protocol::iosurface_pages::entry_gpa_shift(entry, page_shift)
                    == Some(reply_gpa)
            })
            .then_some(id)
    });
    let Some(holder) = holder else {
        crate::runtime::drain::note_store_route("device_info_reply_frame_in_no_mapping");
        return;
    };
    crate::runtime::drain::note_store_route("device_info_reply_frame_in_a_mapping");
    if crate::observe::first_sight("device_info_reply_frame_in_a_mapping", u64::from(reply_pfn)) {
        let resources = state.task_resources.len();
        let tasks = state.tasks.live_count();
        crate::observe::fail(format!(
            "device_info_reply_frame_in_a_mapping pfn={reply_pfn:#x} mapping={holder} \
             minted={minted} tasks={tasks} resources={resources} (the reply page is part \
             of a mapping's surface, so an identity of its own would leave the reply write \
             unordered against every access to that surface)"
        ));
    }
}

/// Which bytes of a mapping's surface a mapper-ref texture occupies.
///
/// The texture's storage is the mapping's, so its extent is measured in the
/// surface's own bytes and not in an allocation of its own. Which part of the
/// surface it is comes from the mapping's published device descriptor, matched
/// against the geometry the texture's own descriptor declares — the same
/// selection every sampling path on this rail makes, through the same function,
/// so an extent and a sample window can never describe different bytes of one
/// plane.
///
/// `None` when the mapping has published no complete device descriptor yet, or
/// when the geometry matches no plane. Both are the guest not having said which
/// plane this is, and a texture given the whole surface as its extent would
/// declare content authority over its siblings' pixels.
fn mapper_ref_surface_extent(state: &DeviceState, descriptor: &[u8]) -> Option<(u64, u64)> {
    let Ok(crate::runtime::decode::resource::Descriptor::IOSurfaceTexture {
        mapping_id,
        pixel_format,
        width,
        height,
        ..
    }) = crate::runtime::decode::resource::decode_iosurface_texture_descriptor(descriptor)
    else {
        return None;
    };
    let mapping = state.mappings.get(&mapping_id)?;
    let (offset, _bytes_per_row, span_end) =
        crate::protocol::iosurface_pages::sample_window_from_device_desc(
            mapping.device_desc_complete(),
            None,
            pixel_format,
            width,
            height,
        )?;
    Some((offset, span_end.checked_sub(offset)?))
}

/// Census the canonical backing identity of every object this device
/// constructs.
///
/// # Why an identity nothing consumes is measured
///
/// `reims_vgpu_core::access::BackingId` is what the replacement model's
/// dependency compiler draws every hazard edge on, and no packet class has
/// moved to that model yet — so [`backing_id_of`] has no production consumer.
/// It will have exactly one, all at once, and the question a cutover cannot ask
/// afterwards is *what fraction of a real guest's objects the identity actually
/// answers for*. A rail where a tenth of constructions land on a refusal is a
/// rail where a tenth of the hazard graph is missing, and it would show up as
/// an intermittently wrong frame rather than as a refusal.
///
/// So the reading is taken now, at the one site every constructed object passes
/// through, against an unmodified guest. `backing_identity_asked` is the
/// denominator — without it a boot with no refusals and a boot with no
/// constructions read the same — and each refusal arm is counted under its own
/// slug.
///
/// What is asked for is the whole *declaration*, not the identity alone, because
/// the declaration is what the cutover needs. An object with an identity and no
/// extent is one the model can name and cannot hold content authority for, and a
/// census that stopped at the identity would score it as an answer. So the
/// `declared_storage_*` counts partition the objects the model can describe —
/// an allocation with an extent, or no bytes at all — and the refusals partition
/// the ones it cannot.
///
/// # Read, on a driven macos-15 boot to the desktop
///
/// ```text
/// backing_identity_asked                 1447
/// backing_identity_minted                1371
/// backing_id_names_no_storage              76
/// backing_id_unresolved                     0
/// backing_id_heap_placed                    0
/// backing_id_mapping_names_no_surface       0
/// ```
///
/// 1371 + 76 = 1447 exactly: **every object this guest constructed got an
/// answer**, and none of them fell to a missing contract term. Both halves of
/// the identity answered — the zero on `mapping_names_no_surface` is the
/// mapper-ref delegation landing identities rather than refusals, and before it
/// every object of that type was unidentifiable by construction. The 5% that
/// name no storage are the samplers, functions, pipeline states and views,
/// which is a description of what they are rather than a shortfall.
///
/// A second boot, once the census asked for the whole declaration:
///
/// ```text
/// backing_identity_asked                 1462
/// declared_storage_dedicated             1393
/// declared_storage_no_bytes                69
/// storage_extent_unrecovered                0
/// backing_id_unresolved                     0
/// backing_id_heap_placed                    0
/// ```
///
/// 1393 + 69 = 1462, again exactly. **Every object a real guest constructs can be
/// declared into the semantic model** — an allocation with the object's own byte
/// window inside it, or no bytes at all. The zero on `extent_unrecovered` is the
/// mapper-ref texture's plane derivation holding on a live guest: before it, that
/// whole class had an identity the model could name and no content authority to
/// go with it.
///
/// A third boot, this one with the object namespace as the device's naming
/// authority rather than a shadow beside it:
///
/// ```text
/// backing_identity_asked                 1638
/// declared_storage_dedicated             1566
/// declared_storage_no_bytes                72
/// declared_storage_placed                   0
/// storage_extent_unrecovered                0
/// backing_id_unresolved                     0
/// backing_id_heap_placed                    0
/// ```
///
/// 1566 + 72 = 1638, exact for the third time, on the boot where the declaration
/// is no longer only measured — the name every object carries is the one this
/// walk issued. The split holds its shape across all three boots (95% owning an
/// allocation, 5% owning nothing) on workloads differing by 13%, which is what
/// says the partition is a property of what a macOS object list *is* rather than
/// of one boot's timing.
///
/// One arm is a fail line as well as a count.
/// [`BackingIdRefusal::HeapPlaced`] is the one open contract term
/// `BackingId`'s own doc still names: a heap's extent is unrecovered, so two
/// placements sharing bytes would come out distinct. No driven boot has yet
/// placed a heap texture, and the *sighting* is what would turn that from a
/// term nobody can look for into a descriptor at a known address.
fn note_backing_identity(
    state: &DeviceState,
    task_id: u32,
    obj_ref: u32,
    entry: &ListObjectEntry,
    descriptor: &[u8],
) {
    use reims_vgpu_core::lifecycle::Storage;
    crate::runtime::drain::note_store_route("backing_identity_asked");
    // The declaration is asked for rather than the identity, because it is the
    // declaration the cutover needs: an identity with no extent describes an
    // object the model can name and cannot hold content authority for, and a
    // census that stopped at the identity would score that as an answer.
    let refusal = match declared_storage(state, task_id, obj_ref, entry, descriptor) {
        Ok(Storage::Dedicated { .. }) => {
            crate::runtime::drain::note_store_route("backing_identity_minted");
            crate::runtime::drain::note_store_route("declared_storage_dedicated");
            return;
        }
        // Not a refusal: the object owns no bytes and the model is told so.
        Ok(Storage::NoBytes) => {
            crate::runtime::drain::note_store_route("declared_storage_no_bytes");
            return;
        }
        // Never produced here — a heap placement arrives as an identity refusal
        // because a heap's extent is unrecovered. Counted so that the day it
        // does, the reading says so rather than the arm being unreachable
        // forever by assumption.
        Ok(Storage::Placed { .. }) => {
            crate::runtime::drain::note_store_route("declared_storage_placed");
            return;
        }
        Err(refusal) => refusal,
    };
    crate::runtime::drain::note_store_route(refusal.slug());
    let StorageRefusal::Backing(refusal) = refusal else {
        // The identity was mintable; only the extent was not.
        crate::runtime::drain::note_store_route("backing_identity_minted");
        return;
    };
    if !matches!(refusal, BackingIdRefusal::HeapPlaced) {
        return;
    }
    if crate::observe::first_sight(
        "backing_id_heap_placed",
        (u64::from(task_id) << 32) | u64::from(entry.descriptor_length),
    ) {
        crate::observe::fail(format!(
            "backing_id_heap_placed task={task_id} type={} desc_len={} desc_gva={:#x} \
             (a heap placement, and a heap's extent is the one identity term still \
             unrecovered — two placements sharing bytes come out distinct, which is a \
             hazard edge the dependency compiler would never draw; the length and \
             address here are where the extent would be read from)",
            entry.object_type, entry.descriptor_length, entry.descriptor_gva,
        ));
    }
}

/// Retrieve or construct the resource named by `task_id` / `obj_ref`.
///
/// A successful construction snapshots the object-list entry and descriptor
/// bytes for the lifetime of that reference. Subsequent binds retrieve the
/// retained resource; guest memory is consulted again only after an explicit
/// resource deletion or task teardown. Failed constructions are not retained,
/// so a descriptor that is still being published can succeed on retry.
pub fn resolve_resource<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    obj_ref: u32,
) -> Result<Arc<TaskResource>, LadderRung> {
    // The namespace decides whether this reference still names anything, and the
    // memo is keyed by its answer. A slot the guest has deleted resolves to
    // nothing, so the retained bytes behind the name it used to have are
    // unreachable here rather than merely unlikely to be reached.
    if let Some(name) = state.object_name(task_id, obj_ref) {
        if let Some(resource) = state.task_resources.get(task_id, name) {
            note_stale_task_resource(state, host, task_id, obj_ref, &resource);
            return Ok(resource);
        }
    }

    let entry = lookup_list_entry(state, host, task_id, obj_ref).ok_or(LadderRung::NoListEntry)?;
    if !object_type_is_resource(entry.object_type) {
        return Err(LadderRung::WrongType {
            got: entry.object_type,
        });
    }
    let bytes = read_descriptor(state, host, task_id, &entry).ok_or(LadderRung::DescRead {
        declared_len: entry.descriptor_length,
    })?;
    let descriptor: Arc<[u8]> = Arc::from(bytes);
    if let Some((base, size)) = backing_window(state.page_shift, &entry, &descriptor) {
        let _ = note_backing_window_alias(state, host, task_id, obj_ref, base, size);
    }
    note_backing_identity(state, task_id, obj_ref, &entry, &descriptor);
    // The one declaration site, carrying what this device established about the
    // object's bytes rather than a flattening of it. See `declare_object_name`
    // for why an undescribable storage is `None` here and not `NoBytes`.
    let storage = declared_storage(state, task_id, obj_ref, &entry, &descriptor).ok();
    // Naming is idempotent, and a name may already exist without a memo: a
    // reference is named the first time *anything* asks what it names, and a
    // lifetime packet asks without constructing. Declaring again here would give
    // the slot a second generation and displace the first — retiring an object
    // the guest never deleted — so the existing name is taken when there is one.
    let name = match state.object_name(task_id, obj_ref) {
        Some(existing) => {
            crate::runtime::drain::note_store_route("object_named_before_constructed");
            existing
        }
        None => declare_object_name(state, task_id, obj_ref, storage)?,
    };
    let resource = Arc::new(TaskResource::new(entry, descriptor));
    Ok(state.task_resources.register(task_id, name, resource))
}

/// Declare a reference into its task's namespace and take the name issued.
///
/// The one call to `DeviceState::declare_object` in this crate, so the
/// displacement it can report is judged once. Callers reach it through
/// [`name_resource`] or through the construction path, both of which have
/// established that the slot holds no live name — which is what makes a
/// displacement here an event rather than the ordinary redeclaration the
/// namespace also supports.
///
/// # The refusal is the owner's now, and it is the one this group was gated on
///
/// `reims_vgpu_core::lifecycle::Lifecycle::create_resource` **refuses**
/// `NoSuchTask` for a declaration into a task no `DefineTask` opened. This
/// device used to create the namespace it needed —
/// `entry(task_id).or_default()` — so the declaration always succeeded, and the
/// difference was counted here against `DeviceState::tasks` as a proxy before
/// the group moved.
///
/// It is not a proxy any more and it is not reachable either. Both callers ask
/// `lookup_list_entry` first, which requires an active `DeviceState::tasks`
/// entry, and that table and the owner's tasks are written by the same pair of
/// functions — `DeviceState::define_task` and `DeviceState::delete_task`. So a
/// refusal here means those two tables have come apart, which is why it is a
/// [`LadderRung::NoTaskSpace`] on the always-on channel rather than a silent
/// `None`.
///
/// # What a driven boot answered
///
/// x86 Vulkan, macos-15, host-driven workload, 8914 draws:
/// **`object_declared_into_a_defined_task=1301`,
/// `object_declared_into_an_undefined_task=0`.** Every declaration this guest
/// makes is into a task a `DefineTask` opened, so the lifecycle owner's
/// `NoSuchTask` would refuse none of them, and the group's one identified
/// refusal costs nothing.
///
/// The same boot re-confirmed the other term the group needs, for the sixth
/// time: `declared_storage_dedicated=1279` + `declared_storage_no_bytes=22` =
/// **1301**, exactly the declarations counted here, with
/// `storage_extent_unrecovered` and `object_declared_over_a_live_name` both
/// silent. Every object this guest constructs is expressible as a
/// `reims_vgpu_core::lifecycle::Storage`, and none would reach the content
/// ledger as a `NoBytes` that has bytes.
fn declare_object_name(
    state: &DeviceState,
    task_id: u32,
    obj_ref: u32,
    storage: Option<reims_vgpu_core::lifecycle::Storage>,
) -> Result<reims_vgpu_core::identity::ResourceId, LadderRung> {
    // **`None` is not `NoBytes`, and collapsing them is the lie this argument
    // exists to stop.** `NoBytes` is a claim — the object owns no memory, which
    // is true of most of a list — and `None` is this device failing to describe
    // storage that exists. Both used to arrive at the namespace as a `None`
    // backing, which is harmless while a slot records only *which* storage a
    // name is over; it stops being harmless the moment these declarations reach
    // `reims_vgpu_core::lifecycle::Lifecycle`, whose `create_resource` tells the
    // content authority that a `NoBytes` object has no bytes to be authoritative
    // about. Declaring that of a resource with bytes would leave its content
    // owned by nothing.
    //
    // So the undescribable case is declared as `NoBytes` here — the namespace
    // has no third answer and a name must still be issued, or the reference
    // resolves to nothing and every packet that mentions it is refused — and it
    // is *counted*, which is what makes it a known floor rather than a silent
    // claim. A driven boot of 15 937 draws read **zero** on this counter
    // directly, which is the reading `storage_extent_unrecovered` had already
    // predicted on six boots by proxy.
    let declared = state.declare_object(
        task_id,
        obj_ref,
        storage.unwrap_or_else(|| {
            crate::runtime::drain::note_store_route("object_declared_with_undescribable_storage");
            reims_vgpu_core::lifecycle::Storage::NoBytes
        }),
    );
    let declared = match declared {
        Ok(declared) => declared,
        Err(refusal) => {
            crate::runtime::drain::note_store_route("object_declared_into_an_undefined_task");
            if crate::observe::first_sight(
                "object_declared_into_an_undefined_task",
                (u64::from(task_id) << 32) | u64::from(obj_ref),
            ) {
                crate::observe::fail(format!(
                    "object_declared_into_an_undefined_task task={task_id} ref={obj_ref} \
                     refusal={} (this reference reached a declaration through an object-list \
                     entry, so DeviceState::tasks holds the task and the lifecycle owner does \
                     not — the two tables are written by one pair of functions and have come \
                     apart)",
                    refusal.slug()
                ));
            }
            return Err(LadderRung::NoTaskSpace);
        }
    };
    crate::runtime::drain::note_store_route("object_declared_into_a_defined_task");
    // Opened once, by name. A declaration names one resource, so the three
    // fields below cannot be filled by it — an address interval, a deferred
    // discard and a task redefinition all belong to other commands — and
    // naming them here is what makes an arm that starts filling one a change at
    // this site rather than a silent loss.
    let crate::model::Acted {
        teardowns,
        remapped,
        at_completion,
        redefined,
    } = declared.acted;
    debug_assert!(
        remapped.is_empty() && at_completion.is_empty() && redefined.is_empty(),
        "a declaration obliges nothing but its displaced occupant's teardown"
    );
    // A declaration over a slot that still held a live object owes that
    // occupant's teardown. It should be unreachable: every caller asked the
    // namespace first and found nothing. Reported rather than assumed away,
    // because if it ever fires the previous occupant's storage is one nothing
    // else will free.
    //
    // The owner carries the displaced occupant's teardown out through the
    // effects rather than handing back the displacement for a caller to
    // discharge — `Lifecycle::create_resource` retires the resident, the heap
    // window and the content authority together, which is exactly what this
    // site could not do when all it held was a namespace.
    for teardown in teardowns {
        crate::runtime::drain::note_store_route("object_declared_over_a_live_name");
        if crate::observe::first_sight(
            "object_declared_over_a_live_name",
            (u64::from(task_id) << 32) | u64::from(obj_ref),
        ) {
            crate::observe::fail(format!(
                "object_declared_over_a_live_name task={task_id} ref={obj_ref} \
                 teardown={teardown:?} (a declaration displaced a live occupant after its \
                 caller had asked the namespace and been told the slot was empty, so the two \
                 reads were not of one namespace)"
            ));
        }
    }
    Ok(declared.id)
}

/// The name a guest reference has in its task, reading the guest's object list
/// to give it one if this is the first sighting.
///
/// # Why naming is separate from constructing
///
/// A driven boot measured it. Of the object refs a real guest names in its
/// resource-lifetime lists — synchronize, invalidate, discard — a namespace
/// populated only by construction answers for **1290 of 1339**, and the
/// remaining **49** are objects this device had never had a reason to construct.
/// Those commands are the guest saying "I am about to touch these with the CPU",
/// and nothing on this interface says a resource must be drawn before it is
/// named.
///
/// 49 is small and it is not a rounding error, because the model's list join
/// resolves every ref in a list or refuses the whole packet — deliberately,
/// since a partial `Invalidate` claims the resources that happened to resolve
/// are the only ones that went stale. This guest's lists carry **one ref each**
/// (1339 lists, 1339 refs), so 49 unnamed refs are 49 refused packets: 49
/// content-authority moves the model would never hear about, spread across the
/// boot rather than clustered at its start.
///
/// Eager declaration is not the alternative. The same boot's
/// `CmdSetObjectList` packets each declare a table of **1 048 576** entries,
/// eighteen times — that number is the list's *capacity* and not its
/// population, and walking it would read twelve megabytes of guest memory per
/// bind to declare a few thousand live objects.
///
/// So a reference is named the first time anything asks what it names, and
/// naming is the cheap half: one list entry and one descriptor read, no host
/// texture, no buffer, no pipeline. [`resolve_resource`] adds the expensive half
/// on top of the same name.
///
/// # It names every object type, not only the resource types
///
/// `resolve_resource` refuses a ref whose entry is not a resource constructor,
/// which is right for construction and wrong for naming: a sampler, a function
/// and a pipeline state are named by their slots exactly as a texture is, they
/// resolve, and they stop resolving when the guest deletes them. See
/// `reims_vgpu_core::namespace`. Their storage is `Storage::NoBytes`, which the
/// declaration carries as `None`.
/// Name the slot a construction has just resolved, and say whether it took.
///
/// # Why construction is where naming has to happen for these kinds
///
/// A destroy arrives after the guest has cleared its object-list slot — 2072
/// of ~2160 on a driven boot — so the destroy itself cannot name what it ends.
/// [`name_resource`] answers from `DeviceState::object_name` before the guest's
/// list, which means a slot named while it was still live is still nameable
/// afterwards; a slot never named is a destroy `SessionModel::admit` refuses
/// with `UnknownRef`. So each kind is named where it is *constructed*, which is
/// a different site per kind, and the boot-measured population says which sites
/// are worth one: `delete_unnamed_*` names the eleven kinds apart precisely so
/// this is aimed rather than guessed.
///
/// The two routes are the caller's because they name the kind, and a caller
/// that passed a `format!` would be a counter no reader greps.
pub fn note_named_at_construction<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    obj_ref: u32,
    named: &'static str,
    unnamed: &'static str,
) {
    crate::runtime::drain::note_store_route(
        if name_resource(state, host, task_id, obj_ref).is_some() {
            named
        } else {
            // The model cannot name the slot even now, while the construction
            // that asked for it is succeeding. Counted rather than silent: it
            // means the list entry the construction read is not the one
            // `name_resource` walks, which is a namespace question and not a
            // missing call.
            unnamed
        },
    );
}

pub fn name_resource<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    obj_ref: u32,
) -> Option<reims_vgpu_core::identity::ResourceId> {
    if let Some(name) = state.object_name(task_id, obj_ref) {
        return Some(name);
    }
    let entry = lookup_list_entry(state, host, task_id, obj_ref)?;
    let descriptor: Arc<[u8]> = Arc::from(read_descriptor(state, host, task_id, &entry)?);
    let storage = declared_storage(state, task_id, obj_ref, &entry, &descriptor).ok();
    declare_object_name(state, task_id, obj_ref, storage).ok()
}

/// Look `obj_ref` up in `task_id`'s list, require its type to be one of `want`,
/// and read its descriptor bytes.
///
/// The first three rungs of the ladder about twenty rails open with, in the one
/// order they can be asked: a type tag cannot be checked before the entry is
/// found, and a descriptor cannot be read before the entry says where it is.
///
/// `want` is a slice because several rails accept more than one tag —
/// `OBJECT_TYPE_TEXTURE` and `OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS` are the standing
/// pair — and a single-tag caller passes a one-element slice rather than a
/// second entry point that would drift from this one.
///
/// `ref_ == 0` is **not** handled here. An unbound ref is a different statement
/// from a ref naming nothing, several rails treat it as expected control flow
/// and stay silent, and `AGENTS.md` names it as one of the things that must not
/// be logged. Callers that care test it before calling.
/// Say whether a memo that refused a caller's type is the guest's own answer.
///
/// **It decides nothing, and what it decides against is a fix.**
/// [`resolve_descriptor`] refuses a memo hit whose type is not wanted without
/// re-reading the guest's list. That is right if the memo is the guest's current
/// answer and wrong if the slot has been recycled under it — and "re-read on a
/// type refusal" is the obvious repair, so the reading below is what says it
/// would repair nothing. `note_stale_task_resource` asks the same question on
/// the memo *hit* path; this asks it where a hit turns into a refusal.
///
/// Read, on a driven macos-15 boot through the ordering model, x86/Vulkan:
///
/// ```text
/// memo_wrong_type                 10216
/// memo_wrong_type_guest_agrees    10216
/// memo_wrong_type_memo_stale          0
/// memo_wrong_type_third_type          0
/// memo_wrong_type_unlisted            0
/// ```
///
/// Ten thousand refusals and not one disagreement: every time a caller was told
/// "that slot is not the type you want", the guest's own object list said the
/// same. The population is one shape — a caller asking for
/// `OBJECT_TYPE_TEXTURE_VIEW` about a slot the guest filled with an
/// `OBJECT_TYPE_REF_TEXTURE` — so it is a question being asked of the wrong
/// slot, not a cache answering for a dead one, and no repair belongs in this
/// resolver.
///
/// [`probe_list_entry`] rather than [`lookup_list_entry`], because this is a
/// question about the slot and not a resolution: the recheck bookkeeping a named
/// lookup does must not count an instrument's read.
fn note_memo_wrong_type<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    obj_ref: u32,
    cached: &TaskResource,
    want: &[u8],
) {
    crate::runtime::drain::note_store_route("memo_wrong_type");
    let live = probe_list_entry(state, host, task_id, obj_ref);
    crate::runtime::drain::note_store_route(match live {
        None => "memo_wrong_type_unlisted",
        Some(ref entry) if entry.object_type == cached.entry.object_type => {
            "memo_wrong_type_guest_agrees"
        }
        Some(ref entry) if want.contains(&entry.object_type) => "memo_wrong_type_memo_stale",
        Some(_) => "memo_wrong_type_third_type",
    });
    if crate::observe::first_sight(
        "memo_wrong_type",
        (u64::from(task_id) << 32) | u64::from(obj_ref),
    ) {
        crate::observe::fail(format!(
            "memo_wrong_type task={task_id} ref={obj_ref} want={want:?} memo_ot={} \
             live_ot={:?} memo_desc_gva={:#x} live_desc_gva={:?} (a retained resolution \
             refused a caller's type without re-reading the guest's list; live_ot is what \
             that list says now)",
            cached.entry.object_type,
            live.as_ref().map(|e| e.object_type),
            cached.entry.descriptor_gva,
            live.as_ref().map(|e| e.descriptor_gva),
        ));
    }
}

pub fn resolve_descriptor<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    obj_ref: u32,
    want: &[u8],
) -> Result<(ListObjectEntry, Arc<[u8]>), LadderRung> {
    if let Some(resource) = state.constructed_object(task_id, obj_ref) {
        if !want.contains(&resource.entry.object_type) {
            note_memo_wrong_type(state, host, task_id, obj_ref, &resource, want);
            return Err(LadderRung::WrongType {
                got: resource.entry.object_type,
            });
        }
        return Ok((resource.entry, Arc::clone(&resource.descriptor)));
    }

    // Keep the ladder's type check ahead of the descriptor read. A caller
    // asking for the wrong kind must not turn an unreadable descriptor into a
    // different refusal merely because resource retention is enabled.
    let entry = lookup_list_entry(state, host, task_id, obj_ref).ok_or(LadderRung::NoListEntry)?;
    if !want.contains(&entry.object_type) {
        return Err(LadderRung::WrongType {
            got: entry.object_type,
        });
    }
    let bytes = read_descriptor(state, host, task_id, &entry).ok_or(LadderRung::DescRead {
        declared_len: entry.descriptor_length,
    })?;
    if !object_type_is_resource(entry.object_type) {
        return Ok((entry, Arc::from(bytes)));
    }
    // Construction is [`resolve_resource`]'s, and this delegates rather than
    // registering a second time. It used to build and publish its own
    // `TaskResource`, which made two sites that could put an object into this
    // device's hands — and only one of them can be the place a name is issued.
    // The cost is re-reading the entry and the descriptor, on a path that is
    // taken once per object lifetime; the alternative is an object with retained
    // bytes and no name.
    let resource = resolve_resource(state, host, task_id, obj_ref)?;
    if !want.contains(&resource.entry.object_type) {
        return Err(LadderRung::WrongType {
            got: resource.entry.object_type,
        });
    }
    Ok((resource.entry, Arc::clone(&resource.descriptor)))
}

/// Why a buffer ref did not yield a backing span.
///
/// The three ways past [`resolve_descriptor`]'s rungs, which is the whole of
/// what [`resolve_buffer_span`] can refuse for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferSpanRefusal {
    /// One of the first three object-list rungs.
    Rung(LadderRung),
    /// The descriptor bytes read, and are not a buffer descriptor.
    Decode,
    /// The descriptor decoded and names no backing allocation — a zero handle
    /// or a zero size. The resource exists; it has nowhere to read from.
    NoBacking,
}

/// Resolve a buffer ref to its `(guest base address, allocation size)`.
///
/// Three rails needed this and each wrote it out: `compute_exec`'s buffer-window
/// read, `icb`'s buffer bind, and `draw`'s vertex/fragment buffer load — which
/// was found last, after this function already existed, and whose five refusals
/// carried no `reason=` field at all, so none of them was in the log's ranking.
/// They agreed on the four steps — resolve as
/// `OBJECT_TYPE_BUFFER`, decode, derive the span from the handle and the
/// device's own `page_shift` — and disagreed on what to say afterwards. The ICB
/// copy named all four refusals; the compute copy returned `Option` and its
/// caller labelled every one of them `no_backing`, which is the *last* of the
/// four and therefore wrong about the other three.
///
/// So the refusal is returned as a value and each rail maps it into its own
/// status vocabulary, rather than one rail's answer being the other's guess.
///
/// `page_shift` comes from the device (x86 12, arm64e 14) and never from the
/// resource decoder's arm-only default; the two place the handle differently.
///
/// `buffer_ref == 0` is not handled here, for the reason [`resolve_descriptor`]
/// gives: an unbound ref is a different statement from a ref naming nothing, and
/// callers that care test it first.
pub fn resolve_buffer_span<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
) -> Result<(u64, u64), BufferSpanRefusal> {
    let resource =
        resolve_resource(state, host, task_id, buffer_ref).map_err(BufferSpanRefusal::Rung)?;
    resolve_buffer_span_from_resource(state, &resource)
}

/// Resolve the backing carried by an encoder-retained buffer object.
pub fn resolve_buffer_span_from_resource(
    state: &DeviceState,
    resource: &crate::model::TaskResource,
) -> Result<(u64, u64), BufferSpanRefusal> {
    if resource.entry.object_type != crate::runtime::decode::resource::OBJECT_TYPE_BUFFER {
        return Err(BufferSpanRefusal::Rung(LadderRung::WrongType {
            got: resource.entry.object_type,
        }));
    }
    let desc = match resource.decoded() {
        Ok(crate::runtime::decode::resource::Descriptor::Buffer(desc)) => desc,
        Ok(_) => {
            return Err(BufferSpanRefusal::Rung(LadderRung::WrongType {
                got: resource.entry.object_type,
            }));
        }
        Err(_) => return Err(BufferSpanRefusal::Decode),
    };
    desc.backing_gva_size(state.page_shift)
        .ok_or(BufferSpanRefusal::NoBacking)
}

/// Resolve object ref and, if mapper-ref-texture, latch mapping geometry + cache the entry.
///
/// Returns the mapping_id for mapper-ref-textures, or None.
pub fn resolve_mapper_ref_texture<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<u32> {
    let resource = match resolve_resource(state, host, task_id, ref_) {
        Ok(resource) => resource,
        Err(LadderRung::DescRead { .. }) => {
            // Keep this failure scoped to a confirmed mapper-ref-texture object. The
            // second lookup is only on the failed-construction path; successful
            // binds retrieve the retained resource without a guest read.
            if let Some(entry) = lookup_list_entry(state, host, task_id, ref_)
                .filter(|entry| entry.object_type == OBJECT_TYPE_MAPPER_REF_TEXTURE)
            {
                note_mapper_ref_texture_fail(
                    task_id,
                    ref_,
                    crate::observe::ladder_slug!("mapper_ref_texture", desc_read),
                    format!(
                        "mapper_ref_texture_resolve_fail reason=mapper_ref_texture_desc_read task={task_id} ref={ref_} obj_type={} desc_gva={:#x} desc_len={}",
                        entry.object_type, entry.descriptor_gva, entry.descriptor_length
                    ),
                );
            }
            return None;
        }
        Err(_) => return None,
    };
    resolve_mapper_ref_texture_resource(state, task_id, ref_, &resource)
}

/// Resolve an already-retained mapper-ref-texture resource to its mapping.
///
/// Draw preparation resolves each bound reference once and threads the
/// resulting object through all of its consumers. Keeping this half separate
/// prevents the mapper-ref-texture branch from looking the same reference up again and
/// reparsing immutable construction bytes on every bind.
pub fn resolve_mapper_ref_texture_resource(
    state: &mut DeviceState,
    task_id: u32,
    ref_: u32,
    resource: &TaskResource,
) -> Option<u32> {
    let entry = resource.entry;
    let desc = &resource.descriptor;
    if entry.object_type != OBJECT_TYPE_MAPPER_REF_TEXTURE {
        // Legitimate: this ref is a different object type, not a texture. Normal
        // control flow (resolve_mapper_ref_textures skips it) — never a failure.
        return None;
    }
    if let Some(mapping_id) = resource.registered_mapper_ref_texture_mapping() {
        return Some(mapping_id);
    }
    // Record the ref as live so the explicit delete path retires its associated
    // host-side content as well as the retained resource object.
    let _ = state.insert_object(task_id, ref_);
    let mapping_id = match resource.decoded() {
        Ok(crate::runtime::decode::resource::Descriptor::IOSurfaceTexture {
            mapping_id,
            pixel_format,
            width,
            height,
            ..
        }) => {
            if !texture::register_mapper_ref_texture_geom(
                state,
                *mapping_id,
                *width,
                *height,
                *pixel_format,
            ) {
                note_mapper_ref_texture_fail(
                    task_id,
                    ref_,
                    "mapper_ref_texture_register",
                    format!(
                        "mapper_ref_texture_resolve_fail reason=mapper_ref_texture_register task={task_id} ref={ref_} desc_len={}",
                        desc.len()
                    ),
                );
                return None;
            }
            *mapping_id
        }
        // Preserve the older headerless decoder as a compatibility rung for
        // descriptors the total object decoder cannot name. This is a cold
        // refusal path; successfully constructed resources take the typed arm.
        Err(_) => {
            if !texture::register_from_descriptor_bytes(state, OBJECT_TYPE_MAPPER_REF_TEXTURE, desc)
            {
                note_mapper_ref_texture_fail(
                    task_id,
                    ref_,
                    "mapper_ref_texture_register",
                    format!(
                        "mapper_ref_texture_resolve_fail reason=mapper_ref_texture_register task={task_id} ref={ref_} desc_len={}",
                        desc.len()
                    ),
                );
                return None;
            }
            u32::from_le_bytes(desc.get(..4)?.try_into().ok()?)
        }
        Ok(_) => return None,
    };
    if mapping_id == 0 {
        // Defensive for a compatibility decoder that ever accepts the sentinel
        // mapping id without registering it.
        note_mapper_ref_texture_fail(
            task_id,
            ref_,
            "mapper_ref_texture_mapping_zero",
            format!(
                "mapper_ref_texture_resolve_fail reason=mapper_ref_texture_mapping_zero task={task_id} ref={ref_} desc_len={}",
                desc.len()
            ),
        );
        return None;
    }
    state.texture_to_mapping.insert((task_id, ref_), mapping_id);
    let mapping_id = resource.register_mapper_ref_texture_mapping(mapping_id);
    // Resolved: re-arm so a later genuine failure on this ref logs again.
    clear_mapper_ref_texture_fail(task_id, ref_);
    Some(mapping_id)
}

/// Make the physical backing of a retained texture bindable.
///
/// A texture bind does not revalidate immutable surface construction input.
/// The guest announces a physical re-point explicitly; that path clears the
/// mapping's page entries and bumps its generation. Consequently a warm bind
/// only checks the retained mapping, while the first bind and the first bind
/// after a re-point rebuild the backing through the full resolver.
pub fn ensure_surface_for_texture_bind<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
) -> bool {
    let ready = state
        .mappings
        .get(&surface_id)
        .is_some_and(|m| m.mapped && !m.page_entries.is_empty() && m.has_geom);
    ready || ensure_surface_for_present(state, host, surface_id)
}

/// The detail line a refused page walk reports.
///
/// # Why the walk status is on it
///
/// `reims_vgpu_paging::resolve::ResolveStatus` distinguishes every check
/// in the guest page-table walk and has done since it was written; this site
/// collapsed all of them into the single word `translate`. Two refusals with
/// opposite remedies were therefore indistinguishable in the log: a leaf PTE the
/// guest has not filled in yet (`zero-pfn` — the surface is mid-map, and the
/// next frame resolves it) and a task root this device could not read at all
/// (`no-directory`, `root(...)` — the walk is aimed at the wrong table and
/// waiting will never help).
///
/// The distinction had to be reconstructed by hand for a whole A/B's worth of
/// refusals, by matching each against later attaches of the same surface id and
/// inferring which it had been. `walk=` states it outright.
///
/// Pure and separate from the emit so the composition is testable: the always-on
/// sink has no in-memory capture, so a test can only reach this line by building
/// it.
fn backing_translate_fail_detail(
    surface_id: u32,
    task_id: u32,
    page: u64,
    page_count: u64,
    gva: u64,
    walk: &str,
) -> String {
    format!(
        "backing_fail reason=translate sid={surface_id} task={task_id} \
         page={page}/{page_count} gva={gva:#x} walk=[{walk}] \
         (no translation in this task; not substituting the GVA)"
    )
}

/// Apply a decoded backing record as page-table backing for `surface_id`.
///
/// `backing_pfn` is a GPU-VA page (same source as normal textures). Translate
/// each consecutive GVA page through the task page table into GPA page entries
/// the scanout path already understands.
fn apply_backing<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    surface_id: u32,
    surf: &BackingRecord,
) -> bool {
    if !crate::model::is_mapping_id(surface_id) {
        defer_backing_fail(
            surface_id,
            "sid_zero",
            None,
            format!(
                "backing_fail reason=sid_zero sid={surface_id} task={task_id} \
                 (0 is the unbound-mapping sentinel; backing it would store pixels \
                 no attachment reader can address)"
            ),
        );
        return false;
    }
    let page_shift = state.page_shift;
    let page_size = page_size_of(page_shift);
    if page_size == 0 {
        defer_backing_fail(
            surface_id,
            "page_size_zero",
            None,
            format!("backing_fail reason=page_size_zero sid={surface_id} task={task_id} page_shift={page_shift}"),
        );
        return false;
    }
    // The backing base this attempt is about. It identifies the refusal for
    // `clear_backing_fail`, which is what lets a later clean attach on the *same*
    // backing be recognised as a recovery — surface ids recycle, addresses do
    // not.
    let backing_base_gva = (surf.backing_pfn as u64) << page_shift;
    let page_count = ((surf.length.saturating_sub(1)) / page_size) + 1;
    // No host MiB budget: page count follows guest `surf.length` only.
    // Fail if zero or not host-addressable as a page-entry vector.
    if page_count == 0 || crate::runtime::draw::host_alloc_len(page_count).is_none() {
        defer_backing_fail(
            surface_id,
            "page_count_oob",
            Some(backing_base_gva),
            format!(
                "backing_fail reason=page_count_oob sid={surface_id} task={task_id} len={:#x} page_count={page_count}",
                surf.length
            ),
        );
        return false;
    }
    let task = match state.tasks.get(task_id) {
        Some(t) if t.active => t,
        _ => {
            defer_backing_fail(
                surface_id,
                "task_inactive",
                Some(backing_base_gva),
                format!("backing_fail reason=task_inactive sid={surface_id} task={task_id}"),
            );
            return false;
        }
    };

    // Contract: backing_pfn is getGPUVirtualAddress>>page_shift (GPU-VA page).
    // Translate each consecutive GVA page through the task directory.
    //
    // A failed walk is not an address. The device used to substitute the guest
    // *virtual* address as a guest *physical* one whenever `read_gpa` could
    // touch it, but that probe asks "is this RAM", which nearly all of low
    // guest memory answers yes to. Two things follow, and the second is why
    // this refuses rather than guesses harder.
    //
    // The fabricated PFN goes into `m.page_entries`, which is the address list
    // every later reader and writer resolves through, so a guess aims real
    // pixel writes at memory the guest allocated for something else — and it
    // stays there, because the guess is cached as the surface's backing.
    //
    // What refusing buys, measured, is a retry. On boot 20260731-192622 both
    // refusals were followed by a full real-walk resolve of the same surface on
    // the same task within one or two frames: the guest had not finished
    // mapping the backing when the device first asked. The callers are
    // per-frame (scanout, bind, draw), so re-asking is already the shape of the
    // code; the guess was standing in for an answer about to be available.
    //
    // Refusing also lets the task search do its job, which a guess ended.
    // `apply_backing` returning `true` stops the loop in
    // `resolve_backing_ex`, and task 0 is probed first, so a guess made
    // task 0 claim surfaces it could not translate. That path is covered by
    // `the_task_search_reaches_the_owner_when_task_zero_cannot_translate`; it
    // has not been observed on the rig, where every attach resolves on task 0.
    let mut entries = Vec::with_capacity(page_count as usize);
    let mut gva_hits = 0u32;
    for i in 0..page_count {
        let gva = ((surf.backing_pfn as u64) + i) << page_shift;
        let Some(gpa) = gva_mem::translate_task_gva(host, task, gva, page_shift) else {
            crate::runtime::drain::note_store_route("backing_translate_refused");
            defer_backing_fail(
                surface_id,
                "translate",
                Some(backing_base_gva),
                backing_translate_fail_detail(
                    surface_id,
                    task_id,
                    i,
                    page_count,
                    gva,
                    &gva_mem::diagnose_task_slot(host, task, task_id, gva, page_shift),
                ),
            );
            return false;
        };
        gva_hits = gva_hits.saturating_add(1);
        let pfn = gpa >> page_shift;
        if pfn > u32::MAX as u64 {
            defer_backing_fail(
                surface_id,
                "pfn_oob",
            Some(backing_base_gva),
                format!("backing_fail reason=pfn_oob sid={surface_id} task={task_id} page={i}/{page_count} gpa={gpa:#x} pfn={pfn:#x}"),
            );
            return false;
        }
        let entry = ((pfn as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        // Sanity: entry_gpa must round-trip.
        if entry_gpa_shift(entry, page_shift) != Some(gpa & !(page_size - 1)) {
            defer_backing_fail(
                surface_id,
                "entry_roundtrip",
            Some(backing_base_gva),
                format!("backing_fail reason=entry_roundtrip sid={surface_id} task={task_id} page={i}/{page_count} gpa={gpa:#x} entry={entry:#x}"),
            );
            return false;
        }
        entries.push(entry);
    }
    // Bring-up probe once per surface_id (first attach).
    let first_attach = state
        .mappings
        .get(&surface_id)
        .map(|m| m.page_entries.is_empty())
        .unwrap_or(true);
    if first_attach && page_count >= 1 {
        let g0 = entry_gpa_shift(entries[0], page_shift).unwrap_or(0);
        // Three fields that used to ride on this line are gone, all of them
        // probe residue rather than census:
        //
        // - `sample0_nz`: a 16-byte `read_gpa` of the first backing page, per
        //   first attach, to count non-zero bytes. It fed no decision, and it
        //   read `0/16` on every one of the 131 attaches across two driven
        //   boots — a content sniff answering a bring-up question that has been
        //   answered.
        // - `plane0_bytes` and its `bpe0`: where the wire did not state
        //   bytes-per-element, a four-branch ladder guessed one from the format
        //   so a log field could be filled. Deriving a number nothing consumes
        //   is how a guess becomes a rule later.
        // - `gpa1`/`gpa2`: the second and third pages, sampled for no stated
        //   reason. `n` and `gva_hits` already say how many pages resolved.
        //
        // What remains is the census the comment below defends: the identity of
        // the backing, so a refusal and a later resolve can be matched.
        //
        // Bring-up census (dims/fmt), not a drop — the genuine
        // backing failures route through note_backing_fail with reason=. On the
        // always-on `off()` sink, not `fail()`: under surface recycling this
        // "first attach" re-fires per recycle (page_entries cleared by the
        // teardown), so on fail() it floods the curated real-error view (~4k
        // lines under a continuously-animating app, burying genuine failures).
        // `gva0` is what the refusal line above prints as `gva=`, so a refusal
        // and a later resolve can be matched by the *backing* they name. Matching
        // them by `sid` alone is unsound: surface ids recycle within a boot and
        // across geometries — sid 145 was a 15x622 scrollbar at t=332488 and a
        // 1225x512 tile 2.8 s later — so "the same surface resolved a frame
        // later" can be a different surface wearing the same id.
        let gva0 = (surf.backing_pfn as u64) << page_shift;
        crate::observe::off(format!(
            "backing pages sid={surface_id} task={task_id} n={page_count} gva_hits={gva_hits} gva0={gva0:#x} gpa0={g0:#x} w={} h={} bpr={} len={:#x} fmt={:#x} planes={} multi={}",
            surf.width,
            surf.height,
            surf.bytes_per_row,
            surf.length,
            surf.pixel_format,
            surf.plane_count,
            backing_is_multiplanar(surf) as u8
        ));
    }

    if !state.map_surface(surface_id) {
        defer_backing_fail(
            surface_id,
            "map_surface",
            Some(backing_base_gva),
            format!(
                "backing_fail reason=map_surface sid={surface_id} task={task_id} n={page_count}"
            ),
        );
        return false;
    }
    // Device desc from backing wire only (single- or multi-plane). No BGRA invent.
    let device_desc = synthesize_device_desc_from_backing(surf);

    let state_page_shift = state.page_shift;
    if let Some(m) = state.mappings.get_mut(&surface_id) {
        // `map_surface` above stashed the prior bindings as the incarnation
        // fingerprint (the notify-vs-eager-resolve rule): compare the fresh
        // plan against it — identical pages are the SAME incarnation (no
        // bump; deferred windows and the resident survive), a change is the
        // recycled-mid rule (bump; stale residents/views must never survive).
        let prior = m
            .condemned_entries
            .take()
            .unwrap_or_else(|| std::mem::take(&mut m.page_entries));
        let changed = prior != entries;
        let replaced = !prior.is_empty() && changed;
        if changed {
            crate::model::DeviceState::bump_map_generation(m);
        }
        if replaced {
            // Recycled-mid backing-refresh census — not a drop. Off the curated
            // fail() view: per-recycle under animation churn it floods the
            // real-error view, at 793 lines in one measured boot.
            crate::observe::off(format!(
                "backing_pages_refreshed sid={surface_id} task={task_id} n={} map_gen={}",
                entries.len(),
                m.map_generation
            ));
            // Present evidence needs no prune here: this branch only runs when
            // the plan changed, which bumped `map_generation` just above, and
            // the evidence is stamped with the incarnation that recorded it.
            // Pruning it unconditionally is what the identical-plan path used
            // to do via `map_surface`, and that demoted a surface the compare
            // had just called the SAME incarnation.
        }
        // The guest-physical footprint this incarnation authorises us to write.
        // See `mapper::entry_gpa_span`; this is the backing adoption site, and it
        // is the one that carried every span in the x86 log.
        //
        // That reading used to be stated as "the page list arrives here, the
        // mapper's own adoption stays silent". It could not have come out any
        // other way: both sites deduped through one `first_sight` namespace on
        // the same key, so this site claimed each footprint it reached first and
        // silenced its peer for that footprint. The namespaces are now
        // `mapper::SPAN_SEEN_TYPE4` and `SPAN_SEEN_MAPPER`, so each site's
        // silence is its own.
        //
        // No `changed=` field, though `changed` is in scope and the mapper's
        // peer emitter prints its own. Here it could only ever be 1: the dedup
        // is `first_sight` on the span, and an unchanged plan has by definition
        // the same span as the plan before it, so the unchanged case is filtered
        // out before reaching this line. The one way to arrive here unchanged is
        // the first visit for a surface, and there `prior` is empty, which makes
        // `changed` true. The mapper's copy is not in that position — its
        // reprieve path can repopulate an emptied entry list with no change.
        if let Some((lo, hi)) = crate::runtime::mapper::entry_gpa_span(&entries, state_page_shift) {
            let key =
                crate::runtime::mapper::span_first_sight_key(surface_id, lo, hi, state_page_shift);
            if crate::observe::first_sight(crate::runtime::mapper::SPAN_SEEN_TYPE4, key) {
                crate::observe::off(format!(
                    "mapping_gpa_span mid={surface_id} gen={} pages={} src=backing \
                     lo={lo:#x} hi={:#x} pn_lo={:#x} pn_hi={:#x}",
                    m.map_generation,
                    entries.len(),
                    hi + (1u64 << state_page_shift),
                    lo >> state_page_shift,
                    hi >> state_page_shift,
                ));
            }
        }
        m.page_entries = entries;
        m.mapped = true;
        m.page_table_kva = 0;
        m.device_desc = device_desc;
        // Latched with the list rather than near it: this records the walk that
        // produced the entries above, so a later reader can repeat that walk —
        // and find out whether the entries still name the guest's memory —
        // without repeating the object search that found the surface. Written at
        // the assignment so the two cannot be updated independently.
        m.backing_walk = Some(crate::model::BackingWalk {
            task_id,
            backing_pfn: surf.backing_pfn,
            map_generation: m.map_generation,
        });
        // Contiguous view must be rebuilt.
        let mut retired_import = None;
        if m.contig_ptr != 0 {
            state.retired_views.push((m.contig_ptr, m.contig_len));
            m.contig_ptr = 0;
            m.contig_len = 0;
            m.contig_footprint = None;
            retired_import = m.contig_import.take().map(|import| {
                import.retire();
                import.id()
            });
        }
        if let Some(import) = retired_import {
            state.retired_guest_imports.push(import);
        }
    }

    // Dims come from plane 0 for a multi-plane surface, which is bookkeeping;
    // the format is `latched_mapping_format`'s, which is a contract.
    if surf.width > 0 && surf.height > 0 {
        let _ = state.set_mapping_geom(
            surface_id,
            surf.width,
            surf.height,
            latched_mapping_format(surf),
        );
    }

    // Backing built cleanly — re-arm the fail latch so a later genuine failure
    // on this surface (flapping backing) is logged again, and report the earlier
    // refusal for *this* backing as the recovery it turned out to be.
    clear_backing_fail(surface_id, backing_base_gva);
    true
}

/// Resolve present `surface_id` to backing pages + geometry.
///
/// Scans active tasks: object-list slot `surface_id` must be backing (heap is
/// indexed by IOSurface surface ID). Returns true when pages were latched.
pub fn resolve_backing<M: HostMemory>(state: &mut DeviceState, host: &M, surface_id: u32) -> bool {
    resolve_backing_ex(state, host, surface_id, false)
}

/// Like [`resolve_backing`] but always re-reads the object list / PT.
pub fn resolve_backing_force<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
) -> bool {
    resolve_backing_ex(state, host, surface_id, true)
}

/// Latch the task that owns `surface_id` as its backing so the next
/// present-path scan tries it right after task 0.
fn record_backing_owner(state: &mut DeviceState, surface_id: u32, task_id: u32) {
    if let Some(m) = state.mappings.get_mut(&surface_id) {
        m.owner_task_hint = task_id;
    }
}

/// Apply `CmdReplacePhysical` (`0x3c`): the guest re-pointed this resource's
/// GPU-VA range at different physical pages.
///
/// The packet is the announcement, and it is the only one there is. The guest
/// releases the range, rewires the pages, re-commits the *same* GPU-VA with the
/// new PFNs, and then emits one of these per attached resource. Nothing else on
/// the wire says the translation moved — GVA, surface id, geometry and length
/// are all unchanged — so a cached GPA list not dropped here stays trusted while
/// naming pages that now back something else.
///
/// Dropping the list is the whole action. It bumps `map_generation`, which is
/// what retires the [`crate::model::BackingWalk`] latch and the resident/deferred
/// state keyed on that incarnation, and the next resolve re-walks the page table
/// the guest has already rewritten.
///
/// The list is *cleared* rather than condemned, which is where this packet parts
/// from `DeleteIOSurfaceBacking2` even though the two look alike. A delete may
/// be trailing a recycled id whose slot already carries a live surface, so it
/// stashes a fingerprint and lets the next resolve decide. A re-point carries no
/// such ambiguity — by its own contract the PFNs under this GPU-VA have already
/// changed — and condemning would leave `map_generation` unmoved, so a resident
/// gathered from the old pages would stay eligible until some later resolve
/// disqualified it. Clearing fails closed on the packet that says the pages
/// moved; on this rail that is the direction every ambiguity resolves toward.
///
/// # The packet names a task-local resource
///
/// The task and object fields are obtained from the same resource object. The
/// object is therefore resolved in the task's namespace; its integer must never
/// be tried as a global mapping id first. Surface ids and resource refs overlap
/// numerically. Treating a ref in task B as mapping `n` can retire the pages of
/// an unrelated backing record `n` owned by task A, after which a compositor
/// draw sees an unbound texture until that surface happens to be mapped again.
///
/// A latched backing walk is the exact provenance of a direct surface mapping.
/// Mapper-ref-texture resources carry the task/ref-to-mapping association established at
/// construction. Both routes require the packet's task; a bare integer match is
/// not ownership.
///
/// What the *unreached* re-points turn out to be is measured rather than
/// assumed, and it is mostly benign: 44 of 46 on a driven boot were
/// `replace_physical_unmapped_no_state` — this device holds nothing at all for
/// the object, so the first resolve of that ref reads the page table the guest
/// has already rewritten. Two held a ref-keyed host copy.
/// [`note_replace_physical_unmapped_after_invalidation`] is what separates
/// those, and it exists
/// because a bare "reached nothing" cannot: an announcement with nothing to
/// apply it to and an announcement that missed a live host copy read the same.
/// The storage a re-pointed reference names *now*, in the lifecycle owner's
/// vocabulary.
///
/// **Asked after the re-point, and that is the whole reason it is a function
/// and not a resolver.** `reims_vgpu_core::lifecycle::ResolveRefusal::NeedsStorage`
/// says a re-point's operation names pages that are not on the wire: the guest
/// wired them itself and re-committed at the same GPU-VA, so the packet carries
/// a task and a ref and nothing about the new memory. A resolver consulted while
/// the operation was being built would answer with the *old* identity, because
/// the storage incarnation has not moved yet — and one that moved it would be a
/// mutation from the one function in the join that performs none.
///
/// So the caller applies this device's re-point first, and then asks what the
/// reference names. That is the moment the answer is the new one.
///
/// # Why the failure is an enum and not a `None`
///
/// It used to be one `None` covering six unrelated facts, and the caller
/// counted it as `replace_physical_storage_undescribable`. A driven macos-15
/// boot put 34 of 204 re-points in that bucket, and the bucket could not say
/// whether they were re-points of references that name no bytes at all — for
/// which telling the model nothing is correct — or re-points of real storage
/// the model was never told had moved, which is content authority left on the
/// old pages. Those are a non-event and a defect, and one counter cannot be
/// read as either. [`RepointStorageRefusal`] is the split.
pub fn repointed_storage<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    obj_ref: u32,
) -> Result<
    (
        reims_vgpu_core::identity::ResourceId,
        reims_vgpu_core::access::BackingId,
        reims_vgpu_core::access::ByteRange,
    ),
    RepointStorageRefusal,
> {
    let name =
        name_resource(state, host, task_id, obj_ref).ok_or(RepointStorageRefusal::Unnamed)?;
    let entry = lookup_list_entry(state, host, task_id, obj_ref)
        .ok_or(RepointStorageRefusal::NoListEntry)?;
    let descriptor = read_descriptor(state, host, task_id, &entry)
        .ok_or(RepointStorageRefusal::DescriptorUnread)?;
    match declared_storage(state, task_id, obj_ref, &entry, &descriptor) {
        Ok(reims_vgpu_core::lifecycle::Storage::Dedicated { backing, extent }) => {
            Ok((name, backing, extent))
        }
        // A placement has no physical of its own to re-point — the owner
        // refuses that by name — and `NoBytes` is a reference with nothing to
        // move. Neither is an operation, and both are named so that a boot can
        // tell them from storage that moved unheard.
        Ok(reims_vgpu_core::lifecycle::Storage::Placed { .. }) => {
            Err(RepointStorageRefusal::Placed)
        }
        Ok(reims_vgpu_core::lifecycle::Storage::NoBytes) => {
            Err(RepointStorageRefusal::NoBytes {
                object_type: entry.object_type,
                // Asked because a re-point whose reference names no bytes in
                // the object list, while the *same integer* names a live
                // mapping surface in this task, is not a no-op: it is storage
                // reached through the other namespace, and the object list
                // cannot describe it. The two namespaces overlap numerically,
                // so this is recorded as a reading and not acted on — what
                // makes it one fact rather than a collision is the type at the
                // list entry, which travels with it.
                mapping_names_a_surface: mapping_backing_id(state, obj_ref).is_ok(),
            })
        }
        Err(refusal) => Err(RepointStorageRefusal::Refused(refusal)),
    }
}

/// Why a re-pointed reference names no storage operation the model can be told
/// about.
///
/// Each variant carries its own census route, because the whole point of the
/// enum is that the routes are read separately: a boot where every one is
/// `NoBytes` has lost nothing, and a boot where they are `Refused` has one
/// silent stale-content hazard per count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepointStorageRefusal {
    /// The reference resolves to no resource name — no task space, or no live
    /// slot. There is nothing for an operation to be about.
    Unnamed,
    /// The name exists and the guest's object list holds no entry at it.
    NoListEntry,
    /// The entry exists and its descriptor did not read out of guest memory.
    DescriptorUnread,
    /// The reference names a heap window. A placement owns no physical of its
    /// own, so a re-point of one is the heap's event, not this reference's.
    Placed,
    /// The reference names an object with no bytes at all — a sampler, a
    /// function, a view. A re-point announcement against one is a no-op, and
    /// the type is carried so that a boot can say which kinds this guest
    /// re-points and whether the answer is plausible for them.
    NoBytes {
        object_type: u8,
        /// Whether the same integer names a live mapping surface in this task.
        ///
        /// If it does, the announcement is about storage the mapping namespace
        /// has and the object list does not, and "no bytes" is the object
        /// list's answer rather than the resource's.
        mapping_names_a_surface: bool,
    },
    /// The reference names storage this device could not describe. This is the
    /// one that costs: the pages moved and the model was not told.
    Refused(StorageRefusal),
}

impl RepointStorageRefusal {
    /// The census route this refusal is counted on.
    #[must_use]
    pub const fn route(self) -> &'static str {
        match self {
            Self::Unnamed => "replace_physical_storage_unnamed",
            Self::NoListEntry => "replace_physical_storage_no_list_entry",
            Self::DescriptorUnread => "replace_physical_storage_descriptor_unread",
            Self::Placed => "replace_physical_storage_placed",
            Self::NoBytes {
                mapping_names_a_surface: true,
                ..
            } => "replace_physical_storage_no_bytes_but_mapping_names_a_surface",
            Self::NoBytes { .. } => self.no_bytes_kind_route(),
            Self::Refused(_) => self.refused_route(),
        }
    }

    /// The kind at the object-list entry, whatever the mapping namespace says.
    ///
    /// [`Self::route`]'s no-bytes arms are shadowed by the mapping arm, and the
    /// mapping arm is exactly the case where the type pair decides whether the
    /// two namespaces are naming one thing or the integer merely collided.
    /// So the caller counts both, and only this one answers "what is the guest's
    /// own list holding at that reference".
    #[must_use]
    pub const fn no_bytes_kind_route(self) -> &'static str {
        match self {
            Self::NoBytes {
                object_type: OBJECT_TYPE_FUNCTION,
                ..
            } => "replace_physical_storage_no_bytes_function",
            Self::NoBytes {
                object_type: OBJECT_TYPE_SERIALIZER_OBJECT,
                ..
            } => "replace_physical_storage_no_bytes_serializer",
            Self::NoBytes {
                object_type: OBJECT_TYPE_TEXTURE_VIEW,
                ..
            } => "replace_physical_storage_no_bytes_texture_view",
            Self::NoBytes {
                object_type: OBJECT_TYPE_REF_TEXTURE,
                ..
            } => "replace_physical_storage_no_bytes_ref_texture",
            Self::NoBytes {
                object_type: OBJECT_TYPE_BACKING,
                ..
            } => "replace_physical_storage_no_bytes_backing",
            Self::NoBytes {
                object_type: OBJECT_TYPE_MAPPER_REF_TEXTURE,
                ..
            } => "replace_physical_storage_no_bytes_mapper_ref_texture",
            Self::NoBytes { .. } => "replace_physical_storage_no_bytes_other",
            _ => "replace_physical_storage_kind_not_asked",
        }
    }

    const fn refused_route(self) -> &'static str {
        match self {
            Self::Refused(StorageRefusal::ExtentUnrecovered { .. }) => {
                "replace_physical_storage_extent_unrecovered"
            }
            Self::Refused(StorageRefusal::Backing(BackingIdRefusal::Unresolved(_))) => {
                "replace_physical_storage_backing_unresolved"
            }
            Self::Refused(StorageRefusal::Backing(BackingIdRefusal::ThroughMapping { .. })) => {
                "replace_physical_storage_mapping_names_no_surface"
            }
            Self::Refused(StorageRefusal::Backing(BackingIdRefusal::HeapPlaced)) => {
                "replace_physical_storage_heap_placed"
            }
            Self::Refused(StorageRefusal::Backing(BackingIdRefusal::NamesNoStorage { .. })) => {
                "replace_physical_storage_names_no_storage"
            }
            _ => "replace_physical_storage_refusal_not_asked",
        }
    }

    /// Whether this refusal leaves the model's content authority on pages the
    /// guest has already re-pointed.
    ///
    /// The four resolution and no-bytes arms do not: there is no storage for
    /// the model to hold authority over. [`Self::Refused`] does — the storage
    /// is real and only this device's vocabulary for it is missing.
    #[must_use]
    pub const fn leaves_authority_stale(self) -> bool {
        matches!(self, Self::Refused(_))
    }
}

pub fn replace_physical<H: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    task_id: u32,
    object_id: u32,
) {
    let target = state
        .mappings
        .get(&object_id)
        .and_then(|mapping| {
            mapping
                .backing_walk
                .filter(|walk| walk.task_id == task_id)
                .map(|_| object_id)
        })
        .or_else(|| {
            state
                .texture_to_mapping
                .get(&(task_id, object_id))
                .copied()
                .filter(|mid| state.mappings.contains_key(mid))
                .inspect(|_| crate::runtime::drain::note_store_route("replace_physical_routed_ref"))
        });

    // A re-point changes every host representation of this resource, whether
    // or not it also owns a mapping. Mapping and ref-keyed caches are different
    // indexes over the same resource lifetime; reaching one does not excuse
    // leaving the other stale.
    let (texture_cache, linear_cache) = state.invalidate_object_host_copies(task_id, object_id);
    // The packet's own contract is that the PFNs under this window have already
    // changed, so it says so whether or not this device was holding anything to
    // invalidate — the announcement is about the guest's memory, not about our
    // caches. Ahead of the unreached arm's early return for that reason.
    //
    // The packet names a reference and the incarnation is kept on the window,
    // so the reference is resolved to one first. Both are asked, cache before
    // guest, because they answer different questions: the cache holds the
    // window *accepted work was planned against*, which is the one that must
    // stop comparing equal, while the guest's list holds whatever the
    // reference names now. A reference this device never constructed has only
    // the second.
    match repointed_window(state, host, task_id, object_id) {
        Some(base) => {
            state.bump_storage_incarnation(task_id, base);
            invalidate_window_peers(state, host, task_id, object_id, base);
        }
        // No window. That is only a loss for storage this device reaches *by
        // address*: an object that owns a mapping carries its incarnation in
        // that mapping's generation instead, which the arm below moves, and
        // the two counters are the two ways storage is reached rather than two
        // answers about one. A driven macos-15 boot puts every one of these in
        // the second class — seven re-points, all in the accelerator's kernel
        // task, naming surfaces registered there — so a line that called them
        // losses would be seven false findings a boot.
        None if target.is_some() => {
            crate::runtime::drain::note_store_route("replace_physical_window_via_mapping");
        }
        // Neither address nor mapping. Now it is a real loss — a later identity
        // for these pages will compare equal to the one accepted work holds —
        // and it is named rather than left to a counter that reads the same as
        // a boot with no re-points at all.
        None => {
            crate::runtime::drain::note_store_route("replace_physical_window_unknown");
            if crate::observe::first_sight(
                "replace_physical_window_unknown",
                u64::from(task_id) << 32 | u64::from(object_id),
            ) {
                crate::observe::fail(format!(
                    "replace_physical_window_unknown task={task_id} object={object_id} \
                     (neither this device's construction of this reference nor the \
                     guest's own list gives it a guest-VA window, so the incarnation \
                     of the pages the packet says have moved was not advanced and a \
                     later identity for them will compare equal to the one accepted \
                     work holds)"
                ));
            }
        }
    }

    let Some(target) = target else {
        note_replace_physical_unmapped_after_invalidation(
            state,
            host,
            task_id,
            object_id,
            texture_cache,
            linear_cache,
        );
        return;
    };
    // A deferred window still owed on this mapping is riding the page plan the
    // guest has just replaced. The generation bump below would refuse it anyway,
    // so taking it here changes no outcome — it changes whether the loss has a
    // name. `drop_windows` reports each one against this packet instead of
    // letting it disappear into a generation mismatch nothing attributes.
    let had = state.invalidate_mapping_pages(target);
    crate::runtime::drain::note_store_route(if had {
        "replace_physical_dropped"
    } else {
        "replace_physical_no_pages"
    });
    // The views the invalidation retired name pages the guest is re-pointing.
    // Handing them back now, rather than at the next poll, keeps this device from
    // holding a host mapping over memory the guest has already rewired — the same
    // reason the trailing delete flushes here.
    crate::runtime::mapper::flush_retired_views(state, host);
    if had {
        crate::observe::off(format!(
            "replace_physical task={task_id} object={object_id} mid={target} \
             (guest re-pointed the backing; cached page list dropped)"
        ));
    }
}

/// Say what a re-point named when no mapping belongs to the resource, and whether this
/// device is holding anything else for it.
///
/// A mapping is not the only place a resource's bytes can be cached. The normal-texture
/// rails key their host copies by object-list ref (`host_texture_surfaces`,
/// `host_linear_textures`) rather than by mapping id, and neither carries a page
/// list to notice a move — so "no mapping for this task-local resource" does
/// not settle whether the re-point had anything to invalidate. It only settles
/// that the *mapping* rail had nothing.
///
/// The counters split three ways. `_unmapped_no_state` is a re-point of a
/// resource this device holds nothing for, which is genuinely a no-op — the
/// first resolve of that ref will read the page table the guest has already
/// rewritten. `_unmapped_texture_invalidated` and `_unmapped_linear_invalidated`
/// are the re-points that reached a live host copy, and they now name a repair
/// rather than a loss.
///
/// They used to be `_unmapped_texture_cache` / `_unmapped_linear_cache` and to
/// count only. A host copy whose pages the guest has re-pointed is a copy of
/// memory that is no longer the object's, and leaving it trusted served the
/// guest a stale frame from bytes it had already rewired — with nothing refusing
/// and nothing to read but these two counters. That is what they were added to
/// measure, and they measured it. Driven x86/PCI boot, `web-content-probe -n 10
/// --churn 1`, run on to a settled desktop:
///
/// ```text
///   replace_physical_unknown_object                 56
///   replace_physical_unmapped_no_state              43
///   replace_physical_unmapped_texture_invalidated   13
///   replace_physical_unmapped_linear_invalidated     0
/// ```
///
/// So roughly a quarter of the re-points reaching this branch find a live host
/// copy. The class is intermittent across boots — an earlier one on the same
/// image and probe read 8 — because it depends on which objects hold a host copy
/// at the moment the packet arrives, and the events cluster late in a boot
/// rather than during the probe.
///
/// **These are `store_routes` counters and they are per-window: sum the samples,
/// do not take the maximum.** The series descends across a boot
/// (`unknown_object` = 30, 12, 8), so a `sort -n | tail -1` reads the busiest
/// window and calls it the total — which is how this measurement was first
/// misreported as 37/32/7/1. The check that catches it is the arithmetic:
/// `no_state + texture + linear` must equal `unknown_object`. It does above
/// (43 + 13 = 56); on the maxima it did not (32 + 7 + 1 vs 37).
///
/// Re-taken on the host-pointer-import tree, same probe, 54 windows summed:
/// `17 + 7 + 0 = 24 == 24`, alongside `replace_physical_dropped` 19 and
/// `_no_pages` 5. So the identity survives the guest-memory rail change, which
/// is the plan's §7 line for it.
///
/// **A window drag cannot check this.** A `window-drag-probe` boot reads all
/// four at zero, so the identity holds as `0 == 0` and proves nothing —
/// verifying it needs `web-content-probe --churn 1`, which is what produced
/// both readings above.
///
/// `invalidate_object_host_copies` is the discharge, and it is the same one
/// `delete_object` has always performed for the same two maps — the difference
/// being that a delete also unnames the object while a re-point only moves its
/// bytes. Kept fail-visible after the repair so the reliance stays measurable;
/// a rising count is now this device correctly following the guest, and its
/// disappearance would mean the packet stopped arriving, not that the bug was
/// fixed twice.
///
/// The guest's own object list supplies the type, which is the only authority on
/// what the id names: `lookup_list_entry` reads it at use time and this device
/// caches no copy of it. The type is on the line rather than in a counter
/// because the interesting reading is which types show up at all, and that is a
/// small set a boot enumerates in a handful of lines.
fn note_replace_physical_unmapped_after_invalidation<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    object_id: u32,
    texture_cache: bool,
    linear_cache: bool,
) {
    let object_type = lookup_list_entry(state, host, task_id, object_id).map(|e| e.object_type);
    crate::runtime::drain::note_store_route("replace_physical_unknown_object");
    if texture_cache {
        crate::runtime::drain::note_store_route("replace_physical_unmapped_texture_invalidated");
    }
    if linear_cache {
        crate::runtime::drain::note_store_route("replace_physical_unmapped_linear_invalidated");
    }
    if !texture_cache && !linear_cache {
        crate::runtime::drain::note_store_route("replace_physical_unmapped_no_state");
    }
    // Deduped per `(task, object)`: a compositor re-pointing the same resource
    // does it every frame, and the class is what matters here.
    if crate::observe::first_sight(
        "replace_physical_unknown_object",
        (u64::from(task_id) << 32) | u64::from(object_id),
    ) {
        let kind = object_type
            .map(|t| t.to_string())
            .unwrap_or_else(|| "absent".to_string());
        crate::observe::fail(format!(
            "replace_physical_unknown_object task={task_id} object={object_id} \
             obj_type={kind} tex_dropped={} lin_dropped={} \
             (no mapping belongs to this task-local resource; ref-keyed host copies dropped)",
            texture_cache as u8, linear_cache as u8
        ));
    }
}

/// The active tasks whose object list holds an `OBJECT_TYPE_BACKING` at slot
/// `surface_id` — every task the search could legitimately have stopped on.
///
/// `lookup_list_entry` already refuses an inactive task, an out-of-range slot
/// and an entry with no descriptor, so a task reaching the type test is one with
/// a real object at that slot.
fn backing_claimant_tasks<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    surface_id: u32,
) -> Vec<u32> {
    // The live ids, not a fixed range. This walked `0..256` while the task table
    // was an array; `lookup_list_entry` refuses an inactive task before reading
    // anything, so the ids in between were never claimants and the answer is
    // unchanged. The walk is now the size of the guest's task set.
    state
        .tasks
        .live_ids()
        .filter(|&task_id| {
            probe_list_entry(state, host, task_id, surface_id)
                .is_some_and(|e| e.object_type == OBJECT_TYPE_BACKING)
        })
        .collect()
}

/// Report how many active tasks claim `surface_id`, once per surface id.
///
/// The search below takes the first task that produces a translatable backing.
/// If two tasks can, probe order decides which of them the guest gets, and
/// nothing on the wire would say it chose wrong — there is no field to verify a
/// candidate against. The object-list entry is `[type | desc_len]` plus
/// `desc_gva` and carries no identity ([`decode_list_object_entry`]), and the
/// backing descriptor is fully consumed: its only undecoded span is the three
/// bytes at `0x11`, which read zero on every distinct shape a driven boot
/// produces (`backing_desc_shape … undecoded_nz=0`).
///
/// So the question the wire cannot answer directly is answered by counting
/// instead. A surface id only ever claimed by one task is a surface whose owner
/// probe order cannot have gotten wrong, whatever order it used.
///
/// The claim test is the object-list slot's type alone, not a descriptor read or
/// a translation: a task that lists a backing record at this slot is a task the
/// search could have stopped on. That keeps the sweep to one 12-byte guest read
/// per active task, and it is taken once per surface id — whether a surface id
/// is claimed twice is a property of the guest's allocation, not of this
/// resolve.
///
/// `claims=1` is the healthy reading and stays on the quiet channel. More than
/// one claimant is the case that makes the search's tie-break load-bearing, so
/// that one is a failure line naming the tasks involved.
fn note_backing_claimants<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    surface_id: u32,
    winner: u32,
) {
    if !crate::observe::first_sight("backing_claimants", surface_id as u64) {
        return;
    }
    let claimants = backing_claimant_tasks(state, host, surface_id);
    let line = format!(
        "backing_claimants sid={surface_id} winner={winner} claims={} tasks={claimants:?}",
        claimants.len()
    );
    if claimants.len() > 1 {
        crate::observe::fail(format!(
            "{line} (more than one task lists this surface id, so probe order \
             chose between them and no wire field can say it chose right)"
        ));
    } else {
        crate::observe::off(line);
    }
}

/// What the object-list slot a heap reference names actually holds.
///
/// Returned as well as reported so the reading can be asserted where a boot is
/// not available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapReference {
    /// The task lists nothing at that slot.
    Unlisted,
    /// It lists an object, of this type and descriptor length.
    Listed {
        object_type: u8,
        descriptor_length: u32,
    },
}

/// Why a guest object reference has no canonical backing identity.
///
/// Three of the four are a contract term this device does not have, named
/// rather than guessed; the fourth is a mapping that names no surface yet,
/// carried whole from the resolver that does have the term. `reims_vgpu_core::access::BackingId`'s own doc says why the
/// guess is not available: an identity that is wrong in the equal direction
/// hands storage back under a live reader, and one that is wrong in the
/// distinct direction drops a hazard edge. Neither failure has a symptom at
/// the point it is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackingIdRefusal {
    /// The reference names nothing in the task's object list, or its
    /// descriptor did not read.
    Unresolved(LadderRung),
    /// The object reaches its storage through a mapping — a mapper-ref texture
    /// — and that mapping names no live surface.
    ///
    /// The refusal is the *mapping's*, carried whole, and it is no longer this
    /// one's own: the descriptor names a mapping id, so the reference is
    /// followed to it and [`mapping_backing_id`] answers. It used to stop here
    /// on the type alone, which read as "this device does not have the term"
    /// when the term was one decode away.
    ///
    /// Following a decoded field is not one resolver answering two namespaces.
    /// The two entry points stay separate, keyed by the two `u32`s that overlap
    /// numerically and name unrelated things — a caller holding a mapping id
    /// asks [`mapping_backing_id`], a caller holding an object reference asks
    /// [`backing_id`] — and a caller that could pass either to one method would
    /// get a plausible number for the wrong question. That is what the split is
    /// for, and it survives the delegation.
    ///
    /// **A dual-plane texture is not one of these and used to be.** It is built
    /// from paging info in its own task like any normal texture, its two planes
    /// share the one object header's handle and allocation size, and
    /// `Descriptor::backing_window` now answers for it — one allocation, two
    /// extents.
    ThroughMapping {
        mapping_id: u32,
        refusal: MappingBackingRefusal,
    },
    /// The object is placed inside a heap, and a heap's extent is unrecovered.
    ///
    /// A placement names a heap reference and an offset. The heap's identity is
    /// settled — it is an object-list object like any other — but without its
    /// *length* nothing can say which two offsets in it overlap, so two
    /// placements sharing bytes would come out distinct. That is a dropped
    /// hazard edge, so the identity is refused until the extent is recovered.
    HeapPlaced,
    /// The object names no storage at all: a function, a serializer object, a
    /// texture view over another object's bytes.
    NamesNoStorage { object_type: u8 },
}

impl BackingIdRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unresolved(_) => "backing_id_unresolved",
            Self::ThroughMapping { .. } => "backing_id_mapping_names_no_surface",
            Self::HeapPlaced => "backing_id_heap_placed",
            Self::NamesNoStorage { .. } => "backing_id_names_no_storage",
        }
    }
}

/// The canonical backing identity of the storage a guest object reference
/// names.
///
/// # What makes this the identity and not a name
///
/// `reims_vgpu_core::access::BackingId` is what two resources sharing backing
/// must share, and the dependency compiler draws its hazard edges on it. So it
/// is derived from *where the storage is* and never from what it is called: two
/// references over one allocation arrive here at one number — which a driven
/// boot proved is not hypothetical, the compositor holds its scanout buffer
/// under two — and one reference across a physical replacement arrives at two,
/// because the pages changed.
///
/// The window is the allocation's, not the texel base's, so two textures placed
/// at different offsets in one allocation are one backing with two extents.
/// The extent is `reims_vgpu_core::access::ByteRange`'s and is not asked here.
///
/// # Errors
///
/// [`BackingIdRefusal`], one variant per contract term this device does not
/// have for the reference. None of them is a failure to try harder.
pub fn backing_id<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    obj_ref: u32,
) -> Result<reims_vgpu_core::access::BackingId, BackingIdRefusal> {
    let entry = lookup_list_entry(state, host, task_id, obj_ref)
        .ok_or(BackingIdRefusal::Unresolved(LadderRung::NoListEntry))?;
    let descriptor = read_descriptor(state, host, task_id, &entry).ok_or(
        BackingIdRefusal::Unresolved(LadderRung::DescRead {
            declared_len: entry.descriptor_length,
        }),
    )?;
    backing_id_of(state, task_id, obj_ref, &entry, &descriptor)
}

/// [`backing_id`] for a caller that has already read the list entry and the
/// descriptor.
///
/// The construction path holds both — it decodes the descriptor to build the
/// resource — so making it go back to guest memory for them would be two reads
/// of the same bytes and, worse, two chances for the identity and the resource
/// to describe different versions of the object.
///
/// # Errors
///
/// [`BackingIdRefusal`], exactly as [`backing_id`].
pub fn backing_id_of(
    state: &DeviceState,
    task_id: u32,
    obj_ref: u32,
    entry: &ListObjectEntry,
    descriptor: &[u8],
) -> Result<reims_vgpu_core::access::BackingId, BackingIdRefusal> {
    match backing_window(state.page_shift, entry, descriptor) {
        Some((base, _)) => Ok(state.backing_identity(task_id, base)),
        // One type's storage is somewhere else rather than not yet named, and
        // the rest are refusals whose variant says what would have to land for
        // them to stop being one. Both live in the classifier below, so the
        // answer is exhaustive over the types rather than a fallthrough.
        None => backing_id_without_window(state, obj_ref, entry, descriptor),
    }
}

/// Why a guest mapping's surface has no canonical backing identity.
///
/// Both arms are "the guest has not said there is a surface here yet", never
/// "this device could not work it out". Storage reached through a mapping needs
/// no descriptor read and no page walk to be *identified* — the mapping id and
/// its generation are the identity — so there is no ladder of ways to fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappingBackingRefusal {
    /// This device holds no entry at that mapping id.
    Unlisted,
    /// An entry exists but the guest has never mapped a surface into it.
    ///
    /// Entries are created by any of several statements about a slot — a
    /// geometry declaration, a validity quad — and one that has never carried a
    /// MAP or an attach names no storage to identify. Minting for it would hand
    /// an identity to a slot number rather than to storage, which is the
    /// name-derived id `reims_vgpu_core::access::BackingId` rules out.
    Unmapped,
}

impl MappingBackingRefusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unlisted => "mapping_backing_id_unlisted",
            Self::Unmapped => "mapping_backing_id_unmapped",
        }
    }
}

/// The canonical backing identity of the storage a guest mapping's surface
/// occupies.
///
/// [`backing_id`]'s other half, and the answer to
/// `reims_vgpu_core::resolve::MappingResolver`. The two are separate functions
/// over separate namespaces — a mapping id and an object-list reference arrive
/// as `u32`s that overlap numerically and name unrelated things — but they mint
/// into one identity space, so a mapping-reached and an address-reached piece of
/// storage can never collide. `crate::model::BackingWindowRefs` states how.
///
/// The incarnation is the mapping's own `map_generation` rather than the window
/// counter, and `DeviceState::mapping_backing_identity` says why that is forced
/// rather than chosen.
///
/// # Errors
///
/// [`MappingBackingRefusal`], for the two ways a mapping id names no surface.
pub fn mapping_backing_id(
    state: &DeviceState,
    mapping_id: u32,
) -> Result<reims_vgpu_core::access::BackingId, MappingBackingRefusal> {
    let entry = state
        .mappings
        .get(&mapping_id)
        .ok_or(MappingBackingRefusal::Unlisted)?;
    if !entry.mapped {
        return Err(MappingBackingRefusal::Unmapped);
    }
    Ok(state.mapping_backing_identity(mapping_id, entry.map_generation))
}

/// What an object that named no guest-VA window is, when it is not a window.
///
/// Split from [`backing_id`] so the answer is exhaustive over the object types
/// the resource constructor accepts, rather than a fallthrough from a decode
/// that happened not to produce a window.
///
/// One arm is not a refusal at all. A mapper-ref texture's storage is a
/// mapping's, and the descriptor in hand names which mapping — so the reference
/// is followed to [`mapping_backing_id`] and only the mapping's own refusal can
/// come back. A zero mapping id, and a descriptor that does not decode, are the
/// descriptor rung rather than a mapping refusal: the bytes are there and they
/// do not yet name a mapping, which is what an unwritten handle is on the
/// address-named types.
fn backing_id_without_window(
    state: &DeviceState,
    obj_ref: u32,
    entry: &ListObjectEntry,
    descriptor: &[u8],
) -> Result<reims_vgpu_core::access::BackingId, BackingIdRefusal> {
    use crate::runtime::decode::resource::{
        decode_iosurface_texture_descriptor, texture_view_opcode, Descriptor, HEAP_TEXTURE_OPCODE,
        HEAP_TEXTURE_WIDE_OPCODE, OBJECT_TYPE_DUAL_PLANE_TEXTURE,
    };
    let unresolved = BackingIdRefusal::Unresolved(LadderRung::DescRead {
        declared_len: entry.descriptor_length,
    });
    Err(match entry.object_type {
        // The surface-backing object, whose storage is the mapping's and whose
        // mapping id is its own object-list slot.
        //
        // This is not the numeric coincidence the two namespaces are otherwise
        // kept apart for. A backing is registered at slot `surface_id` *because*
        // that is what a surface id is here — `backing_claimant_tasks` finds a
        // surface's owner by probing exactly this slot for exactly this type,
        // and `crate::model::BackingWindowRefs` says the address route names no
        // storage for `OBJECT_TYPE_BACKING` precisely because the mapping route
        // owns it. The type is what makes the integer one namespace rather than
        // two, which is why the arm is keyed on the type and not on the id
        // resolving.
        //
        // It answered `NamesNoStorage` before, and a driven macos-15 boot said
        // what that cost: every one of 36 re-points the semantic model was not
        // told about was a backing object whose slot also named a live mapping
        // surface.
        OBJECT_TYPE_BACKING => {
            return mapping_backing_id(state, obj_ref).map_err(|refusal| {
                BackingIdRefusal::ThroughMapping {
                    mapping_id: obj_ref,
                    refusal,
                }
            });
        }
        OBJECT_TYPE_MAPPER_REF_TEXTURE => {
            let Ok(Descriptor::IOSurfaceTexture { mapping_id, .. }) =
                decode_iosurface_texture_descriptor(descriptor)
            else {
                return Err(unresolved);
            };
            if mapping_id == 0 {
                return Err(unresolved);
            }
            return mapping_backing_id(state, mapping_id).map_err(|refusal| {
                BackingIdRefusal::ThroughMapping {
                    mapping_id,
                    refusal,
                }
            });
        }
        // A texture whose descriptor is a heap placement, which is a record at
        // its own opcode rather than an object type of its own -- the placement
        // is how a texture is made, not what kind of object it is.
        OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS
            if matches!(
                texture_view_opcode(descriptor),
                Some(HEAP_TEXTURE_OPCODE) | Some(HEAP_TEXTURE_WIDE_OPCODE)
            ) =>
        {
            BackingIdRefusal::HeapPlaced
        }
        // A texture or buffer that produced no window has a descriptor the
        // guest has not finished writing -- a zero handle or a zero allocation
        // size. That is the ladder's `DescRead` in substance: the bytes are
        // there and they do not yet name an allocation.
        OBJECT_TYPE_TEXTURE
        | OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS
        | OBJECT_TYPE_BUFFER
        | OBJECT_TYPE_DUAL_PLANE_TEXTURE => unresolved,
        other => BackingIdRefusal::NamesNoStorage { object_type: other },
    })
}

/// Whether another constructed reference in this task names the same
/// allocation, asked without touching guest memory.
///
/// The construction cache alone, because the callers are on hot paths and
/// because the cache is the right question for them: it holds the window each
/// reference was *resolved* against, which is the window the device's own
/// per-reference state was built for.
///
/// `None` when this reference has no cached construction, no window, or no
/// company. It is a reading and not a decision — see
/// [`note_reference_shares_storage`] for what a caller does with it.
pub fn cached_window_peer(state: &DeviceState, task_id: u32, obj_ref: u32) -> Option<u32> {
    let base = cached_window(state, task_id, obj_ref)?;
    state
        .task_resources
        .in_task(task_id)
        .into_iter()
        .filter(|&(name, _)| name.slot.0 != obj_ref)
        .find(|(_, other)| other.backing_window(state.page_shift) == Some((base.0, base.1)))
        .map(|(name, _)| name.slot.0)
}

/// The allocation a constructed reference names, from its own decode.
///
/// The construction cache alone and no guest read, which is what makes it
/// affordable on the payment path. A reference this device has not constructed
/// has no cached window and is not one this device is keeping per-reference
/// state for either, so the two absences agree.
#[must_use]
pub fn cached_window(state: &DeviceState, task_id: u32, obj_ref: u32) -> Option<(u64, u64)> {
    state
        .constructed_object(task_id, obj_ref)?
        .backing_window(state.page_shift)
}

/// Report that per-reference state is being kept for a reference that shares
/// its storage with another.
///
/// # What this is asking
///
/// Several of this device's caches and generations are keyed
/// `(task, reference)` and stand for *storage*: the guest-write generation a
/// GVA writeback debt compares against, and the two host-copy caches. A
/// reference is a lawful key for those only if one allocation has one live
/// reference — and it does not. `note_backing_window_alias` found the
/// compositor holding its scanout allocation under two, so a guest write
/// declared through one reference leaves the other's generation unchanged and
/// its stale image authoritative.
///
/// The host-copy caches are fixed: a re-point now sweeps the window's peers.
/// The guest-write generation is **not**, and this is what says whether it
/// needs to be — it is bumped some six thousand times per census window, far
/// too hot to resolve a window on, so the question is asked once per distinct
/// reference.
///
/// # The reading, and why the re-key did not happen
///
/// A driven macos-15 boot asked it of **1285** distinct references and every
/// one of them was the only constructed name for its allocation
/// (`reference_storage_asked=1285`, `reference_storage_shared=0`). So the
/// `(task, reference)` key stands for storage everywhere this path uses it,
/// and re-keying it on the window would have bought nothing at a cost the hot
/// path cannot carry.
///
/// That is not the same as saying the alias does not exist — it does, and
/// `note_backing_window_alias` finds it on the compositor's scanout allocation
/// on every boot. It says the two sets do not overlap: the references that
/// share an allocation are not the references this generation is kept for.
/// Two facts, and only the measurement distinguishes them.
///
/// This stays on, at one scan per distinct reference, because the claim is
/// about a workload and not about the code: a guest that does bind two names
/// over one buffer would make the re-key necessary, and this is what would say
/// so rather than a wrong frame.
///
/// # More than one site asks it
///
/// The GVA writeback debt asks it too. `GvaResourceKey` is the resource
/// because that is what `CmdSynchronizeResources` names, and the debt it holds
/// is owed by *storage* — so a second live name over one allocation could be
/// synchronised while this one's frame stays owed, and the guest would CPU-read
/// pages the device still holds newer pixels for.
///
/// `reference_storage_asked` and `reference_storage_shared` are shared across
/// the sites on purpose: the question they answer is "does any per-reference
/// key that stands for storage alias", which is one question. Which site saw it
/// is on the failure line, and that is where it matters — a non-zero
/// `reference_storage_shared` is read by finding the line, not by reading the
/// counter.
pub fn note_reference_shares_storage(
    state: &DeviceState,
    task_id: u32,
    obj_ref: u32,
    site: &'static str,
) {
    // Once per reference, not per write. The scan is over one task's
    // constructed resources and the hot path cannot afford it per call; the
    // contract question it answers is about the reference, so once is enough.
    if !crate::observe::first_sight(site, u64::from(task_id) << 32 | u64::from(obj_ref)) {
        return;
    }
    crate::runtime::drain::note_store_route("reference_storage_asked");
    let Some(peer) = cached_window_peer(state, task_id, obj_ref) else {
        return;
    };
    crate::runtime::drain::note_store_route("reference_storage_shared");
    // Both names, so a hot path may ask about either in one lookup.
    state.note_aliased_reference(task_id, obj_ref);
    state.note_aliased_reference(task_id, peer);
    crate::observe::fail(format!(
        "{site} task={task_id} ref={obj_ref} peer={peer} (per-reference state is \
         standing for storage that has a second live name, so a statement made \
         through this reference leaves the other one's copy of it unchanged)"
    ));
}

/// Drop the host copies held under *every other* reference over the re-pointed
/// window.
///
/// # The defect this closes
///
/// A re-point invalidated the host copies keyed by the reference its packet
/// named, on the reading that a reference is a resource and a resource has one
/// set of copies. Both host caches are keyed `(task, reference)`, so a second
/// reference over the same allocation kept its texture — built from pages the
/// packet has just said are different pages, and served to every later draw
/// that binds it. That is a wrong surface with no refusal anywhere, which is
/// the failure the whole window derivation exists to prevent, reached through
/// the cache rather than through the identity.
///
/// It is not hypothetical: `note_backing_window_alias` found the compositor
/// holding its 1920×1080 scanout allocation under two live references, and the
/// scanout buffer is the one this device re-points.
///
/// # Why the peers are found by walking the caches
///
/// The set that matters is the set that *has* a copy to drop, and that is
/// exactly the keys in the two caches. It is small — a host copy per bound
/// resource, not per reference the guest published — and it needs no index
/// this device would then have to keep true. A window→references map would be
/// a third thing to invalidate on every lifetime event, and the events it
/// would have to see include the one no packet announces.
fn invalidate_window_peers<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    named: u32,
    base: u64,
) {
    let peers: Vec<u32> = state
        .host_texture_surfaces
        .keys()
        .chain(state.host_linear_textures.keys())
        .filter(|&&(task, ref_)| task == task_id && ref_ != named)
        .map(|&(_, ref_)| ref_)
        .collect();
    // Counted whether or not anything is dropped, for the reason
    // `backing_window_collision` is: a boot where no reference over the
    // re-pointed window held a copy and a boot where the sweep never ran read
    // the same on the drop counter alone, and only the first says the guard
    // cost nothing.
    crate::runtime::drain::note_store_route_n(
        "replace_physical_window_peers_examined",
        peers.len() as u64,
    );
    for peer in peers {
        if repointed_window(state, host, task_id, peer) != Some(base) {
            continue;
        }
        let (texture, linear) = state.invalidate_object_host_copies(task_id, peer);
        if texture || linear {
            crate::runtime::drain::note_store_route("replace_physical_window_peer_dropped");
            if crate::observe::first_sight(
                "replace_physical_window_peer",
                u64::from(task_id) << 32 | u64::from(peer),
            ) {
                crate::observe::fail(format!(
                    "replace_physical_window_peer task={task_id} named={named} peer={peer} \
                     base={base:#x} tex={texture} lin={linear} (a second reference over the \
                     re-pointed window was holding a host copy built from the pages the \
                     packet says have already changed, and every later bind of it would \
                     have been served those pages)"
                ));
            }
        }
    }
}

/// The guest-VA window a re-point packet's reference names.
///
/// The construction cache first: it holds the descriptor the accepted work was
/// planned against, and that is the window whose incarnation has to move.
/// Falling back to the guest's own list covers a reference this device has
/// never constructed, which is 96 of 103 re-points on a driven boot.
fn repointed_window<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    object_id: u32,
) -> Option<u64> {
    if let Some((base, _)) = cached_window(state, task_id, object_id) {
        return Some(base);
    }
    let entry = probe_list_entry(state, host, task_id, object_id)?;
    let descriptor = read_descriptor(state, host, task_id, &entry)?;
    backing_window(state.page_shift, &entry, &descriptor).map(|(base, _)| base)
}

/// Report what a heap-placed texture's heap reference names.
///
/// # The one field of a heap placement this device throws away
///
/// A heap-placed texture's record carries a heap reference and an offset. This
/// device checks the reference against zero, puts it in a log line and resolves
/// it no further: the placed texture becomes host-only storage keyed by the
/// texture's own name, so two textures placed at overlapping offsets in one
/// heap become two independent host images. Content written through one is not
/// visible through the other, which is the thing a heap is for.
///
/// That is a modelling gap and not a decode failure — the placement executes,
/// and correctly, as long as nothing aliases. Reported on the always-on channel
/// because a guest that does alias gets a wrong surface with no refusal
/// anywhere, and because the reading here is what closes it.
///
/// # The claim this tests, and which reading falsifies it
///
/// The claim is that a heap is an ordinary object-list object: the reference
/// arrives in the same `u32` form as the resource's own, immediately after it
/// in the same record, and [`RESOURCE_CONSTRUCTOR_TYPE_MASK`] already accepts
/// heap object tags. If that holds, a heap's canonical identity is the same
/// address-named identity every other object-list object has, and what is left
/// to recover is only the heap's *extent* — the length a placement is bounded
/// by, which the descriptor at this slot would declare.
///
/// [`HeapReference::Unlisted`] is the reading that falsifies it: the heap is
/// then named somewhere this device does not look, and both halves are open
/// again. [`HeapReference::Listed`] confirms it and hands over the two values
/// that say where the extent is — which tag, and how many descriptor bytes.
///
/// One 12-byte guest read per distinct shape. `first_sight` is keyed on the
/// answer rather than on the reference, so a boot placing ten thousand textures
/// in one heap costs one line, and a second *kind* of answer still gets its own.
pub fn note_heap_reference<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    heap_ref: u32,
) -> HeapReference {
    let found = match probe_list_entry(state, host, task_id, heap_ref) {
        Some(entry) => HeapReference::Listed {
            object_type: entry.object_type,
            descriptor_length: entry.descriptor_length,
        },
        None => HeapReference::Unlisted,
    };
    let shape = match found {
        HeapReference::Unlisted => 0,
        HeapReference::Listed {
            object_type,
            descriptor_length,
        } => (u64::from(object_type) << 32) | u64::from(descriptor_length) | (1 << 63),
    };
    if !crate::observe::first_sight("heap_reference", shape) {
        return found;
    }
    let (names, verdict) = match found {
        HeapReference::Unlisted => (
            "nothing".to_string(),
            "a heap is not an object-list object after all, so its identity is open again",
        ),
        HeapReference::Listed {
            object_type,
            descriptor_length,
        } => (
            format!("type={object_type} desc_len={descriptor_length}"),
            "a heap is an object-list object, so it takes the address-named identity; \
             read the extent out of this descriptor",
        ),
    };
    crate::observe::fail(format!(
        "heap_reference_unresolved task={task_id} heap={heap_ref} names={names} \
         (the placement becomes host-only storage keyed by the texture's own name, \
         so two placements sharing bytes in this heap are two independent images; \
         {verdict})"
    ));
    found
}

/// The mapping format a backing latches: single-plane MTL only.
///
/// Multi-plane and unknown-FourCC surfaces get `0`, and that zero is a decoded
/// refusal rather than an absence — stage and paint must not invent BGRA, and
/// mapper-ref-texture selects planes through `device_desc` instead.
/// [`iosurface_pixel_format_to_mtl`] states the same rule for the conversion.
///
/// Named rather than inlined at [`apply_backing`] because
/// [`backing_matches_latched_geom`] has to compute the *same* value: it compares
/// a freshly-read descriptor against `m.format`, which is whatever this returned
/// last time.
fn latched_mapping_format(surf: &BackingRecord) -> u16 {
    if backing_is_multiplanar(surf) {
        return 0;
    }
    iosurface_pixel_format_to_mtl(surf.pixel_format)
}

/// Whether the geometry already latched on this mapping is still the geometry
/// the freshly-read descriptor declares.
///
/// Both arms of [`resolve_backing_ex`]'s freshness test ask this, and they
/// used to ask it differently: the non-force arm compared width **and** height,
/// the force arm compared width only. They are the same question — "may this
/// resolve return without rebuilding" — and the force arm is the one that cannot
/// afford to be looser, because `force_fresh` returns through
/// [`win_backing_search`] *without* calling [`apply_backing`], so neither
/// `set_mapping_geom` nor `synthesize_device_desc_from_backing` runs. A height
/// change that stays inside the same page count therefore left `m.height` and
/// the whole device descriptor describing the previous incarnation, on the exact
/// path `ensure_surface_for_present` calls to catch a wire geometry change.
///
/// Format is compared too, and neither arm used to. A surface id can be recycled
/// at identical dimensions with a different pixel format, and the format is what
/// every read window's bytes-per-pixel comes from — so keeping the old one
/// samples the new backing at the wrong stride. The comparison goes through
/// [`latched_mapping_format`] rather than the wire FourCC, because `m.format` is
/// whatever that function last returned. Comparing the FourCC would report every
/// surface as changed, and comparing the raw conversion would report every
/// multi-plane surface as changed forever, since the latch deliberately discards
/// it in favour of 0.
fn backing_matches_latched_geom(m: &MappingEntry, surf: &BackingRecord) -> bool {
    m.width == surf.width && m.height == surf.height && m.format == latched_mapping_format(surf)
}

/// The order [`resolve_backing_ex`] probes task object lists in: task 0,
/// then the cached owner hint, then every other task once.
///
/// An iterator rather than a materialised list. The order is unchanged, but
/// building it as a `Vec` allocated 257 elements on every call, and this runs
/// from `ensure_surface_for_present` on every present for every resident
/// mapping — thousands of times a boot to read element 0 and stop.
///
/// A hint of 0 contributes nothing: 0 is already first, so admitting it would
/// only cost the `!= hint` filter its meaning. A hint naming no live task is
/// admitted and then skipped by the liveness test at the probe, which is where
/// that question was always answered.
///
/// The tail is the **live** ids rather than `1..256`. It walked the fixed range
/// while the task table was an array; the probe refuses an inactive task, so
/// those ids were only ever skipped, and yielding them cost the caller a
/// liveness test per id per present. Task 0 leads whether or not it is live —
/// see the caller for why the guest's kernel task is named rather than found —
/// so it is filtered out of the tail rather than left to `live_ids` to omit.
fn backing_probe_order(tasks: &TaskTable, hint: u32) -> Vec<u32> {
    std::iter::once(0)
        .chain(Some(hint).filter(|&h| h != 0))
        .chain(tasks.live_ids().filter(move |&tid| tid != 0 && tid != hint))
        .collect()
}

/// Take `task_id` as the owner of `surface_id` and report the search's exposure.
fn win_backing_search<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
    task_id: u32,
) -> bool {
    record_backing_owner(state, surface_id, task_id);
    note_backing_claimants(state, host, surface_id, task_id);
    true
}

fn resolve_backing_ex<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
    force: bool,
) -> bool {
    if !crate::model::is_mapping_id(surface_id) {
        return false;
    }
    // Task probe order: task 0 first, then the cached owner-task hint (so a hot
    // present-path re-scan short-circuits on the owning task instead of walking
    // all 256 slots), then the remaining tasks.
    //
    // Task 0 leads because the guest says so, not because it is where surfaces
    // have happened to be. A ref-texture view carries the owning task at
    // [`TYPE5_OWNER_TASK`] and it is the accelerator's kernel task, whose id is a
    // hardcoded 0 and whose slot the task-id allocator reserves before any client
    // task exists. `note_ref_texture_owner_task` fails loudly if that ever reads
    // otherwise.
    //
    // The remaining 255 probes are not dead weight on that reading. They cost
    // nothing on the path that matters — every successful resolve measured has
    // stopped on the first probe — and they are what makes `backing_claimants` able
    // to say a second task claims the id at all.
    //
    // Built as an iterator rather than a `Vec`. The order is the same one, but
    // materialising it allocated a 257-element vector on every call, and this is
    // called from `ensure_surface_for_present` on every present for every
    // resident mapping — thousands of times a boot to read element 0 and stop.
    let hint = state
        .mappings
        .get(&surface_id)
        .map(|m| m.owner_task_hint)
        .unwrap_or(0);

    for task_id in backing_probe_order(&state.tasks, hint) {
        // Count the guest-read cost of one active-task object-list probe.
        let Some(entry) = probe_list_entry(state, host, task_id, surface_id) else {
            continue;
        };
        if entry.object_type != OBJECT_TYPE_BACKING {
            continue;
        }
        let Some(desc) = read_descriptor(state, host, task_id, &entry) else {
            defer_backing_fail(
                surface_id,
                crate::observe::ladder_slug!("", desc_read),
                None,
                format!(
                    "backing_fail reason=desc_read sid={surface_id} task={task_id} desc_gva={:#x} desc_len={}",
                    entry.descriptor_gva, entry.descriptor_length
                ),
            );
            continue;
        };
        let _ = state.insert_object(task_id, surface_id);
        let Some(surf) = decode_backing(&desc) else {
            defer_backing_fail(
                surface_id,
                crate::observe::ladder_slug!("", desc_decode),
                None,
                format!(
                    "backing_fail reason=desc_decode sid={surface_id} task={task_id} desc_len={} backing_pfn={:#x} length={:#x}",
                    desc.len(),
                    reims_vgpu_wire::device_desc::backing_header(&desc)
                        .map(|h| h.backing_pfn.get())
                        .unwrap_or(0),
                    reims_vgpu_wire::device_desc::backing_header(&desc)
                        .map(|h| h.length.get())
                        .unwrap_or(0)
                ),
            );
            continue;
        };
        // Force path validated the cached pages are still fresh → keep them.
        let mut force_fresh = false;
        // Skip rebuild when pages already match this backing (hot present path).
        if !force {
            let same_geom = state
                .mappings
                .get(&surface_id)
                .map(|m| {
                    m.mapped
                        && !m.page_entries.is_empty()
                        && m.has_geom
                        && backing_matches_latched_geom(m, &surf)
                })
                .unwrap_or(false);
            if same_geom {
                // Same geom + non-empty pages: keep (guest double-buffer
                // may still rewrite page *content* without changing pfn).
                return win_backing_search(state, host, surface_id, task_id);
            }
        } else if let Some(m) = state.mappings.get(&surface_id) {
            // Force: keep the cached table only while the CURRENT task
            // page-table translation of the descriptor's first and last
            // backing pages still matches it. `backing_pfn` is a GPU-VA page;
            // the guest may remap that GVA range onto new physical pages
            // without changing surface id, geometry, or length (early-boot
            // console FB vs the WindowServer reallocation). A same-size guard
            // here kept boot-time pages forever, so presents froze on pages
            // nobody writes.
            if m.mapped && !m.page_entries.is_empty() {
                let page_shift = state.page_shift;
                let page_size = page_size_of(page_shift);
                let need = ((surf.length.saturating_sub(1)) / page_size) + 1;
                if m.page_entries.len() as u64 == need && backing_matches_latched_geom(m, &surf) {
                    let task = state.tasks.get(task_id).filter(|t| t.active);
                    let entry_fresh = |idx: u64, entry: u32| -> bool {
                        let gva = ((surf.backing_pfn as u64) + idx) << page_shift;
                        let cached = entry_gpa_shift(entry, page_shift);
                        match task
                            .and_then(|t| gva_mem::translate_task_gva(host, t, gva, page_shift))
                        {
                            Some(gpa) => cached == Some(gpa & !(page_size - 1)),
                            // No translation now, so nothing here can vouch for
                            // the cached table. The device never caches a
                            // GVA-as-GPA entry, so say stale and let the rebuild
                            // refuse, which is what moves the task search on to
                            // the task that can translate.
                            None => false,
                        }
                    };
                    let last = m.page_entries.len() - 1;
                    if entry_fresh(0, m.page_entries[0])
                        && entry_fresh(last as u64, m.page_entries[last])
                    {
                        force_fresh = true;
                    } else {
                        crate::observe::fail(format!(
                            "backing_pages_stale sid={surface_id} task={task_id} n={} gpa0={:#x} (task PT translation moved; rebuilding)",
                            m.page_entries.len(),
                            entry_gpa_shift(m.page_entries[0], page_shift).unwrap_or(0)
                        ));
                    }
                }
            }
        }
        if force_fresh {
            return win_backing_search(state, host, surface_id, task_id);
        }
        if apply_backing(state, host, task_id, surface_id, &surf) {
            return win_backing_search(state, host, surface_id, task_id);
        }
    }
    // No task could back it. Only now is a probe's refusal a backing failure.
    flush_backing_fail(surface_id);
    false
}

/// Ensure surface backing for present: backing pages when needed, else keep arm
/// MappingInternal path.
///
/// Resolves backing once pages are empty; guest double-buffering uses distinct
/// surface_ids (content updates land in-place on an already-mapped pfn).
pub fn ensure_surface_for_present<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
) -> bool {
    if surface_id == 0 {
        return false;
    }
    let need = state
        .mappings
        .get(&surface_id)
        .map(|m| !m.mapped || m.page_entries.is_empty())
        .unwrap_or(true);
    if need {
        let _ = resolve_backing(state, host, surface_id);
    } else {
        // Opportunistic refresh if wire geom changed (mode switch).
        let _ = resolve_backing_force(state, host, surface_id);
    }
    // Arm/iosfc path: MappingInternal resolve when captured.
    let _ = crate::runtime::mapper::ensure_resolved_for_scanout(state, host, surface_id);
    state
        .mappings
        .get(&surface_id)
        .map(|m| m.mapped && !m.page_entries.is_empty() && m.has_geom)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests;
