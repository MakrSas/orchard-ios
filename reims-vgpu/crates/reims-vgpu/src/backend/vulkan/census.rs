//! The census lines only the Vulkan rail can answer.
//!
//! Every reading here comes out of `backend::vulkan::engine` — its counters,
//! its phase windows, its registries, its mutex — so a rail that is not this
//! one has nothing to take from them, and a build that carries both rails must
//! not print them for a device running on the other. The drain asks through
//! [`crate::backend::Backend::emit_census`] and keeps the ordering; what each
//! site is for is on [`crate::backend::CensusSite`].
//!
//! These emit rather than return their lines. The order within a site is this
//! rail's own — `engine_delta` before `registry_pressure` before `draw_phase`,
//! each dividing the one before it — and that ordering is part of what makes
//! them readable.

/// How many RAMBlocks this device has imported, and how many bytes they cover,
/// as **levels** rather than per-window deltas.
///
/// This is the reading that says whether the one-import-per-RAMBlock model held.
/// The count should be one or two for a whole boot and flat across every window;
/// a count that tracks the workload is the per-resource import the model exists
/// to avoid, which `VK_EXT_external_memory_host` does not guarantee works twice
/// over one allocation and which would pay the driver's page pinning thousands
/// of times a second for an answer that never changes.
///
/// Flat is therefore the healthy reading and a rise is the alarm — the opposite
/// polarity to most lines here, which is why the count is emitted every window
/// rather than once at import time. A single line at import time could not
/// distinguish "imported once" from "imported once per window".
///
/// # Both terms, because the numerator alone is ambiguous
///
/// A backend imports a span at its **first reference**, not at device init, so
/// the imported count is bounded above by the number of spans the shim reported
/// and starts below it. `ramblocks=1` alone cannot distinguish "this machine has
/// one RAMBlock and it is imported" from "this machine has two and the workload
/// has only ever touched one" — and the second is a workload fact, not a defect.
/// The denominator comes from [`crate::runtime::guest_ram_map::span_census`],
/// which is the shim's answer, so the pair reads `imported/reported`.
///
/// This is not hypothetical on the x86 pathway. `vm/boot-x86.sh` boots `-m 16G`
/// and a driven Safari boot measures `ramblocks=1/4 mib=14336/16399`: the shim
/// reports four writable spans and the workload imports one, the 14 GiB half of
/// `-m` above the PCI hole. The numerator alone reads as 2 GiB of guest RAM
/// having gone missing against `-m 16G`, which is how it was first misread.
///
/// The reported set is larger than `-m` — 16399 MiB against 16384 — because the
/// shim walks the flat view rather than `-m`. `guest_ram_span` names each span
/// at build time; on this boot they are:
///
/// ```text
/// n=0/4 gpa=0x0          len=786432      (768 KiB, below the legacy VGA hole)
/// n=1/4 gpa=0x100000     len=2146435072  (2047 MiB, 1 MiB up to the PCI hole)
/// n=2/4 gpa=0x80000000   len=16777216    (16 MiB — this device's own BAR1)
/// n=3/4 gpa=0x100000000  len=15032385536 (14336 MiB, above 4 GiB — imported)
/// ```
///
/// Spans 0, 1 and 3 are the two halves of `-m 16G` either side of the PCI hole,
/// with the low half split again by the legacy hole at `0xA0000`. Span 2 is the
/// 15 MiB of "extra": it is `REIMS_VGPU_PCI_FB_SIZE`, the linear GOP framebuffer
/// `reims-vgpu-pci.c` registers as BAR1 with `memory_region_init_ram`, assigned
/// into the PCI hole at 2 GiB. A plain RAM BAR is not ROM, not ROMD, not a
/// `ram_device` and not readonly, so it passes the shim's filter — that filter
/// screens out memory the guest cannot store into, and the guest *can* store
/// into a GOP framebuffer, which is what a GOP framebuffer is for.
///
/// It is reported and never imported: only span 3 has ever been referenced. The
/// consequence worth knowing is that a GPA landing inside BAR1 would resolve
/// rather than earning `GpaNotInAnyImport`, so it is bounded to this device's
/// own framebuffer rather than refused. That is the host console's bytes, not
/// another RAMBlock's and not this process's private state, and the guest
/// already writes them through the BAR. Narrowing the filter is **not** an
/// obvious improvement: the EFI console path exists precisely because the guest
/// points at BAR1, so excluding it would need evidence that no legitimate
/// reference lands there. Nothing has measured that.
///
/// `mib` is the same level and is not a rate: it is guest RAM the device can
/// currently reach, against what the machine reported.
pub(crate) fn emit_guest_import_levels() {
    let (bytes, count, aliases) = crate::backend::vulkan::engine::guest_import_census();
    let (spans, span_bytes) = crate::runtime::guest_ram_map::span_census();
    // An engine that never imported emits nothing, so a host on a negative
    // `host_pointer` rung — or a boot before the first guest window — costs no
    // line, and a zero here always means the copying rails rather than silence.
    if count == 0 && aliases == 0 {
        return;
    }
    crate::observe::off(format!(
        "guest_import_levels (levels, not per-interval) ramblocks={count}/{spans} aliases={aliases} \
         imported_mib={} ramblock_reported_mib={} (RAMBlock spans import lazily; \
         packed aliases add to imported_mib without changing the reported RAM size)",
        bytes / (1024 * 1024),
        span_bytes / (1024 * 1024),
    ));
}

/// Live entry counts of the caches that hold one entry per distinct guest
/// object, as **levels** rather than per-window deltas.
///
/// These caches carry no capacity and no replacement rule. The argument for that
/// is that each key is a content digest or a complete descriptor of guest state,
/// so the count is the guest's own distinct object set and settles once the
/// guest has finished compiling. That is a claim about a running guest, and this
/// line is what can falsify it: a level still climbing minutes into a boot means
/// some key is carrying per-frame state, and the argument is wrong for that
/// cache. A settling level is the argument holding.
///
/// `m2v` counts translated shaders (`runtime::m2v_cache`); the rest are the
/// Vulkan engine's immutable-object caches.
pub(crate) fn emit_object_cache_levels(state: &crate::model::DeviceState) {
    let [shaders, layouts, attr_sets, passes, pipelines, samplers, compute_pipelines] =
        crate::backend::vulkan::engine::object_cache_levels(state);
    let (_, _, m2v) = crate::runtime::m2v_cache::stats();
    crate::observe::off(format!(
        "object_cache_levels (levels, not per-interval) m2v={m2v} shaders={shaders} \
         layouts={layouts} attr_sets={attr_sets} passes={passes} pipelines={pipelines} \
         samplers={samplers} compute_pipelines={compute_pipelines}"
    ));
}

pub(crate) fn emit_engine_delta() {
    use crate::backend::vulkan::engine::CounterSnapshot;
    static PREV: std::sync::Mutex<Option<CounterSnapshot>> = std::sync::Mutex::new(None);
    let now = crate::backend::vulkan::engine::counter_snapshot();
    let Ok(mut prev) = PREV.lock() else {
        return;
    };
    let d = now.delta_since(&prev.unwrap_or_default());
    *prev = Some(now);
    // Generated from the counter vocabulary rather than named here, so this line
    // cannot fall behind it again; see `CounterSnapshot::delta_fields`.
    let mut line = String::from("engine_delta");
    for (name, value) in d.delta_fields() {
        use std::fmt::Write as _;
        let _ = write!(line, " {name}={value}");
    }
    crate::observe::off(line);
    emit_registry_pressure(&now);
    emit_draw_phase();
}

/// How far the resident registries reached, and what the populations that
/// cannot be given back cost.
///
/// Separate from `engine_delta` because these fields are read **absolute**, and
/// that line reports differences. A high-water mark deltas to nonsense — the
/// difference between two peaks is not a peak, and reads as zero for the rest of
/// the boot once the true maximum is behind the window — so it is taken from the
/// snapshot rather than from `delta_since`.
///
/// `peak` has no `cap` beside it any more, and that is the point: the
/// resident-target population is bounded by the allocator refusing rather than by
/// a slot count (see `ResourcePools::recoverable_residents`). Read `peak` against
/// `peak_mib` — the pair is what says whether a count was ever a proxy for VRAM,
/// and it answered no: 194 slots against 211 MiB on one workload and 41 against
/// 74 MiB on another, a 1.65x spread in MiB per slot between two ordinary
/// desktop workloads.
///
/// `sole_copy` is the half of that population the allocation-failure retry
/// cannot hand back. Its ratio against `peak` is the reading that matters now:
/// near 1 means a retry would find nothing to give, and the copy-out sites are
/// what needs work.
///
/// # Why `resident_samples` is on this line
///
/// It is the *denominator* of `sampled_resident_missing`, which is raised from
/// one place — the `SampledSource::Target` arm of the engine's sampled loop,
/// also the sole increment of `sampled_gpu_binds`. When it is zero, no draw
/// bound a resident as a texture, nothing could have observed a destroyed one,
/// and a zero missing-count is a null instrument rather than a pass.
///
/// This field exists because that denominator was once argued about from two
/// boots that had never been compared. The since-retired slot cap was driven six
/// times over its bound and reported `evicts=1591` against
/// `sampled_resident_missing=0`; a later reading of `sampled_gpu_binds=0` —
/// taken on a *different* workload — was used to call that pair a null
/// instrument. Printing the denominator beside the pair settled it in one boot:
/// `web-content-probe --churn 1` reports `resident_samples=11742`, so the arm
/// does run, the zero was a real measurement, and the null-instrument objection
/// was itself the unfounded claim. The reading still matters: it is what says a
/// draw would have noticed had anything gone missing.
///
/// `cs_sole_copy` is the same protected-population reading over the
/// compute-storage registry. It is worth reading separately rather than summed:
/// that registry holds standalone `VkDeviceMemory` where the target registry
/// holds slab suballocations, so the two say different things about what an
/// allocation failure would have found to give back.
///
/// Neither registry publishes an eviction count any more, because neither has a
/// slot count to evict for. `vram_reclaim_retry` and
/// `vram_compute_storage_reclaim_retry` on the fail channel are what report a
/// reclaim now, and they fire only when an allocation was actually refused.
fn emit_registry_pressure(now: &crate::backend::vulkan::engine::CounterSnapshot) {
    crate::observe::off(format!(
        "registry_pressure (levels, not per-interval) current={}/{}mib \
         recoverable={}/{}mib pinned={}/{}mib peak={} peak_mib={} \
         resident_samples={} resample_peak_ms={}/{} \
         slab_mib={}/{} sole_copy={}/{}mib cs_sole_copy={}/{}mib",
        now.registry_current_count,
        now.registry_current_bytes >> 20,
        now.registry_recoverable_count,
        now.registry_recoverable_bytes >> 20,
        now.registry_pinned_count,
        now.registry_pinned_bytes >> 20,
        now.registry_non_pinned_peak,
        now.registry_non_pinned_peak_bytes >> 20,
        now.sampled_gpu_binds,
        now.resident_resample_peak_ms,
        crate::backend::vulkan::engine::IDLE_MAINTENANCE_START_MS,
        now.slab_carved_bytes >> 20,
        now.slab_held_bytes >> 20,
        now.registry_sole_copy_peak,
        now.registry_sole_copy_peak_bytes >> 20,
        now.compute_storage_sole_copy_peak,
        now.compute_storage_sole_copy_peak_bytes >> 20,
    ));
}

/// The split of `drain_duty`'s `draw_us`, over the same window.
///
/// `drain_duty` says a saturated second is 93-99% `draw_us` and `engine_delta`
/// says ~450 MB/s crosses the bus each way. Those two are consistent with
/// opposite fixes — moving fewer bytes, or stopping the per-draw GPU round trip
/// — and neither line can tell them apart. This one can: `readback_us` and the
/// staging half of `setup_us` scale with bytes, `wait_us` does not.
///
/// Silent when no draw ran, so an idle desktop costs nothing.
fn emit_draw_phase() {
    let Some(w) = crate::backend::vulkan::engine::draw_phase_window() else {
        return;
    };
    crate::observe::off(format!(
        "draw_phase draws={} prep_us={} slot_us={} pipeline_us={} \
         pl_depth_us={} pl_shader_us={} pl_layoutpass_us={} pl_compile_us={} pl_sampler_us={} \
         stage_us={} sg_roles_us={} sg_vertex_us={} sg_index_us={} sg_storage_us={} \
         sg_seed_us={} stage_pass_us={} \
         acquire_us={} acquire_sampled_us={} sampled_upload_us={} acquire_readback_us={} \
         descriptors_us={} \
         record_us={} rec_begin_us={} rec_barrier_us={} rec_pass_us={} rec_state_us={} \
         rec_draw_us={} submit_us={} post_target_us={} post_store_us={} post_sampled_us={} \
         post_park_us={} wait_us={} readback_us={} max_us={} stalls={}",
        w.draws,
        w.prep_us,
        w.slot_us,
        w.pipeline_us,
        w.pipeline_depth_us,
        w.pipeline_shader_us,
        w.pipeline_layout_pass_us,
        w.pipeline_compile_us,
        w.pipeline_sampler_us,
        w.stage_us,
        w.stage_roles_us,
        w.stage_vertex_us,
        w.stage_index_us,
        w.stage_storage_us,
        w.stage_seed_us,
        w.stage_pass_us,
        w.acquire_us,
        w.acquire_sampled_us,
        w.sampled_upload_us,
        w.acquire_readback_us,
        w.descriptors_us,
        w.record_us,
        w.rec_begin_us,
        w.rec_barrier_us,
        w.rec_pass_us,
        w.rec_state_us,
        w.rec_draw_us,
        w.submit_us,
        w.post_target_us,
        w.post_store_us,
        w.post_sampled_us,
        w.post_park_us,
        w.wait_us,
        w.readback_us,
        w.max_us,
        w.stalls,
    ));
    emit_stage_phase();
    emit_gather_phase();
    emit_gpu_span();
}

/// Beside `draw_phase`, because it is the one column in it the GPU wrote.
///
/// `slot_us` above is the drain worker blocked on a ring fence, and every session
/// before this one read that as "the GPU is busy" without a GPU-side number
/// existing. `busy_us` is that number: GPU microseconds summed over the
/// submissions retired this window, from timestamps each command buffer wrote
/// into itself.
///
/// Read the pair and never `busy_us` alone. Against a census second it is
/// utilisation; against `slot_us` it is how much of the worker's wait was this
/// device's own recorded work rather than queue latency, and those are two
/// different questions with two different fixes. `armed`/`sealed`/`read` say
/// whether a low reading is a quiet GPU or a probe that did not close.
///
/// The five `*_us`/`*_n` pairs tile `busy_us`/`read` by what the submission was
/// recorded for, so the shares say which rail owns the device's GPU time without
/// an ablation. `unattributed` is the identity that keeps that honest: it is
/// `read` minus the per-kind counts and must be zero.
///
/// **A per-second `busy_us` is not comparable across boots that delivered
/// different amounts of work.** The guest sets the draw rate on this rail, so a
/// change that slows the guest lowers `busy_us` by lowering the workload. Divide
/// by `draw_phase draws` or by the kind's own `*_n` before comparing two arms —
/// the writeback's own positive control halved the frame rate and lowered
/// `busy_us` by 48 % while per-submission GPU cost moved 1.5 %.
fn emit_gpu_span() {
    let Some(w) = crate::backend::vulkan::engine::gpu_span::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "gpu_span busy_us={} busy_max_us={} read={} armed={} sealed={} unread={} \
         unattributed={} draw_us={} draw_n={} store_us={} store_n={} \
         readback_us={} readback_n={} compute_us={} compute_n={} stamp_us={} stamp_n={}",
        w.busy_us,
        w.busy_max_us,
        w.read,
        w.armed,
        w.sealed,
        w.unread,
        w.unattributed(),
        w.kind_us[0],
        w.kind_n[0],
        w.kind_us[1],
        w.kind_n[1],
        w.kind_us[2],
        w.kind_n[2],
        w.kind_us[3],
        w.kind_n[3],
        w.kind_us[4],
        w.kind_n[4],
    ));
}

/// Where a compute-gather dispatch's CPU cost goes, four ways.
///
/// Emitted only when a gather dispatched, so the line's presence is itself the
/// statement that this boot ran the dispatch arm — see
/// [`crate::backend::vulkan::engine::gather_phase`] for what each part is and
/// what would remove it.
fn emit_gather_phase() {
    let Some(w) = crate::backend::vulkan::engine::gather_phase::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "gather_phase plan_us={} plan_n={} stage_us={} stage_n={} \
         dset_us={} dset_n={} record_us={} record_n={}",
        w.plan_us, w.plan_n, w.stage_us, w.stage_n, w.dset_us, w.dset_n, w.record_us, w.record_n,
    ));
}

/// Under `draw_phase`, dividing its largest column — `stage_us` is 83 % of that
/// phase's second on a driven drag, and the five parts want opposite fixes.
fn emit_stage_phase() {
    let Some(w) = crate::backend::vulkan::engine::stage_phase::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "stage_phase acquire_us={} acquires={} bytes_us={} bytes_n={} bytes_b={} \
         runs_us={} runs_n={} runs_b={} swap_us={} swap_n={} swap_b={} \
         shift_us={} shift_n={} shift_b={} \
         gather_us={} gather_n={} gather_b={}",
        w.acquire_us,
        w.acquires,
        w.bytes_us,
        w.bytes_n,
        w.bytes_b,
        w.runs_us,
        w.runs_n,
        w.runs_b,
        w.swap_us,
        w.swap_n,
        w.swap_b,
        w.shift_us,
        w.shift_n,
        w.shift_b,
        w.gather_us,
        w.gather_n,
        w.gather_b,
    ));
}

/// The engine mutex's wait and hold time over the same window, split by which
/// thread class asked for it.
///
/// Emitted beside `window_publish` because it divides the gap that line opens:
/// `window_publish fresh` is what the device offered the window and
/// `host_window_cadence presents` is what reached the screen, and when the two
/// disagree the first candidate is that the window thread could not have the
/// engine while the worker held it.
pub(crate) fn emit_engine_lock(win_ms: u64) {
    if let Some(line) = crate::backend::vulkan::engine::take_engine_lock_census(win_ms) {
        crate::observe::off(line);
    }
}
/// How much of the workload this rail's two gather caches were asked to hold.
///
/// Beside the engine counters they have to be read against: the eviction routes
/// say which cap fired and these say how much the workload wanted, and neither
/// is interpretable without the other. The second is the same question one rail
/// over, and the one with no cache behind it yet — `buffer_guest_gathers` says
/// how many gathers ran and this says how few distinct windows they were.
pub(crate) fn emit_working_set() {
    if let Some(wanted) = crate::backend::vulkan::engine::sampled_working_set_census() {
        crate::observe::off(wanted);
    }
    if let Some(wanted) = crate::backend::vulkan::engine::buffer_gather_working_set_census() {
        crate::observe::off(wanted);
    }
}
