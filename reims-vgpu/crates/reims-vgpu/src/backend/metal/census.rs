//! The census lines only the Metal rail can answer.
//!
//! Two lines: this rail's compiled-object cache levels, and its retained colour
//! render targets. The Vulkan rail asks the first of those of entirely
//! different tables and has no counterpart for the second, which is why both
//! are reached through [`crate::backend::Backend::emit_census`] rather than
//! through one `cfg`-selected function — a build carrying both rails has to
//! print the lines belonging to the device it is actually running.
//!
//! They live here and not beside the drain's neutral census for the same
//! reason: the only things they read are this rail's own tables, and a
//! reporting module in the neutral runtime forced those tables to be
//! re-exported out of the rail so a stranger could count them.

/// This rail's compiled-object cache levels, as **levels** rather than
/// per-window deltas — the same question and cadence as its Vulkan
/// counterpart, over different tables. This arm builds `MTLFunction` / `MTLRenderPipelineState` /
/// `MTLComputePipelineState` / `MTLSamplerState` / `MTLDepthStencilState` and
/// compute reflections, and holds them in `backend::metal::cache`.
///
/// No `m2v` field: AIR reaches Metal directly on this arm, so
/// `runtime::m2v_cache` is never populated and a zero there would read as an
/// empty cache rather than an absent rail.
pub(crate) fn emit_object_cache_levels() {
    let [functions, render_pso, compute_pso, samplers, depth_stencil, reflections] =
        super::cache::cache_levels();
    crate::observe::off(format!(
        "object_cache_levels (levels, not per-interval) functions={functions} \
         render_pso={render_pso} compute_pso={compute_pso} samplers={samplers} \
         depth_stencil={depth_stencil} reflections={reflections}"
    ));
}

/// This rail's retained colour render targets, as levels.
///
/// Three numbers and not one, because "retained" and "loadable" are different
/// facts and only the second says the rail is working: a `targets` that climbs
/// while `loadable` stays at zero is a rail holding textures and re-uploading
/// into every one of them, which reads as a win on memory and is a loss on
/// everything else. `bytes` is what the budget in
/// [`super::resident`] is spent against — read it beside
/// `metal_resident_evicted`, which is what says the budget is too small.
pub(crate) fn emit_resident_color_levels() {
    let (targets, bytes, loadable) = super::resident::levels();
    crate::observe::off(format!(
        "resident_color_levels (levels, not per-interval) targets={targets} \
         bytes={bytes} loadable={loadable}"
    ));
}
