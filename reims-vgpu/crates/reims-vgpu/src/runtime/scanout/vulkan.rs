//! What the Vulkan rail's resident registry can say about a present surface.
//!
//! Two questions, one file, because they are the same lookup asked for two
//! reasons: *would* a resident carry this present (a census/failure-channel
//! split at the drain), and *does* one, hand me its bytes (the capture). Both
//! resolve the surface through [`crate::backend::vulkan::present_identity`], and a
//! second spelling of that identity in either place would report a frame as
//! carried that the other then cannot find.
//!
//! Reached only through [`crate::backend::Backend`]; the drain and the capture
//! never name this rail.

use crate::backend::vulkan::engine;
use crate::backend::vulkan::present_identity::surface_identity;
use crate::model::DeviceState;

/// Would a resident carry the present this mapping names, at this geometry?
///
/// Asks [`engine::resident_presentable`], which shares `pools::slot_presentable`
/// with the window *publish* — the transaction that decides whether a resident
/// carries a present. Sharing the rule is the point rather than tidiness: a
/// looser predicate here would report a frame as carried that the publish then
/// refuses, which is a disagreement neither call site can see on its own — the
/// same shape as the publish/present split that once blanked the window.
pub fn present_resident_carries(
    state: &DeviceState,
    mapping: u32,
    width: u32,
    height: u32,
) -> Option<bool> {
    let identity = surface_identity(state, mapping, width, height);
    Some(engine::resident_presentable(&identity, width, height))
}

/// Fill `buf` from the mapping's GPU resident, without any guest-page scatter.
///
/// On `true` `buf` holds tight BGRA8; on `false` `buf` is untouched. A miss is
/// an expected steady-state condition (cold mid / no resident yet), so it is
/// counted in the `capture_source` census rather than logged per present.
/// The mapping's published frame from this rail's resident, as tight RGBA8.
///
/// `Backend::published_frame_rgba8` for this rail. `generation` is unused: the
/// engine's [`surface_identity`] already carries the evidence that decides which
/// image answers for a mapping at a geometry, and asking it to also match a
/// number this rail never wrote would decline every frame. The caller's
/// currency check still runs — it is what makes the question legitimate — it
/// simply is not the thing this rail keys on.
pub fn published_frame_rgba8(
    state: &DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    use crate::protocol::pixel_format::RGBA8_BPP;
    let need = (width as usize)
        .checked_mul(RGBA8_BPP as usize)?
        .checked_mul(height as usize)?;
    let identity = surface_identity(state, mapping_id, width, height);
    let mut bgra = engine::read_resident_bgra(&identity, need)?;
    if bgra.len() != need {
        return None;
    }
    // The engine reads BGRA8 for the console; every seed reader wants RGBA8.
    // In place — the same four bytes in a different order, and this readback is
    // already this function's own frame.
    crate::runtime::draw::swap_rb_channels_in_place(&mut bgra);
    Some(bgra)
}

pub fn try_capture_from_resident(
    state: &mut DeviceState,
    buf: &mut Vec<u8>,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> bool {
    let need = buf.len();
    let identity = surface_identity(state, mapping_id, width, height);
    let Some(bgra) = engine::read_resident_bgra(&identity, need) else {
        return false;
    };
    debug_assert_eq!(bgra.len(), need);
    // Move (not copy) the readback in; the untouched scratch returns to the pool.
    state.present.capture_scratch = std::mem::replace(buf, bgra);
    true
}
