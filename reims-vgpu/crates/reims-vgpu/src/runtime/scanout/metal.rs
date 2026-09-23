//! What the Metal rail's resident registry can say about a present surface.
//!
//! The sibling of [`super::vulkan`], and the same two questions asked for the
//! same two reasons: *would* a resident carry this present (a census/failure
//! split at the drain), and *does* one, hand me its bytes (the capture). Both
//! resolve the surface through
//! [`crate::backend::metal::resident::ResidentColorKey::for_surface`] and the
//! generation [`crate::runtime::surface_cache::frame_generation`] names, so a
//! second spelling of either in one of them would report a frame as carried that
//! the other then cannot find.
//!
//! # Why this rail needed one at all
//!
//! Until the Store learned to cede
//! ([`crate::runtime::mapping_write::FramePublication`]) the Metal rail's
//! present capture was always served by the BGRA8 host cache, and
//! `Backend::try_capture_from_resident`'s default `false` was reached only when
//! that cache had no entry — which the doc on the default called "the known
//! arm/Metal breakage this pathway already carries". A ceded frame lives in the
//! resident and nowhere else host-side, so the default would now be reached on
//! *every* present of a rendered surface and the console would hold its prior
//! retain forever.
//!
//! Reached only through [`crate::backend::Backend`]; the drain and the capture
//! never name this rail.

use crate::backend::metal::resident::{self, ResidentColorKey};
use crate::model::DeviceState;

/// Would a resident carry the present this mapping names, at this geometry?
///
/// `Some(true)` when this rail holds a target for the mapping whose pixels are
/// the frame the surface cache currently names, `Some(false)` otherwise — so a
/// present with no host-side frame behind it is reported as the loss it is
/// rather than demoted to a census.
///
/// Asked of the same key and the same generation [`try_capture_from_resident`]
/// reads with. Sharing them is the point rather than tidiness: a looser
/// predicate here would report a frame as carried that the capture then cannot
/// produce, which is a disagreement neither call site can see on its own.
pub fn present_resident_carries(
    state: &DeviceState,
    mapping: u32,
    width: u32,
    height: u32,
) -> Option<bool> {
    let Some(generation) =
        crate::runtime::surface_cache::frame_generation(state, mapping, width, height)
    else {
        return Some(false);
    };
    let key = ResidentColorKey::for_surface(mapping, width, height);
    Some(resident::borrow_published(&key, generation).is_some())
}

/// Fill `buf` from the mapping's resident colour target, without any guest-page
/// scatter.
///
/// On `true` `buf` holds tight BGRA8; on `false` `buf` is untouched. A miss is
/// an expected steady-state condition (a cold mapping, a target the byte budget
/// evicted, a Store that published to the host cache instead), so it is the
/// caller's `capture_source` census rather than a line per present.
pub fn try_capture_from_resident(
    state: &mut DeviceState,
    buf: &mut Vec<u8>,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> bool {
    let need = buf.len();
    let Some(generation) =
        crate::runtime::surface_cache::frame_generation(state, mapping_id, width, height)
    else {
        return false;
    };
    let key = ResidentColorKey::for_surface(mapping_id, width, height);
    let Some(mut rgba) = resident::read_published_rgba8(&key, generation) else {
        return false;
    };
    if rgba.len() != need {
        // The caller sized `buf` from the same geometry this key carries, so a
        // disagreement is a contract break between the two rather than a guest
        // behaviour, and serving a truncated frame would show as a torn console
        // with nothing saying why.
        crate::observe::fail(format!(
            "metal_capture_resident reason=length_mismatch mid={mapping_id} \
             {width}x{height} have={} need={need}",
            rgba.len()
        ));
        return false;
    }
    // The resident is RGBA8 and the console is BGRA8 — see `read_published_rgba8`
    // for why the order is named rather than assumed. In place, because this
    // readback is already this function's own frame.
    crate::runtime::draw::swap_rb_channels_in_place(&mut rgba);
    // Move (not copy) the readback in; the untouched scratch returns to the pool.
    state.present.capture_scratch = std::mem::replace(buf, rgba);
    true
}
