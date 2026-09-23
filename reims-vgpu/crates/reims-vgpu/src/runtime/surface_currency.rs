//! Whether a host-side copy of a mapper-ref-texture surface's pixels is still
//! that surface's content.
//!
//! # The question
//!
//! A mapper-ref-texture surface's pixels are plain guest RAM. The guest CPU
//! stores into them with no device operation at all, so nothing in this
//! device's own command stream marks the moment a host-side copy — a render
//! resident, the [`crate::runtime::surface_cache`] frame, an attachment LOAD
//! seed — stopped being the surface. The hypervisor's dirty bitmap is the only
//! witness, and [`crate::runtime::mapper::mapping_guest_write_verdict`] is how
//! this device reads it.
//!
//! Every consumer of a host-side copy has to ask this before serving one, and
//! there is exactly one right answer per `(mapping, window)` — so it is asked
//! here, once, and each rail names the outcome under its own census keys.
//!
//! # Why it is two stages and not one
//!
//! The tracking token covers the mapping's whole page list, and its generation
//! moves for a write to any page in it. A backing allocation is bigger than the
//! plane a bind samples: `mapper_ref_texture_sample_window` reports a `base_off`
//! precisely because the pixels do not start at offset 0, and an allocation can
//! carry a second plane and end padding past `span_end`. Refusing on the coarse
//! answer alone discarded whole 1920x1080 compositor scanouts whose pixels the
//! GPU had rendered and nothing had touched — measured live as a black desktop
//! at 17 Hz, against 120 Hz and a painted one on the same boot script with the
//! narrowing in place.
//!
//! So the coarse stage decides whether the page-list walk is worth paying for,
//! and the walk decides the answer. The walk is paid on the minority of asks
//! the coarse stage flags.
//!
//! # Which answers are evidence
//!
//! Only [`crate::runtime::mapper::GuestWriteVerdict::Wrote`] is evidence of a
//! guest write. `NoStamp` says "nobody asked the host to watch these pages",
//! which is a statement about this device's arming and not about the guest; on
//! the boot that first measured the ladder it was 14 092 of 14 396 cache binds,
//! so refusing on it would turn every host-side copy off on the strength of a
//! rail that was never armed.
//!
//! The narrowing fails closed in the other direction. Everything the host
//! cannot answer exactly — no token, no enumerable page list, no resolvable
//! sample window, or written GPAs this mapping does not own — is
//! [`SurfaceCurrency::WroteUnknown`], which does not serve. Serving a stale copy
//! is a wrong frame that is then *held*, because the pass composites onto the
//! seed and its Store publishes the composite back over the guest's pages;
//! re-reading the guest's pages costs a copy.

use crate::model::DeviceState;
use crate::protocol::pixel_format;
use crate::runtime::host::HostOps;
use crate::runtime::mapper::{self, GuestWriteVerdict};

/// What the host's dirty tracking says about one pixel window of one mapping,
/// since the Store that stamped this device's copies of it.
///
/// The serving rule is [`Self::serves`] and lives on the type, so a consumer
/// cannot spell it a fourth way: the three call sites that predate this each
/// wrote their own `!matches!(…)` and one of them had already drifted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfaceCurrency {
    /// The host has no evidence of a guest write to this allocation since the
    /// stamp. Carries the verdict that established it, because "the guest never
    /// rewrites its surfaces" and "the witness was never armed" are the same
    /// answer here and completely different findings.
    Unwritten(GuestWriteVerdict),
    /// The guest wrote the allocation, but every written page falls outside this
    /// pixel window — a header, a sibling plane, or end padding.
    WroteElsewhere,
    /// The guest wrote pages inside this pixel window. Carries the
    /// mapping-offset ranges the guest now owns, ascending and merged, which is
    /// exactly the `skip` list a merge that must preserve the guest's stores
    /// needs.
    WrotePixels(Vec<(u64, u64)>),
    /// The guest wrote the allocation and the host cannot name where.
    /// Indistinguishable from [`Self::WrotePixels`] to a consumer that must be
    /// right, but it cannot be merged either — there is no page list to
    /// preserve.
    WroteUnknown,
}

/// The evidence a consumer requires of the witness before it will serve a
/// host-side copy.
///
/// Named at the ask rather than assumed, because the two standards are both
/// correct and choosing wrongly is invisible: on a pathway where the witness
/// never arms, every answer is
/// [`GuestWriteVerdict::NoStamp`], and the two standards then mean "serve
/// every time" and "serve never".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurrencyStandard {
    /// Absence of evidence to the contrary: serve unless the host actually
    /// watched a write land in this window.
    ///
    /// For a consumer that has other corroboration, or whose miss costs
    /// content the guest asked for. `NoStamp` says "nobody asked the host to
    /// watch these pages", which is a statement about this device's arming and
    /// not about the guest; on the boot that first measured the Vulkan sampled
    /// ladder it was 14 092 of 14 396 cache binds, so refusing on it would turn
    /// the rung off on the strength of a rail that was never armed.
    NoContraryEvidence,
    /// Positive evidence: the host watched these pages across the window and
    /// saw nothing written.
    ///
    /// For a consumer whose only check this is. An attachment LOAD seed served
    /// from a host-side copy has no rung under it to correct a stale serve —
    /// the pass composites onto the seed and its Store publishes the composite
    /// back over the guest's pages, so a wrong frame is *held*. Its miss costs
    /// a guest re-read and nothing else, which is the trade
    /// [`NoContraryEvidence`](Self::NoContraryEvidence) cannot make.
    WatchedAndUnwritten,
}

impl SurfaceCurrency {
    /// Whether a host-side copy of this window may be served as the surface's
    /// content, under the evidence standard the consumer states.
    ///
    /// [`Self::WroteElsewhere`] serves under both: the guest wrote the
    /// allocation and not these pixels, and discarding the copy on that is the
    /// 17 Hz black desktop the module doc records. The standards differ only on
    /// the answers that are not about the guest at all — no mapping, no stamp,
    /// an unreadable token — which
    /// [`CurrencyStandard::NoContraryEvidence`] admits and
    /// [`CurrencyStandard::WatchedAndUnwritten`] does not.
    pub fn serves(&self, standard: CurrencyStandard) -> bool {
        match standard {
            CurrencyStandard::NoContraryEvidence => {
                matches!(self, Self::Unwritten(_) | Self::WroteElsewhere)
            }
            CurrencyStandard::WatchedAndUnwritten => matches!(
                self,
                Self::Unwritten(GuestWriteVerdict::Clean) | Self::WroteElsewhere
            ),
        }
    }

    /// The coarse verdict this answer was reached under.
    ///
    /// Derived rather than stored twice: every variant but [`Self::Unwritten`]
    /// exists only because the verdict was
    /// [`GuestWriteVerdict::Wrote`], so a second field holding it could
    /// contradict the variant. Census that reports "which state was a copy
    /// served under" reads it here, so the coarse column keeps its meaning while
    /// the serving decision uses the narrowed one.
    pub fn verdict(&self) -> GuestWriteVerdict {
        match self {
            Self::Unwritten(verdict) => *verdict,
            Self::WroteElsewhere | Self::WrotePixels(_) | Self::WroteUnknown => {
                GuestWriteVerdict::Wrote
            }
        }
    }

    /// The mapping-offset ranges the guest's own stores now own, when the host
    /// could name them.
    ///
    /// Only [`Self::WrotePixels`] has them. [`Self::WroteUnknown`] deliberately
    /// answers `None` rather than an empty slice: an empty skip list reads as
    /// "the guest owns nothing", which is the opposite of what that variant
    /// means.
    pub fn guest_owned_ranges(&self) -> Option<&[(u64, u64)]> {
        match self {
            Self::WrotePixels(ranges) => Some(ranges.as_slice()),
            _ => None,
        }
    }
}

/// Ask the hypervisor's witness whether this device's host-side copies of
/// `mapping_id`'s `width` x `height` pixel window are still that window.
///
/// `width`/`height` are the geometry the consumer is about to serve at, not the
/// mapping's latched geometry, because the window they resolve is what decides
/// [`SurfaceCurrency::WroteElsewhere`].
pub fn surface_currency<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> SurfaceCurrency {
    let verdict = mapper::mapping_guest_write_verdict(state, host, mapping_id);
    if !matches!(verdict, GuestWriteVerdict::Wrote) {
        return SurfaceCurrency::Unwritten(verdict);
    }
    narrow_to_window(state, host, mapping_id, width, height)
}

/// The second stage: where the writes landed relative to the pixel window.
fn narrow_to_window<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> SurfaceCurrency {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return SurfaceCurrency::WroteUnknown;
    };
    let format = if m.format != 0 {
        m.format
    } else {
        pixel_format::MTL_FORMAT_BGRA8_UNORM
    };
    let Some((base_off, _bpr, span_end)) =
        crate::runtime::mapping_write::mapper_ref_texture_sample_window(m, width, height, format)
    else {
        return SurfaceCurrency::WroteUnknown;
    };
    let Some(pages) = host.guest_written_pages(m.guest_write_token, m.guest_write_gen_at_store)
    else {
        return SurfaceCurrency::WroteUnknown;
    };
    let ranges = mapper::mapping_offsets_of_pages(state, mapping_id, &pages);
    if ranges.is_empty() {
        // The set-wide generation moved, so some page of this list was written,
        // yet none of them mapped back to an offset. That is a disagreement
        // between the token and the page list this call resolved against, not a
        // finding about the guest.
        return SurfaceCurrency::WroteUnknown;
    }
    if ranges_touch_window(&ranges, base_off, span_end) {
        SurfaceCurrency::WrotePixels(ranges)
    } else {
        SurfaceCurrency::WroteElsewhere
    }
}

/// Whether any mapping-offset range intersects `[base_off, span_end)`.
///
/// Both bounds half-open, so a range that abuts the plane's end is outside it.
fn ranges_touch_window(ranges: &[(u64, u64)], base_off: u64, span_end: u64) -> bool {
    ranges
        .iter()
        .any(|&(start, end)| start < span_end && end > base_off)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coarse stage admits everything that is not positive evidence.
    ///
    /// Both directions are asserted deliberately. Refusing more than `Wrote`
    /// would be just as wrong: `NoStamp` means this device never armed the
    /// witness, and turning every host-side copy off on that answer would send
    /// consumers to the guest's pages for surfaces whose content the deferred
    /// writeback rail has not landed there yet.
    #[test]
    fn only_a_watched_guest_write_refuses_a_host_side_copy() {
        for verdict in [
            GuestWriteVerdict::Clean,
            GuestWriteVerdict::NoMapping,
            GuestWriteVerdict::NoStamp,
            GuestWriteVerdict::Unreadable,
        ] {
            assert!(
                SurfaceCurrency::Unwritten(verdict).serves(CurrencyStandard::NoContraryEvidence),
                "{verdict:?} is not evidence of a guest write and must not refuse a copy"
            );
        }
        for state in [
            SurfaceCurrency::WrotePixels(vec![(0, 4096)]),
            SurfaceCurrency::WroteUnknown,
        ] {
            for standard in [
                CurrencyStandard::NoContraryEvidence,
                CurrencyStandard::WatchedAndUnwritten,
            ] {
                assert!(
                    !state.serves(standard),
                    "{state:?} is a write this device cannot rule out of the window"
                );
            }
        }
        for standard in [
            CurrencyStandard::NoContraryEvidence,
            CurrencyStandard::WatchedAndUnwritten,
        ] {
            assert!(
                SurfaceCurrency::WroteElsewhere.serves(standard),
                "a write outside the window leaves the window intact under {standard:?}"
            );
        }
    }

    /// The stricter standard admits only positive evidence, and the three
    /// answers it drops are exactly the ones that are about this device's arming
    /// rather than about the guest.
    ///
    /// This is the whole reason the standard is a parameter. A consumer whose
    /// only check this is cannot spend `NoStamp` — on a pathway where the
    /// hypervisor's witness never arms, every ask answers `NoStamp` and
    /// `NoContraryEvidence` degenerates into "serve whatever the cache holds".
    #[test]
    fn the_strict_standard_spends_only_a_watched_clean_answer() {
        assert!(SurfaceCurrency::Unwritten(GuestWriteVerdict::Clean)
            .serves(CurrencyStandard::WatchedAndUnwritten));
        for verdict in [
            GuestWriteVerdict::NoMapping,
            GuestWriteVerdict::NoStamp,
            GuestWriteVerdict::Unreadable,
        ] {
            assert!(
                !SurfaceCurrency::Unwritten(verdict).serves(CurrencyStandard::WatchedAndUnwritten),
                "{verdict:?} is not a watched answer and must not satisfy the strict standard"
            );
        }
    }

    /// Only the variant that can name the guest's bytes offers a skip list. An
    /// empty slice from `WroteUnknown` would read as "the guest owns nothing",
    /// which is the opposite of what it means.
    #[test]
    fn a_skip_list_comes_only_from_named_pages() {
        assert_eq!(
            SurfaceCurrency::WrotePixels(vec![(0, 4096)]).guest_owned_ranges(),
            Some(&[(0u64, 4096u64)][..])
        );
        assert_eq!(SurfaceCurrency::WroteUnknown.guest_owned_ranges(), None);
        assert_eq!(SurfaceCurrency::WroteElsewhere.guest_owned_ranges(), None);
        assert_eq!(
            SurfaceCurrency::Unwritten(GuestWriteVerdict::Clean).guest_owned_ranges(),
            None
        );
    }

    /// The narrowing itself, which is what keeps the coarse stage from being
    /// ruinous. A backing allocation is bigger than the plane a bind samples —
    /// pixels start at `base_off` and padding follows `span_end` — and the
    /// tracking token's generation moves for a write to any page of it.
    #[test]
    fn a_guest_write_outside_the_window_leaves_the_window_current() {
        // A 1920x1080 BGRA8 plane one page into its allocation.
        const BASE: u64 = 4096;
        const END: u64 = BASE + 1920 * 1080 * 4;
        // The header page before the plane is not the pixels.
        assert!(!ranges_touch_window(&[(0, 4096)], BASE, END));
        // Nor is padding after it.
        assert!(!ranges_touch_window(&[(END + 4096, END + 8192)], BASE, END));
        // Abutting the end exactly is still outside — both bounds half-open.
        assert!(!ranges_touch_window(&[(END, END + 4096)], BASE, END));
        // One page anywhere inside the plane is the whole finding.
        assert!(ranges_touch_window(&[(4_198_400, 4_202_496)], BASE, END));
        // A range straddling the plane's first byte counts.
        assert!(ranges_touch_window(&[(0, 8192)], BASE, END));
        // Outside ranges do not mask an inside one.
        assert!(ranges_touch_window(
            &[(0, 4096), (4_198_400, 4_202_496), (END, END + 4096)],
            BASE,
            END
        ));
        assert!(!ranges_touch_window(&[], BASE, END));
    }
}
