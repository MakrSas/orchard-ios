//! Always-on proxy for the sRGB-downgrade bug class.
//!
//! # The class
//!
//! The guest names an sRGB render target or texture; the host binds the linear
//! sibling of that format. The hardware then never applies the sRGB transfer
//! function, so blending and sampling happen in the wrong colour space. The
//! defect is not that the fold exists — several rails genuinely carry raw bytes
//! and cannot encode — it is that the fold used to be **silent**: a lost format
//! qualifier looked exactly like a supported format, with nothing in the fail
//! log.
//!
//! # A zero here does not mean no rail dropped a qualifier
//!
//! Read this before excluding the sRGB family on the strength of an empty
//! census, because a session already did and was wrong.
//!
//! This census reports the sites listed in [`site`] and nothing else. It once
//! carried three more — the linear, mapper-ref-texture and ref-texture **zero-copy** sampled
//! rails — which were removed when those rails were changed to bind the
//! `_SRGB` view and honour the qualifier. They had no emitter for some time
//! before that, so the census was answering for a population it no longer
//! watched.
//!
//! The rail that used to be missing here was the **CPU sampled upload rung**,
//! `runtime::draw::vulkan`'s `SampledSourceRequest::Bytes` arm. It could not be
//! given a site constant, because a site needs the guest's declared format at
//! the moment of the fold and that variant carried a bare `TexelLayout` — into
//! which the qualifier had already been narrowed away by the producer. So the
//! rung downgraded every sRGB CPU upload silently, while the zero-copy rails
//! beside it bound the `_SRGB` view and decoded, and which of the two a bind
//! took was a cost decision.
//!
//! That variant now carries [`crate::protocol::pixel_format::SampledByteFormat`],
//! which pairs the layout with the source format, so the rung *honours* the
//! qualifier and there is nothing left to report for the eight-bit colour
//! orders. What remains is [`site::SAMPLED_BYTE_UPLOAD`]: a loader that
//! converted an sRGB texture into a layout with no sRGB spelling has moved the
//! values out of the encoding's domain, and that is a genuine loss this can
//! finally name.
//!
//! # Reading it
//!
//! `/tmp/reims-vgpu-fail.log`, always-on, one line per (site, format) pair:
//!
//! * `srgb_downgraded reason=srgb_downgraded site=<site> mtl=<fmt> …`
//!
//! **No lines on a healthy boot means the guest never asked for sRGB.** A line
//! is not itself a failure; it says which rail is trading colour-space
//! correctness away, which is the only thing needed to decide where adopting
//! `VK_FORMAT_*_SRGB` would pay. The pair is the unit because a rail hit twice
//! with the same format has nothing more to say the second time.
//!
//! # It watches one direction, and the open report is in the other
//!
//! Every site here reports the guest asking for sRGB and this device binding the
//! linear sibling — a *lost* encode. Nothing in this crate watches the opposite:
//! this device applying the transfer function where the guest did not ask for
//! it, or failing to decode on a read so an already-encoded value is encoded
//! again downstream. Both spell the same thing in the frame — a value lighter
//! than it should be — and neither can produce a line here.
//!
//! That is not hypothetical. `bugs/bug-03` is the macos-13 System Settings
//! sidebar icons, dark mode only, and it reproduces at `418eb35b` with **zero**
//! lines from this census. One driven macos-13 boot, System Settings
//! photographed in both appearances from the same window position, sampling the
//! peak-saturation pixel of each icon badge (17 patches, 51 channels), fitting
//! the dark reading against the light one:
//!
//! ```text
//! model                        RMS levels
//! no transfer function              99.68
//! sRGB encode applied once          48.49
//! sRGB encode applied twice          6.95
//! sRGB encode applied three times   20.72
//! best-fit single power law         54.94   (gamma 1.99)
//! ```
//!
//! So the dark frame is the light one with the sRGB encode applied **twice**,
//! and no single gamma reproduces it. Two things follow. The exponent count is
//! *two*, where `kb/the-settings-icons-are-one-extra-srgb-encode-…` fitted one —
//! that kb averaged a patch per badge, which pulls in antialiased edges and
//! flattens the curve, so the peak-pixel fit is the sharper of the two and the
//! search is for two sites or for one value going round twice. And a defect of
//! this size lives entirely outside this census's field of view, which is the
//! more useful half: a zero here excludes the downgrade direction and nothing
//! else.
//!
//! What *is* excluded, measured on the same boot: the scanout, present and
//! host-window path is colour-exact. The guest's own `screencapture` and this
//! device's host-window capture, taken seconds apart, agree to within two
//! levels on every desktop-wallpaper point no window covers — in both
//! appearances. (`screencapture` from an ssh session has no Screen Recording
//! consent, so it returns the desktop and menu bar and no window contents; the
//! wallpaper is what makes it a usable control rather than a failed probe.) The
//! extra encodes are applied to that window's own content, upstream of present.
//!
//! Measure-only: nothing here gates decode, execute or present.

use std::collections::BTreeSet;
use std::sync::Mutex;

use crate::observe;

/// The slug every downgrade line carries. Kept equal to
/// `TranslateReason::SrgbDowngraded`'s slug by a unit test in the Vulkan
/// backend, so the typed reason and the always-on line cannot drift apart.
pub const SRGB_DOWNGRADED_SLUG: &str = "srgb_downgraded";

/// The rails that can drop an sRGB qualifier, each named for the code path a
/// reader would open. One constant per site so a log line points somewhere
/// specific rather than at "the sampled path".
pub mod site {
    /// `build_secondary_targets` — MRT colour attachment beyond slot 0.
    pub const SECONDARY_COLOR_TARGET: &str = "secondary_color_target";
    /// `translate::pixel::vk_sampled_bytes` — a CPU loader handed back bytes
    /// that are still sRGB-encoded in a layout with no sRGB spelling, so the
    /// linear one is bound and the hardware will not decode.
    ///
    /// The CPU upload rails do not otherwise appear here any more: they carry
    /// the source format alongside the layout and the fold honours it. This
    /// fires only where honouring it is not expressible.
    pub const SAMPLED_BYTE_UPLOAD: &str = "sampled_byte_upload";

    /// Every site, for the completeness test. A new site constant that is not
    /// listed here is one the census cannot report on.
    pub const ALL: &[&str] = &[SECONDARY_COLOR_TARGET, SAMPLED_BYTE_UPLOAD];
}

/// `(site, MTLPixelFormat)` pairs already reported, so a per-draw rail costs one
/// line per distinct pair per boot. Bounded by `site::ALL` times the small set
/// of sRGB formats.
static SEEN: Mutex<BTreeSet<(&'static str, u16)>> = Mutex::new(BTreeSet::new());

/// Record that `site` bound the linear sibling of the sRGB format `mtl`.
///
/// Call this at the moment the qualifier is dropped, not at the moment the
/// format is decoded — the point is to name what was actually traded away.
pub fn note_downgrade(site: &'static str, mtl: u16) {
    let first_sight = SEEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((site, mtl));
    if first_sight {
        observe::fail(format!(
            "srgb_downgraded reason={SRGB_DOWNGRADED_SLUG} site={site} mtl={mtl:#x} \
             (bound the linear sibling; hardware will not apply the sRGB transfer \
             function on this rail)"
        ));
    }
}

/// Drop the first-sight set. Test-only: it is process-global, so a test that
/// asserts a line was emitted must start from a known point.
#[cfg(test)]
pub fn reset_for_tests() {
    SEEN.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::pixel_format::{MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA8_UNORM_SRGB};

    /// A per-draw rail must cost one line per distinct (site, format) pair, not
    /// one per bind — the dedup is what makes it safe to leave on forever, and
    /// a second pair on the same site is still a new event.
    #[test]
    fn each_site_and_format_pair_reports_once() {
        reset_for_tests();
        assert!(SEEN
            .lock()
            .unwrap()
            .insert((site::SAMPLED_BYTE_UPLOAD, MTL_FORMAT_BGRA8_UNORM_SRGB)));
        for _ in 0..64 {
            note_downgrade(site::SAMPLED_BYTE_UPLOAD, MTL_FORMAT_BGRA8_UNORM_SRGB);
        }
        assert_eq!(SEEN.lock().unwrap().len(), 1, "64 binds, one pair");
        note_downgrade(site::SAMPLED_BYTE_UPLOAD, MTL_FORMAT_RGBA8_UNORM_SRGB);
        note_downgrade(site::SECONDARY_COLOR_TARGET, MTL_FORMAT_RGBA8_UNORM_SRGB);
        assert_eq!(
            SEEN.lock().unwrap().len(),
            3,
            "a new format and a new site are each a new event"
        );
        reset_for_tests();
    }

    /// Site names are distinct and log-safe — a duplicate would merge two
    /// rails' counts and a space would break the field split.
    #[test]
    fn site_names_are_distinct_and_log_safe() {
        let mut names = site::ALL.to_vec();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate site name");
        for name in site::ALL {
            assert!(name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
        }
    }
}
