//! Aspect-fit viewport math, shared by every rail's presenter (where the guest
//! frame lands inside the drawable) and by the window's input path (where host
//! pointer coordinates map back into guest coordinates).
//!
//! Presentation and pointer translation MUST move through the same transform
//! as one unit: the rolled-back aspect-fit experiment stayed visible but left
//! clicks in full-window coordinates, so every pointer event was offset and
//! scaled against the letterboxed guest viewport ([[host-window]]).
//!
//! Two rails and an input path are three callers, and the two rails are below
//! the window — which is why this is under [`super`] and not beside the loop
//! that drives it.

/// Where a `src`-sized frame lands inside a `dst`-sized drawable: the largest
/// centered rectangle with `src`'s aspect ratio that fits `dst`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Viewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    /// True when the viewport is exactly the whole destination — no letterbox
    /// bars exist and full-window presentation/input transforms apply.
    pub fn covers(&self, dst: (u32, u32)) -> bool {
        self.x == 0 && self.y == 0 && self.width == dst.0 && self.height == dst.1
    }
}

/// Largest centered `src`-aspect rectangle inside `dst`. Zero dimensions are
/// clamped to 1 so degenerate transition frames can never divide by zero or
/// produce an empty blit.
pub fn aspect_fit(src: (u32, u32), dst: (u32, u32)) -> Viewport {
    let (sw, sh) = (src.0.max(1) as u64, src.1.max(1) as u64);
    let (dw, dh) = (dst.0.max(1) as u64, dst.1.max(1) as u64);
    let (width, height) = if sw * dh >= sh * dw {
        // Source is at least as wide as the destination: fill the width.
        (dw, (sh * dw / sw).max(1))
    } else {
        ((sw * dh / sh).max(1), dh)
    };
    Viewport {
        x: ((dw - width) / 2) as u32,
        y: ((dh - height) / 2) as u32,
        width: width as u32,
        height: height as u32,
    }
}

/// Map a window-space pointer position into guest-frame pixels through the
/// same viewport the presenter draws. Positions inside a letterbox bar clamp
/// to the nearest viewport edge, matching what the user sees under the cursor.
pub fn pointer_to_guest(pos: (f64, f64), window: (u32, u32), guest: (u32, u32)) -> (u32, u32) {
    let vp = aspect_fit(guest, window);
    let gx = (pos.0 - vp.x as f64) * guest.0.max(1) as f64 / vp.width.max(1) as f64;
    let gy = (pos.1 - vp.y as f64) * guest.1.max(1) as f64 / vp.height.max(1) as f64;
    (
        (gx.max(0.0) as u32).min(guest.0.saturating_sub(1)),
        (gy.max(0.0) as u32).min(guest.1.saturating_sub(1)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_aspect_covers_the_destination() {
        let vp = aspect_fit((1440, 1080), (1440, 1080));
        assert_eq!(
            vp,
            Viewport {
                x: 0,
                y: 0,
                width: 1440,
                height: 1080
            }
        );
        assert!(vp.covers((1440, 1080)));
        // Same aspect at a different size still covers (scaled, no bars).
        assert!(aspect_fit((1920, 1080), (960, 540)).covers((960, 540)));
    }

    #[test]
    fn narrower_guest_pillarboxes_centered() {
        // 4:3 guest in a 16:9 drawable: full height, centered horizontally.
        let vp = aspect_fit((1440, 1080), (1920, 1080));
        assert_eq!(
            vp,
            Viewport {
                x: 240,
                y: 0,
                width: 1440,
                height: 1080
            }
        );
        assert!(!vp.covers((1920, 1080)));
    }

    #[test]
    fn wider_guest_letterboxes_centered() {
        // 16:9 guest in a 4:3 drawable: full width, centered vertically.
        let vp = aspect_fit((1920, 1080), (1440, 1080));
        assert_eq!(
            vp,
            Viewport {
                x: 0,
                y: 135,
                width: 1440,
                height: 810
            }
        );
    }

    #[test]
    fn pointer_maps_through_the_viewport_and_clamps_bars() {
        // Full-cover window: identity mapping.
        assert_eq!(
            pointer_to_guest((720.0, 540.0), (1440, 1080), (1440, 1080)),
            (720, 540)
        );
        // Pillarboxed 4:3 in 16:9: the viewport starts at x=240.
        assert_eq!(
            pointer_to_guest((240.0, 0.0), (1920, 1080), (1440, 1080)),
            (0, 0)
        );
        assert_eq!(
            pointer_to_guest((960.0, 540.0), (1920, 1080), (1440, 1080)),
            (720, 540)
        );
        // Inside the left bar: clamps to the viewport's left edge.
        assert_eq!(
            pointer_to_guest((10.0, 540.0), (1920, 1080), (1440, 1080)),
            (0, 540)
        );
        // Past the right edge: clamps to the last guest pixel.
        assert_eq!(
            pointer_to_guest((1919.0, 1079.0), (1920, 1080), (1440, 1080)),
            (1439, 1079)
        );
    }

    #[test]
    fn degenerate_sizes_stay_in_range() {
        let vp = aspect_fit((0, 0), (1920, 1080));
        assert!(vp.width >= 1 && vp.height >= 1);
        assert_eq!(pointer_to_guest((5.0, 5.0), (10, 10), (0, 0)), (0, 0));
    }
}
