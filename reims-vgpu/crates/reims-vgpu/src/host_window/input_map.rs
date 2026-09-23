//! Pure winit-event → neutral-input mapping for the host-owned window
//! ([[host-window]]). No window/thread state here — just total functions from
//! winit input types to the wire forms the C shim replays (`qemu_input_*`):
//!
//! - keys → Linux **evdev** keycodes (`KEY_*`); the shim forwards them to
//!   `qemu_input_event_send_key_linux`, which owns evdev→qcode.
//! - buttons → [`ReimsVgpuButton`]; the shim maps to QEMU `InputButton`.
//! - scroll deltas → a sequence of wheel [`ReimsVgpuButton`] notches (the producer
//!   emits a down+up pair per notch).
//!
//! Unmapped inputs return `None`/empty — never an invented code (unknown wire
//! stays unknown; the producer logs the drop). evdev values are the Linux
//! `input-event-codes.h` ABI (a stable contract, not magic constants).

use winit::event::{MouseButton, MouseScrollDelta};
use winit::keyboard::KeyCode;

use crate::runtime::host::HostAction;
use crate::runtime::input::ReimsVgpuButton;

/// One trackpad/wheel notch is this many pixels of a `PixelDelta` scroll. winit
/// reports high-resolution pixel deltas on precise devices; we quantise to
/// discrete wheel clicks (the guest's pointer only understands notches). 40 px
/// ≈ one line on a typical device; tuned later against a real trackpad.
const PIXELS_PER_NOTCH: f64 = 40.0;

/// Map a winit physical [`KeyCode`] to a Linux evdev keycode (`KEY_*`), or
/// `None` when we don't carry it (the producer drops + logs; QEMU would drop an
/// unknown evdev code anyway). Physical (not logical) keys so the guest keymap
/// owns layout, exactly like a real USB keyboard.
pub fn keycode_to_evdev(code: KeyCode) -> Option<u32> {
    use KeyCode as K;
    Some(match code {
        // Letters (KEY_A = 30, scattered — explicit per key).
        K::KeyA => 30,
        K::KeyB => 48,
        K::KeyC => 46,
        K::KeyD => 32,
        K::KeyE => 18,
        K::KeyF => 33,
        K::KeyG => 34,
        K::KeyH => 35,
        K::KeyI => 23,
        K::KeyJ => 36,
        K::KeyK => 37,
        K::KeyL => 38,
        K::KeyM => 50,
        K::KeyN => 49,
        K::KeyO => 24,
        K::KeyP => 25,
        K::KeyQ => 16,
        K::KeyR => 19,
        K::KeyS => 31,
        K::KeyT => 20,
        K::KeyU => 22,
        K::KeyV => 47,
        K::KeyW => 17,
        K::KeyX => 45,
        K::KeyY => 21,
        K::KeyZ => 44,
        // Number row (KEY_1 = 2 .. KEY_0 = 11).
        K::Digit1 => 2,
        K::Digit2 => 3,
        K::Digit3 => 4,
        K::Digit4 => 5,
        K::Digit5 => 6,
        K::Digit6 => 7,
        K::Digit7 => 8,
        K::Digit8 => 9,
        K::Digit9 => 10,
        K::Digit0 => 11,
        // Punctuation / symbols.
        K::Minus => 12,
        K::Equal => 13,
        K::BracketLeft => 26,
        K::BracketRight => 27,
        K::Backslash => 43,
        K::Semicolon => 39,
        K::Quote => 40,
        K::Backquote => 41,
        K::Comma => 51,
        K::Period => 52,
        K::Slash => 53,
        K::IntlBackslash => 86,
        // Editing / whitespace.
        K::Escape => 1,
        K::Backspace => 14,
        K::Tab => 15,
        K::Enter => 28,
        K::Space => 57,
        K::CapsLock => 58,
        // Modifiers.
        K::ControlLeft => 29,
        K::ShiftLeft => 42,
        K::AltLeft => 56,
        K::ShiftRight => 54,
        K::ControlRight => 97,
        K::AltRight => 100,
        K::SuperLeft => 125,
        K::SuperRight => 126,
        K::ContextMenu => 127,
        // Navigation cluster.
        K::Insert => 110,
        K::Delete => 111,
        K::Home => 102,
        K::End => 107,
        K::PageUp => 104,
        K::PageDown => 109,
        K::ArrowUp => 103,
        K::ArrowDown => 108,
        K::ArrowLeft => 105,
        K::ArrowRight => 106,
        // Function row (KEY_F1 = 59 .. F10 = 68, then F11/F12 = 87/88).
        K::F1 => 59,
        K::F2 => 60,
        K::F3 => 61,
        K::F4 => 62,
        K::F5 => 63,
        K::F6 => 64,
        K::F7 => 65,
        K::F8 => 66,
        K::F9 => 67,
        K::F10 => 68,
        K::F11 => 87,
        K::F12 => 88,
        // System / locks.
        K::PrintScreen => 99,
        K::ScrollLock => 70,
        K::Pause => 119,
        K::NumLock => 69,
        // Keypad (KEY_KP7 = 71 …).
        K::Numpad7 => 71,
        K::Numpad8 => 72,
        K::Numpad9 => 73,
        K::NumpadSubtract => 74,
        K::Numpad4 => 75,
        K::Numpad5 => 76,
        K::Numpad6 => 77,
        K::NumpadAdd => 78,
        K::Numpad1 => 79,
        K::Numpad2 => 80,
        K::Numpad3 => 81,
        K::Numpad0 => 82,
        K::NumpadDecimal => 83,
        K::NumpadEnter => 96,
        K::NumpadDivide => 98,
        K::NumpadMultiply => 55,
        _ => return None,
    })
}

/// Map a winit [`MouseButton`] to a neutral [`ReimsVgpuButton`]. `Other(_)` and any
/// button we have no neutral code for return `None` (dropped, not invented).
pub fn mouse_button(button: MouseButton) -> Option<ReimsVgpuButton> {
    Some(match button {
        MouseButton::Left => ReimsVgpuButton::Left,
        MouseButton::Right => ReimsVgpuButton::Right,
        MouseButton::Middle => ReimsVgpuButton::Middle,
        MouseButton::Back => ReimsVgpuButton::Side,
        MouseButton::Forward => ReimsVgpuButton::Extra,
        MouseButton::Other(_) => return None,
    })
}

/// Turn a scroll delta into a sequence of wheel-notch [`ReimsVgpuButton`]s (the
/// producer emits a down+up pair per element). Sign convention: positive y =
/// wheel up, positive x = wheel right — matching winit's up/right-positive
/// deltas. A sub-notch pixel delta still yields one notch if nonzero, so a slow
/// trackpad drag is never fully swallowed.
pub fn scroll_to_notches(delta: MouseScrollDelta) -> Vec<ReimsVgpuButton> {
    let (dx, dy) = match delta {
        MouseScrollDelta::LineDelta(x, y) => (f64::from(x), f64::from(y)),
        MouseScrollDelta::PixelDelta(pos) => (pos.x / PIXELS_PER_NOTCH, pos.y / PIXELS_PER_NOTCH),
    };
    let mut out = Vec::new();
    push_notches(
        &mut out,
        dy,
        ReimsVgpuButton::WheelUp,
        ReimsVgpuButton::WheelDown,
    );
    push_notches(
        &mut out,
        dx,
        ReimsVgpuButton::WheelRight,
        ReimsVgpuButton::WheelLeft,
    );
    out
}

/// Full `HostAction` sequence for a scroll delta: each wheel notch emitted as a
/// down+up pair (a wheel button is momentary — no held state), so the C shim
/// stays uniform (one `qemu_input_queue_btn` + sync per action). The window
/// event-loop feeds this straight onto the prompt action queue.
pub fn scroll_actions(delta: MouseScrollDelta) -> Vec<HostAction> {
    let mut out = Vec::new();
    for notch in scroll_to_notches(delta) {
        out.push(HostAction::input_pointer_button(notch, true));
        out.push(HostAction::input_pointer_button(notch, false));
    }
    out
}

/// Append `|delta|` rounded-up-to-≥1-if-nonzero notches of `pos`/`neg` by sign.
fn push_notches(
    out: &mut Vec<ReimsVgpuButton>,
    delta: f64,
    pos: ReimsVgpuButton,
    neg: ReimsVgpuButton,
) {
    if delta == 0.0 {
        return;
    }
    let btn = if delta > 0.0 { pos } else { neg };
    let n = delta.abs().round().max(1.0) as usize;
    for _ in 0..n {
        out.push(btn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::dpi::PhysicalPosition;

    /// Anchor a few keys against their known evdev values, and confirm an
    /// unmapped key returns None (no invented fallback code).
    #[test]
    fn keycode_evdev_anchors() {
        assert_eq!(keycode_to_evdev(KeyCode::KeyA), Some(30));
        assert_eq!(keycode_to_evdev(KeyCode::Digit1), Some(2));
        assert_eq!(keycode_to_evdev(KeyCode::Digit0), Some(11));
        assert_eq!(keycode_to_evdev(KeyCode::Enter), Some(28));
        assert_eq!(keycode_to_evdev(KeyCode::Space), Some(57));
        assert_eq!(keycode_to_evdev(KeyCode::Escape), Some(1));
        assert_eq!(keycode_to_evdev(KeyCode::ArrowUp), Some(103));
        assert_eq!(keycode_to_evdev(KeyCode::F1), Some(59));
        assert_eq!(keycode_to_evdev(KeyCode::F12), Some(88));
        assert_eq!(keycode_to_evdev(KeyCode::ControlLeft), Some(29));
        assert_eq!(keycode_to_evdev(KeyCode::SuperLeft), Some(125));
        // A physical key we deliberately don't carry.
        assert_eq!(keycode_to_evdev(KeyCode::F20), None);
    }

    /// No two mapped keys collide on the same evdev code (a duplicate would make
    /// one physical key type as another).
    #[test]
    fn evdev_mapping_is_injective() {
        // Exercise the whole hand-written table via winit's key list.
        let keys = [
            KeyCode::KeyA,
            KeyCode::KeyB,
            KeyCode::KeyC,
            KeyCode::KeyD,
            KeyCode::KeyE,
            KeyCode::KeyF,
            KeyCode::KeyG,
            KeyCode::KeyH,
            KeyCode::KeyI,
            KeyCode::KeyJ,
            KeyCode::KeyK,
            KeyCode::KeyL,
            KeyCode::KeyM,
            KeyCode::KeyN,
            KeyCode::KeyO,
            KeyCode::KeyP,
            KeyCode::KeyQ,
            KeyCode::KeyR,
            KeyCode::KeyS,
            KeyCode::KeyT,
            KeyCode::KeyU,
            KeyCode::KeyV,
            KeyCode::KeyW,
            KeyCode::KeyX,
            KeyCode::KeyY,
            KeyCode::KeyZ,
            KeyCode::Digit1,
            KeyCode::Digit2,
            KeyCode::Digit3,
            KeyCode::Digit4,
            KeyCode::Digit5,
            KeyCode::Digit6,
            KeyCode::Digit7,
            KeyCode::Digit8,
            KeyCode::Digit9,
            KeyCode::Digit0,
            KeyCode::Minus,
            KeyCode::Equal,
            KeyCode::BracketLeft,
            KeyCode::BracketRight,
            KeyCode::Backslash,
            KeyCode::Semicolon,
            KeyCode::Quote,
            KeyCode::Backquote,
            KeyCode::Comma,
            KeyCode::Period,
            KeyCode::Slash,
            KeyCode::IntlBackslash,
            KeyCode::Escape,
            KeyCode::Backspace,
            KeyCode::Tab,
            KeyCode::Enter,
            KeyCode::Space,
            KeyCode::CapsLock,
            KeyCode::ControlLeft,
            KeyCode::ShiftLeft,
            KeyCode::AltLeft,
            KeyCode::ShiftRight,
            KeyCode::ControlRight,
            KeyCode::AltRight,
            KeyCode::SuperLeft,
            KeyCode::SuperRight,
            KeyCode::ContextMenu,
            KeyCode::Insert,
            KeyCode::Delete,
            KeyCode::Home,
            KeyCode::End,
            KeyCode::PageUp,
            KeyCode::PageDown,
            KeyCode::ArrowUp,
            KeyCode::ArrowDown,
            KeyCode::ArrowLeft,
            KeyCode::ArrowRight,
            KeyCode::F1,
            KeyCode::F2,
            KeyCode::F3,
            KeyCode::F4,
            KeyCode::F5,
            KeyCode::F6,
            KeyCode::F7,
            KeyCode::F8,
            KeyCode::F9,
            KeyCode::F10,
            KeyCode::F11,
            KeyCode::F12,
            KeyCode::PrintScreen,
            KeyCode::ScrollLock,
            KeyCode::Pause,
            KeyCode::NumLock,
            KeyCode::Numpad7,
            KeyCode::Numpad8,
            KeyCode::Numpad9,
            KeyCode::NumpadSubtract,
            KeyCode::Numpad4,
            KeyCode::Numpad5,
            KeyCode::Numpad6,
            KeyCode::NumpadAdd,
            KeyCode::Numpad1,
            KeyCode::Numpad2,
            KeyCode::Numpad3,
            KeyCode::Numpad0,
            KeyCode::NumpadDecimal,
            KeyCode::NumpadEnter,
            KeyCode::NumpadDivide,
            KeyCode::NumpadMultiply,
        ];
        let mut seen = std::collections::HashMap::new();
        for k in keys {
            let ev = keycode_to_evdev(k).expect("listed key is mapped");
            if let Some(prev) = seen.insert(ev, k) {
                panic!("evdev {ev} maps from both {prev:?} and {k:?}");
            }
        }
    }

    #[test]
    fn mouse_buttons_map_and_reject_other() {
        assert_eq!(mouse_button(MouseButton::Left), Some(ReimsVgpuButton::Left));
        assert_eq!(
            mouse_button(MouseButton::Right),
            Some(ReimsVgpuButton::Right)
        );
        assert_eq!(
            mouse_button(MouseButton::Middle),
            Some(ReimsVgpuButton::Middle)
        );
        assert_eq!(mouse_button(MouseButton::Back), Some(ReimsVgpuButton::Side));
        assert_eq!(
            mouse_button(MouseButton::Forward),
            Some(ReimsVgpuButton::Extra)
        );
        assert_eq!(mouse_button(MouseButton::Other(9)), None);
    }

    #[test]
    fn scroll_line_delta_maps_to_notches_by_sign() {
        // One line up → one WheelUp notch.
        assert_eq!(
            scroll_to_notches(MouseScrollDelta::LineDelta(0.0, 1.0)),
            vec![ReimsVgpuButton::WheelUp]
        );
        // Three lines down → three WheelDown notches.
        assert_eq!(
            scroll_to_notches(MouseScrollDelta::LineDelta(0.0, -3.0)),
            vec![ReimsVgpuButton::WheelDown; 3]
        );
        // Horizontal right.
        assert_eq!(
            scroll_to_notches(MouseScrollDelta::LineDelta(2.0, 0.0)),
            vec![ReimsVgpuButton::WheelRight; 2]
        );
        // No motion → nothing.
        assert!(scroll_to_notches(MouseScrollDelta::LineDelta(0.0, 0.0)).is_empty());
    }

    /// A 2-notch scroll produces exactly a down,up,down,up HostAction sequence
    /// of `InputPointerButton`s carrying the wheel code — the momentary-wheel
    /// contract the kb documents.
    #[test]
    fn scroll_actions_emit_down_up_pair_per_notch() {
        use crate::runtime::host::HostActionKind;
        let acts = scroll_actions(MouseScrollDelta::LineDelta(0.0, -2.0));
        assert_eq!(acts.len(), 4);
        for a in &acts {
            assert_eq!(a.kind, HostActionKind::InputPointerButton);
            assert_eq!(
                ReimsVgpuButton::from_wire(a.a0 as u32),
                Some(ReimsVgpuButton::WheelDown)
            );
        }
        // down, up, down, up.
        assert_eq!(
            acts.iter().map(|a| a.a1).collect::<Vec<_>>(),
            vec![1, 0, 1, 0]
        );
        // No scroll → no actions.
        assert!(scroll_actions(MouseScrollDelta::LineDelta(0.0, 0.0)).is_empty());
    }

    #[test]
    fn scroll_pixel_delta_quantises_and_never_swallows() {
        // Exactly two notches down.
        let two = scroll_to_notches(MouseScrollDelta::PixelDelta(PhysicalPosition::new(
            0.0,
            -2.0 * PIXELS_PER_NOTCH,
        )));
        assert_eq!(two, vec![ReimsVgpuButton::WheelDown; 2]);
        // A tiny sub-notch nudge still yields exactly one notch (not swallowed).
        let tiny = scroll_to_notches(MouseScrollDelta::PixelDelta(PhysicalPosition::new(
            0.0, 3.0,
        )));
        assert_eq!(tiny, vec![ReimsVgpuButton::WheelUp]);
    }
}
