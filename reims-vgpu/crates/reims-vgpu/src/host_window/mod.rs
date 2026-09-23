//! Host-owned presentation window ([[host-window]]) — a Rust-owned `winit`
//! window that replaces QEMU's built-in UI and presents the finished guest
//! frame directly, keeping the C/QEMU side thin.
//!
//! Gated behind the `host-window` cargo feature, which names no backend. This
//! module knows nothing about which rail draws into the window: it hands the
//! native surface to `backend::Backend::window_attach` and each published frame
//! to `window_present`, and the running rail supplies the drawables — a
//! `VkSurfaceKHR` swapchain on the Vulkan rail, a `CAMetalLayer` on the Metal
//! one. That is what lets one boot be compared against the other with only the
//! executor changed.
//!
//! Three pieces:
//! - [`input_map`] — winit event → neutral [`crate::runtime::HostAction`]. Pure
//!   mapping, no window state, unit-tested off-VM.
//! - [`keyboard`] — which keys the guest believes are held, and whether the
//!   compositor's own shortcuts are being captured. Pure state machine; it owns
//!   the rule that every key-down is eventually closed by a key-up.
//! - [`capture`] — the per-platform request that stops the desktop from
//!   consuming shortcuts before the window sees them. A typed refusal where the
//!   platform cannot honour it.
//! - [`present`] — the window itself: event loop, native surface, and the
//!   publish → fit → present loop. It also drives [`input_map`] and hands each
//!   action to an `InputSink`; `lib.rs` wires that to the device's prompt action
//!   queue through QEMU's thread-safe `notify_actions` callback.
//!
//! The letterbox/scale arithmetic both the blit and the pointer path move
//! through is `backend::window::viewport`, one layer down, because both rails'
//! presenters need it too and `backend` may not reach up into this module.

pub mod capture;
pub mod input_map;
pub mod keyboard;
pub mod present;
