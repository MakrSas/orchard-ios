//! Reims vGPU host path — single crate.
//!
//! | Module | Role |
//! | --- | --- |
//! | [`protocol`] | Stable facts: formats, layouts, pure arithmetic |
//! | [`model`] | Live guest-visible state (regs, rings, objects, present) |
//! | [`runtime`] | Drain / parse / resolve / plan / HostActions |
//! | [`backend`] | Trait + self-contained [`backend::metal`] / [`backend::vulkan`] |
//! | [`qemu`] | QEMU C ABI surface only |
//!
//! Features: at least one of `backend-metal` (default) or `backend-vulkan`, and
//! a build may carry **both**. Vulkan's product path is self-contained `ash`
//! ([`backend::vulkan::engine`]).
//!
//! # The supported arms
//!
//! | Arm | Features | Host GPU API |
//! | --- | --- | --- |
//! | Metal | `backend-metal` | native Metal |
//! | Vulkan / MoltenVK | `backend-vulkan` on macOS | MoltenVK |
//! | Vulkan / native | `backend-vulkan` on linux | native ICD |
//! | Both | `backend-metal,backend-vulkan` on macOS | either, chosen at run time |
//!
//! **Gate the host on `target_os` and nothing else.** `macos` and `linux` are
//! the only two values this crate names, so a reader greps one key to find
//! every host gate.
//!
//! There is **no** host-stub Metal arm. `backend-metal` off macOS has no Metal
//! to call, so it is a compile error rather than a binary that links and cannot
//! draw.
//!
//! # Why the fourth cell exists
//!
//! An Apple host can reach its GPU natively through Metal or through MoltenVK,
//! and those two paths translate the guest's command stream by completely
//! different means. When a frame is wrong, "is this a metal2vulkan defect or a
//! defect in this device" is the first question and it was previously
//! unanswerable without rebuilding — which changes the binary, the caches and
//! the boot. A binary carrying both rails answers it by running the *same guest
//! stream* twice with one variable changed.
//!
//! # What that costs the rest of the crate
//!
//! `cfg` may only ever answer **"what did this build compile"**. It may not
//! answer "which rail is running", because on this fourth cell those are
//! different questions and the compiler cannot tell them apart — a
//! `not(feature = "backend-vulkan")` block meaning "the Metal arm" simply
//! disappears, silently, the moment both features are on. Everything the
//! running rail decides goes through [`backend::Backend`], whose implementation
//! the process picks once in [`backend::select`]; the `cfg`s that remain are
//! module declarations and the arms of that one selection.
//!
//! Do not reintroduce `not(feature = "backend-vulkan")` as a spelling of "the
//! Metal arm". It says what the build is *not*, which stopped being equivalent
//! the moment this cell existed.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(rust_2018_idioms)]

#[cfg(not(any(feature = "backend-metal", feature = "backend-vulkan")))]
compile_error!("select at least one of backend-metal or backend-vulkan");

#[cfg(all(feature = "backend-metal", not(target_os = "macos")))]
compile_error!(
    "backend-metal requires target_os = \"macos\": there is no host-stub Metal \
     arm. Use --no-default-features --features backend-vulkan,host-window on \
     any other host."
);

// Vulkan reaches the GPU through MoltenVK on macOS and a native ICD on
// Linux; Windows hosts use their native ICDs (NVIDIA/AMD/Intel ship
// VK_KHR_win32_surface and Vulkan 1.2+). Any other host is untested rather
// than known-broken — name it here so a new port is a deliberate edit to this
// list, not an accident.
#[cfg(all(
    feature = "backend-vulkan",
    not(any(target_os = "macos", target_os = "linux", target_os = "windows"))
))]
compile_error!(
    "backend-vulkan is supported on target_os = \"macos\" (MoltenVK), \
     target_os = \"linux\", and target_os = \"windows\" (native ICDs)"
);

/// Operator configuration: the switches this device reads, their names, and the
/// one place they parse.
///
/// Named `config` rather than `env` because the environment is how a switch
/// arrives today and not what it *is*. The crate owns the declaration, the
/// parse and the rule — see `reims_vgpu_config` for why a switch may only
/// narrow what this device does and may never widen it.
pub use reims_vgpu_config as config;
/// The backend-neutral protocol vocabulary, in the crate that owns it.
///
/// The first layer allowed to say what a wire tag *means*: formats, layouts,
/// geometry, page arithmetic, the closed ordinal rules, and the refusals they
/// name. It absorbed `reims-vgpu-contract`, which had the same job under a
/// second name — one vocabulary in two crates is two places to look and two
/// places for a rule to drift.
///
/// See `reims_vgpu_protocol` for what the crate boundary makes true that a
/// module boundary only asserted.
pub use reims_vgpu_protocol as protocol;
pub mod model;
/// Crate-wide observability: the always-on fail sink and the decline
/// vocabulary. Above `runtime/` because every subsystem owes the reader a
/// reason, and `translate/` + `caps/` must be able to name one without
/// depending on `runtime/`.
pub mod observe;
pub mod runtime;

pub mod backend;
pub mod qemu;

/// Host-owned presentation window — a Rust-owned `winit` window that replaces
/// QEMU's UI. See [[host-window]].
///
/// The feature names no backend. Which rail fills the window is a run-time
/// answer (`backend::Backend::presents_host_window`) and every rail can:
/// Vulkan drives a swapchain on a `VkSurfaceKHR`, Metal a `CAMetalLayer` on the
/// same native view. It is enabled on every product arm.
#[cfg(feature = "host-window")]
pub mod host_window;

/// The device registry and the entry surface `qemu::abi` wraps. Private, with
/// the names that surface reaches re-exported below — the shape
/// `display_surface` and `window_publish` already use.
mod device;
pub(crate) use device::{
    backend_name, device_console_feed, device_create, device_cursor_glyph_copy,
    device_cursor_glyph_info, device_destroy, device_drain, device_efi_console_copy,
    device_gfx_read, device_gfx_write, device_iosfc_read, device_iosfc_write, device_poll,
    device_pop_action, device_reset, device_scanout_copy, device_scanout_may_paint,
    device_window_run_main, device_window_set_early_fb, device_window_start, device_window_stop,
    unwind_safe, ConsoleFeed, CursorGlyphInfo,
};
