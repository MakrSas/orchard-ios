//! Direct host-Metal backend: pure-Rust Metal encode driven from `runtime/`.
//!
//! macOS only. `backend-metal` on any other target is rejected by the
//! `compile_error!` in `lib.rs`, so there is no non-Apple arm of this module
//! and every `target_os = "macos"` gate below is a statement of that fact
//! rather than a branch.

pub mod abi;
mod constants;
pub mod error;

// ---------------------------------------------------------------------------
// Apple: real Metal encode
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod cache;
/// The census lines only this rail can answer. Reached through
/// [`crate::backend::Backend::emit_census`], never through a `cfg`.
#[cfg(target_os = "macos")]
mod census;
#[cfg(target_os = "macos")]
pub(crate) mod compute;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
pub(crate) mod format;
#[cfg(target_os = "macos")]
mod function;
#[cfg(target_os = "macos")]
pub(crate) mod mipmap;
#[cfg(target_os = "macos")]
pub(crate) mod mtl_enum;
#[cfg(target_os = "macos")]
pub(crate) mod raw_metal;
#[cfg(target_os = "macos")]
pub(crate) mod render;
/// Colour render targets this rail keeps alive across draws, and the one claim
/// that makes loading from one safe. See the module doc.
#[cfg(target_os = "macos")]
pub(crate) mod resident;
#[cfg(target_os = "macos")]
pub(crate) mod runtime;
#[cfg(target_os = "macos")]
pub(crate) mod samplers;
#[cfg(target_os = "macos")]
mod stage_input;
#[cfg(target_os = "macos")]
pub(crate) mod util;

/// This rail's half of the host-owned presentation window: a `CAMetalLayer` on
/// the window's own view, and the blit that fills its drawables.
///
/// Gated on the window's feature, which is the lawful question — whether this
/// build compiled a window at all is a fact about the build.
#[cfg(feature = "host-window")]
pub mod window;

#[cfg(target_os = "macos")]
pub(crate) use device::MetalBackend;
