//! Direct host-Metal backend: pure-Rust Metal encode driven from `runtime/`.
//!
//! Apple only. `backend-metal` on any other target is rejected by the
//! `compile_error!` in `lib.rs`, so there is no non-Apple arm of this module
//! and the modules below carry no platform gate of their own: the feature is
//! already the whole answer to "did this build compile Metal encode".

pub mod abi;
mod constants;
pub mod error;

// ---------------------------------------------------------------------------
// Apple: real Metal encode
// ---------------------------------------------------------------------------

mod cache;
/// The census lines only this rail can answer. Reached through
/// [`crate::backend::Backend::emit_census`], never through a `cfg`.
mod census;
pub(crate) mod compute;
mod device;
pub(crate) mod format;
mod function;
pub(crate) mod mipmap;
pub(crate) mod mtl_enum;
pub(crate) mod raw_metal;
pub(crate) mod render;
/// Colour render targets this rail keeps alive across draws, and the one claim
/// that makes loading from one safe. See the module doc.
pub(crate) mod resident;
pub(crate) mod runtime;
/// Sampled textures kept across draws, keyed by what their decode read.
pub(crate) mod sampled_cache;
pub(crate) mod samplers;
mod stage_input;
pub(crate) mod util;

/// This rail's half of the host-owned presentation window: a `CAMetalLayer` on
/// the window's own view, and the blit that fills its drawables.
///
/// Gated on the window's feature, which is the lawful question — whether this
/// build compiled a window at all is a fact about the build.
#[cfg(feature = "host-window")]
pub mod window;

pub(crate) use device::MetalBackend;
