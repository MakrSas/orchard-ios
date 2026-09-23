//! Normalization and planners.
//!
//! `event_sync` is the live planner: `runtime/exec`, `runtime/fence_exec` and
//! `runtime/resolve` drive fence/event semantics through it. The blit, compute
//! and render planners that used to sit beside it produced values only the
//! `Backend` trait consumed, and nothing called that.

pub mod event_sync;
