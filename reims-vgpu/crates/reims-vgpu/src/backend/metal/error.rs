//! The C error-buffer helper for the Metal encode path, and this rail's name
//! for the device's structured refusal.
//!
//! `Status` is [`crate::backend::refusal::RailRefusal`]. It is aliased rather
//! than re-spelled because 200-odd Metal checks construct one by that name, and
//! because the type is not this rail's: the neutral `EncodeStatus` /
//! `ComputeStatus` vocabularies carry it, and the other rail may build one the
//! day it has structured refusals of its own.

pub use crate::backend::refusal::{FieldValue, RailRefusal as Status};

use std::os::raw::c_char;

/// Copy `msg` into the shim's error buffer, NUL-terminated and truncated to fit.
///
/// # Safety
///
/// `err` must be null, or valid for writes of `err_cap` bytes. Null and a zero
/// capacity are both checked here, so the caller's obligation is only that a
/// non-null pointer really has the capacity it claims — which is the contract
/// `reims_vgpu_qemu_abi.h` states for every `(char *err, size_t err_cap)` pair
/// crossing the boundary.
pub unsafe fn write_err(err: *mut c_char, err_cap: usize, msg: &str) {
    // SAFETY: forwarded unchanged — the caller's promise about `err` and
    // `err_cap` is exactly what `write_c_str` asks for.
    unsafe { crate::qemu::cstr::write_c_str(err, err_cap, msg) };
}
