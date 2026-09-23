//! Host-testable library surface for the UEFI GOP option ROM.
//!
//! [`paint`] is the real Blt/fill/copy implementation shared with the UEFI
//! binary handlers. Host `cargo test` exercises these paths without boot
//! services. The PE (`main.rs`) is a thin PCI + protocol install shell.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod paint;
