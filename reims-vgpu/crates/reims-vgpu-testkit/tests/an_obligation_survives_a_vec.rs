//! **`#[must_use]` on an obligation type says nothing about a `Vec` of them.**
//!
//! Several types in this workspace name something the caller must then do:
//! destroy a retired native object, tear down a replaced swapchain, publish a
//! released stamp, withdraw a stranded ordinal. Each carries `#[must_use]`,
//! and the modules' own docs treat that annotation as the enforcement.
//!
//! `unused_must_use` does not look inside a `Vec`. A function returning
//! `Vec<Retired<T>>` whose result is dropped therefore compiles in silence,
//! and dropping it is exactly the leak the annotation on the element type was
//! written to prevent — the objects go without the device ever destroying
//! them, and nothing anywhere says so.
//!
//! So the annotation has to be on the function too, and whether it is there is
//! a fact about source text rather than about types.
//!
//! # Why this is asked of the workspace and not of one crate
//!
//! It was held at crate scope, which made the crate boundary decide what
//! counts as a leak: `Retired` and `Release` are the semantic model's and the
//! rail is what hands them back, so a rail function returning `Vec<Retired>`
//! was nobody's business. The types are pooled across the replacement crates
//! now, and the scan reads every crate — the third law kept here, beside
//! `slugs` and the fixture loaders, rather than a fourth copy somewhere else.

use std::path::PathBuf;

fn crates_dir() -> PathBuf {
    // This crate sits beside the ones it reads.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("a crate lives under a crates directory")
        .to_path_buf()
}

/// A syntactic instrument's one failure mode is matching nothing and passing
/// everywhere, so the scan is asked what it found before it is believed.
#[test]
fn the_scan_finds_the_obligations_the_workspace_declares() {
    let dir = crates_dir();
    let core = reims_vgpu_testkit::obligations::obligation_types_in(&dir.join("reims-vgpu-core"));
    assert!(
        core.len() > 4,
        "the semantic model declares more obligation types than this: {core:?}"
    );
    let rail = reims_vgpu_testkit::obligations::obligation_types_in(&dir.join("reims-vgpu-vulkan"));
    assert!(
        !rail.is_empty(),
        "and so does the rail, which is the half a per-crate scan could not join"
    );
}

/// The law.
#[test]
fn every_function_handing_out_a_vec_of_obligations_says_so() {
    let found = reims_vgpu_testkit::obligations::unannotated_vec_returns_under(&crates_dir());
    assert!(
        found.is_empty(),
        "these hand back a `Vec` of a `#[must_use]` type without carrying the \
         attribute, so a caller that drops the result is told nothing:\n{}",
        found
            .iter()
            .map(|u| format!(
                "  {}/{} line {}: fn {} -> Vec<{}>",
                u.krate, u.file, u.line, u.function, u.obligation
            ))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
