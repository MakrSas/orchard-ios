//! Reims vGPU core: the semantic model, with no host in it.
//!
//! # What this crate is for
//!
//! The device's execution architecture is being replaced by one where every
//! accepted guest packet is an ordered transaction with an exact dependency
//! envelope, and where the only things that advance semantic state are
//! immutable facts returned by an executor. That model has to exist somewhere
//! that cannot reach a Vulkan handle, a Metal object, a QEMU structure or a
//! guest-RAM pointer — otherwise "backend-neutral" is a habit, and habits do
//! not survive the first convenient import.
//!
//! So it is a crate, and its dependency list is the claim: it can see what a
//! wire tag *means* and nothing about how a host produces it.
//!
//! # The parts
//!
//! - [`identity`] — the names. Guest ordinals are parsed once into total types
//!   and carried; slot reuse produces a new generation; a wrapping completion
//!   timeline is a type that knows it wraps rather than an integer every reader
//!   has to remember about.

#![forbid(unsafe_code)]

pub mod access;
pub mod bind;
pub mod blit;
pub mod compute;
pub mod content;
pub mod control;
pub mod coverage;
pub mod depend;
pub mod encoder;
pub mod exec;
pub mod executor;
pub mod heap;
pub mod icb;
pub mod identity;
pub mod interpret;
pub mod lifecycle;
pub mod namespace;
pub mod operation;
pub mod pass;
pub mod pipeline;
pub mod prereq;
pub mod present;
pub mod publish;
pub mod query;
pub mod range_set;
pub mod ready;
pub mod render;
pub mod resolve;
pub mod resource_state;
pub mod retire;
pub mod schedule;
pub mod session;
/// The storage mode a resource declares, re-exported from the protocol layer.
///
/// A semantic input to placement and coherence, so the model carries it — and
/// the executor that consumes it sees this crate rather than the protocol one.
pub use reims_vgpu_protocol::storage_mode;

/// The three-dimensional extent and mip arithmetic of the guest API,
/// re-exported from the protocol layer.
pub use reims_vgpu_protocol::extent;

/// The `MTLBlitOption` word's closed set, and the plane it selects.
///
/// Re-exported rather than restated: which plane a copy addresses is a term of
/// the wire, and an executor deciding it for itself would be a second reading
/// of the same word. The executor sees this crate rather than the protocol
/// one, and the semantic model names the option in [`crate::blit`] without
/// interpreting it.
pub use reims_vgpu_protocol::blit as blit_option;

/// The guest's pixel-format ordinals and what each one is made of,
/// re-exported from the protocol layer.
///
/// Semantic, not native: which aspects a format has and how wide its texels
/// are decide what a transfer moves and which attachment slot an image can
/// take, on either rail. The executor that turns one of these into a native
/// format sees this crate rather than the protocol one.
pub use reims_vgpu_protocol::pixel_format;

/// What a texture declaration is, re-exported from the protocol layer.
///
/// A checked shape is a semantic input to allocation, view expansion and
/// transfer arithmetic alike, so the model carries it — and the executor that
/// turns one into a native image sees this crate rather than the protocol one.
pub use reims_vgpu_protocol::texture_shape;

/// The sampler state a guest declares, re-exported from the protocol layer.
///
/// Semantic: which filter, which addressing, which comparison — the same
/// questions on either rail. The executor that turns one into a native sampler
/// sees this crate rather than the protocol one.
pub use reims_vgpu_protocol::sampler;

/// The depth-stencil state a guest declares, re-exported from the protocol
/// layer.
///
/// Which comparison, which face is bound, what a face does at each outcome —
/// the same questions on either rail, and neither answer names a native
/// object.
pub use reims_vgpu_protocol::depth_stencil;

/// A colour attachment's blend state, re-exported from the protocol layer.
///
/// Which equation, which factors, which channels are writable — semantic on
/// either rail. Whether a host can run the dual-source factors among them is
/// not, and that question belongs to the executor that queried a device.
pub use reims_vgpu_protocol::blend;

/// The primitive a draw assembles, re-exported from the protocol layer.
///
/// Semantic on either rail, and the class grouping with it — where a rail's
/// pipeline may move without being rebuilt is an executor question, but which
/// types share a class is not.
pub use reims_vgpu_protocol::topology;

/// A vertex attribute's format and the geometry it implies, re-exported from
/// the protocol layer.
///
/// Component count, channel width and footprint are arithmetic that means the
/// same thing on either rail. Which of these formats a host can fetch is not,
/// and that belongs to the executor.
pub use reims_vgpu_protocol::vertex_format;

/// How a vertex buffer layout advances, re-exported from the protocol layer.
pub use reims_vgpu_protocol::vertex_step;

pub mod stream;
pub mod submit;
pub mod sync;
pub mod transaction;
pub mod walk;

/// Command-stream bytes for the suites that want the model driven from them.
#[cfg(test)]
mod testing;

/// The dependency list, read back and checked against the claim above it.
///
/// # Why a test and not a comment
///
/// This crate's doc says its dependency list *is* the claim that it can see
/// what a wire tag means and nothing about how a host produces it. A claim
/// carried only by a comment is one a convenient import silently retires: a
/// `reims-vgpu-vulkan` line added to `Cargo.toml` to reach one type compiles,
/// passes every test in this crate, and leaves the doc above saying the
/// opposite of what the crate is.
///
/// So the manifest is parsed and compared. The parser is deliberately a dozen
/// lines here rather than a shared helper in another crate — a boundary test
/// that had to depend on something to run would be adding an edge to the graph
/// it exists to bound.
///
/// Dev- and build-dependencies are checked separately and not folded in. They
/// are a different claim: a test may read the wire bytes and the shared fixture
/// loader without the library being able to, and the split is the only thing
/// that keeps "what the model can see" from quietly meaning "what the test
/// binary links".
#[cfg(test)]
mod boundary {
    /// The in-workspace crates named in one section of a manifest, in the order
    /// the section lists them.
    fn workspace_deps(manifest: &str, section: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut inside = false;
        for line in manifest.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                inside = trimmed == section;
                continue;
            }
            if !inside {
                continue;
            }
            if let Some(name) = trimmed.split('=').next().map(str::trim) {
                if name.starts_with("reims-vgpu-") {
                    out.push(name.to_string());
                }
            }
        }
        out
    }

    fn manifest() -> String {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
            .expect("a crate can read its own manifest")
    }

    /// The library sees the protocol layer and nothing else in this workspace.
    ///
    /// Named exhaustively rather than as a set of prohibitions, because the
    /// failure this guards against is an edge nobody thought to prohibit.
    #[test]
    fn the_library_depends_on_the_protocol_layer_and_nothing_else() {
        assert_eq!(
            workspace_deps(&manifest(), "[dependencies]"),
            vec!["reims-vgpu-protocol"],
            "this crate's dependency list is its claim to be backend-neutral; \
             a new edge here retires that claim silently"
        );
    }

    /// And the test binary's extra edges are the ones the fixture suites need,
    /// which the library still cannot reach.
    #[test]
    fn the_test_binary_reaches_the_capture_and_the_library_still_does_not() {
        let m = manifest();
        for section in ["[dev-dependencies]", "[build-dependencies]"] {
            let deps = workspace_deps(&m, section);
            for d in &deps {
                assert!(
                    matches!(d.as_str(), "reims-vgpu-testkit" | "reims-vgpu-wire"),
                    "{section} names {d}: a suite may read the capture and the \
                     bytes under it, and nothing else this crate is not allowed \
                     to see"
                );
            }
            assert!(!deps.is_empty(), "{section} names no in-workspace crate");
        }
    }
}
