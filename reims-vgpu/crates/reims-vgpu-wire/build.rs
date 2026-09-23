//! Tells the fixture tests whether Apple's captured records are on disk.
//!
//! See `reims_vgpu_testkit::probe_wire_fixtures` for why the answer is a `cfg`
//! rather than a runtime check, and why it is one function rather than three.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    reims_vgpu_testkit::probe_wire_fixtures(&format!("{manifest}/fixtures"));
}
