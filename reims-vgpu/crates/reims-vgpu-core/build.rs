//! Tells the fixture test whether Apple's captured records are on disk.
//!
//! The probe lives in `reims-vgpu-testkit` so the directory override, the file
//! names and what `REIMS_WIRE_FIXTURES_REQUIRED` means cannot fork between the
//! three suites that stand down without them.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    reims_vgpu_testkit::probe_wire_fixtures(&format!("{manifest}/../reims-vgpu-wire/fixtures"));
}
