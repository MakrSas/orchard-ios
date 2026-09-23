//! **A slug is the only thing a failure-channel reader has.**
//!
//! An observation carries the slug and its payload and nothing that says which
//! module minted it, so two refusals in different modules sharing one slug
//! make every report of either ambiguous — and the reader who investigates
//! goes to the wrong owner. Every module already checks that its own slugs are
//! distinct from each other, which is the check a module can make alone and is
//! not the check that matters: a collision is a fact about all the sources at
//! once, produced by two modules written months apart drawing on the same
//! small vocabulary.
//!
//! The second law is the shape a collision arrives in. A reason is namespaced
//! by its owner and a classification is a bare word, so a reason that lost its
//! prefix is one word in a space of three hundred and the next module to want
//! that word takes it.

use std::collections::BTreeMap;
use std::path::PathBuf;

fn crates_dir() -> PathBuf {
    // This crate sits beside the ones it reads.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("a crate lives under a crates directory")
        .to_path_buf()
}

/// The bare words a slug is allowed to be, and the enums they classify.
///
/// A classification says *which of several kinds* something is, so it is read
/// beside the key that names its subject: a memory topology of `discrete` and
/// a device class of `discrete` are two answers to two questions and neither
/// is a failure at all. That is why they may repeat and reasons may not.
///
/// The list is short on purpose. Growing it is the moment to ask whether the
/// new word is really a classification, because everything else in the
/// workspace answers this by carrying its owner's prefix.
const CLASSIFICATIONS: &[&str] = &[
    // `vulkan::memory::MemoryTopology`
    "unified",
    "discrete",
    // `vulkan::host::DeviceClass`
    "integrated",
    "virtual",
    "cpu",
    "other",
];

#[test]
fn every_reason_is_namespaced_and_names_exactly_one_owner() {
    let slugs = reims_vgpu_testkit::slugs::slugs_under(&crates_dir());

    // A syntactic instrument's one failure mode is matching nothing and
    // passing everywhere. The workspace declares hundreds of these, so a scan
    // that found a handful found none of the collisions either.
    assert!(
        slugs.len() > 250,
        "the scan found {} slugs in the workspace, which is not what it \
         declares; the reader stopped matching",
        slugs.len()
    );
    // And from more than one crate, so a scan that walked one directory and
    // stopped is caught too.
    let crates: std::collections::BTreeSet<&str> = slugs.iter().map(|s| s.krate.as_str()).collect();
    assert!(crates.len() > 3, "slugs found in only {crates:?}");

    let mut owners: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for slug in &slugs {
        owners
            .entry(slug.text.as_str())
            .or_default()
            .push(format!("{}/src/{}", slug.krate, slug.file));
    }

    let collisions: Vec<_> = owners
        .iter()
        .filter(|(text, at)| at.len() > 1 && !CLASSIFICATIONS.contains(text))
        .collect();
    assert!(
        collisions.is_empty(),
        "these reasons are minted in more than one place, so a report of one \
         cannot be told from a report of the other:\n{collisions:#?}"
    );

    let unnamespaced: Vec<_> = slugs
        .iter()
        .filter(|s| !s.namespaced() && !CLASSIFICATIONS.contains(&s.text.as_str()))
        .collect();
    assert!(
        unnamespaced.is_empty(),
        "these slugs carry no owner prefix. A reason must name its owner, and \
         a classification must be listed in CLASSIFICATIONS with the enum it \
         classifies:\n{unnamespaced:#?}"
    );
}
