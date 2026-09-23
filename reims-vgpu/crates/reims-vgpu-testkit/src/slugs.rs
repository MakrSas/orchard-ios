//! The instrument that reads every failure-channel slug in the workspace at
//! once.
//!
//! # The hole this exists for
//!
//! A slug is the stable name a refusal reaches the always-on failure channel
//! under. It is the only thing a reader has: an observation carries the slug
//! and its payload, and nothing beside it says which module minted it. So two
//! refusals in different modules sharing one slug make every report of either
//! ambiguous, and the reader who investigates goes to the wrong owner.
//!
//! Each module tests that *its own* slugs are distinct from each other, which
//! is the check a module can make alone and is not the one that matters.
//! Uniqueness across the workspace is a fact about all the sources at once, so
//! nothing inside a module can hold it — and a collision is exactly the kind
//! of thing two modules written months apart produce, because the words are
//! drawn from the same small vocabulary.
//!
//! # Two shapes, and only one of them has to be unique
//!
//! A slug is either a *reason* --- why something was refused, dropped or
//! declined --- or a *classification*: which memory topology this host has,
//! which class of device it is. Reasons are namespaced by their owner
//! (`vk_transfer_overlapping_self_copy`, `lifecycle_no_such_task`,
//! `stream_encoder_never_ended`) and must be unique. Classifications are bare
//! words (`discrete`, `unified`) and may repeat, because each is read beside
//! the key that names its subject: a memory topology of `discrete` and a
//! device class of `discrete` are two answers to two questions and neither is
//! a failure at all.
//!
//! That gives the second law: **a slug with no namespace is a
//! classification.** A reason that lost its prefix would be one word in a
//! space of three hundred, and it is the shape a collision arrives in.
//!
//! Deliberately syntactic and deliberately narrow, like
//! [`crate::obligations`]: it reads the source text for the two forms a slug is
//! written in, and a form it cannot parse is one it does not find rather than
//! one it passes.

use std::path::Path;

/// One slug, and where it was written.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Slug {
    pub text: String,
    /// The crate directory's name, so a failure message names the owner.
    pub krate: String,
    /// Path relative to that crate's `src`, so the message is the same on
    /// every machine.
    pub file: String,
}

impl Slug {
    /// Whether this slug is namespaced, which is what a reason must be.
    #[must_use]
    pub fn namespaced(&self) -> bool {
        self.text.contains('_')
    }
}

/// Every slug declared by the replacement crates under `crates_dir`.
///
/// Walks the directory rather than taking a list, because a list is a thing to
/// forget a crate from and a forgotten crate is a whole namespace this law
/// stops covering.
///
/// The legacy `reims-vgpu` crate itself is not walked, and its exclusion is a
/// claim rather than a convenience. A replacement module that takes over a
/// fact from a legacy one **reuses its slug on purpose**: the same refusal
/// should reach the channel under the same name whichever rail produced it,
/// so that a report from before the cutover and one from after are the same
/// report. Requiring uniqueness across that seam would forbid exactly the
/// continuity the port wants. Within the replacement crates there is no such
/// pairing, so uniqueness there means what it says.
///
/// Test modules are excluded: a slug written inside `#[cfg(test)]` is a
/// fixture, and holding fixtures to the production namespace would make the
/// law fail for a test that deliberately forges one.
///
/// # Panics
///
/// If `crates_dir` cannot be read.
#[must_use]
pub fn slugs_under(crates_dir: &Path) -> Vec<Slug> {
    let mut out = Vec::new();
    for krate in replacement_crates(crates_dir) {
        let name = krate
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        let src = krate.join("src");
        if !src.is_dir() {
            continue;
        }
        let mut files = Vec::new();
        sources(&src, &src, &mut files);
        for (file, text) in &files {
            for slug in in_text(text) {
                out.push(Slug {
                    text: slug,
                    krate: name.clone(),
                    file: file.clone(),
                });
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The slugs one source file declares, outside its test module.
///
/// Two forms, which are the two the workspace writes:
///
/// - a `fn slug` or `fn reason` returning `&'static str`, whose body's string
///   literals are the slugs;
/// - a `const SLUG`/`const REASON` of `&str`.
fn in_text(text: &str) -> Vec<String> {
    let production = match text.find("\n#[cfg(test)]") {
        Some(at) => &text[..at],
        None => text,
    };
    let mut out = Vec::new();
    let lines: Vec<&str> = production.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("pub const SLUG: &str = ") {
            out.extend(literal(rest));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("const SLUG: &str = ") {
            out.extend(literal(rest));
            continue;
        }
        let is_body = trimmed.contains("fn slug") || trimmed.contains("fn reason");
        if !is_body {
            continue;
        }
        // The signature may wrap, so the return type is looked for over the
        // next few lines and the body ends at the first line whose indentation
        // closes it.
        let mut signature = String::new();
        let mut opened = None;
        for (offset, continued) in lines.iter().enumerate().skip(i).take(6) {
            signature.push_str(continued);
            signature.push(' ');
            if continued.contains('{') {
                opened = Some(offset);
                break;
            }
        }
        let (Some(open), true) = (opened, signature.contains("-> &'static str")) else {
            continue;
        };
        let indent = line.len() - trimmed.len();
        for body in lines.iter().skip(open + 1) {
            let closed = body.len() > indent
                && body.as_bytes()[indent] == b'}'
                && body.trim() == "}"
                && body.len() - body.trim_start().len() == indent;
            if closed {
                break;
            }
            out.extend(literal(body));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The slug-shaped string literals in one line: lower-case, digits and
/// underscores. A literal with anything else in it is prose, not a slug.
fn literal(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut end = start;
        while end < bytes.len() && bytes[end] != b'"' {
            end += 1;
        }
        if end < bytes.len() {
            let text = &line[start..end];
            if !text.is_empty()
                && text
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            {
                out.push(text.to_owned());
            }
        }
        i = end + 1;
    }
    out
}

/// Read every `.rs` file under a directory, as `(relative path, text)`.
/// The replacement crates under `crates_dir`, in name order.
///
/// Walked rather than listed, for [`slugs_under`]'s reason: a list is a thing
/// to forget a crate from. The legacy `reims-vgpu` crate is not one of these —
/// its directory name has no `reims-vgpu-` prefix — which is the exclusion
/// [`slugs_under`] states its claim about.
///
/// # Panics
///
/// If `crates_dir` cannot be read.
pub(crate) fn replacement_crates(crates_dir: &Path) -> Vec<std::path::PathBuf> {
    let entries = std::fs::read_dir(crates_dir).expect("the workspace can read its own crates");
    let mut crates: Vec<_> = entries
        .map(|e| e.expect("a readable entry").path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("reims-vgpu-"))
        })
        .collect();
    crates.sort();
    crates
}

/// Every `.rs` file under `dir`, as (path relative to `root`, text).
pub(crate) fn sources(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
    let entries = std::fs::read_dir(dir).expect("a crate can read its own sources");
    let mut paths: Vec<_> = entries
        .map(|e| e.expect("a readable entry").path())
        .collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            sources(root, &path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).expect("a readable source file");
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            out.push((rel, text));
        }
    }
}
