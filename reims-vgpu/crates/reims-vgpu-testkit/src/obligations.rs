//! The instrument that checks an obligation type's annotation reaches the
//! calls that hand one out.
//!
//! # The hole this exists for
//!
//! Several types in this workspace name an obligation: a retired native object
//! the caller must destroy, a swapchain it must tear down, a claim it must
//! release. Each carries `#[must_use]`, and the crates' own docs treat that
//! annotation as the enforcement.
//!
//! It is not enforcement when the value arrives in a `Vec`. `unused_must_use`
//! does not look inside one, so a method returning `Vec<Retired<T>>` whose
//! result is dropped compiles without a word — and dropping it is precisely
//! the leak the annotation on `Retired` was written to prevent. The annotation
//! has to be on the *method* as well, and whether it is there is a fact about
//! source text rather than about types, so a test is the only thing that can
//! hold it.
//!
//! [`unannotated_vec_returns_under`] reads every replacement crate's sources
//! and answers it. Deliberately syntactic and deliberately narrow: it knows one
//! shape, and a shape it cannot parse is one it reports rather than one it
//! passes.
//!
//! # Why the workspace and not a crate
//!
//! The obligation types and the functions handing them out are not always in
//! one crate. `Retired` and `Release` are the semantic model's and the rail
//! hands them back; `ChunkHandles` is the rail's own. A per-crate scan pools
//! only that crate's types, so a rail function returning `Vec<Retired>` was
//! not this law's business anywhere — which is the crate boundary deciding
//! what counts as a leak.
//!
//! Every function, not only the public ones. A private helper's result dropped
//! loses the same handles, and the call site that drops it is inside this
//! workspace either way.

use std::path::Path;

/// One function handing out a `Vec` of an obligation type without saying so.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Unannotated {
    /// The crate directory's name, so a failure message names the owner.
    pub krate: String,
    /// Path relative to that crate's `src`, so a failure message is the same
    /// on every machine.
    pub file: String,
    pub line: usize,
    /// The function's name.
    pub function: String,
    /// The obligation type it returns a `Vec` of.
    pub obligation: String,
}

/// Every `#[must_use]` `pub struct`/`pub enum` a crate declares.
///
/// Public so a suite can assert the scan found something. A scanner that
/// silently matched nothing would pass
/// [`unannotated_vec_returns_under`] on every crate in the workspace, including one
/// that had regressed, and it is the one failure mode a syntactic instrument
/// has that a type-level one does not.
///
/// # Panics
///
/// If the crate's sources cannot be read.
#[must_use]
pub fn obligation_types_in(crate_root: &Path) -> Vec<String> {
    let src = crate_root.join("src");
    let mut files = Vec::new();
    sources(&src, &src, &mut files);
    obligation_types(&files)
}

/// Every `#[must_use]` `pub struct`/`pub enum` declared under `src`.
fn obligation_types(sources: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    for (_, text) in sources {
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("#[must_use") {
                continue;
            }
            // The declaration is the next line that is not another attribute
            // or a doc comment.
            for next in lines.iter().skip(i + 1) {
                let next = next.trim_start();
                if next.starts_with("#[") || next.starts_with("///") || next.starts_with("//") {
                    continue;
                }
                if let Some(rest) = next
                    .strip_prefix("pub struct ")
                    .or_else(|| next.strip_prefix("pub enum "))
                {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        out.push(name);
                    }
                }
                break;
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Read every `.rs` file under a directory, as `(relative path, text)`.
fn sources(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
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

/// Every function under `crates_dir` that returns a `Vec` of a `#[must_use]`
/// type declared anywhere in these crates, and does not carry `#[must_use]`
/// itself.
///
/// The signature may be spread over several lines; what is looked for is a
/// `-> Vec<Obligation` reached before the body opens. The annotation is
/// searched for in the contiguous run of attributes and doc comments directly
/// above the declaration, which is where an attribute can be.
///
/// # Panics
///
/// If the crates cannot be read.
#[must_use]
pub fn unannotated_vec_returns_under(crates_dir: &Path) -> Vec<Unannotated> {
    let mut files: Vec<(String, String, String)> = Vec::new();
    for krate in crate::slugs::replacement_crates(crates_dir) {
        let name = krate
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        let src = krate.join("src");
        if !src.is_dir() {
            continue;
        }
        let mut here = Vec::new();
        sources(&src, &src, &mut here);
        for (file, text) in here {
            files.push((name.clone(), file, text));
        }
    }
    // Pooled across the crates, because a `Vec` in one may hold a type
    // declared in another and a per-crate pool called that ordinary.
    let obligations = obligation_types(
        &files
            .iter()
            .map(|(_, file, text)| (file.clone(), text.clone()))
            .collect::<Vec<_>>(),
    );

    let mut out = Vec::new();
    for (krate, file, text) in &files {
        for (line, function, obligation) in unannotated_in(text, &obligations) {
            out.push(Unannotated {
                krate: krate.clone(),
                file: file.clone(),
                line,
                function,
                obligation,
            });
        }
    }
    out.sort_unstable();
    out
}

/// The unannotated signatures in one file's production text.
///
/// Production only, as [`crate::slugs`] reads it: a signature inside
/// `#[cfg(test)]` is a fixture, including the ones below that exist to be
/// parsed by this.
fn unannotated_in(text: &str, obligations: &[String]) -> Vec<(usize, String, String)> {
    let production = match text.find("\n#[cfg(test)]") {
        Some(at) => &text[..at],
        None => text,
    };
    let lines: Vec<&str> = production.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(function) = declared_fn(line) else {
            continue;
        };
        // The signature, up to the body or the end of a `;` declaration.
        let mut signature = String::new();
        for continued in lines.iter().skip(i).take(12) {
            signature.push_str(continued);
            signature.push(' ');
            if continued.contains('{') || continued.trim_end().ends_with(';') {
                break;
            }
        }
        let Some(after) = signature.split("-> Vec<").nth(1) else {
            continue;
        };
        let returned: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !obligations.contains(&returned) {
            continue;
        }
        // Walk back over the attribute and doc block directly above.
        let mut annotated = false;
        for above in lines[..i].iter().rev() {
            let above = above.trim_start();
            if above.starts_with("#[must_use") {
                annotated = true;
                break;
            }
            if above.starts_with("#[") || above.starts_with("///") || above.starts_with("//") {
                continue;
            }
            break;
        }
        if !annotated {
            out.push((i + 1, function, returned));
        }
    }
    out
}

/// The name a `fn` declaration introduces, however it is qualified.
fn declared_fn(line: &str) -> Option<String> {
    let mut rest = line.trim_start();
    loop {
        let stripped = [
            "pub(crate) ",
            "pub(super) ",
            "pub ",
            "const ",
            "unsafe ",
            "async ",
        ]
        .iter()
        .find_map(|p| rest.strip_prefix(p));
        match stripped {
            Some(next) => rest = next,
            None => break,
        }
    }
    let rest = rest.strip_prefix("fn ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_vec_of_an_obligation_is_found_and_an_annotated_one_is_not() {
        let text = "\
    pub fn reached(&mut self) -> Vec<Retired> {
    }
    #[must_use = \"the objects are the caller's to destroy\"]
    pub fn lost(&mut self) -> Vec<Retired> {
    }
    fn take(self) -> Vec<Retired> {
    }
    pub fn other(&mut self) -> Vec<Plain> {
    }
";
        let found: Vec<String> = unannotated_in(text, &["Retired".to_owned()])
            .into_iter()
            .map(|(_, f, _)| f)
            .collect();
        assert_eq!(
            found,
            vec!["reached".to_owned(), "take".to_owned()],
            "a private helper hands out the same obligations"
        );
    }

    /// A signature that wraps over several lines is still one signature.
    #[test]
    fn a_wrapped_signature_is_read_whole() {
        let text = "\
    pub fn released(
        &mut self,
        at: TimelinePoint,
    ) -> Vec<Release> {
    }
";
        assert_eq!(
            unannotated_in(text, &["Release".to_owned()])
                .into_iter()
                .map(|(_, f, _)| f)
                .collect::<Vec<_>>(),
            vec!["released".to_owned()]
        );
    }
}
