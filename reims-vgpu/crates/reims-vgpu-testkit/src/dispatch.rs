//! Which opcodes a hand-written dispatch function actually handles.
//!
//! # The third table
//!
//! The packet ledger in the protocol crate says which opcodes exist on each
//! channel and what this device owes for each. The semantic model's
//! `classify` turns a row into a payload class, and the model's own suite pins
//! `classify`'s arm lists against each kind's `Kind::of` — two tables of the
//! same numbers, compared.
//!
//! The device's `process_root_packet` and `process_child_packet` are a third
//! table of those numbers, and nothing compared it against either of the other
//! two. It is the one that decides what a guest packet actually does today, and
//! it is the one the cutover has to replace: a row production handles and the
//! ledger does not know about is a command the replacement will refuse, and a
//! row the ledger admits and production never reaches is a command the guest is
//! silently owed nothing for.
//!
//! Whether a `match` arm names an opcode is a fact about source text — the arms
//! are `const` names, not values a type could carry — so a scan is the only
//! thing that can hold it, for the reason [`crate::obligations`] gives about
//! `#[must_use]` inside a `Vec`.
//!
//! # It reports what it cannot read
//!
//! Every refusal here is a shape the scanner did not understand, and each one
//! is returned rather than skipped. A syntactic instrument that silently
//! matched nothing passes on a codebase that has regressed and on one that has
//! been deleted, which is the failure mode a type-level check does not have.
//! The suites also assert the scan found something, for the same reason.
//!
//! # Why a column-zero brace ends a function
//!
//! The scan takes a function's body as everything up to the next line that is
//! exactly `}`. That is not a guess about formatting: both workspaces are
//! checked rustfmt-clean by the feature matrix, and rustfmt puts a top-level
//! item's closing brace in column zero. A file that stopped being formatted
//! would fail that arm before it reached this one.

use std::collections::BTreeMap;

/// A shape the scan could not read.
///
/// Returned rather than skipped: see the module doc.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScanRefusal {
    /// No `fn <name>` in the source given.
    NoSuchFunction { function: String },
    /// The function had no column-zero closing brace after it, so its body has
    /// no end the scan can name.
    Unterminated { function: String },
    /// A constant's initialiser was neither a hexadecimal literal nor the name
    /// of another constant in the same table.
    UnreadableConstant { name: String, initialiser: String },
    /// A constant alias chain that did not reach a literal. A cycle, or an
    /// initialiser naming something declared elsewhere.
    UnresolvedAlias { name: String },
}

impl std::fmt::Display for ScanRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchFunction { function } => write!(f, "no `fn {function}` in the source"),
            Self::Unterminated { function } => {
                write!(f, "`fn {function}` has no column-zero closing brace")
            }
            Self::UnreadableConstant { name, initialiser } => {
                write!(f, "{name} is initialised by `{initialiser}`")
            }
            Self::UnresolvedAlias { name } => write!(f, "{name} names no literal"),
        }
    }
}

/// Every `pub const <PREFIX>…: u16` in `source`, resolved through alias chains.
///
/// The device writes its opcode space as named constants, some of which are
/// aliases of others because one number is the same command on both channels.
/// A caller wants numbers, so the chain is followed here rather than at each
/// comparison.
///
/// # Errors
///
/// [`ScanRefusal::UnreadableConstant`] for an initialiser that is neither a hex
/// literal nor another constant's name, and [`ScanRefusal::UnresolvedAlias`]
/// for a chain that does not reach one.
pub fn u16_constants(
    source: &str,
    prefixes: &[&str],
) -> Result<BTreeMap<String, u16>, ScanRefusal> {
    let mut raw: BTreeMap<String, String> = BTreeMap::new();
    for line in source.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("pub const ") else {
            continue;
        };
        let Some((name, initialiser)) = rest.split_once(": u16 = ") else {
            continue;
        };
        if !prefixes.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        raw.insert(
            name.to_owned(),
            initialiser.trim_end_matches(';').trim().to_owned(),
        );
    }

    let mut out = BTreeMap::new();
    for name in raw.keys() {
        // Bounded rather than "until it resolves": a cycle is a source the scan
        // cannot read, and hanging on one is the failure mode a report exists
        // to replace.
        let mut at = name.clone();
        let mut value = None;
        for _ in 0..8 {
            let initialiser = &raw[&at];
            if let Some(hex) = initialiser.strip_prefix("0x") {
                let parsed =
                    u16::from_str_radix(hex, 16).map_err(|_| ScanRefusal::UnreadableConstant {
                        name: at.clone(),
                        initialiser: initialiser.clone(),
                    })?;
                value = Some(parsed);
                break;
            }
            if raw.contains_key(initialiser) {
                at.clone_from(initialiser);
                continue;
            }
            return Err(ScanRefusal::UnreadableConstant {
                name: at.clone(),
                initialiser: initialiser.clone(),
            });
        }
        let value = value.ok_or_else(|| ScanRefusal::UnresolvedAlias { name: name.clone() })?;
        out.insert(name.clone(), value);
    }
    Ok(out)
}

/// The opcode constants a function's body names, with their values.
///
/// "Names" and not "matches on": the scan does not parse `match` arms. A
/// constant mentioned anywhere in the body is one this dispatch can reach, and
/// a mention that is not an arm — a comment, an equality test inside an arm —
/// makes the answer wider rather than narrower. That direction is the safe one
/// for what the suites ask: they compare this set against the ledger, and a
/// scan that reported *fewer* opcodes than the function handles would let a
/// genuinely unhandled row pass unnoticed.
///
/// # Errors
///
/// [`ScanRefusal::NoSuchFunction`] and [`ScanRefusal::Unterminated`].
pub fn opcodes_named_in(
    source: &str,
    function: &str,
    constants: &BTreeMap<String, u16>,
) -> Result<BTreeMap<String, u16>, ScanRefusal> {
    let body = function_body(source, function)?;
    Ok(constants
        .iter()
        .filter(|(name, _)| names_identifier(body, name))
        .map(|(name, value)| (name.clone(), *value))
        .collect())
}

/// The text of `fn <function>`'s body, up to the column-zero brace that ends it.
fn function_body<'a>(source: &'a str, function: &str) -> Result<&'a str, ScanRefusal> {
    let needle = format!("fn {function}");
    let start = source
        .find(&needle)
        .ok_or_else(|| ScanRefusal::NoSuchFunction {
            function: function.to_owned(),
        })?;
    let tail = &source[start..];
    let end = tail.find("\n}").ok_or_else(|| ScanRefusal::Unterminated {
        function: function.to_owned(),
    })?;
    Ok(&tail[..end])
}

/// Whether `haystack` contains `name` as a whole identifier.
///
/// Whole, because `CHILD_OP_EXEC` is a prefix of `CHILD_OP_EXEC_INDIRECT2` and
/// a substring search would report the shorter one wherever the longer one
/// appears — an opcode the function does not handle, credited to it.
fn names_identifier(haystack: &str, name: &str) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(at) = haystack[from..].find(name) {
        let at = from + at;
        let before_ok = at == 0 || !haystack[..at].chars().next_back().is_some_and(ident);
        let after_ok = !haystack[at + name.len()..]
            .chars()
            .next()
            .is_some_and(ident);
        if before_ok && after_ok {
            return true;
        }
        from = at + 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGS: &str = "\
pub const CHILD_OP_ONE: u16 = 0x01;
pub const CHILD_OP_EXEC: u16 = 0x37;
pub const ROOT_OP_ONE: u16 = CHILD_OP_ONE;
pub const OTHER: u16 = 0x99;
";

    #[test]
    fn an_alias_resolves_to_the_number_it_names() {
        let c = u16_constants(REGS, &["ROOT_OP_", "CHILD_OP_"]).expect("readable");
        assert_eq!(c["ROOT_OP_ONE"], 1);
        assert_eq!(c["CHILD_OP_EXEC"], 0x37);
        assert!(
            !c.contains_key("OTHER"),
            "a constant outside the prefixes is not an opcode"
        );
    }

    #[test]
    fn an_initialiser_the_scan_cannot_read_is_reported() {
        assert_eq!(
            u16_constants(
                "pub const CHILD_OP_X: u16 = SOMETHING_ELSE;\n",
                &["CHILD_OP_"]
            ),
            Err(ScanRefusal::UnreadableConstant {
                name: "CHILD_OP_X".to_owned(),
                initialiser: "SOMETHING_ELSE".to_owned(),
            })
        );
    }

    #[test]
    fn an_alias_cycle_is_reported_rather_than_looped_on() {
        let source = "\
pub const CHILD_OP_A: u16 = CHILD_OP_B;
pub const CHILD_OP_B: u16 = CHILD_OP_A;
";
        assert_eq!(
            u16_constants(source, &["CHILD_OP_"]),
            Err(ScanRefusal::UnresolvedAlias {
                name: "CHILD_OP_A".to_owned(),
            })
        );
    }

    /// The prefix case: a scan that matched substrings would credit the shorter
    /// constant to every function naming the longer one.
    #[test]
    fn a_constant_that_is_another_ones_prefix_is_not_credited_to_it() {
        let source = "\
fn dispatch() {
    match op {
        CHILD_OP_EXEC_INDIRECT2 => go(),
    }
}
";
        let mut constants = BTreeMap::new();
        constants.insert("CHILD_OP_EXEC".to_owned(), 0x37);
        constants.insert("CHILD_OP_EXEC_INDIRECT2".to_owned(), 0x38);
        let named = opcodes_named_in(source, "dispatch", &constants).expect("found");
        assert_eq!(
            named.keys().collect::<Vec<_>>(),
            ["CHILD_OP_EXEC_INDIRECT2"]
        );
    }

    #[test]
    fn a_body_stops_at_the_column_zero_brace() {
        let source = "\
fn first() {
    CHILD_OP_A;
}
fn second() {
    CHILD_OP_B;
}
";
        let mut constants = BTreeMap::new();
        constants.insert("CHILD_OP_A".to_owned(), 1);
        constants.insert("CHILD_OP_B".to_owned(), 2);
        assert_eq!(
            opcodes_named_in(source, "first", &constants)
                .expect("found")
                .keys()
                .collect::<Vec<_>>(),
            ["CHILD_OP_A"]
        );
    }

    #[test]
    fn a_function_that_is_not_there_is_reported() {
        assert_eq!(
            opcodes_named_in("fn other() {\n}\n", "missing", &BTreeMap::new()),
            Err(ScanRefusal::NoSuchFunction {
                function: "missing".to_owned(),
            })
        );
    }

    #[test]
    fn a_function_with_no_closing_brace_is_reported() {
        assert_eq!(
            opcodes_named_in("fn open() {\n    work();\n", "open", &BTreeMap::new()),
            Err(ScanRefusal::Unterminated {
                function: "open".to_owned(),
            })
        );
    }
}
