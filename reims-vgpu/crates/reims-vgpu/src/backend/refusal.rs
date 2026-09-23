//! How this device says no, with the check that said it.
//!
//! One `Copy`, fixed-size record carrying a class, the registered slug of the
//! check that refused, and up to [`MAX_REFUSAL_FIELDS`] load-bearing facts. The
//! neutral status vocabularies — `EncodeStatus`, `ComputeStatus` — carry one in
//! their `RailRefused` variant, which is why it lives here and not under a
//! rail: those enums are the runtime's, and a variant whose payload named one
//! rail gave them two shapes across a feature boundary and made the *other*
//! rail unable to refuse with structure at all.
//!
//! Only the Metal rail builds one today (`backend::metal::error::Status` is
//! this type). That is a fact about which rail has been given structured
//! refusals so far, not about who may have them.

use crate::observe::Refusal;

/// The device's refusal codes. Historically a C backend header's; no
/// translation unit in this tree reads them today, and [`RailRefusal::code`] is
/// the only thing that maps onto them.
pub const REIMS_VGPU_OK: i32 = 0;
pub const REIMS_VGPU_ERR_ARGS: i32 = 1;
pub const REIMS_VGPU_ERR_TRANSLATE: i32 = 2;
pub const REIMS_VGPU_ERR_EXECUTE: i32 = 3;

/// Maximum structured fields one refusal carries.
///
/// Deliberately fixed-size so [`RailRefusal`] stays `Copy`: a rail passes it
/// through several leaf helpers before the runtime turns it into
/// `EncodeStatus` / `ComputeStatus`. Six covers the widest current
/// check (geometry + expected/actual length) without allocating on success.
const MAX_REFUSAL_FIELDS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldValue {
    Unsigned(u64),
    Signed(i64),
    Text(&'static str),
}

impl std::fmt::Display for FieldValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsigned(value) => value.fmt(f),
            Self::Signed(value) => value.fmt(f),
            Self::Text(value) => value.fmt(f),
        }
    }
}

macro_rules! unsigned_field_value {
    ($($ty:ty),+ $(,)?) => {
        $(impl From<$ty> for FieldValue {
            fn from(value: $ty) -> Self {
                Self::Unsigned(value as u64)
            }
        })+
    };
}

macro_rules! signed_field_value {
    ($($ty:ty),+ $(,)?) => {
        $(impl From<$ty> for FieldValue {
            fn from(value: $ty) -> Self {
                Self::Signed(value as i64)
            }
        })+
    };
}

unsigned_field_value!(u8, u16, u32, u64, usize);
signed_field_value!(i8, i16, i32, i64, isize);

impl From<bool> for FieldValue {
    fn from(value: bool) -> Self {
        Self::Unsigned(value as u64)
    }
}

impl From<&'static str> for FieldValue {
    fn from(value: &'static str) -> Self {
        Self::Text(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RefusalField {
    key: &'static str,
    value: FieldValue,
}

/// Result of a direct-Metal backend operation.
///
/// The C ABI still observes the historical integer code, but Rust never
/// constructs a payload-free `ARGS` / `EXECUTE` value. Every refusal carries
/// the registered slug of the exact check plus its numeric protocol facts.
/// `Ok` remains in the same status type so [`Refusal::refusal`] makes the
/// success-vs-decline judgement exhaustive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefusalClass {
    Ok,
    Args,
    Execute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RailRefusal {
    class: RefusalClass,
    slug: Option<&'static str>,
    fields: [Option<RefusalField>; MAX_REFUSAL_FIELDS],
}

impl RailRefusal {
    pub const OK: Self = Self {
        class: RefusalClass::Ok,
        slug: None,
        fields: [None; MAX_REFUSAL_FIELDS],
    };

    pub fn args(slug: &'static str) -> Self {
        Self {
            class: RefusalClass::Args,
            slug: Some(slug),
            fields: [None; MAX_REFUSAL_FIELDS],
        }
    }

    pub fn execute(slug: &'static str) -> Self {
        Self {
            class: RefusalClass::Execute,
            slug: Some(slug),
            fields: [None; MAX_REFUSAL_FIELDS],
        }
    }

    /// Add one load-bearing numeric or static-token fact to the refusal.
    ///
    /// Overflow is an authoring defect, not a runtime fallback: every current
    /// constructor is kept within [`MAX_REFUSAL_FIELDS`] and the unit test pins
    /// that a seventh field cannot silently replace an earlier one.
    pub fn field(mut self, key: &'static str, value: impl Into<FieldValue>) -> Self {
        if self.class == RefusalClass::Ok {
            return self;
        }
        let Some(slot) = self.fields.iter_mut().find(|slot| slot.is_none()) else {
            panic!("refusal field capacity exceeded for {key}");
        };
        *slot = Some(RefusalField {
            key,
            value: value.into(),
        });
        self
    }

    pub fn code(self) -> i32 {
        match self.class {
            RefusalClass::Ok => REIMS_VGPU_OK,
            RefusalClass::Args => REIMS_VGPU_ERR_ARGS,
            RefusalClass::Execute => REIMS_VGPU_ERR_EXECUTE,
        }
    }

    pub fn is_ok(self) -> bool {
        self.class == RefusalClass::Ok
    }

    pub fn is_args(self) -> bool {
        self.class == RefusalClass::Args
    }
}

impl Refusal for RailRefusal {
    fn refusal(&self) -> Option<&'static str> {
        self.slug
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let class = match self.class {
            RefusalClass::Ok => return Vec::new(),
            RefusalClass::Args => "args",
            RefusalClass::Execute => "execute",
        };
        let mut out = Vec::with_capacity(1 + MAX_REFUSAL_FIELDS);
        out.push(("class", class.to_string()));
        out.extend(
            self.fields
                .iter()
                .flatten()
                .map(|field| (field.key, field.value.to_string())),
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Emit;

    #[test]
    fn status_preserves_abi_class_reason_and_structured_fields() {
        let status = RailRefusal::args("metal_test_bad_extent")
            .field("width", 19u32)
            .field("offset", -4i32);
        assert_eq!(status.code(), REIMS_VGPU_ERR_ARGS);
        assert!(status.is_args());
        assert_eq!(status.refusal(), Some("metal_test_bad_extent"));
        assert_eq!(
            Emit::refusal("metal_backend", &status)
                .expect("a refusal must render")
                .render(),
            "metal_backend reason=metal_test_bad_extent class=args width=19 offset=-4"
        );
    }

    #[test]
    fn ok_cannot_be_emitted_as_a_refusal() {
        assert_eq!(RailRefusal::OK.code(), REIMS_VGPU_OK);
        assert!(RailRefusal::OK.is_ok());
        assert!(Emit::refusal("metal_backend", &RailRefusal::OK).is_none());
    }

    #[test]
    #[should_panic(expected = "refusal field capacity exceeded")]
    fn structured_field_overflow_is_not_silent() {
        let _ = RailRefusal::execute("metal_test_field_overflow")
            .field("a", 1u8)
            .field("b", 2u8)
            .field("c", 3u8)
            .field("d", 4u8)
            .field("e", 5u8)
            .field("f", 6u8)
            .field("g", 7u8);
    }
}
