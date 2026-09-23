//! `MTLSamplerDescriptor`'s ordinals, and the combinations the API itself does
//! not admit.
//!
//! # Why the ordinals are parsed once
//!
//! A sampler is six enumerations, two clamps, an anisotropy and three flags,
//! and every one of the enumerations has a closed set. Folding an unrecognised
//! address mode onto `ClampToEdge` samples the edge texel where the guest
//! wanted a repeat — a visible, plausible, unattributable difference — so each
//! parses to `None` outside its set and the refusal carries the ordinal.
//!
//! # Unnormalized coordinates are a mode, not a flag
//!
//! `MTLSamplerDescriptor.normalizedCoordinates = NO` puts the sampler in a
//! restricted mode in both APIs: one mip level, no anisotropy, no comparison,
//! and only the clamping address modes. Metal documents those restrictions and
//! Vulkan makes them validation rules, so a descriptor that breaks one is not
//! a sampler either API will build. [`SamplerShape::checked`] refuses it here,
//! with the field that broke it, rather than letting each backend discover a
//! different subset of the rules.
//!
//! # What is deliberately not here
//!
//! Whether a *host* can build the sampler. Anisotropy limits, mirror-clamp
//! support and border-colour kinds are physical-device properties and belong
//! to the executor that queried one. This layer answers only what the guest
//! API itself admits.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// `MTLSamplerMinMagFilter`.
pub const MTL_SAMPLER_MIN_MAG_FILTER_NEAREST: u32 = 0;
pub const MTL_SAMPLER_MIN_MAG_FILTER_LINEAR: u32 = 1;

/// `MTLSamplerMipFilter`.
pub const MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED: u32 = 0;
pub const MTL_SAMPLER_MIP_FILTER_NEAREST: u32 = 1;
pub const MTL_SAMPLER_MIP_FILTER_LINEAR: u32 = 2;

/// `MTLSamplerAddressMode`.
pub const MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: u32 = 0;
pub const MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: u32 = 1;
pub const MTL_SAMPLER_ADDRESS_MODE_REPEAT: u32 = 2;
pub const MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT: u32 = 3;
pub const MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO: u32 = 4;
pub const MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR: u32 = 5;

/// `MTLSamplerBorderColor`.
pub const MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK: u32 = 0;
pub const MTL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK: u32 = 1;
pub const MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE: u32 = 2;

/// `MTLCompareFunction`. Shared with the depth-stencil state, which is why it
/// is spelled once.
pub const MTL_COMPARE_FUNCTION_NEVER: u32 = 0;
pub const MTL_COMPARE_FUNCTION_LESS: u32 = 1;
pub const MTL_COMPARE_FUNCTION_EQUAL: u32 = 2;
pub const MTL_COMPARE_FUNCTION_LESS_EQUAL: u32 = 3;
pub const MTL_COMPARE_FUNCTION_GREATER: u32 = 4;
pub const MTL_COMPARE_FUNCTION_NOT_EQUAL: u32 = 5;
pub const MTL_COMPARE_FUNCTION_GREATER_EQUAL: u32 = 6;
pub const MTL_COMPARE_FUNCTION_ALWAYS: u32 = 7;

/// Minification and magnification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Filter {
    Nearest,
    Linear,
}

impl Filter {
    pub const ALL: [Filter; 2] = [Self::Nearest, Self::Linear];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_SAMPLER_MIN_MAG_FILTER_NEAREST => Self::Nearest,
            MTL_SAMPLER_MIN_MAG_FILTER_LINEAR => Self::Linear,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Linear => "linear",
        }
    }
}

/// How levels of a mip chain are combined.
///
/// `NotMipmapped` is a third answer and not a filter: it says only the top
/// level is ever sampled, which is a clamp on the level of detail rather than
/// a choice of how to blend two levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MipFilter {
    NotMipmapped,
    Nearest,
    Linear,
}

impl MipFilter {
    pub const ALL: [MipFilter; 3] = [Self::NotMipmapped, Self::Nearest, Self::Linear];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED => Self::NotMipmapped,
            MTL_SAMPLER_MIP_FILTER_NEAREST => Self::Nearest,
            MTL_SAMPLER_MIP_FILTER_LINEAR => Self::Linear,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NotMipmapped => "not_mipmapped",
            Self::Nearest => "nearest",
            Self::Linear => "linear",
        }
    }
}

/// What happens outside `[0, 1]` on one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AddressMode {
    ClampToEdge,
    MirrorClampToEdge,
    Repeat,
    MirrorRepeat,
    /// Clamp to transparent black. A border mode whose border is fixed, which
    /// is why it is distinct from [`Self::ClampToBorderColor`] rather than a
    /// spelling of it.
    ClampToZero,
    ClampToBorderColor,
}

impl AddressMode {
    pub const ALL: [AddressMode; 6] = [
        Self::ClampToEdge,
        Self::MirrorClampToEdge,
        Self::Repeat,
        Self::MirrorRepeat,
        Self::ClampToZero,
        Self::ClampToBorderColor,
    ];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE => Self::ClampToEdge,
            MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE => Self::MirrorClampToEdge,
            MTL_SAMPLER_ADDRESS_MODE_REPEAT => Self::Repeat,
            MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT => Self::MirrorRepeat,
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO => Self::ClampToZero,
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR => Self::ClampToBorderColor,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ClampToEdge => "clamp_to_edge",
            Self::MirrorClampToEdge => "mirror_clamp_to_edge",
            Self::Repeat => "repeat",
            Self::MirrorRepeat => "mirror_repeat",
            Self::ClampToZero => "clamp_to_zero",
            Self::ClampToBorderColor => "clamp_to_border_color",
        }
    }

    /// Whether this mode reads a border rather than a texel of the image.
    #[must_use]
    pub const fn uses_border(self) -> bool {
        matches!(self, Self::ClampToZero | Self::ClampToBorderColor)
    }

    /// Whether an unnormalized-coordinate sampler may use this mode.
    ///
    /// Only the clamping modes: a repeat over unnormalized coordinates has no
    /// period to repeat over. Both APIs say so, and the restriction is the
    /// same one, so it is stated once here.
    #[must_use]
    pub const fn allows_unnormalized(self) -> bool {
        matches!(
            self,
            Self::ClampToEdge | Self::ClampToZero | Self::ClampToBorderColor
        )
    }
}

/// The border a border mode reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

impl BorderColor {
    pub const ALL: [BorderColor; 3] =
        [Self::TransparentBlack, Self::OpaqueBlack, Self::OpaqueWhite];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK => Self::TransparentBlack,
            MTL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK => Self::OpaqueBlack,
            MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE => Self::OpaqueWhite,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::TransparentBlack => "transparent_black",
            Self::OpaqueBlack => "opaque_black",
            Self::OpaqueWhite => "opaque_white",
        }
    }
}

/// `MTLCompareFunction`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl CompareFunction {
    pub const ALL: [CompareFunction; 8] = [
        Self::Never,
        Self::Less,
        Self::Equal,
        Self::LessEqual,
        Self::Greater,
        Self::NotEqual,
        Self::GreaterEqual,
        Self::Always,
    ];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_COMPARE_FUNCTION_NEVER => Self::Never,
            MTL_COMPARE_FUNCTION_LESS => Self::Less,
            MTL_COMPARE_FUNCTION_EQUAL => Self::Equal,
            MTL_COMPARE_FUNCTION_LESS_EQUAL => Self::LessEqual,
            MTL_COMPARE_FUNCTION_GREATER => Self::Greater,
            MTL_COMPARE_FUNCTION_NOT_EQUAL => Self::NotEqual,
            MTL_COMPARE_FUNCTION_GREATER_EQUAL => Self::GreaterEqual,
            MTL_COMPARE_FUNCTION_ALWAYS => Self::Always,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::Never => MTL_COMPARE_FUNCTION_NEVER,
            Self::Less => MTL_COMPARE_FUNCTION_LESS,
            Self::Equal => MTL_COMPARE_FUNCTION_EQUAL,
            Self::LessEqual => MTL_COMPARE_FUNCTION_LESS_EQUAL,
            Self::Greater => MTL_COMPARE_FUNCTION_GREATER,
            Self::NotEqual => MTL_COMPARE_FUNCTION_NOT_EQUAL,
            Self::GreaterEqual => MTL_COMPARE_FUNCTION_GREATER_EQUAL,
            Self::Always => MTL_COMPARE_FUNCTION_ALWAYS,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::Less => "less",
            Self::Equal => "equal",
            Self::LessEqual => "less_equal",
            Self::Greater => "greater",
            Self::NotEqual => "not_equal",
            Self::GreaterEqual => "greater_equal",
            Self::Always => "always",
        }
    }
}

/// A sampler declaration as the fields arrived.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplerShape {
    pub min_filter: u32,
    pub mag_filter: u32,
    pub mip_filter: u32,
    pub s_address: u32,
    pub t_address: u32,
    pub r_address: u32,
    /// Floored at one by the decoder. One is "no anisotropic filtering".
    pub max_anisotropy: u32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    /// `MTLCompareFunction` ordinal. A sampler with no comparison is
    /// `Never` in Metal's own default, so this is not an `Option` on the wire
    /// — see [`SamplerState::compare`].
    pub compare_function: u32,
    /// Whether a comparison is performed at all. Metal's descriptor has no
    /// separate flag; the guest's serializer does, so it is carried rather
    /// than inferred from the function.
    pub compare_enabled: bool,
    pub border_color: u32,
    pub normalized_coordinates: bool,
}

/// Why a sampler declaration is not one the guest API admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerRefusal {
    UnknownOrdinal {
        field: &'static str,
        ordinal: u32,
    },
    /// An unnormalized-coordinate sampler broke one of the restrictions both
    /// APIs place on it.
    UnnormalizedRestriction {
        field: &'static str,
    },
    /// The level-of-detail clamp is inverted or is not a number.
    BadLodClamp {
        min_bits: u32,
        max_bits: u32,
    },
}

impl reims_vgpu_observe::Decline for SamplerRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownOrdinal { .. } => "sampler_unknown_ordinal",
            Self::UnnormalizedRestriction { .. } => "sampler_unnormalized_restriction",
            Self::BadLodClamp { .. } => "sampler_bad_lod_clamp",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnknownOrdinal { field, ordinal } => vec![
                ("field", (*field).to_string()),
                ("ordinal", ordinal.to_string()),
            ],
            Self::UnnormalizedRestriction { field } => vec![("field", (*field).to_string())],
            Self::BadLodClamp { min_bits, max_bits } => vec![
                ("min_bits", min_bits.to_string()),
                ("max_bits", max_bits.to_string()),
            ],
        }
    }
}

/// A sampler declaration whose fields have been parsed and checked against
/// each other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerState {
    min_filter: Filter,
    mag_filter: Filter,
    mip_filter: MipFilter,
    address: [AddressMode; 3],
    max_anisotropy: u32,
    lod_min_clamp: f32,
    lod_max_clamp: f32,
    compare: Option<CompareFunction>,
    border_color: BorderColor,
    normalized_coordinates: bool,
    /// Whether the unnormalized conformance below changed anything.
    unnormalized_conformed: bool,
}

fn parsed<T>(field: &'static str, ordinal: u32, value: Option<T>) -> Result<T, SamplerRefusal> {
    value.ok_or(SamplerRefusal::UnknownOrdinal { field, ordinal })
}

impl SamplerShape {
    /// Parse and check the declaration.
    ///
    /// # Errors
    ///
    /// [`SamplerRefusal`] naming the field that failed.
    pub fn checked(self) -> Result<SamplerState, SamplerRefusal> {
        let min_filter = parsed(
            "min_filter",
            self.min_filter,
            Filter::parse(self.min_filter),
        )?;
        let mag_filter = parsed(
            "mag_filter",
            self.mag_filter,
            Filter::parse(self.mag_filter),
        )?;
        let mip_filter = parsed(
            "mip_filter",
            self.mip_filter,
            MipFilter::parse(self.mip_filter),
        )?;
        let address = [
            parsed(
                "s_address",
                self.s_address,
                AddressMode::parse(self.s_address),
            )?,
            parsed(
                "t_address",
                self.t_address,
                AddressMode::parse(self.t_address),
            )?,
            parsed(
                "r_address",
                self.r_address,
                AddressMode::parse(self.r_address),
            )?,
        ];
        let border_color = parsed(
            "border_color",
            self.border_color,
            BorderColor::parse(self.border_color),
        )?;
        let compare = self
            .compare_enabled
            .then(|| {
                parsed(
                    "compare_function",
                    self.compare_function,
                    CompareFunction::parse(self.compare_function),
                )
            })
            .transpose()?;

        if !self.lod_min_clamp.is_finite()
            || !self.lod_max_clamp.is_finite()
            || self.lod_min_clamp > self.lod_max_clamp
        {
            return Err(SamplerRefusal::BadLodClamp {
                min_bits: self.lod_min_clamp.to_bits(),
                max_bits: self.lod_max_clamp.to_bits(),
            });
        }

        let mut state = SamplerState {
            min_filter,
            mag_filter,
            mip_filter,
            address,
            // One is "off"; the decoder floors it there and nothing below
            // means anything.
            max_anisotropy: self.max_anisotropy.max(1),
            lod_min_clamp: self.lod_min_clamp,
            lod_max_clamp: self.lod_max_clamp,
            compare,
            border_color,
            normalized_coordinates: self.normalized_coordinates,
            unnormalized_conformed: false,
        };
        if !self.normalized_coordinates {
            state = state.conform_unnormalized()?;
        }
        Ok(state)
    }
}

impl SamplerState {
    /// Bring an unnormalized-coordinate declaration inside the restriction
    /// both APIs place on one.
    ///
    /// # Why this conforms where it could refuse
    ///
    /// Metal and Vulkan agree on the restriction — same filter for min and
    /// mag, no mip chain, no anisotropy, no comparison, and only clamping
    /// address modes — and both state it as a requirement on the caller rather
    /// than as something the API checks. So a declaration that breaks it is a
    /// guest that already left the documented region, and what Metal's own
    /// driver did with it is not knowable from the declaration.
    ///
    /// Between dropping the guest's work and binding the nearest sampler that
    /// is legal in both APIs, the second loses nothing: every field this
    /// changes is one an unnormalized sampler cannot consult. A mip level is
    /// never selected because there is no derivative to select it with, an
    /// anisotropy ratio needs the same derivative, and a repeat has no period
    /// to repeat over. The declared values are unreadable, not overridden.
    ///
    /// The comparison is the exception and stays a refusal: it changes *what
    /// the shader reads* — a comparison result instead of a texel — and there
    /// is no nearby sampler that reads the same thing. Silently dropping it
    /// would hand the shader a value of the wrong kind.
    ///
    /// # Errors
    ///
    /// [`SamplerRefusal::UnnormalizedRestriction`] naming `compare_function`.
    fn conform_unnormalized(self) -> Result<Self, SamplerRefusal> {
        if self.compare.is_some() {
            return Err(SamplerRefusal::UnnormalizedRestriction {
                field: "compare_function",
            });
        }
        let mut conformed = self;
        // Mag rather than min: a pixel-coordinate sampler is reading at or
        // above 1:1 in every use this rail has seen, and magnification is the
        // filter that decides what that reads.
        conformed.min_filter = self.mag_filter;
        conformed.mip_filter = MipFilter::NotMipmapped;
        conformed.max_anisotropy = 1;
        conformed.lod_min_clamp = 0.0;
        conformed.lod_max_clamp = 0.0;
        for mode in &mut conformed.address {
            if !mode.allows_unnormalized() {
                *mode = AddressMode::ClampToEdge;
            }
        }
        // Read before the flag is set, so the comparison is over the declared
        // fields and never over the flag itself.
        let changed = conformed != self;
        conformed.unnormalized_conformed = changed;
        Ok(conformed)
    }

    /// Whether this declaration was an unnormalized one that had to be
    /// conformed, and is therefore not exactly what the guest wrote.
    ///
    /// A backend reports it rather than deciding on it: every field the
    /// conformance touched is one the sampler cannot consult, so there is no
    /// behaviour to branch on — only a population worth counting.
    #[must_use]
    pub const fn unnormalized_conformed(self) -> bool {
        self.unnormalized_conformed
    }

    #[must_use]
    pub const fn min_filter(self) -> Filter {
        self.min_filter
    }

    #[must_use]
    pub const fn mag_filter(self) -> Filter {
        self.mag_filter
    }

    #[must_use]
    pub const fn mip_filter(self) -> MipFilter {
        self.mip_filter
    }

    /// The three axes, in `s`, `t`, `r` order.
    #[must_use]
    pub const fn address(self) -> [AddressMode; 3] {
        self.address
    }

    #[must_use]
    pub const fn max_anisotropy(self) -> u32 {
        self.max_anisotropy
    }

    #[must_use]
    pub const fn lod_clamp(self) -> (f32, f32) {
        (self.lod_min_clamp, self.lod_max_clamp)
    }

    #[must_use]
    pub const fn compare(self) -> Option<CompareFunction> {
        self.compare
    }

    #[must_use]
    pub const fn border_color(self) -> BorderColor {
        self.border_color
    }

    #[must_use]
    pub const fn normalized_coordinates(self) -> bool {
        self.normalized_coordinates
    }

    /// Whether any axis reads a border.
    ///
    /// The question a backend asks before it decides whether the border colour
    /// matters at all — and it is per-sampler in both APIs, which is why the
    /// axes are checked together rather than one at a time.
    #[must_use]
    pub fn uses_border(self) -> bool {
        self.address.iter().any(|m| m.uses_border())
    }

    /// Whether two border modes on different axes disagree about the border.
    ///
    /// `ClampToZero` has a fixed transparent-black border and
    /// `ClampToBorderColor` reads the declared one. A descriptor using both
    /// with a non-transparent border colour is asking for two borders, and
    /// neither API has more than one — so this is the question a backend has
    /// to ask before it picks one.
    #[must_use]
    pub fn border_modes_disagree(self) -> bool {
        let zero = self.address.contains(&AddressMode::ClampToZero);
        let declared = self.address.contains(&AddressMode::ClampToBorderColor);
        zero && declared && self.border_color != BorderColor::TransparentBlack
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use reims_vgpu_observe::Decline;

    fn base() -> SamplerShape {
        SamplerShape {
            min_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mag_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mip_filter: MTL_SAMPLER_MIP_FILTER_LINEAR,
            s_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            t_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            r_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            max_anisotropy: 1,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1000.0,
            compare_function: MTL_COMPARE_FUNCTION_NEVER,
            compare_enabled: false,
            border_color: MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
            normalized_coordinates: true,
        }
    }

    /// The unnormalized mode's own valid base, so a test mutating one field
    /// sees that field's refusal.
    fn unnormalized() -> SamplerShape {
        SamplerShape {
            mip_filter: MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            t_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            r_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            normalized_coordinates: false,
            // Zero, not the base's 1000: an unnormalized sampler selects no
            // level, so a clamp that admits one is itself a thing to conform.
            lod_max_clamp: 0.0,
            ..base()
        }
    }

    #[test]
    fn every_ordinal_set_is_closed_and_round_trips() {
        for (index, filter) in Filter::ALL.iter().enumerate() {
            assert_eq!(Filter::parse(index as u32), Some(*filter));
        }
        assert_eq!(Filter::parse(2), None);

        for (index, mip) in MipFilter::ALL.iter().enumerate() {
            assert_eq!(MipFilter::parse(index as u32), Some(*mip));
        }
        assert_eq!(MipFilter::parse(3), None);

        for (index, mode) in AddressMode::ALL.iter().enumerate() {
            assert_eq!(AddressMode::parse(index as u32), Some(*mode));
        }
        assert_eq!(AddressMode::parse(6), None);

        for (index, border) in BorderColor::ALL.iter().enumerate() {
            assert_eq!(BorderColor::parse(index as u32), Some(*border));
        }
        assert_eq!(BorderColor::parse(3), None);

        for (index, compare) in CompareFunction::ALL.iter().enumerate() {
            assert_eq!(CompareFunction::parse(index as u32), Some(*compare));
            assert_eq!(compare.ordinal(), index as u32);
        }
        assert_eq!(CompareFunction::parse(8), None);
    }

    #[test]
    fn every_name_in_a_set_is_distinct() {
        for names in [
            Filter::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
            MipFilter::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
            AddressMode::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
            BorderColor::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
            CompareFunction::ALL
                .iter()
                .map(|f| f.name())
                .collect::<BTreeSet<_>>(),
        ] {
            assert!(!names.is_empty());
        }
        assert_eq!(
            AddressMode::ALL
                .iter()
                .map(|m| m.name())
                .collect::<BTreeSet<_>>()
                .len(),
            AddressMode::ALL.len()
        );
    }

    #[test]
    fn an_unknown_ordinal_names_its_field_and_is_never_folded() {
        for (mutate, field, ordinal) in [
            (
                SamplerShape {
                    min_filter: 9,
                    ..base()
                },
                "min_filter",
                9u32,
            ),
            (
                SamplerShape {
                    mip_filter: 9,
                    ..base()
                },
                "mip_filter",
                9,
            ),
            (
                SamplerShape {
                    t_address: 9,
                    ..base()
                },
                "t_address",
                9,
            ),
            (
                SamplerShape {
                    border_color: 9,
                    ..base()
                },
                "border_color",
                9,
            ),
        ] {
            assert_eq!(
                mutate.checked(),
                Err(SamplerRefusal::UnknownOrdinal { field, ordinal })
            );
        }
    }

    #[test]
    fn a_comparison_function_is_read_only_where_one_is_enabled() {
        // An unrecognised function is not an error where no comparison is
        // performed: the field is Metal's default and not a request.
        let disabled = SamplerShape {
            compare_function: 99,
            compare_enabled: false,
            ..base()
        };
        assert_eq!(disabled.checked().expect("no comparison").compare(), None);

        let enabled = SamplerShape {
            compare_function: MTL_COMPARE_FUNCTION_LESS_EQUAL,
            compare_enabled: true,
            ..base()
        };
        assert_eq!(
            enabled.checked().expect("a comparison").compare(),
            Some(CompareFunction::LessEqual)
        );

        let bad = SamplerShape {
            compare_function: 99,
            compare_enabled: true,
            ..base()
        };
        assert!(matches!(
            bad.checked(),
            Err(SamplerRefusal::UnknownOrdinal {
                field: "compare_function",
                ..
            })
        ));
    }

    #[test]
    fn an_inverted_or_absent_lod_clamp_refuses() {
        for (min, max) in [(2.0f32, 1.0f32), (f32::NAN, 1.0), (0.0, f32::INFINITY)] {
            assert!(matches!(
                SamplerShape {
                    lod_min_clamp: min,
                    lod_max_clamp: max,
                    ..base()
                }
                .checked(),
                Err(SamplerRefusal::BadLodClamp { .. })
            ));
        }
        // Equal bounds are a clamp to one level, which is legal.
        assert!(SamplerShape {
            lod_min_clamp: 3.0,
            lod_max_clamp: 3.0,
            ..base()
        }
        .checked()
        .is_ok());
    }

    #[test]
    fn every_unnormalized_restriction_but_the_comparison_is_conformed_away() {
        let base = unnormalized().checked().expect("the base is valid");
        assert!(!base.unnormalized_conformed(), "nothing to conform");

        // Each declaration breaks one rule, and each comes back as the nearest
        // sampler both APIs admit rather than as a lost draw.
        for (shape, what) in [
            (
                SamplerShape {
                    min_filter: MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
                    ..unnormalized()
                },
                "filter",
            ),
            (
                SamplerShape {
                    mip_filter: MTL_SAMPLER_MIP_FILTER_LINEAR,
                    ..unnormalized()
                },
                "mip_filter",
            ),
            (
                SamplerShape {
                    max_anisotropy: 4,
                    ..unnormalized()
                },
                "max_anisotropy",
            ),
            (
                SamplerShape {
                    t_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
                    ..unnormalized()
                },
                "t_address",
            ),
            (
                SamplerShape {
                    lod_min_clamp: 1.0,
                    lod_max_clamp: 4.0,
                    ..unnormalized()
                },
                "lod_clamp",
            ),
        ] {
            let state = shape.checked().unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert!(state.unnormalized_conformed(), "{what} was conformed");
            // The conformed value is the same one in every case: an
            // unnormalized sampler has exactly one legal shape per filter.
            assert_eq!(state.min_filter(), state.mag_filter(), "{what}");
            assert_eq!(state.mip_filter(), MipFilter::NotMipmapped, "{what}");
            assert_eq!(state.max_anisotropy(), 1, "{what}");
            assert_eq!(state.lod_clamp(), (0.0, 0.0), "{what}");
            for mode in state.address() {
                assert!(mode.allows_unnormalized(), "{what}: {mode:?}");
            }
        }
    }

    /// The one restriction that is not conformed away, because conforming it
    /// would hand the shader a texel where it asked for a comparison.
    #[test]
    fn an_unnormalized_comparison_is_a_refusal_and_not_a_conformance() {
        assert_eq!(
            SamplerShape {
                compare_enabled: true,
                compare_function: MTL_COMPARE_FUNCTION_LESS,
                ..unnormalized()
            }
            .checked(),
            Err(SamplerRefusal::UnnormalizedRestriction {
                field: "compare_function"
            })
        );
    }

    /// Conforming reports itself only when it changed something. A guest that
    /// already wrote a legal unnormalized sampler is not counted as repaired.
    #[test]
    fn a_conformance_that_changed_nothing_does_not_report_itself() {
        for min_mag in [
            MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
            MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
        ] {
            let state = SamplerShape {
                min_filter: min_mag,
                mag_filter: min_mag,
                ..unnormalized()
            }
            .checked()
            .expect("a legal unnormalized sampler");
            assert!(!state.unnormalized_conformed());
        }
    }

    /// Magnification decides, so a mismatched pair conforms *to mag* and never
    /// to min. Both directions, because a table that took `min` would pass a
    /// test that only checked the two ended up equal.
    #[test]
    fn a_mismatched_filter_pair_conforms_to_the_magnifying_one() {
        for (min, mag, expected) in [
            (
                MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
                MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
                Filter::Linear,
            ),
            (
                MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
                MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
                Filter::Nearest,
            ),
        ] {
            let state = SamplerShape {
                min_filter: min,
                mag_filter: mag,
                ..unnormalized()
            }
            .checked()
            .expect("a conformable sampler");
            assert_eq!(state.min_filter(), expected);
            assert_eq!(state.mag_filter(), expected);
        }
    }

    #[test]
    fn the_restrictions_apply_only_to_the_unnormalized_mode() {
        // Every one of them is ordinary on a normalized sampler.
        let ordinary = SamplerShape {
            min_filter: MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
            mag_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mip_filter: MTL_SAMPLER_MIP_FILTER_LINEAR,
            max_anisotropy: 16,
            compare_enabled: true,
            compare_function: MTL_COMPARE_FUNCTION_GREATER,
            ..base()
        };
        let state = ordinary.checked().expect("a normalized sampler");
        assert_eq!(state.max_anisotropy(), 16);
        assert!(state.normalized_coordinates());
    }

    /// The conformance is per axis and does not disturb the axes that were
    /// already legal — including the border modes, which are legal here and
    /// carry a meaning `ClampToEdge` does not.
    #[test]
    fn conforming_one_axis_leaves_the_others_as_the_guest_wrote_them() {
        let state = SamplerShape {
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
            t_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
            r_address: MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT,
            ..unnormalized()
        }
        .checked()
        .expect("two legal axes and one that conforms");
        assert_eq!(
            state.address(),
            [
                AddressMode::ClampToZero,
                AddressMode::ClampToBorderColor,
                AddressMode::ClampToEdge,
            ]
        );
        assert!(state.unnormalized_conformed());
    }

    #[test]
    fn anisotropy_is_floored_at_one_because_below_it_means_nothing() {
        let state = SamplerShape {
            max_anisotropy: 0,
            ..base()
        }
        .checked()
        .expect("a sampler");
        assert_eq!(state.max_anisotropy(), 1);
    }

    #[test]
    fn a_border_is_used_only_where_an_axis_reads_one() {
        let none = base().checked().expect("repeat on every axis");
        assert!(!none.uses_border());

        for mode in [
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
            MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
        ] {
            let state = SamplerShape {
                t_address: mode,
                ..base()
            }
            .checked()
            .expect("a sampler");
            assert!(state.uses_border());
        }
    }

    #[test]
    fn two_border_modes_disagree_only_when_the_declared_border_is_not_transparent() {
        // `ClampToZero` is a fixed transparent-black border and
        // `ClampToBorderColor` reads the declared one. Together with an opaque
        // declaration that is two borders, and neither API has two.
        let clash = SamplerShape {
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
            t_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
            border_color: MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE,
            ..base()
        }
        .checked()
        .expect("a declaration the guest API admits");
        assert!(clash.border_modes_disagree());

        // The same pair with a transparent declaration is one border twice.
        let agreed = SamplerShape {
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_ZERO,
            t_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
            border_color: MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
            ..base()
        }
        .checked()
        .expect("a sampler");
        assert!(!agreed.border_modes_disagree());

        // And one border mode alone never disagrees, whatever the colour.
        let alone = SamplerShape {
            s_address: MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER_COLOR,
            border_color: MTL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK,
            ..base()
        }
        .checked()
        .expect("a sampler");
        assert!(!alone.border_modes_disagree());
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            SamplerRefusal::UnknownOrdinal {
                field: "min_filter",
                ordinal: 9,
            },
            SamplerRefusal::UnnormalizedRestriction { field: "filter" },
            SamplerRefusal::BadLodClamp {
                min_bits: 1,
                max_bits: 0,
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(Decline::slug).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.slug().starts_with("sampler_"));
            assert!(!refusal.fields().is_empty());
        }
    }

    /// Every declaration the checkpoint admits, checked against the two rules
    /// the checkpoint is for.
    ///
    /// `every_unnormalized_restriction_but_the_comparison_is_conformed_away`
    /// drives five declarations, each breaking one rule, and the conformance
    /// touches six fields on three axes. What that list cannot say is what
    /// happens when two rules break at once, or when the rule breaks on the
    /// `s` or `r` axis rather than the `t` one — and a conformance is exactly
    /// the kind of pass that fixes one field while reading another it forgot
    /// to fix first.
    ///
    /// So it is driven over the product instead. Both directions matter and
    /// both are asserted: an unnormalized state satisfies the restriction, and
    /// a normalized one is untouched. A conformance that ran on every
    /// declaration would satisfy the first alone.
    #[test]
    fn no_admitted_sampler_leaves_the_region_its_mode_allows() {
        let mut unnormalized_seen = 0u32;
        let mut conformed_seen = 0u32;
        let mut normalized_seen = 0u32;
        let mut refused = 0u32;
        // One past each closed set, so an unknown ordinal is driven on every
        // field rather than assumed unreachable.
        for min_filter in 0..=2 {
            for mag_filter in 0..=2 {
                for mip_filter in 0..=3 {
                    for s_address in 0..=6 {
                        for t_address in 0..=6 {
                            for r_address in [0, 2, 5, 6] {
                                for max_anisotropy in [0, 1, 4] {
                                    for (lod_min_clamp, lod_max_clamp) in
                                        [(0.0, 0.0), (1.0, 4.0), (0.0, f32::MAX)]
                                    {
                                        for compare_enabled in [false, true] {
                                            for normalized_coordinates in [false, true] {
                                                let shape = SamplerShape {
                                                    min_filter,
                                                    mag_filter,
                                                    mip_filter,
                                                    s_address,
                                                    t_address,
                                                    r_address,
                                                    max_anisotropy,
                                                    lod_min_clamp,
                                                    lod_max_clamp,
                                                    compare_function: MTL_COMPARE_FUNCTION_LESS,
                                                    compare_enabled,
                                                    border_color:
                                                        MTL_SAMPLER_BORDER_COLOR_OPAQUE_WHITE,
                                                    normalized_coordinates,
                                                };
                                                let Ok(state) = shape.checked() else {
                                                    refused += 1;
                                                    continue;
                                                };

                                                // Whatever came out, the flag
                                                // is only ever set on the mode
                                                // that has a conformance.
                                                assert_eq!(
                                                    state.normalized_coordinates(),
                                                    normalized_coordinates
                                                );
                                                assert!(
                                                    !state.unnormalized_conformed()
                                                        || !normalized_coordinates
                                                );

                                                if normalized_coordinates {
                                                    normalized_seen += 1;
                                                    // Untouched: every field
                                                    // is the parse of the
                                                    // ordinal the guest wrote.
                                                    assert_eq!(
                                                        Some(state.min_filter()),
                                                        Filter::parse(min_filter)
                                                    );
                                                    assert_eq!(
                                                        Some(state.mip_filter()),
                                                        MipFilter::parse(mip_filter)
                                                    );
                                                    assert_eq!(
                                                        state.address(),
                                                        [s_address, t_address, r_address]
                                                            .map(|a| AddressMode::parse(a)
                                                                .expect("admitted"))
                                                    );
                                                    assert_eq!(
                                                        state.max_anisotropy(),
                                                        max_anisotropy.max(1)
                                                    );
                                                    assert_eq!(
                                                        state.lod_clamp(),
                                                        (lod_min_clamp, lod_max_clamp)
                                                    );
                                                    assert_eq!(
                                                        state.compare().is_some(),
                                                        compare_enabled
                                                    );
                                                    assert!(!state.unnormalized_conformed());
                                                    continue;
                                                }

                                                unnormalized_seen += 1;
                                                if state.unnormalized_conformed() {
                                                    conformed_seen += 1;
                                                }
                                                // The region both APIs admit
                                                // for a pixel-coordinate
                                                // sampler, in full.
                                                assert_eq!(state.min_filter(), state.mag_filter());
                                                assert_eq!(
                                                    state.mip_filter(),
                                                    MipFilter::NotMipmapped
                                                );
                                                assert_eq!(state.max_anisotropy(), 1);
                                                assert_eq!(state.lod_clamp(), (0.0, 0.0));
                                                assert!(state.compare().is_none());
                                                for mode in state.address() {
                                                    assert!(mode.allows_unnormalized());
                                                }
                                                // And the flag says whether it
                                                // took any of that to get
                                                // there, which is the one
                                                // thing a backend reports.
                                                let untouched = min_filter == mag_filter
                                                    && mip_filter
                                                        == MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED
                                                    && max_anisotropy <= 1
                                                    && lod_min_clamp == 0.0
                                                    && lod_max_clamp == 0.0
                                                    && [s_address, t_address, r_address]
                                                        .iter()
                                                        .all(|&a| {
                                                            AddressMode::parse(a).is_some_and(
                                                                AddressMode::allows_unnormalized,
                                                            )
                                                        });
                                                assert_eq!(
                                                    state.unnormalized_conformed(),
                                                    !untouched,
                                                    "{shape:?}"
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Floors per outcome. One aggregate would be satisfied by a product
        // that refused everything, or that never reached the conforming arm.
        assert!(normalized_seen > 500, "{normalized_seen}");
        assert!(unnormalized_seen > 500, "{unnormalized_seen}");
        assert!(conformed_seen > 500, "{conformed_seen}");
        assert!(refused > 500, "{refused}");
    }
}
