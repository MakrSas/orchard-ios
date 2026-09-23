//! `MTLRenderPipelineColorAttachmentDescriptor`'s blend ordinals and write
//! mask.
//!
//! # A factor that cannot reach a pixel cannot refuse a pipeline
//!
//! The six factor and operation ordinals sit beside a `blendingEnabled` flag,
//! and with it clear none of them affects any fragment. So
//! [`ColorAttachmentShape::checked`] parses them only behind that flag and
//! [`ColorAttachmentState`] holds `Option<Blend>`: a consumer cannot read a
//! blend that is not in effect, and a declaration whose unused ordinal this
//! device does not recognise is not dropped for it.
//!
//! # The write mask is not part of the blend
//!
//! It is parsed and carried whether or not blending is on. An unblended
//! attachment with a mask still leaves its unwritten channels alone, so gating
//! the mask on blending would drop it exactly where the guest is replacing
//! rather than compositing.
//!
//! Its bit order is also not Vulkan's — Metal runs alpha-first from the low
//! end — but that is a spelling this layer does not perform. What it does is
//! refuse a value no four-bit mask holds, so a mask that is not a mask arrives
//! by name rather than quietly writing the wrong channels.
//!
//! # Two sentinels that are not values
//!
//! `MTLBlendFactorUnspecialized = 19`, `MTLBlendOperationUnspecialized = 5`
//! and `MTLColorWriteMaskUnspecialized = 0x10` are Metal 4's "resolve this at
//! specialization time" markers rather than a factor, an operation and a mask.
//! Nothing here performs that resolution, so each is outside its set — a
//! decision, and the doc on each set says so, because a bound and a decision
//! look identical in a table and only a decision can be wrong.
//!
//! # What is deliberately not here
//!
//! Whether the host can run a dual-source blend. The four `Source1` factors
//! need a device feature on one rail and none on the other;
//! [`BlendFactor::is_dual_source`] names them so an executor can ask, and the
//! asking is the executor's.
//!
//! The blend colour, too. Metal sets it on the encoder, so it changes without
//! the pipeline changing; it is not part of what the guest declared here.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// `MTLBlendFactor`, in `MTLRenderPipeline.h` order.
pub const MTL_BLEND_FACTOR_ZERO: u32 = 0;
pub const MTL_BLEND_FACTOR_ONE: u32 = 1;
pub const MTL_BLEND_FACTOR_SOURCE_COLOR: u32 = 2;
pub const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_COLOR: u32 = 3;
pub const MTL_BLEND_FACTOR_SOURCE_ALPHA: u32 = 4;
pub const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA: u32 = 5;
pub const MTL_BLEND_FACTOR_DESTINATION_COLOR: u32 = 6;
pub const MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_COLOR: u32 = 7;
pub const MTL_BLEND_FACTOR_DESTINATION_ALPHA: u32 = 8;
pub const MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA: u32 = 9;
pub const MTL_BLEND_FACTOR_SOURCE_ALPHA_SATURATED: u32 = 10;
pub const MTL_BLEND_FACTOR_BLEND_COLOR: u32 = 11;
pub const MTL_BLEND_FACTOR_ONE_MINUS_BLEND_COLOR: u32 = 12;
pub const MTL_BLEND_FACTOR_BLEND_ALPHA: u32 = 13;
pub const MTL_BLEND_FACTOR_ONE_MINUS_BLEND_ALPHA: u32 = 14;
pub const MTL_BLEND_FACTOR_SOURCE_1_COLOR: u32 = 15;
pub const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_1_COLOR: u32 = 16;
pub const MTL_BLEND_FACTOR_SOURCE_1_ALPHA: u32 = 17;
pub const MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_1_ALPHA: u32 = 18;

/// `MTLBlendOperation`.
pub const MTL_BLEND_OPERATION_ADD: u32 = 0;
pub const MTL_BLEND_OPERATION_SUBTRACT: u32 = 1;
pub const MTL_BLEND_OPERATION_REVERSE_SUBTRACT: u32 = 2;
pub const MTL_BLEND_OPERATION_MIN: u32 = 3;
pub const MTL_BLEND_OPERATION_MAX: u32 = 4;

/// `MTLColorWriteMask`, alpha-first from the low end.
pub const MTL_COLOR_WRITE_MASK_NONE: u32 = 0;
pub const MTL_COLOR_WRITE_MASK_ALPHA: u32 = 1 << 0;
pub const MTL_COLOR_WRITE_MASK_BLUE: u32 = 1 << 1;
pub const MTL_COLOR_WRITE_MASK_GREEN: u32 = 1 << 2;
pub const MTL_COLOR_WRITE_MASK_RED: u32 = 1 << 3;
pub const MTL_COLOR_WRITE_MASK_ALL: u32 = 0xf;

/// One side of one blend equation.
///
/// Nineteen values, of which the last four read the fragment shader's second
/// colour output. They are in the same enumeration because Metal puts them
/// there; what separates them is [`Self::is_dual_source`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationColor,
    OneMinusDestinationColor,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    SourceAlphaSaturated,
    /// `MTLBlendFactorBlendColor` — the encoder's blend colour, which is not
    /// declared here. See the module doc.
    BlendColor,
    OneMinusBlendColor,
    BlendAlpha,
    OneMinusBlendAlpha,
    Source1Color,
    OneMinusSource1Color,
    Source1Alpha,
    OneMinusSource1Alpha,
}

impl BlendFactor {
    pub const ALL: [BlendFactor; 19] = [
        Self::Zero,
        Self::One,
        Self::SourceColor,
        Self::OneMinusSourceColor,
        Self::SourceAlpha,
        Self::OneMinusSourceAlpha,
        Self::DestinationColor,
        Self::OneMinusDestinationColor,
        Self::DestinationAlpha,
        Self::OneMinusDestinationAlpha,
        Self::SourceAlphaSaturated,
        Self::BlendColor,
        Self::OneMinusBlendColor,
        Self::BlendAlpha,
        Self::OneMinusBlendAlpha,
        Self::Source1Color,
        Self::OneMinusSource1Color,
        Self::Source1Alpha,
        Self::OneMinusSource1Alpha,
    ];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_BLEND_FACTOR_ZERO => Self::Zero,
            MTL_BLEND_FACTOR_ONE => Self::One,
            MTL_BLEND_FACTOR_SOURCE_COLOR => Self::SourceColor,
            MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_COLOR => Self::OneMinusSourceColor,
            MTL_BLEND_FACTOR_SOURCE_ALPHA => Self::SourceAlpha,
            MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA => Self::OneMinusSourceAlpha,
            MTL_BLEND_FACTOR_DESTINATION_COLOR => Self::DestinationColor,
            MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_COLOR => Self::OneMinusDestinationColor,
            MTL_BLEND_FACTOR_DESTINATION_ALPHA => Self::DestinationAlpha,
            MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA => Self::OneMinusDestinationAlpha,
            MTL_BLEND_FACTOR_SOURCE_ALPHA_SATURATED => Self::SourceAlphaSaturated,
            MTL_BLEND_FACTOR_BLEND_COLOR => Self::BlendColor,
            MTL_BLEND_FACTOR_ONE_MINUS_BLEND_COLOR => Self::OneMinusBlendColor,
            MTL_BLEND_FACTOR_BLEND_ALPHA => Self::BlendAlpha,
            MTL_BLEND_FACTOR_ONE_MINUS_BLEND_ALPHA => Self::OneMinusBlendAlpha,
            MTL_BLEND_FACTOR_SOURCE_1_COLOR => Self::Source1Color,
            MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_1_COLOR => Self::OneMinusSource1Color,
            MTL_BLEND_FACTOR_SOURCE_1_ALPHA => Self::Source1Alpha,
            MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_1_ALPHA => Self::OneMinusSource1Alpha,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::Zero => MTL_BLEND_FACTOR_ZERO,
            Self::One => MTL_BLEND_FACTOR_ONE,
            Self::SourceColor => MTL_BLEND_FACTOR_SOURCE_COLOR,
            Self::OneMinusSourceColor => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_COLOR,
            Self::SourceAlpha => MTL_BLEND_FACTOR_SOURCE_ALPHA,
            Self::OneMinusSourceAlpha => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
            Self::DestinationColor => MTL_BLEND_FACTOR_DESTINATION_COLOR,
            Self::OneMinusDestinationColor => MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_COLOR,
            Self::DestinationAlpha => MTL_BLEND_FACTOR_DESTINATION_ALPHA,
            Self::OneMinusDestinationAlpha => MTL_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA,
            Self::SourceAlphaSaturated => MTL_BLEND_FACTOR_SOURCE_ALPHA_SATURATED,
            Self::BlendColor => MTL_BLEND_FACTOR_BLEND_COLOR,
            Self::OneMinusBlendColor => MTL_BLEND_FACTOR_ONE_MINUS_BLEND_COLOR,
            Self::BlendAlpha => MTL_BLEND_FACTOR_BLEND_ALPHA,
            Self::OneMinusBlendAlpha => MTL_BLEND_FACTOR_ONE_MINUS_BLEND_ALPHA,
            Self::Source1Color => MTL_BLEND_FACTOR_SOURCE_1_COLOR,
            Self::OneMinusSource1Color => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_1_COLOR,
            Self::Source1Alpha => MTL_BLEND_FACTOR_SOURCE_1_ALPHA,
            Self::OneMinusSource1Alpha => MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_1_ALPHA,
        }
    }

    /// Whether this factor reads the fragment shader's second colour output.
    ///
    /// A capability question on at least one rail, which is why it is named
    /// rather than left for an executor to re-derive from a range.
    #[must_use]
    pub const fn is_dual_source(self) -> bool {
        matches!(
            self,
            Self::Source1Color
                | Self::OneMinusSource1Color
                | Self::Source1Alpha
                | Self::OneMinusSource1Alpha
        )
    }

    /// Whether this factor reads the encoder's blend colour, which is set
    /// outside the pipeline. See the module doc.
    #[must_use]
    pub const fn reads_blend_color(self) -> bool {
        matches!(
            self,
            Self::BlendColor
                | Self::OneMinusBlendColor
                | Self::BlendAlpha
                | Self::OneMinusBlendAlpha
        )
    }
}

/// How the weighted source and destination are combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl BlendOperation {
    pub const ALL: [BlendOperation; 5] = [
        Self::Add,
        Self::Subtract,
        Self::ReverseSubtract,
        Self::Min,
        Self::Max,
    ];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_BLEND_OPERATION_ADD => Self::Add,
            MTL_BLEND_OPERATION_SUBTRACT => Self::Subtract,
            MTL_BLEND_OPERATION_REVERSE_SUBTRACT => Self::ReverseSubtract,
            MTL_BLEND_OPERATION_MIN => Self::Min,
            MTL_BLEND_OPERATION_MAX => Self::Max,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::Add => MTL_BLEND_OPERATION_ADD,
            Self::Subtract => MTL_BLEND_OPERATION_SUBTRACT,
            Self::ReverseSubtract => MTL_BLEND_OPERATION_REVERSE_SUBTRACT,
            Self::Min => MTL_BLEND_OPERATION_MIN,
            Self::Max => MTL_BLEND_OPERATION_MAX,
        }
    }
}

/// Which channels one attachment writes, in Metal's own bit order.
///
/// A newtype rather than a `u32` because the value meaning "write every
/// channel" is `0xf` and the value a derived `Default` would produce is `0`,
/// which means *write nothing*. Here the omission is unwritable: `Default` is
/// all, which is also what an entry carrying no mask means on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColorWriteMask {
    bits: u32,
}

impl Default for ColorWriteMask {
    fn default() -> Self {
        Self::ALL
    }
}

impl ColorWriteMask {
    pub const ALL: Self = Self {
        bits: MTL_COLOR_WRITE_MASK_ALL,
    };
    pub const NONE: Self = Self {
        bits: MTL_COLOR_WRITE_MASK_NONE,
    };

    /// `None` for a value outside `MTLColorWriteMask`'s four bits.
    #[must_use]
    pub const fn new(bits: u32) -> Option<Self> {
        if bits <= MTL_COLOR_WRITE_MASK_ALL {
            Some(Self { bits })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.bits
    }

    #[must_use]
    pub const fn red(self) -> bool {
        self.bits & MTL_COLOR_WRITE_MASK_RED != 0
    }

    #[must_use]
    pub const fn green(self) -> bool {
        self.bits & MTL_COLOR_WRITE_MASK_GREEN != 0
    }

    #[must_use]
    pub const fn blue(self) -> bool {
        self.bits & MTL_COLOR_WRITE_MASK_BLUE != 0
    }

    #[must_use]
    pub const fn alpha(self) -> bool {
        self.bits & MTL_COLOR_WRITE_MASK_ALPHA != 0
    }

    /// Whether this attachment can change any channel at all.
    #[must_use]
    pub const fn writes_nothing(self) -> bool {
        self.bits == MTL_COLOR_WRITE_MASK_NONE
    }
}

/// One colour attachment's blend declaration as the fields arrived.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorAttachmentShape {
    pub blending_enabled: bool,
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub op_rgb: u32,
    pub src_alpha: u32,
    pub dst_alpha: u32,
    pub op_alpha: u32,
    /// Already range-checked by whoever read it, because the mask has its own
    /// refusal at the point the wire value is seen. See [`ColorWriteMask`].
    pub write_mask: ColorWriteMask,
}

/// Why a colour-attachment declaration is not one the guest API admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendRefusal {
    UnknownOrdinal { field: &'static str, ordinal: u32 },
}

impl reims_vgpu_observe::Decline for BlendRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownOrdinal { .. } => "blend_unknown_ordinal",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnknownOrdinal { field, ordinal } => vec![
                ("field", (*field).to_string()),
                ("ordinal", ordinal.to_string()),
            ],
        }
    }
}

/// The two equations, parsed.
///
/// The colour and alpha halves are three fields each and interchangeable in
/// type, which is exactly why they are named here rather than carried as two
/// arrays: a swap between them produces a valid blend that blends the wrong
/// channel set, and no refusal could report it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blend {
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_operation: BlendOperation,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_operation: BlendOperation,
}

impl Blend {
    /// Whether any of the six factors needs the second fragment output.
    #[must_use]
    pub fn is_dual_source(&self) -> bool {
        [
            self.src_color,
            self.dst_color,
            self.src_alpha,
            self.dst_alpha,
        ]
        .iter()
        .any(|f| f.is_dual_source())
    }

    /// Whether any of the four factors reads the encoder's blend colour.
    #[must_use]
    pub fn reads_blend_color(&self) -> bool {
        [
            self.src_color,
            self.dst_color,
            self.src_alpha,
            self.dst_alpha,
        ]
        .iter()
        .any(|f| f.reads_blend_color())
    }
}

/// One colour attachment, parsed.
///
/// Blending is present or it is not; there is no flag to consult and no
/// equation to read behind a clear one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorAttachmentState {
    blend: Option<Blend>,
    write_mask: ColorWriteMask,
}

fn parsed<T>(field: &'static str, ordinal: u32, value: Option<T>) -> Result<T, BlendRefusal> {
    value.ok_or(BlendRefusal::UnknownOrdinal { field, ordinal })
}

impl ColorAttachmentShape {
    /// Parse the declaration.
    ///
    /// The six blend ordinals are parsed only behind `blending_enabled`; the
    /// write mask always. See the module doc for both.
    ///
    /// # Errors
    ///
    /// [`BlendRefusal`] naming the field that failed.
    pub fn checked(self) -> Result<ColorAttachmentState, BlendRefusal> {
        let blend = if self.blending_enabled {
            Some(Blend {
                src_color: parsed("src_rgb", self.src_rgb, BlendFactor::parse(self.src_rgb))?,
                dst_color: parsed("dst_rgb", self.dst_rgb, BlendFactor::parse(self.dst_rgb))?,
                color_operation: parsed("op_rgb", self.op_rgb, BlendOperation::parse(self.op_rgb))?,
                src_alpha: parsed(
                    "src_alpha",
                    self.src_alpha,
                    BlendFactor::parse(self.src_alpha),
                )?,
                dst_alpha: parsed(
                    "dst_alpha",
                    self.dst_alpha,
                    BlendFactor::parse(self.dst_alpha),
                )?,
                alpha_operation: parsed(
                    "op_alpha",
                    self.op_alpha,
                    BlendOperation::parse(self.op_alpha),
                )?,
            })
        } else {
            None
        };
        Ok(ColorAttachmentState {
            blend,
            write_mask: self.write_mask,
        })
    }
}

impl ColorAttachmentState {
    /// An attachment that replaces rather than composites, writing everything.
    pub const OPAQUE: Self = Self {
        blend: None,
        write_mask: ColorWriteMask::ALL,
    };

    #[must_use]
    pub const fn blend(&self) -> Option<Blend> {
        self.blend
    }

    #[must_use]
    pub const fn write_mask(&self) -> ColorWriteMask {
        self.write_mask
    }

    /// Whether this attachment can change any pixel.
    ///
    /// A mask of nothing is the whole answer: with no channel writable, the
    /// blend equation is arithmetic on a value that is then discarded.
    #[must_use]
    pub const fn is_inert(&self) -> bool {
        self.write_mask.writes_nothing()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use reims_vgpu_observe::Decline;

    fn shape() -> ColorAttachmentShape {
        ColorAttachmentShape {
            blending_enabled: true,
            src_rgb: MTL_BLEND_FACTOR_SOURCE_ALPHA,
            dst_rgb: MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
            op_rgb: MTL_BLEND_OPERATION_ADD,
            src_alpha: MTL_BLEND_FACTOR_ONE,
            dst_alpha: MTL_BLEND_FACTOR_ZERO,
            op_alpha: MTL_BLEND_OPERATION_MAX,
            write_mask: ColorWriteMask::ALL,
        }
    }

    #[test]
    fn every_factor_and_operation_round_trips_and_none_collapses() {
        let factors: BTreeSet<u32> = BlendFactor::ALL.iter().map(|f| f.ordinal()).collect();
        assert_eq!(factors.len(), BlendFactor::ALL.len());
        for f in BlendFactor::ALL {
            assert_eq!(BlendFactor::parse(f.ordinal()), Some(f));
        }
        let ops: BTreeSet<u32> = BlendOperation::ALL.iter().map(|o| o.ordinal()).collect();
        assert_eq!(ops.len(), BlendOperation::ALL.len());
        for o in BlendOperation::ALL {
            assert_eq!(BlendOperation::parse(o.ordinal()), Some(o));
        }
        // The ordinals are contiguous from zero, so the sets are exactly the
        // ranges and nothing in between is missing.
        assert_eq!(factors.iter().copied().max(), Some(18));
        assert_eq!(ops.iter().copied().max(), Some(4));
    }

    /// The two sentinels. Each is one past a set this device closes on
    /// purpose, and the assertion is about the decision rather than the bound:
    /// Apple declares both values, and this device turns them down because it
    /// performs no specialization.
    #[test]
    fn the_unspecialized_sentinels_are_not_values() {
        assert_eq!(BlendFactor::parse(19), None);
        assert_eq!(BlendOperation::parse(5), None);
        assert_eq!(ColorWriteMask::new(0x10), None);
        // And the last real value of each is still a value, so the refusals
        // above are not an off-by-one.
        assert_eq!(
            BlendFactor::parse(18),
            Some(BlendFactor::OneMinusSource1Alpha)
        );
        assert_eq!(BlendOperation::parse(4), Some(BlendOperation::Max));
        assert_eq!(ColorWriteMask::new(0xf), Some(ColorWriteMask::ALL));
    }

    #[test]
    fn exactly_the_four_source_one_factors_are_dual_source() {
        let dual: BTreeSet<u32> = BlendFactor::ALL
            .iter()
            .filter(|f| f.is_dual_source())
            .map(|f| f.ordinal())
            .collect();
        assert_eq!(dual, BTreeSet::from([15, 16, 17, 18]));

        let constant: BTreeSet<u32> = BlendFactor::ALL
            .iter()
            .filter(|f| f.reads_blend_color())
            .map(|f| f.ordinal())
            .collect();
        assert_eq!(constant, BTreeSet::from([11, 12, 13, 14]));
    }

    /// The claim the module doc makes: with blending off the six ordinals
    /// cannot refuse the declaration, and with it on the same values do.
    #[test]
    fn ordinals_behind_a_clear_blending_flag_refuse_nothing() {
        let state = ColorAttachmentShape {
            blending_enabled: false,
            src_rgb: 0xdead_beef,
            dst_rgb: 0xdead_beef,
            op_rgb: 0xdead_beef,
            src_alpha: 0xdead_beef,
            dst_alpha: 0xdead_beef,
            op_alpha: 0xdead_beef,
            ..shape()
        }
        .checked()
        .expect("an equation nothing evaluates cannot refuse a declaration");
        assert_eq!(state.blend(), None);
        assert_eq!(state.write_mask(), ColorWriteMask::ALL);

        let refused = ColorAttachmentShape {
            src_rgb: 0xdead_beef,
            ..shape()
        }
        .checked()
        .expect_err("a live equation is parsed");
        assert_eq!(
            refused,
            BlendRefusal::UnknownOrdinal {
                field: "src_rgb",
                ordinal: 0xdead_beef,
            }
        );
        assert_eq!(refused.slug(), "blend_unknown_ordinal");
    }

    /// The colour and alpha halves are three interchangeable fields each, so a
    /// swap between them is a valid blend of the wrong channel set. Each of
    /// the six must therefore be nameable on its own.
    #[test]
    fn a_refusal_names_which_of_the_six_failed() {
        let mut seen: BTreeSet<&'static str> = BTreeSet::new();
        for field in 0..6 {
            let mut bad = shape();
            match field {
                0 => bad.src_rgb = 99,
                1 => bad.dst_rgb = 99,
                2 => bad.op_rgb = 99,
                3 => bad.src_alpha = 99,
                4 => bad.dst_alpha = 99,
                _ => bad.op_alpha = 99,
            }
            let BlendRefusal::UnknownOrdinal { field, ordinal } =
                bad.checked().expect_err("an unknown ordinal");
            assert_eq!(ordinal, 99);
            assert!(seen.insert(field), "{field} named twice");
        }
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn the_two_halves_are_carried_apart() {
        let blend = shape().checked().expect("legal").blend().expect("enabled");
        assert_eq!(blend.src_color, BlendFactor::SourceAlpha);
        assert_eq!(blend.dst_color, BlendFactor::OneMinusSourceAlpha);
        assert_eq!(blend.color_operation, BlendOperation::Add);
        assert_eq!(blend.src_alpha, BlendFactor::One);
        assert_eq!(blend.dst_alpha, BlendFactor::Zero);
        assert_eq!(blend.alpha_operation, BlendOperation::Max);
        assert!(!blend.is_dual_source());
        assert!(!blend.reads_blend_color());

        let dual = ColorAttachmentShape {
            dst_alpha: MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_1_COLOR,
            src_rgb: MTL_BLEND_FACTOR_BLEND_ALPHA,
            ..shape()
        }
        .checked()
        .expect("legal")
        .blend()
        .expect("enabled");
        assert!(dual.is_dual_source());
        assert!(dual.reads_blend_color());
    }

    /// A mask survives blending being off, because an unblended attachment
    /// with a mask still leaves its unwritten channels alone.
    #[test]
    fn the_write_mask_is_carried_whether_or_not_blending_is_on() {
        for enabled in [false, true] {
            let state = ColorAttachmentShape {
                blending_enabled: enabled,
                write_mask: ColorWriteMask::new(MTL_COLOR_WRITE_MASK_ALPHA).expect("in range"),
                ..shape()
            }
            .checked()
            .expect("legal");
            let mask = state.write_mask();
            assert!(mask.alpha());
            assert!(!mask.red() && !mask.green() && !mask.blue());
            assert!(!mask.writes_nothing());
            assert_eq!(state.blend().is_some(), enabled);
        }
    }

    /// The bit order is Metal's, alpha-first. A red-only mask and an
    /// alpha-only mask are not the same value, which is the whole reason the
    /// bits are named rather than passed through.
    #[test]
    fn the_channel_bits_are_metals_own_order() {
        let red = ColorWriteMask::new(MTL_COLOR_WRITE_MASK_RED).expect("in range");
        assert!(red.red());
        assert!(!red.alpha());
        let alpha = ColorWriteMask::new(MTL_COLOR_WRITE_MASK_ALPHA).expect("in range");
        assert!(alpha.alpha());
        assert!(!alpha.red());
        assert_ne!(red, alpha);

        let all = ColorWriteMask::ALL;
        assert!(all.red() && all.green() && all.blue() && all.alpha());
        assert!(ColorWriteMask::NONE.writes_nothing());
    }

    /// An omitted mask writes everything, not nothing. This is the reason the
    /// type is a newtype at all.
    #[test]
    fn an_omitted_mask_is_every_channel() {
        assert_eq!(ColorWriteMask::default(), ColorWriteMask::ALL);
        assert_eq!(
            ColorAttachmentShape::default()
                .checked()
                .expect("legal")
                .write_mask(),
            ColorWriteMask::ALL
        );
    }

    #[test]
    fn an_attachment_that_writes_no_channel_is_inert_whatever_it_blends() {
        assert!(!ColorAttachmentState::OPAQUE.is_inert());
        assert_eq!(ColorAttachmentState::OPAQUE.blend(), None);
        let masked = ColorAttachmentShape {
            write_mask: ColorWriteMask::NONE,
            ..shape()
        }
        .checked()
        .expect("legal");
        assert!(masked.is_inert());
        assert!(masked.blend().is_some());
    }
}
