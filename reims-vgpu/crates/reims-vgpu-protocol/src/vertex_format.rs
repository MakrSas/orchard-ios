//! `MTLVertexFormat`, and the geometry every rail derives from one.
//!
//! # The byte size is derived, not tabulated
//!
//! A `Short3` occupies six bytes *because* it is three 16-bit components. Held
//! as a second table beside the first, those two facts drift: one arm gets
//! fixed and the other keeps the old answer, and the mismatch surfaces as a
//! stride bug in a shader nobody is looking at. So [`VertexFormat::layout`] is
//! the only table, and the component count, the component width and the total
//! footprint all come out of it.
//!
//! # Four formats do not decompose
//!
//! `Int1010102Normalized` and its unsigned twin pack four channels into one
//! 32-bit word at ten, ten, ten and two bits; `FloatRg11B10` and `FloatRgb9E5`
//! pack three into one word with no byte-aligned channel at all. A component
//! width is not a fact about them, so [`Layout`] has a second arm rather than
//! a component count that would have to lie.
//!
//! # BGRA is a channel order, not a format
//!
//! `UChar4NormalizedBgra` is four normalized bytes like `UChar4Normalized`;
//! what differs is which channel each byte is. Carrying that as
//! [`ChannelOrder`] beside an otherwise identical layout keeps its footprint
//! derived from the same arithmetic as its sibling's.
//!
//! # What is deliberately not here
//!
//! Whether a host can fetch a given format from a vertex buffer. Every rail
//! declines some of these and each declines a different set, so the question
//! belongs to the executor that queried a device.

/// `MTLVertexFormat`.
///
/// `0` is `MTLVertexFormatInvalid`, and `43` and `44` are gaps in Apple's own
/// enumeration rather than values this device turns down.
pub const MTL_VERTEX_FORMAT_U_CHAR2: u32 = 1;
pub const MTL_VERTEX_FORMAT_U_CHAR3: u32 = 2;
pub const MTL_VERTEX_FORMAT_U_CHAR4: u32 = 3;
pub const MTL_VERTEX_FORMAT_CHAR2: u32 = 4;
pub const MTL_VERTEX_FORMAT_CHAR3: u32 = 5;
pub const MTL_VERTEX_FORMAT_CHAR4: u32 = 6;
pub const MTL_VERTEX_FORMAT_U_CHAR2_NORMALIZED: u32 = 7;
pub const MTL_VERTEX_FORMAT_U_CHAR3_NORMALIZED: u32 = 8;
pub const MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED: u32 = 9;
pub const MTL_VERTEX_FORMAT_CHAR2_NORMALIZED: u32 = 10;
pub const MTL_VERTEX_FORMAT_CHAR3_NORMALIZED: u32 = 11;
pub const MTL_VERTEX_FORMAT_CHAR4_NORMALIZED: u32 = 12;
pub const MTL_VERTEX_FORMAT_U_SHORT2: u32 = 13;
pub const MTL_VERTEX_FORMAT_U_SHORT3: u32 = 14;
pub const MTL_VERTEX_FORMAT_U_SHORT4: u32 = 15;
pub const MTL_VERTEX_FORMAT_SHORT2: u32 = 16;
pub const MTL_VERTEX_FORMAT_SHORT3: u32 = 17;
pub const MTL_VERTEX_FORMAT_SHORT4: u32 = 18;
pub const MTL_VERTEX_FORMAT_U_SHORT2_NORMALIZED: u32 = 19;
pub const MTL_VERTEX_FORMAT_U_SHORT3_NORMALIZED: u32 = 20;
pub const MTL_VERTEX_FORMAT_U_SHORT4_NORMALIZED: u32 = 21;
pub const MTL_VERTEX_FORMAT_SHORT2_NORMALIZED: u32 = 22;
pub const MTL_VERTEX_FORMAT_SHORT3_NORMALIZED: u32 = 23;
pub const MTL_VERTEX_FORMAT_SHORT4_NORMALIZED: u32 = 24;
pub const MTL_VERTEX_FORMAT_HALF2: u32 = 25;
pub const MTL_VERTEX_FORMAT_HALF3: u32 = 26;
pub const MTL_VERTEX_FORMAT_HALF4: u32 = 27;
pub const MTL_VERTEX_FORMAT_FLOAT: u32 = 28;
pub const MTL_VERTEX_FORMAT_FLOAT2: u32 = 29;
pub const MTL_VERTEX_FORMAT_FLOAT3: u32 = 30;
pub const MTL_VERTEX_FORMAT_FLOAT4: u32 = 31;
pub const MTL_VERTEX_FORMAT_INT: u32 = 32;
pub const MTL_VERTEX_FORMAT_INT2: u32 = 33;
pub const MTL_VERTEX_FORMAT_INT3: u32 = 34;
pub const MTL_VERTEX_FORMAT_INT4: u32 = 35;
pub const MTL_VERTEX_FORMAT_U_INT: u32 = 36;
pub const MTL_VERTEX_FORMAT_U_INT2: u32 = 37;
pub const MTL_VERTEX_FORMAT_U_INT3: u32 = 38;
pub const MTL_VERTEX_FORMAT_U_INT4: u32 = 39;
pub const MTL_VERTEX_FORMAT_INT1010102_NORMALIZED: u32 = 40;
pub const MTL_VERTEX_FORMAT_U_INT1010102_NORMALIZED: u32 = 41;
pub const MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED_BGRA: u32 = 42;
pub const MTL_VERTEX_FORMAT_U_CHAR: u32 = 45;
pub const MTL_VERTEX_FORMAT_CHAR: u32 = 46;
pub const MTL_VERTEX_FORMAT_U_CHAR_NORMALIZED: u32 = 47;
pub const MTL_VERTEX_FORMAT_CHAR_NORMALIZED: u32 = 48;
pub const MTL_VERTEX_FORMAT_U_SHORT: u32 = 49;
pub const MTL_VERTEX_FORMAT_SHORT: u32 = 50;
pub const MTL_VERTEX_FORMAT_U_SHORT_NORMALIZED: u32 = 51;
pub const MTL_VERTEX_FORMAT_SHORT_NORMALIZED: u32 = 52;
pub const MTL_VERTEX_FORMAT_HALF: u32 = 53;
pub const MTL_VERTEX_FORMAT_FLOAT_RG11_B10: u32 = 54;
pub const MTL_VERTEX_FORMAT_FLOAT_RGB9_E5: u32 = 55;

/// The scalar one channel of a non-packed format holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComponentKind {
    Uint8,
    Sint8,
    Unorm8,
    Snorm8,
    Uint16,
    Sint16,
    Unorm16,
    Snorm16,
    Float16,
    Float32,
    Sint32,
    Uint32,
}

impl ComponentKind {
    pub const ALL: [ComponentKind; 12] = [
        Self::Uint8,
        Self::Sint8,
        Self::Unorm8,
        Self::Snorm8,
        Self::Uint16,
        Self::Sint16,
        Self::Unorm16,
        Self::Snorm16,
        Self::Float16,
        Self::Float32,
        Self::Sint32,
        Self::Uint32,
    ];

    /// Bytes one channel occupies.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Uint8 | Self::Sint8 | Self::Unorm8 | Self::Snorm8 => 1,
            Self::Uint16 | Self::Sint16 | Self::Unorm16 | Self::Snorm16 | Self::Float16 => 2,
            Self::Float32 | Self::Sint32 | Self::Uint32 => 4,
        }
    }

    /// Whether the guest declared this channel signed.
    ///
    /// Named because at least one rail binds signed Metal formats to unsigned
    /// host formats deliberately, and a decision like that needs the fact it
    /// is deciding against to be readable rather than inferred from a name.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::Sint8 | Self::Snorm8 | Self::Sint16 | Self::Snorm16 | Self::Sint32
        )
    }

    /// Whether the channel is read back as a real in `[0, 1]` or `[-1, 1]`
    /// rather than as its stored integer.
    #[must_use]
    pub const fn is_normalized(self) -> bool {
        matches!(
            self,
            Self::Unorm8 | Self::Snorm8 | Self::Unorm16 | Self::Snorm16
        )
    }

    /// Whether the channel is an integer in the shader.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Uint8 | Self::Sint8 | Self::Uint16 | Self::Sint16 | Self::Uint32 | Self::Sint32
        )
    }
}

/// A format whose channels are not byte-aligned.
///
/// Four bytes each, and the channel widths are the format's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Packed {
    /// Four signed normalized channels at 10, 10, 10 and 2 bits.
    Snorm1010102,
    Unorm1010102,
    /// Three unsigned floats at 11, 11 and 10 bits.
    FloatRg11B10,
    /// Three unsigned floats sharing one 5-bit exponent.
    FloatRgb9E5,
}

impl Packed {
    pub const ALL: [Packed; 4] = [
        Self::Snorm1010102,
        Self::Unorm1010102,
        Self::FloatRg11B10,
        Self::FloatRgb9E5,
    ];

    /// One 32-bit word, every one of them.
    pub const BYTES: u32 = 4;

    #[must_use]
    pub const fn components(self) -> u32 {
        match self {
            Self::Snorm1010102 | Self::Unorm1010102 => 4,
            Self::FloatRg11B10 | Self::FloatRgb9E5 => 3,
        }
    }
}

/// Which channel each byte of a four-byte format is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChannelOrder {
    Rgba,
    Bgra,
}

/// How one attribute's bytes are arranged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layout {
    /// `count` channels of `kind`, each whole, in `order`.
    Separate {
        count: u32,
        kind: ComponentKind,
        order: ChannelOrder,
    },
    Packed(Packed),
}

impl Layout {
    /// Bytes this attribute occupies in the guest's vertex buffer.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Separate { count, kind, .. } => count * kind.bytes(),
            Self::Packed(_) => Packed::BYTES,
        }
    }

    #[must_use]
    pub const fn components(self) -> u32 {
        match self {
            Self::Separate { count, .. } => count,
            Self::Packed(packed) => packed.components(),
        }
    }
}

/// One vertex attribute's format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VertexFormat {
    UChar2,
    UChar3,
    UChar4,
    Char2,
    Char3,
    Char4,
    UChar2Normalized,
    UChar3Normalized,
    UChar4Normalized,
    Char2Normalized,
    Char3Normalized,
    Char4Normalized,
    UShort2,
    UShort3,
    UShort4,
    Short2,
    Short3,
    Short4,
    UShort2Normalized,
    UShort3Normalized,
    UShort4Normalized,
    Short2Normalized,
    Short3Normalized,
    Short4Normalized,
    Half2,
    Half3,
    Half4,
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Int2,
    Int3,
    Int4,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    Int1010102Normalized,
    UInt1010102Normalized,
    UChar4NormalizedBgra,
    UChar,
    Char,
    UCharNormalized,
    CharNormalized,
    UShort,
    Short,
    UShortNormalized,
    ShortNormalized,
    Half,
    FloatRg11B10,
    FloatRgb9E5,
}

impl VertexFormat {
    pub const ALL: [VertexFormat; 53] = [
        Self::UChar2,
        Self::UChar3,
        Self::UChar4,
        Self::Char2,
        Self::Char3,
        Self::Char4,
        Self::UChar2Normalized,
        Self::UChar3Normalized,
        Self::UChar4Normalized,
        Self::Char2Normalized,
        Self::Char3Normalized,
        Self::Char4Normalized,
        Self::UShort2,
        Self::UShort3,
        Self::UShort4,
        Self::Short2,
        Self::Short3,
        Self::Short4,
        Self::UShort2Normalized,
        Self::UShort3Normalized,
        Self::UShort4Normalized,
        Self::Short2Normalized,
        Self::Short3Normalized,
        Self::Short4Normalized,
        Self::Half2,
        Self::Half3,
        Self::Half4,
        Self::Float,
        Self::Float2,
        Self::Float3,
        Self::Float4,
        Self::Int,
        Self::Int2,
        Self::Int3,
        Self::Int4,
        Self::UInt,
        Self::UInt2,
        Self::UInt3,
        Self::UInt4,
        Self::Int1010102Normalized,
        Self::UInt1010102Normalized,
        Self::UChar4NormalizedBgra,
        Self::UChar,
        Self::Char,
        Self::UCharNormalized,
        Self::CharNormalized,
        Self::UShort,
        Self::Short,
        Self::UShortNormalized,
        Self::ShortNormalized,
        Self::Half,
        Self::FloatRg11B10,
        Self::FloatRgb9E5,
    ];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_VERTEX_FORMAT_U_CHAR2 => Self::UChar2,
            MTL_VERTEX_FORMAT_U_CHAR3 => Self::UChar3,
            MTL_VERTEX_FORMAT_U_CHAR4 => Self::UChar4,
            MTL_VERTEX_FORMAT_CHAR2 => Self::Char2,
            MTL_VERTEX_FORMAT_CHAR3 => Self::Char3,
            MTL_VERTEX_FORMAT_CHAR4 => Self::Char4,
            MTL_VERTEX_FORMAT_U_CHAR2_NORMALIZED => Self::UChar2Normalized,
            MTL_VERTEX_FORMAT_U_CHAR3_NORMALIZED => Self::UChar3Normalized,
            MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED => Self::UChar4Normalized,
            MTL_VERTEX_FORMAT_CHAR2_NORMALIZED => Self::Char2Normalized,
            MTL_VERTEX_FORMAT_CHAR3_NORMALIZED => Self::Char3Normalized,
            MTL_VERTEX_FORMAT_CHAR4_NORMALIZED => Self::Char4Normalized,
            MTL_VERTEX_FORMAT_U_SHORT2 => Self::UShort2,
            MTL_VERTEX_FORMAT_U_SHORT3 => Self::UShort3,
            MTL_VERTEX_FORMAT_U_SHORT4 => Self::UShort4,
            MTL_VERTEX_FORMAT_SHORT2 => Self::Short2,
            MTL_VERTEX_FORMAT_SHORT3 => Self::Short3,
            MTL_VERTEX_FORMAT_SHORT4 => Self::Short4,
            MTL_VERTEX_FORMAT_U_SHORT2_NORMALIZED => Self::UShort2Normalized,
            MTL_VERTEX_FORMAT_U_SHORT3_NORMALIZED => Self::UShort3Normalized,
            MTL_VERTEX_FORMAT_U_SHORT4_NORMALIZED => Self::UShort4Normalized,
            MTL_VERTEX_FORMAT_SHORT2_NORMALIZED => Self::Short2Normalized,
            MTL_VERTEX_FORMAT_SHORT3_NORMALIZED => Self::Short3Normalized,
            MTL_VERTEX_FORMAT_SHORT4_NORMALIZED => Self::Short4Normalized,
            MTL_VERTEX_FORMAT_HALF2 => Self::Half2,
            MTL_VERTEX_FORMAT_HALF3 => Self::Half3,
            MTL_VERTEX_FORMAT_HALF4 => Self::Half4,
            MTL_VERTEX_FORMAT_FLOAT => Self::Float,
            MTL_VERTEX_FORMAT_FLOAT2 => Self::Float2,
            MTL_VERTEX_FORMAT_FLOAT3 => Self::Float3,
            MTL_VERTEX_FORMAT_FLOAT4 => Self::Float4,
            MTL_VERTEX_FORMAT_INT => Self::Int,
            MTL_VERTEX_FORMAT_INT2 => Self::Int2,
            MTL_VERTEX_FORMAT_INT3 => Self::Int3,
            MTL_VERTEX_FORMAT_INT4 => Self::Int4,
            MTL_VERTEX_FORMAT_U_INT => Self::UInt,
            MTL_VERTEX_FORMAT_U_INT2 => Self::UInt2,
            MTL_VERTEX_FORMAT_U_INT3 => Self::UInt3,
            MTL_VERTEX_FORMAT_U_INT4 => Self::UInt4,
            MTL_VERTEX_FORMAT_INT1010102_NORMALIZED => Self::Int1010102Normalized,
            MTL_VERTEX_FORMAT_U_INT1010102_NORMALIZED => Self::UInt1010102Normalized,
            MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED_BGRA => Self::UChar4NormalizedBgra,
            MTL_VERTEX_FORMAT_U_CHAR => Self::UChar,
            MTL_VERTEX_FORMAT_CHAR => Self::Char,
            MTL_VERTEX_FORMAT_U_CHAR_NORMALIZED => Self::UCharNormalized,
            MTL_VERTEX_FORMAT_CHAR_NORMALIZED => Self::CharNormalized,
            MTL_VERTEX_FORMAT_U_SHORT => Self::UShort,
            MTL_VERTEX_FORMAT_SHORT => Self::Short,
            MTL_VERTEX_FORMAT_U_SHORT_NORMALIZED => Self::UShortNormalized,
            MTL_VERTEX_FORMAT_SHORT_NORMALIZED => Self::ShortNormalized,
            MTL_VERTEX_FORMAT_HALF => Self::Half,
            MTL_VERTEX_FORMAT_FLOAT_RG11_B10 => Self::FloatRg11B10,
            MTL_VERTEX_FORMAT_FLOAT_RGB9_E5 => Self::FloatRgb9E5,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::UChar2 => MTL_VERTEX_FORMAT_U_CHAR2,
            Self::UChar3 => MTL_VERTEX_FORMAT_U_CHAR3,
            Self::UChar4 => MTL_VERTEX_FORMAT_U_CHAR4,
            Self::Char2 => MTL_VERTEX_FORMAT_CHAR2,
            Self::Char3 => MTL_VERTEX_FORMAT_CHAR3,
            Self::Char4 => MTL_VERTEX_FORMAT_CHAR4,
            Self::UChar2Normalized => MTL_VERTEX_FORMAT_U_CHAR2_NORMALIZED,
            Self::UChar3Normalized => MTL_VERTEX_FORMAT_U_CHAR3_NORMALIZED,
            Self::UChar4Normalized => MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED,
            Self::Char2Normalized => MTL_VERTEX_FORMAT_CHAR2_NORMALIZED,
            Self::Char3Normalized => MTL_VERTEX_FORMAT_CHAR3_NORMALIZED,
            Self::Char4Normalized => MTL_VERTEX_FORMAT_CHAR4_NORMALIZED,
            Self::UShort2 => MTL_VERTEX_FORMAT_U_SHORT2,
            Self::UShort3 => MTL_VERTEX_FORMAT_U_SHORT3,
            Self::UShort4 => MTL_VERTEX_FORMAT_U_SHORT4,
            Self::Short2 => MTL_VERTEX_FORMAT_SHORT2,
            Self::Short3 => MTL_VERTEX_FORMAT_SHORT3,
            Self::Short4 => MTL_VERTEX_FORMAT_SHORT4,
            Self::UShort2Normalized => MTL_VERTEX_FORMAT_U_SHORT2_NORMALIZED,
            Self::UShort3Normalized => MTL_VERTEX_FORMAT_U_SHORT3_NORMALIZED,
            Self::UShort4Normalized => MTL_VERTEX_FORMAT_U_SHORT4_NORMALIZED,
            Self::Short2Normalized => MTL_VERTEX_FORMAT_SHORT2_NORMALIZED,
            Self::Short3Normalized => MTL_VERTEX_FORMAT_SHORT3_NORMALIZED,
            Self::Short4Normalized => MTL_VERTEX_FORMAT_SHORT4_NORMALIZED,
            Self::Half2 => MTL_VERTEX_FORMAT_HALF2,
            Self::Half3 => MTL_VERTEX_FORMAT_HALF3,
            Self::Half4 => MTL_VERTEX_FORMAT_HALF4,
            Self::Float => MTL_VERTEX_FORMAT_FLOAT,
            Self::Float2 => MTL_VERTEX_FORMAT_FLOAT2,
            Self::Float3 => MTL_VERTEX_FORMAT_FLOAT3,
            Self::Float4 => MTL_VERTEX_FORMAT_FLOAT4,
            Self::Int => MTL_VERTEX_FORMAT_INT,
            Self::Int2 => MTL_VERTEX_FORMAT_INT2,
            Self::Int3 => MTL_VERTEX_FORMAT_INT3,
            Self::Int4 => MTL_VERTEX_FORMAT_INT4,
            Self::UInt => MTL_VERTEX_FORMAT_U_INT,
            Self::UInt2 => MTL_VERTEX_FORMAT_U_INT2,
            Self::UInt3 => MTL_VERTEX_FORMAT_U_INT3,
            Self::UInt4 => MTL_VERTEX_FORMAT_U_INT4,
            Self::Int1010102Normalized => MTL_VERTEX_FORMAT_INT1010102_NORMALIZED,
            Self::UInt1010102Normalized => MTL_VERTEX_FORMAT_U_INT1010102_NORMALIZED,
            Self::UChar4NormalizedBgra => MTL_VERTEX_FORMAT_U_CHAR4_NORMALIZED_BGRA,
            Self::UChar => MTL_VERTEX_FORMAT_U_CHAR,
            Self::Char => MTL_VERTEX_FORMAT_CHAR,
            Self::UCharNormalized => MTL_VERTEX_FORMAT_U_CHAR_NORMALIZED,
            Self::CharNormalized => MTL_VERTEX_FORMAT_CHAR_NORMALIZED,
            Self::UShort => MTL_VERTEX_FORMAT_U_SHORT,
            Self::Short => MTL_VERTEX_FORMAT_SHORT,
            Self::UShortNormalized => MTL_VERTEX_FORMAT_U_SHORT_NORMALIZED,
            Self::ShortNormalized => MTL_VERTEX_FORMAT_SHORT_NORMALIZED,
            Self::Half => MTL_VERTEX_FORMAT_HALF,
            Self::FloatRg11B10 => MTL_VERTEX_FORMAT_FLOAT_RG11_B10,
            Self::FloatRgb9E5 => MTL_VERTEX_FORMAT_FLOAT_RGB9_E5,
        }
    }

    /// The one table. Everything else here is arithmetic on it.
    #[must_use]
    pub const fn layout(self) -> Layout {
        match self {
            Self::UChar2 => Layout::Separate {
                count: 2,
                kind: ComponentKind::Uint8,
                order: ChannelOrder::Rgba,
            },
            Self::UChar3 => Layout::Separate {
                count: 3,
                kind: ComponentKind::Uint8,
                order: ChannelOrder::Rgba,
            },
            Self::UChar4 => Layout::Separate {
                count: 4,
                kind: ComponentKind::Uint8,
                order: ChannelOrder::Rgba,
            },
            Self::Char2 => Layout::Separate {
                count: 2,
                kind: ComponentKind::Sint8,
                order: ChannelOrder::Rgba,
            },
            Self::Char3 => Layout::Separate {
                count: 3,
                kind: ComponentKind::Sint8,
                order: ChannelOrder::Rgba,
            },
            Self::Char4 => Layout::Separate {
                count: 4,
                kind: ComponentKind::Sint8,
                order: ChannelOrder::Rgba,
            },
            Self::UChar2Normalized => Layout::Separate {
                count: 2,
                kind: ComponentKind::Unorm8,
                order: ChannelOrder::Rgba,
            },
            Self::UChar3Normalized => Layout::Separate {
                count: 3,
                kind: ComponentKind::Unorm8,
                order: ChannelOrder::Rgba,
            },
            Self::UChar4Normalized => Layout::Separate {
                count: 4,
                kind: ComponentKind::Unorm8,
                order: ChannelOrder::Rgba,
            },
            Self::Char2Normalized => Layout::Separate {
                count: 2,
                kind: ComponentKind::Snorm8,
                order: ChannelOrder::Rgba,
            },
            Self::Char3Normalized => Layout::Separate {
                count: 3,
                kind: ComponentKind::Snorm8,
                order: ChannelOrder::Rgba,
            },
            Self::Char4Normalized => Layout::Separate {
                count: 4,
                kind: ComponentKind::Snorm8,
                order: ChannelOrder::Rgba,
            },
            Self::UShort2 => Layout::Separate {
                count: 2,
                kind: ComponentKind::Uint16,
                order: ChannelOrder::Rgba,
            },
            Self::UShort3 => Layout::Separate {
                count: 3,
                kind: ComponentKind::Uint16,
                order: ChannelOrder::Rgba,
            },
            Self::UShort4 => Layout::Separate {
                count: 4,
                kind: ComponentKind::Uint16,
                order: ChannelOrder::Rgba,
            },
            Self::Short2 => Layout::Separate {
                count: 2,
                kind: ComponentKind::Sint16,
                order: ChannelOrder::Rgba,
            },
            Self::Short3 => Layout::Separate {
                count: 3,
                kind: ComponentKind::Sint16,
                order: ChannelOrder::Rgba,
            },
            Self::Short4 => Layout::Separate {
                count: 4,
                kind: ComponentKind::Sint16,
                order: ChannelOrder::Rgba,
            },
            Self::UShort2Normalized => Layout::Separate {
                count: 2,
                kind: ComponentKind::Unorm16,
                order: ChannelOrder::Rgba,
            },
            Self::UShort3Normalized => Layout::Separate {
                count: 3,
                kind: ComponentKind::Unorm16,
                order: ChannelOrder::Rgba,
            },
            Self::UShort4Normalized => Layout::Separate {
                count: 4,
                kind: ComponentKind::Unorm16,
                order: ChannelOrder::Rgba,
            },
            Self::Short2Normalized => Layout::Separate {
                count: 2,
                kind: ComponentKind::Snorm16,
                order: ChannelOrder::Rgba,
            },
            Self::Short3Normalized => Layout::Separate {
                count: 3,
                kind: ComponentKind::Snorm16,
                order: ChannelOrder::Rgba,
            },
            Self::Short4Normalized => Layout::Separate {
                count: 4,
                kind: ComponentKind::Snorm16,
                order: ChannelOrder::Rgba,
            },
            Self::Half2 => Layout::Separate {
                count: 2,
                kind: ComponentKind::Float16,
                order: ChannelOrder::Rgba,
            },
            Self::Half3 => Layout::Separate {
                count: 3,
                kind: ComponentKind::Float16,
                order: ChannelOrder::Rgba,
            },
            Self::Half4 => Layout::Separate {
                count: 4,
                kind: ComponentKind::Float16,
                order: ChannelOrder::Rgba,
            },
            Self::Float => Layout::Separate {
                count: 1,
                kind: ComponentKind::Float32,
                order: ChannelOrder::Rgba,
            },
            Self::Float2 => Layout::Separate {
                count: 2,
                kind: ComponentKind::Float32,
                order: ChannelOrder::Rgba,
            },
            Self::Float3 => Layout::Separate {
                count: 3,
                kind: ComponentKind::Float32,
                order: ChannelOrder::Rgba,
            },
            Self::Float4 => Layout::Separate {
                count: 4,
                kind: ComponentKind::Float32,
                order: ChannelOrder::Rgba,
            },
            Self::Int => Layout::Separate {
                count: 1,
                kind: ComponentKind::Sint32,
                order: ChannelOrder::Rgba,
            },
            Self::Int2 => Layout::Separate {
                count: 2,
                kind: ComponentKind::Sint32,
                order: ChannelOrder::Rgba,
            },
            Self::Int3 => Layout::Separate {
                count: 3,
                kind: ComponentKind::Sint32,
                order: ChannelOrder::Rgba,
            },
            Self::Int4 => Layout::Separate {
                count: 4,
                kind: ComponentKind::Sint32,
                order: ChannelOrder::Rgba,
            },
            Self::UInt => Layout::Separate {
                count: 1,
                kind: ComponentKind::Uint32,
                order: ChannelOrder::Rgba,
            },
            Self::UInt2 => Layout::Separate {
                count: 2,
                kind: ComponentKind::Uint32,
                order: ChannelOrder::Rgba,
            },
            Self::UInt3 => Layout::Separate {
                count: 3,
                kind: ComponentKind::Uint32,
                order: ChannelOrder::Rgba,
            },
            Self::UInt4 => Layout::Separate {
                count: 4,
                kind: ComponentKind::Uint32,
                order: ChannelOrder::Rgba,
            },
            Self::Int1010102Normalized => Layout::Packed(Packed::Snorm1010102),
            Self::UInt1010102Normalized => Layout::Packed(Packed::Unorm1010102),
            Self::UChar4NormalizedBgra => Layout::Separate {
                count: 4,
                kind: ComponentKind::Unorm8,
                order: ChannelOrder::Bgra,
            },
            Self::UChar => Layout::Separate {
                count: 1,
                kind: ComponentKind::Uint8,
                order: ChannelOrder::Rgba,
            },
            Self::Char => Layout::Separate {
                count: 1,
                kind: ComponentKind::Sint8,
                order: ChannelOrder::Rgba,
            },
            Self::UCharNormalized => Layout::Separate {
                count: 1,
                kind: ComponentKind::Unorm8,
                order: ChannelOrder::Rgba,
            },
            Self::CharNormalized => Layout::Separate {
                count: 1,
                kind: ComponentKind::Snorm8,
                order: ChannelOrder::Rgba,
            },
            Self::UShort => Layout::Separate {
                count: 1,
                kind: ComponentKind::Uint16,
                order: ChannelOrder::Rgba,
            },
            Self::Short => Layout::Separate {
                count: 1,
                kind: ComponentKind::Sint16,
                order: ChannelOrder::Rgba,
            },
            Self::UShortNormalized => Layout::Separate {
                count: 1,
                kind: ComponentKind::Unorm16,
                order: ChannelOrder::Rgba,
            },
            Self::ShortNormalized => Layout::Separate {
                count: 1,
                kind: ComponentKind::Snorm16,
                order: ChannelOrder::Rgba,
            },
            Self::Half => Layout::Separate {
                count: 1,
                kind: ComponentKind::Float16,
                order: ChannelOrder::Rgba,
            },
            Self::FloatRg11B10 => Layout::Packed(Packed::FloatRg11B10),
            Self::FloatRgb9E5 => Layout::Packed(Packed::FloatRgb9E5),
        }
    }

    /// Bytes this attribute occupies in the guest's vertex buffer.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        self.layout().bytes()
    }

    /// How many channels the shader sees.
    ///
    /// The fact that decides whether a widening substitute exists when a host
    /// declines the format: a three-channel attribute can be fetched as the
    /// four-channel one beside it and the fourth discarded, and a
    /// four-channel one has nothing wider to widen to.
    #[must_use]
    pub const fn components(self) -> u32 {
        self.layout().components()
    }

    /// The format with the same channel kind and order and one more channel,
    /// when there is one.
    ///
    /// A geometric relation and not a policy: components `0..count` sit at
    /// identical byte offsets in both, because these layouts are
    /// component-packed. Whether substituting the wider one is *safe* depends
    /// on what the shader declares and on whether the wider read still fits
    /// the vertex stride, and both of those belong to the executor.
    ///
    /// `None` for a four-channel format, which has nothing wider, and for
    /// every packed one, whose channels are not whole.
    #[must_use]
    pub const fn widened(self) -> Option<Self> {
        let Layout::Separate { count, kind, order } = self.layout() else {
            return None;
        };
        if count >= 4 {
            return None;
        }
        let mut i = 0;
        while i < Self::ALL.len() {
            let candidate = Self::ALL[i];
            if let Layout::Separate {
                count: wider,
                kind: k,
                order: o,
            } = candidate.layout()
            {
                if wider == count + 1 && kind_eq(k, kind) && order_eq(o, order) {
                    return Some(candidate);
                }
            }
            i += 1;
        }
        None
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::UChar2 => "uchar2",
            Self::UChar3 => "uchar3",
            Self::UChar4 => "uchar4",
            Self::Char2 => "char2",
            Self::Char3 => "char3",
            Self::Char4 => "char4",
            Self::UChar2Normalized => "uchar2_normalized",
            Self::UChar3Normalized => "uchar3_normalized",
            Self::UChar4Normalized => "uchar4_normalized",
            Self::Char2Normalized => "char2_normalized",
            Self::Char3Normalized => "char3_normalized",
            Self::Char4Normalized => "char4_normalized",
            Self::UShort2 => "ushort2",
            Self::UShort3 => "ushort3",
            Self::UShort4 => "ushort4",
            Self::Short2 => "short2",
            Self::Short3 => "short3",
            Self::Short4 => "short4",
            Self::UShort2Normalized => "ushort2_normalized",
            Self::UShort3Normalized => "ushort3_normalized",
            Self::UShort4Normalized => "ushort4_normalized",
            Self::Short2Normalized => "short2_normalized",
            Self::Short3Normalized => "short3_normalized",
            Self::Short4Normalized => "short4_normalized",
            Self::Half2 => "half2",
            Self::Half3 => "half3",
            Self::Half4 => "half4",
            Self::Float => "float",
            Self::Float2 => "float2",
            Self::Float3 => "float3",
            Self::Float4 => "float4",
            Self::Int => "int",
            Self::Int2 => "int2",
            Self::Int3 => "int3",
            Self::Int4 => "int4",
            Self::UInt => "uint",
            Self::UInt2 => "uint2",
            Self::UInt3 => "uint3",
            Self::UInt4 => "uint4",
            Self::Int1010102Normalized => "int1010102_normalized",
            Self::UInt1010102Normalized => "uint1010102_normalized",
            Self::UChar4NormalizedBgra => "uchar4_normalized_bgra",
            Self::UChar => "uchar",
            Self::Char => "char",
            Self::UCharNormalized => "uchar_normalized",
            Self::CharNormalized => "char_normalized",
            Self::UShort => "ushort",
            Self::Short => "short",
            Self::UShortNormalized => "ushort_normalized",
            Self::ShortNormalized => "short_normalized",
            Self::Half => "half",
            Self::FloatRg11B10 => "float_rg11b10",
            Self::FloatRgb9E5 => "float_rgb9e5",
        }
    }
}

/// `PartialEq` is not `const`, and this is a `const fn`.
const fn kind_eq(a: ComponentKind, b: ComponentKind) -> bool {
    a as u32 == b as u32
}

const fn order_eq(a: ChannelOrder, b: ChannelOrder) -> bool {
    a as u32 == b as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::{BTreeMap, BTreeSet};

    #[test]
    fn every_format_round_trips_and_no_two_share_an_ordinal_or_a_name() {
        let ordinals: BTreeSet<u32> = VertexFormat::ALL.iter().map(|f| f.ordinal()).collect();
        assert_eq!(ordinals.len(), VertexFormat::ALL.len());
        for f in VertexFormat::ALL {
            assert_eq!(VertexFormat::parse(f.ordinal()), Some(f));
        }
        let names: BTreeSet<&str> = VertexFormat::ALL.iter().map(|f| f.name()).collect();
        assert_eq!(names.len(), VertexFormat::ALL.len());
    }

    /// `0` is Apple's own `Invalid` and `43`/`44` are gaps in Apple's
    /// enumeration. Neither is this device declining anything, which is why
    /// the set runs `1..=42` and `45..=55` with nothing in between.
    #[test]
    fn the_set_is_apples_two_runs_and_the_gaps_are_apples() {
        let ordinals: BTreeSet<u32> = VertexFormat::ALL.iter().map(|f| f.ordinal()).collect();
        let expected: BTreeSet<u32> = (1..=42).chain(45..=55).collect();
        assert_eq!(ordinals, expected);
        for absent in [0, 43, 44, 56] {
            assert_eq!(VertexFormat::parse(absent), None);
        }
    }

    /// The byte size is arithmetic on the layout, so this states it
    /// independently: a format's name says how many channels of what width,
    /// and the footprint must be their product.
    #[test]
    fn the_footprint_is_the_channels_times_their_width() {
        // Spot the four widths at each channel count, so no arm of `bytes`
        // goes unvisited.
        for (format, bytes, components) in [
            (VertexFormat::UChar, 1, 1),
            (VertexFormat::Char4, 4, 4),
            (VertexFormat::Short3, 6, 3),
            (VertexFormat::Half4, 8, 4),
            (VertexFormat::Float3, 12, 3),
            (VertexFormat::UInt4, 16, 4),
            (VertexFormat::UChar4NormalizedBgra, 4, 4),
            (VertexFormat::Int1010102Normalized, 4, 4),
            (VertexFormat::FloatRg11B10, 4, 3),
            (VertexFormat::FloatRgb9E5, 4, 3),
        ] {
            assert_eq!(format.bytes(), bytes, "{}", format.name());
            assert_eq!(format.components(), components, "{}", format.name());
        }

        // And over the whole set: a separate layout's footprint is exactly its
        // product, a packed one's is exactly a word, and nothing is zero.
        for format in VertexFormat::ALL {
            match format.layout() {
                Layout::Separate { count, kind, .. } => {
                    assert_eq!(format.bytes(), count * kind.bytes());
                    assert!((1..=4).contains(&count));
                }
                Layout::Packed(packed) => {
                    assert_eq!(format.bytes(), Packed::BYTES);
                    assert_eq!(format.components(), packed.components());
                }
            }
            assert!(format.bytes() > 0);
            assert!((1..=4).contains(&format.components()));
        }
    }

    /// A widening substitute exists only where something wider shares the
    /// channel kind. This is the fact an executor reads when a host declines a
    /// three-channel format, so the grouping has to be real.
    #[test]
    fn every_three_channel_format_has_a_four_channel_sibling() {
        let mut by_kind: BTreeMap<(ComponentKind, ChannelOrder), BTreeSet<u32>> = BTreeMap::new();
        for format in VertexFormat::ALL {
            if let Layout::Separate { count, kind, order } = format.layout() {
                by_kind.entry((kind, order)).or_default().insert(count);
            }
        }
        let mut threes = 0;
        for (key, counts) in &by_kind {
            if counts.contains(&3) {
                threes += 1;
                assert!(counts.contains(&4), "{key:?} has a three and no four");
            }
        }
        // Non-vacuity: there really are three-channel families.
        assert!(
            threes >= 8,
            "only {threes} families had a three-channel arm"
        );
    }

    #[test]
    fn a_component_kind_is_one_of_signed_normalized_and_integer_in_the_ways_it_can_be() {
        for kind in ComponentKind::ALL {
            assert!([1, 2, 4].contains(&kind.bytes()));
            // Normalized and integer are exclusive: a normalized channel is
            // read as a real, which is the whole difference.
            assert!(!(kind.is_normalized() && kind.is_integer()));
        }
        assert!(ComponentKind::Snorm16.is_signed());
        assert!(ComponentKind::Snorm16.is_normalized());
        assert!(!ComponentKind::Snorm16.is_integer());
        assert!(ComponentKind::Sint32.is_signed());
        assert!(ComponentKind::Sint32.is_integer());
        assert!(!ComponentKind::Uint8.is_signed());
        // Floats are neither normalized nor integer, and float16 is not
        // "signed" in the sense the flag means — it names an integer or fixed
        // point channel's sign, not whether a real can be negative.
        assert!(!ComponentKind::Float32.is_integer());
        assert!(!ComponentKind::Float32.is_normalized());
        assert!(!ComponentKind::Float32.is_signed());
        assert!(!ComponentKind::Float16.is_signed());
    }

    /// BGRA differs from RGBA in channel order and in nothing else — same
    /// count, same kind, same footprint.
    #[test]
    fn bgra_is_its_sibling_with_the_bytes_relabelled() {
        let bgra = VertexFormat::UChar4NormalizedBgra.layout();
        let rgba = VertexFormat::UChar4Normalized.layout();
        assert_ne!(bgra, rgba);
        assert_eq!(bgra.bytes(), rgba.bytes());
        assert_eq!(bgra.components(), rgba.components());
        let (Layout::Separate { order: b, .. }, Layout::Separate { order: r, .. }) = (bgra, rgba)
        else {
            panic!("both are separate layouts");
        };
        assert_eq!(b, ChannelOrder::Bgra);
        assert_eq!(r, ChannelOrder::Rgba);
        // And it is the only one, so a rail need not scan for others.
        let reordered: BTreeSet<u32> = VertexFormat::ALL
            .iter()
            .filter(|f| {
                matches!(
                    f.layout(),
                    Layout::Separate {
                        order: ChannelOrder::Bgra,
                        ..
                    }
                )
            })
            .map(|f| f.ordinal())
            .collect();
        assert_eq!(reordered, BTreeSet::from([42]));
    }

    /// The widening relation, and the two facts that make it usable: the
    /// wider format's leading channels sit at the same offsets, and it is one
    /// step wider rather than any wider format at all.
    #[test]
    fn widening_climbs_one_channel_and_stops_at_four() {
        assert_eq!(VertexFormat::Short3.widened(), Some(VertexFormat::Short4));
        assert_eq!(
            VertexFormat::UChar2Normalized.widened(),
            Some(VertexFormat::UChar3Normalized)
        );
        assert_eq!(VertexFormat::Short4.widened(), None);
        assert_eq!(VertexFormat::FloatRgb9E5.widened(), None);
        assert_eq!(VertexFormat::Int1010102Normalized.widened(), None);

        let mut widened = 0;
        for format in VertexFormat::ALL {
            let Layout::Separate { count, kind, order } = format.layout() else {
                assert_eq!(format.widened(), None, "{} is packed", format.name());
                continue;
            };
            match format.widened() {
                Some(wider) => {
                    widened += 1;
                    assert!(count < 4);
                    let Layout::Separate {
                        count: c,
                        kind: k,
                        order: o,
                    } = wider.layout()
                    else {
                        panic!("a widened format is still separate");
                    };
                    // One step, same channels, and therefore exactly one more
                    // channel's worth of bytes.
                    assert_eq!(c, count + 1);
                    assert_eq!(k, kind);
                    assert_eq!(o, order);
                    assert_eq!(wider.bytes(), format.bytes() + kind.bytes());
                }
                None => assert_eq!(count, 4, "{} has no wider sibling", format.name()),
            }
        }
        // Non-vacuity: most of this set widens.
        assert!(widened >= 30, "only {widened} formats widened");

        // The BGRA format is four channels, so it never widens and never
        // becomes the substitute for an RGBA one.
        assert_eq!(VertexFormat::UChar4NormalizedBgra.widened(), None);
        assert_eq!(
            VertexFormat::UChar3Normalized.widened(),
            Some(VertexFormat::UChar4Normalized)
        );
    }

    #[test]
    fn every_packed_format_is_one_word_and_they_are_four() {
        let packed: BTreeSet<u32> = VertexFormat::ALL
            .iter()
            .filter(|f| matches!(f.layout(), Layout::Packed(_)))
            .map(|f| f.ordinal())
            .collect();
        assert_eq!(packed, BTreeSet::from([40, 41, 54, 55]));
        for p in Packed::ALL {
            assert!((3..=4).contains(&p.components()));
        }
        assert_eq!(Packed::BYTES, 4);
    }
}
