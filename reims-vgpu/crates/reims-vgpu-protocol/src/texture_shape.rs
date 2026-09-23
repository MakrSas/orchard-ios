//! What a texture declaration is: its `MTLTextureType`, the dimensions that
//! type actually uses, and the pairs the guest API does not admit.
//!
//! # Why the shape is one type and not nine fields
//!
//! A serialized texture descriptor carries a type ordinal, a width, a height,
//! a depth, a mip count, a sample count and an array length, and only some
//! combinations of those seven mean anything. A 3D texture has no array
//! length. A cube's array length counts *cubes*, and its slice count is six
//! times that. A multisample texture has exactly one mip level. A 1D texture
//! has no height. Every one of those is a rule a reader can forget, and the
//! cost of forgetting one is not a refusal — it is an allocation of the wrong
//! size, or a view over slices that do not exist, and a wrong frame a long way
//! downstream.
//!
//! So the fields are read once into [`TextureShape`], checked once by
//! [`TextureShape::checked`], and carried afterwards as [`Texture`] — which
//! answers "how many layers", "what extent", "how many samples" from the type
//! rather than from whichever field the caller reached for. A backend that has
//! a `Texture` cannot ask a question whose answer depends on a rule it did not
//! apply.
//!
//! # The six faces are a constant, not a literal
//!
//! A cube is six 2D slices and a cube array is six per element. That six
//! appears in the layer count, in every attachment view a render target
//! expands into, and in the byte arithmetic of a cube upload. It is
//! [`CUBE_FACES`] in all of them, because three independently written sixes
//! are three chances for one of them to be a five.
//!
//! # What this module does not decide
//!
//! Whether the *host* can allocate the shape. Extent limits, mip limits, layer
//! limits and sample-count support are properties of a physical device and
//! belong to the executor that queried one. This module answers only whether
//! the declaration is a texture the guest API itself admits, which is the same
//! answer on every host and therefore belongs here.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::extent::Extent3;

/// The faces of a cube texture.
///
/// One cube is six 2D slices, and a cube array of `n` is `6 * n` of them. The
/// guest API fixes this; it is not a host property and not a policy.
pub const CUBE_FACES: u32 = 6;

/// `MTLTextureType`, as the closed set of ordinals the wire carries.
///
/// Ordinals are `MTLTextureType.h`'s and are not contiguous by accident:
/// `D2MultisampleArray` was added after `D3` and therefore sits above it.
/// Parsing goes through [`Self::from_ordinal`] so that an ordinal outside the
/// set is a refusal with the value in it, rather than a `match` arm somewhere
/// silently falling through to 2D.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureKind {
    D1,
    D1Array,
    D2,
    D2Array,
    D2Multisample,
    Cube,
    CubeArray,
    D3,
    D2MultisampleArray,
}

/// How many of the extent's three dimensions a texture type uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dimensions {
    One,
    Two,
    Three,
}

impl TextureKind {
    /// Every declared type, in ordinal order. The totality tests sweep this.
    pub const ALL: [TextureKind; 9] = [
        Self::D1,
        Self::D1Array,
        Self::D2,
        Self::D2Array,
        Self::D2Multisample,
        Self::Cube,
        Self::CubeArray,
        Self::D3,
        Self::D2MultisampleArray,
    ];

    /// The type an ordinal names, or `None` when it names none.
    #[must_use]
    pub const fn from_ordinal(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            0 => Self::D1,
            1 => Self::D1Array,
            2 => Self::D2,
            3 => Self::D2Array,
            4 => Self::D2Multisample,
            5 => Self::Cube,
            6 => Self::CubeArray,
            7 => Self::D3,
            8 => Self::D2MultisampleArray,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::D1 => 0,
            Self::D1Array => 1,
            Self::D2 => 2,
            Self::D2Array => 3,
            Self::D2Multisample => 4,
            Self::Cube => 5,
            Self::CubeArray => 6,
            Self::D3 => 7,
            Self::D2MultisampleArray => 8,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::D1 => "1d",
            Self::D1Array => "1d_array",
            Self::D2 => "2d",
            Self::D2Array => "2d_array",
            Self::D2Multisample => "2d_multisample",
            Self::Cube => "cube",
            Self::CubeArray => "cube_array",
            Self::D3 => "3d",
            Self::D2MultisampleArray => "2d_multisample_array",
        }
    }

    /// Which of width, height and depth this type uses.
    ///
    /// The ones it does not use are required to be one, not merely ignored —
    /// see [`TextureShape::checked`]. A descriptor that sets a height on a 1D
    /// texture is describing something the guest API cannot make, and reading
    /// past it would size the allocation from a field the guest did not intend
    /// as an extent.
    #[must_use]
    pub const fn dimensions(self) -> Dimensions {
        match self {
            Self::D1 | Self::D1Array => Dimensions::One,
            Self::D2
            | Self::D2Array
            | Self::D2Multisample
            | Self::Cube
            | Self::CubeArray
            | Self::D2MultisampleArray => Dimensions::Two,
            Self::D3 => Dimensions::Three,
        }
    }

    /// Whether `array_length` counts anything for this type.
    ///
    /// `Cube` is deliberately *not* arrayed while `CubeArray` is: one cube is
    /// six faces and not an array of one cube, which is exactly the distinction
    /// [`Texture::layers`] has to get right.
    #[must_use]
    pub const fn is_arrayed(self) -> bool {
        matches!(
            self,
            Self::D1Array | Self::D2Array | Self::CubeArray | Self::D2MultisampleArray
        )
    }

    /// Whether the extent's third axis is a depth rather than a layer count.
    ///
    /// Spelled here rather than as `dimensions() == Three` at each reader,
    /// because it is the same question `is_arrayed` and `is_cube` answer and a
    /// reader asking all three should ask them the same way.
    #[must_use]
    pub const fn is_volume(self) -> bool {
        matches!(self.dimensions(), Dimensions::Three)
    }

    /// Whether the type has one spatial axis.
    #[must_use]
    pub const fn is_one_dim(self) -> bool {
        matches!(self.dimensions(), Dimensions::One)
    }

    /// Whether each array element is six faces rather than one slice.
    #[must_use]
    pub const fn is_cube(self) -> bool {
        matches!(self, Self::Cube | Self::CubeArray)
    }

    /// Whether this type carries more than one sample per texel.
    ///
    /// A multisample texture has exactly one mip level and cannot be a cube or
    /// a volume; those are the checks that hang off this answer.
    #[must_use]
    pub const fn is_multisample(self) -> bool {
        matches!(self, Self::D2Multisample | Self::D2MultisampleArray)
    }
}

/// `MTLTextureUsage`: what the guest declared the texture would be used for.
///
/// Held as a mask rather than a set of booleans because the guest sends a mask
/// and an unknown bit has to stay distinguishable from an absent one — a
/// backend that turned this into four flags could not tell "the guest asked for
/// something this build does not know about" from "the guest asked for
/// nothing".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureUsage(pub u32);

impl TextureUsage {
    /// `MTLTextureUsageUnknown`. Metal reads this as "any usage", which is the
    /// widest possible declaration and not the narrowest.
    pub const UNKNOWN: TextureUsage = TextureUsage(0);
    pub const SHADER_READ: TextureUsage = TextureUsage(1 << 0);
    pub const SHADER_WRITE: TextureUsage = TextureUsage(1 << 1);
    pub const RENDER_TARGET: TextureUsage = TextureUsage(1 << 2);
    /// A view of this texture may reinterpret its pixel format.
    pub const PIXEL_FORMAT_VIEW: TextureUsage = TextureUsage(1 << 4);
    pub const SHADER_ATOMIC: TextureUsage = TextureUsage(1 << 5);

    /// Every bit this crate has a meaning for.
    pub const DECLARED: TextureUsage = TextureUsage(
        Self::SHADER_READ.0
            | Self::SHADER_WRITE.0
            | Self::RENDER_TARGET.0
            | Self::PIXEL_FORMAT_VIEW.0
            | Self::SHADER_ATOMIC.0,
    );

    #[must_use]
    pub const fn contains(self, bit: TextureUsage) -> bool {
        self.0 & bit.0 == bit.0
    }

    /// Bits set that this crate has no meaning for.
    ///
    /// Not a refusal by itself: an unknown usage bit narrows nothing a backend
    /// does, and refusing the texture for it would lose a frame over a hint.
    /// It is reportable, which is why it is a value rather than a `bool`.
    #[must_use]
    pub const fn undeclared(self) -> u32 {
        self.0 & !Self::DECLARED.0
    }

    /// Whether the guest declared no usage at all, which Metal reads as every
    /// usage.
    ///
    /// The one place a backend must not treat "no bits" as "no capabilities":
    /// an image created without render-target capability for an `UNKNOWN`
    /// declaration fails at the first pass that attaches it, and the failure
    /// names the pass rather than this descriptor.
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        self.0 == Self::UNKNOWN.0
    }
}

impl core::ops::BitOr for TextureUsage {
    type Output = TextureUsage;

    fn bitor(self, rhs: TextureUsage) -> TextureUsage {
        TextureUsage(self.0 | rhs.0)
    }
}

/// A texture declaration as the fields arrived, before anything checked that
/// they agree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureShape {
    /// Raw `MTLTextureType`. Not yet a [`TextureKind`]: an unknown ordinal has
    /// to survive as far as the refusal that names it.
    pub kind: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mipmap_level_count: u32,
    pub sample_count: u32,
    pub array_length: u32,
    /// `MTLPixelFormat`. Zero is `MTLPixelFormatInvalid`, so it is an absent
    /// format and not a format.
    pub pixel_format: u16,
    pub usage: TextureUsage,
}

/// Why a declaration is not a texture the guest API admits.
///
/// Every variant carries the values that made it true, because the useful
/// question after one of these appears on the fail channel is which field
/// disagreed with which — not that some field did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeRefusal {
    UnknownKind {
        ordinal: u32,
    },
    /// No pixel format. `MTLPixelFormatInvalid` is zero, so a short record and
    /// a record of zeroes both arrive as one.
    NoPixelFormat,
    /// A dimension the type uses is zero. A texture with a zero extent is not
    /// a one-texel texture, and clamping it up would size a four-byte payload
    /// and find any backing long enough.
    ZeroExtent {
        width: u32,
        height: u32,
        depth: u32,
    },
    /// A dimension the type does not use is set to something other than one.
    UnusedDimension {
        kind: TextureKind,
        dimension: &'static str,
        found: u32,
    },
    /// A cube type whose faces are not square.
    ///
    /// The guest API builds a cube from one `size`, so its six faces are
    /// square by construction and a declaration with two different extents is
    /// not describing a cube. Refused rather than squared off: which of the
    /// two the guest meant is not in the record, and every downstream answer —
    /// the pyramid's depth, the face's footprint, the view a sampler binds —
    /// is a different number under each reading.
    ///
    /// It is also invalid usage one layer down. A `VkImage` created
    /// `CUBE_COMPATIBLE` must have equal width and height, and the flag is set
    /// from the type alone, so a non-square cube reaching the executor is a
    /// `vkCreateImage` no driver is obliged to reject.
    CubeNotSquare {
        width: u32,
        height: u32,
    },
    /// `array_length` is zero, or is not one on a type that is not arrayed.
    ArrayLength {
        kind: TextureKind,
        found: u32,
    },
    /// A multisample type declares more than one mip level, or a
    /// single-sample type declares none.
    MipLevels {
        kind: TextureKind,
        found: u32,
    },
    /// More mip levels than the extent has. `floor(log2(max dimension)) + 1` is
    /// the whole pyramid; anything past it names a level with no texels.
    MipLevelsBeyondExtent {
        declared: u32,
        available: u32,
    },
    /// `sample_count` disagrees with the type: not one on a single-sample
    /// type, or one (or not a power of two) on a multisample one.
    SampleCount {
        kind: TextureKind,
        found: u32,
    },
    /// The (mip level, layer) pairs this declaration names cannot be counted.
    ///
    /// [`Texture::layers`] and [`Texture::subresources`] are the answers this
    /// type exists to give, and both are products of guest fields: a cube's
    /// layer count is `array_length * 6`, and a subresource count is that
    /// times the mip levels. `array_length` is otherwise bounded only by not
    /// being zero, so a declaration can name more pairs than a `u32` holds —
    /// and then the answer is a panic in a checked build and a small wrong
    /// number in an unchecked one, from a method whose signature promises
    /// neither.
    ///
    /// Refused at the checkpoint rather than made fallible at the doors. The
    /// point of [`Texture`] is that a holder does not re-derive the rules, and
    /// a `layers()` returning `Option` would push exactly that back out to
    /// every caller.
    SubresourceCount {
        kind: TextureKind,
        array_length: u32,
        mip_levels: u32,
    },
}

impl reims_vgpu_observe::Decline for ShapeRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownKind { .. } => "texture_shape_unknown_kind",
            Self::NoPixelFormat => "texture_shape_no_pixel_format",
            Self::ZeroExtent { .. } => "texture_shape_zero_extent",
            Self::UnusedDimension { .. } => "texture_shape_unused_dimension",
            Self::CubeNotSquare { .. } => "texture_shape_cube_not_square",
            Self::ArrayLength { .. } => "texture_shape_array_length",
            Self::MipLevels { .. } => "texture_shape_mip_levels",
            Self::MipLevelsBeyondExtent { .. } => "texture_shape_mip_levels_beyond_extent",
            Self::SampleCount { .. } => "texture_shape_sample_count",
            Self::SubresourceCount { .. } => "texture_shape_subresource_count",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoPixelFormat => Vec::new(),
            Self::UnknownKind { ordinal } => vec![("ordinal", ordinal.to_string())],
            Self::ZeroExtent {
                width,
                height,
                depth,
            } => vec![
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("depth", depth.to_string()),
            ],
            Self::UnusedDimension {
                kind,
                dimension,
                found,
            } => vec![
                ("kind", kind.name().to_string()),
                ("dimension", (*dimension).to_string()),
                ("found", found.to_string()),
            ],
            Self::CubeNotSquare { width, height } => {
                vec![("width", width.to_string()), ("height", height.to_string())]
            }
            Self::ArrayLength { kind, found } | Self::MipLevels { kind, found } => vec![
                ("kind", kind.name().to_string()),
                ("found", found.to_string()),
            ],
            Self::SampleCount { kind, found } => vec![
                ("kind", kind.name().to_string()),
                ("found", found.to_string()),
            ],
            Self::MipLevelsBeyondExtent {
                declared,
                available,
            } => vec![
                ("declared", declared.to_string()),
                ("available", available.to_string()),
            ],
            Self::SubresourceCount {
                kind,
                array_length,
                mip_levels,
            } => vec![
                ("kind", kind.name().to_string()),
                ("array_length", array_length.to_string()),
                ("mip_levels", mip_levels.to_string()),
            ],
        }
    }
}

/// How many mip levels an extent has room for.
///
/// `floor(log2(max dimension)) + 1`, over exactly the dimensions the type uses
/// — a 2D texture's depth is not part of its pyramid, and including it would
/// admit a level count a 3D texture would have and this one does not.
#[must_use]
pub fn mip_levels_available(kind: TextureKind, extent: Extent3) -> u32 {
    let longest = match kind.dimensions() {
        Dimensions::One => extent.x,
        Dimensions::Two => extent.x.max(extent.y),
        Dimensions::Three => extent.x.max(extent.y).max(extent.z),
    };
    if longest == 0 {
        return 0;
    }
    32 - longest.leading_zeros()
}

/// A texture declaration whose fields have been checked against each other.
///
/// The only way to make one is [`TextureShape::checked`], so a holder of this
/// value knows the seven fields agree — and asks it for layers, extent and
/// samples rather than reading the fields and reapplying the rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Texture {
    kind: TextureKind,
    extent: Extent3,
    mip_levels: u32,
    sample_count: u32,
    array_length: u32,
    pixel_format: u16,
    usage: TextureUsage,
}

impl TextureShape {
    /// Check the declaration against itself.
    ///
    /// # Errors
    ///
    /// [`ShapeRefusal`] naming the two fields that disagree. Checks run from
    /// the type outwards, because every later rule is a rule *of* the type: an
    /// unknown ordinal cannot be told what its unused dimensions are.
    pub fn checked(self) -> Result<Texture, ShapeRefusal> {
        let kind = TextureKind::from_ordinal(self.kind)
            .ok_or(ShapeRefusal::UnknownKind { ordinal: self.kind })?;
        if self.pixel_format == 0 {
            return Err(ShapeRefusal::NoPixelFormat);
        }

        // The dimensions this type does not use must be one, not merely
        // ignored: a set field there is a descriptor for something else.
        let dimensions = kind.dimensions();
        if dimensions == Dimensions::One && self.height != 1 {
            return Err(ShapeRefusal::UnusedDimension {
                kind,
                dimension: "height",
                found: self.height,
            });
        }
        if dimensions != Dimensions::Three && self.depth != 1 {
            return Err(ShapeRefusal::UnusedDimension {
                kind,
                dimension: "depth",
                found: self.depth,
            });
        }
        let extent = Extent3 {
            x: self.width,
            y: self.height,
            z: self.depth,
        };
        if extent.x == 0 || extent.y == 0 || extent.z == 0 {
            return Err(ShapeRefusal::ZeroExtent {
                width: extent.x,
                height: extent.y,
                depth: extent.z,
            });
        }

        // A cube is built from one size, so its faces are square. Checked
        // before the pyramid below it, which takes the longest dimension: on a
        // declaration that is not a cube at all, "the longest" is a reading of
        // a field that does not mean what it is being read as.
        if kind.is_cube() && extent.x != extent.y {
            return Err(ShapeRefusal::CubeNotSquare {
                width: extent.x,
                height: extent.y,
            });
        }

        // An arrayed type needs at least one element; a non-arrayed one has
        // exactly one, and a descriptor saying otherwise is not describing this
        // type.
        if self.array_length == 0 || (!kind.is_arrayed() && self.array_length != 1) {
            return Err(ShapeRefusal::ArrayLength {
                kind,
                found: self.array_length,
            });
        }

        // A multisample texture is one level by construction: there is no
        // filtered reduction of samples to define a smaller one from.
        if self.mipmap_level_count == 0 || (kind.is_multisample() && self.mipmap_level_count != 1) {
            return Err(ShapeRefusal::MipLevels {
                kind,
                found: self.mipmap_level_count,
            });
        }
        let available = mip_levels_available(kind, extent);
        if self.mipmap_level_count > available {
            return Err(ShapeRefusal::MipLevelsBeyondExtent {
                declared: self.mipmap_level_count,
                available,
            });
        }

        // One sample is what a non-multisample type has, and a multisample one
        // has a power of two greater than one — no host offers three samples,
        // and a count that is not a power of two cannot become a
        // `VkSampleCountFlags` bit at all.
        let multisample = kind.is_multisample();
        let sample_ok = if multisample {
            self.sample_count > 1 && self.sample_count.is_power_of_two()
        } else {
            self.sample_count == 1
        };
        if !sample_ok {
            return Err(ShapeRefusal::SampleCount {
                kind,
                found: self.sample_count,
            });
        }

        // Last, because it is a rule about the answers the checked type gives
        // rather than about any one field: `layers` and `subresources` are
        // products, `array_length` is bounded only by not being zero, and a
        // declaration naming more pairs than a `u32` holds has no countable
        // subresources. Both products are checked here so neither door has to
        // be fallible.
        let layers = if kind.is_cube() {
            self.array_length.checked_mul(CUBE_FACES)
        } else {
            Some(self.array_length)
        };
        let countable = layers.and_then(|layers| layers.checked_mul(self.mipmap_level_count));
        if countable.is_none() {
            return Err(ShapeRefusal::SubresourceCount {
                kind,
                array_length: self.array_length,
                mip_levels: self.mipmap_level_count,
            });
        }

        Ok(Texture {
            kind,
            extent,
            mip_levels: self.mipmap_level_count,
            sample_count: self.sample_count,
            array_length: self.array_length,
            pixel_format: self.pixel_format,
            usage: self.usage,
        })
    }
}

impl Texture {
    #[must_use]
    pub const fn kind(self) -> TextureKind {
        self.kind
    }

    #[must_use]
    pub const fn extent(self) -> Extent3 {
        self.extent
    }

    #[must_use]
    pub const fn mip_levels(self) -> u32 {
        self.mip_levels
    }

    #[must_use]
    pub const fn sample_count(self) -> u32 {
        self.sample_count
    }

    /// The guest's `arrayLength`, which counts *elements* and not slices. A
    /// cube array of two is two cubes and twelve slices; see [`Self::layers`].
    #[must_use]
    pub const fn array_length(self) -> u32 {
        self.array_length
    }

    #[must_use]
    pub const fn pixel_format(self) -> u16 {
        self.pixel_format
    }

    #[must_use]
    pub const fn usage(self) -> TextureUsage {
        self.usage
    }

    /// How many addressable slices this texture has.
    ///
    /// The one answer a caller must never derive itself, because the three
    /// cases look alike and the wrong one is a view over slices that do not
    /// exist: a cube is [`CUBE_FACES`] regardless of `arrayLength`, a cube
    /// array is that many per element, and everything else is `arrayLength`.
    ///
    /// A 3D texture is one layer with a depth, never `depth` layers. Those are
    /// different objects: layers are addressed independently and a volume's
    /// slices are filtered between.
    #[must_use]
    pub const fn layers(self) -> u32 {
        if self.kind.is_cube() {
            // `array_length` is 1 for `Cube` by the check above, so one
            // expression covers both cube types.
            self.array_length * CUBE_FACES
        } else {
            self.array_length
        }
    }

    /// Every (mip level, layer) pair this texture has, which is what a render
    /// target expands into one attachment view per.
    #[must_use]
    pub const fn subresources(self) -> u32 {
        self.mip_levels * self.layers()
    }

    /// The extent of one mip level, over the dimensions the type uses.
    ///
    /// A level past the top returns `None` rather than a clamped extent: a
    /// caller asking for a level this texture does not have is asking about
    /// texels that were never allocated.
    #[must_use]
    pub fn level_extent(self, level: u32) -> Option<Extent3> {
        if level >= self.mip_levels {
            return None;
        }
        let reduce = |value: u32| crate::extent::mip_extent(value, level);
        Some(match self.kind.dimensions() {
            Dimensions::One => Extent3 {
                x: reduce(self.extent.x),
                y: 1,
                z: 1,
            },
            Dimensions::Two => Extent3 {
                x: reduce(self.extent.x),
                y: reduce(self.extent.y),
                z: 1,
            },
            Dimensions::Three => Extent3 {
                x: reduce(self.extent.x),
                y: reduce(self.extent.y),
                z: reduce(self.extent.z),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use reims_vgpu_observe::Decline;

    /// A 2D texture that passes every check, as the base for one-field
    /// mutations.
    fn base() -> TextureShape {
        TextureShape {
            kind: TextureKind::D2.ordinal(),
            width: 64,
            height: 32,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            pixel_format: 70,
            usage: TextureUsage::SHADER_READ,
        }
    }

    /// The base shape adjusted to be valid for one type, so that a test
    /// mutating a single field sees that field's refusal and not another's.
    fn shaped(kind: TextureKind) -> TextureShape {
        TextureShape {
            kind: kind.ordinal(),
            height: if kind.dimensions() == Dimensions::One {
                1
            } else if kind.is_cube() {
                // A cube's faces are square, so the base's oblong extent is
                // not a valid declaration of one --- and a test mutating some
                // other field would see this refusal instead of that field's.
                base().width
            } else {
                base().height
            },
            sample_count: if kind.is_multisample() { 4 } else { 1 },
            mipmap_level_count: 1,
            ..base()
        }
    }

    #[test]
    fn the_ordinal_set_is_closed_and_round_trips() {
        for kind in TextureKind::ALL {
            assert_eq!(TextureKind::from_ordinal(kind.ordinal()), Some(kind));
        }
        // Contiguous from zero and nothing past the top: the one shape a
        // hand-written match arm gets wrong is an ordinal that falls through.
        for ordinal in 0..9 {
            assert!(TextureKind::from_ordinal(ordinal).is_some());
        }
        for ordinal in [9, 10, 255, u32::MAX] {
            assert_eq!(TextureKind::from_ordinal(ordinal), None);
        }
        let names: BTreeSet<&str> = TextureKind::ALL.iter().map(|k| k.name()).collect();
        assert_eq!(names.len(), TextureKind::ALL.len());
    }

    #[test]
    fn only_the_cube_types_multiply_by_six() {
        for kind in TextureKind::ALL {
            assert_eq!(
                kind.is_cube(),
                matches!(kind, TextureKind::Cube | TextureKind::CubeArray),
                "{}",
                kind.name()
            );
        }
    }

    #[test]
    fn a_cube_is_six_slices_and_a_cube_array_is_six_per_element() {
        let cube = shaped(TextureKind::Cube).checked().expect("a cube");
        assert_eq!(cube.array_length(), 1);
        assert_eq!(cube.layers(), CUBE_FACES);

        let array = TextureShape {
            array_length: 3,
            ..shaped(TextureKind::CubeArray)
        }
        .checked()
        .expect("a cube array");
        assert_eq!(array.array_length(), 3);
        assert_eq!(array.layers(), 3 * CUBE_FACES);
    }

    #[test]
    fn a_volume_is_one_layer_with_a_depth_and_not_depth_layers() {
        let volume = TextureShape {
            depth: 8,
            ..shaped(TextureKind::D3)
        }
        .checked()
        .expect("a volume");
        assert_eq!(volume.layers(), 1);
        assert_eq!(volume.extent().z, 8);
    }

    #[test]
    fn a_dimension_the_type_does_not_use_must_be_one() {
        let refusal = TextureShape {
            height: 4,
            ..shaped(TextureKind::D1)
        }
        .checked()
        .expect_err("1d has no height");
        assert_eq!(
            refusal,
            ShapeRefusal::UnusedDimension {
                kind: TextureKind::D1,
                dimension: "height",
                found: 4,
            }
        );

        for kind in TextureKind::ALL {
            if kind.dimensions() == Dimensions::Three {
                continue;
            }
            let shape = TextureShape {
                depth: 2,
                height: if kind.dimensions() == Dimensions::One {
                    1
                } else {
                    32
                },
                ..shaped(kind)
            };
            assert!(
                matches!(
                    shape.checked(),
                    Err(ShapeRefusal::UnusedDimension {
                        dimension: "depth",
                        ..
                    })
                ),
                "{} accepted a depth",
                kind.name()
            );
        }
    }

    #[test]
    fn a_zero_extent_is_not_a_one_texel_texture() {
        for zeroed in [
            TextureShape { width: 0, ..base() },
            TextureShape {
                height: 0,
                ..base()
            },
        ] {
            assert!(matches!(
                zeroed.checked(),
                Err(ShapeRefusal::ZeroExtent { .. })
            ));
        }
    }

    #[test]
    fn array_length_is_one_on_every_type_that_is_not_arrayed() {
        for kind in TextureKind::ALL {
            let two = TextureShape {
                array_length: 2,
                ..shaped(kind)
            };
            assert_eq!(
                two.checked().is_ok(),
                kind.is_arrayed(),
                "{} with arrayLength 2",
                kind.name()
            );
            let zero = TextureShape {
                array_length: 0,
                ..shaped(kind)
            };
            assert_eq!(
                zero.checked(),
                Err(ShapeRefusal::ArrayLength { kind, found: 0 }),
                "{} with arrayLength 0",
                kind.name()
            );
        }
    }

    #[test]
    fn a_multisample_texture_is_exactly_one_level_and_more_than_one_sample() {
        for kind in [TextureKind::D2Multisample, TextureKind::D2MultisampleArray] {
            assert_eq!(shaped(kind).checked().map(Texture::sample_count), Ok(4));

            let one_sample = TextureShape {
                sample_count: 1,
                ..shaped(kind)
            };
            assert_eq!(
                one_sample.checked(),
                Err(ShapeRefusal::SampleCount { kind, found: 1 })
            );

            let three = TextureShape {
                sample_count: 3,
                ..shaped(kind)
            };
            assert_eq!(
                three.checked(),
                Err(ShapeRefusal::SampleCount { kind, found: 3 })
            );

            let mipped = TextureShape {
                sample_count: 4,
                mipmap_level_count: 2,
                ..shaped(kind)
            };
            assert_eq!(
                mipped.checked(),
                Err(ShapeRefusal::MipLevels { kind, found: 2 })
            );
        }
    }

    #[test]
    fn a_single_sample_type_refuses_a_sample_count_it_cannot_have() {
        for kind in TextureKind::ALL {
            if kind.is_multisample() {
                continue;
            }
            let shape = TextureShape {
                sample_count: 4,
                ..shaped(kind)
            };
            assert_eq!(
                shape.checked(),
                Err(ShapeRefusal::SampleCount { kind, found: 4 }),
                "{}",
                kind.name()
            );
        }
    }

    #[test]
    fn the_pyramid_is_measured_over_the_dimensions_the_type_uses() {
        // 64x32: seven levels from the width, and the height's six do not
        // shorten it.
        assert_eq!(
            mip_levels_available(TextureKind::D2, Extent3 { x: 64, y: 32, z: 1 }),
            7
        );
        // The same extent as a 1D texture asks only about the width, and the
        // same as a volume also weighs the depth.
        assert_eq!(
            mip_levels_available(TextureKind::D1, Extent3 { x: 64, y: 32, z: 1 }),
            7
        );
        assert_eq!(
            mip_levels_available(TextureKind::D3, Extent3 { x: 4, y: 4, z: 256 }),
            9
        );
        assert_eq!(
            mip_levels_available(TextureKind::D2, Extent3 { x: 1, y: 1, z: 1 }),
            1
        );
    }

    #[test]
    fn a_level_count_past_the_pyramid_is_refused() {
        let shape = TextureShape {
            width: 8,
            height: 8,
            mipmap_level_count: 5,
            ..base()
        };
        assert_eq!(
            shape.checked(),
            Err(ShapeRefusal::MipLevelsBeyondExtent {
                declared: 5,
                available: 4,
            })
        );
        let full = TextureShape {
            mipmap_level_count: 4,
            ..shape
        };
        assert_eq!(full.checked().map(Texture::mip_levels), Ok(4));
    }

    #[test]
    fn a_level_extent_shrinks_only_the_dimensions_the_type_uses() {
        let volume = TextureShape {
            width: 8,
            height: 4,
            depth: 2,
            mipmap_level_count: 4,
            ..shaped(TextureKind::D3)
        }
        .checked()
        .expect("a volume");
        assert_eq!(volume.level_extent(0), Some(Extent3 { x: 8, y: 4, z: 2 }));
        assert_eq!(volume.level_extent(2), Some(Extent3 { x: 2, y: 1, z: 1 }));
        assert_eq!(volume.level_extent(4), None);

        let flat = TextureShape {
            width: 8,
            height: 4,
            mipmap_level_count: 4,
            ..base()
        }
        .checked()
        .expect("a 2d texture");
        // The depth stays one at every level rather than being reduced from a
        // field this type does not use.
        assert_eq!(flat.level_extent(3), Some(Extent3 { x: 1, y: 1, z: 1 }));
    }

    #[test]
    fn a_render_target_expands_into_one_view_per_level_and_layer() {
        let cube = TextureShape {
            width: 16,
            height: 16,
            mipmap_level_count: 5,
            array_length: 2,
            usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            ..shaped(TextureKind::CubeArray)
        }
        .checked()
        .expect("a cube array render target");
        assert_eq!(cube.layers(), 12);
        assert_eq!(cube.subresources(), 60);
        assert!(cube.usage().contains(TextureUsage::RENDER_TARGET));
        assert!(cube.usage().contains(TextureUsage::SHADER_READ));
        assert!(!cube.usage().contains(TextureUsage::SHADER_WRITE));
    }

    #[test]
    fn an_absent_format_is_not_a_format() {
        assert_eq!(
            TextureShape {
                pixel_format: 0,
                ..base()
            }
            .checked(),
            Err(ShapeRefusal::NoPixelFormat)
        );
    }

    #[test]
    fn an_unknown_type_refuses_with_its_ordinal_before_any_other_rule() {
        // Every other field is nonsense too; the type is what is reported,
        // because nothing else can be checked without it.
        let shape = TextureShape {
            kind: 9,
            width: 0,
            height: 0,
            depth: 7,
            mipmap_level_count: 0,
            sample_count: 3,
            array_length: 0,
            pixel_format: 0,
            usage: TextureUsage::UNKNOWN,
        };
        assert_eq!(
            shape.checked(),
            Err(ShapeRefusal::UnknownKind { ordinal: 9 })
        );
    }

    #[test]
    fn an_unknown_usage_bit_is_reported_and_never_refuses() {
        let usage = TextureUsage(TextureUsage::SHADER_READ.0 | 1 << 30);
        assert_eq!(usage.undeclared(), 1 << 30);
        assert!(!usage.is_unknown());
        let texture = TextureShape { usage, ..base() }
            .checked()
            .expect("an unknown usage bit narrows nothing");
        assert_eq!(texture.usage(), usage);

        assert!(TextureUsage::UNKNOWN.is_unknown());
        assert_eq!(TextureUsage::DECLARED.undeclared(), 0);
    }

    #[test]
    fn every_refusal_names_itself_and_carries_what_disagreed() {
        let refusals = [
            ShapeRefusal::UnknownKind { ordinal: 9 },
            ShapeRefusal::NoPixelFormat,
            ShapeRefusal::ZeroExtent {
                width: 0,
                height: 1,
                depth: 1,
            },
            ShapeRefusal::UnusedDimension {
                kind: TextureKind::D1,
                dimension: "height",
                found: 4,
            },
            ShapeRefusal::ArrayLength {
                kind: TextureKind::D2,
                found: 0,
            },
            ShapeRefusal::MipLevels {
                kind: TextureKind::D2Multisample,
                found: 2,
            },
            ShapeRefusal::MipLevelsBeyondExtent {
                declared: 5,
                available: 4,
            },
            ShapeRefusal::SampleCount {
                kind: TextureKind::D2,
                found: 4,
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(Decline::slug).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.slug().starts_with("texture_shape_"));
            assert_eq!(
                refusal.fields().is_empty(),
                refusal == ShapeRefusal::NoPixelFormat
            );
        }
    }

    /// The failure this exists to prevent: `layers` and `subresources` are the
    /// two answers this type is for, and both are products of guest fields.
    /// `array_length` was bounded only by not being zero, so a cube array
    /// naming more than `u32::MAX / 6` elements — or any type naming more
    /// (level, layer) pairs than a `u32` holds — reached a `layers()` that
    /// panicked in a checked build and wrapped to a small number in an
    /// unchecked one. A wrapped layer count is a view over slices that do not
    /// exist.
    #[test]
    fn a_declaration_with_more_subresources_than_can_be_counted_refuses() {
        let base = TextureShape {
            kind: TextureKind::CubeArray.ordinal(),
            width: 16,
            height: 16,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            pixel_format: 70,
            usage: TextureUsage::SHADER_READ,
        };
        // The cube multiply is the first of the two products.
        let limit = u32::MAX / CUBE_FACES;
        assert!(
            TextureShape {
                array_length: limit,
                ..base
            }
            .checked()
            .is_ok(),
            "exactly as many faces as fit is a countable declaration"
        );
        assert_eq!(
            TextureShape {
                array_length: limit + 1,
                ..base
            }
            .checked()
            .expect_err("one element past what six faces of it can be counted as"),
            ShapeRefusal::SubresourceCount {
                kind: TextureKind::CubeArray,
                array_length: limit + 1,
                mip_levels: 1,
            }
        );

        // And the mip multiply is the second, on a type with no cube factor at
        // all --- so a check that only guarded the cube product would pass this.
        let flat = TextureShape {
            kind: TextureKind::D2Array.ordinal(),
            width: 1 << 15,
            height: 1,
            mipmap_level_count: 16,
            array_length: 1 << 28,
            ..base
        };
        assert_eq!(
            flat.checked()
                .expect_err("sixteen levels of two hundred and sixty-eight million layers"),
            ShapeRefusal::SubresourceCount {
                kind: TextureKind::D2Array,
                array_length: 1 << 28,
                mip_levels: 16,
            }
        );
        assert!(
            TextureShape {
                mipmap_level_count: 15,
                ..flat
            }
            .checked()
            .is_ok(),
            "one level fewer is exactly countable"
        );
    }

    /// Every declaration that passes the checkpoint, asked every question the
    /// checked type answers.
    ///
    /// Driven over a product of the fields rather than over named cases: what
    /// the type promises is that its answers agree *with each other*, and a
    /// hand-picked declaration checks one point of a space whose corners are
    /// where the products overflow and the pyramids bottom out.
    #[test]
    fn no_answer_a_checked_texture_gives_disagrees_with_another() {
        let interesting: [u32; 8] = [
            1,
            2,
            3,
            16,
            1 << 15,
            1 << 16,
            u32::MAX / CUBE_FACES,
            u32::MAX,
        ];
        let mut accepted = 0u32;
        let mut refused = 0u32;
        for kind in TextureKind::ALL {
            for &width in &interesting {
                for &height in &[1u32, 2, 33, 1 << 15] {
                    for &array_length in &interesting {
                        for &mipmap_level_count in &[1u32, 2, 5, 17, 32, u32::MAX] {
                            let shape = TextureShape {
                                kind: kind.ordinal(),
                                width,
                                height: if kind.dimensions() == Dimensions::One {
                                    1
                                } else if kind.is_cube() {
                                    width
                                } else {
                                    height
                                },
                                depth: 1,
                                mipmap_level_count,
                                sample_count: if kind.is_multisample() { 4 } else { 1 },
                                array_length: if kind.is_arrayed() { array_length } else { 1 },
                                pixel_format: 70,
                                usage: TextureUsage::SHADER_READ,
                            };
                            let Ok(texture) = shape.checked() else {
                                refused += 1;
                                continue;
                            };
                            accepted += 1;

                            // The two products are countable, which is the
                            // whole of the new rule and is asserted by asking
                            // rather than by re-deriving --- these panic in
                            // this build if they are not.
                            let layers = texture.layers();
                            assert_eq!(texture.subresources(), layers * texture.mip_levels());

                            // A cube is six faces per element and nothing else
                            // is.
                            assert_eq!(
                                layers,
                                if kind.is_cube() {
                                    texture.array_length() * CUBE_FACES
                                } else {
                                    texture.array_length()
                                }
                            );

                            // A level exists exactly when it is below the
                            // declared count, and level zero is the extent the
                            // declaration named.
                            assert!(texture.level_extent(texture.mip_levels()).is_none());
                            assert_eq!(texture.level_extent(0), Some(texture.extent()));

                            // The pyramid never grows, never reaches zero, and
                            // leaves the axes the type does not use at one.
                            let mut previous = texture.extent();
                            for level in 1..texture.mip_levels() {
                                let e = texture.level_extent(level).expect("below the count");
                                for (now, before) in
                                    [(e.x, previous.x), (e.y, previous.y), (e.z, previous.z)]
                                {
                                    assert!(now <= before && now >= 1, "{kind:?} level {level}");
                                }
                                if texture.kind().dimensions() != Dimensions::Three {
                                    assert_eq!(e.z, 1);
                                }
                                if texture.kind().dimensions() == Dimensions::One {
                                    assert_eq!(e.y, 1);
                                }
                                previous = e;
                            }
                        }
                    }
                }
            }
        }
        // Floors per outcome, not one total: a product that refused everything
        // would satisfy any bound written on the sweep as a whole.
        assert!(accepted > 500, "{accepted}");
        assert!(refused > 500, "{refused}");
    }

    /// The failure this exists to prevent: the guest API builds a cube from
    /// one `size`, so a declaration naming two different extents is not
    /// describing a cube — and it reached the executor, which sets
    /// `CUBE_COMPATIBLE` from the type alone. A `VkImage` created with that
    /// flag must have equal width and height, so the result was a
    /// `vkCreateImage` no driver is obliged to reject and a cube view reading
    /// past the end of its own face.
    #[test]
    fn a_cube_whose_faces_are_not_square_is_not_a_cube() {
        for kind in TextureKind::ALL {
            let oblong = TextureShape {
                width: 64,
                height: 32,
                ..shaped(kind)
            };
            // A 1D type has no height to disagree with, so it is not part of
            // this question at all.
            if kind.dimensions() == Dimensions::One {
                continue;
            }
            let answer = oblong.checked();
            if kind.is_cube() {
                assert_eq!(
                    answer.expect_err("a cube with two extents"),
                    ShapeRefusal::CubeNotSquare {
                        width: 64,
                        height: 32,
                    },
                    "{}",
                    kind.name()
                );
            } else {
                assert!(
                    answer.is_ok(),
                    "{} is not a cube and owes nothing about its aspect",
                    kind.name()
                );
            }
            // And the square form of the same declaration is admitted, so the
            // refusal is the aspect and not the extent.
            assert!(TextureShape {
                height: 64,
                ..oblong
            }
            .checked()
            .is_ok());
        }
    }
}
