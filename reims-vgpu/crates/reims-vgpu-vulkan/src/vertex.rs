//! What a vertex attribute becomes here: a `VkFormat`, a binding rate, a
//! divisor, and the three device questions any of those can fail on.
//!
//! # The signedness ABI
//!
//! Twelve signed Metal formats map to **unsigned** Vulkan formats, and that is
//! deliberate rather than a slip. The shader translator that produces this
//! device's SPIR-V mints every integer as `OpTypeInt <width> 0` and declares a
//! vertex attribute's `Input` variable with the source parameter's own type
//! verbatim, so every integer vertex stage input in the modules this rail
//! executes is unsigned. Vulkan's numeric-type match rule then makes a
//! `*_UINT` attribute format the only conforming pairing; a `*_SINT` format
//! against those modules is the undefined case. For the 32-bit arms the
//! pairing is bit-exact — nothing is extended, and the shader body's signed
//! instructions read the intended value.
//!
//! This is a coupling to a component outside this crate, so
//! [`SIGNED_AS_UNSIGNED`] enumerates the arms that rest on it. If the
//! translator ever emits a genuinely signed stage input, every one of them
//! becomes wrong at once and silently, and the blast radius should be
//! readable rather than a comment on one line of twelve.
//!
//! # Not every format can be fetched
//!
//! Vulkan mandates only a subset of formats as vertex attributes. **Every
//! three-channel 8- and 16-bit format is outside it** — `R8G8B8_*` and
//! `R16G16B16_*` are optional while their four-channel siblings and every
//! `R32G32B32_*` are mandatory. So support is measured, once, into
//! [`VertexFormatSupport`], and never assumed from a format's shape.
//!
//! # The widening substitute, and why it is conditional
//!
//! A declined three-channel format has a four-channel sibling that is
//! mandatory everywhere, and channels `0..3` sit at identical byte offsets in
//! both. The first three values the shader reads are the same bytes.
//!
//! **The fourth is not, and that is the whole difficulty.** Vulkan supplies a
//! vertex input's missing channels from `(0, 0, 0, 1)`, so a shader input
//! declared four-wide over the guest's three-channel format takes
//! `(x, y, z, 1.0)`. Over the substitute it takes `(x, y, z, whatever those
//! bytes hold)` — the next attribute's data, or the vertex's padding —
//! because the channel is now *supplied* rather than defaulted. Nothing about
//! the bytes changed; Vulkan stopped filling in a default the guest relied on.
//!
//! So the substitution is made only where it is invisible, and what makes it
//! invisible is what the shader declares at that location:
//!
//! | shader declares | verdict |
//! |---|---|
//! | three channels or fewer | widen — Vulkan discards what the format oversupplies |
//! | four channels | refuse, by name |
//! | nothing at this location | widen — an input nothing reads |
//! | something the reader could not measure | refuse, by name |
//!
//! Refusing rather than widening anyway is the rule this device follows
//! everywhere: a wrong `w` reaching a vertex shader is a geometry error with
//! nothing downstream able to name it.
//!
//! Widening also has to stay inside the buffer. The widened attribute reads
//! more bytes at the same offset, so the last vertex's fetch runs past the end
//! unless `offset + widened bytes <= stride`. That is checked, and a
//! substitution that would not fit refuses by name — a read past the end of a
//! vertex buffer is a violation, not a degraded frame.
//!
//! # Constant is an instance rate with no advance
//!
//! Vulkan has `VERTEX` and `INSTANCE` and nothing else.
//! `MTLVertexStepFunctionConstant` fetches once for the whole draw, which is
//! `INSTANCE` with a divisor of **zero** — and a zero divisor is itself a
//! capability, `vertexAttributeInstanceRateZeroDivisor`. A divisor above one
//! needs `vertexAttributeInstanceRateDivisor` and must be within
//! `maxVertexAttribDivisor`. All three come off the census, and each refuses
//! by its own name so a guest that asked for a constant attribute is
//! distinguishable from one that asked for a divisor this host cannot reach.
//!
//! # Planned, not created
//!
//! Nothing here builds a pipeline. Every mapping is tested with no GPU.

use ash::vk;
use reims_vgpu_core::vertex_format::VertexFormat;
use reims_vgpu_core::vertex_step::StepFunction;

/// `MTLVertexFormat` → `VkFormat`. Total.
///
/// Written out, and the twelve signed-to-unsigned arms are the reason it must
/// be: see the module doc.
#[must_use]
pub const fn format(guest: VertexFormat) -> vk::Format {
    match guest {
        VertexFormat::UChar2 => vk::Format::R8G8_UINT,
        VertexFormat::UChar3 => vk::Format::R8G8B8_UINT,
        VertexFormat::UChar4 => vk::Format::R8G8B8A8_UINT,
        VertexFormat::Char2 => vk::Format::R8G8_UINT,
        VertexFormat::Char3 => vk::Format::R8G8B8_UINT,
        VertexFormat::Char4 => vk::Format::R8G8B8A8_UINT,
        VertexFormat::UChar2Normalized => vk::Format::R8G8_UNORM,
        VertexFormat::UChar3Normalized => vk::Format::R8G8B8_UNORM,
        VertexFormat::UChar4Normalized => vk::Format::R8G8B8A8_UNORM,
        VertexFormat::Char2Normalized => vk::Format::R8G8_SNORM,
        VertexFormat::Char3Normalized => vk::Format::R8G8B8_SNORM,
        VertexFormat::Char4Normalized => vk::Format::R8G8B8A8_SNORM,
        VertexFormat::UShort2 => vk::Format::R16G16_UINT,
        VertexFormat::UShort3 => vk::Format::R16G16B16_UINT,
        VertexFormat::UShort4 => vk::Format::R16G16B16A16_UINT,
        VertexFormat::Short2 => vk::Format::R16G16_UINT,
        VertexFormat::Short3 => vk::Format::R16G16B16_UINT,
        VertexFormat::Short4 => vk::Format::R16G16B16A16_UINT,
        VertexFormat::UShort2Normalized => vk::Format::R16G16_UNORM,
        VertexFormat::UShort3Normalized => vk::Format::R16G16B16_UNORM,
        VertexFormat::UShort4Normalized => vk::Format::R16G16B16A16_UNORM,
        VertexFormat::Short2Normalized => vk::Format::R16G16_SNORM,
        VertexFormat::Short3Normalized => vk::Format::R16G16B16_SNORM,
        VertexFormat::Short4Normalized => vk::Format::R16G16B16A16_SNORM,
        VertexFormat::Half2 => vk::Format::R16G16_SFLOAT,
        VertexFormat::Half3 => vk::Format::R16G16B16_SFLOAT,
        VertexFormat::Half4 => vk::Format::R16G16B16A16_SFLOAT,
        VertexFormat::Float => vk::Format::R32_SFLOAT,
        VertexFormat::Float2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Int => vk::Format::R32_UINT,
        VertexFormat::Int2 => vk::Format::R32G32_UINT,
        VertexFormat::Int3 => vk::Format::R32G32B32_UINT,
        VertexFormat::Int4 => vk::Format::R32G32B32A32_UINT,
        VertexFormat::UInt => vk::Format::R32_UINT,
        VertexFormat::UInt2 => vk::Format::R32G32_UINT,
        VertexFormat::UInt3 => vk::Format::R32G32B32_UINT,
        VertexFormat::UInt4 => vk::Format::R32G32B32A32_UINT,
        VertexFormat::Int1010102Normalized => vk::Format::A2B10G10R10_SNORM_PACK32,
        VertexFormat::UInt1010102Normalized => vk::Format::A2B10G10R10_UNORM_PACK32,
        VertexFormat::UChar4NormalizedBgra => vk::Format::B8G8R8A8_UNORM,
        VertexFormat::UChar => vk::Format::R8_UINT,
        VertexFormat::Char => vk::Format::R8_UINT,
        VertexFormat::UCharNormalized => vk::Format::R8_UNORM,
        VertexFormat::CharNormalized => vk::Format::R8_SNORM,
        VertexFormat::UShort => vk::Format::R16_UINT,
        VertexFormat::Short => vk::Format::R16_UINT,
        VertexFormat::UShortNormalized => vk::Format::R16_UNORM,
        VertexFormat::ShortNormalized => vk::Format::R16_SNORM,
        VertexFormat::Half => vk::Format::R16_SFLOAT,
        VertexFormat::FloatRg11B10 => vk::Format::B10G11R11_UFLOAT_PACK32,
        VertexFormat::FloatRgb9E5 => vk::Format::E5B9G9R9_UFLOAT_PACK32,
    }
}

/// The signed Metal formats deliberately bound to unsigned Vulkan formats.
///
/// Enumerated rather than commented so the coupling has a blast radius that
/// can be read. Every entry is correct only while the shader translator emits
/// unsigned integer stage inputs.
pub const SIGNED_AS_UNSIGNED: [VertexFormat; 12] = [
    VertexFormat::Char,
    VertexFormat::Char2,
    VertexFormat::Char3,
    VertexFormat::Char4,
    VertexFormat::Short,
    VertexFormat::Short2,
    VertexFormat::Short3,
    VertexFormat::Short4,
    VertexFormat::Int,
    VertexFormat::Int2,
    VertexFormat::Int3,
    VertexFormat::Int4,
];

/// Which formats this device reported usable as a vertex attribute.
///
/// A bitset keyed by `MTLVertexFormat` ordinal, which fits one word because
/// the enumeration stops at 55. `Copy` and `Eq`, so it lives in the census
/// beside every other measured fact rather than being re-queried per pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexFormatSupport {
    bits: u64,
}

/// The width claim above, checked rather than written down.
///
/// Every shift in this bitset is `1u64 << guest.ordinal()`, and the ordinal is
/// `MTLVertexFormat`'s own value in another crate — not a dense index this
/// crate controls. A format added there at 64 or beyond does not fail here: on
/// x86 the shift amount is masked, so ordinal 64 sets ordinal *zero's* bit and
/// a device declining `UChar2` would report declining the new format as well.
/// A silently wrong capability answer, from a comment that was true when it was
/// written.
const _: () = {
    let mut index = 0;
    while index < VertexFormat::ALL.len() {
        assert!(
            VertexFormat::ALL[index].ordinal() < u64::BITS,
            "a vertex format ordinal no longer fits this bitset's word"
        );
        index += 1;
    }
};

impl VertexFormatSupport {
    /// A device that declined everything. The conservative state, and the one
    /// a `Default` census is in.
    pub const NONE: Self = Self { bits: 0 };

    /// Build from a predicate asked once per format.
    ///
    /// Takes the whole enumeration rather than a caller-supplied list, so a
    /// format that gains a variant is queried without anybody remembering to
    /// add it.
    pub fn measured(mut supported: impl FnMut(vk::Format) -> bool) -> Self {
        let mut bits = 0u64;
        for guest in VertexFormat::ALL {
            if supported(format(guest)) {
                bits |= 1 << guest.ordinal();
            }
        }
        Self { bits }
    }

    /// Every format. Only for tests and for a host that reported so.
    #[must_use]
    pub fn all() -> Self {
        Self::measured(|_| true)
    }

    #[must_use]
    pub const fn has(self, guest: VertexFormat) -> bool {
        self.bits & (1 << guest.ordinal()) != 0
    }

    #[must_use]
    pub const fn without(self, guest: VertexFormat) -> Self {
        Self {
            bits: self.bits & !(1 << guest.ordinal()),
        }
    }

    #[must_use]
    pub const fn with(self, guest: VertexFormat) -> Self {
        Self {
            bits: self.bits | (1 << guest.ordinal()),
        }
    }

    /// How many of the enumeration this device accepted.
    #[must_use]
    pub const fn count(self) -> u32 {
        self.bits.count_ones()
    }
}

/// What this host offers for the parts of a vertex binding that are not
/// mappings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexCell {
    pub formats: VertexFormatSupport,
    /// `VkPhysicalDeviceVertexAttributeDivisorFeatures::vertexAttributeInstanceRateDivisor`,
    /// or 1.4 core.
    pub instance_rate_divisor: bool,
    /// `…::vertexAttributeInstanceRateZeroDivisor`. Separate because a
    /// constant-rate attribute needs exactly this one and a divisor of two
    /// needs exactly the other.
    pub zero_divisor: bool,
    /// `VkPhysicalDeviceVertexAttributeDivisorProperties::maxVertexAttribDivisor`.
    /// Meaningless without `instance_rate_divisor`, and never read without it.
    pub max_divisor: u32,
}

/// What the shader declares at an attribute's location, as far as widening
/// cares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderInput {
    /// The input exists and this many channels are read.
    Channels(u32),
    /// No input is declared at this location.
    Absent,
    /// An input is declared and the reader could not measure its width.
    Unreadable,
}

/// Why a vertex attribute cannot be bound here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The device declined the format and it has no wider sibling to
    /// substitute — a four-channel or packed format.
    NoFormat { guest: VertexFormat },
    /// The substitute is available but the shader reads four channels, so the
    /// fourth would stop being the `1.0` Vulkan defaults in.
    WidenReadAsFour { guest: VertexFormat },
    /// The substitute is available and the shader's declaration could not be
    /// measured, so there is no way to know the fourth channel is unread.
    WidenShaderUnreadable { guest: VertexFormat },
    /// The wider read would run past the end of a vertex.
    WidenPastStride {
        guest: VertexFormat,
        offset: u32,
        widened_bytes: u32,
        stride: u32,
    },
    /// A tessellation step function on a rail that builds no tessellation
    /// pipeline. Recognised, and declined for what it is.
    TessellationStep { step: StepFunction },
    /// `MTLVertexStepFunctionConstant` without
    /// `vertexAttributeInstanceRateZeroDivisor`.
    NoZeroDivisor,
    /// A per-instance divisor above one without
    /// `vertexAttributeInstanceRateDivisor`.
    NoInstanceDivisor { divisor: u32 },
    /// A per-instance divisor above `maxVertexAttribDivisor`.
    DivisorTooLarge { divisor: u32, limit: u32 },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoFormat { .. } => "vk_vertex_no_format",
            Self::WidenReadAsFour { .. } => "vk_vertex_widen_read_as_four",
            Self::WidenShaderUnreadable { .. } => "vk_vertex_widen_shader_unreadable",
            Self::WidenPastStride { .. } => "vk_vertex_widen_past_stride",
            Self::TessellationStep { .. } => "vk_vertex_tessellation_step",
            Self::NoZeroDivisor => "vk_vertex_no_zero_divisor",
            Self::NoInstanceDivisor { .. } => "vk_vertex_no_instance_divisor",
            Self::DivisorTooLarge { .. } => "vk_vertex_divisor_too_large",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFormat { guest }
            | Self::WidenReadAsFour { guest }
            | Self::WidenShaderUnreadable { guest } => {
                write!(f, "{} format={}", self.slug(), guest.name())
            }
            Self::WidenPastStride {
                guest,
                offset,
                widened_bytes,
                stride,
            } => write!(
                f,
                "{} format={} offset={offset} widened_bytes={widened_bytes} stride={stride}",
                self.slug(),
                guest.name()
            ),
            Self::TessellationStep { step } => write!(f, "{} step={}", self.slug(), step.name()),
            Self::NoZeroDivisor => f.write_str(self.slug()),
            Self::NoInstanceDivisor { divisor } => {
                write!(f, "{} divisor={divisor}", self.slug())
            }
            Self::DivisorTooLarge { divisor, limit } => {
                write!(f, "{} divisor={divisor} limit={limit}", self.slug())
            }
        }
    }
}

/// One attribute, as `VkVertexInputAttributeDescription` would hold it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttributePlan {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
    /// `Some(original)` when the device declined the attribute's own format
    /// and the wider sibling was substituted. Carried so a widened pipeline is
    /// reported rather than assumed.
    pub widened_from: Option<VertexFormat>,
}

impl AttributePlan {
    pub const fn native(self) -> vk::VertexInputAttributeDescription {
        vk::VertexInputAttributeDescription {
            location: self.location,
            binding: self.binding,
            format: self.format,
            offset: self.offset,
        }
    }
}

/// One buffer layout, as `VkVertexInputBindingDescription` and its divisor
/// would hold it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindingPlan {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: vk::VertexInputRate,
    /// The instance divisor. One under `PerVertex` and `PerInstance` with a
    /// rate of one, zero under `Constant`, and the guest's rate otherwise.
    ///
    /// Carried even where it is one, because "this binding has no divisor
    /// structure" and "this binding's divisor is one" are the same pipeline
    /// and a caller should not have to re-derive which.
    pub divisor: u32,
}

impl BindingPlan {
    pub const fn native(self) -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: self.binding,
            stride: self.stride,
            input_rate: self.input_rate,
        }
    }

    /// Whether this binding needs a `VkVertexInputBindingDivisorDescription`
    /// chained onto the pipeline at all.
    #[must_use]
    pub const fn needs_divisor_structure(self) -> bool {
        self.divisor != 1
    }
}

/// `MTLVertexStepFunction` → the rate a binding is created with.
///
/// `Constant` has no spelling of its own: Metal advances a constant-rate
/// attribute never, which is `INSTANCE` with a zero divisor. This decides only
/// the rate; the divisor is chosen beside it in [`binding`].
///
/// # Errors
///
/// [`Refusal::TessellationStep`] for the two step functions that only mean
/// something inside a tessellation pipeline.
pub const fn input_rate(step: StepFunction) -> Result<vk::VertexInputRate, Refusal> {
    Ok(match step {
        StepFunction::PerVertex => vk::VertexInputRate::VERTEX,
        StepFunction::Constant | StepFunction::PerInstance => vk::VertexInputRate::INSTANCE,
        StepFunction::PerPatch | StepFunction::PerPatchControlPoint => {
            return Err(Refusal::TessellationStep { step })
        }
    })
}

/// Plan one buffer layout.
///
/// `rate` is the guest's `stepRate`, whose contract with `step` is checked one
/// layer down — a zero is legal under `Constant` and under nothing else.
///
/// # Errors
///
/// [`Refusal`], with nothing partially planned.
pub fn binding(
    index: u32,
    stride: u32,
    step: StepFunction,
    rate: u32,
    cell: VertexCell,
) -> Result<BindingPlan, Refusal> {
    let input_rate = input_rate(step)?;
    let divisor = match step {
        StepFunction::PerVertex => 1,
        // Fetch once for the whole draw. Its own capability, and its own
        // refusal, because a host may have divisors and not this.
        StepFunction::Constant => {
            if !cell.zero_divisor {
                return Err(Refusal::NoZeroDivisor);
            }
            0
        }
        StepFunction::PerInstance => {
            if rate != 1 {
                if !cell.instance_rate_divisor {
                    return Err(Refusal::NoInstanceDivisor { divisor: rate });
                }
                if rate > cell.max_divisor {
                    return Err(Refusal::DivisorTooLarge {
                        divisor: rate,
                        limit: cell.max_divisor,
                    });
                }
            }
            rate
        }
        StepFunction::PerPatch | StepFunction::PerPatchControlPoint => {
            return Err(Refusal::TessellationStep { step })
        }
    };
    Ok(BindingPlan {
        binding: index,
        stride,
        input_rate,
        divisor,
    })
}

/// Plan one attribute, substituting a wider format only where doing so is
/// invisible.
///
/// `shader` is what is declared at `location`, which the caller reads from the
/// module it is about to bind. It is consulted only when the substitution is
/// actually needed — on a host that declines nothing, no module is walked —
/// and only after the two structural questions below have been answered, so a
/// substitution that could never be made does not cost a module walk.
///
/// Takes the format support rather than the whole [`VertexCell`] because that
/// is all it reads. The divisor capabilities belong to [`binding`], and a
/// signature naming them here would oblige a caller that has measured formats
/// and nothing else to invent them.
///
/// # Errors
///
/// [`Refusal`], with nothing partially planned.
pub fn attribute(
    location: u32,
    binding: u32,
    guest: VertexFormat,
    offset: u32,
    stride: u32,
    formats: VertexFormatSupport,
    shader: impl FnOnce() -> ShaderInput,
) -> Result<AttributePlan, Refusal> {
    if formats.has(guest) {
        return Ok(AttributePlan {
            location,
            binding,
            format: format(guest),
            offset,
            widened_from: None,
        });
    }
    let Some(wider) = guest.widened() else {
        return Err(Refusal::NoFormat { guest });
    };
    if !formats.has(wider) {
        // The substitute is mandatory in Vulkan for every case this reaches,
        // so a device declining it has declined the whole family; there is no
        // second rung to climb to.
        return Err(Refusal::NoFormat { guest });
    }
    // Asked before the shader, because it is a comparison of three integers
    // the caller already holds and the shader is a walk of a whole module. An
    // attribute whose substitute cannot fit inside the vertex is refused
    // whatever the shader reads, so the walk would buy nothing but a different
    // name for the same refusal.
    let widened_bytes = wider.bytes();
    let fits = offset
        .checked_add(widened_bytes)
        .is_some_and(|end| end <= stride);
    if !fits {
        return Err(Refusal::WidenPastStride {
            guest,
            offset,
            widened_bytes,
            stride,
        });
    }
    match shader() {
        ShaderInput::Channels(read) if read > guest.components() => {
            return Err(Refusal::WidenReadAsFour { guest })
        }
        ShaderInput::Unreadable => return Err(Refusal::WidenShaderUnreadable { guest }),
        ShaderInput::Channels(_) | ShaderInput::Absent => {}
    }
    Ok(AttributePlan {
        location,
        binding,
        format: format(wider),
        offset,
        widened_from: Some(guest),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::vertex_format::{ComponentKind, Layout};
    use std::collections::BTreeSet;

    fn cell() -> VertexCell {
        VertexCell {
            formats: VertexFormatSupport::all(),
            instance_rate_divisor: true,
            zero_divisor: true,
            max_divisor: 16,
        }
    }

    /// Never consulted in the paths that must not consult it.
    fn unreached() -> ShaderInput {
        panic!("the shader was walked on a host that declines nothing")
    }

    #[test]
    fn every_format_maps_and_the_width_of_the_mapping_matches_the_guest_footprint() {
        // No two guest formats with different footprints may share a Vulkan
        // format, and the twelve signed-to-unsigned pairs are the ones that
        // legitimately collide.
        let mut by_vk: std::collections::BTreeMap<i32, Vec<VertexFormat>> = Default::default();
        for guest in VertexFormat::ALL {
            by_vk.entry(format(guest).as_raw()).or_default().push(guest);
        }
        for (vk_format, guests) in &by_vk {
            let footprints: BTreeSet<u32> = guests.iter().map(|g| g.bytes()).collect();
            assert_eq!(
                footprints.len(),
                1,
                "vk format {vk_format} is shared by formats of different sizes: {guests:?}"
            );
            let components: BTreeSet<u32> = guests.iter().map(|g| g.components()).collect();
            assert_eq!(components.len(), 1, "vk format {vk_format} spans widths");
        }
        // Non-vacuity: collisions really do happen, and only where a signed
        // guest format shares its unsigned twin's spelling.
        let collided: BTreeSet<&str> = by_vk
            .values()
            .filter(|g| g.len() > 1)
            .flat_map(|g| g.iter().map(|f| f.name()))
            .collect();
        assert!(collided.len() >= 12);
    }

    /// The coupling this rail rests on, stated as a set rather than a comment.
    #[test]
    fn every_signed_arm_lands_on_an_unsigned_format_and_the_list_is_complete() {
        for guest in SIGNED_AS_UNSIGNED {
            let Layout::Separate { kind, .. } = guest.layout() else {
                panic!("{} is not a separate layout", guest.name());
            };
            assert!(kind.is_signed(), "{} is not signed", guest.name());
            assert!(kind.is_integer(), "{} is not an integer", guest.name());
            // And its Vulkan spelling is the one its unsigned twin gets.
            assert!(
                format!("{:?}", format(guest)).contains("UINT"),
                "{} maps to {:?}",
                guest.name(),
                format(guest)
            );
        }

        // The list is every signed *integer* format and no others. The signed
        // normalized ones map to `SNORM` and are not part of the coupling.
        let listed: BTreeSet<u32> = SIGNED_AS_UNSIGNED.iter().map(|f| f.ordinal()).collect();
        let signed_integers: BTreeSet<u32> = VertexFormat::ALL
            .iter()
            .filter(|f| match f.layout() {
                Layout::Separate { kind, .. } => kind.is_signed() && kind.is_integer(),
                Layout::Packed(_) => false,
            })
            .map(|f| f.ordinal())
            .collect();
        assert_eq!(listed, signed_integers);
        assert_eq!(listed.len(), 12);

        // And the signed normalized ones are untouched by it.
        assert_eq!(
            format(VertexFormat::Char4Normalized),
            vk::Format::R8G8B8A8_SNORM
        );
        assert!(!ComponentKind::Snorm8.is_integer());
    }

    #[test]
    fn a_supported_format_is_bound_as_itself_and_no_module_is_walked() {
        let plan = attribute(3, 1, VertexFormat::Short3, 8, 32, cell().formats, unreached)
            .expect("this host supports everything");
        assert_eq!(plan.format, vk::Format::R16G16B16_UINT);
        assert_eq!(plan.widened_from, None);
        assert_eq!(plan.location, 3);
        assert_eq!(plan.binding, 1);
        assert_eq!(plan.offset, 8);
        assert_eq!(plan.native().format, vk::Format::R16G16B16_UINT);
    }

    /// The optional formats really are the three-channel 8- and 16-bit ones,
    /// so the widening path is the one a real host reaches.
    #[test]
    fn a_declined_three_channel_format_widens_when_the_shader_reads_three() {
        let narrow = VertexCell {
            formats: VertexFormatSupport::all().without(VertexFormat::Short3),
            ..cell()
        };
        let plan = attribute(0, 0, VertexFormat::Short3, 0, 8, narrow.formats, || {
            ShaderInput::Channels(3)
        })
        .expect("the wider sibling is mandatory");
        assert_eq!(plan.format, vk::Format::R16G16B16A16_UINT);
        assert_eq!(plan.widened_from, Some(VertexFormat::Short3));
        // The bytes it reads grew by exactly one channel.
        assert_eq!(
            VertexFormat::Short4.bytes(),
            VertexFormat::Short3.bytes() + 2
        );

        // An input nothing declares widens too — there is no fourth channel
        // for anybody to read.
        assert!(
            attribute(0, 0, VertexFormat::Short3, 0, 8, narrow.formats, || {
                ShaderInput::Absent
            })
            .is_ok()
        );
    }

    /// The whole difficulty: Vulkan stops defaulting the fourth channel.
    #[test]
    fn a_shader_that_reads_four_channels_refuses_rather_than_widening() {
        let narrow = VertexCell {
            formats: VertexFormatSupport::all().without(VertexFormat::Short3),
            ..cell()
        };
        let refused = attribute(0, 0, VertexFormat::Short3, 0, 8, narrow.formats, || {
            ShaderInput::Channels(4)
        })
        .expect_err("the fourth channel would stop being 1.0");
        assert_eq!(
            refused,
            Refusal::WidenReadAsFour {
                guest: VertexFormat::Short3
            }
        );
        assert_eq!(refused.slug(), "vk_vertex_widen_read_as_four");

        // Unmeasurable is refused separately, because "reads four" and "we
        // could not tell" are different reports.
        let unreadable = attribute(0, 0, VertexFormat::Short3, 0, 8, narrow.formats, || {
            ShaderInput::Unreadable
        })
        .expect_err("no way to know the fourth channel is unread");
        assert_eq!(
            unreadable,
            Refusal::WidenShaderUnreadable {
                guest: VertexFormat::Short3
            }
        );
        assert_ne!(refused.slug(), unreadable.slug());
    }

    /// A read past the end of a vertex is a violation, not a degraded frame.
    #[test]
    fn a_widened_read_that_would_leave_the_vertex_refuses() {
        let narrow = VertexCell {
            formats: VertexFormatSupport::all().without(VertexFormat::Short3),
            ..cell()
        };
        // Six bytes at offset 2 fits a stride of 8; eight bytes does not.
        let refused = attribute(0, 0, VertexFormat::Short3, 2, 8, narrow.formats, || {
            ShaderInput::Channels(3)
        })
        .expect_err("2 + 8 > 8");
        assert_eq!(
            refused,
            Refusal::WidenPastStride {
                guest: VertexFormat::Short3,
                offset: 2,
                widened_bytes: 8,
                stride: 8,
            }
        );
        // And the unwidened attribute at the same offset would have fit, so
        // the refusal is about the substitution rather than the declaration.
        assert!(2 + VertexFormat::Short3.bytes() <= 8);
        // One more byte of stride and it fits.
        assert!(
            attribute(0, 0, VertexFormat::Short3, 2, 10, narrow.formats, || {
                ShaderInput::Channels(3)
            })
            .is_ok()
        );
    }

    /// **A substitution that cannot fit is refused without walking a module.**
    ///
    /// The stride comparison is three integers the caller already holds; the
    /// shader is a walk of a whole SPIR-V module. An attribute whose substitute
    /// runs past the end of a vertex is refused whatever the shader declares,
    /// so consulting it first would buy a different name for the same refusal
    /// at the cost of the walk.
    #[test]
    fn a_substitution_that_cannot_fit_never_consults_the_shader() {
        let narrow = VertexFormatSupport::all().without(VertexFormat::Short3);
        let refused =
            attribute(0, 0, VertexFormat::Short3, 2, 8, narrow, unreached).expect_err("2 + 8 > 8");
        assert_eq!(
            refused,
            Refusal::WidenPastStride {
                guest: VertexFormat::Short3,
                offset: 2,
                widened_bytes: 8,
                stride: 8,
            },
            "and it refuses for the reason that is actually true"
        );
    }

    /// **Every optional three-channel format has a substitute this rail will
    /// actually make**, and the mandatory ones never reach the substitution.
    ///
    /// The nine are the whole of what Vulkan leaves optional at three channels:
    /// the 8- and 16-bit ones. Asserted through [`attribute`] rather than
    /// through [`VertexFormat::widened`], because the sibling relation is
    /// geometry and hands *every* format below four channels a sibling — what
    /// this claims is that the plan comes back widened, which is a different
    /// statement.
    #[test]
    fn every_optional_three_channel_format_widens_and_the_mandatory_ones_do_not() {
        use VertexFormat as F;
        let optional = [
            F::UChar3,
            F::Char3,
            F::UChar3Normalized,
            F::Char3Normalized,
            F::UShort3,
            F::Short3,
            F::UShort3Normalized,
            F::Short3Normalized,
            F::Half3,
        ];
        assert_eq!(optional.len(), 9);
        for guest in optional {
            let narrow = VertexFormatSupport::all().without(guest);
            let plan = attribute(0, 0, guest, 0, 64, narrow, || ShaderInput::Channels(3))
                .unwrap_or_else(|r| panic!("{guest:?} must widen, got {r}"));
            assert_eq!(plan.widened_from, Some(guest), "{guest:?}");
            assert_eq!(
                plan.format,
                format(guest.widened().expect("a wider sibling")),
                "{guest:?}"
            );
        }
        // The 32-bit three-channel formats are mandatory, so a device that
        // reports them never reaches the substitution at all.
        for guest in [F::Float3, F::Int3, F::UInt3] {
            assert!(guest.widened().is_some(), "{guest:?}");
            let plan = attribute(0, 0, guest, 0, 64, VertexFormatSupport::all(), unreached)
                .expect("mandatory");
            assert_eq!(plan.widened_from, None, "{guest:?}");
        }
    }

    /// **A shader may not read past the channels the guest's own format
    /// supplies, whatever that count is.**
    ///
    /// The comparison is against the declared format's channel count and not
    /// the literal four. Written as four it would be right only while the
    /// sibling lookup was itself restricted to three-channel formats: a
    /// one-channel format widened under that rule lets a shader reading two
    /// channels see a real second component where Vulkan had been defaulting a
    /// zero — no refusal, no log line, a wrong vertex stream.
    #[test]
    fn the_channel_comparison_is_against_the_declared_format_and_not_the_literal_four() {
        use VertexFormat as F;
        for (guest, channels) in [(F::Half, 1u32), (F::Half2, 2), (F::Half3, 3)] {
            assert_eq!(guest.components(), channels, "{guest:?}");
            let narrow = VertexFormatSupport::all().without(guest);
            let planned = attribute(0, 0, guest, 0, 64, narrow, || ShaderInput::Channels(2));
            if channels >= 2 {
                assert!(
                    planned.is_ok_and(|p| p.widened_from.is_some()),
                    "{guest:?} supplies {channels} and the shader reads two"
                );
            } else {
                assert_eq!(
                    planned.expect_err("one channel, two read"),
                    Refusal::WidenReadAsFour { guest },
                    "{guest:?} supplies one channel and the shader reads two"
                );
            }
        }
    }

    /// A measured support set covers every format an attribute can resolve to,
    /// substitutes included.
    ///
    /// [`VertexFormatSupport::measured`] asks about the enumeration, and a
    /// substitute is a member of it, so nothing an attribute can be bound as
    /// goes unqueried. Stated as a test because the alternative — a probe
    /// driven by a hand-written list — is what this replaced.
    #[test]
    fn the_measured_set_covers_every_format_an_attribute_can_resolve_to() {
        let mut asked: BTreeSet<i32> = BTreeSet::new();
        let _ = VertexFormatSupport::measured(|f| {
            asked.insert(f.as_raw());
            true
        });
        for guest in VertexFormat::ALL {
            assert!(asked.contains(&format(guest).as_raw()), "{guest:?}");
            if let Some(wider) = guest.widened() {
                assert!(
                    asked.contains(&format(wider).as_raw()),
                    "{guest:?} substitute went unqueried"
                );
            }
        }
        assert_eq!(VertexFormat::ALL.len(), 53);
        assert!(asked.len() >= 30, "queried {}", asked.len());
    }

    #[test]
    fn a_four_channel_or_packed_format_has_nothing_to_widen_to() {
        for guest in [VertexFormat::Short4, VertexFormat::FloatRgb9E5] {
            let narrow = VertexCell {
                formats: VertexFormatSupport::all().without(guest),
                ..cell()
            };
            let refused = attribute(0, 0, guest, 0, 64, narrow.formats, || {
                ShaderInput::Channels(1)
            })
            .expect_err("nothing wider exists");
            assert_eq!(refused, Refusal::NoFormat { guest });
        }
    }

    #[test]
    fn the_support_set_is_keyed_by_ordinal_and_holds_every_format() {
        let all = VertexFormatSupport::all();
        assert_eq!(all.count(), VertexFormat::ALL.len() as u32);
        for guest in VertexFormat::ALL {
            assert!(all.has(guest));
            assert!(!all.without(guest).has(guest));
            assert!(VertexFormatSupport::NONE.with(guest).has(guest));
            // Removing one leaves every other alone.
            let minus = all.without(guest);
            assert_eq!(minus.count(), all.count() - 1);
        }
        assert_eq!(VertexFormatSupport::NONE.count(), 0);
        // Measured from a predicate over Vulkan formats: a device that
        // declines all three-channel 16-bit formats declines exactly the guest
        // formats that use them.
        let measured = VertexFormatSupport::measured(|f| f != vk::Format::R16G16B16_UINT);
        assert!(!measured.has(VertexFormat::Short3));
        assert!(!measured.has(VertexFormat::UShort3));
        assert!(measured.has(VertexFormat::Short4));
        assert_eq!(measured.count(), all.count() - 2);
    }

    #[test]
    fn per_vertex_is_the_vertex_rate_with_a_divisor_of_one() {
        let plan = binding(0, 32, StepFunction::PerVertex, 1, cell()).expect("legal");
        assert_eq!(plan.input_rate, vk::VertexInputRate::VERTEX);
        assert_eq!(plan.divisor, 1);
        assert!(!plan.needs_divisor_structure());
        assert_eq!(plan.native().stride, 32);
        assert_eq!(plan.native().input_rate, vk::VertexInputRate::VERTEX);

        // A per-vertex layout ignores the rate entirely, which is why the
        // rate's contract with the step function is checked elsewhere.
        assert_eq!(
            binding(0, 32, StepFunction::PerVertex, 7, cell()).expect("legal"),
            plan
        );
    }

    /// Constant is an instance rate that never advances, and the zero divisor
    /// that spells it is its own capability.
    #[test]
    fn constant_is_the_instance_rate_with_a_zero_divisor() {
        let plan = binding(2, 16, StepFunction::Constant, 0, cell()).expect("legal");
        assert_eq!(plan.input_rate, vk::VertexInputRate::INSTANCE);
        assert_eq!(plan.divisor, 0);
        assert!(plan.needs_divisor_structure());

        let without = VertexCell {
            zero_divisor: false,
            ..cell()
        };
        assert_eq!(
            binding(2, 16, StepFunction::Constant, 0, without),
            Err(Refusal::NoZeroDivisor)
        );
        // And a host without the zero divisor can still take a plain instanced
        // binding, so the two capabilities really are separate.
        assert!(binding(2, 16, StepFunction::PerInstance, 1, without).is_ok());
    }

    #[test]
    fn an_instance_divisor_above_one_needs_a_feature_and_stays_under_the_limit() {
        let plan = binding(1, 8, StepFunction::PerInstance, 4, cell()).expect("legal");
        assert_eq!(plan.input_rate, vk::VertexInputRate::INSTANCE);
        assert_eq!(plan.divisor, 4);
        assert!(plan.needs_divisor_structure());

        // A divisor of one needs nothing, which is why the gate is `!= 1`.
        let bare = VertexCell {
            instance_rate_divisor: false,
            max_divisor: 0,
            ..cell()
        };
        let one = binding(1, 8, StepFunction::PerInstance, 1, bare).expect("legal");
        assert_eq!(one.divisor, 1);
        assert!(!one.needs_divisor_structure());

        assert_eq!(
            binding(1, 8, StepFunction::PerInstance, 4, bare),
            Err(Refusal::NoInstanceDivisor { divisor: 4 })
        );
        // Reported but over the limit is a different refusal from not
        // reported at all.
        let limited = VertexCell {
            max_divisor: 2,
            ..cell()
        };
        assert_eq!(
            binding(1, 8, StepFunction::PerInstance, 4, limited),
            Err(Refusal::DivisorTooLarge {
                divisor: 4,
                limit: 2
            })
        );
        assert!(binding(1, 8, StepFunction::PerInstance, 2, limited).is_ok());
    }

    /// Recognised and declined, rather than unknown.
    #[test]
    fn a_tessellation_step_function_is_declined_for_what_it_is() {
        for step in [StepFunction::PerPatch, StepFunction::PerPatchControlPoint] {
            assert_eq!(input_rate(step), Err(Refusal::TessellationStep { step }));
            assert_eq!(
                binding(0, 4, step, 1, cell()),
                Err(Refusal::TessellationStep { step })
            );
        }
        // The three that are not tessellation all have a rate.
        for step in [
            StepFunction::Constant,
            StepFunction::PerVertex,
            StepFunction::PerInstance,
        ] {
            assert!(input_rate(step).is_ok());
        }
    }

    /// Every refusal has its own slug, or a report cannot say which happened.
    #[test]
    fn no_two_refusals_share_a_slug() {
        let slugs: BTreeSet<&str> = [
            Refusal::NoFormat {
                guest: VertexFormat::Short3,
            },
            Refusal::WidenReadAsFour {
                guest: VertexFormat::Short3,
            },
            Refusal::WidenShaderUnreadable {
                guest: VertexFormat::Short3,
            },
            Refusal::WidenPastStride {
                guest: VertexFormat::Short3,
                offset: 0,
                widened_bytes: 8,
                stride: 4,
            },
            Refusal::TessellationStep {
                step: StepFunction::PerPatch,
            },
            Refusal::NoZeroDivisor,
            Refusal::NoInstanceDivisor { divisor: 2 },
            Refusal::DivisorTooLarge {
                divisor: 2,
                limit: 1,
            },
        ]
        .iter()
        .map(|r| r.slug())
        .collect();
        assert_eq!(slugs.len(), 8);
    }

    /// Bytes per element read out of the Vulkan format's own name, which is a
    /// derivation this crate shares nothing with.
    ///
    /// `R8G8B8A8_UINT` is four eight-bit channels however it got chosen, and a
    /// `_PACK32` format is four bytes by definition. So the width the table
    /// implies can be recovered from the answer alone, with no reference to
    /// the guest ordinal that produced it.
    fn width_from_vulkan_name(f: vk::Format) -> Option<u32> {
        let name = format!("{f:?}");
        if let Some(at) = name.find("PACK") {
            return name[at + 4..].parse::<u32>().ok().map(|bits| bits / 8);
        }
        let mut bits = 0u32;
        let mut digits = String::new();
        for ch in name.split('_').next()?.chars() {
            if ch.is_ascii_digit() {
                digits.push(ch);
            } else if !digits.is_empty() {
                bits += digits.parse::<u32>().ok()?;
                digits.clear();
            }
        }
        if !digits.is_empty() {
            bits += digits.parse::<u32>().ok()?;
        }
        (bits > 0).then_some(bits / 8)
    }

    /// Every arm of the fifty-three-arm table fetches as many bytes as the
    /// contract says the guest format occupies.
    ///
    /// This is the failure a hand-written mapping table has: not an unknown
    /// format, which refuses, but a *plausible neighbour* --- `Half3` bound to
    /// `R16G16_SFLOAT`, `Short4` to `R16G16B16_UINT`. The pipeline builds, the
    /// draw runs, and every vertex after the first reads from the wrong offset,
    /// because the attribute's stride advance and the format's fetch width
    /// stopped agreeing. `attribute`'s own bounds check cannot see it: it
    /// checks the *widened* width against the stride, and a narrower wrong
    /// answer fits.
    ///
    /// The two sides are independent. One is `VertexFormat::bytes`, from the
    /// decode contract. The other is read out of the `VkFormat`'s name, so a
    /// wrong arm has to be wrong in both places by the same amount to survive.
    #[test]
    fn every_arm_fetches_the_width_the_contract_declares() {
        for guest in VertexFormat::ALL {
            let native = format(guest);
            assert_eq!(
                width_from_vulkan_name(native),
                Some(guest.bytes()),
                "{guest:?} is {} bytes by the contract and {native:?} on this host",
                guest.bytes()
            );
        }
    }

    /// `SIGNED_AS_UNSIGNED` is exactly the arms that are signed on one side and
    /// unsigned on the other.
    ///
    /// The list exists so the coupling to the shader translator has a blast
    /// radius somebody can read, and a list that is merely *near* the truth is
    /// worse than none: an arm missing from it is a coupling nobody knows
    /// about, and an arm listed that is not one sends a reader looking for a
    /// dependency that is not there. Both halves are derived --- signedness
    /// from the guest format's own name, unsignedness from the Vulkan
    /// format's --- so the list is checked rather than trusted.
    #[test]
    fn the_signed_as_unsigned_list_is_exactly_the_arms_that_are() {
        for guest in VertexFormat::ALL {
            let guest_name = format!("{guest:?}");
            let signed_integer = (guest_name.starts_with("Char")
                || guest_name.starts_with("Short")
                || guest_name.starts_with("Int"))
                && !guest_name.contains("Normalized");
            let native_unsigned = format!("{:?}", format(guest)).ends_with("_UINT");
            assert_eq!(
                SIGNED_AS_UNSIGNED.contains(&guest),
                signed_integer && native_unsigned,
                "{guest:?} maps to {:?}",
                format(guest)
            );
        }
    }
}
