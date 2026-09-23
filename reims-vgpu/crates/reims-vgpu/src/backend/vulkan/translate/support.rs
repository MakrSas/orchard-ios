//! The device-capability half of the translation boundary, which this crate no
//! longer implements.
//!
//! Translation answers what a Metal value *means*; capability answers what the
//! bound GPU can do with it. The second is
//! [`reims_vgpu_vulkan::vertex`]'s — it is the layer that turns host
//! capabilities into policy, and a vertex format the device declines is exactly
//! that. What is left here is the join: which support set this engine measured,
//! how this crate's shader reader spells what the rail asks for, and how the
//! rail's refusals are named in this crate's vocabulary.
//!
//! The widening substitute and the conditions under which it is invisible are
//! documented once, on [`reims_vgpu_vulkan::vertex`], and tested there.

use ash::vk;

use super::reason::TranslateReason;
use crate::backend::vulkan::engine::VertexAttributeFormat;
use crate::runtime::spirv_vertex_input::InputWidth;
use reims_vgpu_vulkan::vertex::{self, Refusal, ShaderInput, VertexFormatSupport};

/// What the pipeline should bind for one vertex attribute on this device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexBinding {
    pub format: vk::Format,
    /// `Some(original)` when the device declined the attribute's own format and
    /// the mandatory wider sibling was substituted. Callers report this so a
    /// widened pipeline is visible rather than assumed.
    pub widened_from: Option<vk::Format>,
}

/// Probe every format the rail can emit for a vertex attribute.
///
/// The enumeration is the rail's, so the probe cannot fall behind the mapping
/// table: a format that gains a variant is queried without anybody remembering
/// to add it here.
///
/// # Safety
///
/// `instance` and `pd` must be live. The call itself only reads properties.
pub fn probe(instance: &ash::Instance, pd: vk::PhysicalDevice) -> VertexFormatSupport {
    VertexFormatSupport::measured(|format| {
        // SAFETY: reading physical-device format properties requires only a
        // live instance and physical device, both of which the caller owns.
        let props = unsafe { instance.get_physical_device_format_properties(pd, format) };
        props
            .buffer_features
            .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
    })
}

/// How this crate's SPIR-V reader spells what the rail asks about.
///
/// Total, and a mapping rather than a re-export, because the reader is this
/// crate's and must stay nameable on a build with no Vulkan rail compiled in.
const fn declared(width: InputWidth) -> ShaderInput {
    match width {
        InputWidth::Components(n) => ShaderInput::Channels(n),
        InputWidth::Absent => ShaderInput::Absent,
        InputWidth::Unreadable => ShaderInput::Unreadable,
    }
}

/// The rail's refusals in this crate's vocabulary.
///
/// Total over the variants [`vertex::attribute`] can return. The two the rail
/// tells apart that this vocabulary does not — no sibling at all, and a
/// substitute that would read past the vertex — are both "the device will not
/// take this format", which is what the guest-visible reason has always said.
/// The rail's names survive in the fail-channel text; this is only what the
/// pipeline cache branches on.
fn declined(refusal: Refusal, native: vk::Format) -> TranslateReason {
    let raw = native.as_raw();
    match refusal {
        Refusal::NoFormat { .. } | Refusal::WidenPastStride { .. } => {
            TranslateReason::FormatNotVertexBuffer(raw)
        }
        Refusal::WidenReadAsFour { .. } => TranslateReason::VertexFormatWidenReadAsFour(raw),
        Refusal::WidenShaderUnreadable { .. } => {
            TranslateReason::VertexFormatWidenShaderUnreadable(raw)
        }
        // Not reachable from an attribute: the step function and divisor
        // questions belong to `vertex::binding`, which this join does not call.
        // Named rather than wildcarded so a variant this join starts being able
        // to see stops the build.
        Refusal::TessellationStep { .. }
        | Refusal::NoZeroDivisor
        | Refusal::NoInstanceDivisor { .. }
        | Refusal::DivisorTooLarge { .. } => TranslateReason::FormatNotVertexBuffer(raw),
    }
}

/// Resolve one attribute against what this device accepts.
///
/// `offset` and `stride` are the attribute's own placement in the vertex
/// buffer and decide whether the widening substitute is in bounds.
///
/// `shader_width` is called at most once, and only on the path that would
/// substitute a wider format and could actually place it — see
/// [`reims_vgpu_vulkan::vertex::attribute`]. It must answer for *this
/// attribute's* `Location`.
///
/// # Errors
///
/// The rail's refusal, named in this crate's vocabulary.
pub fn resolve(
    formats: VertexFormatSupport,
    format: VertexAttributeFormat,
    offset: u32,
    stride: u32,
    shader_width: impl FnOnce() -> InputWidth,
) -> Result<VertexBinding, TranslateReason> {
    // The location and binding numbers are the pipeline's, and this join does
    // not build the descriptor; zero is what the plan carries back unread.
    match vertex::attribute(0, 0, format, offset, stride, formats, || {
        declared(shader_width())
    }) {
        Ok(plan) => Ok(VertexBinding {
            format: plan.format,
            widened_from: plan.widened_from.map(vertex::format),
        }),
        Err(refusal) => Err(declined(refusal, vertex::format(format))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every refusal the rail can raise for an attribute has a name here.**
    ///
    /// `declined` is total by construction — the match names every variant —
    /// and this holds the mapping to the three reasons the pipeline cache
    /// branches on, so a rail refusal cannot start arriving as a reason that
    /// means something else.
    #[test]
    fn the_rails_attribute_refusals_keep_their_meaning_here() {
        let guest = VertexAttributeFormat::Short3;
        let native = vertex::format(guest);
        let raw = native.as_raw();
        for (refusal, expected) in [
            (
                Refusal::NoFormat { guest },
                TranslateReason::FormatNotVertexBuffer(raw),
            ),
            (
                Refusal::WidenPastStride {
                    guest,
                    offset: 2,
                    widened_bytes: 8,
                    stride: 8,
                },
                TranslateReason::FormatNotVertexBuffer(raw),
            ),
            (
                Refusal::WidenReadAsFour { guest },
                TranslateReason::VertexFormatWidenReadAsFour(raw),
            ),
            (
                Refusal::WidenShaderUnreadable { guest },
                TranslateReason::VertexFormatWidenShaderUnreadable(raw),
            ),
        ] {
            assert_eq!(declined(refusal, native), expected, "{refusal}");
        }
    }

    /// The shader reader's three answers reach the rail as its three, and the
    /// component count travels rather than being recomputed.
    #[test]
    fn the_shader_readers_answers_map_one_for_one() {
        assert_eq!(declared(InputWidth::Absent), ShaderInput::Absent);
        assert_eq!(declared(InputWidth::Unreadable), ShaderInput::Unreadable);
        for n in 1..=4 {
            assert_eq!(
                declared(InputWidth::Components(n)),
                ShaderInput::Channels(n)
            );
        }
    }

    /// A host that accepts every format resolves to the attribute's own format
    /// and never reads a shader.
    #[test]
    fn a_permissive_device_widens_nothing() {
        for guest in VertexAttributeFormat::ALL {
            let binding = resolve(VertexFormatSupport::all(), guest, 0, 64, || {
                panic!("the shader was walked on a host that declines nothing")
            })
            .expect("a permissive device accepts every format");
            assert_eq!(binding.format, vertex::format(guest));
            assert_eq!(binding.widened_from, None);
        }
    }

    /// A declined format substitutes its wider sibling, and the binding says
    /// which format the guest actually asked for.
    #[test]
    fn a_declined_format_carries_the_one_it_was_substituted_for() {
        let guest = VertexAttributeFormat::Short3;
        let narrow = VertexFormatSupport::all().without(guest);
        let binding = resolve(narrow, guest, 0, 8, || InputWidth::Components(3))
            .expect("the wider sibling is mandatory");
        assert_eq!(
            binding.format,
            vertex::format(guest.widened().expect("Short3 widens"))
        );
        assert_eq!(binding.widened_from, Some(vertex::format(guest)));
    }
}
