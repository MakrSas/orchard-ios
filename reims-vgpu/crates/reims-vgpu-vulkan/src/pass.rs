//! What a render-pass descriptor becomes here: the attachments, their load and
//! store operations, and the clear values read the way the format says.
//!
//! # A clear colour is four doubles and a union member
//!
//! `MTLClearColor` has one form — four `double`s — and Metal interprets them
//! by the attachment's format. Vulkan does not. A clear written into the float
//! member of `VkClearColorValue` for an integer attachment is read as a bit
//! pattern, and the attachment clears to garbage no later stage can attribute
//! to it. So the member is chosen from the guest's format code, and the
//! conversion is a value conversion rather than a transmute:
//! `4.0` clears an integer attachment to four, not to `0x40800000`.
//!
//! # Depth and stencil are two slots to the guest and one attachment here
//!
//! `MTLRenderPassDescriptor` has independent depth and stencil attachments.
//! Vulkan has one depth-stencil attachment with two pairs of operations. When
//! the guest attaches both, they must be the same texture — a combined
//! `D24_UNORM_S8_UINT` image is one image — and a descriptor naming two
//! different textures is refused rather than resolved to one of them. Metal
//! permits the two to be separate images for separate formats; this rail
//! cannot express that and says so instead of dropping one.
//!
//! # A resolve is a store action, and it needs somewhere to resolve to
//!
//! `MultisampleResolve` and `StoreAndMultisampleResolve` differ in exactly one
//! thing: whether the multisample attachment itself is also stored. Both
//! resolve, and both need a resolve target. A descriptor that asks to resolve
//! and names no target is refused, because the alternative is a pass that runs
//! and silently produces nothing where the guest expected the resolved image.
//!
//! # Which mechanism carries the pass
//!
//! Vulkan 1.2 is the baseline, so dynamic rendering is an extension there and
//! core from 1.3. [`select`] is the same shape as
//! [`crate::descriptor::select`]: a capability ladder whose bottom rung —
//! `VkRenderPass` and `VkFramebuffer` — is Vulkan 1.0 and therefore always
//! available, so the choice is total.
//!
//! # Planned, not recorded
//!
//! Nothing here begins a pass. The plan is a value, and the attachment order,
//! the operations, the clear members and every refusal are tested with no GPU.

use ash::vk;
use reims_vgpu_core::identity::ResourceId;
use reims_vgpu_core::pass::{
    Attachment, AttachmentSlot, LoadAction, PassDescriptor, StoreAction, COLOR_ATTACHMENTS,
};
use reims_vgpu_core::pixel_format::{clear_class, ClearClass};

/// Whether this host has dynamic rendering.
///
/// Core from Vulkan 1.3 and `VK_KHR_dynamic_rendering` below it; the census is
/// what joins the two.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassCell {
    pub dynamic_rendering: bool,
}

/// An operator switch, which may narrow what this rail uses and never widen
/// it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Narrowing {
    /// Force the render-pass object path even where dynamic rendering exists.
    /// The one way to compare the two on one host.
    pub no_dynamic_rendering: bool,
}

impl Narrowing {
    /// The veto an operator set.
    ///
    /// [`reims_vgpu_config::DYNAMIC_RENDERING`] is narrowing-only for the
    /// reason every switch here is: `on` would have to widen to mean anything,
    /// and there is nothing above the rung it names.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            no_dynamic_rendering: reims_vgpu_config::switch(reims_vgpu_config::DYNAMIC_RENDERING)
                == reims_vgpu_config::Switch::Off,
        }
    }
}

/// Which mechanism carries a pass here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Carrier {
    /// `vkCmdBeginRendering`. No render-pass object, no framebuffer, and no
    /// compatibility rules between a pipeline and the pass it was made for.
    DynamicRendering,
    /// `VkRenderPass` and `VkFramebuffer`. Vulkan 1.0, so this rung is always
    /// available and the ladder is total.
    RenderPassObject,
}

impl Carrier {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DynamicRendering => "dynamic_rendering",
            Self::RenderPassObject => "render_pass_object",
        }
    }
}

/// Why the higher rung was not taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Declined {
    /// Neither core here nor enumerated.
    Unsupported,
    /// An operator switch narrowed it away.
    Narrowed,
}

impl Declined {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Narrowed => "narrowed",
        }
    }
}

/// The carrier this host uses, and why the other one was not taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub carrier: Carrier,
    /// `None` when dynamic rendering was taken.
    pub dynamic_rendering: Option<Declined>,
}

/// Choose the carrier. Total: the bottom rung needs nothing.
#[must_use]
pub const fn select(cell: PassCell, narrowing: Narrowing) -> Selection {
    if !cell.dynamic_rendering {
        return Selection {
            carrier: Carrier::RenderPassObject,
            dynamic_rendering: Some(Declined::Unsupported),
        };
    }
    if narrowing.no_dynamic_rendering {
        return Selection {
            carrier: Carrier::RenderPassObject,
            dynamic_rendering: Some(Declined::Narrowed),
        };
    }
    Selection {
        carrier: Carrier::DynamicRendering,
        dynamic_rendering: None,
    }
}

/// Why a pass descriptor cannot be expressed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No slot is attached. A pass with no attachments renders nowhere.
    NoAttachments,
    /// The guest attached depth and stencil to different textures. One Vulkan
    /// attachment cannot be two images.
    SplitDepthStencil {
        depth: ResourceId,
        stencil: ResourceId,
    },
    /// A store action that resolves, with no resolve target named.
    ResolveWithoutTarget { slot: AttachmentSlot },
    /// The guest attached depth and stencil to one texture but disagreed about
    /// which part of it, or about where it resolves.
    ///
    /// One Vulkan depth-stencil attachment has one subresource and one resolve
    /// target, so there is nowhere to put a second answer. Taking the depth
    /// slot's and dropping the stencil slot's runs the pass and silently writes
    /// nothing where the guest expected the stencil's resolve --- which is the
    /// failure [`Self::ResolveWithoutTarget`] exists to prevent, reached from
    /// the other side. `field` is the one the two slots differ on, so a report
    /// says whether the guest asked for two subresources or two resolves.
    DepthStencilDisagree { field: &'static str },
    /// The depth slot clears to a value no depth buffer can hold.
    ///
    /// `VkClearDepthStencilValue::depth` is an `f32` and Vulkan requires it in
    /// `[0, 1]`; a NaN or an infinity is neither, and it is what every
    /// fragment's depth test in the pass is then against. The colour side
    /// cannot reach this --- a colour clear is read through its format, and
    /// every member of `VkClearColorValue` is a value conversion that
    /// saturates. A depth clear is one number and this is its only door.
    ///
    /// The *range* is deliberately not narrowed here, for the reason
    /// [`crate::raster::viewport`] gives about a depth range: both models use
    /// `[0, 1]` and refusing on the boundary would lose passes that work.
    NonFiniteDepthClear { value: u64 },
    /// The render-target extent does not fit the 32-bit fields a pass carries.
    ExtentTooLarge { axis: &'static str, value: u64 },
    /// The render target has no extent at all.
    ZeroExtent { width: u64, height: u64 },
    /// One attachment named both an array slice and a depth plane.
    ///
    /// They are the same axis — [`Attachment::subresource`] says so and adds
    /// them for exactly that reason — and which of the two a texture has is
    /// the texture's property, so only one of them is ever non-zero. Both
    /// non-zero is a contradiction for every texture there is: a volume has no
    /// array slices and an array has no depth planes. Summing them anyway
    /// attaches a third slice, and the pass runs and renders into it.
    SliceAndDepthPlane {
        slot: AttachmentSlot,
        /// The attachment itself, or the target it resolves to.
        which: &'static str,
        slice: u16,
        depth_plane: u16,
    },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NoAttachments => "vk_pass_no_attachments",
            Self::SplitDepthStencil { .. } => "vk_pass_split_depth_stencil",
            Self::ResolveWithoutTarget { .. } => "vk_pass_resolve_without_target",
            Self::DepthStencilDisagree { .. } => "vk_pass_depth_stencil_disagree",
            Self::NonFiniteDepthClear { .. } => "vk_pass_non_finite_depth_clear",
            Self::ExtentTooLarge { .. } => "vk_pass_extent_too_large",
            Self::ZeroExtent { .. } => "vk_pass_zero_extent",
            Self::SliceAndDepthPlane { .. } => "vk_pass_slice_and_depth_plane",
        }
    }
}

fn slot_name(slot: AttachmentSlot) -> String {
    match slot {
        AttachmentSlot::Color(index) => format!("color{index}"),
        AttachmentSlot::Depth => "depth".to_string(),
        AttachmentSlot::Stencil => "stencil".to_string(),
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAttachments => f.write_str(self.slug()),
            Self::SplitDepthStencil { depth, stencil } => write!(
                f,
                "{} depth={} stencil={}",
                self.slug(),
                depth.slot.0,
                stencil.slot.0
            ),
            Self::ResolveWithoutTarget { slot } => {
                write!(f, "{} slot={}", self.slug(), slot_name(*slot))
            }
            Self::DepthStencilDisagree { field } => {
                write!(f, "{} field={field}", self.slug())
            }
            Self::NonFiniteDepthClear { value } => {
                write!(f, "{} value={value:#x}", self.slug())
            }
            Self::ExtentTooLarge { axis, value } => {
                write!(f, "{} axis={axis} value={value}", self.slug())
            }
            Self::ZeroExtent { width, height } => {
                write!(f, "{} width={width} height={height}", self.slug())
            }
            Self::SliceAndDepthPlane {
                slot,
                which,
                slice,
                depth_plane,
            } => write!(
                f,
                "{} slot={} which={which} slice={slice} depth_plane={depth_plane}",
                self.slug(),
                slot_name(*slot)
            ),
        }
    }
}

/// A clear colour in the member the format says to read.
///
/// Not `vk::ClearValue`, which is a union and therefore neither comparable nor
/// `Eq` — and a plan whose clears cannot be compared is one whose conversion
/// cannot be asserted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearColor {
    Float([f32; 4]),
    Uint([u32; 4]),
    Sint([i32; 4]),
}

impl ClearColor {
    /// The guest's four doubles, read the way `format` says.
    ///
    /// A value conversion and never a reinterpretation: a guest clearing an
    /// integer attachment to `4.0` means four.
    #[must_use]
    pub fn of(format: u16, components: [f64; 4]) -> Self {
        match clear_class(format) {
            ClearClass::Float => Self::Float(components.map(|c| c as f32)),
            ClearClass::Uint => Self::Uint(components.map(|c| c as u32)),
            ClearClass::Sint => Self::Sint(components.map(|c| c as i32)),
        }
    }

    #[must_use]
    pub const fn native(self) -> vk::ClearColorValue {
        match self {
            Self::Float(float32) => vk::ClearColorValue { float32 },
            Self::Uint(uint32) => vk::ClearColorValue { uint32 },
            Self::Sint(int32) => vk::ClearColorValue { int32 },
        }
    }
}

/// Where a multisample attachment resolves to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolve {
    pub texture: ResourceId,
    pub level: u32,
    pub slice: u32,
}

/// One attachment, translated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttachmentPlan {
    pub slot: AttachmentSlot,
    pub texture: ResourceId,
    pub level: u32,
    pub slice: u32,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    /// `None` unless the load operation is a clear. Carried only then, so a
    /// clear value on a `LOAD` attachment is not representable.
    pub clear: Option<ClearColor>,
    pub resolve: Option<Resolve>,
}

/// The depth-stencil attachment, which is one image and two pairs of
/// operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthStencilPlan {
    pub texture: ResourceId,
    pub level: u32,
    pub slice: u32,
    /// `None` when the guest attached only stencil.
    pub depth: Option<Ops>,
    /// `None` when the guest attached only depth.
    pub stencil: Option<Ops>,
    pub clear_depth: f32,
    pub clear_stencil: u32,
    pub resolve: Option<Resolve>,
}

/// One aspect's load and store operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ops {
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
}

/// A whole pass, translated.
#[derive(Clone, Debug, PartialEq)]
pub struct PassPlan {
    /// Attached colour slots, in slot order. Vulkan's colour attachment index
    /// is the position in this list, which is what a fragment output location
    /// refers to — a gap in the guest's slots therefore renumbers, and this is
    /// the one place that happens.
    pub color: Vec<AttachmentPlan>,
    pub depth_stencil: Option<DepthStencilPlan>,
    pub extent: vk::Extent2D,
    pub layers: u32,
}

/// The load operation an action becomes.
///
/// `DontCare` maps to `DONT_CARE` and not to `LOAD`: the guest said the prior
/// contents are not needed, and loading them anyway is bandwidth spent on
/// bytes the pass is about to overwrite.
#[must_use]
pub const fn load_op(action: LoadAction) -> vk::AttachmentLoadOp {
    match action {
        LoadAction::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        LoadAction::Load => vk::AttachmentLoadOp::LOAD,
        LoadAction::Clear => vk::AttachmentLoadOp::CLEAR,
    }
}

/// The store operation an action becomes.
///
/// The two resolving actions differ in exactly one thing — whether the
/// multisample attachment itself is also stored — so the resolve half is not
/// here. It is [`Resolve`], produced beside this.
#[must_use]
pub const fn store_op(action: StoreAction) -> vk::AttachmentStoreOp {
    match action {
        StoreAction::DontCare | StoreAction::MultisampleResolve => vk::AttachmentStoreOp::DONT_CARE,
        StoreAction::Store | StoreAction::StoreAndMultisampleResolve => {
            vk::AttachmentStoreOp::STORE
        }
    }
}

/// The one subresource an attachment names, and the one its resolve target
/// receives.
///
/// [`Attachment::subresource`] and [`Attachment::resolve_subresource`] already
/// derive these, and the model owns the rule they rest on: the array slice and
/// the depth plane are the same axis, and "only one of the fields is non-zero
/// for a given" texture. Read from there rather than added again here — a
/// second derivation of one decision is how two layers come to disagree about
/// which slice a pass renders into.
///
/// The rule is checked and not assumed, and it needs no texture to check: a
/// volume has no array slices and an array has no depth planes, so both
/// non-zero contradicts every texture there is. Summing them regardless names
/// a third slice, and a pass that renders into it runs.
fn subresource_of(attachment: &Attachment) -> Result<(u32, u32), Refusal> {
    if attachment.slice != 0 && attachment.depth_plane != 0 {
        return Err(Refusal::SliceAndDepthPlane {
            slot: attachment.slot,
            which: "attachment",
            slice: attachment.slice,
            depth_plane: attachment.depth_plane,
        });
    }
    let range = attachment.subresource();
    Ok((range.base_level, range.base_slice))
}

fn resolve_of(attachment: &Attachment) -> Result<Option<Resolve>, Refusal> {
    if !attachment.store.resolves() {
        return Ok(None);
    }
    let texture = attachment
        .resolve_texture
        .ok_or(Refusal::ResolveWithoutTarget {
            slot: attachment.slot,
        })?;
    if attachment.resolve_slice != 0 && attachment.resolve_depth_plane != 0 {
        return Err(Refusal::SliceAndDepthPlane {
            slot: attachment.slot,
            which: "resolve",
            slice: attachment.resolve_slice,
            depth_plane: attachment.resolve_depth_plane,
        });
    }
    let range = attachment.resolve_subresource();
    Ok(Some(Resolve {
        texture,
        level: range.base_level,
        slice: range.base_slice,
    }))
}

/// Translate a resolved pass descriptor.
///
/// `format_of` answers the guest pixel format of an attached texture, which is
/// what decides a clear colour's union member. It is a callback rather than a
/// residency borrow so this stays testable with no resources published, and
/// because a pass may be planned against textures a caller resolves however it
/// likes.
///
/// # Errors
///
/// [`Refusal`], with nothing partially translated.
pub fn plan(
    descriptor: &PassDescriptor,
    mut format_of: impl FnMut(ResourceId) -> u16,
) -> Result<PassPlan, Refusal> {
    let width = u32::try_from(descriptor.extent.width).map_err(|_| Refusal::ExtentTooLarge {
        axis: "width",
        value: descriptor.extent.width,
    })?;
    let height = u32::try_from(descriptor.extent.height).map_err(|_| Refusal::ExtentTooLarge {
        axis: "height",
        value: descriptor.extent.height,
    })?;
    if width == 0 || height == 0 {
        return Err(Refusal::ZeroExtent {
            width: descriptor.extent.width,
            height: descriptor.extent.height,
        });
    }
    let layers = u32::try_from(descriptor.extent.array_length.max(1)).map_err(|_| {
        Refusal::ExtentTooLarge {
            axis: "array_length",
            value: descriptor.extent.array_length,
        }
    })?;

    let mut color = Vec::with_capacity(COLOR_ATTACHMENTS);
    for attachment in &descriptor.color {
        let Some(texture) = attachment.texture else {
            continue;
        };
        let clear = (attachment.load == LoadAction::Clear)
            .then(|| {
                attachment
                    .clear_color()
                    .map(|components| ClearColor::of(format_of(texture), components))
            })
            .flatten();
        let (level, slice) = subresource_of(attachment)?;
        color.push(AttachmentPlan {
            slot: attachment.slot,
            texture,
            level,
            slice,
            load: load_op(attachment.load),
            store: store_op(attachment.store),
            clear,
            resolve: resolve_of(attachment)?,
        });
    }

    let depth_stencil = depth_stencil_of(descriptor)?;
    if color.is_empty() && depth_stencil.is_none() {
        return Err(Refusal::NoAttachments);
    }

    Ok(PassPlan {
        color,
        depth_stencil,
        extent: vk::Extent2D { width, height },
        layers,
    })
}

fn depth_stencil_of(descriptor: &PassDescriptor) -> Result<Option<DepthStencilPlan>, Refusal> {
    let depth = descriptor.depth.texture;
    let stencil = descriptor.stencil.texture;
    let texture = match (depth, stencil) {
        (None, None) => return Ok(None),
        (Some(depth), Some(stencil)) if depth != stencil => {
            return Err(Refusal::SplitDepthStencil { depth, stencil })
        }
        (Some(texture), _) | (None, Some(texture)) => texture,
    };

    // The two aspects of one image, so one level and one slice. Taken from
    // whichever slot the guest attached; when both are attached they must agree
    // about which part of the image they mean and about where it resolves,
    // because there is one place here to record either. A descriptor whose two
    // slots disagree is describing two attachments again, and it is refused
    // rather than resolved to the depth slot's answer --- the stencil half of
    // which would then be silently dropped.
    let carrier = if depth.is_some() {
        &descriptor.depth
    } else {
        &descriptor.stencil
    };
    if depth.is_some() && stencil.is_some() {
        let (from_depth, from_stencil) = (&descriptor.depth, &descriptor.stencil);
        for (field, agree) in [
            ("level", from_depth.level == from_stencil.level),
            ("slice", from_depth.slice == from_stencil.slice),
            (
                "depth_plane",
                from_depth.depth_plane == from_stencil.depth_plane,
            ),
            // Both are resolved before either is compared, so a stencil slot
            // that resolves to nowhere is that refusal rather than this one.
            (
                "resolve",
                resolve_of(from_depth)? == resolve_of(from_stencil)?,
            ),
        ] {
            if !agree {
                return Err(Refusal::DepthStencilDisagree { field });
            }
        }
    }

    // Read only where the load action clears, which is the rule the colour side
    // already states: the clear bits a `LOAD` attachment still carries are not
    // a clear, and carrying them onward would put a value the pass never uses
    // into a `VkClearValue` --- where, on the depth side, it can be a NaN.
    let clear_depth = if descriptor.depth.load == LoadAction::Clear {
        let value = descriptor.depth.clear_depth().unwrap_or(0.0);
        let narrowed = value as f32;
        if !narrowed.is_finite() {
            return Err(Refusal::NonFiniteDepthClear {
                value: value.to_bits(),
            });
        }
        narrowed
    } else {
        0.0
    };
    let clear_stencil = if descriptor.stencil.load == LoadAction::Clear {
        descriptor.stencil.clear_stencil().unwrap_or(0)
    } else {
        0
    };

    let (level, slice) = subresource_of(carrier)?;
    Ok(Some(DepthStencilPlan {
        texture,
        level,
        slice,
        depth: depth.map(|_| Ops {
            load: load_op(descriptor.depth.load),
            store: store_op(descriptor.depth.store),
        }),
        stencil: stencil.map(|_| Ops {
            load: load_op(descriptor.stencil.load),
            store: store_op(descriptor.stencil.store),
        }),
        // Depth clears are `[0, 1]` in both models, so this is a narrowing and
        // not a mapping. The stencil clear is the guest's low word.
        clear_depth,
        clear_stencil,
        resolve: resolve_of(carrier)?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::identity::{ObjectListRef, SlotGeneration};
    use reims_vgpu_core::pass::RenderTargetExtent;
    use reims_vgpu_core::pixel_format::{
        MTL_FORMAT_DEPTH32_FLOAT, MTL_FORMAT_RGBA8_SINT, MTL_FORMAT_RGBA8_UINT,
        MTL_FORMAT_RGBA8_UNORM,
    };
    use std::collections::BTreeSet;

    fn id(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn colour() -> impl FnMut(ResourceId) -> u16 {
        |_| MTL_FORMAT_RGBA8_UNORM
    }

    fn empty() -> PassDescriptor {
        let mut descriptor = PassDescriptor::empty();
        descriptor.extent = RenderTargetExtent {
            width: 128,
            height: 64,
            array_length: 1,
        };
        descriptor
    }

    fn attach(descriptor: &mut PassDescriptor, index: usize, texture: ResourceId) {
        descriptor.color[index].texture = Some(texture);
        descriptor.color[index].load = LoadAction::Load;
        descriptor.color[index].store = StoreAction::Store;
    }

    #[test]
    fn the_ladder_is_total_and_the_bottom_rung_needs_nothing() {
        let bare = select(PassCell::default(), Narrowing::default());
        assert_eq!(bare.carrier, Carrier::RenderPassObject);
        assert_eq!(bare.dynamic_rendering, Some(Declined::Unsupported));

        let modern = select(
            PassCell {
                dynamic_rendering: true,
            },
            Narrowing::default(),
        );
        assert_eq!(modern.carrier, Carrier::DynamicRendering);
        assert_eq!(modern.dynamic_rendering, None);
    }

    #[test]
    fn no_narrowing_can_widen_the_carrier() {
        for dynamic_rendering in [false, true] {
            for no_dynamic_rendering in [false, true] {
                let cell = PassCell { dynamic_rendering };
                let narrowed = select(
                    cell,
                    Narrowing {
                        no_dynamic_rendering,
                    },
                );
                let open = select(cell, Narrowing::default());
                // Never above what the unnarrowed answer was.
                if open.carrier == Carrier::RenderPassObject {
                    assert_eq!(narrowed.carrier, Carrier::RenderPassObject);
                }
                if no_dynamic_rendering {
                    assert_eq!(narrowed.carrier, Carrier::RenderPassObject);
                }
            }
        }
        // Non-vacuity: the switch does something on a host that has it.
        let cell = PassCell {
            dynamic_rendering: true,
        };
        assert_ne!(
            select(cell, Narrowing::default()).carrier,
            select(
                cell,
                Narrowing {
                    no_dynamic_rendering: true
                }
            )
            .carrier
        );
        assert_eq!(
            select(
                cell,
                Narrowing {
                    no_dynamic_rendering: true
                }
            )
            .dynamic_rendering,
            Some(Declined::Narrowed)
        );
    }

    /// The array slice and the depth plane are one axis, and the model says so
    /// by adding them. Both non-zero contradicts every texture there is — a
    /// volume has no array slices, an array has no depth planes — so the sum
    /// would name a third slice and the pass would render into it. Refused on
    /// the attachment and on its resolve target, which are two subresources
    /// and two chances to say it.
    #[test]
    fn an_attachment_that_names_both_a_slice_and_a_depth_plane_is_refused() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(10));
        descriptor.color[0].slice = 1;
        descriptor.color[0].depth_plane = 2;
        assert_eq!(
            plan(&descriptor, colour()),
            Err(Refusal::SliceAndDepthPlane {
                slot: AttachmentSlot::Color(0),
                which: "attachment",
                slice: 1,
                depth_plane: 2,
            })
        );

        // Either one alone is the same slice on the one axis, and legal.
        descriptor.color[0].depth_plane = 0;
        assert_eq!(
            plan(&descriptor, colour()).expect("plannable").color[0].slice,
            1
        );
        descriptor.color[0].slice = 0;
        descriptor.color[0].depth_plane = 2;
        assert_eq!(
            plan(&descriptor, colour()).expect("plannable").color[0].slice,
            2
        );

        // And the resolve target is checked in its own right.
        descriptor.color[0].depth_plane = 0;
        descriptor.color[0].store = StoreAction::StoreAndMultisampleResolve;
        descriptor.color[0].resolve_texture = Some(id(11));
        descriptor.color[0].resolve_slice = 3;
        descriptor.color[0].resolve_depth_plane = 4;
        assert_eq!(
            plan(&descriptor, colour()),
            Err(Refusal::SliceAndDepthPlane {
                slot: AttachmentSlot::Color(0),
                which: "resolve",
                slice: 3,
                depth_plane: 4,
            })
        );
    }

    /// One derivation, not two: the subresources this plan carries are the
    /// model's own answer for the same attachment.
    #[test]
    fn the_plans_subresources_are_the_models_own() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(10));
        descriptor.color[0].level = 3;
        descriptor.color[0].depth_plane = 5;
        descriptor.color[0].store = StoreAction::StoreAndMultisampleResolve;
        descriptor.color[0].resolve_texture = Some(id(11));
        descriptor.color[0].resolve_level = 2;
        descriptor.color[0].resolve_slice = 6;

        let plan = plan(&descriptor, colour()).expect("plannable");
        let model = descriptor.color[0].subresource();
        assert_eq!(plan.color[0].level, model.base_level);
        assert_eq!(plan.color[0].slice, model.base_slice);
        let resolve = plan.color[0].resolve.expect("a resolve target");
        let model = descriptor.color[0].resolve_subresource();
        assert_eq!(resolve.level, model.base_level);
        assert_eq!(resolve.slice, model.base_slice);
    }

    #[test]
    fn attached_colour_slots_renumber_to_their_position() {
        let mut descriptor = empty();
        // The guest attached slots two and five and left the rest empty. A
        // fragment output location refers to the *Vulkan* index, so these
        // become zero and one — the one place a renumber happens.
        attach(&mut descriptor, 2, id(10));
        attach(&mut descriptor, 5, id(11));

        let plan = plan(&descriptor, colour()).expect("plannable");
        assert_eq!(plan.color.len(), 2);
        assert_eq!(plan.color[0].texture, id(10));
        assert_eq!(plan.color[0].slot, AttachmentSlot::Color(2));
        assert_eq!(plan.color[1].texture, id(11));
        assert_eq!(plan.color[1].slot, AttachmentSlot::Color(5));
        assert_eq!(
            plan.extent,
            vk::Extent2D {
                width: 128,
                height: 64
            }
        );
        assert_eq!(plan.layers, 1);
    }

    #[test]
    fn each_load_and_store_action_becomes_its_own_operation() {
        for (action, expected) in [
            (LoadAction::DontCare, vk::AttachmentLoadOp::DONT_CARE),
            (LoadAction::Load, vk::AttachmentLoadOp::LOAD),
            (LoadAction::Clear, vk::AttachmentLoadOp::CLEAR),
        ] {
            assert_eq!(load_op(action), expected);
        }
        // The resolving pair differ only in whether the multisample
        // attachment is stored, which is what these two answers say.
        assert_eq!(
            store_op(StoreAction::MultisampleResolve),
            vk::AttachmentStoreOp::DONT_CARE
        );
        assert_eq!(
            store_op(StoreAction::StoreAndMultisampleResolve),
            vk::AttachmentStoreOp::STORE
        );
        assert_eq!(store_op(StoreAction::Store), vk::AttachmentStoreOp::STORE);
        assert_eq!(
            store_op(StoreAction::DontCare),
            vk::AttachmentStoreOp::DONT_CARE
        );
    }

    #[test]
    fn a_clear_colour_lands_in_the_member_the_format_names() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(1));
        descriptor.color[0].load = LoadAction::Clear;
        // Four, as a double. An integer attachment clears to four and not to
        // the bit pattern of 4.0.
        descriptor.color[0].clear_bits = [4.0f64.to_bits(); 4];

        let float = plan(&descriptor, |_| MTL_FORMAT_RGBA8_UNORM).expect("plannable");
        assert_eq!(float.color[0].clear, Some(ClearColor::Float([4.0; 4])));

        let unsigned = plan(&descriptor, |_| MTL_FORMAT_RGBA8_UINT).expect("plannable");
        assert_eq!(unsigned.color[0].clear, Some(ClearColor::Uint([4; 4])));

        let signed = plan(&descriptor, |_| MTL_FORMAT_RGBA8_SINT).expect("plannable");
        assert_eq!(signed.color[0].clear, Some(ClearColor::Sint([4; 4])));
    }

    #[test]
    fn a_clear_value_exists_only_where_the_load_action_clears() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(1));
        descriptor.color[0].clear_bits = [1.0f64.to_bits(); 4];
        // Loading, so the clear bits the record still carries are not a clear.
        let plan = plan(&descriptor, colour()).expect("plannable");
        assert_eq!(plan.color[0].clear, None);
        assert_eq!(plan.color[0].load, vk::AttachmentLoadOp::LOAD);
    }

    #[test]
    fn a_resolve_needs_a_target_and_refuses_without_one() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(1));
        descriptor.color[0].store = StoreAction::MultisampleResolve;

        assert_eq!(
            plan(&descriptor, colour()).err(),
            Some(Refusal::ResolveWithoutTarget {
                slot: AttachmentSlot::Color(0),
            })
        );

        descriptor.color[0].resolve_texture = Some(id(2));
        descriptor.color[0].resolve_level = 1;
        descriptor.color[0].resolve_slice = 3;
        let plan = plan(&descriptor, colour()).expect("plannable");
        assert_eq!(
            plan.color[0].resolve,
            Some(Resolve {
                texture: id(2),
                level: 1,
                slice: 3,
            })
        );
    }

    #[test]
    fn a_store_that_does_not_resolve_names_no_target_even_when_one_is_set() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(1));
        // The record carries a resolve texture and the action does not resolve
        // — which is an ordinary state for a descriptor the guest reuses.
        descriptor.color[0].resolve_texture = Some(id(2));
        let plan = plan(&descriptor, colour()).expect("plannable");
        assert_eq!(plan.color[0].resolve, None);
    }

    #[test]
    fn depth_and_stencil_on_one_texture_are_one_attachment_with_two_ops() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(1));
        descriptor.depth.texture = Some(id(9));
        descriptor.depth.load = LoadAction::Clear;
        descriptor.depth.store = StoreAction::Store;
        descriptor.depth.clear_bits = [0.5f64.to_bits(), 0, 0, 0];
        descriptor.stencil.texture = Some(id(9));
        descriptor.stencil.load = LoadAction::DontCare;
        descriptor.stencil.store = StoreAction::DontCare;
        descriptor.stencil.clear_bits = [7, 0, 0, 0];

        let plan = plan(&descriptor, |_| MTL_FORMAT_DEPTH32_FLOAT).expect("plannable");
        let ds = plan.depth_stencil.expect("a depth-stencil attachment");
        assert_eq!(ds.texture, id(9));
        assert_eq!(
            ds.depth,
            Some(Ops {
                load: vk::AttachmentLoadOp::CLEAR,
                store: vk::AttachmentStoreOp::STORE,
            })
        );
        assert_eq!(
            ds.stencil,
            Some(Ops {
                load: vk::AttachmentLoadOp::DONT_CARE,
                store: vk::AttachmentStoreOp::DONT_CARE,
            })
        );
        assert!((ds.clear_depth - 0.5).abs() < f32::EPSILON);
        // The stencil slot does not clear, so the bits it still carries are
        // not a clear --- the same rule the colour side states, and the reason
        // a `LOAD` attachment cannot carry a value the pass never uses into a
        // `VkClearValue`.
        assert_eq!(ds.clear_stencil, 0);
    }

    /// The stencil clear is the guest's low word, asserted where the slot
    /// actually clears.
    #[test]
    fn a_clearing_stencil_slot_takes_the_low_word_of_its_bits() {
        let mut descriptor = empty();
        descriptor.stencil.texture = Some(id(9));
        descriptor.stencil.load = LoadAction::Clear;
        descriptor.stencil.store = StoreAction::Store;
        descriptor.stencil.clear_bits = [0x1_0000_0007, 0, 0, 0];
        let plan = plan(&descriptor, |_| MTL_FORMAT_DEPTH32_FLOAT).expect("plannable");
        assert_eq!(
            plan.depth_stencil.expect("attached").clear_stencil,
            7,
            "the low word, not the whole 64-bit value"
        );
    }

    /// A depth clear that is not a number reaches `VkClearDepthStencilValue`
    /// as one, and every fragment's depth test in the pass is against it.
    #[test]
    fn a_depth_clear_that_is_not_a_number_refuses() {
        for bits in [
            f64::NAN.to_bits(),
            f64::INFINITY.to_bits(),
            1.0e300f64.to_bits(),
        ] {
            let mut descriptor = empty();
            descriptor.depth.texture = Some(id(9));
            descriptor.depth.load = LoadAction::Clear;
            descriptor.depth.clear_bits = [bits, 0, 0, 0];
            assert_eq!(
                plan(&descriptor, |_| MTL_FORMAT_DEPTH32_FLOAT).err(),
                Some(Refusal::NonFiniteDepthClear { value: bits }),
                "{bits:#x}"
            );

            // The same bits on a slot that does not clear are not a clear at
            // all, so there is nothing to refuse.
            descriptor.depth.load = LoadAction::Load;
            let planned = plan(&descriptor, |_| MTL_FORMAT_DEPTH32_FLOAT).expect("plannable");
            assert_eq!(planned.depth_stencil.expect("attached").clear_depth, 0.0);
        }
    }

    #[test]
    fn one_aspect_attached_alone_leaves_the_other_absent() {
        let mut descriptor = empty();
        descriptor.depth.texture = Some(id(9));
        descriptor.depth.load = LoadAction::Clear;
        let depth_only = plan(&descriptor, |_| MTL_FORMAT_DEPTH32_FLOAT).expect("plannable");
        let ds = depth_only.depth_stencil.expect("depth");
        assert!(ds.depth.is_some());
        assert_eq!(ds.stencil, None);

        let mut descriptor = empty();
        descriptor.stencil.texture = Some(id(9));
        descriptor.stencil.level = 2;
        let stencil_only = plan(&descriptor, |_| MTL_FORMAT_DEPTH32_FLOAT).expect("plannable");
        let ds = stencil_only.depth_stencil.expect("stencil");
        assert_eq!(ds.depth, None);
        assert!(ds.stencil.is_some());
        // The level comes from whichever slot is attached, not from the empty
        // one.
        assert_eq!(ds.level, 2);
    }

    #[test]
    fn depth_and_stencil_on_two_textures_refuse_rather_than_picking_one() {
        let mut descriptor = empty();
        descriptor.depth.texture = Some(id(9));
        descriptor.stencil.texture = Some(id(10));
        assert_eq!(
            plan(&descriptor, colour()).err(),
            Some(Refusal::SplitDepthStencil {
                depth: id(9),
                stencil: id(10),
            })
        );
    }

    #[test]
    fn a_depth_only_pass_is_a_pass() {
        let mut descriptor = empty();
        descriptor.depth.texture = Some(id(9));
        let plan = plan(&descriptor, |_| MTL_FORMAT_DEPTH32_FLOAT).expect("plannable");
        assert!(plan.color.is_empty());
        assert!(plan.depth_stencil.is_some());
    }

    #[test]
    fn a_pass_with_nothing_attached_refuses() {
        assert_eq!(plan(&empty(), colour()).err(), Some(Refusal::NoAttachments));
    }

    #[test]
    fn an_extent_of_no_texels_refuses_rather_than_rendering_nowhere() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(1));
        descriptor.extent.height = 0;
        assert_eq!(
            plan(&descriptor, colour()).err(),
            Some(Refusal::ZeroExtent {
                width: 128,
                height: 0,
            })
        );
    }

    #[test]
    fn an_extent_wider_than_a_native_field_refuses_rather_than_truncating() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(1));
        descriptor.extent.width = u64::from(u32::MAX) + 1;
        assert_eq!(
            plan(&descriptor, colour()).err(),
            Some(Refusal::ExtentTooLarge {
                axis: "width",
                value: u64::from(u32::MAX) + 1,
            })
        );
    }

    #[test]
    fn a_layerless_render_target_still_has_one_layer() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(1));
        // A descriptor that never set an array length is one layer, not zero:
        // a pass with no layers renders nothing.
        descriptor.extent.array_length = 0;
        assert_eq!(plan(&descriptor, colour()).expect("plannable").layers, 1);
        descriptor.extent.array_length = 6;
        assert_eq!(plan(&descriptor, colour()).expect("plannable").layers, 6);
    }

    #[test]
    fn a_depth_plane_and_an_array_slice_are_the_same_axis() {
        let mut descriptor = empty();
        attach(&mut descriptor, 0, id(1));
        descriptor.color[0].depth_plane = 3;
        assert_eq!(
            plan(&descriptor, colour()).expect("plannable").color[0].slice,
            3
        );
        descriptor.color[0].depth_plane = 0;
        descriptor.color[0].slice = 3;
        assert_eq!(
            plan(&descriptor, colour()).expect("plannable").color[0].slice,
            3
        );
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::NoAttachments,
            Refusal::SplitDepthStencil {
                depth: id(1),
                stencil: id(2),
            },
            Refusal::ResolveWithoutTarget {
                slot: AttachmentSlot::Depth,
            },
            Refusal::ExtentTooLarge {
                axis: "width",
                value: 1,
            },
            Refusal::ZeroExtent {
                width: 0,
                height: 0,
            },
            Refusal::DepthStencilDisagree { field: "level" },
            Refusal::NonFiniteDepthClear { value: 1 },
            Refusal::SliceAndDepthPlane {
                slot: AttachmentSlot::Depth,
                which: "attachment",
                slice: 1,
                depth_plane: 2,
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_pass_"));
        }
        assert!(Refusal::ResolveWithoutTarget {
            slot: AttachmentSlot::Color(3)
        }
        .to_string()
        .contains("color3"));
    }

    /// The failure this exists to prevent: one Vulkan depth-stencil attachment
    /// has one subresource and one resolve target, and taking the depth slot's
    /// answers ran the pass with the stencil slot's silently dropped.
    #[test]
    fn depth_and_stencil_on_one_texture_must_agree_about_which_part_of_it() {
        let both = || {
            let mut descriptor = empty();
            for slot in [&mut descriptor.depth, &mut descriptor.stencil] {
                slot.texture = Some(id(9));
                slot.load = LoadAction::Load;
                slot.store = StoreAction::Store;
                slot.level = 2;
                slot.slice = 1;
                slot.depth_plane = 0;
            }
            descriptor
        };
        // Agreeing is the ordinary case and still plans.
        let agreed = plan(&both(), colour()).expect("plannable");
        let attachment = agreed.depth_stencil.expect("one attachment");
        assert_eq!((attachment.level, attachment.slice), (2, 1));
        assert!(attachment.depth.is_some() && attachment.stencil.is_some());

        for (field, break_it) in [
            (
                "level",
                (|d: &mut PassDescriptor| d.stencil.level = 3) as fn(&mut PassDescriptor),
            ),
            ("slice", |d: &mut PassDescriptor| d.stencil.slice = 0),
            ("depth_plane", |d: &mut PassDescriptor| {
                d.stencil.depth_plane = 1
            }),
        ] {
            let mut descriptor = both();
            break_it(&mut descriptor);
            assert_eq!(
                plan(&descriptor, colour()).err(),
                Some(Refusal::DepthStencilDisagree { field }),
            );
        }
    }

    /// A resolve named on only one of the two slots, or named differently on
    /// each, has nowhere to go. This is `ResolveWithoutTarget`'s failure
    /// reached from the other side: the pass runs and the guest's resolved
    /// stencil image is never written.
    #[test]
    fn depth_and_stencil_on_one_texture_must_agree_about_where_it_resolves() {
        let mut descriptor = empty();
        for slot in [&mut descriptor.depth, &mut descriptor.stencil] {
            slot.texture = Some(id(9));
            slot.load = LoadAction::Load;
            slot.store = StoreAction::Store;
        }
        // Only the stencil half resolves --- so the carrier, which is the depth
        // slot, answers `None` and the resolve disappears.
        descriptor.stencil.store = StoreAction::StoreAndMultisampleResolve;
        descriptor.stencil.resolve_texture = Some(id(10));
        assert_eq!(
            plan(&descriptor, colour()).err(),
            Some(Refusal::DepthStencilDisagree { field: "resolve" })
        );

        // Both resolve, to different places.
        descriptor.depth.store = StoreAction::StoreAndMultisampleResolve;
        descriptor.depth.resolve_texture = Some(id(11));
        assert_eq!(
            plan(&descriptor, colour()).err(),
            Some(Refusal::DepthStencilDisagree { field: "resolve" })
        );

        // Both resolve to the same place, which is what one attachment can
        // express.
        descriptor.depth.resolve_texture = Some(id(10));
        let planned = plan(&descriptor, colour()).expect("plannable");
        assert_eq!(
            planned.depth_stencil.expect("one attachment").resolve,
            Some(Resolve {
                texture: id(10),
                level: 0,
                slice: 0,
            })
        );

        // A stencil slot that resolves to nowhere is still its own refusal and
        // not a disagreement --- both sides are resolved before either is
        // compared.
        descriptor.stencil.resolve_texture = None;
        assert_eq!(
            plan(&descriptor, colour()).err(),
            Some(Refusal::ResolveWithoutTarget {
                slot: AttachmentSlot::Stencil,
            })
        );
    }
}
