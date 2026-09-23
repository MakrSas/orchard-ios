//! What both rungs of [`crate::pass::Carrier`] need built, and the two
//! different keys the objects they build are cached under.
//!
//! # A pipeline is compatible with a pass it was never built for
//!
//! Vulkan's render-pass compatibility rules say two passes are compatible when
//! their attachments agree on *format* and *sample count*. Load and store
//! operations and image layouts are explicitly not part of it. So a pipeline
//! built against a pass that clears is usable in one that loads, and the two
//! are different `VkRenderPass` objects.
//!
//! That is why there are two keys here and not one. [`Compatibility`] is what
//! a pipeline cache is keyed on; [`Signature`] is what a `VkRenderPass` cache
//! is keyed on, and it *contains* a `Compatibility` rather than restating it,
//! so the two cannot disagree about a format. Keying pipelines on the
//! signature would be correct and would recompile every pipeline in a frame
//! the first time a guest changed a load action from clear to load — which
//! guests do between the first pass of a frame and the rest of it.
//!
//! # The same key on both rungs
//!
//! On the dynamic-rendering rung there is no `VkRenderPass` at all: a pipeline
//! carries `VkPipelineRenderingCreateInfo`, which names the same formats and
//! the same sample count. So [`Compatibility`] is the pipeline cache key on
//! *both* rungs, and [`Compatibility::rendering_info`] is how it becomes the
//! one the dynamic rung wants. That is what makes [`crate::pass::select`] a
//! choice of rung rather than two pipelines caches with two key types.
//!
//! # One sample count for the whole pass
//!
//! Metal requires every attachment of a render pass to have the same sample
//! count, and Vulkan requires the same of a subpass — with the single
//! exception of resolve targets, which are always single-sampled. So
//! [`Compatibility`] holds one count rather than one per attachment, and a set
//! of attachments that disagree is refused by name rather than resolved to one
//! of them.
//!
//! # A framebuffer is not a pass
//!
//! `VkFramebuffer` binds *images* to a pass, so it is keyed on the views and
//! the extent as well as the pass — three attachments to the same textures at
//! different mip levels are three framebuffers over one render pass. Kept as
//! its own key for that reason, and because a framebuffer is invalidated by a
//! resource replacement while the pass it belongs to is not.
//!
//! # Built, not created
//!
//! Nothing here calls `vkCreateRenderPass` or `vkCmdBeginRendering`. Each
//! build owns the arrays its create info points at and hands out a borrowed
//! info, so the structure a driver would receive is inspectable with no GPU.
//!
//! [`Cache`] keeps that property. It memoizes handles the caller creates and
//! hands back the ones the caller must destroy; it never touches a device
//! itself. So the whole cache — including the reverse index that decides
//! *which* framebuffers a dead image view invalidates — is testable without a
//! GPU, and the one thing that would not be is the `vkDestroy` call the caller
//! makes on what it gives back.

use crate::pass::{ClearColor, Ops, PassPlan};
use ash::vk;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

/// What two passes must agree on for a pipeline built against one to run in
/// the other.
///
/// Deliberately holds no operation and no layout: those are not part of
/// Vulkan's compatibility rule, and including them would make this a second
/// spelling of [`Signature`].
///
/// It does hold [`Self::resolve`], which is not an operation. Vulkan's
/// compatibility rule names "color, input, **resolve**, and depth/stencil
/// attachment references", and a reference naming a real attachment is
/// compatible only with another naming one — never with the null pointer a
/// subpass without resolve targets has. So two passes that differ only in
/// whether a colour attachment resolves are *not* compatible, and a pipeline
/// built for one cannot be bound in the other.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Compatibility {
    /// Colour attachment formats, in the pass's own attachment order.
    pub color: Vec<vk::Format>,
    /// Whether each colour attachment resolves, in the same order.
    ///
    /// A resolve target is an extra attachment in the description and an extra
    /// reference in the subpass, so it changes both the render pass object and
    /// the compatibility class — see the type's doc.
    ///
    /// Not read by [`Self::rendering_info`], because the dynamic-rendering rung
    /// has no resolve in the pipeline at all: it is a field of
    /// `VkRenderingAttachmentInfo`, supplied when the pass is begun. So that
    /// rung compiles one pipeline where it could have shared one. That is the
    /// price of one compatibility class serving both carriers, and it is the
    /// direction the price has to fall — a class loose enough for dynamic
    /// rendering binds an incompatible pipeline on the render-pass rung, while
    /// a class tight enough for the render-pass rung merely compiles a second
    /// one here.
    pub resolve: Vec<bool>,
    /// `None` when the pass has no depth-stencil attachment, and otherwise the
    /// one format that carries both aspects the guest attached.
    pub depth_stencil: Option<vk::Format>,
    /// Whether the depth-stencil attachment's depth aspect is used, which
    /// `VkPipelineRenderingCreateInfo` spells as a format per aspect.
    pub depth: bool,
    pub stencil: bool,
    /// One count for the whole pass. See the module doc.
    pub samples: vk::SampleCountFlags,
}

impl Compatibility {
    /// The structure a pipeline is built with on the dynamic-rendering rung.
    ///
    /// The borrow is this value's: the create info points at
    /// [`Self::color`], so it may not outlive it.
    pub fn rendering_info(&self) -> vk::PipelineRenderingCreateInfo<'_> {
        let mut info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&self.color);
        // Per aspect, because an attachment carrying only stencil must leave
        // the depth format undefined — naming it would declare a depth
        // attachment the pass does not have.
        if let Some(format) = self.depth_stencil {
            if self.depth {
                info = info.depth_attachment_format(format);
            }
            if self.stencil {
                info = info.stencil_attachment_format(format);
            }
        }
        info
    }
}

/// One attachment's operations, as `VkAttachmentDescription` holds them.
///
/// Separate from [`Compatibility`] for the reason the module doc gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttachmentOps {
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub stencil_load: vk::AttachmentLoadOp,
    pub stencil_store: vk::AttachmentStoreOp,
    pub initial_layout: vk::ImageLayout,
    pub final_layout: vk::ImageLayout,
}

/// Everything a `VkRenderPass` is created from.
///
/// Contains its [`Compatibility`] rather than restating the formats, so a
/// cache lookup on one and a pipeline lookup on the other cannot disagree.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    pub compatibility: Compatibility,
    /// One per colour attachment, in the same order.
    pub color: Vec<AttachmentOps>,
    pub depth_stencil: Option<AttachmentOps>,
}

/// Everything a `VkFramebuffer` is created from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FramebufferKey {
    pub render_pass: vk::RenderPass,
    /// In attachment order: every colour attachment, then its resolve target
    /// where it has one, then the depth-stencil attachment.
    pub views: Vec<vk::ImageView>,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

/// The single-sampled image a multisample attachment resolves into.
///
/// Carries its own format and sample count, because they are not the
/// attachment's to assume. Vulkan requires a resolve attachment to have the
/// *same* format as the colour attachment it resolves
/// (VUID-VkSubpassDescription-pResolveAttachments-00866) and a sample count of
/// exactly one (VUID-VkSubpassDescription-pResolveAttachments-00869), and both
/// are properties of a different guest texture that the guest chose
/// independently.
///
/// Describing it from the attachment's own [`Bound`] would state those two
/// requirements as facts instead of checking them. The attachment description
/// would then claim a format the view does not have, and `vkCreateFramebuffer`
/// would refuse it with a message naming neither texture — a driver error
/// standing in for a refusal this rail can give by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolveTarget {
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    /// What this view covers. See [`Coverage`] --- a resolve target is an
    /// element of the framebuffer's attachment array like any other, so the
    /// render area has to fit inside it too.
    pub coverage: Coverage,
}

/// What one attachment view actually covers, which the render area must fit
/// inside.
///
/// # Why the view carries this rather than the area assuming it
///
/// A `VkFramebuffer` is created with a width, a height and a layer count, and
/// every view it is given must be at least that large on all three
/// (VUID-VkFramebufferCreateInfo-flags-04533, -04534 and -04535). The area
/// comes from the guest's `renderTargetWidth`/`Height`/`ArrayLength`, and the
/// views come from textures the guest attached; nothing on the wire makes them
/// agree, and the pass descriptor's own extent is a *separate* declaration the
/// guest may write anything into.
///
/// So this is measured and not assumed, for the reason [`ResolveTarget`]'s own
/// doc gives about format and samples: describing the view from the area would
/// state the requirement as a fact instead of checking it, and
/// `vkCreateFramebuffer` would then refuse it with a message naming neither
/// the texture nor the field the guest wrote.
///
/// The extent is the *view's* --- the mip level it was made over, not level
/// zero --- because that is what the framebuffer is measured against. Likewise
/// `layers` is the view's `layerCount` and not the image's array size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coverage {
    pub extent: vk::Extent2D,
    pub layers: u32,
}

/// One attachment's image, resolved by the caller from the plan's resource
/// names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bound {
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub view: vk::ImageView,
    /// What this view covers. See [`Coverage`].
    pub coverage: Coverage,
    /// The image this attachment resolves into. Required exactly when the
    /// plan's attachment asks to resolve.
    pub resolve: Option<ResolveTarget>,
}

/// Why a pass cannot be built here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The caller supplied a different number of images than the plan has
    /// colour attachments.
    ColorCountMismatch { planned: usize, bound: usize },
    /// The plan has a depth-stencil attachment and no image was supplied, or
    /// the reverse.
    DepthStencilMismatch { planned: bool, bound: bool },
    /// Attachments disagree about how many samples they have. See the module
    /// doc.
    SampleCountMismatch {
        first: vk::SampleCountFlags,
        found: vk::SampleCountFlags,
    },
    /// The plan asks this attachment to resolve and no resolve image was
    /// supplied — or one was supplied for an attachment that does not resolve.
    ResolveMismatch { index: usize, planned: bool },
    /// The plan asks for a depth or stencil multisample resolve, and this rail
    /// has no path that performs one.
    ///
    /// A depth-stencil resolve is `VK_KHR_depth_stencil_resolve`, core from
    /// Vulkan 1.2: `VkSubpassDescriptionDepthStencilResolve` on the
    /// render-pass rung — which also requires `VkRenderPass2` and
    /// `VkSubpassDescription2`, neither of which is built here — and
    /// `resolveMode` on the dynamic rung. Both need the device's reported
    /// `depthResolveModes` and `stencilResolveModes`, because which reductions
    /// a host performs is not fixed and `SAMPLE_ZERO` is the only one every
    /// implementation must offer.
    ///
    /// Refused by name rather than dropped. The information reaches here —
    /// [`crate::pass::plan`] produces the [`crate::pass::Resolve`] — and
    /// building the pass without it produces a pass that runs and never writes
    /// the guest's resolved depth image, which is exactly what
    /// [`Self::ResolveMismatch`] exists to prevent on the colour side.
    DepthStencilResolveUnsupported,
    /// A resolve target's format differs from the attachment it resolves.
    ///
    /// Vulkan admits only the same format, and the two are different guest
    /// textures — so this is the guest naming a pair no host can resolve
    /// between, and the refusal names both.
    ResolveFormatMismatch {
        index: usize,
        attachment: vk::Format,
        resolve: vk::Format,
    },
    /// A resolve target is itself multisampled. It is where a resolve *lands*,
    /// so it is single-sampled or it is not a resolve target.
    ResolveIsMultisampled {
        index: usize,
        samples: vk::SampleCountFlags,
    },
    /// An attachment view is smaller than the area the pass renders over.
    ///
    /// The render area is the guest's `renderTargetWidth`, `Height` and
    /// `ArrayLength`, written on the pass descriptor; the views are textures
    /// the guest attached. They are separate declarations and the wire makes
    /// neither bound the other, so a pass may name an area larger than
    /// something it renders into --- which is a `vkCreateFramebuffer` every
    /// one of whose views is too small on that axis
    /// (VUID-VkFramebufferCreateInfo-flags-04533, -04534 and -04535), and a
    /// driver that does not check writes past the end of a texture.
    ///
    /// One variant with an axis rather than three, because the sentence is the
    /// same each time. `slot` names which view, since a pass has up to
    /// seventeen of them and "an attachment" names none.
    AttachmentSmallerThanArea {
        slot: AttachmentName,
        axis: &'static str,
        covered: u32,
        area: u32,
    },
}

/// Which of a pass's views a refusal is about.
///
/// A pass carries up to eight colour attachments, a resolve target for each,
/// and one depth-stencil attachment. An index alone would name the colour slot
/// and its resolve identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentName {
    Color(usize),
    ColorResolve(usize),
    DepthStencil,
}

impl std::fmt::Display for AttachmentName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Color(index) => write!(f, "color{index}"),
            Self::ColorResolve(index) => write!(f, "color{index}_resolve"),
            Self::DepthStencil => f.write_str("depth_stencil"),
        }
    }
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ColorCountMismatch { .. } => "vk_renderpass_color_count",
            Self::DepthStencilMismatch { .. } => "vk_renderpass_depth_stencil_count",
            Self::SampleCountMismatch { .. } => "vk_renderpass_sample_count",
            Self::ResolveMismatch { .. } => "vk_renderpass_resolve",
            Self::DepthStencilResolveUnsupported => "vk_renderpass_depth_stencil_resolve",
            Self::ResolveFormatMismatch { .. } => "vk_renderpass_resolve_format",
            Self::ResolveIsMultisampled { .. } => "vk_renderpass_resolve_multisampled",
            Self::AttachmentSmallerThanArea { .. } => "vk_renderpass_attachment_smaller_than_area",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColorCountMismatch { planned, bound } => {
                write!(f, "{} planned={planned} bound={bound}", self.slug())
            }
            Self::DepthStencilMismatch { planned, bound } => {
                write!(f, "{} planned={planned} bound={bound}", self.slug())
            }
            Self::SampleCountMismatch { first, found } => write!(
                f,
                "{} first={:?} found={:?}",
                self.slug(),
                first.as_raw(),
                found.as_raw()
            ),
            Self::ResolveMismatch { index, planned } => {
                write!(f, "{} index={index} planned={planned}", self.slug())
            }
            Self::ResolveFormatMismatch {
                index,
                attachment,
                resolve,
            } => write!(
                f,
                "{} index={index} attachment={:?} resolve={:?}",
                self.slug(),
                attachment.as_raw(),
                resolve.as_raw()
            ),
            Self::ResolveIsMultisampled { index, samples } => write!(
                f,
                "{} index={index} samples={:?}",
                self.slug(),
                samples.as_raw()
            ),
            Self::AttachmentSmallerThanArea {
                slot,
                axis,
                covered,
                area,
            } => write!(
                f,
                "{} slot={slot} axis={axis} covered={covered} area={area}",
                self.slug()
            ),
            Self::DepthStencilResolveUnsupported => f.write_str(self.slug()),
        }
    }
}

/// The layout an attachment is in for the whole pass.
///
/// One layout for the initial and the final, because this rail transitions
/// attachments with explicit barriers around the pass rather than letting the
/// pass do it: a pass that changed the layout would move an image the layout
/// tracker believes is somewhere else, and the tracker is what every other
/// barrier is planned against. `VK_IMAGE_LAYOUT_UNDEFINED` as an initial
/// layout would additionally discard the contents a `LOAD` attachment is
/// loading.
const fn color_layout() -> vk::ImageLayout {
    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
}

const fn depth_stencil_layout() -> vk::ImageLayout {
    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
}

/// A pass, built into everything either rung needs.
///
/// Owns the arrays the create infos point at, so a borrowed info is valid for
/// as long as this value is.
///
/// `Debug` is written rather than derived: `VkClearValue` is a union and
/// `VkRenderingAttachmentInfo` carries a raw `p_next`, and neither has one.
/// What a report wants is the signature and the shape, which is what it
/// prints.
#[derive(Clone)]
pub struct Build {
    signature: Signature,
    attachments: Vec<vk::AttachmentDescription>,
    color_refs: Vec<vk::AttachmentReference>,
    resolve_refs: Vec<vk::AttachmentReference>,
    depth_ref: Option<vk::AttachmentReference>,
    views: Vec<vk::ImageView>,
    clears: Vec<vk::ClearValue>,
    color_rendering: Vec<vk::RenderingAttachmentInfo<'static>>,
    depth_rendering: Option<vk::RenderingAttachmentInfo<'static>>,
    stencil_rendering: Option<vk::RenderingAttachmentInfo<'static>>,
    area: vk::Rect2D,
    layers: u32,
}

impl std::fmt::Debug for Build {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Build")
            .field("signature", &self.signature)
            .field("attachments", &self.attachments.len())
            .field("views", &self.views)
            .field("area", &self.area)
            .field("layers", &self.layers)
            .finish()
    }
}

fn ops_of(
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
    layout: vk::ImageLayout,
) -> AttachmentOps {
    AttachmentOps {
        load,
        store,
        // A colour attachment has no stencil aspect; Vulkan ignores these and
        // `DONT_CARE` is the value that says so rather than one that claims a
        // load nothing performs.
        stencil_load: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: layout,
        final_layout: layout,
    }
}

/// What a render pass does with an aspect of its depth-stencil attachment that
/// the guest did not attach: preserve it.
///
/// `DepthStencilPlan::depth` is `None` when the guest attached *only stencil*,
/// and `stencil` is `None` when it attached only depth — one image with two
/// aspects, of which this pass uses one. That is not permission to throw the
/// other away, and `DONT_CARE` as a store op is exactly that permission: on a
/// combined format the driver may discard the aspect, and a later pass that
/// attaches it reads whatever is left.
///
/// It is also what made the two rungs of one [`Build`] disagree about guest
/// pixels. Dynamic rendering attaches nothing for an unplanned aspect, so it
/// leaves it untouched; the render-pass object has one attachment carrying both
/// pairs of operations and cannot say "not attached" — the nearest thing it can
/// say is load it and store it back, which is what leaving it alone means here.
///
/// Ignored by the driver for a format that has no such aspect, so a pure-depth
/// attachment pays nothing for the stencil pair.
const UNATTACHED_ASPECT: Ops = Ops {
    load: vk::AttachmentLoadOp::LOAD,
    store: vk::AttachmentStoreOp::STORE,
};

fn attachment_description(
    format: vk::Format,
    samples: vk::SampleCountFlags,
    ops: AttachmentOps,
) -> vk::AttachmentDescription {
    vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format,
        samples,
        load_op: ops.load,
        store_op: ops.store,
        stencil_load_op: ops.stencil_load,
        stencil_store_op: ops.stencil_store,
        initial_layout: ops.initial_layout,
        final_layout: ops.final_layout,
    }
}

/// One attachment for the dynamic-rendering rung.
///
/// `format` is the attachment's, and it is here for one reason: a resolve mode
/// is not a constant. An integer colour attachment must resolve by sample zero
/// — see [`crate::pixel::color_resolve_mode`] — and the render-pass rung, which
/// has no mode field, already does. A hardcoded `AVERAGE` made the two rungs
/// disagree with one of them invalid.
fn rendering_attachment(
    view: vk::ImageView,
    format: vk::Format,
    layout: vk::ImageLayout,
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
    clear: vk::ClearValue,
    resolve: Option<vk::ImageView>,
) -> vk::RenderingAttachmentInfo<'static> {
    let mut info = vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(layout)
        .load_op(load)
        .store_op(store)
        .clear_value(clear);
    if let Some(target) = resolve {
        info = info
            .resolve_mode(crate::pixel::color_resolve_mode(format))
            .resolve_image_view(target)
            .resolve_image_layout(layout);
    }
    info
}

/// The clear a plan carries for one colour attachment, or a zero the driver
/// ignores because the load operation is not a clear.
fn color_clear(clear: Option<ClearColor>) -> vk::ClearValue {
    vk::ClearValue {
        color: clear.map_or(
            vk::ClearColorValue { float32: [0.0; 4] },
            ClearColor::native,
        ),
    }
}

/// Build a planned pass against the images the caller resolved for it.
///
/// A [`PassPlan`] with no attachment at all cannot arrive here: [`crate::pass::plan`]
/// refuses that descriptor, and re-checking it would put one invariant in two
/// places with only one of them tested against a real descriptor.
///
/// # Errors
///
/// [`Refusal`], with nothing partially built.
pub fn build(
    plan: &PassPlan,
    color: &[Bound],
    depth_stencil: Option<Bound>,
) -> Result<Build, Refusal> {
    if plan.color.len() != color.len() {
        return Err(Refusal::ColorCountMismatch {
            planned: plan.color.len(),
            bound: color.len(),
        });
    }
    if plan.depth_stencil.is_some() != depth_stencil.is_some() {
        return Err(Refusal::DepthStencilMismatch {
            planned: plan.depth_stencil.is_some(),
            bound: depth_stencil.is_some(),
        });
    }

    // One count for the whole pass, taken from the first attachment and
    // checked against every other. Resolve targets are excluded because they
    // are single-sampled by definition.
    let mut samples: Option<vk::SampleCountFlags> = None;
    for bound in color.iter().chain(depth_stencil.iter()) {
        match samples {
            None => samples = Some(bound.samples),
            Some(first) if first != bound.samples => {
                return Err(Refusal::SampleCountMismatch {
                    first,
                    found: bound.samples,
                })
            }
            Some(_) => {}
        }
    }
    let samples = samples.unwrap_or(vk::SampleCountFlags::TYPE_1);

    // Every view the framebuffer will be given, against the area it will be
    // created with. Before any description is written, and over the resolve
    // targets as well: a resolve target is an element of the same attachment
    // array, so the requirement is its too. The plan's own extent is already
    // narrowed and non-zero --- see `crate::pass::plan` --- so what is left is
    // whether the images the names resolved to are big enough to hold it.
    let area = Coverage {
        extent: plan.extent,
        layers: plan.layers,
    };
    let named = color
        .iter()
        .enumerate()
        .flat_map(|(index, bound)| {
            [Some((AttachmentName::Color(index), bound.coverage))]
                .into_iter()
                .chain([bound
                    .resolve
                    .map(|target| (AttachmentName::ColorResolve(index), target.coverage))])
        })
        .flatten()
        .chain(
            depth_stencil
                .iter()
                .map(|bound| (AttachmentName::DepthStencil, bound.coverage)),
        );
    for (slot, coverage) in named {
        for (axis, covered, wanted) in [
            ("width", coverage.extent.width, area.extent.width),
            ("height", coverage.extent.height, area.extent.height),
            ("layers", coverage.layers, area.layers),
        ] {
            if covered < wanted {
                return Err(Refusal::AttachmentSmallerThanArea {
                    slot,
                    axis,
                    covered,
                    area: wanted,
                });
            }
        }
    }

    for (index, (planned, bound)) in plan.color.iter().zip(color).enumerate() {
        if planned.resolve.is_some() != bound.resolve.is_some() {
            return Err(Refusal::ResolveMismatch {
                index,
                planned: planned.resolve.is_some(),
            });
        }
        // The two requirements a resolve target carries in its own right,
        // checked before its description is written rather than asserted by
        // writing one. See [`ResolveTarget`].
        if let Some(target) = bound.resolve {
            if target.format != bound.format {
                return Err(Refusal::ResolveFormatMismatch {
                    index,
                    attachment: bound.format,
                    resolve: target.format,
                });
            }
            if target.samples != vk::SampleCountFlags::TYPE_1 {
                return Err(Refusal::ResolveIsMultisampled {
                    index,
                    samples: target.samples,
                });
            }
        }
    }
    // The depth-stencil pair, which the loop above does not reach. A resolve
    // the plan asks for has no path here; a resolve image the caller supplied
    // for an attachment the plan does not resolve is the colour side's own
    // mismatch, at the index one past the last colour attachment.
    if plan
        .depth_stencil
        .as_ref()
        .is_some_and(|planned| planned.resolve.is_some())
    {
        return Err(Refusal::DepthStencilResolveUnsupported);
    }
    if depth_stencil.is_some_and(|bound| bound.resolve.is_some()) {
        return Err(Refusal::ResolveMismatch {
            index: plan.color.len(),
            planned: false,
        });
    }

    let mut attachments = Vec::new();
    let mut color_refs = Vec::new();
    let mut resolve_refs = Vec::new();
    let mut views = Vec::new();
    let mut clears = Vec::new();
    let mut color_ops = Vec::new();
    let mut resolve_flags = Vec::new();
    let mut color_rendering = Vec::new();

    for (planned, bound) in plan.color.iter().zip(color) {
        let ops = ops_of(planned.load, planned.store, color_layout());
        color_refs.push(vk::AttachmentReference {
            attachment: u32::try_from(attachments.len()).unwrap_or(u32::MAX),
            layout: color_layout(),
        });
        attachments.push(attachment_description(bound.format, samples, ops));
        views.push(bound.view);
        let clear = color_clear(planned.clear);
        clears.push(clear);
        color_ops.push(ops);
        resolve_flags.push(planned.resolve.is_some());
        color_rendering.push(rendering_attachment(
            bound.view,
            bound.format,
            color_layout(),
            planned.load,
            planned.store,
            clear,
            bound.resolve.map(|target| target.view),
        ));

        if let Some(target) = bound.resolve {
            resolve_refs.push(vk::AttachmentReference {
                attachment: u32::try_from(attachments.len()).unwrap_or(u32::MAX),
                layout: color_layout(),
            });
            // The target's own format and sample count, both already checked
            // against what Vulkan admits. Taking the attachment's instead
            // would describe an image nobody looked at.
            attachments.push(attachment_description(
                target.format,
                target.samples,
                ops_of(
                    vk::AttachmentLoadOp::DONT_CARE,
                    vk::AttachmentStoreOp::STORE,
                    color_layout(),
                ),
            ));
            views.push(target.view);
            clears.push(vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            });
        } else {
            // Vulkan wants either no resolve array or one entry per colour
            // attachment, so an attachment that does not resolve contributes
            // `VK_ATTACHMENT_UNUSED` rather than being skipped — skipping it
            // would shift every later attachment's resolve target onto the
            // wrong one.
            resolve_refs.push(vk::AttachmentReference {
                attachment: vk::ATTACHMENT_UNUSED,
                layout: vk::ImageLayout::UNDEFINED,
            });
        }
    }

    let mut depth_ref = None;
    let mut depth_stencil_ops = None;
    let mut depth_rendering = None;
    let mut stencil_rendering = None;
    let mut depth_stencil_format = None;
    let (mut has_depth, mut has_stencil) = (false, false);

    if let (Some(planned), Some(bound)) = (plan.depth_stencil.as_ref(), depth_stencil) {
        let depth = planned.depth.unwrap_or(UNATTACHED_ASPECT);
        let stencil = planned.stencil.unwrap_or(UNATTACHED_ASPECT);
        let ops = AttachmentOps {
            load: depth.load,
            store: depth.store,
            stencil_load: stencil.load,
            stencil_store: stencil.store,
            initial_layout: depth_stencil_layout(),
            final_layout: depth_stencil_layout(),
        };
        depth_ref = Some(vk::AttachmentReference {
            attachment: u32::try_from(attachments.len()).unwrap_or(u32::MAX),
            layout: depth_stencil_layout(),
        });
        attachments.push(attachment_description(bound.format, samples, ops));
        views.push(bound.view);
        let clear = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: planned.clear_depth,
                stencil: planned.clear_stencil,
            },
        };
        clears.push(clear);
        depth_stencil_ops = Some(ops);
        depth_stencil_format = Some(bound.format);
        has_depth = planned.depth.is_some();
        has_stencil = planned.stencil.is_some();
        if has_depth {
            depth_rendering = Some(rendering_attachment(
                bound.view,
                bound.format,
                depth_stencil_layout(),
                depth.load,
                depth.store,
                clear,
                None,
            ));
        }
        if has_stencil {
            stencil_rendering = Some(rendering_attachment(
                bound.view,
                bound.format,
                depth_stencil_layout(),
                stencil.load,
                stencil.store,
                clear,
                None,
            ));
        }
    }

    let signature = Signature {
        compatibility: Compatibility {
            color: color.iter().map(|b| b.format).collect(),
            resolve: resolve_flags,
            depth_stencil: depth_stencil_format,
            depth: has_depth,
            stencil: has_stencil,
            samples,
        },
        color: color_ops,
        depth_stencil: depth_stencil_ops,
    };

    Ok(Build {
        signature,
        attachments,
        color_refs,
        resolve_refs,
        depth_ref,
        views,
        clears,
        color_rendering,
        depth_rendering,
        stencil_rendering,
        area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: plan.extent,
        },
        layers: plan.layers,
    })
}

impl Build {
    /// The key a `VkRenderPass` cache stores this under.
    #[must_use]
    pub fn signature(&self) -> &Signature {
        &self.signature
    }

    /// The key a *pipeline* cache stores against this pass, on either rung.
    #[must_use]
    pub fn compatibility(&self) -> &Compatibility {
        &self.signature.compatibility
    }

    /// The key a `VkFramebuffer` cache stores this under, given the pass it
    /// was created against.
    #[must_use]
    pub fn framebuffer_key(&self, render_pass: vk::RenderPass) -> FramebufferKey {
        FramebufferKey {
            render_pass,
            views: self.views.clone(),
            width: self.area.extent.width,
            height: self.area.extent.height,
            layers: self.layers,
        }
    }

    /// Every attachment description, in the order the references index.
    pub fn attachments(&self) -> &[vk::AttachmentDescription] {
        &self.attachments
    }

    /// The clear values a `vkCmdBeginRenderPass` takes, one per attachment and
    /// in the same order.
    #[must_use]
    pub fn clear_values(&self) -> &[vk::ClearValue] {
        &self.clears
    }

    /// Whether any colour attachment resolves, which is what decides whether
    /// the subpass carries a resolve array at all.
    #[must_use]
    pub fn resolves(&self) -> bool {
        self.signature.compatibility.resolve.iter().any(|r| *r)
    }

    /// Hand `f` the `VkRenderPassCreateInfo` this build describes.
    ///
    /// A closure rather than a returned structure because the subpass
    /// description points at arrays this value owns, and a returned info would
    /// have to carry a lifetime through the subpass as well.
    pub fn with_render_pass_create_info<R>(
        &self,
        f: impl FnOnce(&vk::RenderPassCreateInfo<'_>) -> R,
    ) -> R {
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&self.color_refs);
        if self.resolves() {
            subpass = subpass.resolve_attachments(&self.resolve_refs);
        }
        if let Some(reference) = self.depth_ref.as_ref() {
            subpass = subpass.depth_stencil_attachment(reference);
        }
        let subpasses = [subpass];
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&self.attachments)
            .subpasses(&subpasses);
        f(&info)
    }

    /// Hand `f` the `VkFramebufferCreateInfo` for `render_pass`.
    pub fn with_framebuffer_create_info<R>(
        &self,
        render_pass: vk::RenderPass,
        f: impl FnOnce(&vk::FramebufferCreateInfo<'_>) -> R,
    ) -> R {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&self.views)
            .width(self.area.extent.width)
            .height(self.area.extent.height)
            .layers(self.layers);
        f(&info)
    }

    /// Hand `f` the `VkRenderingInfo` for `vkCmdBeginRendering`.
    pub fn with_rendering_info<R>(&self, f: impl FnOnce(&vk::RenderingInfo<'_>) -> R) -> R {
        let mut info = vk::RenderingInfo::default()
            .render_area(self.area)
            .layer_count(self.layers)
            .color_attachments(&self.color_rendering);
        if let Some(depth) = self.depth_rendering.as_ref() {
            info = info.depth_attachment(depth);
        }
        if let Some(stencil) = self.stencil_rendering.as_ref() {
            info = info.stencil_attachment(stencil);
        }
        f(&info)
    }
}

/// What a [`Cache`] has done, for the report line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheCensus {
    pub render_pass_hits: u64,
    pub render_pass_misses: u64,
    pub framebuffer_hits: u64,
    pub framebuffer_misses: u64,
    /// Framebuffers dropped because an image view they named went away. Not
    /// evictions: see [`Cache`].
    pub framebuffers_invalidated: u64,
    /// Live now, which is what the two counters above cannot say.
    pub render_passes: usize,
    pub framebuffers: usize,
}

/// Everything one device epoch's [`Cache`] was holding when it ended.
///
/// Two vectors and not one, because the two are destroyed by different
/// functions and a caller that confused them would be calling
/// `vkDestroyRenderPass` on a framebuffer.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Retired {
    pub render_passes: Vec<vk::RenderPass>,
    pub framebuffers: Vec<vk::Framebuffer>,
}

impl Retired {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.render_passes.is_empty() && self.framebuffers.is_empty()
    }
}

/// The `VkRenderPass` and `VkFramebuffer` objects one device epoch has built.
///
/// # Two keys, two granularities, on purpose
///
/// Render passes are keyed on [`Signature`] and framebuffers on
/// [`FramebufferKey`], for the reason the module doc gives. Neither is
/// [`Compatibility`], which is the *pipeline* cache's key — a pass that clears
/// and a pass that loads are two objects here and one class there, and
/// collapsing either direction would either recompile every pipeline when a
/// guest changes a load action or begin a pass with the wrong operations.
///
/// # Nothing is evicted, and invalidation is not eviction
///
/// There is no capacity bound and no LRU, for the reason
/// [`crate::variant`] gives: a cached object the guest can still reach is
/// state that represents guest work, and a bound here would silently rebuild
/// under load.
///
/// [`Cache::forget_view`] is not an exception to that. A framebuffer names
/// image views by handle, so when the guest releases the texture behind one,
/// every framebuffer naming it is *already* invalid — keeping it would hand a
/// later pass a framebuffer over freed memory. That is a lifetime the guest
/// ended, not a bound this cache chose, and it is why the reverse index below
/// exists at all: without it the only way to find those framebuffers would be
/// to walk every key, and the only way to be safe without walking would be to
/// keep none.
///
/// # It creates nothing and destroys nothing
///
/// Every entry point takes a closure for the miss and returns handles for the
/// caller to destroy. A cache that called `vkDestroyFramebuffer` itself would
/// be destroying objects a recorded command buffer may still name; the caller
/// has the completion fact and queues them through
/// [`reims_vgpu_core::retire::NativeRetirement`], which is the type that owns
/// when a native object may die.
#[derive(Debug, Default)]
pub struct Cache {
    passes: HashMap<Signature, vk::RenderPass>,
    framebuffers: HashMap<FramebufferKey, vk::Framebuffer>,
    /// Which framebuffer keys name each image view.
    ///
    /// A `HashSet` and not a `Vec` because a framebuffer may name one view in
    /// two slots — a colour attachment and its own resolve target cannot be
    /// the same view, but a depth attachment read by two subpass-less passes
    /// at the same level can be, and a duplicate would hand the same
    /// framebuffer back twice for one destroy.
    by_view: HashMap<vk::ImageView, HashSet<FramebufferKey>>,
    census: CacheCensus,
}

impl Cache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// What this cache has done and is holding.
    #[must_use]
    pub fn census(&self) -> CacheCensus {
        CacheCensus {
            render_passes: self.passes.len(),
            framebuffers: self.framebuffers.len(),
            ..self.census
        }
    }

    /// The `VkRenderPass` for `signature`, creating it on a miss.
    ///
    /// `create` runs at most once per signature per epoch. A creation that
    /// fails is not cached: the driver may have refused for a reason that does
    /// not recur, and a cached failure would turn one refusal into a permanent
    /// one.
    ///
    /// # Errors
    ///
    /// Whatever `create` returns, unchanged and unrecorded.
    pub fn render_pass<E>(
        &mut self,
        signature: &Signature,
        create: impl FnOnce() -> Result<vk::RenderPass, E>,
    ) -> Result<vk::RenderPass, E> {
        if let Some(&pass) = self.passes.get(signature) {
            self.census.render_pass_hits += 1;
            return Ok(pass);
        }
        let pass = create()?;
        self.census.render_pass_misses += 1;
        self.passes.insert(signature.clone(), pass);
        Ok(pass)
    }

    /// The `VkFramebuffer` for `key`, creating it on a miss.
    ///
    /// Every view in the key is indexed, so [`Self::forget_view`] finds this
    /// framebuffer through any one of them.
    ///
    /// # Errors
    ///
    /// Whatever `create` returns, unchanged and unindexed — a framebuffer that
    /// was never created must not be reachable from a view.
    pub fn framebuffer<E>(
        &mut self,
        key: &FramebufferKey,
        create: impl FnOnce() -> Result<vk::Framebuffer, E>,
    ) -> Result<vk::Framebuffer, E> {
        if let Some(&framebuffer) = self.framebuffers.get(key) {
            self.census.framebuffer_hits += 1;
            return Ok(framebuffer);
        }
        let framebuffer = create()?;
        self.census.framebuffer_misses += 1;
        self.framebuffers.insert(key.clone(), framebuffer);
        for &view in &key.views {
            self.by_view.entry(view).or_default().insert(key.clone());
        }
        Ok(framebuffer)
    }

    /// Drop every framebuffer naming `view`, and hand back the handles.
    ///
    /// Called when the image behind `view` stops being the one the guest
    /// named — a resource replacement, a reallocation, or a release. The
    /// returned handles are still live and are the caller's to retire.
    ///
    /// Idempotent: a view nothing names returns nothing, and a second call
    /// returns nothing because the first removed the index entry.
    #[must_use]
    pub fn forget_view(&mut self, view: vk::ImageView) -> Vec<vk::Framebuffer> {
        let Some(keys) = self.by_view.remove(&view) else {
            return Vec::new();
        };
        let mut dropped = Vec::with_capacity(keys.len());
        for key in keys {
            let Some(framebuffer) = self.framebuffers.remove(&key) else {
                // Unreachable while the two maps agree: the sibling cleanup
                // below is what keeps a key from surviving in the index after
                // its framebuffer is gone, and `retire` clears both. Skipping
                // rather than unwrapping because the failure this would be is
                // a drifted index, and handing back a handle that is not in
                // the framebuffer map is a second destroy of one already
                // returned.
                continue;
            };
            // The framebuffer's *other* views must stop naming it, or the
            // index grows for the life of the epoch and a later forget hands
            // back a handle this call already returned.
            for &other in &key.views {
                if other == view {
                    continue;
                }
                if let Entry::Occupied(mut entry) = self.by_view.entry(other) {
                    entry.get_mut().remove(&key);
                    if entry.get().is_empty() {
                        entry.remove();
                    }
                }
            }
            dropped.push(framebuffer);
        }
        self.census.framebuffers_invalidated += dropped.len() as u64;
        dropped
    }

    /// Everything this cache holds, at the end of the device epoch.
    ///
    /// Leaves the cache empty, so a caller that retires and keeps using it is
    /// building against the next epoch rather than handing out handles from
    /// the one that ended.
    #[must_use]
    pub fn retire(&mut self) -> Retired {
        self.by_view.clear();
        Retired {
            render_passes: self.passes.drain().map(|(_, pass)| pass).collect(),
            framebuffers: self
                .framebuffers
                .drain()
                .map(|(_, framebuffer)| framebuffer)
                .collect(),
        }
    }

    /// How many views the reverse index is tracking.
    ///
    /// A cache whose framebuffers have all been forgotten tracks none, and
    /// that is the assertion that catches an index leak.
    #[must_use]
    pub fn indexed_views(&self) -> usize {
        self.by_view.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass::plan;
    use ash::vk::Handle;
    use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
    use reims_vgpu_core::pass::{LoadAction, StoreAction};
    use reims_vgpu_core::pass::{PassDescriptor, RenderTargetExtent};
    use reims_vgpu_core::pixel_format::{
        MTL_FORMAT_DEPTH32_FLOAT, MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, MTL_FORMAT_RGBA8_UNORM,
    };
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn id(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn view(n: u64) -> vk::ImageView {
        vk::ImageView::from_raw(n)
    }

    fn descriptor() -> PassDescriptor {
        let mut d = PassDescriptor::empty();
        d.extent = RenderTargetExtent {
            width: 128,
            height: 64,
            array_length: 1,
        };
        d
    }

    fn one_colour(load: LoadAction, store: StoreAction) -> PassPlan {
        let mut d = descriptor();
        d.color[0].texture = Some(id(1));
        d.color[0].load = load;
        d.color[0].store = store;
        plan(&d, |_| MTL_FORMAT_RGBA8_UNORM).expect("a legal descriptor")
    }

    /// A view exactly covering [`descriptor`]'s render area, so a test that
    /// is not about coverage never trips the coverage check.
    const COVERS: Coverage = Coverage {
        extent: vk::Extent2D {
            width: 128,
            height: 64,
        },
        layers: 1,
    };

    fn bound(view_id: u64) -> Bound {
        Bound {
            format: vk::Format::R8G8B8A8_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            view: view(view_id),
            coverage: COVERS,
            resolve: None,
        }
    }

    /// A legal resolve target for [`bound`]: its format, and one sample.
    fn resolve_target(view_id: u64) -> ResolveTarget {
        ResolveTarget {
            view: view(view_id),
            format: vk::Format::R8G8B8A8_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            coverage: COVERS,
        }
    }

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// A resolve target is a different guest texture, so the two facts Vulkan
    /// demands of it — the attachment's format, and exactly one sample — are
    /// its own and not the attachment's. Checked here, where the refusal can
    /// name both, rather than asserted by writing a description and letting
    /// `vkCreateFramebuffer` refuse a view whose format it does not match.
    #[test]
    fn a_resolve_target_answers_for_its_own_format_and_sample_count() {
        let mut plan = one_colour(LoadAction::Load, StoreAction::Store);
        plan.color[0].resolve = Some(crate::pass::Resolve {
            texture: plan.color[0].texture,
            level: 0,
            slice: 0,
        });

        let legal = Bound {
            resolve: Some(resolve_target(2)),
            ..bound(1)
        };
        assert!(build(&plan, &[legal], None).is_ok());

        let wrong_format = Bound {
            resolve: Some(ResolveTarget {
                format: vk::Format::B8G8R8A8_UNORM,
                ..resolve_target(2)
            }),
            ..bound(1)
        };
        assert_eq!(
            build(&plan, &[wrong_format], None).expect_err("two formats"),
            Refusal::ResolveFormatMismatch {
                index: 0,
                attachment: vk::Format::R8G8B8A8_UNORM,
                resolve: vk::Format::B8G8R8A8_UNORM,
            }
        );

        let multisampled = Bound {
            resolve: Some(ResolveTarget {
                samples: vk::SampleCountFlags::TYPE_4,
                ..resolve_target(2)
            }),
            ..bound(1)
        };
        assert_eq!(
            build(&plan, &[multisampled], None).expect_err("a multisample target"),
            Refusal::ResolveIsMultisampled {
                index: 0,
                samples: vk::SampleCountFlags::TYPE_4,
            }
        );
    }

    /// A resolve is not an operation. Vulkan's compatibility rule names
    /// "color, input, **resolve**, and depth/stencil attachment references",
    /// and a reference naming a real attachment is compatible only with
    /// another naming one — never with the null pointer a subpass without
    /// resolve targets has. So two passes that differ only in whether a colour
    /// attachment resolves are two render passes *and* two pipeline classes,
    /// and a class that conflated them would bind a pipeline built against an
    /// incompatible pass.
    #[test]
    fn a_resolve_changes_the_pass_object_and_the_pipeline_key() {
        // Both plans store; the resolve is injected onto one of them, so it is
        // the only thing the two differ in.
        let mut resolving = one_colour(LoadAction::Load, StoreAction::Store);
        resolving.color[0].resolve = Some(crate::pass::Resolve {
            texture: resolving.color[0].texture,
            level: 0,
            slice: 0,
        });
        let plain = one_colour(LoadAction::Load, StoreAction::Store);

        let with = build(
            &resolving,
            &[Bound {
                resolve: Some(resolve_target(2)),
                ..bound(1)
            }],
            None,
        )
        .expect("legal");
        let without = build(&plain, &[bound(1)], None).expect("legal");

        assert!(with.resolves());
        assert!(!without.resolves());
        assert_ne!(with.signature(), without.signature());
        assert_ne!(
            with.compatibility(),
            without.compatibility(),
            "one pipeline cannot serve both passes"
        );
        // And the difference is the resolve flags and nothing else: the
        // formats, the aspects and the sample count all agree.
        assert_eq!(with.compatibility().color, without.compatibility().color);
        assert_eq!(
            with.compatibility().samples,
            without.compatibility().samples
        );
        assert_eq!(with.compatibility().resolve, vec![true]);
        assert_eq!(without.compatibility().resolve, vec![false]);
    }

    /// The dynamic-rendering rung has no resolve in the pipeline — it is a
    /// field of `VkRenderingAttachmentInfo`, supplied when the pass is begun —
    /// so the flags must not reach `VkPipelineRenderingCreateInfo`.
    #[test]
    fn the_resolve_flags_do_not_reach_the_dynamic_rendering_structure() {
        let mut compatibility = Compatibility {
            color: vec![vk::Format::B8G8R8A8_UNORM],
            resolve: vec![false],
            depth_stencil: None,
            depth: false,
            stencil: false,
            samples: vk::SampleCountFlags::TYPE_1,
        };
        let plain = compatibility.rendering_info();
        let plain = (plain.color_attachment_count, plain.depth_attachment_format);
        compatibility.resolve = vec![true];
        let resolving = compatibility.rendering_info();
        assert_eq!(
            (
                resolving.color_attachment_count,
                resolving.depth_attachment_format
            ),
            plain
        );
    }

    /// The claim the module exists for: two passes that differ only in a load
    /// action are two render passes and one pipeline.
    #[test]
    fn a_load_action_changes_the_pass_object_and_not_the_pipeline_key() {
        let clearing = build(
            &one_colour(LoadAction::Clear, StoreAction::Store),
            &[bound(1)],
            None,
        )
        .expect("legal");
        let loading = build(
            &one_colour(LoadAction::Load, StoreAction::Store),
            &[bound(1)],
            None,
        )
        .expect("legal");

        assert_ne!(clearing.signature(), loading.signature());
        assert_eq!(clearing.compatibility(), loading.compatibility());
        // And the keys hash apart and together the same way, because a cache
        // looks them up by hash before it compares them.
        assert_ne!(hash_of(clearing.signature()), hash_of(loading.signature()));
        assert_eq!(
            hash_of(clearing.compatibility()),
            hash_of(loading.compatibility())
        );

        // The difference really is the load op and nothing else.
        assert_eq!(
            clearing.attachments()[0].load_op,
            vk::AttachmentLoadOp::CLEAR
        );
        assert_eq!(loading.attachments()[0].load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(
            clearing.attachments()[0].format,
            loading.attachments()[0].format
        );
    }

    /// A store action is not part of compatibility either, and a *format*
    /// change is.
    #[test]
    fn the_pipeline_key_moves_with_a_format_and_not_with_a_store() {
        let stored = build(
            &one_colour(LoadAction::Load, StoreAction::Store),
            &[bound(1)],
            None,
        )
        .expect("legal");
        let dropped = build(
            &one_colour(LoadAction::Load, StoreAction::DontCare),
            &[bound(1)],
            None,
        )
        .expect("legal");
        assert_eq!(stored.compatibility(), dropped.compatibility());
        assert_ne!(stored.signature(), dropped.signature());

        let other_format = build(
            &one_colour(LoadAction::Load, StoreAction::Store),
            &[Bound {
                format: vk::Format::R16G16B16A16_SFLOAT,
                ..bound(1)
            }],
            None,
        )
        .expect("legal");
        assert_ne!(stored.compatibility(), other_format.compatibility());

        // As is a sample count.
        let multisampled = build(
            &one_colour(LoadAction::Load, StoreAction::Store),
            &[Bound {
                samples: vk::SampleCountFlags::TYPE_4,
                ..bound(1)
            }],
            None,
        )
        .expect("legal");
        assert_ne!(stored.compatibility(), multisampled.compatibility());
    }

    /// The framebuffer key moves with the images while the pass does not:
    /// three attachments to three textures are one render pass.
    #[test]
    fn the_framebuffer_key_moves_with_the_views_and_the_pass_does_not() {
        let plan = one_colour(LoadAction::Load, StoreAction::Store);
        let first = build(&plan, &[bound(1)], None).expect("legal");
        let second = build(&plan, &[bound(2)], None).expect("legal");
        assert_eq!(first.signature(), second.signature());

        let pass = vk::RenderPass::from_raw(7);
        assert_ne!(first.framebuffer_key(pass), second.framebuffer_key(pass));
        assert_eq!(first.framebuffer_key(pass), first.framebuffer_key(pass));
        // And the pass is part of the key, because a framebuffer belongs to
        // one.
        assert_ne!(
            first.framebuffer_key(pass),
            first.framebuffer_key(vk::RenderPass::from_raw(8))
        );
        let key = first.framebuffer_key(pass);
        assert_eq!(key.views, vec![view(1)]);
        assert_eq!((key.width, key.height, key.layers), (128, 64, 1));
    }

    #[test]
    fn the_render_pass_create_info_names_every_attachment_once() {
        let mut d = descriptor();
        d.color[0].texture = Some(id(1));
        d.color[0].load = LoadAction::Clear;
        d.color[0].store = StoreAction::Store;
        d.color[2].texture = Some(id(2));
        d.color[2].load = LoadAction::Load;
        d.color[2].store = StoreAction::Store;
        d.depth.texture = Some(id(3));
        d.depth.load = LoadAction::Clear;
        d.depth.store = StoreAction::Store;
        let plan = plan(&d, |r| {
            if r == id(3) {
                MTL_FORMAT_DEPTH32_FLOAT
            } else {
                MTL_FORMAT_RGBA8_UNORM
            }
        })
        .expect("a legal descriptor");

        let depth = Bound {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            view: view(30),
            coverage: COVERS,
            resolve: None,
        };
        let built = build(&plan, &[bound(10), bound(20)], Some(depth)).expect("legal");

        // The guest's slot two is Vulkan's colour attachment one: a gap in the
        // guest's slots renumbers, and it renumbers exactly once.
        assert_eq!(built.compatibility().color.len(), 2);
        assert_eq!(built.attachments().len(), 3);
        assert_eq!(built.clear_values().len(), 3);
        assert!(!built.resolves());

        built.with_render_pass_create_info(|info| {
            assert_eq!(info.attachment_count, 3);
            assert_eq!(info.subpass_count, 1);
            // SAFETY: the pointer is into arrays `built` owns and outlives.
            let subpass = unsafe { &*info.p_subpasses };
            assert_eq!(subpass.color_attachment_count, 2);
            assert!(!subpass.p_depth_stencil_attachment.is_null());
            // No resolve array where nothing resolves.
            assert!(subpass.p_resolve_attachments.is_null());
            // SAFETY: two colour references were pushed.
            let refs = unsafe { std::slice::from_raw_parts(subpass.p_color_attachments, 2) };
            assert_eq!(refs[0].attachment, 0);
            assert_eq!(refs[1].attachment, 1);
            // SAFETY: the depth reference is non-null, checked above.
            let depth_ref = unsafe { &*subpass.p_depth_stencil_attachment };
            assert_eq!(depth_ref.attachment, 2);
        });

        built.with_framebuffer_create_info(vk::RenderPass::from_raw(1), |info| {
            assert_eq!(info.attachment_count, 3);
            assert_eq!((info.width, info.height, info.layers), (128, 64, 1));
        });
    }

    /// An attachment that does not resolve still occupies a slot in the
    /// resolve array, or every later attachment's target shifts onto the wrong
    /// one.
    #[test]
    fn a_resolve_array_is_one_entry_per_colour_attachment_or_none_at_all() {
        let mut d = descriptor();
        for slot in 0..2 {
            d.color[slot].texture = Some(id(slot as u32 + 1));
            d.color[slot].load = LoadAction::Load;
            d.color[slot].store = StoreAction::Store;
        }
        d.color[1].store = StoreAction::MultisampleResolve;
        d.color[1].resolve_texture = Some(id(9));
        let plan = plan(&d, |_| MTL_FORMAT_RGBA8_UNORM).expect("a legal descriptor");

        let resolving = Bound {
            resolve: Some(resolve_target(99)),
            samples: vk::SampleCountFlags::TYPE_4,
            ..bound(2)
        };
        let plain = Bound {
            samples: vk::SampleCountFlags::TYPE_4,
            ..bound(1)
        };
        let built = build(&plan, &[plain, resolving], None).expect("legal");
        assert!(built.resolves());
        // Two colour attachments plus one resolve target.
        assert_eq!(built.attachments().len(), 3);
        // The resolve target is single-sampled whatever the pass is.
        assert_eq!(built.compatibility().samples, vk::SampleCountFlags::TYPE_4);
        assert_eq!(built.attachments()[2].samples, vk::SampleCountFlags::TYPE_1);

        built.with_render_pass_create_info(|info| {
            // SAFETY: one subpass was pushed.
            let subpass = unsafe { &*info.p_subpasses };
            assert!(!subpass.p_resolve_attachments.is_null());
            // SAFETY: the array has one entry per colour attachment.
            let resolves = unsafe { std::slice::from_raw_parts(subpass.p_resolve_attachments, 2) };
            assert_eq!(resolves[0].attachment, vk::ATTACHMENT_UNUSED);
            assert_eq!(resolves[1].attachment, 2);
        });
    }

    /// The dynamic rung takes the same compatibility key and turns it into the
    /// structure its pipelines are built with.
    #[test]
    fn the_dynamic_rung_builds_a_pipeline_from_the_same_key() {
        let mut d = descriptor();
        d.color[0].texture = Some(id(1));
        d.color[0].load = LoadAction::Clear;
        d.color[0].store = StoreAction::Store;
        d.depth.texture = Some(id(2));
        d.depth.load = LoadAction::Clear;
        d.depth.store = StoreAction::Store;
        let plan = plan(&d, |r| {
            if r == id(2) {
                MTL_FORMAT_DEPTH32_FLOAT
            } else {
                MTL_FORMAT_RGBA8_UNORM
            }
        })
        .expect("a legal descriptor");
        let depth = Bound {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            view: view(2),
            coverage: COVERS,
            resolve: None,
        };
        let built = build(&plan, &[bound(1)], Some(depth)).expect("legal");

        let compatibility = built.compatibility();
        assert!(compatibility.depth);
        // A depth-only attachment leaves the stencil format undefined, so the
        // pipeline does not declare a stencil attachment the pass lacks.
        assert!(!compatibility.stencil);
        let info = compatibility.rendering_info();
        assert_eq!(info.color_attachment_count, 1);
        assert_eq!(info.depth_attachment_format, vk::Format::D32_SFLOAT);
        assert_eq!(info.stencil_attachment_format, vk::Format::UNDEFINED);

        built.with_rendering_info(|info| {
            assert_eq!(info.color_attachment_count, 1);
            assert_eq!(info.layer_count, 1);
            assert_eq!(
                info.render_area.extent,
                vk::Extent2D {
                    width: 128,
                    height: 64,
                }
            );
            assert!(!info.p_depth_attachment.is_null());
            assert!(info.p_stencil_attachment.is_null());
            // SAFETY: the colour array has one entry.
            let color = unsafe { std::slice::from_raw_parts(info.p_color_attachments, 1) };
            assert_eq!(color[0].image_view, view(1));
            assert_eq!(color[0].load_op, vk::AttachmentLoadOp::CLEAR);
            assert_eq!(
                color[0].image_layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            );
        });
    }

    /// Attachments that disagree about sample count are refused, not resolved
    /// to one of them.
    #[test]
    fn attachments_that_disagree_about_samples_refuse() {
        let mut d = descriptor();
        for slot in 0..2 {
            d.color[slot].texture = Some(id(slot as u32 + 1));
            d.color[slot].load = LoadAction::Load;
            d.color[slot].store = StoreAction::Store;
        }
        let plan = plan(&d, |_| MTL_FORMAT_RGBA8_UNORM).expect("a legal descriptor");
        let refused = build(
            &plan,
            &[
                bound(1),
                Bound {
                    samples: vk::SampleCountFlags::TYPE_4,
                    ..bound(2)
                },
            ],
            None,
        )
        .expect_err("one pass, one sample count");
        assert_eq!(
            refused,
            Refusal::SampleCountMismatch {
                first: vk::SampleCountFlags::TYPE_1,
                found: vk::SampleCountFlags::TYPE_4,
            }
        );
        assert_eq!(refused.slug(), "vk_renderpass_sample_count");

        // Agreeing at four is fine, so the refusal is about disagreement and
        // not about multisampling.
        assert!(build(
            &plan,
            &[
                Bound {
                    samples: vk::SampleCountFlags::TYPE_4,
                    ..bound(1)
                },
                Bound {
                    samples: vk::SampleCountFlags::TYPE_4,
                    ..bound(2)
                },
            ],
            None,
        )
        .is_ok());
    }

    #[test]
    fn a_caller_that_supplied_the_wrong_images_is_refused_by_name() {
        let plan = one_colour(LoadAction::Load, StoreAction::Store);
        assert_eq!(
            build(&plan, &[], None).expect_err("no images for one attachment"),
            Refusal::ColorCountMismatch {
                planned: 1,
                bound: 0
            }
        );
        assert_eq!(
            build(&plan, &[bound(1)], Some(bound(2)))
                .expect_err("a depth image for a pass with no depth"),
            Refusal::DepthStencilMismatch {
                planned: false,
                bound: true
            }
        );
        // A colour attachment that does not resolve, handed a resolve image.
        assert_eq!(
            build(
                &plan,
                &[Bound {
                    resolve: Some(resolve_target(9)),
                    ..bound(1)
                }],
                None
            )
            .expect_err("a resolve image for an attachment that does not resolve"),
            Refusal::ResolveMismatch {
                index: 0,
                planned: false
            }
        );

        // A pass with nothing attached never reaches here: the plan owns that
        // refusal, and this is the assertion that keeps it owned there.
        assert!(crate::pass::plan(&descriptor(), |_| MTL_FORMAT_RGBA8_UNORM).is_err());

        let slugs: std::collections::BTreeSet<&str> = [
            Refusal::ColorCountMismatch {
                planned: 1,
                bound: 0,
            },
            Refusal::DepthStencilMismatch {
                planned: true,
                bound: false,
            },
            Refusal::SampleCountMismatch {
                first: vk::SampleCountFlags::TYPE_1,
                found: vk::SampleCountFlags::TYPE_2,
            },
            Refusal::ResolveMismatch {
                index: 0,
                planned: true,
            },
        ]
        .iter()
        .map(|r| r.slug())
        .collect();
        assert_eq!(slugs.len(), 4);
    }
    /// A depth-stencil plan that asks to resolve. The `Resolve` reaches here
    /// --- `pass::plan` produced it from the guest's store action --- and
    /// nothing in `build` reads it, so before this refusal existed the pass was
    /// built without one and the guest's resolved depth image was never
    /// written.
    fn depth_stencil_that_resolves(resolves: bool) -> PassPlan {
        let mut d = descriptor();
        d.color[0].texture = Some(id(1));
        d.color[0].load = LoadAction::Load;
        d.color[0].store = StoreAction::Store;
        d.depth.texture = Some(id(2));
        d.depth.load = LoadAction::Load;
        d.depth.store = if resolves {
            StoreAction::StoreAndMultisampleResolve
        } else {
            StoreAction::Store
        };
        if resolves {
            d.depth.resolve_texture = Some(id(3));
        }
        plan(&d, |_| MTL_FORMAT_DEPTH32_FLOAT).expect("a legal descriptor")
    }

    /// One image, two aspects, and this pass uses one of them. The other is not
    /// this pass's to throw away.
    ///
    /// `DepthStencilPlan::depth` is `None` when the guest attached only
    /// stencil, and `stencil` is `None` when it attached only depth. Both used
    /// to become `DONT_CARE`, which on a combined format licenses the driver to
    /// discard an aspect of a texture the guest still owns — so a depth prepass
    /// on a `Depth32Float_Stencil8` target destroyed the stencil mask a later
    /// pass tested against.
    ///
    /// The two rungs disagreed about it, which is the sharper form of the same
    /// claim: dynamic rendering attaches nothing for an unplanned aspect and so
    /// leaves it alone, while the render-pass object discarded it. One `Build`
    /// producing two different guest-visible results is what the fourth
    /// supported pathway exists to measure, and it would have been read as a
    /// metal2vulkan difference.
    #[test]
    fn an_aspect_this_pass_does_not_attach_is_preserved_on_both_rungs() {
        let mut d = descriptor();
        d.color[0].texture = Some(id(1));
        d.color[0].load = LoadAction::Load;
        d.color[0].store = StoreAction::Store;
        // Depth attached, stencil not — the depth-prepass shape.
        d.depth.texture = Some(id(2));
        d.depth.load = LoadAction::Clear;
        d.depth.store = StoreAction::Store;
        let planned = plan(&d, |_| MTL_FORMAT_DEPTH32_FLOAT_STENCIL8).expect("a legal descriptor");
        let ds = planned.depth_stencil.as_ref().expect("attached");
        assert!(ds.depth.is_some());
        assert!(
            ds.stencil.is_none(),
            "the guest attached no stencil, which is what this is about"
        );

        let combined = Bound {
            format: vk::Format::D32_SFLOAT_S8_UINT,
            ..bound(2)
        };
        let built = build(&planned, &[bound(1)], Some(combined)).expect("builds");

        let ds_attachment = built
            .attachments()
            .iter()
            .find(|a| a.format == vk::Format::D32_SFLOAT_S8_UINT)
            .expect("the depth-stencil attachment is described");
        assert_ne!(
            ds_attachment.stencil_store_op,
            vk::AttachmentStoreOp::DONT_CARE,
            "the render pass may not discard an aspect this pass never touched"
        );
        assert_eq!(ds_attachment.stencil_store_op, vk::AttachmentStoreOp::STORE);
        assert_eq!(ds_attachment.stencil_load_op, vk::AttachmentLoadOp::LOAD);
        // And the aspect this pass *does* use keeps what the guest asked for.
        assert_eq!(ds_attachment.load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(ds_attachment.store_op, vk::AttachmentStoreOp::STORE);

        // The other rung's answer to the same question, which is what the two
        // have to agree about: no stencil attachment at all.
        built.with_rendering_info(|info| {
            assert!(
                info.p_stencil_attachment.is_null(),
                "dynamic rendering attaches no aspect the guest did not"
            );
            assert!(!info.p_depth_attachment.is_null());
        });
    }

    #[test]
    fn a_depth_stencil_resolve_is_refused_and_not_quietly_left_out() {
        let asking = depth_stencil_that_resolves(true);
        assert!(
            asking
                .depth_stencil
                .as_ref()
                .expect("attached")
                .resolve
                .is_some(),
            "the plan really does carry the resolve this build cannot make"
        );
        let depth = Bound {
            format: vk::Format::D32_SFLOAT,
            ..bound(2)
        };
        assert_eq!(
            build(&asking, &[bound(1)], Some(depth)).expect_err("no path for it"),
            Refusal::DepthStencilResolveUnsupported
        );

        // The same pass without the resolve builds, so the refusal is about the
        // resolve and not about attaching depth at all.
        assert!(build(
            &depth_stencil_that_resolves(false),
            &[bound(1)],
            Some(depth)
        )
        .is_ok());
    }

    /// The other direction: a resolve image supplied for a depth-stencil
    /// attachment the plan does not resolve was ignored, exactly as a colour
    /// attachment's would not have been.
    #[test]
    fn a_depth_stencil_resolve_image_nobody_asked_for_is_a_mismatch() {
        let quiet = depth_stencil_that_resolves(false);
        let depth = Bound {
            format: vk::Format::D32_SFLOAT,
            resolve: Some(resolve_target(7)),
            ..bound(2)
        };
        assert_eq!(
            build(&quiet, &[bound(1)], Some(depth)).expect_err("nothing asked for it"),
            Refusal::ResolveMismatch {
                index: 1,
                planned: false,
            }
        );
    }

    /// The failure this exists to prevent: a `VkFramebuffer` is created with a
    /// width, a height and a layer count, and every view it is given must be
    /// at least that large on all three. The area is the guest's
    /// `renderTargetWidth`/`Height`/`ArrayLength` and the views are textures
    /// the guest attached --- two separate declarations the wire does not make
    /// agree. `Bound` did not carry what its view covered, so the build had
    /// nothing to compare and the pass was created against images too small
    /// for it.
    #[test]
    fn an_attachment_smaller_than_the_render_area_refuses_on_its_own_axis() {
        let plan = one_colour(LoadAction::Clear, StoreAction::Store);
        assert_eq!(
            plan.extent,
            vk::Extent2D {
                width: 128,
                height: 64
            }
        );
        assert!(build(&plan, &[bound(10)], None).is_ok(), "exactly covering");

        // Each axis independently, and one texel short on each.
        for (axis, coverage) in [
            (
                "width",
                Coverage {
                    extent: vk::Extent2D {
                        width: 127,
                        height: 64,
                    },
                    layers: 1,
                },
            ),
            (
                "height",
                Coverage {
                    extent: vk::Extent2D {
                        width: 128,
                        height: 63,
                    },
                    layers: 1,
                },
            ),
            (
                "layers",
                Coverage {
                    extent: vk::Extent2D {
                        width: 128,
                        height: 64,
                    },
                    layers: 0,
                },
            ),
        ] {
            let short = Bound {
                coverage,
                ..bound(10)
            };
            let refusal = build(&plan, &[short], None).expect_err("one axis short");
            assert!(
                matches!(
                    refusal,
                    Refusal::AttachmentSmallerThanArea {
                        slot: AttachmentName::Color(0),
                        axis: named,
                        ..
                    } if named == axis
                ),
                "{axis}: {refusal}"
            );
        }

        // A view larger than the area is legal --- the framebuffer renders
        // into part of it --- which is what makes the refusal a bound and not
        // an equality.
        let larger = Bound {
            coverage: Coverage {
                extent: vk::Extent2D {
                    width: 256,
                    height: 128,
                },
                layers: 4,
            },
            ..bound(10)
        };
        assert!(build(&plan, &[larger], None).is_ok());
    }

    /// And the resolve target is measured too: it is an element of the same
    /// framebuffer attachment array, so the requirement is equally its. The
    /// colour attachment beside it covers the area, so a check that ran only
    /// over the attachments would pass this.
    #[test]
    fn a_resolve_target_smaller_than_the_render_area_refuses_by_its_own_name() {
        let mut d = descriptor();
        d.color[0].texture = Some(id(1));
        d.color[0].load = LoadAction::Clear;
        d.color[0].store = StoreAction::MultisampleResolve;
        d.color[0].resolve_texture = Some(id(2));
        let plan = plan(&d, |_| MTL_FORMAT_RGBA8_UNORM).expect("a legal descriptor");

        let short = ResolveTarget {
            coverage: Coverage {
                extent: vk::Extent2D {
                    width: 64,
                    height: 64,
                },
                layers: 1,
            },
            ..resolve_target(20)
        };
        assert_eq!(
            build(
                &plan,
                &[Bound {
                    resolve: Some(short),
                    ..bound(10)
                }],
                None,
            )
            .expect_err("the resolve target is half as wide"),
            Refusal::AttachmentSmallerThanArea {
                slot: AttachmentName::ColorResolve(0),
                axis: "width",
                covered: 64,
                area: 128,
            }
        );
        assert!(build(
            &plan,
            &[Bound {
                resolve: Some(resolve_target(20)),
                ..bound(10)
            }],
            None,
        )
        .is_ok());
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::ColorCountMismatch {
                planned: 1,
                bound: 0,
            },
            Refusal::DepthStencilMismatch {
                planned: true,
                bound: false,
            },
            Refusal::SampleCountMismatch {
                first: vk::SampleCountFlags::TYPE_1,
                found: vk::SampleCountFlags::TYPE_4,
            },
            Refusal::ResolveMismatch {
                index: 0,
                planned: true,
            },
            Refusal::DepthStencilResolveUnsupported,
            Refusal::ResolveFormatMismatch {
                index: 0,
                attachment: vk::Format::R8G8B8A8_UNORM,
                resolve: vk::Format::B8G8R8A8_UNORM,
            },
            Refusal::ResolveIsMultisampled {
                index: 0,
                samples: vk::SampleCountFlags::TYPE_4,
            },
            Refusal::AttachmentSmallerThanArea {
                slot: AttachmentName::ColorResolve(3),
                axis: "height",
                covered: 32,
                area: 64,
            },
        ];
        let mut slugs: Vec<&str> = refusals.iter().map(|r| r.slug()).collect();
        let count = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two refusals share a slug");
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_renderpass_"));
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use ash::vk::Handle;

    fn compatibility(color: &[vk::Format]) -> Compatibility {
        Compatibility {
            resolve: vec![false; color.len()],
            color: color.to_vec(),
            depth_stencil: None,
            depth: false,
            stencil: false,
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    fn ops(load: vk::AttachmentLoadOp) -> AttachmentOps {
        AttachmentOps {
            load,
            store: vk::AttachmentStoreOp::STORE,
            stencil_load: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }

    fn signature(load: vk::AttachmentLoadOp) -> Signature {
        Signature {
            compatibility: compatibility(&[vk::Format::B8G8R8A8_UNORM]),
            color: vec![ops(load)],
            depth_stencil: None,
        }
    }

    fn framebuffer_key(pass: u64, views: &[u64]) -> FramebufferKey {
        FramebufferKey {
            render_pass: vk::RenderPass::from_raw(pass),
            views: views.iter().map(|&v| vk::ImageView::from_raw(v)).collect(),
            width: 64,
            height: 32,
            layers: 1,
        }
    }

    /// A counter that stands in for the driver, so a second creation is
    /// visible rather than merely wasteful.
    #[derive(Default)]
    struct Creations(std::cell::Cell<u64>);

    impl Creations {
        fn pass(&self, raw: u64) -> Result<vk::RenderPass, ()> {
            self.0.set(self.0.get() + 1);
            Ok(vk::RenderPass::from_raw(raw))
        }

        fn framebuffer(&self, raw: u64) -> Result<vk::Framebuffer, ()> {
            self.0.set(self.0.get() + 1);
            Ok(vk::Framebuffer::from_raw(raw))
        }
    }

    /// A driven history of the cache against a shadow that owns no index.
    ///
    /// The shadow is one map from key to handle, written only when a `create`
    /// closure actually returned a handle and read only to say what a
    /// `forget_view` should have handed back. It has no reverse index at all,
    /// which is the point: the reverse index is the thing under test, and a
    /// shadow that kept one would go wrong in the same way.
    ///
    /// Three laws, and every one of them is a native object's lifetime:
    ///
    /// - **No handle comes back twice.** A framebuffer handed to the caller
    ///   twice is destroyed twice, and the second destroy is on a handle the
    ///   driver has already reused. Held over the whole history, across
    ///   `forget_view` and `retire` together, not per call.
    /// - **No handle is lost.** Everything created and not forgotten comes back
    ///   from `retire`. A framebuffer the cache forgot about is one the epoch
    ///   ends without destroying.
    /// - **The index says exactly what is live.** `indexed_views` counts the
    ///   distinct views of the live framebuffers and nothing else — an index
    ///   that keeps a dead key hands the same handle back on a later forget,
    ///   which is the first law again by a longer route.
    ///
    /// `forget_view` is driven mostly on views something actually names, and
    /// the counter below says how often, because a sweep that forgets views
    /// nobody used proves only that the empty case returns nothing.
    ///
    /// A `create` that fails is driven too. Its key must stay a miss: a cached
    /// failure turns one driver refusal into a permanent one, and an *indexed*
    /// failure would hand back a handle that was never made.
    #[test]
    fn a_driven_history_hands_every_framebuffer_back_exactly_once() {
        use std::collections::{HashMap as Map, HashSet as Set};

        let mut rng: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        let (mut created, mut hits, mut refused_creates) = (0u64, 0u64, 0u64);
        let (mut forgets_that_dropped, mut empty_forgets, mut retires) = (0u64, 0u64, 0u64);
        let (mut pass_hits, mut pass_misses) = (0u64, 0u64);

        for _ in 0..400 {
            let mut cache = Cache::new();
            // What the sweep believes is live, and every handle ever returned
            // to the caller. Both are written from the calls alone.
            let mut live: Map<FramebufferKey, vk::Framebuffer> = Map::new();
            let mut returned: Set<u64> = Set::new();
            let mut handles = 1u64;

            for _ in 0..40 {
                match next() % 10 {
                    // A framebuffer, from a small pool of views so that one
                    // view names several framebuffers and several views name
                    // one.
                    0..=4 => {
                        let count = 1 + (next() % 2) as usize;
                        let views: Vec<vk::ImageView> = (0..count)
                            .map(|_| vk::ImageView::from_raw(1 + next() % 3))
                            .collect();
                        let key = FramebufferKey {
                            render_pass: vk::RenderPass::from_raw(1),
                            views,
                            width: 64,
                            height: 32,
                            layers: 1,
                        };
                        let fail = next() % 8 == 0;
                        let expected = live.get(&key).copied();
                        let mut ran = false;
                        let answer = cache.framebuffer(&key, || {
                            ran = true;
                            if fail {
                                Err(())
                            } else {
                                handles += 1;
                                Ok(vk::Framebuffer::from_raw(handles))
                            }
                        });
                        match (expected, answer) {
                            (Some(known), Ok(got)) => {
                                assert_eq!(got, known, "a hit answered with another handle");
                                assert!(!ran, "a hit called the driver");
                                hits += 1;
                            }
                            (None, Ok(got)) => {
                                assert!(ran);
                                created += 1;
                                live.insert(key, got);
                            }
                            (None, Err(())) => {
                                assert!(ran);
                                refused_creates += 1;
                            }
                            (Some(_), Err(())) => panic!("a live key reached the driver"),
                        }
                    }
                    5..=7 => {
                        // Mostly a view something names, so the interesting
                        // half of `forget_view` is actually driven.
                        let view = vk::ImageView::from_raw(if next() % 8 == 0 {
                            90 + next() % 4
                        } else {
                            1 + next() % 3
                        });
                        let mut expected: Vec<vk::Framebuffer> = Vec::new();
                        live.retain(|key, framebuffer| {
                            if key.views.contains(&view) {
                                expected.push(*framebuffer);
                                false
                            } else {
                                true
                            }
                        });
                        let mut dropped = cache.forget_view(view);
                        dropped.sort_by_key(|f| f.as_raw());
                        expected.sort_by_key(|f| f.as_raw());
                        assert_eq!(dropped, expected, "forget_view answered the wrong set");
                        if dropped.is_empty() {
                            empty_forgets += 1;
                        } else {
                            forgets_that_dropped += 1;
                        }
                        for framebuffer in dropped {
                            assert!(
                                returned.insert(framebuffer.as_raw()),
                                "a framebuffer came back twice"
                            );
                        }
                    }
                    8 => {
                        let signature = signature(if next() % 2 == 0 {
                            vk::AttachmentLoadOp::CLEAR
                        } else {
                            vk::AttachmentLoadOp::LOAD
                        });
                        let before = cache.census();
                        let _ = cache.render_pass(&signature, || {
                            handles += 1;
                            Ok::<_, ()>(vk::RenderPass::from_raw(handles))
                        });
                        let after = cache.census();
                        pass_hits += after.render_pass_hits - before.render_pass_hits;
                        pass_misses += after.render_pass_misses - before.render_pass_misses;
                    }
                    // Gated inside the arm rather than given a narrower
                    // range: a retirement empties the cache, and too many of
                    // them leave every history too short for one view to name
                    // several framebuffers.
                    _ if next() % 3 == 0 => {
                        let retired = cache.retire();
                        retires += 1;
                        let mut got: Vec<u64> =
                            retired.framebuffers.iter().map(|f| f.as_raw()).collect();
                        let mut want: Vec<u64> = live.values().map(|f| f.as_raw()).collect();
                        got.sort_unstable();
                        want.sort_unstable();
                        assert_eq!(got, want, "retire lost or invented a framebuffer");
                        for raw in got {
                            assert!(returned.insert(raw), "a framebuffer came back twice");
                        }
                        live.clear();
                        assert_eq!(cache.indexed_views(), 0, "an index survived retirement");
                    }
                    _ => {}
                }

                // The index holds exactly the live framebuffers' views, after
                // every operation. An index entry for a framebuffer that is
                // gone is how the same handle comes back a second time.
                let distinct: Set<vk::ImageView> =
                    live.keys().flat_map(|k| k.views.iter().copied()).collect();
                assert_eq!(
                    cache.indexed_views(),
                    distinct.len(),
                    "the reverse index and the live set disagree"
                );
                assert_eq!(cache.census().framebuffers, live.len());
            }

            // Nothing is lost at the end of the history either.
            let retired = cache.retire();
            for framebuffer in retired.framebuffers {
                assert!(
                    returned.insert(framebuffer.as_raw()),
                    "a framebuffer came back twice"
                );
            }
        }

        assert!(created > 5_000, "created={created}");
        assert!(hits > 1_500, "hits={hits}");
        assert!(refused_creates > 500, "refused={refused_creates}");
        assert!(
            forgets_that_dropped > 2_000,
            "forgets_that_dropped={forgets_that_dropped}"
        );
        assert!(empty_forgets > 2_000, "empty_forgets={empty_forgets}");
        assert!(retires > 400, "retires={retires}");
        assert!(pass_hits > 500, "pass_hits={pass_hits}");
        assert!(pass_misses > 800, "pass_misses={pass_misses}");
    }

    /// A repeated signature is one object, and the driver is asked once.
    #[test]
    fn a_repeated_signature_is_one_render_pass() {
        let mut cache = Cache::new();
        let made = Creations::default();
        let first = cache.render_pass(&signature(vk::AttachmentLoadOp::CLEAR), || made.pass(1));
        let second = cache.render_pass(&signature(vk::AttachmentLoadOp::CLEAR), || made.pass(2));
        assert_eq!(first, Ok(vk::RenderPass::from_raw(1)));
        assert_eq!(second, Ok(vk::RenderPass::from_raw(1)));
        assert_eq!(made.0.get(), 1, "the driver was asked twice");
        assert_eq!(cache.census().render_pass_hits, 1);
        assert_eq!(cache.census().render_pass_misses, 1);
        assert_eq!(cache.census().render_passes, 1);
    }

    /// A load action makes a different render pass and the *same* pipeline
    /// class. That is the whole reason there are two keys: a cache keyed on
    /// compatibility would begin the second pass with the first's operations,
    /// and a pipeline cache keyed on the signature would recompile every
    /// pipeline in the frame the first time a guest loaded instead of cleared.
    #[test]
    fn a_load_action_is_a_different_pass_and_the_same_compatibility() {
        let mut cache = Cache::new();
        let made = Creations::default();
        let clear = signature(vk::AttachmentLoadOp::CLEAR);
        let load = signature(vk::AttachmentLoadOp::LOAD);
        assert_ne!(clear, load);
        assert_eq!(clear.compatibility, load.compatibility);

        let a = cache.render_pass(&clear, || made.pass(1)).expect("created");
        let b = cache.render_pass(&load, || made.pass(2)).expect("created");
        assert_ne!(a, b);
        assert_eq!(made.0.get(), 2);
        assert_eq!(cache.census().render_passes, 2);
    }

    /// A failed creation is not cached. A driver that refused once for a
    /// reason that does not recur would otherwise refuse for the epoch.
    #[test]
    fn a_refused_creation_is_not_remembered_as_a_refusal() {
        let mut cache = Cache::new();
        let signature = signature(vk::AttachmentLoadOp::CLEAR);
        assert_eq!(
            cache.render_pass(&signature, || Err::<vk::RenderPass, _>("refused")),
            Err("refused")
        );
        assert_eq!(cache.census().render_passes, 0);
        assert_eq!(cache.census().render_pass_misses, 0);
        assert_eq!(
            cache.render_pass(&signature, || Ok::<_, &str>(vk::RenderPass::from_raw(1))),
            Ok(vk::RenderPass::from_raw(1))
        );

        // Same for a framebuffer, and the failed one leaves no view behind.
        let key = framebuffer_key(1, &[10, 11]);
        assert!(cache
            .framebuffer(&key, || Err::<vk::Framebuffer, _>("refused"))
            .is_err());
        assert_eq!(
            cache.indexed_views(),
            0,
            "a view names a framebuffer that does not exist"
        );
    }

    /// Every view of a framebuffer finds it, not only the first.
    #[test]
    fn a_framebuffer_is_reachable_through_each_view_it_names() {
        for through in 0..3u64 {
            let mut cache = Cache::new();
            let made = Creations::default();
            let key = framebuffer_key(1, &[10, 11, 12]);
            cache
                .framebuffer(&key, || made.framebuffer(100))
                .expect("created");
            assert_eq!(cache.indexed_views(), 3);
            let dropped = cache.forget_view(vk::ImageView::from_raw(10 + through));
            assert_eq!(
                dropped,
                vec![vk::Framebuffer::from_raw(100)],
                "not found through view {through}"
            );
            // And the index no longer holds any of the three, so a later
            // forget cannot hand the same handle back a second time.
            assert_eq!(cache.indexed_views(), 0);
            for other in 0..3u64 {
                assert!(cache
                    .forget_view(vk::ImageView::from_raw(10 + other))
                    .is_empty());
            }
            assert_eq!(cache.census().framebuffers_invalidated, 1);
            assert_eq!(cache.census().framebuffers, 0);
        }
    }

    /// One view shared by several framebuffers invalidates all of them, and
    /// leaves the ones that do not name it alone.
    #[test]
    fn a_shared_view_invalidates_exactly_the_framebuffers_that_name_it() {
        let mut cache = Cache::new();
        let made = Creations::default();
        // Three attachments to the same depth view at different colour views,
        // and one that shares nothing.
        for (raw, views) in [
            (100, vec![10, 99]),
            (101, vec![11, 99]),
            (102, vec![12, 99]),
            (103, vec![13, 98]),
        ] {
            cache
                .framebuffer(&framebuffer_key(1, &views), || made.framebuffer(raw))
                .expect("created");
        }
        assert_eq!(cache.census().framebuffers, 4);

        let mut dropped = cache.forget_view(vk::ImageView::from_raw(99));
        dropped.sort_by_key(|f| f.as_raw());
        assert_eq!(
            dropped,
            vec![
                vk::Framebuffer::from_raw(100),
                vk::Framebuffer::from_raw(101),
                vk::Framebuffer::from_raw(102),
            ]
        );
        assert_eq!(cache.census().framebuffers, 1, "the unrelated one survived");
        // Its own views are still indexed and nothing else is.
        assert_eq!(cache.indexed_views(), 2);
        assert_eq!(
            cache.forget_view(vk::ImageView::from_raw(98)),
            vec![vk::Framebuffer::from_raw(103)]
        );
        assert_eq!(cache.indexed_views(), 0);
    }

    /// A view named twice by one framebuffer hands the handle back once.
    /// Destroying it twice is a driver crash.
    #[test]
    fn a_view_a_framebuffer_names_twice_is_returned_once() {
        let mut cache = Cache::new();
        let made = Creations::default();
        let key = framebuffer_key(1, &[10, 10]);
        cache
            .framebuffer(&key, || made.framebuffer(100))
            .expect("created");
        assert_eq!(cache.indexed_views(), 1, "one view, named twice");
        assert_eq!(
            cache.forget_view(vk::ImageView::from_raw(10)),
            vec![vk::Framebuffer::from_raw(100)]
        );
        assert_eq!(cache.indexed_views(), 0);
    }

    /// After a forget, the same key misses again rather than returning the
    /// handle the caller was told to destroy.
    #[test]
    fn a_forgotten_framebuffer_is_rebuilt_and_not_handed_back() {
        let mut cache = Cache::new();
        let made = Creations::default();
        let key = framebuffer_key(1, &[10]);
        cache
            .framebuffer(&key, || made.framebuffer(100))
            .expect("created");
        assert_eq!(
            cache.forget_view(vk::ImageView::from_raw(10)),
            vec![vk::Framebuffer::from_raw(100)]
        );
        let rebuilt = cache
            .framebuffer(&key, || made.framebuffer(200))
            .expect("created");
        assert_eq!(rebuilt, vk::Framebuffer::from_raw(200));
        assert_eq!(made.0.get(), 2);
        assert_eq!(cache.census().framebuffer_hits, 0);
        assert_eq!(cache.census().framebuffer_misses, 2);
        // The rebuild is indexed too, or the second one leaks.
        assert_eq!(cache.indexed_views(), 1);
    }

    /// The extent is part of the key: the same views at two sizes are two
    /// framebuffers.
    #[test]
    fn a_framebuffer_key_separates_extents_over_the_same_views() {
        let mut cache = Cache::new();
        let made = Creations::default();
        let small = framebuffer_key(1, &[10]);
        let large = FramebufferKey {
            width: 128,
            ..small.clone()
        };
        cache
            .framebuffer(&small, || made.framebuffer(100))
            .expect("created");
        cache
            .framebuffer(&large, || made.framebuffer(101))
            .expect("created");
        assert_eq!(cache.census().framebuffers, 2);
        // Both reachable through the one view they share.
        let mut dropped = cache.forget_view(vk::ImageView::from_raw(10));
        dropped.sort_by_key(|f| f.as_raw());
        assert_eq!(
            dropped,
            vec![
                vk::Framebuffer::from_raw(100),
                vk::Framebuffer::from_raw(101)
            ]
        );
    }

    /// Retirement hands back everything exactly once and leaves nothing
    /// reachable, including the reverse index.
    #[test]
    fn retiring_an_epoch_hands_back_everything_once_and_leaves_no_index() {
        let mut cache = Cache::new();
        let made = Creations::default();
        cache
            .render_pass(&signature(vk::AttachmentLoadOp::CLEAR), || made.pass(1))
            .expect("created");
        cache
            .render_pass(&signature(vk::AttachmentLoadOp::LOAD), || made.pass(2))
            .expect("created");
        for (raw, views) in [(100, vec![10, 11]), (101, vec![12])] {
            cache
                .framebuffer(&framebuffer_key(1, &views), || made.framebuffer(raw))
                .expect("created");
        }

        let retired = cache.retire();
        assert_eq!(retired.render_passes.len(), 2);
        assert_eq!(retired.framebuffers.len(), 2);
        assert!(!retired.is_empty());
        assert_eq!(cache.census().render_passes, 0);
        assert_eq!(cache.census().framebuffers, 0);
        assert_eq!(cache.indexed_views(), 0, "the index outlived the epoch");
        // Nothing is handed back twice, which would be two destroys.
        assert!(cache.retire().is_empty());
        assert!(cache.forget_view(vk::ImageView::from_raw(10)).is_empty());
    }

    /// A view nothing names forgets nothing, whatever else is cached.
    #[test]
    fn forgetting_an_unnamed_view_drops_nothing() {
        let mut cache = Cache::new();
        let made = Creations::default();
        cache
            .framebuffer(&framebuffer_key(1, &[10]), || made.framebuffer(100))
            .expect("created");
        assert!(cache.forget_view(vk::ImageView::from_raw(77)).is_empty());
        assert_eq!(cache.census().framebuffers, 1);
        assert_eq!(cache.census().framebuffers_invalidated, 0);
    }

    /// A build's own key finds the object the build made, so the cache and the
    /// builder cannot disagree about what identifies a pass.
    #[test]
    fn a_builds_signature_and_framebuffer_key_are_the_cache_keys() {
        use crate::pass::plan;
        use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
        use reims_vgpu_core::pass::{LoadAction, PassDescriptor, RenderTargetExtent, StoreAction};

        let mut descriptor = PassDescriptor::empty();
        descriptor.extent = RenderTargetExtent {
            width: 64,
            height: 32,
            array_length: 1,
        };
        descriptor.color[0].texture = Some(ResourceId {
            slot: ObjectListRef(1),
            generation: SlotGeneration(1),
        });
        descriptor.color[0].load = LoadAction::Clear;
        descriptor.color[0].store = StoreAction::Store;
        let plan = plan(&descriptor, |_| {
            reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM
        })
        .expect("a legal pass");
        let built = build(
            &plan,
            &[Bound {
                format: vk::Format::B8G8R8A8_UNORM,
                samples: vk::SampleCountFlags::TYPE_1,
                view: vk::ImageView::from_raw(10),
                // Exactly the render area declared above.
                coverage: Coverage {
                    extent: vk::Extent2D {
                        width: 64,
                        height: 32,
                    },
                    layers: 1,
                },
                resolve: None,
            }],
            None,
        )
        .expect("one colour attachment");

        let mut cache = Cache::new();
        let made = Creations::default();
        let pass = cache
            .render_pass(built.signature(), || made.pass(1))
            .expect("created");
        // The second lookup of the same build is a hit, which is the property
        // that makes a per-frame pass cost one creation rather than one per
        // frame.
        assert_eq!(
            cache.render_pass(built.signature(), || made.pass(2)),
            Ok(pass)
        );
        assert_eq!(made.0.get(), 1);

        let key = built.framebuffer_key(pass);
        cache
            .framebuffer(&key, || made.framebuffer(100))
            .expect("created");
        // And the view the build named is the one the guest's texture death
        // will invalidate it through.
        assert_eq!(
            cache.forget_view(vk::ImageView::from_raw(10)),
            vec![vk::Framebuffer::from_raw(100)]
        );
    }
}
