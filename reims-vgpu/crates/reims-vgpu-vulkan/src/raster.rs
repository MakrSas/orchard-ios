//! The fixed-function render state a guest sets, translated — and the two
//! pieces of it that are host capabilities rather than mappings.
//!
//! # The Y axis points the other way
//!
//! Metal's clip space has +Y up. Vulkan's framebuffer coordinates have +Y
//! down, so the same vertices rasterize vertically mirrored unless something
//! flips them. There are three places that flip could go — the shader, the
//! projection the guest supplied, or the viewport — and only one of them is
//! this device's to touch.
//!
//! So it is the viewport: a negative height with the origin moved to the
//! bottom edge. `VK_KHR_maintenance1` made that legal and Vulkan 1.1 promoted
//! it, so on a 1.2 baseline it is always available and needs no gate. Flipping
//! in the shader would mean rewriting translated guest code; flipping the
//! projection would mean recognising one, and a guest that renders without a
//! conventional projection matrix would be silently wrong.
//!
//! [`viewport`] is therefore the one place the flip happens, and its test is
//! that the bottom of the guest's rectangle is where the negative height
//! points.
//!
//! # Two states are capabilities and not mappings
//!
//! `MTLDepthClipModeClamp` needs `depthClamp`, and `MTLTriangleFillModeLines`
//! needs `fillModeNonSolid`. Both are optional Vulkan features, and neither
//! has a substitute: clipping where the guest asked to clamp throws away
//! geometry it expected to keep, and filling where it asked for lines draws
//! solid triangles over a wireframe. Both refuse by name, and both refuse only
//! for the mode that needs the feature — the default mode of each is always
//! available, so a host without either feature still runs every guest that
//! does not ask.
//!
//! # Four of these states are encoder commands, and Vulkan bakes them
//!
//! Metal sets the cull mode, the winding, the fill mode and the depth clip
//! mode on the *encoder*: `setCullMode:` and its three siblings may be called
//! between two draws of one pass, and the second draw rasterizes differently
//! with the same pipeline state object. Vulkan puts all four in
//! `VkPipelineRasterizationStateCreateInfo`, where changing one means a
//! different pipeline.
//!
//! Taken literally that is a pipeline per state change — a guest that toggles
//! culling around a draw compiles a second pipeline for it, and compilation is
//! the most expensive thing this rail does. The way out is the dynamic state
//! Vulkan offers for exactly these members, so [`RasterCell`] carries the
//! three feature bits that reach them and [`plan`] spends them:
//!
//! - `extendedDynamicState`, core in 1.3, reaches `vkCmdSetCullMode` and
//!   `vkCmdSetFrontFace`. One bit, two states, which is why the cell has one
//!   field for the pair.
//! - `extendedDynamicState3PolygonMode` reaches `vkCmdSetPolygonModeEXT`.
//! - `extendedDynamicState3DepthClampEnable` reaches
//!   `vkCmdSetDepthClampEnableEXT`.
//!
//! A state this host made dynamic is **normalized out of**
//! [`RasterizationState`] and carried in [`DynamicRaster`] instead. That is the
//! whole benefit: if the guest's value stayed in the pipeline state it would
//! still be a cache dimension, two draws differing only in a dynamic cull mode
//! would still hash apart, and nothing would have been saved. The same
//! normalization is why [`blend`](crate::blend) fixes a disabled attachment's
//! factors and [`depth_stencil`](crate::depth_stencil) fixes a disabled
//! stencil's faces.
//!
//! **Making a state dynamic does not make its capability free.** A device
//! without `fillModeNonSolid` refuses `vkCmdSetPolygonModeEXT(LINE)` exactly
//! as it refuses the pipeline member, and the same holds for `depthClamp`. So
//! the two refusals below run on the guest's ordinal whether or not the state
//! is dynamic, and the dynamic path is a placement decision made after the
//! translation, never instead of it.
//!
//! # Depth bias is always enabled
//!
//! `setDepthBias:slopeScale:clamp:` is an encoder command with no enable bit:
//! Metal has no way to say "this pipeline cannot be biased". Vulkan does, and
//! it is a pipeline member — `depthBiasEnable` clear makes
//! `vkCmdSetDepthBias` a no-op. So the pipeline always sets it, and the values
//! arrive dynamically. Zero bias with the enable on rasterizes identically to
//! the enable off, so this costs a guest that never biases nothing; leaving it
//! clear would silently drop the bias of a guest that does.
//!
//! # The line width is real, and it is only a question where lines are drawn
//!
//! `setLineWidth:` is a render-encoder command carrying one `float`, and this
//! project has Apple's own record for it. It is encoder state like the four
//! above — the guest changes it between draws of one pipeline — and
//! `VK_DYNAMIC_STATE_LINE_WIDTH` is Vulkan 1.0 core, so it is dynamic on every
//! host this rail admits and is never a pipeline-cache dimension. That is why
//! it is not a member of [`RasterizationState`] at all: two draws differing
//! only in their width are one pipeline everywhere, so a field for it in the
//! key would be a dimension no host needs.
//!
//! What does cost a capability is a width other than 1.0:
//! `VkPhysicalDeviceFeatures::wideLines`, bounded by `lineWidthRange`.
//! Without the feature, `vkCmdSetLineWidth` with anything but 1.0 is invalid
//! use — not a wider line drawn thin, but undefined behaviour — so a width
//! this host cannot serve is refused by name rather than clamped into range.
//!
//! Quantisation is not refused. `lineWidthGranularity` says an implementation
//! supports some subset of its range and rounds within it, and the spec asks
//! nothing of the caller for that; a width the host rounds is the host's own
//! rasterization, not a substitution this device made. 1.0 is always exact,
//! because `lineWidthRange` always contains it.
//!
//! The width is a **conditional** question: Vulkan applies it to line
//! primitives and to `POLYGON_MODE_LINE`, and to nothing else. A guest that
//! sets a width and then draws filled triangles has asked this device for
//! nothing, so refusing that draw would be refusing it for a state it never
//! uses — and the honest width to record for it is 1.0, which every device
//! takes. Whether a draw rasterizes lines is a joint fact of this state and
//! the topology, which is why [`rasterizes_lines`] takes both and
//! [`line_width`] takes its answer rather than deciding it: neither half can
//! answer alone, and a module that took only its own half would refuse a draw
//! that never drew a line, or pass one that did.
//! # An unknown ordinal is not the default
//!
//! Every parse here is a closed set returning `None` outside it. Folding an
//! unrecognised cull mode onto `None` draws back faces the guest culled;
//! folding an unrecognised winding onto clockwise culls the wrong ones. The
//! refusal carries the ordinal.

use ash::vk;
use reims_vgpu_core::render::{ScissorRect, Viewport};
use reims_vgpu_core::topology::TopologyClass;

/// `MTLCullMode`.
pub const MTL_CULL_MODE_NONE: u64 = 0;
pub const MTL_CULL_MODE_FRONT: u64 = 1;
pub const MTL_CULL_MODE_BACK: u64 = 2;

/// `MTLWinding`.
pub const MTL_WINDING_CLOCKWISE: u64 = 0;
pub const MTL_WINDING_COUNTER_CLOCKWISE: u64 = 1;

/// `MTLDepthClipMode`.
pub const MTL_DEPTH_CLIP_MODE_CLIP: u64 = 0;
pub const MTL_DEPTH_CLIP_MODE_CLAMP: u64 = 1;

/// `MTLTriangleFillMode`.
pub const MTL_TRIANGLE_FILL_MODE_FILL: u64 = 0;
pub const MTL_TRIANGLE_FILL_MODE_LINES: u64 = 1;

/// The optional Vulkan features two of these states need.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RasterCell {
    /// `VkPhysicalDeviceFeatures::depthClamp`.
    pub depth_clamp: bool,
    /// `VkPhysicalDeviceFeatures::fillModeNonSolid`.
    pub fill_mode_non_solid: bool,
    /// `VkPhysicalDeviceExtendedDynamicStateFeaturesEXT::extendedDynamicState`,
    /// or 1.3 core.
    ///
    /// One field for two states because it is one feature bit: it reaches
    /// `vkCmdSetCullMode` and `vkCmdSetFrontFace` together, and two fields
    /// that a census could only ever set alike would be two spellings of one
    /// fact.
    pub dynamic_cull_and_winding: bool,
    /// `…ExtendedDynamicState3FeaturesEXT::extendedDynamicState3PolygonMode`.
    pub dynamic_polygon_mode: bool,
    /// `…ExtendedDynamicState3FeaturesEXT::extendedDynamicState3DepthClampEnable`.
    pub dynamic_depth_clamp: bool,
}

/// How many viewport and scissor rectangles this host lets one pipeline
/// declare.
///
/// Its own cell, like [`LineWidthCell`] and for the same reason: it is spent
/// at the seam where the *count* is fixed, which is before a pipeline key
/// exists, and it has nothing to say about how a pipeline is built once one
/// does.
///
/// `Default` is one slot, which every device offers and no feature gates ---
/// so a cell that was never filled in admits exactly the guest that never
/// asked for more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewportCell {
    /// `VkPhysicalDeviceFeatures::multiViewport`.
    pub multi_viewport: bool,
    /// `VkPhysicalDeviceLimits::maxViewports`.
    ///
    /// Required to be at least one on every device, and required to be one
    /// exactly where `multiViewport` is off. Carried beside the feature rather
    /// than folded into it because the two refuse differently: a host that
    /// offers no multiple viewports at all is a different finding from one
    /// that offers four and was asked for eight.
    pub max_viewports: u32,
}

impl ViewportCell {
    /// The one slot every device has.
    pub const SINGLE: Self = Self {
        multi_viewport: false,
        max_viewports: 1,
    };
}

impl Default for ViewportCell {
    fn default() -> Self {
        Self::SINGLE
    }
}

/// How many viewport and scissor rectangles a pipeline declares, checked
/// against the host that will build it.
///
/// A checked type rather than a `u32` because the count is a *pipeline member*
/// --- `VkPipelineViewportStateCreateInfo::viewportCount`, which the with-count
/// dynamic states are needed to move --- so the check has to happen before the
/// pipeline key exists, and a key holding a bare number could be assembled by
/// a caller that never asked. The only ways to reach one are
/// [`viewport_slots`], which asks the host, and [`ViewportSlots::ONE`],
/// which is the count no host can refuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ViewportSlots(u32);

impl ViewportSlots {
    /// One viewport, which needs no capability at all.
    pub const ONE: Self = Self(1);

    /// The count, for `viewportCount` and `scissorCount`.
    ///
    /// Vulkan requires those two to be equal unless both are dynamic with
    /// count, so one number answers both.
    #[must_use]
    pub const fn count(self) -> u32 {
        self.0
    }
}

/// How many viewports this pipeline may declare on this host.
///
/// # Errors
///
/// [`Refusal::NoViewport`] for a pipeline that would rasterize nothing, and
/// [`Refusal::ViewportSlots`] for a count past what this host offers.
/// `VkPipelineViewportStateCreateInfo::viewportCount` must be at least one
/// (VUID-VkPipelineViewportStateCreateInfo-scissorCount-arraylength), must not
/// exceed `maxViewports` (VUID-…-viewportCount-01218), and must be exactly one
/// where `multiViewport` is not enabled (VUID-…-viewportCount-01216).
///
/// Refused rather than clamped for the reason [`line_width`] refuses: a
/// pipeline built with fewer viewports than the guest bound rasterizes the
/// geometry it did bind through the wrong rectangle, and says nothing.
pub fn viewport_slots(requested: u32, cell: ViewportCell) -> Result<ViewportSlots, Refusal> {
    if requested == 0 {
        return Err(Refusal::NoViewport);
    }
    // The feature first, because it is the more specific of the two answers:
    // a host without `multiViewport` reports `maxViewports` of one, so the
    // limit check alone would refuse it as "the limit is one" and lose the
    // fact that it has no multiple viewports at all.
    if requested > 1 && !cell.multi_viewport {
        return Err(Refusal::ViewportSlots {
            requested,
            limit: 1,
            multi_viewport: false,
        });
    }
    if requested > cell.max_viewports {
        return Err(Refusal::ViewportSlots {
            requested,
            limit: cell.max_viewports,
            multi_viewport: cell.multi_viewport,
        });
    }
    Ok(ViewportSlots(requested))
}

/// What this host will accept from `vkCmdSetLineWidth`.
///
/// Separate from [`RasterCell`] because it is spent at a different seam: the
/// cell above decides how a *pipeline* is built and what it declares dynamic,
/// and this decides what one *draw* may be given. `VK_DYNAMIC_STATE_LINE_WIDTH`
/// is 1.0 core, so there is no "is it dynamic" bit here to pair with the three
/// in `RasterCell` — it always is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineWidthCell {
    /// `VkPhysicalDeviceFeatures::wideLines`.
    pub wide_lines: bool,
    /// `VkPhysicalDeviceLimits::lineWidthRange`, `[min, max]`.
    ///
    /// Always contains 1.0, which is what makes [`Self::NARROW`] — the cell of
    /// a device that advertises nothing — a cell every guest that never sets a
    /// width still runs on.
    pub range: [f32; 2],
}

impl LineWidthCell {
    /// A device without `wideLines`: one width, and it is the default.
    pub const NARROW: Self = Self {
        wide_lines: false,
        range: [DEFAULT_LINE_WIDTH, DEFAULT_LINE_WIDTH],
    };
}

impl Default for LineWidthCell {
    fn default() -> Self {
        Self::NARROW
    }
}

/// The width a render encoder rasterizes with before the guest sets one.
///
/// Metal's own default, and the one width `VkPhysicalDeviceLimits::lineWidthRange`
/// is required to contain — so it is both "what the guest means by silence" and
/// "what every device takes", and those being the same value is why a guest that
/// never calls `setLineWidth:` needs no capability.
pub const DEFAULT_LINE_WIDTH: f32 = 1.0;

/// Why a state the guest set cannot be honoured here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// An ordinal outside the closed set. Never folded onto a neighbour: see
    /// the module doc.
    UnknownOrdinal { state: &'static str, ordinal: u64 },
    /// The guest asked to clamp depth and this device cannot.
    NoDepthClamp,
    /// The guest asked for a wireframe fill and this device cannot.
    NoNonSolidFill,
    /// A viewport or scissor dimension outside the fields Vulkan carries.
    OutOfRange { field: &'static str, value: u64 },
    /// The draw rasterizes lines at a width other than 1.0 and this device has
    /// no `wideLines`.
    ///
    /// The width travels as its bit pattern so this type stays `Eq`, which is
    /// what lets a refusal be compared in a test and held in a key. Displayed
    /// as the float it is.
    NoWideLines { width_bits: u32 },
    /// The draw rasterizes lines at a width outside `lineWidthRange`. Refused
    /// rather than clamped: a clamp would draw a line the guest did not ask
    /// for and say nothing.
    LineWidthOutOfRange {
        width_bits: u32,
        min_bits: u32,
        max_bits: u32,
    },
    /// A viewport with no area.
    ///
    /// `VkViewport::width` must be greater than zero
    /// (VUID-VkViewport-width-01770), and a zero or negative height is the
    /// same degenerate rectangle on the other axis --- negative is what the Y
    /// flip *produces*, so it cannot also be what the guest is allowed to
    /// declare. Refused rather than recorded: `vkCmdSetViewport` on it is
    /// invalid usage, and clamping it to one pixel would rasterize geometry
    /// the guest asked to see none of.
    ///
    /// The bits are the guest's own doubles, before the narrowing, because
    /// that is where the number came from and it is what a report should name.
    NonPositiveViewport { width_bits: u64, height_bits: u64 },
    /// A pipeline that declares no viewport at all. It rasterizes nothing, and
    /// `VkPipelineViewportStateCreateInfo` requires at least one.
    NoViewport,
    /// More viewports than this host offers.
    ///
    /// `limit` is `maxViewports` where `multiViewport` is enabled and one
    /// where it is not, and both travel because "the host refused four" reads
    /// very differently from "the host refused four because it offers no
    /// multiple viewports at all" --- only the second is a whole missing
    /// feature.
    ViewportSlots {
        requested: u32,
        limit: u32,
        multi_viewport: bool,
    },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UnknownOrdinal { .. } => "vk_raster_unknown_ordinal",
            Self::NoDepthClamp => "vk_raster_no_depth_clamp",
            Self::NoNonSolidFill => "vk_raster_no_non_solid_fill",
            Self::OutOfRange { .. } => "vk_raster_out_of_range",
            Self::NoWideLines { .. } => "vk_raster_no_wide_lines",
            Self::LineWidthOutOfRange { .. } => "vk_raster_line_width_out_of_range",
            Self::NonPositiveViewport { .. } => "vk_raster_non_positive_viewport",
            Self::NoViewport => "vk_raster_no_viewport",
            Self::ViewportSlots { .. } => "vk_raster_viewport_slots",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownOrdinal { state, ordinal } => {
                write!(f, "{} state={state} ordinal={ordinal}", self.slug())
            }
            Self::NoDepthClamp | Self::NoNonSolidFill | Self::NoViewport => {
                f.write_str(self.slug())
            }
            Self::ViewportSlots {
                requested,
                limit,
                multi_viewport,
            } => write!(
                f,
                "{} requested={requested} limit={limit} multi_viewport={}",
                self.slug(),
                u8::from(*multi_viewport)
            ),
            Self::OutOfRange { field, value } => {
                write!(f, "{} field={field} value={value}", self.slug())
            }
            Self::NoWideLines { width_bits } => {
                write!(f, "{} width={}", self.slug(), f32::from_bits(*width_bits))
            }
            Self::LineWidthOutOfRange {
                width_bits,
                min_bits,
                max_bits,
            } => write!(
                f,
                "{} width={} min={} max={}",
                self.slug(),
                f32::from_bits(*width_bits),
                f32::from_bits(*min_bits),
                f32::from_bits(*max_bits)
            ),
            Self::NonPositiveViewport {
                width_bits,
                height_bits,
            } => write!(
                f,
                "{} width={} height={}",
                self.slug(),
                f64::from_bits(*width_bits),
                f64::from_bits(*height_bits)
            ),
        }
    }
}

/// Which faces a draw discards.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`] outside the closed set.
pub const fn cull_mode(ordinal: u64) -> Result<vk::CullModeFlags, Refusal> {
    Ok(match ordinal {
        MTL_CULL_MODE_NONE => vk::CullModeFlags::NONE,
        MTL_CULL_MODE_FRONT => vk::CullModeFlags::FRONT,
        MTL_CULL_MODE_BACK => vk::CullModeFlags::BACK,
        _ => {
            return Err(Refusal::UnknownOrdinal {
                state: "cull_mode",
                ordinal,
            })
        }
    })
}

/// Which winding is the front face.
///
/// The mapping is direct and stays direct *because* the Y flip is in the
/// viewport. A flip done anywhere else would reverse the winding of every
/// triangle, and the correction would have to be here — two states each
/// half-describing one transform, with nothing to say which is which.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`] outside the closed set.
pub const fn front_face(ordinal: u64) -> Result<vk::FrontFace, Refusal> {
    Ok(match ordinal {
        MTL_WINDING_CLOCKWISE => vk::FrontFace::CLOCKWISE,
        MTL_WINDING_COUNTER_CLOCKWISE => vk::FrontFace::COUNTER_CLOCKWISE,
        _ => {
            return Err(Refusal::UnknownOrdinal {
                state: "winding",
                ordinal,
            })
        }
    })
}

/// Whether fragments outside the depth range are clamped or clipped.
///
/// Returns the `depthClampEnable` this state means.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`], or [`Refusal::NoDepthClamp`] when the guest
/// asked to clamp on a device without the feature. Clipping instead throws
/// away geometry the guest expected to keep.
pub const fn depth_clamp(ordinal: u64, cell: RasterCell) -> Result<bool, Refusal> {
    match ordinal {
        MTL_DEPTH_CLIP_MODE_CLIP => Ok(false),
        MTL_DEPTH_CLIP_MODE_CLAMP => {
            if cell.depth_clamp {
                Ok(true)
            } else {
                Err(Refusal::NoDepthClamp)
            }
        }
        _ => Err(Refusal::UnknownOrdinal {
            state: "depth_clip_mode",
            ordinal,
        }),
    }
}

/// How a triangle is filled.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`], or [`Refusal::NoNonSolidFill`] when the guest
/// asked for lines on a device without the feature. Filling instead draws
/// solid triangles over a wireframe.
pub const fn polygon_mode(ordinal: u64, cell: RasterCell) -> Result<vk::PolygonMode, Refusal> {
    match ordinal {
        MTL_TRIANGLE_FILL_MODE_FILL => Ok(vk::PolygonMode::FILL),
        MTL_TRIANGLE_FILL_MODE_LINES => {
            if cell.fill_mode_non_solid {
                Ok(vk::PolygonMode::LINE)
            } else {
                Err(Refusal::NoNonSolidFill)
            }
        }
        _ => Err(Refusal::UnknownOrdinal {
            state: "triangle_fill_mode",
            ordinal,
        }),
    }
}

/// The four encoder states a guest sets, as ordinals.
///
/// An aggregate rather than four arguments, because four `u64` in a row is a
/// call site that compiles with two of them transposed. Each field is the raw
/// guest ordinal: parsing is [`plan`]'s, and an unrecognised one refuses there
/// rather than being folded here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GuestRasterState {
    pub cull_mode: u64,
    pub winding: u64,
    pub depth_clip_mode: u64,
    pub fill_mode: u64,
}

impl GuestRasterState {
    /// What an encoder rasterizes with before the guest sets anything.
    ///
    /// Metal's own defaults, and every one of them is the mode that needs no
    /// optional feature — which is what makes a guest that sets none of these
    /// runnable on the barest device this rail admits.
    pub const DEFAULT: Self = Self {
        cull_mode: MTL_CULL_MODE_NONE,
        winding: MTL_WINDING_CLOCKWISE,
        depth_clip_mode: MTL_DEPTH_CLIP_MODE_CLIP,
        fill_mode: MTL_TRIANGLE_FILL_MODE_FILL,
    };
}

impl Default for GuestRasterState {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Which rasterization members this host supplies per draw rather than per
/// pipeline.
///
/// Derived from [`RasterCell`] and never assembled by hand, so the flags and
/// the values in [`DynamicRaster`] cannot disagree about which is which.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RasterDynamic {
    /// `vkCmdSetCullMode` and `vkCmdSetFrontFace`. One flag, because one
    /// feature bit reaches both.
    pub cull_and_winding: bool,
    pub polygon_mode: bool,
    pub depth_clamp_enable: bool,
}

impl RasterDynamic {
    /// What this host offers, from the cell.
    #[must_use]
    pub const fn of(cell: RasterCell) -> Self {
        Self {
            cull_and_winding: cell.dynamic_cull_and_winding,
            polygon_mode: cell.dynamic_polygon_mode,
            depth_clamp_enable: cell.dynamic_depth_clamp,
        }
    }

    /// Nothing dynamic: every state is baked.
    pub const NONE: Self = Self {
        cull_and_winding: false,
        polygon_mode: false,
        depth_clamp_enable: false,
    };

    /// The states a pipeline built with these flags must declare.
    ///
    /// `DEPTH_BIAS` and `LINE_WIDTH` are unconditional, and are here rather
    /// than in the caller's list for the reason the module doc gives. The
    /// pipeline always enables biasing, so it must always take the values
    /// dynamically; and the line width is Vulkan 1.0 core dynamic state, so
    /// there is no host on which this rail bakes it and no cell that could say
    /// otherwise. Both being here rather than at the call site is what keeps
    /// the declared list and the recorded values one decision — see
    /// [`Plan::polygon_mode`] and [`line_width`] for the values.
    ///
    /// The viewport and scissor are the caller's — they are set per pass, not
    /// per rasterizer state, and this type would be claiming a decision it
    /// does not make.
    #[must_use]
    pub fn states(self) -> Vec<vk::DynamicState> {
        let mut out = vec![vk::DynamicState::DEPTH_BIAS, vk::DynamicState::LINE_WIDTH];
        if self.cull_and_winding {
            out.push(vk::DynamicState::CULL_MODE);
            out.push(vk::DynamicState::FRONT_FACE);
        }
        if self.polygon_mode {
            out.push(vk::DynamicState::POLYGON_MODE_EXT);
        }
        if self.depth_clamp_enable {
            out.push(vk::DynamicState::DEPTH_CLAMP_ENABLE_EXT);
        }
        out
    }
}

/// The rasterization state a pipeline is created with.
///
/// Spelled out rather than held as the ash structure, which is not `Eq` — and
/// a pipeline key that cannot be compared is one whose cache cannot be
/// asserted.
///
/// **A member this host made dynamic carries its default here, not the
/// guest's value.** See the module doc: the guest's value is in
/// [`DynamicRaster`], and leaving it here as well would keep it a cache
/// dimension and save nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RasterizationState {
    pub depth_clamp_enable: bool,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    /// Always true. See the module doc.
    pub depth_bias_enable: bool,
    /// Which members above are placeholders because the encoder supplies them.
    pub dynamic: RasterDynamic,
}

impl RasterizationState {
    pub const fn native(self) -> vk::PipelineRasterizationStateCreateInfo<'static> {
        vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: as_bool32(self.depth_clamp_enable),
            // Never. Metal has no equivalent, and discarding before
            // rasterization would drop every fragment of every draw.
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: self.polygon_mode,
            cull_mode: self.cull_mode,
            front_face: self.front_face,
            depth_bias_enable: as_bool32(self.depth_bias_enable),
            // Zero, and dynamic: the encoder's `setDepthBias:` supplies all
            // three, and zero is what a guest that never calls it means.
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            // A placeholder, and only ever one: `LINE_WIDTH` is in every
            // dynamic-state list this module produces, so the encoder supplies
            // the guest's width before every draw and this member is never
            // read. It is the default rather than an arbitrary number for the
            // reason `BAKED_DEFAULT` exists — a placeholder that drifted from
            // the value a caller would compute is a difference nothing checks.
            line_width: DEFAULT_LINE_WIDTH,
            _marker: core::marker::PhantomData,
        }
    }
}

const fn as_bool32(value: bool) -> vk::Bool32 {
    if value {
        vk::TRUE
    } else {
        vk::FALSE
    }
}

/// The guest's values for the members this host made dynamic.
///
/// `Some` exactly where [`RasterizationState`] holds a placeholder, so a
/// caller that sets every `Some` has reproduced the guest's state and a caller
/// that sets none of them is a bug this type can name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DynamicRaster {
    pub cull_mode: Option<vk::CullModeFlags>,
    pub front_face: Option<vk::FrontFace>,
    pub polygon_mode: Option<vk::PolygonMode>,
    pub depth_clamp_enable: Option<bool>,
}

/// A translated rasterizer state: what the pipeline is built with, and what
/// the encoder sets before each draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub state: RasterizationState,
    pub dynamic: DynamicRaster,
}

impl Plan {
    /// The polygon mode this draw actually rasterizes with, whichever half of
    /// the plan holds it.
    ///
    /// Every caller that asks "does this draw fill or wireframe" must ask it
    /// this way, because the answer moved between the two halves the moment
    /// the host offered `vkCmdSetPolygonModeEXT`: reading `state.polygon_mode`
    /// alone reports `FILL` for a wireframe draw on a dynamic host, and
    /// reading `dynamic.polygon_mode` alone reports nothing at all on a baking
    /// one. It is a method here rather than an `unwrap_or` at each call site
    /// so there is one place that knows which half is authoritative.
    #[must_use]
    pub const fn polygon_mode(self) -> vk::PolygonMode {
        match self.dynamic.polygon_mode {
            Some(mode) => mode,
            None => self.state.polygon_mode,
        }
    }
}

/// Whether a draw puts line segments through the rasterizer.
///
/// The joint fact the module doc names, and the reason it is a free function
/// taking both halves: the polygon mode is this module's and the topology
/// class is [`crate::topology`]'s, and neither can answer it alone.
///
/// - A line-class topology rasterizes lines whatever the polygon mode is. The
///   polygon mode applies to polygons, and a line is not one.
/// - A triangle-class topology rasterizes lines exactly when the polygon mode
///   is `LINE` — which is `MTLTriangleFillModeLines`.
/// - A point-class topology rasterizes points, and no polygon mode changes
///   that.
#[must_use]
pub const fn rasterizes_lines(polygon_mode: vk::PolygonMode, class: TopologyClass) -> bool {
    match class {
        TopologyClass::Point => false,
        TopologyClass::Line => true,
        TopologyClass::Triangle => polygon_mode.as_raw() == vk::PolygonMode::LINE.as_raw(),
    }
}

/// The width to hand `vkCmdSetLineWidth` before this draw.
///
/// `guest` is what the stream's `setLineWidth:` last set, or `None` where it
/// set none — which means [`DEFAULT_LINE_WIDTH`], and means it here rather
/// than at the caller so no caller can name a different silence.
///
/// `draws_lines` is [`rasterizes_lines`]' answer. A draw that rasterizes none
/// takes the default whatever the guest set, because Vulkan applies the width
/// to nothing that draw produces: refusing it would be refusing a draw for a
/// state it never uses, and forwarding a width the host cannot take would be
/// invalid use for no benefit at all.
///
/// # Errors
///
/// [`Refusal::NoWideLines`] where the draw rasterizes lines at a non-default
/// width on a device without the feature, and
/// [`Refusal::LineWidthOutOfRange`] where the width — including a NaN, which
/// fails both comparisons — is outside `lineWidthRange`. Neither is clamped:
/// see the module doc.
pub fn line_width(
    guest: Option<f32>,
    draws_lines: bool,
    cell: LineWidthCell,
) -> Result<f32, Refusal> {
    let width = guest.unwrap_or(DEFAULT_LINE_WIDTH);
    if !draws_lines {
        return Ok(DEFAULT_LINE_WIDTH);
    }
    // Bit equality, not `==`: the question is whether the guest wrote *the*
    // literal that needs no feature, and `-0.0 == 0.0` is the kind of answer a
    // float comparison gives that a capability check must not act on.
    if width.to_bits() == DEFAULT_LINE_WIDTH.to_bits() {
        return Ok(DEFAULT_LINE_WIDTH);
    }
    if !cell.wide_lines {
        return Err(Refusal::NoWideLines {
            width_bits: width.to_bits(),
        });
    }
    let [min, max] = cell.range;
    if !(width >= min && width <= max) {
        return Err(Refusal::LineWidthOutOfRange {
            width_bits: width.to_bits(),
            min_bits: min.to_bits(),
            max_bits: max.to_bits(),
        });
    }
    Ok(width)
}

/// The rasterization members a pipeline bakes when nothing is dynamic.
///
/// Also what a dynamic member is normalized to, which is why it is one
/// constant and not two: a placeholder that drifted from the baked default
/// would make an all-static host and an all-dynamic host build different
/// pipelines for the same guest state.
const BAKED_DEFAULT: (bool, vk::PolygonMode, vk::CullModeFlags, vk::FrontFace) = (
    false,
    vk::PolygonMode::FILL,
    vk::CullModeFlags::NONE,
    vk::FrontFace::CLOCKWISE,
);

/// Translate a guest rasterizer state against this host.
///
/// Every ordinal is parsed and every capability is checked, whether or not
/// the state it names ends up dynamic — see the module doc for why making a
/// state dynamic does not make its feature free.
///
/// # Errors
///
/// [`Refusal::UnknownOrdinal`] for an ordinal outside a closed set, and
/// [`Refusal::NoDepthClamp`] or [`Refusal::NoNonSolidFill`] for a mode this
/// device has no feature for. Nothing is partially translated.
pub fn plan(guest: GuestRasterState, cell: RasterCell) -> Result<Plan, Refusal> {
    // All four first, so a refusal is decided before any placement is. A host
    // that made a state dynamic must still refuse the ordinal it cannot serve.
    let cull = cull_mode(guest.cull_mode)?;
    let winding = front_face(guest.winding)?;
    let clamp = depth_clamp(guest.depth_clip_mode, cell)?;
    let fill = polygon_mode(guest.fill_mode, cell)?;

    let dynamic = RasterDynamic::of(cell);
    let (default_clamp, default_fill, default_cull, default_winding) = BAKED_DEFAULT;
    Ok(Plan {
        state: RasterizationState {
            depth_clamp_enable: if dynamic.depth_clamp_enable {
                default_clamp
            } else {
                clamp
            },
            polygon_mode: if dynamic.polygon_mode {
                default_fill
            } else {
                fill
            },
            cull_mode: if dynamic.cull_and_winding {
                default_cull
            } else {
                cull
            },
            front_face: if dynamic.cull_and_winding {
                default_winding
            } else {
                winding
            },
            depth_bias_enable: true,
            dynamic,
        },
        dynamic: DynamicRaster {
            cull_mode: dynamic.cull_and_winding.then_some(cull),
            front_face: dynamic.cull_and_winding.then_some(winding),
            polygon_mode: dynamic.polygon_mode.then_some(fill),
            depth_clamp_enable: dynamic.depth_clamp_enable.then_some(clamp),
        },
    })
}

/// The viewport a guest rectangle becomes, with the Y flip in it.
///
/// The origin moves to the bottom edge and the height is negated, which is
/// what makes +Y-up clip space rasterize the way the guest drew it. See the
/// module doc for why the flip is here and not in the shader or the
/// projection.
///
/// # Errors
///
/// [`Refusal::OutOfRange`] for a dimension outside the `f32` fields Vulkan
/// carries. The guest's are doubles.
///
/// # The narrowing is where the check has to be
///
/// Finiteness of the guest's own doubles is not the same question. A finite
/// double past `f32::MAX` casts to an infinity rather than saturating at it,
/// so a viewport that passes every input check can still hand a driver an
/// infinite width --- and the bottom edge is a *sum*, so two halves that each
/// fit can produce one that does not. Both are decided on the value that
/// actually lands in the field.
///
/// The depth pair goes through the same door. `minDepth` and `maxDepth` are
/// `f32` like the rest, and a NaN in either is a viewport a driver may hang
/// on; they were the two of six nothing looked at. Their *range* is
/// deliberately not narrowed here: Metal and Vulkan agree on `[0, 1]` and a
/// guest that reverses its own range is doing something both models allow.
///
/// # And a viewport has to have an area
///
/// Vulkan requires `width` greater than zero, and a zero or negative height
/// is the same degenerate rectangle on the other axis. Read before the flip
/// negates the height, so a legal rectangle is not mistaken for a degenerate
/// one --- and read *after* the narrowing, because a positive double small
/// enough narrows to zero and the field the driver reads is the one that has
/// to be legal. See [`Refusal::NonPositiveViewport`].
pub fn viewport(guest: Viewport) -> Result<vk::Viewport, Refusal> {
    let x = f64::from_bits(guest.origin_x_bits);
    let y = f64::from_bits(guest.origin_y_bits);
    let width = f64::from_bits(guest.width_bits);
    let height = f64::from_bits(guest.height_bits);
    let near = f64::from_bits(guest.z_near_bits);
    let far = f64::from_bits(guest.z_far_bits);
    let narrowed = |field: &'static str, value: f64| -> Result<f32, Refusal> {
        let narrow = value as f32;
        if narrow.is_finite() {
            Ok(narrow)
        } else {
            Err(Refusal::OutOfRange {
                field,
                value: value.to_bits(),
            })
        }
    };
    let native = vk::Viewport {
        x: narrowed("origin_x", x)?,
        // The bottom edge, because the height below is negative and a viewport
        // is measured from its `y`.
        y: narrowed("origin_y", y + height)?,
        width: narrowed("width", width)?,
        // Negated last, so the positivity check below reads the guest's
        // rectangle and not this function's own flip.
        height: narrowed("height", height)?,
        min_depth: narrowed("z_near", near)?,
        max_depth: narrowed("z_far", far)?,
    };
    // On the narrowed values and not on the guest's doubles. A positive double
    // small enough --- `1e-300` --- narrows to zero, so a rectangle that has an
    // area in the guest's arithmetic has none in the field the driver reads,
    // and the field the driver reads is the one that has to be legal. Both are
    // finite by now, so this decides rather than falling through.
    if native.width <= 0.0 || native.height <= 0.0 {
        return Err(Refusal::NonPositiveViewport {
            width_bits: guest.width_bits,
            height_bits: guest.height_bits,
        });
    }
    Ok(vk::Viewport {
        height: -native.height,
        ..native
    })
}

/// The scissor a guest rectangle becomes.
///
/// No flip: a scissor is in framebuffer pixels in both models, with the origin
/// at the top left. Flipping it as well would move the rectangle off the
/// geometry the viewport just placed.
///
/// # Errors
///
/// [`Refusal::OutOfRange`] for a coordinate outside the 32-bit fields Vulkan
/// carries.
pub fn scissor(guest: ScissorRect) -> Result<vk::Rect2D, Refusal> {
    let offset_x = i32::try_from(guest.x).map_err(|_| Refusal::OutOfRange {
        field: "x",
        value: guest.x,
    })?;
    let offset_y = i32::try_from(guest.y).map_err(|_| Refusal::OutOfRange {
        field: "y",
        value: guest.y,
    })?;
    let width = u32::try_from(guest.width).map_err(|_| Refusal::OutOfRange {
        field: "width",
        value: guest.width,
    })?;
    let height = u32::try_from(guest.height).map_err(|_| Refusal::OutOfRange {
        field: "height",
        value: guest.height,
    })?;
    Ok(vk::Rect2D {
        offset: vk::Offset2D {
            x: offset_x,
            y: offset_y,
        },
        extent: vk::Extent2D { width, height },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::render::FloatBits;
    use std::collections::BTreeSet;

    /// Every capability, and every state dynamic.
    fn all() -> RasterCell {
        RasterCell {
            depth_clamp: true,
            fill_mode_non_solid: true,
            dynamic_cull_and_winding: true,
            dynamic_polygon_mode: true,
            dynamic_depth_clamp: true,
        }
    }

    /// Every capability, and nothing dynamic — the host that bakes all four
    /// states into the pipeline.
    fn baked() -> RasterCell {
        RasterCell {
            dynamic_cull_and_winding: false,
            dynamic_polygon_mode: false,
            dynamic_depth_clamp: false,
            ..all()
        }
    }

    /// A guest that set all four states away from their defaults, so a plan
    /// that dropped one on the floor is visible rather than accidentally
    /// right.
    const ALL_SET: GuestRasterState = GuestRasterState {
        cull_mode: MTL_CULL_MODE_BACK,
        winding: MTL_WINDING_COUNTER_CLOCKWISE,
        depth_clip_mode: MTL_DEPTH_CLIP_MODE_CLAMP,
        fill_mode: MTL_TRIANGLE_FILL_MODE_LINES,
    };

    fn guest_viewport(x: f64, y: f64, width: f64, height: f64) -> Viewport {
        Viewport {
            origin_x_bits: x.to_bits(),
            origin_y_bits: y.to_bits(),
            width_bits: width.to_bits(),
            height_bits: height.to_bits(),
            z_near_bits: 0.0f64.to_bits(),
            z_far_bits: 1.0f64.to_bits(),
        }
    }

    /// A viewport count is checked against the host that will build the
    /// pipeline, and against both of the two facts that bound it.
    ///
    /// The regression: nothing checked it at all.
    /// `pipeline::GraphicsKey::viewports` was a bare `u32` refused only for
    /// being zero, so a key naming three viewports produced a
    /// `VkPipelineViewportStateCreateInfo` with `viewportCount: 3` on a host
    /// with `multiViewport` off --- invalid use
    /// (VUID-VkPipelineViewportStateCreateInfo-viewportCount-01216) that no
    /// census fact could have caught, because the census carried neither the
    /// feature nor the limit.
    #[test]
    fn a_viewport_count_is_admitted_by_the_host_or_it_does_not_exist() {
        // The one slot every device has needs no capability.
        assert_eq!(
            viewport_slots(1, ViewportCell::SINGLE),
            Ok(ViewportSlots::ONE)
        );
        assert_eq!(ViewportSlots::ONE.count(), 1);

        // Zero rasterizes nothing and is not a pipeline.
        assert_eq!(viewport_slots(0, MULTI), Err(Refusal::NoViewport));
        assert_eq!(
            viewport_slots(0, ViewportCell::SINGLE).unwrap_err().slug(),
            "vk_raster_no_viewport"
        );

        // The feature and the limit refuse differently, and the feature is
        // asked first: a host without it reports a limit of one, so the limit
        // alone would report "the limit is one" and lose the fact that this
        // host has no viewport arrays at all.
        let refused = viewport_slots(4, ViewportCell::SINGLE).expect_err("no multiViewport");
        assert_eq!(
            refused,
            Refusal::ViewportSlots {
                requested: 4,
                limit: 1,
                multi_viewport: false,
            }
        );
        assert_eq!(refused.slug(), "vk_raster_viewport_slots");
        assert!(refused.to_string().contains("multi_viewport=0"));

        let over = viewport_slots(17, MULTI).expect_err("past maxViewports");
        assert_eq!(
            over,
            Refusal::ViewportSlots {
                requested: 17,
                limit: 16,
                multi_viewport: true,
            }
        );
        assert!(over.to_string().contains("multi_viewport=1"));

        // And the whole admitted range on a capable host, up to the limit and
        // not past it.
        for requested in 1..=MULTI.max_viewports {
            assert_eq!(
                viewport_slots(requested, MULTI).map(ViewportSlots::count),
                Ok(requested)
            );
        }
    }

    /// A host that offers viewport arrays.
    const MULTI: ViewportCell = ViewportCell {
        multi_viewport: true,
        max_viewports: 16,
    };

    #[test]
    fn each_cull_ordinal_is_its_own_face_and_nothing_else_is_a_face() {
        assert_eq!(cull_mode(MTL_CULL_MODE_NONE), Ok(vk::CullModeFlags::NONE));
        assert_eq!(cull_mode(MTL_CULL_MODE_FRONT), Ok(vk::CullModeFlags::FRONT));
        assert_eq!(cull_mode(MTL_CULL_MODE_BACK), Ok(vk::CullModeFlags::BACK));
        for ordinal in [3u64, 4, u64::MAX] {
            // Not folded onto `NONE`, which would draw the back faces the
            // guest culled.
            assert_eq!(
                cull_mode(ordinal),
                Err(Refusal::UnknownOrdinal {
                    state: "cull_mode",
                    ordinal,
                })
            );
        }
    }

    #[test]
    fn each_winding_is_its_own_and_nothing_else_is_a_winding() {
        assert_eq!(
            front_face(MTL_WINDING_CLOCKWISE),
            Ok(vk::FrontFace::CLOCKWISE)
        );
        assert_eq!(
            front_face(MTL_WINDING_COUNTER_CLOCKWISE),
            Ok(vk::FrontFace::COUNTER_CLOCKWISE)
        );
        assert!(front_face(2).is_err());
    }

    #[test]
    fn the_default_of_each_capability_state_needs_no_capability() {
        let none = RasterCell::default();
        // A host with neither feature still runs every guest that does not ask
        // for one.
        assert_eq!(depth_clamp(MTL_DEPTH_CLIP_MODE_CLIP, none), Ok(false));
        assert_eq!(
            polygon_mode(MTL_TRIANGLE_FILL_MODE_FILL, none),
            Ok(vk::PolygonMode::FILL)
        );
    }

    #[test]
    fn clamping_and_wireframe_refuse_where_the_feature_is_absent() {
        let none = RasterCell::default();
        assert_eq!(
            depth_clamp(MTL_DEPTH_CLIP_MODE_CLAMP, none),
            Err(Refusal::NoDepthClamp)
        );
        assert_eq!(
            polygon_mode(MTL_TRIANGLE_FILL_MODE_LINES, none),
            Err(Refusal::NoNonSolidFill)
        );
        // And are honoured where it is present, so neither refusal is
        // unconditional.
        assert_eq!(depth_clamp(MTL_DEPTH_CLIP_MODE_CLAMP, all()), Ok(true));
        assert_eq!(
            polygon_mode(MTL_TRIANGLE_FILL_MODE_LINES, all()),
            Ok(vk::PolygonMode::LINE)
        );
    }

    #[test]
    fn an_unknown_ordinal_refuses_as_that_and_not_as_a_missing_capability() {
        // The distinction matters: one is a decode or contract gap and the
        // other is a host that cannot do what the guest asked.
        assert!(matches!(
            depth_clamp(7, all()),
            Err(Refusal::UnknownOrdinal { .. })
        ));
        assert!(matches!(
            polygon_mode(7, all()),
            Err(Refusal::UnknownOrdinal { .. })
        ));
        // And on a host with neither feature, an unknown ordinal is still an
        // unknown ordinal rather than the capability refusal.
        assert!(matches!(
            polygon_mode(7, RasterCell::default()),
            Err(Refusal::UnknownOrdinal { .. })
        ));
    }

    #[test]
    fn the_viewport_flips_y_and_the_bottom_edge_is_where_it_starts() {
        // A 800x600 viewport at the origin. Vulkan measures from `y` with a
        // negative height, so `y` is the bottom edge and the covered band is
        // still 0..600.
        let flipped = viewport(guest_viewport(0.0, 0.0, 800.0, 600.0)).expect("finite");
        assert!((flipped.x - 0.0).abs() < f32::EPSILON);
        assert!((flipped.y - 600.0).abs() < f32::EPSILON);
        assert!((flipped.width - 800.0).abs() < f32::EPSILON);
        assert!((flipped.height + 600.0).abs() < f32::EPSILON);
        assert!(flipped.height < 0.0, "the flip is the negative height");
        // The band the viewport covers is unchanged; only its direction is
        // reversed.
        assert!((flipped.y + flipped.height - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn an_offset_viewport_keeps_its_band_after_the_flip() {
        // 100..340 vertically, which must still be 100..340 after the flip.
        let flipped = viewport(guest_viewport(50.0, 100.0, 200.0, 240.0)).expect("finite");
        assert!((flipped.y - 340.0).abs() < f32::EPSILON);
        assert!((flipped.y + flipped.height - 100.0).abs() < f32::EPSILON);
        assert!((flipped.x - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn the_depth_range_is_carried_and_not_flipped() {
        // Metal and Vulkan both use [0, 1] depth, so this is a narrowing and
        // never a remap. A guest that reversed its own range keeps it.
        let mut guest = guest_viewport(0.0, 0.0, 1.0, 1.0);
        guest.z_near_bits = 1.0f64.to_bits();
        guest.z_far_bits = 0.0f64.to_bits();
        let flipped = viewport(guest).expect("finite");
        assert!((flipped.min_depth - 1.0).abs() < f32::EPSILON);
        assert!((flipped.max_depth - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_viewport_that_is_not_a_number_refuses_rather_than_rasterizing() {
        for bits in [f64::NAN.to_bits(), f64::INFINITY.to_bits()] {
            let mut guest = guest_viewport(0.0, 0.0, 1.0, 1.0);
            guest.width_bits = bits;
            assert!(matches!(
                viewport(guest),
                Err(Refusal::OutOfRange { field: "width", .. })
            ));
        }
    }

    /// The failure this exists to prevent: `VkViewport::width` must be greater
    /// than zero, and a zero or negative height is the same degenerate
    /// rectangle on the other axis. Production's own draw validation refuses
    /// it as `NonPositiveViewport`; this rail narrowed the numbers, checked
    /// them for finiteness, and recorded the rectangle.
    #[test]
    fn a_viewport_with_no_area_refuses_rather_than_rasterizing_nothing() {
        assert!(
            viewport(guest_viewport(0.0, 0.0, 1.0, 1.0)).is_ok(),
            "one pixel is an area"
        );
        for (width, height) in [
            (0.0, 600.0),
            (800.0, 0.0),
            (0.0, 0.0),
            (-800.0, 600.0),
            (800.0, -600.0),
        ] {
            let guest = guest_viewport(0.0, 0.0, width, height);
            assert_eq!(
                viewport(guest).expect_err("no area"),
                Refusal::NonPositiveViewport {
                    width_bits: width.to_bits(),
                    height_bits: height.to_bits(),
                },
                "{width}x{height}"
            );
        }
        // A width or height smaller than a pixel is still an area, and a
        // rectangle a guest may legitimately draw into.
        assert!(viewport(guest_viewport(0.0, 0.0, 0.5, 0.25)).is_ok());

        // And a dimension positive in the guest's arithmetic and zero in the
        // field the driver reads is the degenerate case, not the legal one.
        // This is why the check is after the narrowing and not before it.
        assert!(1e-300_f64 > 0.0 && 1e-300_f64 as f32 == 0.0, "the premise");
        assert_eq!(
            viewport(guest_viewport(0.0, 0.0, 1e-300, 600.0))
                .expect_err("positive as a double, zero as a float"),
            Refusal::NonPositiveViewport {
                width_bits: 1e-300_f64.to_bits(),
                height_bits: 600.0_f64.to_bits(),
            }
        );
    }

    /// And a dimension that is not a number stays the range failure it is.
    ///
    /// The two checks are on the same two fields and one of them runs first.
    /// Reading the sign before the narrowing would report a NaN width as a
    /// rectangle with no area, which names the wrong thing about it --- a NaN
    /// compares false against every bound, including this one.
    ///
    /// The height's NaN surfaces on `origin_y` rather than on `height`,
    /// because the bottom edge is `y + height` and is narrowed first. That is
    /// the sum genuinely not being a number, and the field it names is the one
    /// whose value could not be produced.
    #[test]
    fn a_dimension_that_is_not_a_number_outranks_the_area_check() {
        for (field, set) in [
            (
                "width",
                (|v: &mut Viewport, b| v.width_bits = b) as fn(&mut Viewport, u64),
            ),
            ("origin_y", |v: &mut Viewport, b| v.height_bits = b),
        ] {
            let mut guest = guest_viewport(0.0, 0.0, 1.0, 1.0);
            set(&mut guest, f64::NAN.to_bits());
            assert!(
                matches!(
                    viewport(guest).expect_err("not a number"),
                    Refusal::OutOfRange { field: named, .. } if named == field
                ),
                "{field}"
            );
        }
    }

    /// The two of six nothing looked at. A NaN depth bound reaches a driver as
    /// a NaN, and the range is what every fragment's depth test is against.
    #[test]
    fn a_depth_bound_that_is_not_a_number_refuses_like_every_other_field() {
        for (field, set) in [
            (
                "z_near",
                (|v: &mut Viewport, b| v.z_near_bits = b) as fn(&mut Viewport, u64),
            ),
            ("z_far", |v: &mut Viewport, b| v.z_far_bits = b),
        ] {
            for bits in [f64::NAN.to_bits(), f64::INFINITY.to_bits()] {
                let mut guest = guest_viewport(0.0, 0.0, 1.0, 1.0);
                set(&mut guest, bits);
                assert_eq!(
                    viewport(guest).expect_err("not a number"),
                    Refusal::OutOfRange { field, value: bits }
                );
            }
        }
    }

    /// A finite double past `f32::MAX` casts to an infinity rather than
    /// saturating, so the input being finite proves nothing about the field it
    /// lands in --- which is what the doc claimed and the code did not check.
    #[test]
    fn a_finite_dimension_that_does_not_fit_an_f32_refuses_rather_than_becoming_infinite() {
        let huge = 1.0e300_f64;
        assert!(huge.is_finite() && !(huge as f32).is_finite());
        let mut guest = guest_viewport(0.0, 0.0, 1.0, 1.0);
        guest.width_bits = huge.to_bits();
        assert_eq!(
            viewport(guest).expect_err("infinite once narrowed"),
            Refusal::OutOfRange {
                field: "width",
                value: huge.to_bits(),
            }
        );

        // The bottom edge is a sum, so two halves that each fit can produce one
        // that does not. Nothing checking the inputs alone can see this.
        let half = f64::from(f32::MAX) * 0.75;
        assert!((half as f32).is_finite() && !((half + half) as f32).is_finite());
        let composed = guest_viewport(0.0, half, 1.0, half);
        assert_eq!(
            viewport(composed).expect_err("the sum does not fit"),
            Refusal::OutOfRange {
                field: "origin_y",
                value: (half + half).to_bits(),
            }
        );
    }

    #[test]
    fn a_scissor_is_not_flipped() {
        let rect = scissor(ScissorRect {
            x: 10,
            y: 20,
            width: 30,
            height: 40,
        })
        .expect("in range");
        // Same origin, same corner: flipping this as well would move the
        // rectangle off the geometry the viewport just placed.
        assert_eq!(rect.offset, vk::Offset2D { x: 10, y: 20 });
        assert_eq!(
            rect.extent,
            vk::Extent2D {
                width: 30,
                height: 40
            }
        );
    }

    #[test]
    fn a_scissor_outside_the_native_fields_refuses_rather_than_truncating() {
        assert_eq!(
            scissor(ScissorRect {
                x: u64::from(u32::MAX) + 1,
                y: 0,
                width: 1,
                height: 1,
            }),
            Err(Refusal::OutOfRange {
                field: "x",
                value: u64::from(u32::MAX) + 1,
            })
        );
        assert!(matches!(
            scissor(ScissorRect {
                x: 0,
                y: 0,
                width: u64::from(u32::MAX) + 1,
                height: 1,
            }),
            Err(Refusal::OutOfRange { field: "width", .. })
        ));
    }

    #[test]
    fn float_state_narrows_without_reinterpretation() {
        // The bit-pattern types exist so a state table can compare; they are
        // not a wire encoding to be passed through.
        assert!((FloatBits::from_f32(0.25).to_f32() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::UnknownOrdinal {
                state: "cull_mode",
                ordinal: 9,
            },
            Refusal::NoDepthClamp,
            Refusal::NoNonSolidFill,
            Refusal::OutOfRange {
                field: "x",
                value: 1,
            },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_raster_"));
        }
    }

    /// Nothing dynamic: every state the guest set lands in the pipeline, and
    /// the encoder is asked for none of them.
    #[test]
    fn a_host_with_no_dynamic_state_bakes_all_four_states_into_the_pipeline() {
        let plan = plan(ALL_SET, baked()).expect("every capability is present");
        assert_eq!(plan.state.cull_mode, vk::CullModeFlags::BACK);
        assert_eq!(plan.state.front_face, vk::FrontFace::COUNTER_CLOCKWISE);
        assert_eq!(plan.state.polygon_mode, vk::PolygonMode::LINE);
        assert!(plan.state.depth_clamp_enable);
        assert_eq!(plan.dynamic, DynamicRaster::default());
        // Only the unconditional two: the depth bias, whose enable this rail
        // always sets, and the line width, which is 1.0 core dynamic state and
        // so is never baked on any host.
        assert_eq!(
            plan.state.dynamic.states(),
            vec![vk::DynamicState::DEPTH_BIAS, vk::DynamicState::LINE_WIDTH]
        );
    }

    /// Every state dynamic: the guest's values move to the encoder and the
    /// pipeline holds the default in each place.
    ///
    /// This is the whole point of the cell. If the guest's value stayed in the
    /// pipeline state as well, it would still be a cache dimension and the
    /// dynamic state would have bought nothing — so the assertion is that the
    /// pipeline members are *not* what the guest asked for.
    #[test]
    fn a_dynamic_state_leaves_the_pipeline_holding_the_default_and_not_the_guests_value() {
        let plan = plan(ALL_SET, all()).expect("every capability is present");
        assert_eq!(plan.state.cull_mode, vk::CullModeFlags::NONE);
        assert_eq!(plan.state.front_face, vk::FrontFace::CLOCKWISE);
        assert_eq!(plan.state.polygon_mode, vk::PolygonMode::FILL);
        assert!(!plan.state.depth_clamp_enable);
        assert_eq!(
            plan.dynamic,
            DynamicRaster {
                cull_mode: Some(vk::CullModeFlags::BACK),
                front_face: Some(vk::FrontFace::COUNTER_CLOCKWISE),
                polygon_mode: Some(vk::PolygonMode::LINE),
                depth_clamp_enable: Some(true),
            }
        );
    }

    /// The one claim a cache rests on: on a fully dynamic host, every guest
    /// rasterizer state plans to one pipeline state, so the whole sixteen-way
    /// product compiles once.
    #[test]
    fn every_guest_state_collapses_to_one_pipeline_where_all_four_are_dynamic() {
        let mut states = BTreeSet::new();
        let mut baked_states = BTreeSet::new();
        let mut count = 0;
        for cull_mode in [MTL_CULL_MODE_NONE, MTL_CULL_MODE_FRONT, MTL_CULL_MODE_BACK] {
            for winding in [MTL_WINDING_CLOCKWISE, MTL_WINDING_COUNTER_CLOCKWISE] {
                for depth_clip_mode in [MTL_DEPTH_CLIP_MODE_CLIP, MTL_DEPTH_CLIP_MODE_CLAMP] {
                    for fill_mode in [MTL_TRIANGLE_FILL_MODE_FILL, MTL_TRIANGLE_FILL_MODE_LINES] {
                        let guest = GuestRasterState {
                            cull_mode,
                            winding,
                            depth_clip_mode,
                            fill_mode,
                        };
                        let dynamic = plan(guest, all()).expect("capable host");
                        states.insert(format!("{:?}", dynamic.state));
                        // The same guest states against a host that bakes
                        // them. This is what makes the count above a saving
                        // rather than a plan that forgot to read the guest:
                        // the two hosts must disagree about how many pipelines
                        // this product needs.
                        let baked = plan(guest, baked()).expect("capable host");
                        baked_states.insert(format!("{:?}", baked.state));
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 24, "the whole product was walked");
        // Injective the other way: baking gives a pipeline per state, and the
        // three cull modes, two windings, two clip modes and two fill modes
        // are all distinguishable. Twenty-four compilations against one.
        assert_eq!(baked_states.len(), 24, "baking collapses nothing");
        assert_eq!(
            states.len(),
            1,
            "one pipeline serves every rasterizer state"
        );
    }

    /// A dynamic state still needs its feature. `vkCmdSetPolygonModeEXT` with
    /// `LINE` is as invalid without `fillModeNonSolid` as the pipeline member
    /// is, so moving the state to the encoder must not move the refusal.
    #[test]
    fn making_a_state_dynamic_does_not_make_its_capability_free() {
        // A host that offers the dynamic member and not the feature it needs.
        // Contrived, and exactly the host the refusal exists for.
        let cell = RasterCell {
            depth_clamp: false,
            fill_mode_non_solid: false,
            dynamic_cull_and_winding: true,
            dynamic_polygon_mode: true,
            dynamic_depth_clamp: true,
        };
        assert_eq!(
            plan(
                GuestRasterState {
                    fill_mode: MTL_TRIANGLE_FILL_MODE_LINES,
                    ..GuestRasterState::DEFAULT
                },
                cell
            ),
            Err(Refusal::NoNonSolidFill)
        );
        assert_eq!(
            plan(
                GuestRasterState {
                    depth_clip_mode: MTL_DEPTH_CLIP_MODE_CLAMP,
                    ..GuestRasterState::DEFAULT
                },
                cell
            ),
            Err(Refusal::NoDepthClamp)
        );
        // And the default of each still runs on it, dynamic or not.
        assert!(plan(GuestRasterState::DEFAULT, cell).is_ok());
    }

    /// An unrecognised ordinal refuses whether or not its state is dynamic.
    /// The parse is not skipped because the value is going to a command.
    #[test]
    fn an_unknown_ordinal_refuses_on_the_dynamic_path_too() {
        for cell in [all(), baked()] {
            assert!(plan(
                GuestRasterState {
                    cull_mode: 7,
                    ..GuestRasterState::DEFAULT
                },
                cell
            )
            .is_err());
            assert!(plan(
                GuestRasterState {
                    winding: 7,
                    ..GuestRasterState::DEFAULT
                },
                cell
            )
            .is_err());
        }
    }

    /// The two halves cannot disagree: a member is a placeholder in the
    /// pipeline state exactly where the encoder is asked to supply it.
    #[test]
    fn a_member_is_dynamic_in_exactly_one_of_the_two_halves() {
        // Each of the eight subsets a host can offer, against a guest that set
        // every state away from its default.
        for bits in 0u8..8 {
            let cell = RasterCell {
                depth_clamp: true,
                fill_mode_non_solid: true,
                dynamic_cull_and_winding: bits & 1 != 0,
                dynamic_polygon_mode: bits & 2 != 0,
                dynamic_depth_clamp: bits & 4 != 0,
            };
            let p = plan(ALL_SET, cell).expect("every capability is present");
            let d = p.state.dynamic;
            assert_eq!(d, RasterDynamic::of(cell));
            assert_eq!(p.dynamic.cull_mode.is_some(), d.cull_and_winding);
            assert_eq!(p.dynamic.front_face.is_some(), d.cull_and_winding);
            assert_eq!(p.dynamic.polygon_mode.is_some(), d.polygon_mode);
            assert_eq!(p.dynamic.depth_clamp_enable.is_some(), d.depth_clamp_enable);
            // The placeholder is the default, and the guest asked for
            // something else, so "dynamic" and "baked" are visibly different
            // in the pipeline state as well.
            assert_eq!(
                p.state.cull_mode == vk::CullModeFlags::NONE,
                d.cull_and_winding
            );
            assert_eq!(
                p.state.polygon_mode == vk::PolygonMode::FILL,
                d.polygon_mode
            );
            assert_eq!(!p.state.depth_clamp_enable, d.depth_clamp_enable);
        }
    }

    /// The dynamic-state list has no duplicates and grows only with the cell.
    #[test]
    fn the_dynamic_state_list_names_each_member_once_and_two_of_them_always() {
        for bits in 0u8..8 {
            let dynamic = RasterDynamic {
                cull_and_winding: bits & 1 != 0,
                polygon_mode: bits & 2 != 0,
                depth_clamp_enable: bits & 4 != 0,
            };
            let states = dynamic.states();
            let unique: BTreeSet<_> = states.iter().map(|s| s.as_raw()).collect();
            assert_eq!(unique.len(), states.len(), "{dynamic:?} repeats a state");
            assert!(states.contains(&vk::DynamicState::DEPTH_BIAS));
            // Every pipeline, on every host: `VK_DYNAMIC_STATE_LINE_WIDTH` is
            // 1.0 core and no cell reaches it, so a pipeline that failed to
            // declare it would be one whose `vkCmdSetLineWidth` is invalid
            // use and whose baked 1.0 would silently ignore the guest.
            assert!(states.contains(&vk::DynamicState::LINE_WIDTH));
            // Two unconditional entries, two for the pair, one each for the
            // other two.
            let expected = 2
                + usize::from(dynamic.cull_and_winding) * 2
                + usize::from(dynamic.polygon_mode)
                + usize::from(dynamic.depth_clamp_enable);
            assert_eq!(states.len(), expected);
        }
        assert_eq!(
            RasterDynamic::NONE.states(),
            vec![vk::DynamicState::DEPTH_BIAS, vk::DynamicState::LINE_WIDTH]
        );
    }

    /// Depth bias is enabled on every pipeline this rail builds, and its three
    /// values start at zero. Metal has no pipeline-level enable, so a clear
    /// one would silently drop `setDepthBias:` — see the module doc.
    #[test]
    fn every_pipeline_permits_a_depth_bias_and_starts_at_none() {
        for cell in [all(), baked(), RasterCell::default()] {
            let p = plan(GuestRasterState::DEFAULT, cell).expect("the defaults need no feature");
            assert!(p.state.depth_bias_enable);
            let native = p.state.native();
            assert_eq!(native.depth_bias_enable, vk::TRUE);
            assert_eq!(native.depth_bias_constant_factor, 0.0);
            assert_eq!(native.depth_bias_clamp, 0.0);
            assert_eq!(native.depth_bias_slope_factor, 0.0);
            // Metal has neither, and both would drop or widen geometry.
            assert_eq!(native.rasterizer_discard_enable, vk::FALSE);
            assert_eq!(native.line_width, 1.0);
        }
    }

    /// The native structure carries what the plan says, field for field.
    #[test]
    fn the_native_structure_is_the_plan_and_not_the_guests_state() {
        let p = plan(ALL_SET, baked()).expect("every capability is present");
        let native = p.state.native();
        assert_eq!(native.cull_mode, vk::CullModeFlags::BACK);
        assert_eq!(native.front_face, vk::FrontFace::COUNTER_CLOCKWISE);
        assert_eq!(native.polygon_mode, vk::PolygonMode::LINE);
        assert_eq!(native.depth_clamp_enable, vk::TRUE);

        let dyn_ = plan(ALL_SET, all()).expect("every capability is present");
        let native = dyn_.state.native();
        assert_eq!(native.cull_mode, vk::CullModeFlags::NONE);
        assert_eq!(native.depth_clamp_enable, vk::FALSE);
    }

    /// A device that advertises nothing.
    const WIDE: LineWidthCell = LineWidthCell {
        wide_lines: true,
        range: [0.5, 8.0],
    };

    /// The effective polygon mode is one question with two storage sites, and
    /// asking the wrong one reports a filled draw as a wireframe or the other
    /// way round. Both hosts, one answer.
    #[test]
    fn the_effective_polygon_mode_is_the_guests_on_a_baking_and_a_dynamic_host() {
        for (guest_fill, expected) in [
            (MTL_TRIANGLE_FILL_MODE_FILL, vk::PolygonMode::FILL),
            (MTL_TRIANGLE_FILL_MODE_LINES, vk::PolygonMode::LINE),
        ] {
            let guest = GuestRasterState {
                fill_mode: guest_fill,
                ..GuestRasterState::DEFAULT
            };
            for cell in [all(), baked()] {
                let p = plan(guest, cell).expect("every capability is present");
                assert_eq!(p.polygon_mode(), expected, "{cell:?}");
            }
        }
        // And the two halves really do differ, so the method is not reading a
        // value both of them happen to hold.
        let wire = GuestRasterState {
            fill_mode: MTL_TRIANGLE_FILL_MODE_LINES,
            ..GuestRasterState::DEFAULT
        };
        let dynamic = plan(wire, all()).expect("every capability is present");
        assert_eq!(dynamic.state.polygon_mode, vk::PolygonMode::FILL);
        assert_eq!(dynamic.dynamic.polygon_mode, Some(vk::PolygonMode::LINE));
    }

    /// The joint fact. A triangle draw rasterizes lines only where the fill
    /// mode says so; a line-class draw always does; a point-class draw never
    /// does.
    #[test]
    fn only_a_line_topology_or_a_wireframe_triangle_rasterizes_lines() {
        for mode in [vk::PolygonMode::FILL, vk::PolygonMode::LINE] {
            assert!(!rasterizes_lines(mode, TopologyClass::Point));
            assert!(rasterizes_lines(mode, TopologyClass::Line));
        }
        assert!(!rasterizes_lines(
            vk::PolygonMode::FILL,
            TopologyClass::Triangle
        ));
        assert!(rasterizes_lines(
            vk::PolygonMode::LINE,
            TopologyClass::Triangle
        ));
    }

    /// A draw that rasterizes nothing linear takes the default whatever the
    /// guest set and whatever the host offers — including on a host that would
    /// have refused the width. This is what stops a `setLineWidth:` before a
    /// filled scene from refusing every draw in it.
    #[test]
    fn a_draw_that_draws_no_lines_takes_the_default_and_is_never_refused() {
        for width in [None, Some(1.0), Some(4.0), Some(-3.0), Some(f32::NAN)] {
            for cell in [LineWidthCell::NARROW, WIDE] {
                assert_eq!(line_width(width, false, cell), Ok(DEFAULT_LINE_WIDTH));
            }
        }
    }

    /// The default needs no feature, which is what makes every guest that
    /// never calls `setLineWidth:` runnable on the barest device.
    #[test]
    fn the_default_width_is_served_by_a_device_without_wide_lines() {
        assert_eq!(
            line_width(None, true, LineWidthCell::NARROW),
            Ok(DEFAULT_LINE_WIDTH)
        );
        assert_eq!(
            line_width(Some(1.0), true, LineWidthCell::NARROW),
            Ok(DEFAULT_LINE_WIDTH)
        );
    }

    /// Anything else needs `wideLines`, and is refused by name rather than
    /// clamped to 1.0 — a clamp would draw a hairline where the guest asked
    /// for a thick one and report nothing.
    #[test]
    fn a_non_default_width_without_wide_lines_is_refused_and_not_clamped() {
        assert_eq!(
            line_width(Some(4.0), true, LineWidthCell::NARROW),
            Err(Refusal::NoWideLines {
                width_bits: 4.0f32.to_bits()
            })
        );
        assert_eq!(
            line_width(Some(4.0), true, LineWidthCell::NARROW)
                .unwrap_err()
                .slug(),
            "vk_raster_no_wide_lines"
        );
    }

    /// Inside the range it is the guest's own number, bit for bit: this device
    /// scales nothing, so a float that survives the checks must survive them
    /// unchanged.
    #[test]
    fn a_width_inside_the_range_reaches_the_encoder_unchanged() {
        for width in [0.5f32, 1.5, 2.5, 8.0] {
            assert_eq!(
                line_width(Some(width), true, WIDE).map(f32::to_bits),
                Ok(width.to_bits()),
            );
        }
    }

    /// Outside it — and a NaN, which fails both comparisons rather than
    /// passing one — refuses with the bounds in the message.
    #[test]
    fn a_width_outside_the_range_or_a_nan_is_refused_with_its_bounds() {
        for width in [0.25f32, 8.5, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                line_width(Some(width), true, WIDE),
                Err(Refusal::LineWidthOutOfRange {
                    width_bits: width.to_bits(),
                    min_bits: 0.5f32.to_bits(),
                    max_bits: 8.0f32.to_bits(),
                }),
                "{width}"
            );
        }
        let refusal = line_width(Some(9.0), true, WIDE).unwrap_err();
        assert_eq!(refusal.slug(), "vk_raster_line_width_out_of_range");
        assert_eq!(
            refusal.to_string(),
            "vk_raster_line_width_out_of_range width=9 min=0.5 max=8"
        );
    }

    /// A cell built from a device that reports no `wideLines` still has a
    /// range containing the default, because Vulkan requires it to. Without
    /// that the narrow cell would refuse the width every guest gets by
    /// default.
    #[test]
    fn the_narrow_cell_serves_the_default_and_nothing_else() {
        assert_eq!(LineWidthCell::default(), LineWidthCell::NARROW);
        assert_eq!(
            line_width(Some(DEFAULT_LINE_WIDTH), true, LineWidthCell::NARROW),
            Ok(DEFAULT_LINE_WIDTH)
        );
        assert!(line_width(Some(1.0001), true, LineWidthCell::NARROW).is_err());
    }
}
