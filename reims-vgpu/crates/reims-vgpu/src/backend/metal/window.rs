//! The Metal rail's half of the host-owned presentation window
//! ([[host-window]]): a `CAMetalLayer` on the window's own `NSView`, and the
//! blit that puts a published guest frame into its next drawable.
//!
//! The mirror of `backend::vulkan::engine::window_present`, and the two are
//! reached through the same six [`crate::backend::Backend`] methods, so the
//! window itself never learns which one it is talking to. That is the whole
//! point of running this rail: the same guest stream, the same window, the same
//! input, with only the executor changed — which is what makes a wrong frame
//! attributable to `metal2vulkan` rather than to this device.
//!
//! # What this rail presents from
//!
//! CPU bytes, always. The Metal rail retains no texture past the encode that
//! produced it — every colour target in [`super::render`] is created,
//! CPU-seeded, read back and dropped inside one `render_core_mrt` — so there is
//! no resident for a present to name, and [`crate::backend::window::WindowResident`]
//! carries no Metal variant. That is the same path the QEMU display took: the
//! frame already crossed host memory before this window existed, and nothing
//! here adds a copy that the Cocoa path did not also pay.
//!
//! # Why a render pass and not a blit encoder
//!
//! The guest frame has to be aspect-fitted into the drawable, and a
//! `MTLBlitCommandEncoder` copies without scaling. Vulkan's presenter uses
//! `vkCmdBlitImage`, which does scale; the Metal equivalent is a one-triangle
//! render pass with the viewport set to the fitted rectangle and the clear
//! colour showing through the letterbox bars. It also puts the fit in this
//! file, where [`crate::backend::window::viewport::aspect_fit`] is the same
//! function the pointer path maps through — presentation and input move as one
//! unit, or a click lands where the pixel is not.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use core_graphics_types::geometry::{CGPoint, CGRect, CGSize};
use metal::{
    Device, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRegion,
    MTLStoreAction, MTLTextureType, MTLTextureUsage, MTLViewport, MetalLayer, RenderPassDescriptor,
    RenderPipelineDescriptor, RenderPipelineState, SamplerState, Texture, TextureDescriptor,
};
use objc::runtime::{Object, YES};
use objc::{msg_send, sel, sel_impl};
use raw_window_handle::RawWindowHandle;

use super::raw_metal;
use super::runtime::{cached_default_sampler, system_device, thread_queue};
use crate::backend::window::viewport::aspect_fit;
use crate::backend::window::{WindowCpuFrame, WindowPresentOutcome, WindowSurface};
use crate::observe::Decline;

/// The layer's and the staging texture's pixel format.
///
/// One constant for both, because they are one decision: the frames this device
/// publishes are BGRA8 (`crate::backend::window::WindowCpuFrame`), and a
/// drawable in any other format would need a conversion that nothing here
/// performs. The render pipeline is built against it too, and Metal rejects a
/// pipeline whose colour attachment format differs from the attachment it is
/// encoded into — so a mismatch is a refusal at attach rather than a wrong
/// picture at present.
const SURFACE_FORMAT: MTLPixelFormat = MTLPixelFormat::BGRA8Unorm;

/// Bytes per pixel of [`SURFACE_FORMAT`], and therefore of every published
/// frame.
const SURFACE_BYTES_PER_PIXEL: usize = 4;

/// `kCALayerWidthSizable | kCALayerHeightSizable`, the two bits of
/// `CAAutoresizingMask` that make a sublayer track its superlayer's size.
///
/// Spelled here because CoreAnimation's constants are not in any crate this
/// build links; the values are the framework's own and are pinned by
/// `the_layer_autoresizing_mask_is_width_and_height`.
const LAYER_SIZABLE: u32 = (1 << 1) | (1 << 4);

const BLIT_VERTEX_FN: &str = "reims_window_blit_vertex";
const BLIT_FRAGMENT_FN: &str = "reims_window_blit_fragment";

/// The presenter's own blit, compiled once when it attaches.
///
/// Three vertices and a sample. The vertex stage builds a triangle that covers
/// the clip cube from `vertex_id` alone, so there is no vertex buffer to
/// allocate, bind or keep alive; the viewport the encoder sets is what confines
/// it to the fitted rectangle, and the render pass's clear colour is what fills
/// the bars outside it.
///
/// `uv` is `(0,0)` at the top left, which is where row zero of a published
/// frame is. Getting that backwards inverts the desktop, so the mapping is
/// written out rather than left to the reader: `vertex_id` 0, 1, 2 produce
/// `uv` `(0,0)`, `(2,0)`, `(0,2)` and clip positions `(-1,1)`, `(3,1)`,
/// `(-1,-3)` — the covered square's corners are `uv` `(0,0)` top-left and
/// `(1,1)` bottom-right.
const BLIT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ReimsBlitVertex {
    float4 position [[position]];
    float2 uv;
};

vertex ReimsBlitVertex reims_window_blit_vertex(uint vertex_id [[vertex_id]]) {
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    ReimsBlitVertex out;
    out.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

fragment float4 reims_window_blit_fragment(ReimsBlitVertex in [[stage_in]],
                                           texture2d<float> source [[texture(0)]],
                                           sampler source_sampler [[sampler(0)]]) {
    return source.sample(source_sampler, in.uv);
}
"#;

/// Why this rail could not attach to, or present into, the host window.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalWindowDecline {
    /// No `MTLDevice`. The same refusal every other Metal path opens with, and
    /// on this one it means the window can never be filled rather than that one
    /// frame was lost.
    NoDevice,
    /// The window handed over a handle that is not an AppKit view. Metal
    /// presents through `CAMetalLayer` and nothing else, so there is no
    /// fallback to take — this is a host this rail does not run on.
    NotAppKitWindow,
    /// The presenter's own blit did not compile. Its source is a constant in
    /// this file, so this is a driver or OS refusal rather than anything the
    /// guest did.
    ShaderCompile(String),
    /// The compiled library does not export one of the two entry points this
    /// file names. Separate from [`Self::ShaderCompile`] because it is a
    /// rename that got away, not a compile failure.
    ShaderFunctionMissing(&'static str),
    /// `newRenderPipelineStateWithDescriptor:` refused the blit pipeline.
    PipelineCreate(String),
    /// The staging texture for a frame of this geometry could not be created —
    /// the device declining an allocation.
    StagingTexture { width: u32, height: u32 },
    /// The queue declined to issue a command buffer.
    CommandBuffer,
    /// The drawable's render pass produced no encoder.
    RenderEncoder,
    /// The present's command buffer completed in an error state.
    Submit(String),
    /// No presenter. On a running window that means it was torn down, or that a
    /// present arrived on a thread that does not own it.
    NotAttached,
}

impl Decline for MetalWindowDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoDevice => "metal_window_no_device",
            Self::NotAppKitWindow => "metal_window_not_appkit",
            Self::ShaderCompile(_) => "metal_window_shader_compile",
            Self::ShaderFunctionMissing(_) => "metal_window_shader_function_missing",
            Self::PipelineCreate(_) => "metal_window_pipeline_create",
            Self::StagingTexture { .. } => "metal_window_staging_texture",
            Self::CommandBuffer => "metal_window_command_buffer",
            Self::RenderEncoder => "metal_window_render_encoder",
            Self::Submit(_) => "metal_window_submit",
            Self::NotAttached => "metal_window_not_attached",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoDevice
            | Self::NotAppKitWindow
            | Self::CommandBuffer
            | Self::RenderEncoder
            | Self::NotAttached => Vec::new(),
            Self::ShaderCompile(detail) | Self::PipelineCreate(detail) | Self::Submit(detail) => {
                vec![("detail", log_safe(detail))]
            }
            Self::ShaderFunctionMissing(name) => vec![("function", (*name).to_string())],
            Self::StagingTexture { width, height } => {
                vec![("width", width.to_string()), ("height", height.to_string())]
            }
        }
    }
}

crate::observe::decline_display!(MetalWindowDecline);

impl std::error::Error for MetalWindowDecline {}

/// Fold a driver's multi-word error text into one `k=v` field.
///
/// The always-on line is whitespace-separated `k=v` pairs, so an unfolded
/// `detail=` swallows every field after it into the reader's idea of one value.
fn log_safe(detail: &str) -> String {
    detail.split_whitespace().collect::<Vec<_>>().join("_")
}

/// Whether this rail currently holds a presenter.
///
/// An atomic beside the thread-local below, for the reason Vulkan's flag is an
/// atomic: the publish path on the drain worker has to know whether a window is
/// taking frames *before* it decides what to publish, and it is not the thread
/// that owns the presenter.
static ATTACHED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// The presenter, owned by the thread that runs the window's event loop.
    ///
    /// A thread-local and not a global mutex, because that is the real
    /// ownership and this makes the compiler keep it: the layer is created
    /// against an `NSView`, and every call that touches it — attach from
    /// `resumed`, present from `draw`, resize from `Resized`, detach from
    /// `exiting` — is a `winit` callback on the loop's own thread, which on
    /// macOS is the process main thread. Nothing else may reach a `CAMetalLayer`,
    /// and with this spelling nothing else can.
    static PRESENTER: RefCell<Option<Presenter>> = const { RefCell::new(None) };
}

/// The staging texture and what it currently holds.
///
/// The sequence is what makes a forced redraw free: a resize, or a
/// self-healing draw, re-presents the frame already on the GPU rather than
/// re-uploading a framebuffer that has not changed.
struct Staged {
    texture: Texture,
    width: u32,
    height: u32,
    seq: u64,
}

struct Presenter {
    /// The `NSView` the layer hangs under.
    ///
    /// Held so a resize can re-read the backing scale factor — see
    /// [`Self::set_drawable_size`]. Not owned: the window that vended it
    /// outlives this presenter, because the window detaches from `exiting`
    /// before it drops the view, and the pointer never leaves the thread the
    /// presenter lives on (a raw pointer in a `thread_local` is exactly that
    /// claim, checked).
    view: *mut Object,
    layer: MetalLayer,
    pipeline: RenderPipelineState,
    sampler: SamplerState,
    staged: Option<Staged>,
    /// The drawable geometry the layer was last told to produce, in pixels.
    ///
    /// Held rather than read back from the layer because it is also the
    /// destination [`aspect_fit`] fits into, and a `drawableSize` read between
    /// a resize request and the layer honouring it would fit the frame to a
    /// rectangle the next drawable does not have.
    drawable: (u32, u32),
}

/// Build this rail's presenter for `surface`.
///
/// Idempotent: a presenter already attached on this thread is left alone, which
/// is what lets the window call this both at creation and to rebuild.
pub fn attach(surface: &WindowSurface) -> Result<(), MetalWindowDecline> {
    PRESENTER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_some() {
            return Ok(());
        }
        let presenter = Presenter::create(surface)?;
        *slot = Some(presenter);
        ATTACHED.store(true, Ordering::Release);
        Ok(())
    })
}

/// Whether a presenter is taking frames. Callable from any thread; see
/// [`ATTACHED`].
pub fn attached() -> bool {
    ATTACHED.load(Ordering::Acquire)
}

/// Ask the layer for drawables of a new size.
///
/// Silent without a presenter: the window resizes before it attaches and after
/// it detaches, and neither is a lost frame.
pub fn resize(width: u32, height: u32) {
    PRESENTER.with(|cell| {
        if let Some(presenter) = cell.borrow_mut().as_mut() {
            presenter.set_drawable_size(width, height);
        }
    });
}

/// Put one frame on the screen.
pub fn present(
    cpu: Option<WindowCpuFrame<'_>>,
) -> Result<WindowPresentOutcome, MetalWindowDecline> {
    PRESENTER.with(|cell| {
        let mut slot = cell.borrow_mut();
        let presenter = slot.as_mut().ok_or(MetalWindowDecline::NotAttached)?;
        presenter.present(cpu)
    })
}

/// Release the presenter while the native window is still alive.
///
/// Dropping the layer is what detaches it from the view; the view is still
/// there to be detached from, which is the ordering the window's `exiting`
/// callback exists to guarantee.
pub fn detach() {
    PRESENTER.with(|cell| {
        if cell.borrow_mut().take().is_some() {
            ATTACHED.store(false, Ordering::Release);
        }
    });
}

impl Presenter {
    fn create(surface: &WindowSurface) -> Result<Self, MetalWindowDecline> {
        let RawWindowHandle::AppKit(handle) = surface.window else {
            return Err(MetalWindowDecline::NotAppKitWindow);
        };
        let device = system_device().ok_or(MetalWindowDecline::NoDevice)?;
        let pipeline = build_blit_pipeline(device)?;
        let width = surface.width.max(1);
        let height = surface.height.max(1);

        let layer = MetalLayer::new();
        layer.set_device(device);
        layer.set_pixel_format(SURFACE_FORMAT);
        // Nothing reads a drawable back, and the guest frame covers the whole
        // window: the compositor may treat this layer as opaque and skip
        // blending it, and the driver may give it write-only storage.
        layer.set_framebuffer_only(true);
        layer.set_opaque(true);
        // The present is committed from the encode below, not handed to
        // CoreAnimation's transaction — which is what `presentsWithTransaction`
        // would require, and which would mean blocking this thread inside
        // AppKit's own commit.
        layer.set_presents_with_transaction(false);
        layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));

        // SAFETY: `ns_view` is the window's own `NSView`, valid for as long as
        // the window that vended the handle — which outlives this presenter,
        // because the window detaches from `exiting` before it drops the view.
        // Every selector here is AppKit's or CoreAnimation's, and this runs on
        // the loop's thread, which on macOS is the main thread they require.
        let view: *mut Object = handle.ns_view.as_ptr().cast();
        unsafe {
            // A **sublayer** of the view's own layer, not a replacement for it.
            // `setLayer:` makes a view layer-hosted, and `winit`'s view is
            // already layer-backed — AppKit owns that layer, sizes it, and is
            // free to replace it, so handing it a `CAMetalLayer` puts two
            // owners on one object. Hosting underneath is what MoltenVK does
            // for a `VkSurfaceKHR` on this platform, which is also why the two
            // rails end up presenting through the same kind of layer.
            let _: () = msg_send![view, setWantsLayer: YES];
            let root: *mut Object = msg_send![view, layer];
            if root.is_null() {
                return Err(MetalWindowDecline::NotAppKitWindow);
            }
            // Top-left anchor, so the frame below places the layer rather than
            // centring it on the origin.
            let _: () = msg_send![&*layer, setAnchorPoint: CGPoint::new(0.0, 0.0)];
            let bounds: CGRect = msg_send![root, bounds];
            let _: () = msg_send![&*layer, setFrame: bounds];
            // The root layer resizes with the view; this makes the sublayer
            // follow it, so a window resize never leaves the picture inset in a
            // corner while the `Resized` event is in flight.
            let _: () = msg_send![&*layer, setAutoresizingMask: LAYER_SIZABLE];
            // Points to pixels. The window hands over `inner_size`, which winit
            // reports in physical pixels, and the frame above is in points —
            // this is the factor between them, and getting it wrong scales the
            // whole desktop by the display's backing factor.
            layer.set_contents_scale(view_backing_scale(view));
            let _: () = msg_send![root, addSublayer: &*layer];
        }

        Ok(Self {
            view,
            layer,
            pipeline,
            sampler: cached_default_sampler(device),
            staged: None,
            drawable: (width, height),
        })
    }

    /// Match the drawables to the size the window system actually granted.
    ///
    /// The scale is re-read with the size, and that is the case worth naming:
    /// dragging the window between a Retina display and an external one changes
    /// the backing factor, winit answers with a `Resized` in the new display's
    /// pixels, and a layer still holding the old `contentsScale` would show the
    /// whole desktop scaled by the ratio between them. One event, one pair of
    /// facts, set together.
    fn set_drawable_size(&mut self, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size == self.drawable {
            return;
        }
        self.drawable = size;
        // SAFETY: `view` is this presenter's own, alive for as long as it — see
        // the field's doc.
        let scale = unsafe { view_backing_scale(self.view) };
        self.layer.set_contents_scale(scale);
        self.layer
            .set_drawable_size(CGSize::new(f64::from(size.0), f64::from(size.1)));
    }

    /// Upload `frame` into the staging texture, recreating it when the geometry
    /// changed.
    ///
    /// A no-op for bytes already staged: the sequence is the device's own
    /// publish counter, so equal sequences are the same pixels and a forced
    /// redraw re-presents rather than re-uploads.
    fn stage(
        &mut self,
        device: &Device,
        frame: &WindowCpuFrame<'_>,
    ) -> Result<(), MetalWindowDecline> {
        let fits = self
            .staged
            .as_ref()
            .is_some_and(|staged| staged.width == frame.width && staged.height == frame.height);
        if fits
            && self
                .staged
                .as_ref()
                .is_some_and(|staged| staged.seq == frame.seq)
        {
            return Ok(());
        }
        if !fits {
            let descriptor = TextureDescriptor::new();
            descriptor.set_texture_type(MTLTextureType::D2);
            descriptor.set_pixel_format(SURFACE_FORMAT);
            descriptor.set_width(u64::from(frame.width));
            descriptor.set_height(u64::from(frame.height));
            descriptor.set_mipmap_level_count(1);
            // Shared, so the upload below writes straight into the texture's
            // own storage: this host has one physical memory and a private
            // texture would cost a staging buffer and a blit to reach the same
            // bytes.
            descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
            descriptor.set_usage(MTLTextureUsage::ShaderRead);
            let texture = raw_metal::new_texture(device, &descriptor).ok_or(
                MetalWindowDecline::StagingTexture {
                    width: frame.width,
                    height: frame.height,
                },
            )?;
            self.staged = Some(Staged {
                texture,
                width: frame.width,
                height: frame.height,
                seq: frame.seq,
            });
        }
        let staged = self.staged.as_mut().expect("staged was just established");
        let row = frame.width as usize * SURFACE_BYTES_PER_PIXEL;
        staged.texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(frame.width), u64::from(frame.height)),
            0,
            frame.bgra.as_ptr().cast(),
            row as u64,
        );
        staged.seq = frame.seq;
        Ok(())
    }

    /// Encode and commit one present.
    ///
    /// Wrapped in an autorelease pool of its own: `nextDrawable` and the
    /// command buffer are autoreleased objects, and this runs once per frame
    /// for the life of the VM. AppKit drains its own pool around each event,
    /// so this is belt and braces rather than a leak that was measured — but
    /// a drawable held past its present is the one object on this path that
    /// stalls the next acquire, so the pool is explicit.
    fn present(
        &mut self,
        cpu: Option<WindowCpuFrame<'_>>,
    ) -> Result<WindowPresentOutcome, MetalWindowDecline> {
        objc::rc::autoreleasepool(|| self.present_pooled(cpu))
    }

    fn present_pooled(
        &mut self,
        cpu: Option<WindowCpuFrame<'_>>,
    ) -> Result<WindowPresentOutcome, MetalWindowDecline> {
        let device = system_device().ok_or(MetalWindowDecline::NoDevice)?;
        // An incomplete frame leaves the staging alone rather than clearing it.
        // A short buffer is a torn frame, not a new one, and the last good
        // picture is a better answer than black — the same rule the neutral
        // `WindowCpuFrame::complete` states for both rails.
        if let Some(frame) = cpu.filter(WindowCpuFrame::complete) {
            self.stage(device, &frame)?;
        }
        let Some(drawable) = self.layer.next_drawable() else {
            // Every drawable is still in flight. Nothing was consumed; the
            // window comes back within its redraw backstop.
            return Ok(WindowPresentOutcome::Busy);
        };

        let queue = thread_queue(device);
        let command_buffer = raw_metal::new_command_buffer(&queue)
            .ok_or(MetalWindowDecline::CommandBuffer)?
            .to_owned();
        let pass = RenderPassDescriptor::new();
        let color = pass
            .color_attachments()
            .object_at(0)
            .ok_or(MetalWindowDecline::RenderEncoder)?;
        color.set_texture(Some(drawable.texture()));
        color.set_load_action(MTLLoadAction::Clear);
        // The letterbox bars are this clear and nothing else — no second draw
        // fills them, so the colour here is the colour the user sees beside a
        // frame whose aspect does not match the window.
        color.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
        color.set_store_action(MTLStoreAction::Store);
        let encoder = raw_metal::new_render_command_encoder(&command_buffer, pass)
            .ok_or(MetalWindowDecline::RenderEncoder)?;
        if let Some(staged) = self.staged.as_ref() {
            encode_fitted_blit(
                encoder,
                &self.pipeline,
                &self.sampler,
                &staged.texture,
                (staged.width, staged.height),
                self.drawable,
            );
        }
        encoder.end_encoding();
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
        // Waited for, and this is the ordering that makes the staging texture
        // safe to overwrite: it is one texture, the next present rewrites it,
        // and the GPU is still sampling from it until this returns. The Metal
        // rail is synchronous everywhere else for the same reason — see
        // `super::mipmap` — and the wait costs the window thread one frame,
        // which `nextDrawable` above was already going to charge.
        command_buffer.wait_until_completed();
        if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
            return Err(MetalWindowDecline::Submit(
                raw_metal::command_buffer_error_description(&command_buffer),
            ));
        }
        Ok(WindowPresentOutcome::Presented {
            // No resident carries a Metal present; see this module's own doc.
            direct: false,
            width: self.drawable.0,
            height: self.drawable.1,
            buffers: self.layer.maximum_drawable_count() as usize,
            // A `CAMetalLayer` has no equivalent of a suboptimal swapchain: the
            // drawable size is this presenter's own to set, and it is set from
            // the size the window system actually granted.
            suboptimal: false,
        })
    }
}

impl Drop for Presenter {
    /// Take the layer back off the view.
    ///
    /// `addSublayer:` retains, so dropping the Rust handle alone would leave a
    /// live `CAMetalLayer` on a window that no longer has a presenter — the
    /// last frame frozen on screen, and a layer holding an `MTLDevice` for the
    /// rest of the process. The window detaches from `exiting`, while the view
    /// is still alive to be removed from.
    fn drop(&mut self) {
        // SAFETY: `removeFromSuperlayer` on a layer this presenter created and
        // has held since. Safe when there is no superlayer — it is a no-op —
        // which is what happens if the view was torn down first.
        unsafe {
            let _: () = msg_send![&*self.layer, removeFromSuperlayer];
        }
    }
}

/// Draw `source` into the encoder's colour attachment, aspect-fitted to a
/// `target`-sized destination.
///
/// The whole of what this rail does to a frame, in one place so the same encode
/// is what a test renders into an ordinary texture and what a present renders
/// into a drawable. Nothing outside the fitted rectangle is written, so the
/// bars are whatever the pass loaded — the render pass's clear colour, on the
/// one caller that presents.
fn encode_fitted_blit(
    encoder: &metal::RenderCommandEncoderRef,
    pipeline: &metal::RenderPipelineStateRef,
    sampler: &metal::SamplerStateRef,
    source: &metal::TextureRef,
    source_size: (u32, u32),
    target: (u32, u32),
) {
    let fit = aspect_fit(source_size, target);
    encoder.set_render_pipeline_state(pipeline);
    encoder.set_viewport(MTLViewport {
        originX: f64::from(fit.x),
        originY: f64::from(fit.y),
        width: f64::from(fit.width),
        height: f64::from(fit.height),
        znear: 0.0,
        zfar: 1.0,
    });
    encoder.set_fragment_texture(0, Some(source));
    encoder.set_fragment_sampler_state(0, Some(sampler));
    encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
}

fn build_blit_pipeline(device: &Device) -> Result<RenderPipelineState, MetalWindowDecline> {
    let library = raw_metal::new_library_with_source(device, BLIT_SHADER)
        .map_err(MetalWindowDecline::ShaderCompile)?;
    let vertex = raw_metal::new_function(&library, BLIT_VERTEX_FN)
        .ok_or(MetalWindowDecline::ShaderFunctionMissing(BLIT_VERTEX_FN))?;
    let fragment = raw_metal::new_function(&library, BLIT_FRAGMENT_FN)
        .ok_or(MetalWindowDecline::ShaderFunctionMissing(BLIT_FRAGMENT_FN))?;
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_vertex_function(Some(&vertex));
    descriptor.set_fragment_function(Some(&fragment));
    descriptor
        .color_attachments()
        .object_at(0)
        .ok_or(MetalWindowDecline::PipelineCreate(
            "no colour attachment 0".to_string(),
        ))?
        .set_pixel_format(SURFACE_FORMAT);
    raw_metal::new_render_pipeline_state(device, &descriptor)
        .map_err(MetalWindowDecline::PipelineCreate)
}

/// The backing scale factor of the screen a view is on, or 1.0 before it has a
/// window.
///
/// # Safety
///
/// `view` is a live `NSView`.
unsafe fn view_backing_scale(view: *mut Object) -> f64 {
    // SAFETY: the caller's contract. A view not yet in a window answers nil to
    // `window`, which is the pre-`resumed` case rather than an error.
    unsafe {
        let window: *mut Object = msg_send![view, window];
        if window.is_null() {
            return 1.0;
        }
        msg_send![window, backingScaleFactor]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every refusal this rail can hand the window names itself, and does it in
    /// a shape the always-on line can carry.
    ///
    /// A driver's error text is a sentence, and the log is whitespace-separated
    /// `k=v` — so an unfolded `detail=` swallows every field after it into the
    /// reader's idea of one value. Checked here rather than trusted, because
    /// the text comes from the OS and nothing in this file chooses it.
    #[test]
    fn every_metal_window_refusal_names_itself_log_safe() {
        let all = [
            MetalWindowDecline::NoDevice,
            MetalWindowDecline::NotAppKitWindow,
            MetalWindowDecline::ShaderCompile("two words here".to_string()),
            MetalWindowDecline::ShaderFunctionMissing(BLIT_VERTEX_FN),
            MetalWindowDecline::PipelineCreate("a b".to_string()),
            MetalWindowDecline::StagingTexture {
                width: 1440,
                height: 900,
            },
            MetalWindowDecline::CommandBuffer,
            MetalWindowDecline::RenderEncoder,
            MetalWindowDecline::Submit("Insufficient Memory".to_string()),
            MetalWindowDecline::NotAttached,
        ];
        let mut slugs = std::collections::BTreeSet::new();
        for decline in &all {
            let slug = decline.slug();
            assert!(
                slug.starts_with("metal_window_"),
                "{slug} does not say which rail refused"
            );
            assert!(slugs.insert(slug), "{slug} is claimed by two checks");
            let line = decline.to_string();
            assert!(line.starts_with(&format!("reason={slug}")));
            for field in line.split_whitespace().skip(1) {
                assert!(
                    field.contains('='),
                    "{line} carries a field the reader cannot split"
                );
            }
        }
        assert_eq!(slugs.len(), all.len());
    }

    /// Render `source` (BGRA8, `src` pixels) through the presenter's own blit
    /// into a `dst`-sized texture, and hand back the result as BGRA8 rows.
    ///
    /// The same encode a present runs, into an ordinary texture rather than a
    /// drawable — which is the only difference, and the one that makes the
    /// pixels readable. `None` when this host has no `MTLDevice`.
    fn render_through_the_blit(source: &[u8], src: (u32, u32), dst: (u32, u32)) -> Option<Vec<u8>> {
        let device = system_device()?;
        let pipeline = build_blit_pipeline(device).expect("the presenter's own blit must compile");
        let sampler = cached_default_sampler(device);

        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(SURFACE_FORMAT);
        descriptor.set_width(u64::from(src.0));
        descriptor.set_height(u64::from(src.1));
        descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::ShaderRead);
        let source_texture = raw_metal::new_texture(device, &descriptor)?;
        source_texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(src.0), u64::from(src.1)),
            0,
            source.as_ptr().cast(),
            u64::from(src.0) * SURFACE_BYTES_PER_PIXEL as u64,
        );

        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(SURFACE_FORMAT);
        descriptor.set_width(u64::from(dst.0));
        descriptor.set_height(u64::from(dst.1));
        descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let target = raw_metal::new_texture(device, &descriptor)?;

        let queue = thread_queue(device);
        let command_buffer = raw_metal::new_command_buffer(&queue)?.to_owned();
        let pass = RenderPassDescriptor::new();
        let color = pass.color_attachments().object_at(0)?;
        color.set_texture(Some(&target));
        color.set_load_action(MTLLoadAction::Clear);
        color.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
        color.set_store_action(MTLStoreAction::Store);
        let encoder = raw_metal::new_render_command_encoder(&command_buffer, pass)?;
        encode_fitted_blit(encoder, &pipeline, &sampler, &source_texture, src, dst);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_ne!(
            command_buffer.status(),
            metal::MTLCommandBufferStatus::Error,
            "{}",
            raw_metal::command_buffer_error_description(&command_buffer)
        );

        let row = dst.0 as usize * SURFACE_BYTES_PER_PIXEL;
        let mut out = vec![0u8; row * dst.1 as usize];
        target.get_bytes(
            out.as_mut_ptr().cast(),
            row as u64,
            MTLRegion::new_2d(0, 0, u64::from(dst.0), u64::from(dst.1)),
            0,
        );
        Some(out)
    }

    fn pixel(frame: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let offset = (y as usize * width as usize + x as usize) * SURFACE_BYTES_PER_PIXEL;
        frame[offset..offset + 4].try_into().expect("four bytes")
    }

    const BLUE: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const RED: [u8; 4] = [0, 0, 255, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];
    const BLACK: [u8; 4] = [0, 0, 0, 255];

    /// The frame reaches the screen the way up it was published.
    ///
    /// Row zero of a published frame is the top of the guest's desktop, `uv`
    /// `(0,0)` is the top left of the drawable, and the shader is what ties
    /// those together. Inverting either axis compiles, presents, and reports
    /// every counter healthy — it is only visible by looking at the window,
    /// which on a headless measurement host is not available. So the encode is
    /// run for real, into a texture that can be read back.
    ///
    /// A 2x2 into a 2x2 puts every destination pixel centre exactly on a source
    /// texel centre, so the linear sampler returns the texel unblended and the
    /// comparison is exact rather than a tolerance.
    #[test]
    fn the_blit_puts_each_source_texel_where_the_window_shows_it() {
        let mut source = Vec::new();
        for texel in [BLUE, GREEN, RED, WHITE] {
            source.extend_from_slice(&texel);
        }
        let Some(out) = render_through_the_blit(&source, (2, 2), (2, 2)) else {
            eprintln!("no MTLDevice on this host; skipping");
            return;
        };
        assert_eq!(pixel(&out, 2, 0, 0), BLUE, "top-left");
        assert_eq!(pixel(&out, 2, 1, 0), GREEN, "top-right");
        assert_eq!(pixel(&out, 2, 0, 1), RED, "bottom-left");
        assert_eq!(pixel(&out, 2, 1, 1), WHITE, "bottom-right");
    }

    /// A frame wider than the window is fitted, and the bars are the clear.
    ///
    /// Nothing draws the letterbox: the viewport confines the triangle and the
    /// render pass's clear colour is what the user sees beside the picture. A
    /// viewport that quietly covered the whole target instead would stretch the
    /// desktop, which is the defect this asserts is absent — a 2x1 source in a
    /// 2x2 target fits to the top row, and the bottom row must still be black.
    #[test]
    fn a_frame_that_does_not_fill_the_window_is_fitted_and_the_bars_are_clear() {
        let mut source = Vec::new();
        for texel in [BLUE, GREEN] {
            source.extend_from_slice(&texel);
        }
        let Some(out) = render_through_the_blit(&source, (2, 1), (2, 2)) else {
            eprintln!("no MTLDevice on this host; skipping");
            return;
        };
        assert_eq!(aspect_fit((2, 1), (2, 2)).height, 1, "the fit under test");
        assert_eq!(pixel(&out, 2, 0, 0), BLUE, "fitted row, left");
        assert_eq!(pixel(&out, 2, 1, 0), GREEN, "fitted row, right");
        assert_eq!(pixel(&out, 2, 0, 1), BLACK, "letterbox bar, left");
        assert_eq!(pixel(&out, 2, 1, 1), BLACK, "letterbox bar, right");
    }

    /// CoreAnimation's own values, pinned because nothing this build links
    /// declares them and a wrong bit is a layer that silently stops tracking
    /// its view — visible only as a picture that stays the old size after a
    /// resize, on the one host that can run this rail.
    #[test]
    fn the_layer_autoresizing_mask_is_width_and_height() {
        const K_CA_LAYER_WIDTH_SIZABLE: u32 = 2;
        const K_CA_LAYER_HEIGHT_SIZABLE: u32 = 16;
        assert_eq!(
            LAYER_SIZABLE,
            K_CA_LAYER_WIDTH_SIZABLE | K_CA_LAYER_HEIGHT_SIZABLE
        );
    }

    /// The blit's two entry points are the two the shader defines.
    ///
    /// A rename in the source with the constant left behind compiles, attaches,
    /// and then refuses at run time on the only host that can run it — which is
    /// the machine a boot is being measured on. Cheap to check here instead.
    #[test]
    fn the_blit_shader_defines_the_functions_the_pipeline_asks_for() {
        assert!(BLIT_SHADER.contains(&format!("vertex ReimsBlitVertex {BLIT_VERTEX_FN}(")));
        assert!(BLIT_SHADER.contains(&format!("fragment float4 {BLIT_FRAGMENT_FN}(")));
    }
}
