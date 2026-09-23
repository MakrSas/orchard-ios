//! L2–L7 immutable object caches (content/descriptor keyed, negative + hit/miss).

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering;

// ash Handle trait not required here.

use super::context::DeviceContext;
use super::counters::{CreateSite, EngineCounters};
use super::digest::Digest128;
use super::pools::{DeferredHandle, ResourcePools};
use super::types::{
    BlendKey, ColorWriteMask, DrawError, SamplerStateKey, VertexAttributeFormat, VertexStepFunction,
};
use super::vk_call::{VkCall, VkOp};

pub(crate) fn vk_sample_count(count: u32) -> vk::SampleCountFlags {
    match count {
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}

/// A device-specific widening of an optional three-component vertex format.
///
/// The draw remains executable, but the pipeline is not byte-for-byte what the
/// guest requested; keep the substitution and the affected attribute visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VertexFormatWidenDecline {
    from: vk::Format,
    to: vk::Format,
    location: u32,
    offset: u32,
    stride: u32,
}

impl crate::observe::Decline for VertexFormatWidenDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self { .. } => "vk_vertex_format_widened",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("from", format!("{:?}", self.from)),
            ("to", format!("{:?}", self.to)),
            ("location", self.location.to_string()),
            ("offset", self.offset.to_string()),
            ("stride", self.stride.to_string()),
        ]
    }
}
use crate::backend::vulkan::translate;
use crate::runtime::spirv_vertex_input::VertexInputWidths;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct AttrKey {
    pub location: u32,
    pub binding: u32,
    pub format: VertexAttributeFormat,
    pub offset: u32,
    pub stride: u32,
    pub step_function: VertexStepFunction,
    pub step_rate: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct BindingSig {
    pub binding: u32,
    pub ty: u32, // vk::DescriptorType as u32
    pub stages: u32,
    pub count: u32,
}

/// Content identity for a slice, issued in order and confirmed by comparison.
///
/// A digest picks the bucket and the *retained* slice decides the hit, so an id
/// is an identity and not a fingerprint — the shape [`LayoutTable`] uses for
/// binding signatures, without the Vulkan handles that make that one its own
/// type. [`super::digest`]'s header argues a digest should be read this way, and
/// both users here are the argument's own case: a cache key that owns a `Vec`
/// can only be probed with a `Vec`, which is a heap allocation on a path whose
/// whole job is a lookup that usually hits.
///
/// Unbounded, on the same argument [`ObjectCache`] is: the population is the
/// guest's distinct object set and plateaus. [`ObjectCaches::levels`] publishes
/// the count so that argument can be falsified.
struct SliceIntern<T> {
    /// Indexed by id. Never removed except by [`Self::clear`], so an id held in
    /// a cache key cannot name a hole.
    entries: Vec<Vec<T>>,
    buckets: HashMap<Digest128, Vec<u32>>,
    /// Last hit. Consecutive draws overwhelmingly repeat one set.
    front: Option<u32>,
}

impl<T: Clone + Eq + std::hash::Hash> SliceIntern<T> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            buckets: HashMap::new(),
            front: None,
        }
    }

    /// The id for `items`, issuing one if this is the first time it is seen.
    fn intern(&mut self, items: &[T]) -> u32 {
        if let Some(index) = self.front {
            if self.entries[index as usize] == items {
                return index;
            }
        }
        let digest = Digest128::of_items(&(), items);
        let bucket = self.buckets.entry(digest).or_default();
        if let Some(index) = bucket
            .iter()
            .copied()
            .find(|index| self.entries[*index as usize] == items)
        {
            self.front = Some(index);
            return index;
        }
        let index = self.entries.len() as u32;
        bucket.push(index);
        self.entries.push(items.to_vec());
        self.front = Some(index);
        index
    }

    fn get(&self, index: u32) -> &[T] {
        &self.entries[index as usize]
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.buckets.clear();
        self.front = None;
    }
}

/// Identity of one draw's vertex-attribute set.
///
/// Same reason as [`LayoutId`]: `PipelineKey` carries it, and a key owning a
/// `Vec<AttrKey>` is a heap allocation on every draw that has attributes —
/// which is every draw a guest actually sends.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct AttrsId(u32);

/// Identity of one descriptor-set + pipeline layout, for the life of this
/// device's caches.
///
/// A `Copy` handle and not the layout's bindings, because [`PipelineKey`] and
/// [`ComputePipelineKey`] carry the layout: a key holding an owned
/// `Vec<BindingSig>` is a heap allocation and a clone on **every draw**, on a
/// path whose whole purpose is a lookup that usually hits.
///
/// Deliberately not a digest. [`super::digest`]'s own header argues against a
/// digest standing alone as an identity, and nothing here needs one to:
/// [`LayoutTable`] issues these in order, retains each layout's bindings, and
/// compares them on every hit. The digest is the bucket; the bindings decide.
/// An id can therefore only name a layout that exists, and two distinct
/// binding sets can never share one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct LayoutId(u32);

/// What a resolved layout gives its caller.
///
/// `push_descriptors` travels with the handles rather than being asked again
/// from the bindings, because it is the flag the `VkDescriptorSetLayout` was
/// *created* with. Asking twice — once at create, once at bind — is two answers
/// to one question, and a host whose capability report moved between them would
/// write an allocated set into a push layout.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedLayout {
    pub id: LayoutId,
    pub dsl: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub push_descriptors: bool,
}

/// A layout create this device refused, kept so the next identical ask replays
/// the reason rather than paying the driver call again.
struct LayoutRefusal {
    bindings: Vec<BindingSig>,
    push_constant: Option<(u32, u32)>,
    error: DrawError,
}

struct LayoutEntry {
    /// Retained, and compared on every hit. This is what makes [`LayoutId`] an
    /// identity rather than a fingerprint.
    bindings: Vec<BindingSig>,
    /// Reflected compute push-constant byte range. Graphics layouts carry none.
    push_constant: Option<(u32, u32)>,
    dsl: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    push_descriptors: bool,
}

/// Whether a layout with these bindings is represented by command-buffer-local
/// push descriptors on this device.
///
/// Called once, when the layout is created; every later consumer reads
/// [`ResolvedLayout::push_descriptors`] instead.
fn layout_uses_push_descriptors(
    bindings: &[BindingSig],
    caps: crate::backend::vulkan::caps::PushDescriptorCaps,
) -> bool {
    !bindings.is_empty() && caps.supports_counts(bindings.iter().map(|b| b.count))
}

/// The device's descriptor/pipeline layouts, looked up by a *slice* of binding
/// signatures rather than by an owned key.
///
/// That is the whole reason it is not an [`ObjectCache`]. A `HashMap<LayoutKey,
/// _>` can only be probed with a `LayoutKey`, so every draw built one — a
/// `Vec<BindingSig>` — and then cloned it into its pipeline key, which was two
/// of the last four heap allocations on the steady-state draw path. Bucketing
/// by a digest of the slice and confirming against retained bindings answers
/// the same question from a borrowed slice.
struct LayoutTable {
    /// Indexed by [`LayoutId`]. Entries are never removed except by
    /// [`Self::clear`] and [`Self::take_all`], which empty the whole table, so
    /// an id outstanding in a `PipelineKey` cannot name a hole.
    entries: Vec<LayoutEntry>,
    /// Digest of `(push_constant, bindings)` to the ids that digest could name.
    /// A bucket with more than one entry is a digest collision between
    /// different layouts, which the comparison below resolves.
    buckets: HashMap<Digest128, Vec<u32>>,
    /// Last positive resolve, for the same reason [`ObjectCache::front`] exists:
    /// a render encoder repeats one layout for long runs, and the front hit
    /// costs one slice comparison instead of a hash.
    front: Option<u32>,
    /// Create failures worth replaying, as `(bindings, push range, error)`.
    /// A `Vec` and not a map because it is empty on a healthy boot and bounded
    /// by [`NEGATIVE_CAP`]; scanning it is cheaper than hashing to ask.
    negative: VecDeque<LayoutRefusal>,
}

impl LayoutTable {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            buckets: HashMap::new(),
            front: None,
            negative: VecDeque::new(),
        }
    }

    fn digest(bindings: &[BindingSig], push_constant: Option<(u32, u32)>) -> Digest128 {
        Digest128::of_items(&push_constant, bindings)
    }

    fn matches(entry: &LayoutEntry, bindings: &[BindingSig], pc: Option<(u32, u32)>) -> bool {
        entry.push_constant == pc && entry.bindings == bindings
    }

    fn resolved(&self, index: u32) -> ResolvedLayout {
        let entry = &self.entries[index as usize];
        ResolvedLayout {
            id: LayoutId(index),
            dsl: entry.dsl,
            pipeline_layout: entry.pipeline_layout,
            push_descriptors: entry.push_descriptors,
        }
    }

    fn get(&mut self, bindings: &[BindingSig], pc: Option<(u32, u32)>) -> Option<ResolvedLayout> {
        if let Some(index) = self.front {
            if Self::matches(&self.entries[index as usize], bindings, pc) {
                return Some(self.resolved(index));
            }
        }
        let candidates = self.buckets.get(&Self::digest(bindings, pc))?;
        let index = *candidates
            .iter()
            .find(|index| Self::matches(&self.entries[**index as usize], bindings, pc))?;
        self.front = Some(index);
        Some(self.resolved(index))
    }

    fn insert(
        &mut self,
        bindings: &[BindingSig],
        pc: Option<(u32, u32)>,
        dsl: vk::DescriptorSetLayout,
        pipeline_layout: vk::PipelineLayout,
        push_descriptors: bool,
    ) -> ResolvedLayout {
        self.negative
            .retain(|refusal| !(refusal.push_constant == pc && refusal.bindings == bindings));
        let index = self.entries.len() as u32;
        self.entries.push(LayoutEntry {
            bindings: bindings.to_vec(),
            push_constant: pc,
            dsl,
            pipeline_layout,
            push_descriptors,
        });
        self.buckets
            .entry(Self::digest(bindings, pc))
            .or_default()
            .push(index);
        self.front = Some(index);
        self.resolved(index)
    }

    fn get_negative(&self, bindings: &[BindingSig], pc: Option<(u32, u32)>) -> Option<DrawError> {
        self.negative
            .iter()
            .find(|refusal| refusal.push_constant == pc && refusal.bindings == bindings)
            .map(|refusal| refusal.error.clone())
    }

    /// Remember a create failure, under the rule [`ObjectCache::insert_negative`]
    /// states: a refusal about how full the device is right now is not a fact
    /// about the request and is not remembered.
    fn insert_negative(&mut self, bindings: &[BindingSig], pc: Option<(u32, u32)>, err: DrawError) {
        if err.out_of_memory() {
            return;
        }
        if self.get_negative(bindings, pc).is_some() {
            return;
        }
        if self.negative.len() >= NEGATIVE_CAP {
            self.negative.pop_front();
        }
        self.negative.push_back(LayoutRefusal {
            bindings: bindings.to_vec(),
            push_constant: pc,
            error: err,
        });
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.buckets.clear();
        self.front = None;
        self.negative.clear();
    }

    fn take_all(&mut self) -> Vec<(vk::DescriptorSetLayout, vk::PipelineLayout)> {
        self.buckets.clear();
        self.front = None;
        self.negative.clear();
        self.entries
            .drain(..)
            .map(|entry| (entry.dsl, entry.pipeline_layout))
            .collect()
    }
}

/// Sort by binding number, refuse two different signatures on one binding, and
/// drop exact duplicates — in place, on the caller's buffer.
///
/// In place because both callers hold reusable scratch: a draw's bindings live
/// in the command buffer's graphics scratch, and returning a fresh `Vec` would
/// put the allocation back that moving them there removed.
pub(crate) fn canonicalize_layout_bindings(
    bindings: &mut Vec<BindingSig>,
) -> Result<(), super::DrawError> {
    bindings.sort_by_key(|binding| binding.binding);
    for pair in bindings.windows(2) {
        if pair[0].binding == pair[1].binding && pair[0] != pair[1] {
            return Err(super::DrawError::Unsupported(
                super::reason::DrawReason::DescriptorBindingConflict {
                    binding: pair[0].binding,
                    first_type: pair[0].ty,
                    first_count: pair[0].count,
                    second_type: pair[1].ty,
                    second_count: pair[1].count,
                },
            ));
        }
    }
    bindings.dedup();
    Ok(())
}

/// Max secondary color attachments (MRT slot 1..): every colour slot Apple's
/// serialized render pass can carry, less the primary at slot 0.
///
/// The fourth spelling of one number, and the last one to be pinned. The wire
/// record's colour-slot array is the truth,
/// [`crate::runtime::render_pass::PASS_MAX_COLOR_ATTACHMENTS`] derives from
/// it, and `backend::metal::REIMS_VGPU_METAL_MAX_COLOR_RTS` is held equal to it
/// by an assertion beside itself. This one is that bound minus one, on the arm
/// the other assertion cannot reach — `REIMS_VGPU_METAL_MAX_COLOR_RTS` is behind
/// `feature = "backend-metal"`, so nothing in a Vulkan build compared the two.
///
/// A drift here is refused rather than lost: `execute_draw_inner` returns
/// [`super::reason::DrawReason::SecondaryAttachmentCap`] for a request past this
/// count, so a shortfall costs the whole draw and says so. That makes the
/// failure loud and still wrong — a guest sending the eighth colour slot the
/// wire format allows would have every MRT draw refused — which is what this
/// assertion is for.
pub(crate) const MAX_SECONDARY_ATTACH: usize = 7;
const _: () =
    assert!(1 + MAX_SECONDARY_ATTACH == crate::runtime::render_pass::PASS_MAX_COLOR_ATTACHMENTS);
const _: () = assert!(MAX_SECONDARY_ATTACH < u8::BITS as usize);

/// A secondary MRT attachment's contribution to the render-pass / pipeline key.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub(crate) struct SecondaryAttachKey {
    pub format: ash::vk::Format,
    /// true = LOAD existing content, false = CLEAR.
    pub load: bool,
}

/// A depth attachment's contribution to the render-pass key. `None` on `PassKey`
/// ⇒ no depth attachment (the 2D UI path). Depth-only uses D32_SFLOAT; when
/// `stencil` is set the attachment is the device-queried combined
/// depth-stencil format (`DeviceContext::depth_stencil_format`) with a live
/// STENCIL aspect (load/store), so it must partition the pass cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct DepthAttachKey {
    /// true = LOAD existing depth, false = CLEAR at pass start.
    pub load: bool,
    /// true = combined depth-stencil attachment (stencil test active).
    pub stencil: bool,
}

/// How slot 0 obtains its contents at the start of a pass.
///
/// # Why this is not a bool
///
/// The guest's load action has three values and this key used to carry one bit,
/// spelled `load_seed: bool` and resolved as "LOAD when true, CLEAR when
/// false". Two different requests collapse onto `false`: a pass that asked to
/// **clear** to the guest's own colour, and a pass that promised to **keep**
/// its prior contents and arrived with none of them. They are not the same
/// request and they do not have the same lawful answer.
///
/// [`reims_vgpu_protocol::pass_action::LoadAction::preserves_prior_contents`]
/// states the term: `MTLLoadActionDontCare` declares the prior contents
/// *undefined*, and undefined permits any contents — including the ones already
/// there. Clearing is the one reading that destroys them, and the colour it
/// destroys them with is not a colour the guest ever supplied: `target_clear`
/// is assigned only on the `Clear` arm, so an unseeded preserving pass cleared
/// to `[0.0; 4]`, a transparent black this device invented.
///
/// Measured cost of the collapse on rail macos-15, boot `s5`: 461 partial draws
/// and 2 107 399 texels overwritten with that invented colour in one boot, over
/// live guest content, on a rail where the guest declares DontCare and then
/// redraws only its damage rectangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum Color0Load {
    /// Prior contents are available and the pass begins with them.
    Preserve,
    /// The guest asked for a clear, to the value it supplied.
    Clear,
    /// The guest's action permits undefined contents and this device found no
    /// prior contents to offer. Writing none of the attachment is lawful and is
    /// strictly cheaper than the full-surface clear it replaces; inventing a
    /// colour is neither.
    Undefined,
}

impl Color0Load {
    /// The Vulkan load op and the layout the attachment must already be in.
    ///
    /// `Undefined` declares `UNDEFINED` as its initial layout, the same as a
    /// clear, and **must**: images are created with
    /// `initial_layout(vk::ImageLayout::UNDEFINED)`, and this arm is reachable
    /// on the *first* pass into a fresh attachment — that is exactly what a
    /// DontCare into a newly allocated plane is. Naming `color0_final` there
    /// would tell Vulkan the image is in a layout it has never been
    /// transitioned into. The `Preserve` arm may name it because prior contents
    /// are what it is preserving, so a previous pass has already left it there.
    ///
    /// What that costs: `UNDEFINED` licenses an implementation to discard the
    /// memory, so this arm removes the invented colour without *promising* the
    /// prior contents. That promise needs the resident to be loaded from —
    /// `Preserve` — and electing it when the resident is ready is the repair
    /// still outstanding; its witness is
    /// `a_preserving_gva_attachment_reaches_the_encoder_able_to_preserve`.
    /// Not writing the attachment is strictly better than clearing it either
    /// way, and it is the cheaper of the two.
    pub(crate) fn ops(
        self,
        color0_final: ash::vk::ImageLayout,
    ) -> (ash::vk::AttachmentLoadOp, ash::vk::ImageLayout) {
        match self {
            Self::Preserve => (ash::vk::AttachmentLoadOp::LOAD, color0_final),
            Self::Clear => (
                ash::vk::AttachmentLoadOp::CLEAR,
                ash::vk::ImageLayout::UNDEFINED,
            ),
            Self::Undefined => (
                ash::vk::AttachmentLoadOp::DONT_CARE,
                ash::vk::ImageLayout::UNDEFINED,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PassKey {
    /// How slot 0 obtains its contents. See [`Color0Load`].
    pub color0_load: Color0Load,
    /// Slot 0 aliases memory the host may modify between submissions.
    ///
    /// A linear image is host-accessible only in `PREINITIALIZED` or `GENERAL`.
    /// The former is an image's one-way birth layout, so a retained imported
    /// attachment must use `GENERAL` for both its pass layout and its exit
    /// layout. This bit partitions render passes and pipelines from ordinary
    /// device-local attachments, which retain their dedicated layouts.
    pub host_accessible_color0: bool,
    /// Slot-0 attachment format, as a format rather than a channel-order flag.
    ///
    /// This used to be `bgra: bool`, meaning `B8G8R8A8_UNORM` or
    /// `R8G8B8A8_UNORM` and nothing else, which made slot 0 the only attachment
    /// in this key that could not name a format — [`SecondaryAttachKey`] has
    /// carried a real [`ash::vk::Format`] since MRT landed. The asymmetry was
    /// not cosmetic: it is the reason a render target's resident is always
    /// eight bits per channel whatever the guest declared, because the *only*
    /// thing downstream could reconstruct from the flag was one of those two.
    ///
    /// It must stay part of the key. A render pass and a pipeline are both
    /// compiled against the attachment's format, so two draws differing only
    /// here need two of each; a key that omitted it would hand the second draw
    /// a pipeline built for the first one's format.
    pub color0_format: ash::vk::Format,
    /// Secondary color attachments (slot 1..). `secondary_count == 0` ⇒ the
    /// classic single-attachment pass, byte-identical to the pre-MRT engine.
    pub secondary: [SecondaryAttachKey; MAX_SECONDARY_ATTACH],
    pub secondary_count: u8,
    /// Depth attachment. `None` ⇒ no depth (byte-identical to the pre-depth
    /// pass); the depth attachment is always appended AFTER color + secondaries
    /// so slot 0 stays the primary color (the zero-copy readback assumes this).
    pub depth: Option<DepthAttachKey>,
    /// Attachment 0 is ALSO referenced as a subpass input (framebuffer fetch).
    /// Both references use GENERAL layout and the subpass carries a BY_REGION
    /// self-dependency — the Vulkan feedback-loop form MoltenVK lowers to Metal
    /// programmable blending. `false` keeps the pass byte-identical.
    pub color_input: bool,
    /// Bit N says colour attachment N is also sampled through
    /// `VK_EXT_attachment_feedback_loop_layout`. The decoded render pass has at
    /// most eight colour attachments, so the wire-derived attachment table is
    /// the bound and one byte carries the whole set.
    pub feedback_colors: u8,
    /// Sample count of the colour attachment pipelines rasterize into.
    pub sample_count: u32,
    /// Attachment zero resolves into a single-sample attachment appended
    /// immediately after it.
    pub multisample_resolve: bool,
}

impl PassKey {
    /// Single-color-attachment pass (the pre-MRT constructor).
    pub(crate) fn single(color0_load: Color0Load, color0_format: ash::vk::Format) -> Self {
        Self {
            color0_load,
            host_accessible_color0: false,
            color0_format,
            secondary: [SecondaryAttachKey::default(); MAX_SECONDARY_ATTACH],
            secondary_count: 0,
            depth: None,
            color_input: false,
            feedback_colors: 0,
            sample_count: 1,
            multisample_resolve: false,
        }
    }

    pub(crate) fn color_feedback(self, index: usize) -> bool {
        index < u8::BITS as usize && self.feedback_colors & (1u8 << index) != 0
    }

    /// The part of a render pass that Vulkan requires to agree for pipeline,
    /// framebuffer, and in-instance compatibility.
    ///
    /// Load actions describe how a newly begun pass obtains attachment
    /// contents; they are deliberately excluded. A serialized render encoder
    /// rewrites a continuation segment to LOAD, but an uninterrupted segment
    /// remains inside the pass begun with the encoder's original action. Store
    /// actions and initial/final layouts are functions of these same fields in
    /// this backend, so there is no second compatibility spelling to normalize.
    ///
    /// `feedback_colors` is erased **exactly when it changes nothing about the
    /// pass** — that is, when [`color_feedback_layout`] and
    /// [`color0_pass_exit_layout`] are the same layout. The self-dependency is
    /// declared on every pass, so once the layouts also coincide a feedback draw
    /// and an ordinary one want a byte-identical `VkRenderPass` and there is
    /// nothing left to keep them apart. Which is the point: whether a draw samples
    /// the target it is writing is a property of the *draw*, exactly as it is in
    /// Metal, and it stops closing the render pass.
    ///
    /// The condition is not decoration. Under [`crate::config::COLOR_GENERAL`]`=off`
    /// the resting layout admits no feedback loop, the feedback slots really are
    /// in a different layout, and erasing the field there would merge two draws
    /// whose attachment is in two different layouts — a pass naming a layout its
    /// image is not in.
    pub(crate) fn compatibility(self) -> PassCompatibilityKey {
        let mut key = self;
        // Erased to one arbitrary-but-fixed value, exactly as the old bool was
        // erased to `false`: which load op a pass begins with is not part of
        // Vulkan's compatibility, so all three must collapse here or two
        // compatible passes would read as incompatible. This is the one place
        // the collapse is correct.
        key.color0_load = Color0Load::Clear;
        for secondary in &mut key.secondary {
            secondary.load = false;
        }
        if let Some(depth) = &mut key.depth {
            depth.load = false;
        }
        if color_feedback_layout() == color0_pass_exit_layout() {
            key.feedback_colors = 0;
        }
        PassCompatibilityKey(key)
    }

    /// The render-pass state Vulkan uses to decide framebuffer compatibility.
    ///
    /// Attachment load actions, attachment-reference layouts, and subpass
    /// dependencies do not participate. Host accessibility changes only the
    /// primary attachment's layouts and external dependency; feedback changes
    /// layouts and a dependency. Neither requires a framebuffer to be rebuilt.
    /// Attachment formats and the subpass attachment-reference shape remain in
    /// the key.
    pub(crate) fn framebuffer_compatibility(self) -> FramebufferCompatibilityKey {
        let mut key = self.compatibility().0;
        key.host_accessible_color0 = false;
        key.feedback_colors = 0;
        FramebufferCompatibilityKey(key)
    }

    pub(crate) fn color_layout(self, index: usize) -> vk::ImageLayout {
        if self.color_feedback(index) {
            color_feedback_layout()
        } else if index == 0 && (self.color_input || self.host_accessible_color0) {
            vk::ImageLayout::GENERAL
        } else {
            // The same layout the pass exits at, so an ordinary pass performs no
            // transition of its own at either end. Written as the exit rather
            // than as `COLOR_ATTACHMENT_OPTIMAL` for the reason
            // `color0_pass_exit_layout` gives: there is one spelling.
            color0_pass_exit_layout()
        }
    }

    pub(crate) fn color_final_layout(self, index: usize) -> vk::ImageLayout {
        if index == 0 && self.host_accessible_color0 {
            vk::ImageLayout::GENERAL
        } else if self.color_feedback(index) {
            color_feedback_layout()
        } else {
            color0_pass_exit_layout()
        }
    }
}

/// The layout a colour attachment a draw also samples is placed in.
///
/// The resting layout itself whenever that admits a feedback loop, which is the
/// shipping arm — so a slot the guest samples and a slot it does not are in the
/// **same** layout, the pass declares no transition for either, and there is no
/// second layout for a colour target anywhere in this device. Only the ablation
/// arm, where the resting layout is `COLOR_ATTACHMENT_OPTIMAL` and admits
/// nothing, reaches the extension's dedicated layout.
///
/// One function because four places name this: the subpass attachment reference,
/// the `finalLayout`, the sampled descriptor
/// ([`super::exec::PreparedSampled::descriptor_layout`]) and the registry record
/// the pass leaves behind. A descriptor naming a layout the attachment reference
/// does not is undefined behaviour, and it is not an error anywhere.
pub(crate) fn color_feedback_layout() -> vk::ImageLayout {
    let resting = color0_pass_exit_layout();
    if layout_admits_color_feedback(resting) {
        resting
    } else {
        vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
    }
}

/// Whether a colour attachment resting in `layout` may also be sampled by a draw
/// inside the same render pass instance — a Vulkan *feedback loop*.
///
/// Two layouts admit one. `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL` is the dedicated
/// spelling `VK_EXT_attachment_feedback_loop_layout` adds, and `GENERAL` is the
/// core one: a sampled-image descriptor may name `GENERAL`, and the core rule for
/// an attachment written earlier in the subpass permits it to be accessed "as an
/// attachment, storage image, or sampled image" by a later command. So the
/// extension layout is an *optimisation over* `GENERAL`, never a requirement.
///
/// This is the whole reason feedback is not a second layout. While
/// [`color0_pass_exit_layout`] answers `GENERAL`, a slot the guest samples and a
/// slot it does not are in the same layout, so the render pass declares no
/// transition for either and the registry's record is true for both.
///
/// It matters that this is a question about the layout and not a switch read.
/// Under [`crate::config::COLOR_GENERAL`]`=off` the resting layout is
/// `COLOR_ATTACHMENT_OPTIMAL`, which admits no feedback loop at all, and the
/// extension layout has to come back — naming the resting layout there would be a
/// sampled read of an attachment in a layout that forbids it, which is undefined
/// behaviour rather than an error. The ablation therefore restores two layouts
/// because the contract says it must, not because a flag says so.
const fn layout_admits_color_feedback(layout: vk::ImageLayout) -> bool {
    matches!(
        layout,
        vk::ImageLayout::GENERAL | vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
    )
}

/// A normalized [`PassKey`] containing exactly Vulkan render-pass
/// compatibility state. Construction is private to [`PassKey::compatibility`]
/// so a load action cannot accidentally enter a pipeline or framebuffer key.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PassCompatibilityKey(PassKey);

/// The subset of [`PassKey`] that Vulkan requires to agree when a framebuffer
/// created against one render pass is used with another.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct FramebufferCompatibilityKey(PassKey);

impl PassCompatibilityKey {
    pub(crate) fn secondary_count(self) -> usize {
        self.0.secondary_count as usize
    }

    pub(crate) fn has_depth(self) -> bool {
        self.0.depth.is_some()
    }

    /// Which field makes two compatibility keys disagree, or `None` when they
    /// are equal.
    ///
    /// A `passdiff_compat` firing says a draw could not continue its
    /// predecessor's render pass because Vulkan would not call the two passes
    /// compatible, and on a driven Maps leg that is the dominant merge blocker
    /// once the framebuffer identity one is fixed. On its own it names no
    /// repair: this key carries nine independent things and a change in any of
    /// them lands in the same bucket. A colour format change is the guest
    /// drawing into a different target and is not repairable at all; a
    /// `sample_count` or `feedback_colors` change might be this device's own
    /// bookkeeping.
    ///
    /// The order is arbitrary — unlike [`super::pools::PassEchoField`]'s, where
    /// an earlier field makes a later one unreachable — so an answer here is
    /// *a* difference and not the only one. That is what the census needs: the
    /// question is which field to look at first, and any field that ever
    /// differs is worth a reading.
    ///
    /// The destructure is exhaustive on purpose. A tenth field added to
    /// [`PassKey`] fails this function to compile rather than joining a bucket
    /// that silently stops being a partition.
    pub(crate) fn first_difference(self, other: Self) -> Option<PassCompatField> {
        let PassKey {
            // Load actions are erased by `PassKey::compatibility`, so they are
            // equal here by construction and cannot be a difference.
            color0_load: _,
            host_accessible_color0,
            color0_format,
            secondary,
            secondary_count,
            depth,
            color_input,
            feedback_colors,
            sample_count,
            multisample_resolve,
        } = self.0;
        let them = other.0;
        if color0_format != them.color0_format {
            return Some(PassCompatField::Color0Format);
        }
        if secondary_count != them.secondary_count {
            return Some(PassCompatField::SecondaryCount);
        }
        if secondary != them.secondary {
            return Some(PassCompatField::SecondaryFormat);
        }
        if depth != them.depth {
            return Some(PassCompatField::Depth);
        }
        if host_accessible_color0 != them.host_accessible_color0 {
            return Some(PassCompatField::HostAccessibleColor0);
        }
        if color_input != them.color_input {
            return Some(PassCompatField::ColorInput);
        }
        if feedback_colors != them.feedback_colors {
            return Some(PassCompatField::FeedbackColors);
        }
        if sample_count != them.sample_count {
            return Some(PassCompatField::SampleCount);
        }
        if multisample_resolve != them.multisample_resolve {
            return Some(PassCompatField::MultisampleResolve);
        }
        None
    }
}

/// Which field of a [`PassCompatibilityKey`] two draws disagreed about.
///
/// See [`PassCompatibilityKey::first_difference`] for why the split exists and
/// why the order carries no meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassCompatField {
    Color0Format,
    SecondaryCount,
    SecondaryFormat,
    Depth,
    HostAccessibleColor0,
    ColorInput,
    FeedbackColors,
    SampleCount,
    MultisampleResolve,
}

impl PassCompatField {
    pub(crate) fn route(self) -> &'static str {
        match self {
            Self::Color0Format => "passcompat_color0_format",
            Self::SecondaryCount => "passcompat_secondary_count",
            Self::SecondaryFormat => "passcompat_secondary_format",
            Self::Depth => "passcompat_depth",
            Self::HostAccessibleColor0 => "passcompat_host_accessible",
            Self::ColorInput => "passcompat_color_input",
            Self::FeedbackColors => "passcompat_feedback",
            Self::SampleCount => "passcompat_sample_count",
            Self::MultisampleResolve => "passcompat_ms_resolve",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PipelineKey {
    pub vert: Digest128,
    pub frag: Digest128,
    pub attrs: AttrsId,
    /// What this pipeline is identified by as far as the primitive type is
    /// concerned — the guest's exact type on a host that bakes it, its
    /// topology class where `vkCmdSetPrimitiveTopology` may move within a
    /// class, and one key for everything where the device also reports
    /// `dynamicPrimitiveTopologyUnrestricted`.
    ///
    /// Not the guest's type, for the reason [`Self::raster`] is not the
    /// guest's ordinals: on a dynamic host a triangle list and a triangle
    /// strip are one pipeline, and a key holding the type could not say so.
    /// `reims_vgpu_vulkan::topology::key` is the only place that decides which
    /// of the three rungs this device is on, and
    /// `TopologyKey::input_assembly` derives the declared topology back out of
    /// the key — never out of the guest's type — so two draws sharing a key
    /// cannot describe two different pipelines.
    pub topology: reims_vgpu_vulkan::topology::TopologyKey,
    pub blend: Option<BlendKey>,
    /// Per-slot blend for secondary colour attachments, parallel to
    /// `pass.secondary[..pass.secondary_count]`. Entries past the count are
    /// `None` and inert.
    ///
    /// This is part of the key, not just the builder input: two draws sharing
    /// shaders and pass shape but blending different secondary slots need
    /// different pipelines, and before this they would have aliased onto
    /// whichever was created first.
    pub secondary_blend: [Option<BlendKey>; MAX_SECONDARY_ATTACH],
    /// Per-slot `MTLColorWriteMask`, index 0 the primary attachment and index
    /// `n` the secondary parallel to `pass.secondary[n - 1]`.
    ///
    /// In the key, not just the builder input: two draws sharing shaders, pass
    /// shape and blend but masking different channels need different
    /// pipelines. Vulkan's write mask is pipeline state with no dynamic
    /// spelling below `VK_EXT_extended_dynamic_state3`.
    pub color_write_mask: [ColorWriteMask; 1 + MAX_SECONDARY_ATTACH],
    pub pass: PassCompatibilityKey,
    /// Which colour attachments this draw samples while writing them.
    ///
    /// Pipeline state, not pass state. `VK_PIPELINE_CREATE_COLOR_ATTACHMENT_-
    /// FEEDBACK_LOOP_BIT_EXT` is what "feedback loop is enabled" means for the
    /// draw-time rules, and it is fixed at pipeline creation — so a feedback draw
    /// and an ordinary one need two pipelines even when they share a render pass.
    ///
    /// It lives here rather than being read back out of [`Self::pass`] because
    /// [`PassKey::compatibility`] erases it precisely when the render pass stops
    /// depending on it, which is the shipping arm. Reading it from there would
    /// silently drop the create flag off every feedback pipeline, and the result
    /// is a draw sampling an attachment it is writing with no feedback loop
    /// enabled — undefined behaviour, reported nowhere.
    pub feedback_colors: u8,
    /// The rasterization state this pipeline is built with — already parsed,
    /// already normalized against this host.
    ///
    /// The one member of this key that is **not** the guest's raw ordinals,
    /// and deliberately: on a host that supplies the cull mode, the winding,
    /// the fill mode or the depth-clip mode per draw, those members carry
    /// their baked default here and the guest's values ride to the encoder
    /// instead. Two draws differing only in a dynamic member are then the same
    /// key and share one pipeline, which is the whole payoff — a key holding
    /// the ordinals could not express that, because it could not tell which of
    /// them this device still bakes.
    ///
    /// `reims_vgpu_vulkan::raster::plan` is still the one layer that decides
    /// what an ordinal means; it is called at the draw seam that builds this
    /// key rather than here, because the same call produces the encoder half.
    /// So this field cannot hold an ordinal nobody parsed.
    pub raster: reims_vgpu_vulkan::raster::RasterizationState,
    /// The depth-stencil state this pipeline is built with — already
    /// translated, already placed against this host and this pass.
    ///
    /// Four key terms before this: a test flag, a write flag, a compare
    /// function and an optional stencil-op pair. They are one value now for the
    /// reason [`Self::raster`] is: on a host with
    /// `VK_EXT_extended_dynamic_state` the guest's whole
    /// `MTLDepthStencilState` rides to the encoder and this carries a fixed
    /// placeholder, so every depth-stencil state a guest can bind is one
    /// pipeline. Four separate terms could not express that, because none of
    /// them knows whether this device still bakes it.
    ///
    /// Meaningful only when `pass.has_depth()`; Vulkan attaches no
    /// depth-stencil state to a pipeline whose subpass has no depth
    /// attachment, and `reims_vgpu_vulkan::depth_stencil::plan` is told so and
    /// makes nothing dynamic there. The reference value is excluded on both
    /// rungs — it is a separate Metal encoder command — so distinct references
    /// have always reused one pipeline.
    pub depth_stencil: reims_vgpu_vulkan::depth_stencil::DepthStencilPlan,
    /// How many viewport/scissor slots the pipeline declares.
    ///
    /// In the key because `VkPipelineViewportStateCreateInfo::viewportCount` is
    /// **not** dynamic below `VK_EXT_extended_dynamic_state`
    /// (`vkCmdSetViewportWithCount`), which is core in 1.3 and this device's
    /// floor is 1.2. So the count is baked, `vkCmdSetViewport` must bind exactly
    /// that many, and two draws sharing shaders and pass shape but rasterizing
    /// into different numbers of viewports need different pipelines. It is one
    /// number for both counts because Vulkan requires `scissorCount` to equal
    /// `viewportCount`; [`super::viewport_slot_count`] is the only place that
    /// decides it.
    pub viewport_slots: u32,
    pub layout: LayoutId,
}

/// Lc: compute pipeline cache key — SPIR-V content digest + entry name + layout
/// + the workgroup size the module is specialized to. Never funcId / pipeline ref.
///
/// The local size belongs in the key because an exact-thread module leaves it
/// specializable: one translated kernel yields up to eight pipelines whose only
/// difference is the boundary workgroup size, and they share every other term.
/// `None` is a module that baked its local size as a constant.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct ComputePipelineKey {
    pub spirv: Digest128,
    pub entry: String,
    pub layout: LayoutId,
    pub local_size: Option<[u32; 3]>,
}

/// A shader module and the words the driver compiles from it.
///
/// They travel together because a pipeline create needs the handle and the
/// crash breadcrumb needs the source, and two parameters is how a caller ends
/// up passing one shader's handle beside another's words.
#[derive(Clone, Copy)]
pub(crate) struct ShaderModuleSource<'a> {
    pub module: vk::ShaderModule,
    pub spirv: &'a [u32],
}

/// How many distinct never-creatable keys a cache remembers the refusal for.
///
/// This bounds the **negative** map only, and it is the one bound in this file
/// that is not a fidelity question: an evicted negative entry costs a re-attempt
/// of a create that has already been measured to fail, never a dropped guest
/// object. The positive maps are deliberately unbounded — see [`ObjectCache`].
const NEGATIVE_CAP: usize = 1024;

/// A content-keyed cache of immutable Vulkan objects, plus the typed refusal for
/// keys whose create failed.
///
/// **The positive map is unbounded, and that is the contract.** Every key here
/// is a content digest or a full descriptor of guest-decoded state — a shader's
/// SPIR-V digest, a pipeline's complete key, a sampler's state. So the live
/// entry count is the number of *distinct* objects the guest has asked for,
/// which is a property of its own program and state set rather than of how long
/// the device has run.
///
/// It used to hold 1024 (64 for render passes) and evict in **insertion** order.
/// Insertion order is the worst possible choice here for the same reason it was
/// in `runtime::m2v_cache`: the first pipeline a boot creates is the
/// compositor's, and it is bound on every frame until the guest shuts down, so
/// the first thing a cap crossing discards is the entry that is still hot. The
/// re-create is `vkCreateGraphicsPipelines` — a driver-side shader compile, not
/// a lookup — so a thrashing cache pays one compile per frame per evicted
/// pipeline, forever.
///
/// The bound also never engaged on this arm. A driven x86 boot, window-drag
/// probe against Safari, settles at `pipelines=92 shaders=75 layouts=33
/// passes=4 samplers=14 compute_pipelines=16` — read directly off
/// [`ObjectCaches::levels`], which is what the `object_cache_levels` census
/// publishes. Every level is flat from roughly 38 s in through the end of the
/// run, including across the drag probe's compositing, so the caps only stood
/// ready to evict the hot set on a heavier guest.
///
/// Two of those numbers matter beyond this arm. `passes=4` against the 64 this
/// cache carried is the widest margin here; and `pipelines=92` is *above* the
/// 64-slot render-pipeline table the Metal arm carried, which is how that arm's
/// cap was shown to be binding — see [`crate::model::content_cache`].
///
/// Unbounded is also the faithful failure mode. When a guest really does ask for
/// more distinct pipelines than the host can hold, the create itself returns
/// `VK_ERROR_OUT_OF_DEVICE_MEMORY` and that is reported as a typed [`DrawError`].
/// That is a GPU refusing because its memory is full — the behavior we are
/// emulating — rather than a device that silently forgets an object the guest
/// still has bound. It is deliberately *not* remembered; see
/// [`ObjectCache::insert_negative`] for why a refusal about this instant must not
/// outlive the instant.
struct ObjectCache<K, V> {
    map: HashMap<K, V>,
    /// Last positive lookup, retained as the exact key and value. Render
    /// encoders commonly repeat one pipeline for long runs; equality against
    /// that key avoids hashing the same composite state on every draw.
    front: Option<(K, V)>,
    negative: HashMap<K, DrawError>,
    /// FIFO order for `negative`, bounded by [`NEGATIVE_CAP`]. Negative entries
    /// are only added on create failures that a second identical attempt would
    /// meet again — a Vulkan create call refusing for a reason inherent to the
    /// request (a typed [`VkCall`]) or a device-capability refusal
    /// (`DrawError::Unsupported`, e.g. an unsupported vertex divisor) — empty on
    /// a healthy boot, but a guest that keeps submitting distinct
    /// never-creatable objects would grow `negative` without limit if it were
    /// unbounded. The value is the exact typed [`DrawError`] the create refused
    /// with, so the cheap re-attempt replays that reason — slug and all — rather
    /// than a re-formatted `Vulkan(String)` that dropped it.
    negative_order: VecDeque<K>,
    negative_cap: usize,
}

impl<K: Clone + Eq + std::hash::Hash, V> ObjectCache<K, V> {
    fn new() -> Self {
        Self::with_negative_cap(NEGATIVE_CAP)
    }

    fn with_negative_cap(negative_cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            front: None,
            negative: HashMap::new(),
            negative_order: VecDeque::new(),
            negative_cap,
        }
    }

    fn get(&mut self, k: &K) -> Option<V>
    where
        V: Copy,
    {
        self.get_routed(k).map(|(value, _)| value)
    }

    /// Positive lookup and whether the one-entry front index answered it.
    fn get_routed(&mut self, k: &K) -> Option<(V, bool)>
    where
        V: Copy,
    {
        if let Some((front_key, value)) = &self.front {
            if front_key == k {
                return Some((*value, true));
            }
        }
        let value = *self.map.get(k)?;
        self.front = Some((k.clone(), value));
        Some((value, false))
    }

    fn get_negative(&self, k: &K) -> Option<DrawError> {
        // The healthy hot path has never cached a refusal. Avoid hashing the
        // full object key merely to ask an empty table; render pipeline keys in
        // particular carry attribute, attachment and descriptor arrays, and a
        // positive hit immediately hashes the same key again below.
        if self.negative.is_empty() {
            return None;
        }
        self.negative.get(k).cloned()
    }

    /// Insert. Returns the value a *replace* displaced, so the caller can
    /// destroy the Vulkan object it owned; a fresh key returns `None`. Nothing
    /// is ever displaced for capacity.
    fn insert(&mut self, k: K, v: V) -> Option<V>
    where
        V: Copy,
    {
        self.negative.remove(&k);
        let old = self.map.insert(k.clone(), v);
        self.front = Some((k, v));
        old
    }

    /// Remember a create failure so the next identical ask replays it without
    /// paying the driver call again.
    ///
    /// **A refusal about this instant is not remembered at all.** Out of memory
    /// describes how much the device is holding right now, not anything about
    /// the request — the guest can free a texture atlas and ask for the very
    /// same pipeline a frame later, and by then the create succeeds. Memoizing
    /// one turns a GPU that refuses while full into a GPU that refuses forever,
    /// which is the failure a real one does not have: nothing here can clear a
    /// negative entry short of device teardown, because the lookup consults
    /// `negative` before the create and so the create that would displace it
    /// never runs.
    ///
    /// The predicate is [`DrawError::out_of_memory`], the crate's single
    /// statement of which refusals a second attempt could answer differently;
    /// the resident image and command-buffer allocators already reclaim and
    /// retry on it. Deciding it here rather than at the call sites is
    /// deliberate — thirteen of them insert negatives, and a rule spread over
    /// thirteen sites is a rule that will be half-applied.
    ///
    /// Declining to memoize costs a repeated failing create while the device
    /// stays full. That is the same bargain the resident allocators take, and
    /// it is bounded by the guest's own retry rate rather than by anything
    /// here.
    fn insert_negative(&mut self, k: K, err: DrawError) {
        if err.out_of_memory() {
            return;
        }
        if self.negative.insert(k.clone(), err).is_some() {
            // Already tracked (error refreshed); order stays as-is.
            return;
        }
        self.negative_order.push_back(k);
        // Bound the negative map, oldest-first. Pops skip stale order entries
        // (keys since promoted into the positive map by `insert`).
        while self.negative.len() > self.negative_cap {
            match self.negative_order.pop_front() {
                Some(old) => {
                    self.negative.remove(&old);
                }
                None => break,
            }
        }
        // Compact the order deque if promotions left many stale entries, so it
        // can never itself grow unbounded (rare; error path only).
        if self.negative_order.len() > self.negative_cap.saturating_mul(2) {
            self.negative_order
                .retain(|key| self.negative.contains_key(key));
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn clear(&mut self) {
        self.front = None;
        self.map.clear();
        self.negative.clear();
        self.negative_order.clear();
    }

    fn take_all(&mut self) -> Vec<V> {
        self.front = None;
        self.negative.clear();
        self.negative_order.clear();
        self.map.drain().map(|(_, v)| v).collect()
    }
}

/// Entries the front index holds before it starts over.
///
/// Each entry is an address, a `Digest128` and an `Arc` clone — tens of bytes,
/// plus the words the `Arc` keeps alive, which the runtime owns for the
/// shader's lifetime anyway. A driven macos-13 boot binds a few hundred
/// distinct modules, so this is a ceiling with an order of magnitude of
/// headroom and `shader_digest_reset` firing is the boot saying the guest's
/// module set is not what that assumed.
const SHADER_DIGEST_ENTRIES: usize = 4096;

/// `Arc<Vec<u32>>` allocation address → the digest that module finally hashes
/// to, so a repeat bind can skip three whole-module walks.
///
/// # Why an address is a sound key
///
/// Only because the entry holds the `Arc`. While it does, the allocation cannot
/// be freed, so nothing else can be given that address and the key cannot come
/// to mean a different module. Drop the `Arc` from the entry and this becomes a
/// use-after-free dressed as a cache hit.
///
/// `usize` rather than `*const Vec<u32>` because a raw pointer is not `Send` and
/// `Caches` is held behind the engine lock and moved between threads. The
/// address is never dereferenced — it is compared, and the `Arc` beside it is
/// what keeps it meaningful.
///
/// # What it skips, and why that is safe
///
/// [`ObjectCaches::get_or_create_shader`] walks the module three times before it
/// can look anything up: `required_image_capabilities`, the digest, and (on the
/// patch path) a rebuild. All three are pure functions of the words, and the
/// words behind an `Arc<Vec<u32>>` cannot change. So the digest recorded here is
/// the *final* one — after any capability patch — and a hit is the same answer
/// those three walks would have produced.
///
/// A hit still consults [`ObjectCaches::shaders`], positive and negative. That
/// keeps this index from depending on `ObjectCache` never evicting, which is a
/// property it happens to have and does not promise: a miss there simply falls
/// through to the full path, which recomputes and re-inserts.
#[derive(Default)]
struct ShaderDigestIndex {
    map: std::collections::HashMap<usize, (std::sync::Arc<Vec<u32>>, Digest128)>,
}

impl ShaderDigestIndex {
    /// The digest this allocation's module hashes to, if it has been walked
    /// before.
    fn get(&self, words: &std::sync::Arc<Vec<u32>>) -> Option<Digest128> {
        self.map
            .get(&(std::sync::Arc::as_ptr(words) as usize))
            .map(|(_, digest)| *digest)
    }

    /// Record what a full walk of this allocation produced.
    ///
    /// The bound is enforced here because this is the only way in: past
    /// [`SHADER_DIGEST_ENTRIES`] the whole index is dropped rather than evicting
    /// one entry, because there is no recency to evict *by* — every entry is
    /// equally cheap to rebuild, and a boot that reaches the bound is reporting
    /// something rather than asking for a policy.
    fn insert(&mut self, words: &std::sync::Arc<Vec<u32>>, digest: Digest128) {
        if self.map.len() >= SHADER_DIGEST_ENTRIES {
            crate::observe::off(format!(
                "shader_digest_reset entries={} words={}",
                self.map.len(),
                words.len()
            ));
            self.map.clear();
        }
        self.map.insert(
            std::sync::Arc::as_ptr(words) as usize,
            (std::sync::Arc::clone(words), digest),
        );
    }

    fn clear(&mut self) {
        self.map.clear();
    }
}

pub(crate) struct ObjectCaches {
    shaders: ObjectCache<Digest128, vk::ShaderModule>,
    layouts: LayoutTable,
    attr_sets: SliceIntern<AttrKey>,
    passes: ObjectCache<PassKey, vk::RenderPass>,
    pipelines: ObjectCache<PipelineKey, vk::Pipeline>,
    /// Exact last Vulkan variant for each retained guest pipeline object.
    /// Values are weakly tied to the runtime object's lifetime; the content
    /// cache remains the authority and owns every Vulkan handle.
    pipeline_objects: ObjectVariantIndex<PipelineKey, vk::Pipeline>,
    samplers: ObjectCache<SamplerStateKey, vk::Sampler>,
    /// Lc: compute pipelines (content digest + entry + layout).
    compute_pipelines: ObjectCache<ComputePipelineKey, vk::Pipeline>,
    /// The allocation a shader's words live in → the digest its module hashes
    /// to, so a repeat bind of the same module does not walk it three times to
    /// find that out.
    shader_digests: ShaderDigestIndex,
}

struct ObjectVariantIndex<K, V> {
    map: HashMap<std::num::NonZeroU64, (std::sync::Weak<super::types::PipelineObjectLife>, K, V)>,
}

impl<K, V> Default for ObjectVariantIndex<K, V> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}

impl<K: Clone + Eq, V: Copy> ObjectVariantIndex<K, V> {
    /// The variant this object last resolved to, if the object is still alive
    /// and still asks for the same one.
    ///
    /// One probe, and `strong_count` rather than `upgrade`: this answers 96 % of
    /// every draw's pipeline lookups on a driven Maps boot, so the second
    /// `HashMap::get` and the `Arc` that `upgrade` creates only to drop —
    /// two more atomics on the hottest path in the engine — were both paid per
    /// draw for nothing. A `Weak` reads `strong_count() == 0` exactly when its
    /// value has been dropped, which is the same question `upgrade().is_none()`
    /// asked.
    fn get(&mut self, identity: &super::types::PipelineObjectIdentity, key: &K) -> Option<V> {
        let id = identity.id();
        let (life, held_key, pipeline) = self.map.get(&id)?;
        if life.strong_count() == 0 {
            self.map.remove(&id);
            return None;
        }
        (held_key == key).then_some(*pipeline)
    }

    fn remember(&mut self, identity: &super::types::PipelineObjectIdentity, key: &K, value: V) {
        let id = identity.id();
        if !self.map.contains_key(&id) {
            // Object construction is rare. Reap identities whose runtime
            // object has gone before admitting the new one, so the index
            // follows object lifetime without a capacity or eviction policy.
            self.map.retain(|_, (life, _, _)| life.strong_count() != 0);
        }
        self.map
            .insert(id, (identity.downgrade(), key.clone(), value));
    }

    fn clear(&mut self) {
        self.map.clear();
    }
}

/// The layout every colour attachment of every render pass this device builds is
/// left in when the pass ends.
///
/// # This is the one spelling, and the registry derives from it
///
/// [`super::pools::ResourcePools::registry_mark_ready_at`] records the layout a
/// finished pass left its target in, and it must name the same layout this
/// `finalLayout` does or every subsequent barrier is issued with the wrong
/// `oldLayout` — which is undefined behaviour, not a validation error, because
/// nothing in Vulkan re-checks it. It reads this constant rather than repeating
/// the name.
///
/// # Why it is not `TRANSFER_SRC_OPTIMAL`
///
/// It used to be, so that a present blit or a readback copy could read the
/// target without transitioning it. The trade was badly priced, and the
/// mispricing is structural rather than workload-specific: on a driven
/// macos-13 sustained-animation boot the target was read ~1 200 times a second
/// and drawn into ~24 000 times a second, and **every one of those draws paid a
/// barrier back to `COLOR_ATTACHMENT_OPTIMAL` to undo an exit that only 5 % of
/// them would ever have used.** It is charged as
/// `passmerge_outside_target_layout` and took 82 % of draws on macos-13, 37 % on
/// macos-11 and 29 % on macos-12.
///
/// On a discrete GPU that round trip is a barrier and probably little else. On a
/// GPU with framebuffer compression — every Intel iGPU (CCS), every AMD part
/// (DCC), every tiler — a transition out of `COLOR_ATTACHMENT_OPTIMAL` and back
/// is a decompress and recompress of the whole attachment, per draw.
///
/// Nothing depended on the old exit. Every consumer that reads a colour target
/// — the present blit, the readback copy, the writeback copy, the copy-on-sample
/// snapshot, and both seed copies — already issues its own barrier into
/// `TRANSFER_SRC_OPTIMAL` first, unconditionally, because the barrier is what
/// carries the *dependency* and a matching layout would not have removed the
/// need for it. So those ~1 200 reads a second gain a real transition each and
/// the ~24 000 draws lose one, which is the whole change.
///
/// # And it is `GENERAL`, because a colour target here is also a texture
///
/// The layout above is where a pass *leaves* the attachment. This one is where
/// the attachment **lives**, and it is `GENERAL` for the reason Metal has no
/// layouts at all: a `MTLTexture` a render encoder writes is the same object a
/// later fragment shader samples, and nothing in that API marks the crossing. In
/// Vulkan the crossing is an image layout, and every layout that is optimal for
/// one of the two uses is illegal for the other — so a device that picks
/// `COLOR_ATTACHMENT_OPTIMAL` has to transition on every sample, and a
/// transition is exactly what a render pass instance may not contain. That is
/// `passmerge_outside_resident_layout`, 25 344 of 176 914 pass begins on a driven
/// macos-13 Maps boot, each closing a pass worth ~100 µs of GPU.
///
/// `GENERAL` is legal for both, so the crossing disappears and the resident is
/// where the next user wants it whichever user that is. What it gives up is
/// framebuffer compression, which on this host is real hardware —
/// Intel Arrow Lake CCS.
///
/// **Measured twice, and the second chain refuted the first.** Both are
/// interleaved driven macos-13 Maps boots of one binary with the layout moved and
/// nothing else, scored by `scripts/boot-score` on `sum us/draw`, every boot at
/// `throttle_ms=0`:
///
/// ```text
///                    chain C (/tmp/wb-outC0..C5)   chain D (/tmp/wb-outD0..D5)
/// COLOR_ATTACHMENT   22.95, 22.73, 25.50          17.44, 19.07, 18.05
/// GENERAL            21.43, 22.33, 21.93          20.86, 20.38
/// ```
///
/// Chain C alone reads as a disjoint −7.7 %. Chain D reads the *other way* by a
/// similar margin, and pooled the two arms overlap completely — 21.39 mean for
/// `GENERAL` against 20.96 for `COLOR_ATTACHMENT`. (Chain D also produced one boot
/// at `sum` 52.04 with `d/frame` 225, a different workload regime, excluded from
/// both means.)
///
/// So **this layout is perf-neutral as far as this host can measure**, and the
/// three-boot disjointness in chain C was chain position, not the arm. It is worth
/// stating why the earlier reading was believed: three position-matched pairs
/// agreeing one by one looks like a controlled result and is not one, because the
/// pairs share their position in a chain whose spread is larger than the effect.
///
/// The change stays, on **correctness** rather than on speed: one resting layout
/// is what lets every spelling below be one function instead of six, and it is
/// what makes a feedback colour slot legal (see [`PassKey::color_layout`]).
/// Do not re-quote the −7.7 %.
///
/// # It is a function, and [`crate::config::COLOR_GENERAL`] is the ablation
///
/// **Every** spelling of the layout has to move together — the pass's
/// `finalLayout`, the `initialLayout` a `LOAD` pass names, the subpass
/// reference, the registry's record of where the pass left the image, the layout
/// a sampled descriptor declares, and the comparisons
/// [`super::exec::pass_exit_needs_no_barrier`] and
/// [`super::pools::ResidentAccess::covered_by_pass_entry`] make. A `const` plus a
/// switch read beside it would be that second spelling, and two of them
/// disagreeing is a barrier naming an `oldLayout` the image is not in, which is
/// undefined behaviour and not an error. So there is one function and no
/// constant, and `REIMS_VGPU_COLOR_GENERAL=off` moves all of them back at once —
/// a narrowing, since it restores a transition rather than removing one.
pub(crate) fn color0_pass_exit_layout() -> vk::ImageLayout {
    if unified_color_layout() {
        vk::ImageLayout::GENERAL
    } else {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    }
}

/// Whether a colour target rests in one layout for its whole life. **Default
/// on**; `REIMS_VGPU_COLOR_GENERAL=off` is the ablation arm.
///
/// Read once. This decides the content of cached `VkRenderPass` objects and of
/// registry records that outlive them, so an answer that changed mid-boot would
/// leave both built under two layouts in one cache.
pub(crate) fn unified_color_layout() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            crate::config::read(crate::config::COLOR_GENERAL).0,
            crate::config::Switch::Off
        )
    })
}

/// The two `VK_SUBPASS_EXTERNAL` dependencies every render pass this device
/// builds must carry, covering **every** attachment class the pass has.
///
/// # Why this is unconditional, and what it cost to be conditional
///
/// Vulkan supplies an implicit external dependency for an attachment only *"if
/// there is no subpass dependency from `VK_SUBPASS_EXTERNAL` to the first
/// subpass that uses"* it — and the implicit one is per render pass, not per
/// attachment. So the moment a pass declares one explicit external dependency
/// for **any** reason, every attachment loses its implicit one.
///
/// This pass used to declare a pair only when it had a depth attachment, and
/// that pair named the `EARLY`/`LATE_FRAGMENT_TESTS` stages and the
/// depth-stencil accesses alone. The colour attachment silently lost the
/// synchronization it had been getting for free, so on a depth pass:
///
/// - the incoming transition into `COLOR_ATTACHMENT_OPTIMAL` was not ordered
///   against the `loadOp` clear that follows it, and
/// - the outgoing transition into `TRANSFER_SRC_OPTIMAL` was not ordered
///   against the subpass's own colour store, nor against the copy that reads
///   the target afterwards.
///
/// All three were reported by the Khronos synchronization validation layer on a
/// driven macos-11 boot, as `SYNC-HAZARD-WRITE-AFTER-WRITE` at
/// `vkCmdBeginRenderPass` and `vkCmdEndRenderPass` and
/// `SYNC-HAZARD-READ-AFTER-WRITE` at the `vkCmdCopyImage` /
/// `vkCmdCopyImageToBuffer` that follows.
///
/// Building both dependencies here, always, from the pass's own composition is
/// what makes the split unrepresentable: there is no longer an arm that adds a
/// dependency for one attachment class without stating the others.
///
/// # The outgoing `dst` scope covers every way the attachment is read next
///
/// Every slot now exits at [`color0_pass_exit_layout`], so the nearest consumer
/// is usually the next draw into the same target — which is why the attachment
/// stages and accesses are in the destination scope, and why
/// [`super::exec::pass_exit_needs_no_barrier`] may then drop that draw's own
/// barrier entirely. `TRANSFER` and `FRAGMENT_SHADER` stay named because a
/// readback, a present blit or a later sample can follow instead; each of those
/// issues its own transition, and this is the scope that transition orders
/// against.
///
/// # The incoming dependency is what makes the skip legal
///
/// `VK_SUBPASS_EXTERNAL` as `srcSubpass` scopes every command submitted before
/// the render pass instance, in submission order. So the incoming dependency
/// here — colour writes to attachment reads and writes — already orders the
/// previous draw's store against this pass's `loadOp`, with no barrier from the
/// draw. Weakening its source scope would silently make that skip unsound.
fn external_dependencies(
    has_depth: bool,
    color_input: bool,
    host_accessible_color0: bool,
    // Taken as an argument rather than read here, so both arms are reachable
    // from a test. The switch is read once, at the one call site that builds a
    // pass; see [`pass_exit_scope_narrow`].
    exit_scope_narrow: bool,
) -> [vk::SubpassDependency; 2] {
    // Colour is unconditional: every pass this device builds has slot 0.
    let mut attach_stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    let mut attach_writes = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    let mut attach_reads = vk::AccessFlags::COLOR_ATTACHMENT_READ;
    if has_depth {
        attach_stages |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        attach_writes |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        attach_reads |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
    }
    // Framebuffer fetch reads attachment 0 through the fragment stage, so the
    // incoming transition has to be visible to that read too. The intra-subpass
    // ordering is the separate `BY_REGION` dependency; this is the entry.
    //
    // The shader stages are unconditional, and they are what
    // [`super::pools::ResidentAccess::covered_by_pass_entry`] rests on. A draw
    // inside this pass may sample a resident an *earlier* pass wrote, and with
    // one resting layout there is no transition left for that draw to record —
    // only a visibility request, which this entry then carries for every such
    // draw at once instead of each of them closing the pass to state it.
    // Weakening this to the attachment stages makes that skip a missing
    // dependency, which is a stale frame and no error.
    let (in_dst_stages, mut in_dst_access) = (
        attach_stages
            | vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER,
        attach_writes | attach_reads | vk::AccessFlags::SHADER_READ,
    );
    if color_input {
        in_dst_access |= vk::AccessFlags::INPUT_ATTACHMENT_READ;
    }
    let mut source_stages = attach_stages | vk::PipelineStageFlags::TRANSFER;
    let mut source_access = attach_writes | vk::AccessFlags::TRANSFER_WRITE;
    if host_accessible_color0 {
        source_stages |= vk::PipelineStageFlags::HOST;
        source_access |= vk::AccessFlags::HOST_WRITE;
    }
    [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            // Whatever last wrote these images: a previous pass's colour store,
            // a depth store, or the transfer that seeded a LOAD attachment.
            .src_stage_mask(source_stages)
            .src_access_mask(source_access)
            .dst_stage_mask(in_dst_stages)
            .dst_access_mask(in_dst_access),
        {
            // Narrowing this to the attachment stages alone is the probe: see
            // [`pass_exit_scope_narrow`] for what it is asking and what it must
            // not break.
            let (dst_stages, dst_access) = if exit_scope_narrow {
                (attach_stages, attach_writes | attach_reads)
            } else {
                (
                    vk::PipelineStageFlags::TRANSFER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER
                        | attach_stages,
                    vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::SHADER_READ
                        | attach_writes
                        | attach_reads,
                )
            };
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(attach_stages)
                .src_access_mask(attach_writes)
                .dst_stage_mask(dst_stages)
                .dst_access_mask(dst_access)
        },
    ]
}

/// The colour write a feedback draw's own sampled read must be ordered after.
pub(crate) const COLOR_FEEDBACK_SRC: (vk::PipelineStageFlags, vk::AccessFlags) = (
    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
);

/// The sampled read a feedback draw performs of the attachment it is writing.
///
/// `FRAGMENT_SHADER` alone, and that is a rule rather than a choice. A subpass
/// self-dependency whose `srcStageMask` contains a framebuffer-space stage may
/// name only framebuffer-space stages in its `dstStageMask`
/// (`VUID-VkSubpassDependency-srcSubpass-06809`), and `VERTEX_SHADER` is not one.
/// It was never right on its own terms either: a feedback loop is a fragment
/// reading the pixel it is about to write, and `BY_REGION` below is that
/// same-pixel claim. A vertex stage reading the attachment is not a feedback loop
/// and could not be ordered by this dependency whatever it named.
pub(crate) const COLOR_FEEDBACK_DST: (vk::PipelineStageFlags, vk::AccessFlags) = (
    vk::PipelineStageFlags::FRAGMENT_SHADER,
    vk::AccessFlags::SHADER_READ,
);

/// The subpass self-dependency that orders a feedback draw's sampled read after
/// the colour writes of the draws before it in the same pass instance.
///
/// **Declared on every pass this device builds, whether or not any draw in it
/// feeds back.** A self-dependency costs nothing until a
/// `vkCmdPipelineBarrier` inside the pass invokes it — but its *presence* changes
/// `dependencyCount`, and Vulkan render-pass compatibility spares initial/final
/// layouts, attachment-reference layouts and load/store ops while sparing nothing
/// about dependencies. So a pass built with this dependency and one built without
/// it are **incompatible**, and a `VkFramebuffer` created against either cannot be
/// used with the other. Declaring it conditionally is what produced
/// `VUID-VkRenderPassBeginInfo-renderPass-00904` (`dependencyCount is
/// incompatible`) on a driven Maps boot, on both layout arms.
///
/// The general rule, which is the one to keep: **a render pass may only vary in
/// ways [`PassKey::framebuffer_compatibility`] preserves.** Anything that key
/// erases must not reach the `VkRenderPassCreateInfo`.
///
/// `dependency_flags` derives the extension bit from the attachment layout via
/// [`super::feedback_transition_dependency`], so the arm that has no feedback
/// layout cannot ask for the extension's flag.
fn color_feedback_self_dependency(color0_layout: vk::ImageLayout) -> vk::SubpassDependency {
    vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(0)
        .src_stage_mask(COLOR_FEEDBACK_SRC.0)
        .src_access_mask(COLOR_FEEDBACK_SRC.1)
        .dst_stage_mask(COLOR_FEEDBACK_DST.0)
        .dst_access_mask(COLOR_FEEDBACK_DST.1)
        .dependency_flags(
            vk::DependencyFlags::BY_REGION | super::feedback_transition_dependency(color0_layout),
        )
}

/// Whether the outgoing external dependency names only the attachment stages.
///
/// **Probe, default off.** See [`crate::config::PASS_EXIT_NARROW`] for the whole
/// argument; in one line, the shipping scope names `TRANSFER | FRAGMENT_SHADER`
/// with `TRANSFER_READ | SHADER_READ`, which asks this driver for a render-cache
/// flush and a texture-cache invalidate at **every** `vkCmdEndRenderPass`, and a
/// pass boundary is the single largest cost in this device on the iGPU pathway.
///
/// Read once. This decides the content of a cached `VkRenderPass`, so a value
/// that changed mid-boot would leave passes built under both answers in one
/// cache and make the arm unreadable.
fn pass_exit_scope_narrow() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            crate::config::read(crate::config::PASS_EXIT_NARROW).0,
            crate::config::Switch::On
        )
    })
}

/// The guest's sampler declaration, in the vocabulary of the layer that owns
/// what a wire tag means.
///
/// `key` is the guest's request and stays the cache index, so the cache and the
/// negative cache still answer for what was asked for; this is a projection of
/// it and is never written back.
fn sampler_shape(key: &SamplerStateKey) -> reims_vgpu_core::sampler::SamplerShape {
    reims_vgpu_core::sampler::SamplerShape {
        min_filter: key.min_filter,
        mag_filter: key.mag_filter,
        mip_filter: key.mip_filter,
        s_address: key.address_mode_u,
        t_address: key.address_mode_v,
        r_address: key.address_mode_w,
        max_anisotropy: key.max_anisotropy,
        lod_min_clamp: f32::from_bits(key.lod_min),
        lod_max_clamp: f32::from_bits(key.lod_max),
        compare_function: key.compare_function.mtl_ordinal(),
        // Metal's descriptor has no separate flag and this key does not either:
        // a sampler that compares with `Never` and one that does not compare
        // are indistinguishable by the time they reach here, and `Never` is
        // Metal's own default for the field.
        compare_enabled: key.compare_function != super::types::SamplerCompareFunction::Never,
        border_color: key.border_color,
        normalized_coordinates: !key.unnormalized_coordinates,
    }
}

/// One colour attachment's blend terms, in the vocabulary of the layer that
/// owns what they mean.
///
/// `None` is `blendingEnabled` clear, which is the whole of what the flag
/// means: with it clear Metal evaluates no equation, so the six ordinals it
/// left behind are not parsed and cannot refuse anything. The mask is outside
/// that, because `MTLColorWriteMask` applies whether or not the slot blends.
///
/// Unlike [`depth_stencil_state`] this one can genuinely fail. The ordinals
/// travel from the guest unparsed, so a value outside `MTLBlendFactor` or
/// `MTLBlendOperation` arrives here rather than at the decoder — see
/// [`super::types::BlendStateResource`] for why that is the arrangement.
fn color_attachment_state(
    blend: Option<BlendKey>,
    mask: ColorWriteMask,
) -> Result<reims_vgpu_core::blend::ColorAttachmentState, reims_vgpu_core::blend::BlendRefusal> {
    reims_vgpu_core::blend::ColorAttachmentShape {
        blending_enabled: blend.is_some(),
        src_rgb: blend.map_or(0, |b| b.src_rgb),
        dst_rgb: blend.map_or(0, |b| b.dst_rgb),
        op_rgb: blend.map_or(0, |b| b.op_rgb),
        src_alpha: blend.map_or(0, |b| b.src_alpha),
        dst_alpha: blend.map_or(0, |b| b.dst_alpha),
        op_alpha: blend.map_or(0, |b| b.op_alpha),
        write_mask: mask,
    }
    .checked()
}

impl ObjectCaches {
    pub(crate) fn new() -> Self {
        Self {
            shaders: ObjectCache::new(),
            layouts: LayoutTable::new(),
            attr_sets: SliceIntern::new(),
            passes: ObjectCache::new(),
            pipelines: ObjectCache::new(),
            pipeline_objects: ObjectVariantIndex::default(),
            samplers: ObjectCache::new(),
            compute_pipelines: ObjectCache::new(),
            shader_digests: ShaderDigestIndex::default(),
        }
    }

    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        // This index borrows handles owned by `pipelines`; forget those echoes
        // before destroying the authoritative objects.
        self.pipeline_objects.clear();
        for p in self.pipelines.take_all() {
            device.destroy_pipeline(p, None);
        }
        for p in self.compute_pipelines.take_all() {
            device.destroy_pipeline(p, None);
        }
        for (dsl, pl) in self.layouts.take_all() {
            device.destroy_pipeline_layout(pl, None);
            if dsl != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(dsl, None);
            }
        }
        // No handles of its own; it is emptied here so a fresh device does not
        // inherit ids issued against the one that just went away.
        self.attr_sets.clear();
        for rp in self.passes.take_all() {
            device.destroy_render_pass(rp, None);
        }
        for s in self.shaders.take_all() {
            device.destroy_shader_module(s, None);
        }
        for s in self.samplers.take_all() {
            device.destroy_sampler(s, None);
        }
    }

    /// Live entries in each cache, in the order
    /// `(shaders, layouts, attribute sets, passes, pipelines, samplers,
    /// compute_pipelines)`.
    ///
    /// Published because [`ObjectCache`] is unbounded on the argument that its
    /// entry count is the guest's distinct object set and therefore plateaus.
    /// That is a claim about a running guest, and this is the reading that can
    /// falsify it: a level that climbs for the life of a boot instead of
    /// settling means some key is carrying per-frame state and the argument is
    /// wrong for that cache. Levels, not deltas — the census line says so.
    pub(crate) fn levels(&self) -> [usize; 7] {
        [
            self.shaders.len(),
            self.layouts.len(),
            self.attr_sets.len(),
            self.passes.len(),
            self.pipelines.len(),
            self.samplers.len(),
            self.compute_pipelines.len(),
        ]
    }

    pub(crate) fn clear_logical(&mut self) {
        // Before the modules it indexes, so no window exists where a front-index
        // hit names a digest whose module has already gone.
        self.shader_digests.clear();
        self.shaders.clear();
        self.layouts.clear();
        self.attr_sets.clear();
        self.passes.clear();
        self.pipeline_objects.clear();
        self.pipelines.clear();
        self.samplers.clear();
        self.compute_pipelines.clear();
    }

    /// Report a driver call this device refused to repeat, and hand back the
    /// error every one of the three call sites caches negatively.
    ///
    /// One place rather than three so the three cannot drift into three
    /// different accounts of the same event — and so the line always carries
    /// both the key (which identifies the call) and what the dead process called
    /// it (which is the only human-readable thing about it).
    fn note_quarantined(
        &self,
        site: &'static str,
        hit: &super::driver_breadcrumb::quarantine::Quarantined,
    ) -> DrawError {
        let reason = super::reason::DrawReason::DriverCallQuarantined;
        crate::observe::Emit::decline("driver_quarantine", &reason)
            .field("site", site)
            .field("key", &hit.key)
            .field("previously", &hit.previously)
            .field(
                "list",
                super::driver_breadcrumb::quarantine::list_path().display(),
            )
            .fail();
        DrawError::Unsupported(reason)
    }

    /// [`Self::get_or_create_shader`] with the three whole-module walks skipped
    /// for an allocation that has been through it before.
    ///
    /// The draw path is the caller that needs this: it binds two modules a draw
    /// at ~30 000 draws a second, from `Arc`s the runtime holds for each
    /// shader's lifetime, into a cache that on a driven macos-13 boot reports
    /// `shader_misses=0`. `pl_shader_us` was **63 ms of every second** — the
    /// largest single item inside `engine_us` — spent deriving a key for a
    /// module already in hand.
    ///
    /// The compute path calls the walking form directly and deliberately: its
    /// `spirv` is an owned `Vec` with no stable allocation to key on, and a
    /// dispatch is three orders rarer than a draw.
    pub(crate) unsafe fn get_or_create_shader_memoized(
        &mut self,
        ctx: &DeviceContext,
        words: &std::sync::Arc<Vec<u32>>,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<(Digest128, vk::ShaderModule), DrawError> {
        if let Some(key) = self.shader_digests.get(words) {
            // Negative before positive, in the order the walking form asks them:
            // a module this device refused is refused again without being
            // rebuilt, and without the front index quietly promoting it.
            if let Some(err) = self.shaders.get_negative(&key) {
                counters.shader_misses.fetch_add(1, Ordering::Relaxed);
                return Err(err);
            }
            if let Some(module) = self.shaders.get(&key) {
                counters.shader_hits.fetch_add(1, Ordering::Relaxed);
                counters.shader_digest_hits.fetch_add(1, Ordering::Relaxed);
                return Ok((key, module));
            }
            // The module was evicted or destroyed under a digest this index
            // still names. Falling through re-walks and re-creates it, which is
            // why the index may hold a digest the cache does not.
        }
        let (key, module) = self.get_or_create_shader(ctx, words, counters, pools)?;
        self.shader_digests.insert(words, key);
        Ok((key, module))
    }

    pub(crate) unsafe fn get_or_create_shader(
        &mut self,
        ctx: &DeviceContext,
        words: &[u32],
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<(Digest128, vk::ShaderModule), DrawError> {
        // Declare the storage-image capabilities this module's own contents
        // require, before it is keyed or validated.
        //
        // This is the only place every module from every path passes through, so
        // it is the only place the question can be asked once. It used to be
        // asked at one producer instead, and phrased as provenance — "did *this
        // device* retarget a binding to `Unknown`?" — which is a claim about how
        // a module came to need the capability rather than whether it does. The
        // translator emits `Unknown`-format storage images and extended formats
        // of its own accord; those modules arrived here undeclared and were
        // rejected, losing the dispatch. Both x86 rails lose compute work to it.
        //
        // A capability whose Vulkan feature was not enabled at device creation
        // cannot be declared — that is invalid usage, and an invalid module is
        // undefined behaviour inside a driver rather than an error it returns —
        // so an unsupported requirement is a named decline instead.
        // Both passes below walk the whole module, and so does the digest, so the
        // charge is levied once here on the words the caller handed over rather
        // than at each walk.
        counters
            .shader_hash_words
            .fetch_add(words.len() as u64, Ordering::Relaxed);
        let need = crate::runtime::spirv_bind::required_image_capabilities(words);
        let mut patched;
        let words: &[u32] = if need.any() {
            let missing = (need.extended_formats && !ctx.spirv_storage_extended_formats)
                || (need.write_without_format && !ctx.spirv_storage_write_without_format)
                || (need.read_without_format && !ctx.spirv_storage_read_without_format);
            if missing {
                let err = DrawError::Unsupported(super::reason::DrawReason::SpirvInvalid);
                crate::observe::fail(format!(
                    "spirv_capability reason=host_lacks_feature words={} \
                     need_extended={} need_write={} need_read={} \
                     have_extended={} have_write={} have_read={}",
                    words.len(),
                    need.extended_formats,
                    need.write_without_format,
                    need.read_without_format,
                    ctx.spirv_storage_extended_formats,
                    ctx.spirv_storage_write_without_format,
                    ctx.spirv_storage_read_without_format,
                ));
                let key = Digest128::of_u32_words(words);
                self.shaders.insert_negative(key, err.clone());
                return Err(err);
            }
            patched = words.to_vec();
            let added = crate::runtime::spirv_bind::ensure_image_capabilities(&mut patched, &need);
            if added.any() {
                crate::observe::off(format!(
                    "spirv_capability added extended={} write={} read={} words={}",
                    added.extended_formats,
                    added.write_without_format,
                    added.read_without_format,
                    patched.len()
                ));
            }
            &patched
        } else {
            words
        };
        let key = Digest128::of_u32_words(words);
        if let Some(err) = self.shaders.get_negative(&key) {
            counters.shader_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(m) = self.shaders.get(&key) {
            counters.shader_hits.fetch_add(1, Ordering::Relaxed);
            return Ok((key, m));
        }
        counters.shader_misses.fetch_add(1, Ordering::Relaxed);
        // Last gate before the driver, and the only place every module from
        // every path passes through exactly once. An invalid module is
        // undefined behaviour inside a driver rather than an error it returns,
        // and one has been observed ending the VM process — so it becomes a
        // negative cache entry here and the guest's work is declined by name.
        // See `crate::runtime::spirv_bind::validate`.
        if let crate::runtime::spirv_bind::SpirvValidation::Rejected(why) =
            crate::runtime::spirv_bind::validate(words)
        {
            let err = DrawError::Unsupported(super::reason::DrawReason::SpirvInvalid);
            // Print what the capability derivation saw alongside the
            // validator's complaint. When the two disagree the difference is
            // the whole bug, and neither one alone says which walk is wrong.
            crate::observe::fail(format!(
                "spirv_validate reason=module_rejected words={} need={:?} imgs={:?} detail={why}",
                words.len(),
                crate::runtime::spirv_bind::required_image_capabilities(words),
                crate::runtime::spirv_bind::image_type_census(words),
            ));
            // The complaint above names instructions by result id, which cannot
            // be read without the module they belong to. Keep it.
            super::driver_breadcrumb::keep_rejected_module(
                &format!("{:016x}{:016x}", key.a, key.b),
                words,
            );
            self.shaders.insert_negative(key, err.clone());
            return Err(err);
        }
        // The driver parses SPIR-V here, so this is one of the three calls that
        // can end the process on a module this device assembled — the other two
        // being the compute and graphics pipeline compiles below. See
        // `driver_breadcrumb` for why the words go to disk across it.
        let breadcrumb = match super::driver_breadcrumb::DriverBreadcrumb::arm(
            "create_shader_module",
            &[("module", words)],
        ) {
            Ok(breadcrumb) => breadcrumb,
            Err(hit) => {
                let err = self.note_quarantined("create_shader_module", &hit);
                self.shaders.insert_negative(key, err.clone());
                return Err(err);
            }
        };
        let created = ctx
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None);
        breadcrumb.disarm();
        let module = created.map_err(|e| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateShaderModule, e));
            self.shaders.insert_negative(key, err.clone());
            err
        })?;
        counters.note_create(CreateSite::ShaderModule);
        if let Some(old) = self.shaders.insert(key, module) {
            pools.dispose(&ctx.device, DeferredHandle::ShaderModule(old));
        }
        Ok((key, module))
    }

    /// The id for this draw's vertex-attribute set, issuing one on first sight.
    ///
    /// Takes a slice for the reason [`Self::get_or_create_layout`] does: the
    /// caller's attributes live in reusable scratch, and the id is what its
    /// pipeline key carries.
    pub(crate) fn intern_attrs(&mut self, attrs: &[AttrKey]) -> AttrsId {
        AttrsId(self.attr_sets.intern(attrs))
    }

    /// Resolve the layout these canonical bindings name, creating it once.
    ///
    /// Takes a **slice**, not a key: the caller's bindings live in reusable
    /// scratch, and the returned [`ResolvedLayout::id`] is what its pipeline key
    /// carries. Takes no `&mut ResourcePools` either, because the only thing the
    /// old signature used it for was disposing a layout an insert displaced, and
    /// [`LayoutTable`] appends — a content lookup that missed cannot displace
    /// the entry it just failed to find.
    pub(crate) unsafe fn get_or_create_layout(
        &mut self,
        ctx: &DeviceContext,
        bindings: &[BindingSig],
        push_constant: Option<(u32, u32)>,
        counters: &EngineCounters,
    ) -> Result<ResolvedLayout, DrawError> {
        if let Some(err) = self.layouts.get_negative(bindings, push_constant) {
            counters.layout_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(resolved) = self.layouts.get(bindings, push_constant) {
            counters.layout_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(resolved);
        }
        counters.layout_misses.fetch_add(1, Ordering::Relaxed);
        let push_descriptors = layout_uses_push_descriptors(bindings, ctx.caps.push_descriptor);
        let layout_bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(vk::DescriptorType::from_raw(b.ty as i32))
                    .descriptor_count(b.count)
                    .stage_flags(vk::ShaderStageFlags::from_raw(b.stages))
            })
            .collect();
        let dsl = if layout_bindings.is_empty() {
            vk::DescriptorSetLayout::null()
        } else {
            let binding_flags: Vec<_> = bindings
                .iter()
                .map(|binding| {
                    if binding.count > 1 {
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    } else {
                        vk::DescriptorBindingFlags::empty()
                    }
                })
                .collect();
            let mut flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&binding_flags);
            let mut create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&layout_bindings)
                .push_next(&mut flags);
            if push_descriptors {
                create_info =
                    create_info.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
            }
            let d = ctx
                .device
                .create_descriptor_set_layout(&create_info, None)
                .map_err(|e| {
                    let err =
                        DrawError::VkCall(VkCall::new(VkOp::CachesCreateDescriptorSetLayout, e));
                    self.layouts
                        .insert_negative(bindings, push_constant, err.clone());
                    err
                })?;
            counters.note_create(CreateSite::DescriptorSetLayout);
            d
        };
        let set_layouts: Vec<vk::DescriptorSetLayout> = if dsl == vk::DescriptorSetLayout::null() {
            Vec::new()
        } else {
            vec![dsl]
        };
        let push_ranges: Vec<_> = push_constant
            .map(|(offset, size)| {
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(offset)
                    .size(size)
            })
            .into_iter()
            .collect();
        let pl = ctx
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )
            .map_err(|e| {
                if dsl != vk::DescriptorSetLayout::null() {
                    ctx.device.destroy_descriptor_set_layout(dsl, None);
                }
                let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreatePipelineLayout, e));
                self.layouts
                    .insert_negative(bindings, push_constant, err.clone());
                err
            })?;
        counters.note_create(CreateSite::PipelineLayout);
        Ok(self
            .layouts
            .insert(bindings, push_constant, dsl, pl, push_descriptors))
    }

    pub(crate) unsafe fn get_or_create_pass(
        &mut self,
        ctx: &DeviceContext,
        key: PassKey,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<vk::RenderPass, DrawError> {
        if let Some(err) = self.passes.get_negative(&key) {
            counters.pass_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(rp) = self.passes.get(&key) {
            counters.pass_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(rp);
        }
        counters.pass_misses.fetch_add(1, Ordering::Relaxed);
        let target_format = key.color0_format;
        // A `LOAD` pass names the layout the *previous* pass left the attachment
        // in, so it reads the exit constant rather than respelling the layout.
        // These two agreeing is what lets `exec::pass_exit_needs_no_barrier`
        // drop the transition between consecutive draws into one target; a
        // second spelling here would make that skip a missing transition the
        // first time somebody changed one of them.
        let color0_final = key.color_final_layout(0);
        let (load_op, initial) = key.color0_load.ops(color0_final);
        // Slot 0 (primary) and the secondary attachments (slot 1..) now exit the
        // same way, at [`color0_pass_exit_layout`], and for the same reason: a
        // consumer's barrier is what establishes the dependency, so leaving the
        // mask at COLOR_ATTACHMENT_OPTIMAL forces that barrier to fire with a
        // colour-write source scope rather than being skipped as a no-op. The
        // registry tracks this layout.
        let mut attachments = vec![vk::AttachmentDescription::default()
            .format(target_format)
            .samples(vk_sample_count(key.sample_count))
            .load_op(load_op)
            // Vulkan render-pass splits are an implementation detail inside one
            // guest encoder. Preserve the scratch source across such a split;
            // the guest's resolve-only store action still exposes only the
            // single-sample destination.
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial)
            .final_layout(color0_final)];
        // Framebuffer fetch: when attachment 0 is also a subpass input, BOTH
        // references must use GENERAL (same-attachment color+input requires it);
        // the pass still transitions initial→GENERAL→final automatically.
        let color0_layout = key.color_layout(0);
        let mut color_ref = vec![vk::AttachmentReference::default()
            .attachment(0)
            .layout(color0_layout)];
        for (i, sec) in key.secondary[..key.secondary_count as usize]
            .iter()
            .enumerate()
        {
            let attachment_index = i + 1;
            let final_layout = key.color_final_layout(attachment_index);
            let (sload, sinitial) = if sec.load {
                (vk::AttachmentLoadOp::LOAD, final_layout)
            } else {
                (vk::AttachmentLoadOp::CLEAR, vk::ImageLayout::UNDEFINED)
            };
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(sec.format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(sload)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(sinitial)
                    .final_layout(final_layout),
            );
            color_ref.push(
                vk::AttachmentReference::default()
                    .attachment(1 + i as u32)
                    .layout(key.color_layout(attachment_index)),
            );
        }
        let mut resolve_ref = Vec::new();
        if key.multisample_resolve {
            let resolve_index = attachments.len() as u32;
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(target_format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(color0_layout)
                    .final_layout(color0_final),
            );
            resolve_ref.push(
                vk::AttachmentReference::default()
                    .attachment(resolve_index)
                    .layout(color0_layout),
            );
            resolve_ref.extend((1..color_ref.len()).map(|_| {
                vk::AttachmentReference::default()
                    .attachment(vk::ATTACHMENT_UNUSED)
                    .layout(vk::ImageLayout::UNDEFINED)
            }));
        }
        // Depth attachment is appended LAST (after color + secondaries), so its
        // index is the current attachment count and color slot 0 is untouched.
        let depth_ref = key.depth.map(|d| {
            let (dload, dinitial) = if d.load {
                (
                    vk::AttachmentLoadOp::LOAD,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                )
            } else {
                (vk::AttachmentLoadOp::CLEAR, vk::ImageLayout::UNDEFINED)
            };
            // Stencil test active ⇒ combined format with a live STENCIL aspect
            // (CLEAR/STORE mirroring depth). Depth-only stays D32_SFLOAT with
            // DONT_CARE stencil, byte-identical to the pre-stencil pass.
            let (dformat, sload, sstore) = if d.stencil {
                (
                    ctx.depth_stencil_format,
                    dload,
                    vk::AttachmentStoreOp::STORE,
                )
            } else {
                (
                    translate::pixel::TRANSIENT_DEPTH_FORMAT,
                    vk::AttachmentLoadOp::DONT_CARE,
                    vk::AttachmentStoreOp::DONT_CARE,
                )
            };
            let index = attachments.len() as u32;
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(dformat)
                    .samples(vk_sample_count(key.sample_count))
                    .load_op(dload)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(sload)
                    .stencil_store_op(sstore)
                    .initial_layout(dinitial)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            );
            vk::AttachmentReference::default()
                .attachment(index)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        });
        let input_ref = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(key.color_layout(0))];
        let mut subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref);
        if !resolve_ref.is_empty() {
            subpass_desc = subpass_desc.resolve_attachments(&resolve_ref);
        }
        if key.color_input {
            subpass_desc = subpass_desc.input_attachments(&input_ref);
        }
        if let Some(depth_ref) = &depth_ref {
            subpass_desc = subpass_desc.depth_stencil_attachment(depth_ref);
        }
        let subpass = [subpass_desc];
        // Framebuffer-fetch feedback loop: the same-pixel color-write →
        // input-read ordering within the one subpass. BY_REGION keeps it
        // framebuffer-local (the form MoltenVK lowers to tile-memory fetch).
        let fetch_dep = vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
            .dependency_flags(vk::DependencyFlags::BY_REGION);
        let mut deps: Vec<vk::SubpassDependency> = external_dependencies(
            key.depth.is_some(),
            key.color_input,
            key.host_accessible_color0,
            pass_exit_scope_narrow(),
        )
        .to_vec();
        if key.color_input {
            deps.push(fetch_dep);
        }
        deps.push(color_feedback_self_dependency(key.color_layout(0)));
        let rp_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpass)
            .dependencies(&deps);
        let rp = ctx.device.create_render_pass(&rp_info, None).map_err(|e| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateRenderPass, e));
            self.passes.insert_negative(key, err.clone());
            err
        })?;
        counters.note_create(CreateSite::RenderPass);
        if let Some(old) = self.passes.insert(key, rp) {
            pools.dispose(&ctx.device, DeferredHandle::RenderPass(old));
        }
        Ok(rp)
    }

    pub(crate) unsafe fn get_or_create_sampler(
        &mut self,
        ctx: &DeviceContext,
        key: &SamplerStateKey,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<vk::Sampler, DrawError> {
        if let Some(err) = self.samplers.get_negative(key) {
            counters.sampler_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(s) = self.samplers.get(key) {
            counters.sampler_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(s);
        }
        counters.sampler_misses.fetch_add(1, Ordering::Relaxed);
        let shape = sampler_shape(key);
        let state = match shape.checked() {
            Ok(state) => state,
            Err(refusal) => {
                crate::observe::Emit::decline("vk_engine_sampler", &refusal).fail();
                let err =
                    DrawError::Unsupported(super::reason::DrawReason::SamplerDeclaration(refusal));
                self.samplers.insert_negative(*key, err.clone());
                return Err(err);
            }
        };
        // One line per distinct conformed sampler this boot creates — the cache
        // is what bounds it, so a workload with three of them logs three lines
        // however many million binds it does.
        //
        // # What this is measuring, and why it is not a decline
        //
        // `VUID-vkCmdDispatch-None-08610`/`-08611` forbid an unnormalized
        // sampler being *used* by an implicit-LOD, `Proj`, `Dref`, `Bias` or
        // `Offset` sample, and `-08611` is the one violation a driven macos-11
        // boot under the Khronos validation layer still reports after every
        // other one here was fixed. That is a property of the SPIR-V
        // instruction, so this device cannot repair it — but it can say which
        // samplers are candidates.
        //
        // # The VUID is real and it is **not** what hangs this GPU
        //
        // A probe forced `unnormalized_coordinates` false here, which makes
        // `-08611` unreachable by construction: the workload froze anyway, with
        // the same two device recreates. So the violation is a passenger. Do not
        // read a future `sampler_unnormalized` line as a lost frame — it says a
        // sampler exists whose declaration was brought inside the restriction
        // both APIs place on it, and nothing about what was drawn with it.
        if state.unnormalized_conformed() {
            crate::observe::off(format!(
                "sampler_unnormalized min_mag_differed={} \
                 min={} mag={} mip={} address_u={} address_v={} aniso={}",
                key.min_filter != key.mag_filter,
                key.min_filter,
                key.mag_filter,
                key.mip_filter,
                key.address_mode_u,
                key.address_mode_v,
                key.max_anisotropy,
            ));
        }
        let plan = match reims_vgpu_vulkan::sampler::plan(state, ctx.sampler_cell()) {
            Ok(plan) => plan,
            Err(refusal) => {
                // Fail-visible here, at the check, and exactly once per sampler
                // key: the negative cache means a replay returns without
                // reaching this line, and the returned `DrawError` reaches the
                // log only if some caller happens to render it.
                crate::observe::Emit::decline("vk_engine_sampler", &refusal).fail();
                // The negative cache stores the typed `DrawError`, so a replay
                // returns this exact decline — slug and all — not a re-rendered
                // `Vulkan(String)` that would drop the reason to
                // `vk_engine_vk_untyped`.
                let err = DrawError::Unsupported(super::reason::DrawReason::SamplerDevice(refusal));
                self.samplers.insert_negative(*key, err.clone());
                return Err(err);
            }
        };
        let sampler = ctx
            .device
            .create_sampler(&plan.create_info(), None)
            .map_err(|e| {
                let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateSampler, e));
                self.samplers.insert_negative(*key, err.clone());
                err
            })?;
        counters.note_create(CreateSite::Sampler);
        if let Some(old) = self.samplers.insert(*key, sampler) {
            pools.dispose(&ctx.device, DeferredHandle::Sampler(old));
        }
        Ok(sampler)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "pipeline creation mirrors the Vulkan shader, layout, pass, and cache handles"
    )]
    pub(crate) unsafe fn get_or_create_pipeline(
        &mut self,
        ctx: &DeviceContext,
        key: &PipelineKey,
        pipeline_object: Option<&super::types::PipelineObjectIdentity>,
        vert_module: vk::ShaderModule,
        // The post-relocation words `vert_module` was built from. Read only to
        // answer how wide this shader's stage-in reads are, and only on a host
        // that substitutes a vertex format; see the resolution loop below.
        vert_spirv: &[u32],
        frag_module: vk::ShaderModule,
        // Read only by the driver breadcrumb: a graphics compile consumes both
        // stages and nothing outside the driver can say which one it choked on,
        // so both go to disk across the call.
        frag_spirv: &[u32],
        pipeline_layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<vk::Pipeline, DrawError> {
        if let Some(identity) = pipeline_object {
            if let Some(pipeline) = self.pipeline_objects.get(identity, key) {
                counters.pipeline_hits.fetch_add(1, Ordering::Relaxed);
                counters
                    .pipeline_object_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(pipeline);
            }
        }
        if let Some(err) = self.pipelines.get_negative(key) {
            counters.pipeline_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some((p, front)) = self.pipelines.get_routed(key) {
            counters.pipeline_hits.fetch_add(1, Ordering::Relaxed);
            if front {
                counters.pipeline_front_hits.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(identity) = pipeline_object {
                self.pipeline_objects.remember(identity, key, p);
            }
            return Ok(p);
        }
        counters.pipeline_misses.fetch_add(1, Ordering::Relaxed);

        // Every colour attachment, parsed once and planned once.
        //
        // Built here rather than beside the create-info below because both
        // things that can go wrong are refusals of the whole pipeline, and a
        // refusal belongs with the other capability checks in this run: typed
        // decline, cached negatively so a replay returns this exact reason.
        //
        // Every attachment participates, not just slot 0. The secondaries
        // carry their own decoded blend, and both questions are asked of the
        // set: a pipeline is invalid if *any* attachment names a `SRC1_*`
        // factor without `dualSrcBlend`, and invalid if they *disagree*
        // without `independentBlend` — which no single attachment can be
        // blamed for, which is why the second check takes the list.
        let blend_cell = reims_vgpu_vulkan::blend::BlendCell {
            dual_source: ctx.features.dual_src_blend,
            independent: ctx.features.independent_blend,
        };
        let mut blend_plans = Vec::with_capacity(1 + key.pass.secondary_count());
        {
            let refuse = |reason: super::reason::DrawReason| {
                crate::observe::Emit::decline("vk_engine_pipeline", &reason).fail();
                DrawError::Unsupported(reason)
            };
            for slot in 0..=key.pass.secondary_count() {
                let blend = if slot == 0 {
                    key.blend
                } else {
                    key.secondary_blend[slot - 1]
                };
                let attempt = color_attachment_state(blend, key.color_write_mask[slot])
                    .map_err(|r| refuse(super::reason::DrawReason::BlendDeclaration(r)))
                    .and_then(|state| {
                        reims_vgpu_vulkan::blend::plan(&state, blend_cell)
                            .map_err(|r| refuse(super::reason::DrawReason::BlendDevice(r)))
                    });
                match attempt {
                    Ok(plan) => blend_plans.push(plan),
                    Err(err) => {
                        self.pipelines.insert_negative(key.clone(), err.clone());
                        return Err(err);
                    }
                }
            }
            if let Err(r) = reims_vgpu_vulkan::blend::independent(&blend_plans, blend_cell) {
                let err = refuse(super::reason::DrawReason::BlendDevice(r));
                self.pipelines.insert_negative(key.clone(), err.clone());
                return Err(err);
            }
        }

        // Every colour attachment, parsed once and planned once.
        //
        // Built here rather than beside the create-info below because both
        // things that can go wrong are refusals of the whole pipeline, and a
        // refusal belongs with the other capability checks in this run: typed
        // decline, cached negatively so a replay returns this exact reason.
        //
        // Every attachment participates, not just slot 0. The secondaries
        // carry their own decoded blend, and both questions are asked of the
        // set: a pipeline is invalid if *any* attachment names a `SRC1_*`
        // factor without `dualSrcBlend`, and invalid if they *disagree*
        // without `independentBlend` — which no single attachment can be
        // blamed for, which is why the second check takes the list.
        let blend_cell = reims_vgpu_vulkan::blend::BlendCell {
            dual_source: ctx.features.dual_src_blend,
            independent: ctx.features.independent_blend,
        };
        let mut blend_plans = Vec::with_capacity(1 + key.pass.secondary_count());
        {
            let refuse = |reason: super::reason::DrawReason| {
                crate::observe::Emit::decline("vk_engine_pipeline", &reason).fail();
                DrawError::Unsupported(reason)
            };
            for slot in 0..=key.pass.secondary_count() {
                let blend = if slot == 0 {
                    key.blend
                } else {
                    key.secondary_blend[slot - 1]
                };
                let attempt = color_attachment_state(blend, key.color_write_mask[slot])
                    .map_err(|r| refuse(super::reason::DrawReason::BlendDeclaration(r)))
                    .and_then(|state| {
                        reims_vgpu_vulkan::blend::plan(&state, blend_cell)
                            .map_err(|r| refuse(super::reason::DrawReason::BlendDevice(r)))
                    });
                match attempt {
                    Ok(plan) => blend_plans.push(plan),
                    Err(err) => {
                        self.pipelines.insert_negative(key.clone(), err.clone());
                        return Err(err);
                    }
                }
            }
            if let Err(r) = reims_vgpu_vulkan::blend::independent(&blend_plans, blend_cell) {
                let err = refuse(super::reason::DrawReason::BlendDevice(r));
                self.pipelines.insert_negative(key.clone(), err.clone());
                return Err(err);
            }
        }

        // Resolve every attribute against what this device accepts as a vertex
        // buffer format. Vulkan makes the three-component 8/16-bit formats
        // optional, so the format the guest decoded is not automatically
        // bindable; `translate::support` either confirms it, substitutes the
        // mandatory wider sibling, or declines by name.
        //
        // A substitution is only invisible to a shader that does not read the
        // component it oversupplies, so `resolve` asks what this shader
        // declares at the attribute's location. Walked at most once per
        // pipeline miss and only when some attribute really needs substituting:
        // on a host that accepts every format — every host this project has run
        // on — `vert_spirv` is never read at all.
        //
        // Copied out of the intern table rather than borrowed from it, because
        // every refusal below takes `&mut self` to record a negative entry. One
        // allocation on the *miss* path, which is the path that is about to
        // compile a pipeline.
        let attrs: Vec<AttrKey> = self.attr_sets.get(key.attrs.0).to_vec();
        let mut shader_inputs: Option<VertexInputWidths> = None;
        let mut attribute_formats = Vec::with_capacity(attrs.len());
        let mut binding_rates = Vec::with_capacity(attrs.len());
        let mut binding_divisors: Vec<Option<vk::VertexInputBindingDivisorDescriptionKHR>> =
            Vec::with_capacity(attrs.len());
        for attr in &attrs {
            let binding = match translate::support::resolve(
                ctx.vertex_formats,
                attr.format,
                attr.offset,
                attr.stride,
                || {
                    shader_inputs
                        .get_or_insert_with(|| VertexInputWidths::from_spirv(vert_spirv))
                        .at(attr.location)
                },
            ) {
                Ok(binding) => binding,
                Err(translate_reason) => {
                    let err = DrawError::Unsupported(super::reason::DrawReason::VertexFormat(
                        translate_reason,
                    ));
                    crate::observe::Emit::decline("vk_engine_vertex_format", &translate_reason)
                        .fail_once(
                            (u64::from(attr.location) << 32) | u64::from(translate_reason.value()),
                        );
                    self.pipelines.insert_negative(key.clone(), err.clone());
                    return Err(err);
                }
            };
            if let Some(narrow) = binding.widened_from {
                // Fail-visible because a widened attribute is a device-specific
                // difference from what the guest asked for, even though
                // `resolve` has just established that no shader input can
                // observe it: without this line a substitution is invisible in
                // a bug report from a host nobody here owns.
                let decline = VertexFormatWidenDecline {
                    from: narrow,
                    to: binding.format,
                    location: attr.location,
                    offset: attr.offset,
                    stride: attr.stride,
                };
                crate::observe::Emit::decline("vk_engine_vertex_format", &decline).fail_once(
                    (u64::from(attr.location) << 32) | u64::from(narrow.as_raw() as u32),
                );
            }
            attribute_formats.push(binding.format);
            // Vulkan has `VERTEX` and `INSTANCE` and nothing else, so the two
            // tessellation step functions have no rate here at all. They are
            // declined before a request reaches the engine — the translation
            // layer refuses them by name — and the arm is a refusal rather
            // than a panic because "unreachable" is a claim about a call site
            // and this one is reached from two.
            let divisor = match attr.step_function {
                VertexStepFunction::Constant => Some(0),
                VertexStepFunction::PerVertex => None,
                VertexStepFunction::PerInstance if attr.step_rate == 1 => None,
                VertexStepFunction::PerInstance => Some(attr.step_rate),
                step
                @ (VertexStepFunction::PerPatch | VertexStepFunction::PerPatchControlPoint) => {
                    let reason = super::reason::DrawReason::VertexStep(
                        reims_vgpu_vulkan::vertex::Refusal::TessellationStep { step },
                    );
                    crate::observe::Emit::decline("vk_engine_pipeline", &reason).fail();
                    let err = DrawError::Unsupported(reason);
                    self.pipelines.insert_negative(key.clone(), err.clone());
                    return Err(err);
                }
            };
            binding_divisors.push(divisor.map(|divisor| {
                vk::VertexInputBindingDivisorDescriptionKHR::default()
                    .binding(attr.binding)
                    .divisor(divisor)
            }));
            // The rate the binding is created with, from the layer that owns
            // which step functions have one.
            binding_rates.push(
                reims_vgpu_vulkan::vertex::input_rate(attr.step_function)
                    .unwrap_or(vk::VertexInputRate::VERTEX),
            );
            if divisor == Some(0) && !ctx.vertex_divisor.zero_divisor {
                let err =
                    DrawError::Unsupported(super::reason::DrawReason::ConstantVertexAttribute);
                self.pipelines.insert_negative(key.clone(), err.clone());
                return Err(err);
            }
            if divisor.is_some_and(|v| v > 1) {
                if !ctx.vertex_divisor.instance_rate_divisor {
                    let err = DrawError::Unsupported(
                        super::reason::DrawReason::InstanceRateDivisorUnsupported {
                            step_rate: attr.step_rate,
                        },
                    );
                    self.pipelines.insert_negative(key.clone(), err.clone());
                    return Err(err);
                }
                if attr.step_rate > ctx.vertex_divisor.max_divisor {
                    let err = DrawError::Unsupported(
                        super::reason::DrawReason::InstanceRateDivisorOverLimit {
                            step_rate: attr.step_rate,
                            limit: ctx.vertex_divisor.max_divisor,
                        },
                    );
                    self.pipelines.insert_negative(key.clone(), err.clone());
                    return Err(err);
                }
            }
        }

        let main_c = super::context::main_entry();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&main_c),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&main_c),
        ];
        // Both lists were rebuilt here from the attributes with the step function
        // matched a second and a third time. They are built once, in the loop
        // that already refused every step function without a rate, so the two
        // spellings cannot answer differently for one attribute.
        let vertex_binding_descs: Vec<_> = attrs
            .iter()
            .zip(&binding_rates)
            .map(|(attribute, rate)| {
                vk::VertexInputBindingDescription::default()
                    .binding(attribute.binding)
                    .stride(attribute.stride)
                    .input_rate(*rate)
            })
            .collect();
        // A divisor of one is what Vulkan already does, so declaring it would
        // pull in the extension structure for nothing.
        let vertex_binding_divisors: Vec<_> = binding_divisors.iter().flatten().copied().collect();
        let vertex_attribute_descs: Vec<_> = attrs
            .iter()
            .zip(&attribute_formats)
            .map(|(attribute, format)| {
                vk::VertexInputAttributeDescription::default()
                    .location(attribute.location)
                    .binding(attribute.binding)
                    // The device-resolved format, which equals the attribute's
                    // own on every host seen so far and its mandatory wider
                    // sibling where the device declined the optional one.
                    .format(*format)
                    .offset(attribute.offset)
            })
            .collect();
        let mut vertex_divisor_state = vk::PipelineVertexInputDivisorStateCreateInfoKHR::default()
            .vertex_binding_divisors(&vertex_binding_divisors);
        let mut vtx_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attribute_descs);
        if !vertex_binding_divisors.is_empty() {
            vtx_input = vtx_input.push_next(&mut vertex_divisor_state);
        }
        // Derived from the key, never from the guest's primitive type: the
        // key is what two draws share, so the declared topology has to be a
        // function of it or one cache entry would describe two pipelines. On
        // the baseline rung the key *is* the guest's type and this is the
        // topology it always was.
        let input_asm_plan = key.topology.input_assembly();
        let input_asm = input_asm_plan.native();
        // Dynamic viewport/scissor so L5 key need not include extent (flip flag is static).
        // Stencil reference is dynamic (Metal's `SetStencilReferenceValue` is a
        // command distinct from the state object) so distinct references reuse
        // one pipeline; only listed for stencil pipelines.
        // The blend colour is dynamic on every graphics pipeline this cache
        // builds, whether or not any attachment names a constant factor.
        // Metal sets it on the encoder, so it changes without the pipeline
        // changing: baking it in would key this cache on a value that is not
        // part of a pipeline, and a guest animating a fade would compile one
        // per frame. Unconditional rather than keyed on whether a factor reads
        // it, because a key dimension that only decides which dynamic states
        // are declared is a second way to spell the same pipeline.
        let mut dynamic_states = vec![
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
        ];
        // `DEPTH_BIAS`, and whichever rasterization members this host supplies
        // per draw — the cull mode and winding under
        // `VK_EXT_extended_dynamic_state`, the fill mode and depth-clip mode
        // under `…_state3`, and none of them on a host with neither. Metal has
        // no way to say "this pipeline cannot be biased", so the pipeline
        // always enables biasing and always takes the three values
        // dynamically; see `reims_vgpu_vulkan::raster`.
        //
        // Read off the key rather than recomputed, because the key *is* the
        // plan's baked half: the states listed here and the placeholders in
        // `key.raster` are two readings of one `RasterDynamic`, and a second
        // derivation could disagree with it.
        dynamic_states.extend(key.raster.dynamic.states());
        // `PRIMITIVE_TOPOLOGY` where the input assembly above declared a
        // stand-in for a class rather than the guest's own type. Taken from
        // the same plan that chose the stand-in: a pipeline that declares one
        // without declaring the state rasterizes the stand-in, and on a host
        // with no validation layers nothing says so.
        dynamic_states.extend_from_slice(input_asm_plan.states());
        // The stencil reference, and on a host that supplies the whole
        // `MTLDepthStencilState` per draw the eight states that go with it.
        // Read off the key's own state rather than re-derived from the host,
        // for the reason above: the placeholder and the list that replaces it
        // are two readings of one decision.
        dynamic_states.extend_from_slice(key.depth_stencil.states());
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        // Both counts are the key's one number: the viewports and scissors
        // themselves are dynamic, but how many of them there are is not, and
        // Vulkan requires the two counts to be equal.
        let vp_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(key.viewport_slots)
            .scissor_count(key.viewport_slots);
        let raster = key.raster.native();
        // `rasterSampleCount` is a property of `MTLRenderPipelineDescriptor`,
        // so it reaches this device inside the serializer-object pipeline's own
        // compact-TLV block. The pass key carries that decoded count, and the
        // render-pass attachment is created with the same value; unsupported
        // count/attachment combinations are refused before either object is
        // built.
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk_sample_count(key.pass.0.sample_count));
        // One blend attachment state per color attachment; Vulkan requires the
        // count to match the render pass. Every slot uses its own decoded
        // blend, slot 0 from `key.blend` and slot n from
        // `key.secondary_blend[n-1]`.
        //
        // The secondaries used to be forced unblended here, justified by a
        // comment saying the decode side did not carry per-attachment blend
        // state. It did, and had all along — the Metal arm reads exactly these
        // fields per slot. Only this key collapsed them, so a guest MRT
        // pipeline that asked to blend slot 1 silently got a raw store.
        //
        // The colour write mask comes from the guest too, and it is applied on
        // both arms because `MTLColorWriteMask` is independent of
        // `blendingEnabled` — an unblended masked attachment still leaves its
        // unwritten channels alone. Metal's bits are alpha-first and Vulkan's
        // are red-first, so the exchange is a reordering rather than a cast;
        // `reims_vgpu_vulkan::blend` performs it, above.
        let blend_att: Vec<vk::PipelineColorBlendAttachmentState> =
            blend_plans.iter().map(|p| p.native()).collect();
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_att);
        // Depth-stencil state: attached ONLY when the pass carries a depth
        // attachment (Vulkan requires the pipeline's depth-stencil state to be
        // consistent with the subpass). Without it the color-only pipeline is
        // byte-identical to the pre-depth engine. The reference field is left 0
        // here and supplied dynamically per draw, and on a dynamic host so is
        // everything else — see the key's own doc.
        let depth_stencil = key.depth_stencil.native();
        let mut gpci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vtx_input)
            .input_assembly_state(&input_asm)
            .viewport_state(&vp_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);
        if key.feedback_colors != 0 {
            gpci = gpci.flags(vk::PipelineCreateFlags::COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT);
        }
        if key.pass.has_depth() {
            gpci = gpci.depth_stencil_state(&depth_stencil);
        }
        // The third call that compiles a module this device assembled, and the
        // only one that compiles two at once. A macOS 15 guest's CoreAnimation
        // uber fragment shader has been observed keeping NVIDIA's compiler in
        // here for over ten minutes with the device lock held; see
        // `crate::observe::driver_watch`, which this arming also starts.
        let breadcrumb = match super::driver_breadcrumb::DriverBreadcrumb::arm(
            &format!(
                "create_graphics_pipelines vert_words={} frag_words={}",
                vert_spirv.len(),
                frag_spirv.len()
            ),
            &[("vert", vert_spirv), ("frag", frag_spirv)],
        ) {
            Ok(breadcrumb) => breadcrumb,
            Err(hit) => {
                let err = self.note_quarantined("create_graphics_pipelines", &hit);
                self.pipelines.insert_negative(key.clone(), err.clone());
                return Err(err);
            }
        };
        let created = ctx
            .device
            .create_graphics_pipelines(ctx.pipeline_cache, &[gpci], None);
        breadcrumb.disarm();
        let pipe = created.map_err(|(_, e)| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateGraphicsPipelines, e));
            self.pipelines.insert_negative(key.clone(), err.clone());
            err
        })?[0];
        counters.note_create(CreateSite::GraphicsPipeline);
        // A fresh pipeline compile grew the VkPipelineCache — persist it so
        // the next boot warm-starts (file write is off-thread, debounced).
        ctx.persist_pipeline_cache();
        if let Some(old) = self.pipelines.insert(key.clone(), pipe) {
            pools.dispose(&ctx.device, DeferredHandle::Pipeline(old));
        }
        if let Some(identity) = pipeline_object {
            self.pipeline_objects.remember(identity, key, pipe);
        }
        Ok(pipe)
    }

    pub(crate) unsafe fn get_or_create_compute_pipeline(
        &mut self,
        ctx: &DeviceContext,
        key: &ComputePipelineKey,
        shader: ShaderModuleSource<'_>,
        pipeline_layout: vk::PipelineLayout,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<vk::Pipeline, DrawError> {
        if let Some(err) = self.compute_pipelines.get_negative(key) {
            counters
                .compute_pipeline_misses
                .fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(p) = self.compute_pipelines.get(key) {
            counters
                .compute_pipeline_hits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(p);
        }
        counters
            .compute_pipeline_misses
            .fetch_add(1, Ordering::Relaxed);
        let entry_c = std::ffi::CString::new(key.entry.as_str()).map_err(|_| {
            DrawError::ComputeValidation(
                super::compute_validation::ComputeValidationDecline::EntryInteriorNul,
            )
        })?;
        // The translator decorates its three local-size specialization
        // constants with `KERNEL_LOCAL_SIZE_SPEC_IDS`, so the map entries are
        // its ids and the data is the three `u32` in that order.
        let spec_data: Vec<u8> = key
            .local_size
            .iter()
            .flat_map(|size| size.iter().flat_map(|value| value.to_ne_bytes()))
            .collect();
        let spec_entries: Vec<vk::SpecializationMapEntry> = key
            .local_size
            .iter()
            .flat_map(|_| {
                metal2vulkan::reflect::KERNEL_LOCAL_SIZE_SPEC_IDS
                    .into_iter()
                    .enumerate()
                    .map(|(dimension, id)| {
                        vk::SpecializationMapEntry::default()
                            .constant_id(id)
                            .offset((dimension * std::mem::size_of::<u32>()) as u32)
                            .size(std::mem::size_of::<u32>())
                    })
            })
            .collect();
        let spec_info = vk::SpecializationInfo::default()
            .map_entries(&spec_entries)
            .data(&spec_data);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&entry_c);
        if key.local_size.is_some() {
            stage = stage.specialization_info(&spec_info);
        }
        let cpci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        // The other call that compiles the module, and the one an NVIDIA driver
        // has been observed dying inside on a macos-14 guest's first dispatch.
        let breadcrumb = match super::driver_breadcrumb::DriverBreadcrumb::arm(
            &format!("create_compute_pipelines entry={}", key.entry),
            &[("kernel", shader.spirv)],
        ) {
            Ok(breadcrumb) => breadcrumb,
            Err(hit) => {
                let err = self.note_quarantined("create_compute_pipelines", &hit);
                self.compute_pipelines
                    .insert_negative(key.clone(), err.clone());
                return Err(err);
            }
        };
        let created = ctx
            .device
            .create_compute_pipelines(ctx.pipeline_cache, &[cpci], None);
        breadcrumb.disarm();
        let pipe = created.map_err(|(_, e)| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateComputePipelines, e));
            self.compute_pipelines
                .insert_negative(key.clone(), err.clone());
            err
        })?[0];
        counters.note_create(CreateSite::ComputePipeline);
        // Same warm-start persistence as the graphics path.
        ctx.persist_pipeline_cache();
        if let Some(old) = self.compute_pipelines.insert(key.clone(), pipe) {
            pools.dispose(&ctx.device, DeferredHandle::Pipeline(old));
        }
        Ok(pipe)
    }
}

#[cfg(test)]
mod color0_load_tests {
    use super::*;
    use reims_vgpu_protocol::pass_action::LoadAction;

    /// A pass that promises its prior contents and arrives with none of them
    /// must not write the attachment.
    ///
    /// This is the external invariant the black rectangle behind a closing dock
    /// menu violates. The guest declares `MTLLoadActionDontCare`, redraws only
    /// its damage rectangle, and expects everything it did not cover to still
    /// be there — which `LoadAction::preserves_prior_contents` says is a lawful
    /// reading, because undefined permits the prior contents. Resolving that to
    /// `CLEAR` writes the whole attachment with a colour the guest never
    /// supplied, so the test is about the *load op*, not about the colour: a
    /// clear is disqualified whatever value it would have used.
    #[test]
    fn a_preserving_pass_with_no_prior_contents_does_not_write_the_attachment() {
        let resting = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let (op, initial) = Color0Load::Undefined.ops(resting);
        assert_ne!(
            op,
            vk::AttachmentLoadOp::CLEAR,
            "an unseeded preserving pass may not invent a colour for the texels \
             the guest did not draw"
        );
        assert_eq!(op, vk::AttachmentLoadOp::DONT_CARE);
        // `UNDEFINED`, and not the resting layout: this arm runs on the first
        // pass into a freshly created attachment, which really is in
        // `UNDEFINED`, so claiming the resting layout would describe a
        // transition that never happened.
        assert_eq!(
            initial,
            vk::ImageLayout::UNDEFINED,
            "an attachment that may never have been transitioned may not be \
             declared to be in the resting layout"
        );
    }

    /// The two answers that DO write, and the one that does not, stay apart.
    #[test]
    fn each_slot0_load_keeps_its_own_answer() {
        let resting = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        assert_eq!(
            Color0Load::Preserve.ops(resting),
            (vk::AttachmentLoadOp::LOAD, resting)
        );
        assert_eq!(
            Color0Load::Clear.ops(resting),
            (vk::AttachmentLoadOp::CLEAR, vk::ImageLayout::UNDEFINED)
        );
        // "writes the whole attachment" is exactly "the load op is CLEAR";
        // there is no second spelling of it to drift from.
        for (load, writes) in [
            (Color0Load::Preserve, false),
            (Color0Load::Clear, true),
            (Color0Load::Undefined, false),
        ] {
            assert_eq!(
                load.ops(resting).0 == vk::AttachmentLoadOp::CLEAR,
                writes,
                "{load:?} must{} write the whole attachment",
                if writes { "" } else { " not" }
            );
        }
    }

    /// A pass key partitions on the load, so the three answers cannot share a
    /// cached render pass.
    ///
    /// Without this a clear and an undefined begin would collide in
    /// `ObjectCaches` and the second one served would begin with the first
    /// one's load op — which is the same defect this change removes, moved one
    /// layer down.
    #[test]
    fn the_three_slot0_loads_are_three_pass_keys() {
        let keys = [
            PassKey::single(Color0Load::Preserve, vk::Format::B8G8R8A8_UNORM),
            PassKey::single(Color0Load::Clear, vk::Format::B8G8R8A8_UNORM),
            PassKey::single(Color0Load::Undefined, vk::Format::B8G8R8A8_UNORM),
        ];
        for (i, a) in keys.iter().enumerate() {
            for (j, b) in keys.iter().enumerate() {
                assert_eq!(i == j, a == b, "slot-0 load must partition the pass cache");
            }
        }
    }

    /// Every declared action this device can receive lands on an answer that
    /// does not invent a colour, unless the guest asked for one.
    ///
    /// Swept over the ordinal space rather than the three names, for the reason
    /// `every_preserving_load_action_with_no_prior_contents_shares_one_bucket`
    /// gives: a fourth ordinal arriving from the wire must not be able to
    /// acquire the clearing reading by falling through a catch-all.
    #[test]
    fn only_a_declared_clear_may_invent_a_colour() {
        for raw in (0u16..=8).chain([u16::MAX]) {
            let declared = LoadAction::from_declared(raw);
            let unseeded = if declared.preserves_prior_contents() {
                Color0Load::Undefined
            } else {
                Color0Load::Clear
            };
            assert_eq!(
                unseeded.ops(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL).0
                    == vk::AttachmentLoadOp::CLEAR,
                !declared.preserves_prior_contents(),
                "ordinal {raw} declared {declared:?}: only a clear may write the \
                 whole attachment when no prior contents were found"
            );
        }
    }
}

#[cfg(test)]
mod object_cache_tests {
    use super::*;

    #[derive(Clone)]
    struct CountingKey {
        value: u32,
        hashes: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl PartialEq for CountingKey {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Eq for CountingKey {}

    impl std::hash::Hash for CountingKey {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.hashes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.value.hash(state);
        }
    }

    #[test]
    fn an_empty_negative_cache_does_not_hash_the_object_key() {
        let hashes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let key = CountingKey {
            value: 7,
            hashes: std::sync::Arc::clone(&hashes),
        };
        let mut cache: ObjectCache<CountingKey, u32> = ObjectCache::new();

        assert_eq!(cache.get_negative(&key), None);
        assert_eq!(hashes.load(std::sync::atomic::Ordering::Relaxed), 0);

        cache.insert_negative(
            CountingKey {
                value: 9,
                hashes: std::sync::Arc::clone(&hashes),
            },
            DrawError::VkCall(VkCall::new(
                VkOp::CachesCreateShaderModule,
                vk::Result::ERROR_UNKNOWN,
            )),
        );
        hashes.store(0, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(cache.get_negative(&key), None);
        assert_eq!(hashes.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn push_layout_selection_uses_the_device_limit_and_keeps_the_fallback() {
        let caps = crate::backend::vulkan::caps::PushDescriptorCaps {
            max_descriptors: 32,
        };
        let bindings = |counts: &[u32]| -> Vec<BindingSig> {
            counts
                .iter()
                .enumerate()
                .map(|(binding, &count)| BindingSig {
                    binding: binding as u32,
                    ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
                    stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
                    count,
                })
                .collect()
        };
        assert!(layout_uses_push_descriptors(&bindings(&[16, 16]), caps));
        assert!(!layout_uses_push_descriptors(&bindings(&[16, 17]), caps));
        assert!(!layout_uses_push_descriptors(&bindings(&[]), caps));
    }

    /// Pipeline and live-pass compatibility exclude load actions but retain
    /// attachment formats and subpass shape.
    #[test]
    fn pass_compatibility_ignores_only_load_actions() {
        let mut clear = PassKey::single(Color0Load::Clear, vk::Format::B8G8R8A8_UNORM);
        clear.secondary_count = 1;
        clear.secondary[0] = SecondaryAttachKey {
            format: vk::Format::R16G16_SFLOAT,
            load: false,
        };
        clear.depth = Some(DepthAttachKey {
            load: false,
            stencil: true,
        });

        let mut load = clear;
        load.color0_load = Color0Load::Preserve;
        load.secondary[0].load = true;
        load.depth.as_mut().unwrap().load = true;
        assert_eq!(clear.compatibility(), load.compatibility());

        let mut different_format = load;
        different_format.secondary[0].format = vk::Format::R32_SFLOAT;
        assert_ne!(clear.compatibility(), different_format.compatibility());

        let mut different_subpass = load;
        different_subpass.color_input = true;
        assert_ne!(clear.compatibility(), different_subpass.compatibility());

        let mut different_depth = load;
        different_depth.depth.as_mut().unwrap().stencil = false;
        assert_ne!(clear.compatibility(), different_depth.compatibility());
    }

    /// `first_difference` answers `None` on exactly the pairs that compare
    /// equal, which is what makes `passcompat_*` a partition of
    /// `passdiff_compat` rather than a second opinion about it.
    ///
    /// This is the property that matters, and it is not the same as "every
    /// field has a variant": a field the destructure names but the body forgets
    /// to compare would still let two unequal keys answer `None`, and the caller
    /// would then charge the *next* echo field — reporting a framebuffer change
    /// where an attachment shape moved. So the assertion is over mutations, one
    /// per field, and it is made in both directions.
    #[test]
    fn every_compatibility_difference_is_named_and_equal_keys_name_none() {
        let mut base = PassKey::single(Color0Load::Clear, vk::Format::B8G8R8A8_UNORM);
        base.secondary_count = 1;
        base.secondary[0] = SecondaryAttachKey {
            format: vk::Format::R16G16_SFLOAT,
            load: false,
        };
        base.depth = Some(DepthAttachKey {
            load: false,
            stencil: true,
        });
        base.sample_count = 1;

        assert_eq!(
            base.compatibility().first_difference(base.compatibility()),
            None
        );

        // A load action is erased by `compatibility`, so it is not a difference
        // — the one mutation below that must answer `None`.
        /// One named mutation of a [`PassKey`] and the difference it must
        /// produce. `None` for a mutation `compatibility` erases.
        type Mutation = (&'static str, fn(&mut PassKey), Option<PassCompatField>);
        let mutations: &[Mutation] = &[
            (
                "load actions",
                |k| {
                    k.color0_load = Color0Load::Preserve;
                    k.secondary[0].load = true;
                    k.depth.as_mut().unwrap().load = true;
                },
                None,
            ),
            (
                "color0 format",
                |k| k.color0_format = vk::Format::R8G8B8A8_UNORM,
                Some(PassCompatField::Color0Format),
            ),
            (
                "secondary count",
                |k| k.secondary_count = 2,
                Some(PassCompatField::SecondaryCount),
            ),
            (
                "secondary format",
                |k| k.secondary[0].format = vk::Format::R32_SFLOAT,
                Some(PassCompatField::SecondaryFormat),
            ),
            ("depth", |k| k.depth = None, Some(PassCompatField::Depth)),
            (
                "host accessible",
                |k| k.host_accessible_color0 = true,
                Some(PassCompatField::HostAccessibleColor0),
            ),
            (
                "color input",
                |k| k.color_input = true,
                Some(PassCompatField::ColorInput),
            ),
            // Feedback is a property of the draw, and it makes two passes
            // incompatible only on the arm where it still moves a layout. On the
            // shipping arm `compatibility` erases it, which is what stops a
            // feedback draw closing the render pass an ordinary one opened.
            (
                "feedback",
                |k| k.feedback_colors = 1,
                if color_feedback_layout() == color0_pass_exit_layout() {
                    None
                } else {
                    Some(PassCompatField::FeedbackColors)
                },
            ),
            (
                "sample count",
                |k| k.sample_count = 4,
                Some(PassCompatField::SampleCount),
            ),
            (
                "resolve",
                |k| k.multisample_resolve = true,
                Some(PassCompatField::MultisampleResolve),
            ),
        ];

        for (name, mutate, expected) in mutations {
            let mut moved = base;
            mutate(&mut moved);
            let (a, b) = (base.compatibility(), moved.compatibility());
            assert_eq!(a.first_difference(b), *expected, "{name}");
            assert_eq!(b.first_difference(a), *expected, "{name}, reversed");
            // The partition itself: `None` iff equal, on every input above.
            assert_eq!(
                a.first_difference(b).is_none(),
                a == b,
                "{name}: a named difference and key equality must agree"
            );
        }
    }

    /// Framebuffers bind attachment views, not load actions, layouts, or
    /// dependencies. A transport change on the same retained attachment must
    /// therefore keep the framebuffer, while a changed input-attachment shape
    /// must not.
    #[test]
    fn framebuffer_compatibility_ignores_transport_and_dependency_state() {
        let plain = PassKey::single(Color0Load::Clear, vk::Format::B8G8R8A8_UNORM);
        let mut transported = plain;
        transported.color0_load = Color0Load::Preserve;
        transported.host_accessible_color0 = true;
        transported.feedback_colors = 1;

        assert_eq!(
            plain.framebuffer_compatibility(),
            transported.framebuffer_compatibility()
        );
        assert_ne!(plain.compatibility(), transported.compatibility());

        let mut input = transported;
        input.color_input = true;
        assert_ne!(
            plain.framebuffer_compatibility(),
            input.framebuffer_compatibility()
        );
    }

    /// A `LayoutId` is an identity and not a fingerprint, and this is the claim
    /// that says so: the table is forced to bucket two *different* layouts
    /// together, and it still tells them apart.
    ///
    /// Buckets are keyed by a digest of the bindings, which is why the question
    /// exists at all. A cache that stopped at the digest would return the first
    /// entry in the bucket — a `VkPipelineLayout` built for someone else's
    /// descriptors — and the draw would bind against a layout it does not match.
    /// Nothing downstream could notice: the handles are opaque and the draw
    /// records normally. Retaining the bindings and comparing them is what makes
    /// that unreachable, and this test reaches into `buckets` to prove the
    /// comparison is load-bearing rather than waiting for a natural collision.
    #[test]
    fn two_layouts_sharing_a_digest_bucket_keep_separate_identities() {
        use ash::vk::Handle as _;
        let sig = |binding, count| BindingSig {
            binding,
            ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count,
        };
        let first = vec![sig(0, 1)];
        let second = vec![sig(1, 2), sig(2, 3)];

        let mut table = LayoutTable::new();
        let a = table.insert(
            &first,
            None,
            vk::DescriptorSetLayout::from_raw(0x11),
            vk::PipelineLayout::from_raw(0xaa),
            true,
        );
        let b = table.insert(
            &second,
            Some((0, 16)),
            vk::DescriptorSetLayout::from_raw(0x22),
            vk::PipelineLayout::from_raw(0xbb),
            false,
        );
        assert_ne!(a.id, b.id);

        // Force the collision the digest is allowed to have and the table is
        // not: both ids in one bucket, under both digests.
        let keys: Vec<Digest128> = table.buckets.keys().copied().collect();
        assert_eq!(keys.len(), 2, "the two layouts digested apart on their own");
        for key in keys {
            table.buckets.insert(key, vec![a.id.0, b.id.0]);
        }

        table.front = None;
        let found_first = table.get(&first, None).expect("first layout still found");
        assert_eq!(found_first.id, a.id);
        assert_eq!(
            found_first.pipeline_layout,
            vk::PipelineLayout::from_raw(0xaa)
        );
        assert!(found_first.push_descriptors);

        table.front = None;
        let found_second = table
            .get(&second, Some((0, 16)))
            .expect("second layout still found");
        assert_eq!(found_second.id, b.id);
        assert_eq!(
            found_second.pipeline_layout,
            vk::PipelineLayout::from_raw(0xbb)
        );
        assert!(!found_second.push_descriptors);

        // The push range is part of the identity, not decoration: the same
        // bindings under a different range are a different pipeline layout.
        table.front = None;
        assert!(table.get(&second, None).is_none());
    }

    /// The front index answers without touching a bucket, and it must answer
    /// about the layout that was actually asked for. A front that matched on
    /// anything less than the full bindings would hand every draw in a run the
    /// first layout of that run.
    #[test]
    fn the_layout_front_does_not_answer_for_a_different_binding_set() {
        use ash::vk::Handle as _;
        let sig = |binding| BindingSig {
            binding,
            ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count: 1,
        };
        let mut table = LayoutTable::new();
        let a = table.insert(
            &[sig(0)],
            None,
            vk::DescriptorSetLayout::null(),
            vk::PipelineLayout::from_raw(0xaa),
            false,
        );
        assert_eq!(table.front, Some(a.id.0), "insert primes the front");
        assert!(
            table.get(&[sig(1)], None).is_none(),
            "the front answered for a binding set it does not hold"
        );
        assert_eq!(table.get(&[sig(0)], None).map(|found| found.id), Some(a.id));
    }

    /// A create failure is replayed for the bindings that produced it and for no
    /// others, and a refusal about how full the device is right now is not
    /// remembered at all — the rule `ObjectCache::insert_negative` states, which
    /// the layout table had to restate because it is not an `ObjectCache`.
    #[test]
    fn a_layout_refusal_is_replayed_only_for_its_own_bindings() {
        use ash::vk::Handle as _;
        let sig = |binding| BindingSig {
            binding,
            ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count: 1,
        };
        let mut table = LayoutTable::new();
        table.insert_negative(
            &[sig(0)],
            None,
            DrawError::VkCall(VkCall::new(
                VkOp::CachesCreatePipelineLayout,
                vk::Result::ERROR_UNKNOWN,
            )),
        );
        assert!(table.get_negative(&[sig(0)], None).is_some());
        assert!(table.get_negative(&[sig(1)], None).is_none());
        assert!(table.get_negative(&[sig(0)], Some((0, 4))).is_none());

        table.insert_negative(
            &[sig(2)],
            None,
            DrawError::VkCall(VkCall::new(
                VkOp::CachesCreatePipelineLayout,
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            )),
        );
        assert!(
            table.get_negative(&[sig(2)], None).is_none(),
            "out of memory describes this instant, not the request"
        );

        // A later success clears the refusal, so a device that recovers is not
        // answered from a memory of when it had not.
        table.insert(
            &[sig(0)],
            None,
            vk::DescriptorSetLayout::null(),
            vk::PipelineLayout::from_raw(0xaa),
            false,
        );
        assert!(table.get_negative(&[sig(0)], None).is_none());
    }

    #[test]
    fn layout_bindings_coalesce_array_elements_and_refuse_conflicting_shapes() {
        let sig = |count| BindingSig {
            binding: 32,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count,
        };
        let mut duplicated = vec![sig(8), sig(8)];
        assert_eq!(canonicalize_layout_bindings(&mut duplicated), Ok(()));
        assert_eq!(duplicated, vec![sig(8)]);
        let mut conflicting = vec![sig(8), sig(4)];
        assert!(matches!(
            canonicalize_layout_bindings(&mut conflicting),
            Err(super::super::DrawError::Unsupported(
                super::super::reason::DrawReason::DescriptorBindingConflict {
                    binding: 32,
                    first_count: 8,
                    second_count: 4,
                    ..
                }
            ))
        ));
    }

    fn unnormalized_key() -> SamplerStateKey {
        use reims_vgpu_core::sampler as mtl;
        SamplerStateKey {
            min_filter: mtl::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
            mag_filter: mtl::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mip_filter: mtl::MTL_SAMPLER_MIP_FILTER_LINEAR,
            address_mode_u: mtl::MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            address_mode_v: mtl::MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT,
            address_mode_w: mtl::MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            border_color: mtl::MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
            compare_function: super::super::types::SamplerCompareFunction::Never,
            lod_min: 1.0f32.to_bits(),
            lod_max: 8.0f32.to_bits(),
            max_anisotropy: 16,
            unnormalized_coordinates: true,
        }
    }

    /// Every constraint `vkCreateSampler` puts on `unnormalizedCoordinates`, on
    /// a key that violates all of them at once.
    ///
    /// This site owns neither the conformance nor the Vulkan spelling any more
    /// — `reims_vgpu_core::sampler` conforms the declaration and
    /// `reims_vgpu_vulkan::sampler` plans it, and each has its own tests. What
    /// it owns is the *join*: that the key this cache is indexed by reaches
    /// those two as the declaration the guest actually wrote. A projection that
    /// silently swapped two axes would pass every test on either side of it.
    #[test]
    fn the_key_reaches_the_owning_layers_as_a_plannable_sampler() {
        let plan = reims_vgpu_vulkan::sampler::plan(
            sampler_shape(&unnormalized_key())
                .checked()
                .expect("an unnormalized declaration is conformed, not refused"),
            reims_vgpu_vulkan::sampler::SamplerCell {
                mirror_clamp_to_edge: true,
                anisotropy: true,
                max_anisotropy: 16.0,
            },
        )
        .expect("a plannable sampler");
        assert!(plan.unnormalized_coordinates);
        // -01072, and it is magFilter that survives.
        assert_eq!(plan.min_filter, ash::vk::Filter::LINEAR);
        assert_eq!(plan.mag_filter, ash::vk::Filter::LINEAR);
        // -01073, -01074, -01076.
        assert_eq!(plan.mipmap_mode, ash::vk::SamplerMipmapMode::NEAREST);
        assert_eq!((plan.min_lod, plan.max_lod), (0.0, 0.0));
        assert!(!plan.anisotropy_enable);
        // -01075, U and V.
        assert_eq!(plan.address[0], ash::vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(plan.address[1], ash::vk::SamplerAddressMode::CLAMP_TO_EDGE);
    }

    /// The six blend ordinals reach the six fields of their own names.
    ///
    /// They are interchangeable `u32`s three-for-three, so a swap between the
    /// RGB and alpha halves produces a perfectly valid blend that composites
    /// the wrong channel set — no refusal, no log line. A test on either side
    /// of this projection alone cannot see it: the key would still hold what it
    /// was given and the shape would still parse. Distinct values in all six
    /// positions is the only arrangement that can.
    #[test]
    fn the_six_blend_ordinals_do_not_cross_on_the_way_to_the_owning_layer() {
        use reims_vgpu_core::blend::{
            BlendFactor, BlendOperation, MTL_BLEND_FACTOR_DESTINATION_ALPHA, MTL_BLEND_FACTOR_ONE,
            MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA, MTL_BLEND_FACTOR_SOURCE_ALPHA,
            MTL_BLEND_OPERATION_MAX, MTL_BLEND_OPERATION_REVERSE_SUBTRACT,
        };
        let key = BlendKey {
            src_rgb: MTL_BLEND_FACTOR_SOURCE_ALPHA,
            dst_rgb: MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
            op_rgb: MTL_BLEND_OPERATION_REVERSE_SUBTRACT,
            src_alpha: MTL_BLEND_FACTOR_ONE,
            dst_alpha: MTL_BLEND_FACTOR_DESTINATION_ALPHA,
            op_alpha: MTL_BLEND_OPERATION_MAX,
        };
        let blend = color_attachment_state(Some(key), ColorWriteMask::ALL)
            .expect("every ordinal is one the guest API declares")
            .blend()
            .expect("the key carried a blend");
        assert_eq!(blend.src_color, BlendFactor::SourceAlpha);
        assert_eq!(blend.dst_color, BlendFactor::OneMinusSourceAlpha);
        assert_eq!(blend.color_operation, BlendOperation::ReverseSubtract);
        assert_eq!(blend.src_alpha, BlendFactor::One);
        assert_eq!(blend.dst_alpha, BlendFactor::DestinationAlpha);
        assert_eq!(blend.alpha_operation, BlendOperation::Max);
    }

    /// `blendingEnabled` clear is the absence of a blend, and the mask is not
    /// part of it.
    ///
    /// Both halves matter. Parsing the six ordinals behind a clear flag would
    /// refuse a pipeline over a value that can never reach a pixel; dropping
    /// the mask with them would make an unblended attachment that writes only
    /// alpha write everything, which is a colour channel the guest asked to
    /// keep.
    #[test]
    fn an_unblended_attachment_keeps_its_mask_and_parses_no_equation() {
        let masked = color_attachment_state(
            None,
            ColorWriteMask::new(reims_vgpu_core::blend::MTL_COLOR_WRITE_MASK_ALPHA)
                .expect("in range"),
        )
        .expect("nothing to parse");
        assert!(masked.blend().is_none());
        assert!(!masked.write_mask().red());
        assert!(masked.write_mask().alpha());
    }

    /// An ordinal the guest API does not declare refuses the pipeline by name.
    ///
    /// It used to make the *slot* unblended and let the pipeline build, which
    /// is a compositing attachment silently becoming a raw store.
    #[test]
    fn an_ordinal_outside_the_guest_api_refuses_rather_than_unblending() {
        let refusal = color_attachment_state(
            Some(BlendKey {
                src_rgb: reims_vgpu_core::blend::MTL_BLEND_FACTOR_ONE,
                dst_rgb: 99,
                op_rgb: 0,
                src_alpha: 1,
                dst_alpha: 0,
                op_alpha: 0,
            }),
            ColorWriteMask::ALL,
        )
        .expect_err("99 is not an MTLBlendFactor");
        assert_eq!(
            refusal,
            reims_vgpu_core::blend::BlendRefusal::UnknownOrdinal {
                field: "dst_rgb",
                ordinal: 99,
            }
        );
    }

    /// The projection carries the guest's axes in the guest's order.
    ///
    /// Three distinct modes, because a swap of two axes is the failure this
    /// catches and a key whose axes agree could not show one.
    #[test]
    fn the_projection_carries_each_axis_to_the_field_that_means_it() {
        use reims_vgpu_core::sampler as mtl;
        let mut key = unnormalized_key();
        key.unnormalized_coordinates = false;
        key.address_mode_u = mtl::MTL_SAMPLER_ADDRESS_MODE_REPEAT;
        key.address_mode_v = mtl::MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT;
        key.address_mode_w = mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        let shape = sampler_shape(&key);
        assert_eq!(shape.s_address, mtl::MTL_SAMPLER_ADDRESS_MODE_REPEAT);
        assert_eq!(shape.t_address, mtl::MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT);
        assert_eq!(shape.r_address, mtl::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        assert!(shape.normalized_coordinates);
        assert_eq!(shape.min_filter, key.min_filter);
        assert_eq!(shape.mag_filter, key.mag_filter);
        assert_eq!(shape.lod_min_clamp, 1.0);
        assert_eq!(shape.lod_max_clamp, 8.0);
    }

    /// This key has no separate "compares" flag and the protocol shape does, so
    /// the projection has to supply one. `Never` is Metal's default for a
    /// sampler that does not compare, and it is the only value that may mean
    /// "off" — every other function is a comparison the guest asked for.
    #[test]
    fn a_comparison_is_enabled_for_every_function_except_metals_default() {
        use super::super::types::SamplerCompareFunction as C;
        for function in [
            C::Never,
            C::Less,
            C::Equal,
            C::LessEqual,
            C::Greater,
            C::NotEqual,
            C::GreaterEqual,
            C::Always,
        ] {
            let mut key = unnormalized_key();
            key.unnormalized_coordinates = false;
            key.compare_function = function;
            let shape = sampler_shape(&key);
            assert_eq!(shape.compare_enabled, function != C::Never, "{function:?}");
            assert_eq!(shape.compare_function, function.mtl_ordinal());
        }
    }

    /// `-01077` is the one constraint that is not observationally neutral, so it
    /// stays a named refusal and never becomes a repair.
    #[test]
    fn an_unnormalized_sampler_with_a_compare_function_is_refused_by_name() {
        use crate::observe::Decline as _;
        let mut key = unnormalized_key();
        key.compare_function = super::super::types::SamplerCompareFunction::LessEqual;
        let refusal = sampler_shape(&key)
            .checked()
            .expect_err("a comparison under pixel coordinates is refused");
        assert_eq!(
            super::super::reason::DrawReason::SamplerDeclaration(refusal).slug(),
            "sampler_unnormalized_restriction"
        );
    }

    #[test]
    fn vertex_format_widening_names_both_formats_and_attribute() {
        use crate::observe::Decline as _;
        let guest = VertexAttributeFormat::UChar3Normalized;
        let binding = translate::support::resolve(
            translate::VertexFormatSupport::all().without(guest),
            guest,
            12,
            32,
            || crate::runtime::spirv_vertex_input::InputWidth::Components(3),
        )
        .unwrap();
        let decline = VertexFormatWidenDecline {
            from: binding.widened_from.unwrap(),
            to: binding.format,
            location: 3,
            offset: 12,
            stride: 32,
        };
        assert_eq!(decline.slug(), "vk_vertex_format_widened");
        assert_eq!(
            crate::observe::Emit::decline("vk_engine_vertex_format", &decline).render(),
            "vk_engine_vertex_format reason=vk_vertex_format_widened \
             from=R8G8B8_UNORM to=R8G8B8A8_UNORM location=3 offset=12 stride=32"
        );
    }

    #[test]
    fn negative_map_is_bounded_by_cap() {
        // Negative entries (create failures) must not grow without bound: a
        // guest submitting endless distinct never-creatable objects would
        // otherwise leak one entry per distinct key forever.
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
        for k in 0..100u32 {
            c.insert_negative(
                k,
                DrawError::VkCall(VkCall::new(
                    VkOp::CachesCreateShaderModule,
                    vk::Result::ERROR_UNKNOWN,
                )),
            );
        }
        assert_eq!(c.negative.len(), 4, "negative map bounded by cap");
        assert!(
            c.negative_order.len() <= 8,
            "order deque bounded (<= 2*cap): {}",
            c.negative_order.len()
        );
        // The newest 4 keys survive (oldest-first eviction).
        for k in 96..100u32 {
            assert!(c.get_negative(&k).is_some(), "recent negative {k} retained");
        }
        assert!(c.get_negative(&0).is_none(), "oldest negative evicted");
    }

    /// The first pipeline a boot creates is the compositor's, and it stays bound
    /// for the life of the guest. Under the retired insertion-order cap it was
    /// also the first thing a cap crossing threw away. Drive far past every cap
    /// this file used to carry (1024, and 64 for render passes) and assert the
    /// first key is still served and nothing was displaced for capacity.
    #[test]
    fn the_first_key_survives_far_past_every_retired_capacity() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::new();
        c.insert(0, 0xC0FFEE);
        for k in 1..4096u32 {
            assert!(
                c.insert(k, k).is_none(),
                "a fresh key displaces nothing: {k}"
            );
        }
        assert_eq!(
            c.get(&0),
            Some(0xC0FFEE),
            "the hot first entry is still served after 4095 later ones"
        );
        assert_eq!(c.get_routed(&0), Some((0xC0FFEE, true)));
        assert_eq!(c.map.len(), 4096, "every distinct key retained");
    }

    /// A replace hands the displaced handle back so the caller can destroy it.
    /// The retired implementation overwrote in place and returned `None`, which
    /// leaked the Vulkan object it had just dropped the last reference to.
    #[test]
    fn replacing_a_key_returns_the_displaced_value_to_destroy() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::new();
        assert_eq!(c.insert(1, 10), None);
        assert_eq!(
            c.insert(1, 20),
            Some(10),
            "the displaced handle comes back for disposal"
        );
        assert_eq!(c.get(&1), Some(20));
    }

    #[test]
    fn clearing_the_cache_forgets_the_front_value_with_the_owned_object() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::new();
        c.insert(1, 20);
        assert_eq!(c.get_routed(&1), Some((20, true)));

        c.clear();

        assert_eq!(c.get_routed(&1), None);
        assert!(c.front.is_none());
    }

    #[test]
    fn retained_object_front_requires_the_same_identity_and_exact_variant() {
        let mut index: ObjectVariantIndex<u32, u32> = ObjectVariantIndex::default();
        let first = super::super::types::PipelineObjectIdentity::new();
        let second = super::super::types::PipelineObjectIdentity::new();

        index.remember(&first, &7, 70);
        assert_eq!(index.get(&first, &7), Some(70));
        assert_eq!(index.get(&first, &8), None, "a Vulkan-only variant differs");
        assert_eq!(
            index.get(&second, &7),
            None,
            "equal content under another guest object is not this object's front"
        );

        index.remember(&first, &8, 80);
        assert_eq!(index.get(&first, &7), None, "one exact last variant");
        assert_eq!(index.get(&first, &8), Some(80));
    }

    #[test]
    fn retained_object_front_reaps_dead_identities_without_capacity_eviction() {
        let mut index: ObjectVariantIndex<u32, u32> = ObjectVariantIndex::default();
        let first = super::super::types::PipelineObjectIdentity::new();
        index.remember(&first, &1, 10);
        assert_eq!(index.map.len(), 1);
        drop(first);

        let second = super::super::types::PipelineObjectIdentity::new();
        index.remember(&second, &2, 20);
        assert_eq!(index.map.len(), 1, "the expired object's weak entry went");
        assert_eq!(index.get(&second, &2), Some(20));

        index.clear();
        assert!(index.map.is_empty());
        assert_eq!(index.get(&second, &2), None);
    }

    #[test]
    fn positive_insert_clears_negative_for_the_key() {
        // A key that failed then later succeeds must not keep serving the stale
        // negative error.
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
        c.insert_negative(
            7,
            DrawError::VkCall(VkCall::new(
                VkOp::CachesCreateShaderModule,
                vk::Result::ERROR_UNKNOWN,
            )),
        );
        assert!(c.get_negative(&7).is_some());
        c.insert(7, 42);
        assert!(c.get_negative(&7).is_none(), "promotion clears negative");
        assert_eq!(c.get(&7), Some(42));
    }

    #[test]
    fn reinserting_same_negative_does_not_duplicate_order() {
        // Both results here are inherent to the request, so both are remembered.
        // They used to be the two out-of-memory results, which no longer reach
        // the map at all — see `an_out_of_memory_refusal_is_never_remembered`.
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
        let a = DrawError::VkCall(VkCall::new(
            VkOp::CachesCreateShaderModule,
            vk::Result::ERROR_UNKNOWN,
        ));
        let b = DrawError::VkCall(VkCall::new(
            VkOp::CachesCreateShaderModule,
            vk::Result::ERROR_INITIALIZATION_FAILED,
        ));
        c.insert_negative(1, a);
        c.insert_negative(1, b.clone());
        assert_eq!(c.negative_order.len(), 1, "same key tracked once");
        assert_eq!(c.get_negative(&1), Some(b), "error refreshed");
    }

    /// Out of memory says what the device holds *now*. The lookup consults
    /// `negative` before the create, so a remembered one is never displaced by a
    /// later success — the create that would displace it never runs. Remembering
    /// it turns "refused while full" into "refused forever", which is the failure
    /// mode a real GPU does not have.
    #[test]
    fn an_out_of_memory_refusal_is_never_remembered() {
        for result in [
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ] {
            let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateGraphicsPipelines, result));
            assert!(err.out_of_memory(), "{result:?} is the retryable class");
            c.insert_negative(1, err);
            assert_eq!(
                c.get_negative(&1),
                None,
                "{result:?} must not short-circuit the next create"
            );
            assert!(c.negative.is_empty(), "{result:?} left no entry");
            assert!(c.negative_order.is_empty(), "{result:?} left no order slot");
        }
    }

    /// The converse, so the test above cannot pass by disabling the map. A
    /// refusal inherent to the request — malformed SPIR-V, or a capability this
    /// host does not have — is worth remembering, because a second identical
    /// attempt meets it again.
    #[test]
    fn a_refusal_inherent_to_the_request_is_still_remembered() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);

        let bad_shader = DrawError::VkCall(VkCall::new(
            VkOp::CachesCreateShaderModule,
            vk::Result::ERROR_INVALID_SHADER_NV,
        ));
        assert!(!bad_shader.out_of_memory());
        c.insert_negative(1, bad_shader.clone());
        assert_eq!(c.get_negative(&1), Some(bad_shader));

        let unsupported = DrawError::Unsupported(
            super::super::reason::DrawReason::InstanceRateDivisorUnsupported { step_rate: 3 },
        );
        assert!(!unsupported.out_of_memory());
        c.insert_negative(2, unsupported.clone());
        assert_eq!(c.get_negative(&2), Some(unsupported));
    }

    /// A pipeline the guest still wants is asked for again after the memory it
    /// needed came back. This is the whole point of the rule, written as the
    /// sequence a guest actually produces: create fails while an atlas is
    /// resident, the guest frees the atlas, the guest re-binds the same
    /// pipeline. Before the rule, step three replayed a stale error and the
    /// driver was never asked.
    #[test]
    fn a_key_that_ran_out_of_memory_is_created_on_the_next_ask() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
        let key = 0xB0BAu32;

        // Frame N: the create refuses because the device is full.
        c.insert_negative(
            key,
            DrawError::VkCall(VkCall::new(
                VkOp::CachesCreateGraphicsPipelines,
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            )),
        );

        // Frame N+1: the guest asks again. Nothing short-circuits it, so the
        // caller reaches its create.
        assert_eq!(
            c.get_negative(&key),
            None,
            "the second ask must reach the driver"
        );
        assert_eq!(c.get(&key), None, "and it is still a miss, not a stale hit");

        // The memory came back, so this time the create succeeds.
        c.insert(key, 0x5EED);
        assert_eq!(c.get(&key), Some(0x5EED));
    }

    /// The index is keyed on the *allocation*, not on the contents, and that is
    /// the whole of its soundness argument: it holds the `Arc`, so the address
    /// cannot be reused while the entry lives.
    ///
    /// Two `Arc`s over identical words are two allocations and therefore two
    /// entries. That is not a miss to fix — a content key is what the digest
    /// already is, and rederiving it is what this index exists to avoid.
    #[test]
    fn the_shader_digest_index_keys_the_allocation_and_not_the_contents() {
        let mut index = ShaderDigestIndex::default();
        let words = std::sync::Arc::new(vec![0x0723_0203u32, 0x0001_0000, 0x000d_000b]);
        let twin = std::sync::Arc::new((*words).clone());
        let digest = Digest128 {
            a: 0xA1,
            b: 0xB2,
            len: 3,
        };

        assert_eq!(index.get(&words), None, "nothing walked yet");
        index.insert(&words, digest);

        assert_eq!(index.get(&words), Some(digest));
        assert_eq!(
            index.get(&twin),
            None,
            "identical words in a second allocation are a second entry"
        );
        let alias = std::sync::Arc::clone(&words);
        assert_eq!(
            index.get(&alias),
            Some(digest),
            "a clone of the same Arc is the same allocation and the same entry"
        );
    }

    /// A dropped module's address may be handed to the next allocation, and the
    /// index must not answer for it. It cannot: the entry holds an `Arc`, so
    /// while it lives the allocation is not freed and the address is not
    /// available to hand out.
    ///
    /// This asserts the mechanism rather than the hazard — a test that freed an
    /// allocation and hoped for the address back would be testing the allocator.
    #[test]
    fn a_shader_digest_entry_keeps_its_words_alive() {
        let mut index = ShaderDigestIndex::default();
        let words = std::sync::Arc::new(vec![1u32, 2, 3]);
        index.insert(&words, Digest128 { a: 1, b: 2, len: 3 });
        assert_eq!(
            std::sync::Arc::strong_count(&words),
            2,
            "the index holds one, which is what makes its key an address"
        );
        index.clear();
        assert_eq!(std::sync::Arc::strong_count(&words), 1, "and releases it");
    }

    /// The bound is the container's and it starts over rather than evicting,
    /// because every entry is equally cheap to rebuild and there is no recency
    /// to evict by.
    #[test]
    fn the_shader_digest_index_starts_over_at_its_bound() {
        let mut index = ShaderDigestIndex::default();
        let held: Vec<std::sync::Arc<Vec<u32>>> = (0..SHADER_DIGEST_ENTRIES)
            .map(|i| std::sync::Arc::new(vec![i as u32]))
            .collect();
        for (i, words) in held.iter().enumerate() {
            index.insert(
                words,
                Digest128 {
                    a: i as u64,
                    b: 0,
                    len: 1,
                },
            );
        }
        assert_eq!(index.map.len(), SHADER_DIGEST_ENTRIES);
        assert!(index.get(&held[0]).is_some());

        let one_more = std::sync::Arc::new(vec![0xFFFF_FFFFu32]);
        index.insert(&one_more, Digest128 { a: 9, b: 9, len: 1 });

        assert_eq!(index.map.len(), 1, "the bound holds by starting over");
        assert_eq!(index.get(&one_more), Some(Digest128 { a: 9, b: 9, len: 1 }));
        assert!(
            index.get(&held[0]).is_none(),
            "and the reset is total, so nothing survives to be answered stale"
        );
    }

    /// Every pass shape states the colour scope on **both** external
    /// dependencies, because declaring one explicit external dependency for any
    /// reason removes the implicit one from every attachment.
    ///
    /// This is the check the previous shape could not pass. It built the pair
    /// only for a depth pass and named depth-stencil stages and accesses alone,
    /// so on `(depth, _)` the colour attachment's transitions were ordered
    /// against nothing — which the synchronization validation layer reported at
    /// both `vkCmdBeginRenderPass` and `vkCmdEndRenderPass`. Asserting the
    /// colour terms across all four shapes is what stops a future edit adding a
    /// The narrow probe removes the transfer and sampler destinations and
    /// changes nothing else — in particular it keeps the colour-attachment
    /// destination, which is the one consumer that deliberately issues no
    /// barrier of its own.
    ///
    /// `super::exec::pass_exit_needs_no_barrier` drops the next draw's barrier
    /// when it renders into the same target, and the only thing that then orders
    /// that draw's writes after this pass's is the outgoing dependency. So a
    /// narrowing that took `COLOR_ATTACHMENT_OUTPUT` out of the destination
    /// scope would be a write-after-write hazard on the most common draw in the
    /// device, and it would be silent. The source scope must not move either:
    /// what is being priced is the *visibility* request, not the ordering one.
    ///
    /// See [`crate::config::PASS_EXIT_NARROW`] for what the probe is asking and for
    /// the validation-layer run it owes before it could ever be a default.
    #[test]
    fn the_narrow_pass_exit_keeps_the_consumer_that_issues_no_barrier() {
        for has_depth in [false, true] {
            for color_input in [false, true] {
                let wide = external_dependencies(has_depth, color_input, false, false)[1];
                let narrow = external_dependencies(has_depth, color_input, false, true)[1];
                let shape = format!("depth={has_depth} color_input={color_input}");

                assert_eq!(
                    (narrow.src_stage_mask, narrow.src_access_mask),
                    (wide.src_stage_mask, wide.src_access_mask),
                    "{shape}: the probe prices visibility and must not weaken ordering"
                );
                assert!(
                    narrow
                        .dst_stage_mask
                        .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        && narrow
                            .dst_access_mask
                            .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                    "{shape}: the next draw into this target issues no barrier of \
                     its own, so its stage must survive the narrowing"
                );
                assert!(
                    !narrow
                        .dst_stage_mask
                        .contains(vk::PipelineStageFlags::TRANSFER)
                        && !narrow
                            .dst_stage_mask
                            .contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
                    "{shape}: the probe removes exactly the two destinations whose \
                     consumers barrier for themselves"
                );
                assert!(
                    wide.dst_stage_mask
                        .contains(vk::PipelineStageFlags::TRANSFER)
                        && wide
                            .dst_stage_mask
                            .contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
                    "{shape}: and the shipping arm is unchanged, or the probe is \
                     measuring against itself"
                );
                if has_depth {
                    assert!(
                        narrow
                            .dst_stage_mask
                            .contains(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS),
                        "{shape}: depth is an attachment stage and stays"
                    );
                }
            }
        }
    }

    /// dependency for one attachment class and dropping the others again.
    #[test]
    fn both_external_dependencies_name_the_colour_scope_in_every_pass_shape() {
        for has_depth in [false, true] {
            for color_input in [false, true] {
                for host_accessible in [false, true] {
                    // The shipping scope. The probe arm has its own test.
                    let [incoming, outgoing] =
                        external_dependencies(has_depth, color_input, host_accessible, false);
                    let shape = format!(
                        "depth={has_depth} color_input={color_input} host={host_accessible}"
                    );

                    // Incoming: the transition into the attachment layout has to be
                    // ordered against the loadOp that writes it.
                    assert!(
                        incoming
                            .dst_stage_mask
                            .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
                        "{shape}: the loadOp clear runs at COLOR_ATTACHMENT_OUTPUT"
                    );
                    assert!(
                        incoming
                            .dst_access_mask
                            .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                        "{shape}: and it is a colour write"
                    );

                    // Outgoing: the final transition has to be ordered against the
                    // subpass's own store, and the store made visible to the copy
                    // that reads the target after the pass.
                    assert!(
                        outgoing
                            .src_stage_mask
                            .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
                        "{shape}: the store runs at COLOR_ATTACHMENT_OUTPUT"
                    );
                    assert!(
                        outgoing
                            .src_access_mask
                            .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                        "{shape}: and it is a colour write"
                    );
                    assert!(
                        outgoing
                            .dst_stage_mask
                            .contains(vk::PipelineStageFlags::TRANSFER)
                            && outgoing
                                .dst_access_mask
                                .contains(vk::AccessFlags::TRANSFER_READ),
                        "{shape}: a reader still barriers slot 0 into TRANSFER_SRC_OPTIMAL \
                     for itself, and this is the scope its transition orders against"
                    );

                    // Depth is stated only where the pass has a depth attachment —
                    // the fix is to add the missing class, not to name every class
                    // on every pass.
                    assert_eq!(
                        incoming
                            .dst_access_mask
                            .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
                        has_depth,
                        "{shape}: depth terms follow the depth attachment"
                    );

                    // Framebuffer fetch reads attachment 0 in the fragment stage, so
                    // the entry transition must be visible to that read as well.
                    assert_eq!(
                        incoming
                            .dst_access_mask
                            .contains(vk::AccessFlags::INPUT_ATTACHMENT_READ),
                        color_input,
                        "{shape}: input-attachment terms follow the fetch"
                    );
                    assert_eq!(
                        incoming
                            .src_stage_mask
                            .contains(vk::PipelineStageFlags::HOST)
                            && incoming
                                .src_access_mask
                                .contains(vk::AccessFlags::HOST_WRITE),
                        host_accessible,
                        "{shape}: host writes source only a host-accessible attachment"
                    );
                }
            }
        }
    }

    #[test]
    fn a_host_accessible_primary_stays_general_between_every_pass_shape() {
        let mut key = PassKey::single(Color0Load::Preserve, vk::Format::R8G8B8A8_UNORM);
        key.host_accessible_color0 = true;
        for color_input in [false, true] {
            for feedback in [false, true] {
                key.color_input = color_input;
                key.feedback_colors = u8::from(feedback);
                assert_eq!(
                    key.color_layout(0),
                    if feedback {
                        color_feedback_layout()
                    } else {
                        vk::ImageLayout::GENERAL
                    }
                );
                assert_eq!(key.color_final_layout(0), vk::ImageLayout::GENERAL);
            }
        }
        // Host accessibility is a property of slot 0 only, so a secondary is an
        // ordinary colour slot and rests where every ordinary colour slot does.
        assert_eq!(key.color_layout(1), color0_pass_exit_layout());
    }

    /// The pass key is the single source of truth for every place that names an
    /// attachment layout, and a feedback slot names the one
    /// [`color_feedback_layout`] answers at both ends of the pass.
    ///
    /// Asserted against that function rather than against a spelled layout, so it
    /// states the relation on both arms: under the shipping layout every slot is
    /// in the *same* layout feedback or not, and under
    /// [`crate::config::COLOR_GENERAL`]`=off` the feedback slots separate because
    /// `COLOR_ATTACHMENT_OPTIMAL` admits no feedback loop. The failure it guards
    /// is a descriptor and a subpass reference naming one image differently,
    /// which is undefined behaviour reported nowhere.
    #[test]
    fn feedback_attachment_layout_is_derived_consistently_from_the_mask() {
        let mut key = PassKey::single(Color0Load::Preserve, vk::Format::R8G8B8A8_UNORM);
        key.color_input = true;
        key.feedback_colors = (1 << 0) | (1 << 3);

        for index in 0..=MAX_SECONDARY_ATTACH {
            let feedback = index == 0 || index == 3;
            assert_eq!(key.color_feedback(index), feedback);
            let want = if feedback {
                color_feedback_layout()
            } else {
                color0_pass_exit_layout()
            };
            assert_eq!(key.color_layout(index), want);
            assert_eq!(key.color_final_layout(index), want);
        }
        assert!(!key.color_feedback(u8::BITS as usize));

        // Whatever the arm, the layout a feedback slot lands in must be one a
        // feedback loop is legal in. This is the check that would have caught the
        // shipping defect: with the resting layout moved to GENERAL, the slot was
        // still being placed in the extension layout while the image sat in
        // GENERAL.
        assert!(layout_admits_color_feedback(color_feedback_layout()));
    }

    /// One layout for a colour target means one for a sampled one too.
    ///
    /// The whole point of the repair: while the resting layout admits a feedback
    /// loop, a slot the guest samples and a slot it does not are in the *same*
    /// layout, so the render pass declares no transition for either and a
    /// framebuffer built for one serves the other.
    #[test]
    fn a_sampled_colour_slot_rests_where_every_other_colour_slot_rests() {
        if layout_admits_color_feedback(color0_pass_exit_layout()) {
            assert_eq!(color_feedback_layout(), color0_pass_exit_layout());
        } else {
            // The ablation arm, where the contract forces the second layout back.
            assert_eq!(
                color_feedback_layout(),
                vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
            );
        }
    }

    /// A render pass may only vary in ways
    /// [`PassKey::framebuffer_compatibility`] preserves, and dependencies are not
    /// among the things Vulkan's compatibility rule spares.
    ///
    /// `feedback_colors` is erased by that key, so the feedback self-dependency
    /// must be declared on every pass rather than only on feedback ones — a
    /// conditional one made `dependencyCount` differ between two passes the key
    /// called interchangeable, which is what the validation layer reported as
    /// `VUID-VkRenderPassBeginInfo-renderPass-00904` on a driven Maps boot.
    #[test]
    fn the_dependency_count_does_not_move_with_anything_the_framebuffer_key_erases() {
        let base = PassKey::single(Color0Load::Preserve, vk::Format::R8G8B8A8_UNORM);
        for feedback in [0u8, 1, (1 << 0) | (1 << 3)] {
            for host_accessible in [false, true] {
                let mut key = base;
                key.feedback_colors = feedback;
                key.host_accessible_color0 = host_accessible;
                assert_eq!(
                    key.framebuffer_compatibility(),
                    base.framebuffer_compatibility(),
                    "the framebuffer key must erase these"
                );
                // Same framebuffer key ⇒ the passes must agree on dependency
                // count, because a framebuffer built against one is used with the
                // other.
                assert_eq!(
                    pass_dependency_count(key),
                    pass_dependency_count(base),
                    "feedback={feedback} host_accessible={host_accessible}"
                );
            }
        }
    }

    /// The dependency list a pass of this shape is built with, counted without a
    /// device. Mirrors the `deps` construction in `get_or_create_render_pass`.
    fn pass_dependency_count(key: PassKey) -> usize {
        external_dependencies(
            key.depth.is_some(),
            key.color_input,
            key.host_accessible_color0,
            pass_exit_scope_narrow(),
        )
        .len()
            + usize::from(key.color_input)
            + 1
    }

    /// Erasing feedback from the pass key must not erase it from the device.
    ///
    /// `compatibility()` drops `feedback_colors` on the shipping arm so a feedback
    /// draw can continue an ordinary draw's render pass. The create flag
    /// `VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT` is what makes
    /// that draw legal, it is fixed at pipeline creation, and it is therefore the
    /// one thing that must **not** follow the field out of the pass key. This
    /// asserts the split: same pass compatibility, different pipeline key.
    ///
    /// Without it, the pass-merge win silently turns every feedback draw into a
    /// sampled read of an attachment it is writing with no feedback loop enabled.
    #[test]
    fn feedback_leaves_pass_compatibility_without_leaving_the_pipeline() {
        let plain = PassKey::single(Color0Load::Preserve, vk::Format::R8G8B8A8_UNORM);
        let mut feeds = plain;
        feeds.feedback_colors = 1;

        if color_feedback_layout() == color0_pass_exit_layout() {
            assert_eq!(
                plain.compatibility(),
                feeds.compatibility(),
                "a feedback draw must be able to continue an ordinary draw's pass"
            );
        }
        // Whatever the pass key did, the draw's own answer is still reachable and
        // is what the pipeline key is built from.
        assert_eq!(plain.feedback_colors, 0);
        assert_eq!(feeds.feedback_colors, 1);
    }

    /// A subpass self-dependency sourced in a framebuffer-space stage may name
    /// only framebuffer-space stages as its destination
    /// (`VUID-VkSubpassDependency-srcSubpass-06809`). `VERTEX_SHADER` is not one,
    /// and naming it made every render pass this device created invalid.
    #[test]
    fn the_feedback_self_dependency_stays_in_framebuffer_space() {
        const FRAMEBUFFER_SPACE: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
            vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw()
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw()
                | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw(),
        );
        assert!(FRAMEBUFFER_SPACE.contains(COLOR_FEEDBACK_SRC.0));
        assert!(FRAMEBUFFER_SPACE.contains(COLOR_FEEDBACK_DST.0));

        // The in-pass barrier in `exec` is built from these same two constants,
        // which is what keeps it inside what the self-dependency declares.
        let dep = color_feedback_self_dependency(color_feedback_layout());
        assert_eq!(dep.src_stage_mask, COLOR_FEEDBACK_SRC.0);
        assert_eq!(dep.dst_stage_mask, COLOR_FEEDBACK_DST.0);
        assert!(dep
            .dependency_flags
            .contains(vk::DependencyFlags::BY_REGION));
        assert_eq!(dep.src_subpass, dep.dst_subpass);
    }

    /// An ordinary colour slot names one layout at all three points a pass can
    /// name one — the `initialLayout` a `LOAD` names, the subpass reference, and
    /// the `finalLayout` — so the pass performs no transition of its own at
    /// either end and the registry's record of where the image was left is the
    /// layout it is actually in.
    ///
    /// This is the relation, not a restatement of a constant: it holds for
    /// whatever [`color0_pass_exit_layout`] answers, including under
    /// [`crate::config::COLOR_GENERAL`], which is the whole reason that answer is a
    /// function. It fails if any of the three grows a second spelling — which is
    /// how the MRT secondary arm in `exec` came to publish a hand-written
    /// `COLOR_ATTACHMENT_OPTIMAL` beside a feedback arm that derived.
    #[test]
    fn an_ordinary_colour_slot_enters_and_leaves_a_pass_at_one_layout() {
        let key = PassKey::single(Color0Load::Preserve, vk::Format::B8G8R8A8_UNORM);
        for index in 0..=MAX_SECONDARY_ATTACH {
            assert_eq!(
                key.color_layout(index),
                color0_pass_exit_layout(),
                "{index}"
            );
            assert_eq!(
                key.color_final_layout(index),
                color0_pass_exit_layout(),
                "{index}"
            );
        }
    }
}
