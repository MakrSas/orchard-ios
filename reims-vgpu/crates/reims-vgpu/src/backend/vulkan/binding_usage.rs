//! What this rail's translated shaders do with each bound slot.
//!
//! # The number this exists to move
//!
//! `reims_vgpu_core::access::AccessMode::Unknown` conflicts with everything, so
//! a bound slot nobody has reflected orders against every other access to the
//! same resource. The model has always said that the layer able to narrow it is
//! the executor that compiled the shader, and until this module no executor
//! did: a driven macos-15 boot read `access_mode_unknown=342 987` against
//! 81 116 known — 81 % of every access its dependency graph ordered against,
//! about 2.1 per draw.
//!
//! # Two different answers this must not confuse
//!
//! `reims_vgpu_core::pipeline::BindingUsage` reads a slot past the end of its
//! table as **unreferenced** — contributing no participation at all — which is
//! only true if whoever built the table enumerated the pipeline's whole binding
//! set. So a stage is published *whole* or not at all. A partial table would
//! silently delete the participations it left out, and a deleted participation
//! is a missing hazard edge rather than a coarser one.
//!
//! That is why a binding this mapping cannot place refuses the stage rather
//! than being skipped. The only bindings left out are the ones whose index
//! cannot be a guest slot at all — sampler state, a framebuffer fetch, and the
//! translator's two band placeholders — where leaving them out says nothing
//! about a slot the guest can spell.
//!
//! # The vertex fetch is not in the reflection, and it still reads
//!
//! A `[[stage_in]]` vertex shader names no `[[buffer(n)]]` for the data the
//! fixed-function fetch pulls: the vertex descriptor names those buffers, the
//! translator lowers them to vertex *input attributes*, and the reflection's
//! binding list therefore does not mention them. Published from the reflection
//! alone, every vertex buffer a guest binds would come back "unreferenced" and
//! a write to one would stop being ordered before the draw that reads it.
//!
//! [`VertexBindPlan::attribute_slots`] is the other half — every buffer index
//! the pipeline's attribute list names — and the vertex stage's table is the
//! union. A slot in it that the reflection did not describe is a read, because
//! the fetch reads it; a slot both describe keeps the shader's answer, which is
//! never weaker.
//!
//! # The compute fetch has the same hole and no descriptor to fill it
//!
//! A kernel `[[stage_in]]` is the same fixed-function read, and the compute
//! pipeline's stage-input descriptor is the vertex descriptor's twin — but it
//! does not reach here, and the reflection's index for the kind is one the
//! translator invented rather than one the guest spelled. So a kernel that
//! declares one refuses instead of publishing a table missing the slot the
//! guest actually bound. See [`UnpublishableStage::KernelStageInput`].

use reims_vgpu_core::access::AccessMode;
use reims_vgpu_core::pipeline::{BindingUsage, PublishedUsage};

use metal2vulkan::reflect::{ResourceAccess, ResourceKind, ShaderReflection};

use super::pipeline_resolve::VertexBindPlan;

/// Why a stage could not be published.
///
/// Each is a reason to leave the stage at `Unknown`, which costs ordering and
/// never correctness. They are named rather than collapsed because the counts
/// say which of them is worth closing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnpublishableStage {
    /// A binding whose Metal index is synthesized rather than the guest's, so
    /// it cannot be placed in a table the guest's slot numbers index.
    SyntheticIndex,
    /// A kernel `[[stage_in]]` attribute array: the fixed-function compute
    /// fetch, and the compute form of the problem the vertex stage solves with
    /// [`VertexBindPlan::attribute_slots`].
    ///
    /// **The contract term this is missing.** Metal supplies stage-input data
    /// through ordinary buffer-table slots the *pipeline descriptor* selects,
    /// and the reflection cannot name those: its `metal_index` for this kind is
    /// a slot allocated as the lowest index the kernel's own declared buffers
    /// do not occupy, which is a translator ABI decision and not a number the
    /// guest ever spelled. So the guest binds a buffer the dispatch reads and
    /// the table has no row that names it.
    ///
    /// Publishing anyway would call that slot unreferenced and delete a read
    /// edge, so the stage is refused whole. What closes it is the compute
    /// pipeline's stage-input descriptor — the compute twin of the vertex
    /// descriptor — which this device decodes for the dispatch and does not
    /// hand to this mapping.
    KernelStageInput,
    /// An authored acceleration structure or function table bound as an object
    /// rather than as bytes this device placed.
    ///
    /// It occupies a buffer index, so leaving it out of the table would read as
    /// unreferenced — and this mapping has no basis for saying that about a
    /// binding whose guest-side resource class it does not model.
    AuthoredObject,
    /// A slot number that does not fit the table this builds. The guest may
    /// bind any slot — nothing refuses a high one — so this is a real answer
    /// and not an impossibility.
    SlotTooHigh,
}

impl UnpublishableStage {
    pub(crate) const fn slug(self) -> &'static str {
        match self {
            Self::SyntheticIndex => "binding_usage_synthetic_index",
            Self::KernelStageInput => "binding_usage_kernel_stage_in",
            Self::AuthoredObject => "binding_usage_authored_object",
            Self::SlotTooHigh => "binding_usage_slot_too_high",
        }
    }
}

/// The highest slot this builds a table up to.
///
/// A bound slot above it refuses the stage rather than being dropped, for the
/// reason the module doc gives: a table that stops short reads as "unreferenced
/// from here on". Well above Apple's argument-table sizes, so a stage refused
/// here is a shader doing something this mapping has never seen rather than an
/// ordinary one.
const MAX_SLOT: u32 = 4095;

/// One stage's two tables, built from its reflection.
///
/// The vectors rather than a finished [`BindingUsage`], because the vertex
/// stage widens the buffer table afterwards and rebuilding one from the other's
/// accessors would be the same table written twice.
type Tables = (Vec<Option<AccessMode>>, Vec<Option<AccessMode>>);

fn stage_tables(reflection: &ShaderReflection) -> Result<Tables, UnpublishableStage> {
    let mut buffers: Vec<Option<AccessMode>> = Vec::new();
    let mut textures: Vec<Option<AccessMode>> = Vec::new();

    // `None` for the mode is `ResourceAccess::Unused`: the shader declares the
    // resource and never dereferences it, which is *unreferenced* and leaves
    // the slot unset. Every other answer, including ignorance, is written down
    // — see the module doc on why an unset slot is not a place to put "I do not
    // know".
    let put = |table: &mut Vec<Option<AccessMode>>,
               slot: u32,
               mode: Option<AccessMode>|
     -> Result<(), UnpublishableStage> {
        if slot > MAX_SLOT {
            return Err(UnpublishableStage::SlotTooHigh);
        }
        let Some(mode) = mode else {
            return Ok(());
        };
        let slot = slot as usize;
        if table.len() <= slot {
            table.resize(slot + 1, None);
        }
        // Two variables on one slot: keep the wider answer rather than the
        // later one, so the order the reflection lists them in cannot decide a
        // hazard.
        table[slot] = Some(match table[slot] {
            Some(existing) if existing != mode => AccessMode::Unknown,
            _ => mode,
        });
        Ok(())
    };

    for binding in &reflection.bindings {
        if binding.embedded_source.is_some() {
            return Err(UnpublishableStage::SyntheticIndex);
        }
        let slot = binding.metal_index;
        match binding.kind {
            ResourceKind::Buffer | ResourceKind::AccelerationStructureShadow => {
                // A constant-address-space buffer reflects as `ReadOnly`; a
                // device buffer's direction needs IR dataflow the facade does
                // not carry and stays `None`. `Unknown` is the honest answer
                // for the second, and it is still an improvement on the slot
                // being unnamed: the *other* slots of the same draw narrow.
                put(
                    &mut buffers,
                    slot,
                    match binding.access {
                        Some(ResourceAccess::Unused) => None,
                        Some(ResourceAccess::ReadOnly) => Some(AccessMode::Read),
                        Some(ResourceAccess::WriteOnly) => Some(AccessMode::Write),
                        Some(ResourceAccess::ReadWrite) => Some(AccessMode::ReadWrite),
                        // A texture classification on a buffer binding is a
                        // disagreement between the kind and the access, and
                        // guessing which is right is exactly what this must not
                        // do.
                        Some(ResourceAccess::Sampled | ResourceAccess::Storage) | None => {
                            Some(AccessMode::Unknown)
                        }
                    },
                )?;
            }
            // Workgroup memory, which the guest binds no resource for. It
            // occupies a buffer index all the same, so it is declared rather
            // than left out — an unnamed slot would read as unreferenced, and
            // this mapping has no basis for saying that about a slot it does
            // not understand.
            ResourceKind::ThreadgroupBuffer => {
                put(&mut buffers, slot, Some(AccessMode::Unknown))?;
            }
            ResourceKind::Texture | ResourceKind::TextureArray | ResourceKind::StorageImage => {
                put(
                    &mut textures,
                    slot,
                    match binding.access {
                        Some(ResourceAccess::Unused) => None,
                        Some(ResourceAccess::Sampled | ResourceAccess::ReadOnly) => {
                            Some(AccessMode::Read)
                        }
                        Some(ResourceAccess::WriteOnly) => Some(AccessMode::Write),
                        // A storage image is read *and* written directly, and
                        // both `read_write` and the bare storage qualifier land
                        // here. `ReadWrite` orders as `Unknown` does without
                        // claiming ignorance, which is the distinction the
                        // variant exists for.
                        Some(ResourceAccess::Storage | ResourceAccess::ReadWrite) => {
                            Some(AccessMode::ReadWrite)
                        }
                        None => Some(AccessMode::Unknown),
                    },
                )?;
            }
            // Bind no memory the footprint is about, and none of them can
            // shadow a guest slot. `Sampler` and `StaticSampler` are sampler
            // state; `ColorInput` is a framebuffer fetch the pass already
            // orders. The two synthesized placeholders sit at the first binding
            // in their band that no Metal resource claims, which is by
            // construction an index the guest did not bind — so leaving all
            // five out of both tables says nothing about a slot the guest can
            // spell.
            ResourceKind::Sampler
            | ResourceKind::StaticSampler
            | ResourceKind::ColorInput
            | ResourceKind::SynthesizedNullTexture
            | ResourceKind::SynthesizedReadSampler => {}
            // Indices the translator invented. Placing one would answer for
            // whatever the guest bound at the same number, and the answer would
            // be *narrower* than `Unknown`, which is the direction that loses
            // edges.
            ResourceKind::EmbeddedArgBufferTexture
            | ResourceKind::EmbeddedArgBufferBuffer
            | ResourceKind::BufferAddressTable => return Err(UnpublishableStage::SyntheticIndex),
            ResourceKind::KernelStageInput => return Err(UnpublishableStage::KernelStageInput),
            ResourceKind::PrimitiveAccelerationStructure
            | ResourceKind::VisibleFunctionTable
            | ResourceKind::IntersectionFunctionTable => {
                return Err(UnpublishableStage::AuthoredObject)
            } // No wildcard, on purpose. `ResourceKind` is exhaustively named
              // above, so a translator that grows a kind fails to *compile* here
              // rather than quietly answering `Unknown` for it — which is the
              // difference between a decision this mapping made and one it never
              // noticed it was making. The compiler is the only reader that can
              // catch that at the moment the kind arrives.
        }
    }
    Ok((buffers, textures))
}

/// One stage's table, built from its reflection.
fn stage_usage(reflection: &ShaderReflection) -> Result<BindingUsage, UnpublishableStage> {
    let (buffers, textures) = stage_tables(reflection)?;
    Ok(BindingUsage::new(buffers, textures))
}

/// The vertex stage's table: its reflection, widened by the buffers the
/// pipeline's attribute list fetches from.
fn vertex_usage(
    reflection: &ShaderReflection,
    plan: &VertexBindPlan,
) -> Result<BindingUsage, UnpublishableStage> {
    let (mut buffers, textures) = stage_tables(reflection)?;
    for &slot in plan.attribute_slots() {
        if slot > MAX_SLOT {
            return Err(UnpublishableStage::SlotTooHigh);
        }
        let slot = slot as usize;
        if buffers.len() <= slot {
            buffers.resize(slot + 1, None);
        }
        // Only where the shader said nothing. The fixed-function fetch reads,
        // and a shader that also writes the same buffer has the stronger claim.
        if buffers[slot].is_none() {
            buffers[slot] = Some(AccessMode::Read);
        }
    }
    Ok(BindingUsage::new(buffers, textures))
}

/// What this rail can say about one render pipeline, or why it cannot.
///
/// Both stages or neither: `PublishedUsage` states them separately, but a
/// pipeline whose fragment stage refuses is one whose vertex answer is still
/// exactly true, so the two are published independently and each `Err` leaves
/// its own stage `Unknown`.
pub(crate) fn render(
    vertex: &ShaderReflection,
    plan: &VertexBindPlan,
    fragment: &ShaderReflection,
) -> (PublishedUsage, [Option<UnpublishableStage>; 2]) {
    let v = vertex_usage(vertex, plan);
    let f = stage_usage(fragment);
    let refusals = [v.as_ref().err().copied(), f.as_ref().err().copied()];
    (PublishedUsage::render(v.ok(), f.ok()), refusals)
}

/// What this rail can say about one compute pipeline, or why it cannot.
///
/// One stage and no widening: a dispatch has no fixed-function fetch, so the
/// kernel's reflection is the whole of what it reads and writes and there is no
/// second table to union in. Whole or nothing for the reason the module doc
/// gives, which is why this is a `Result` rather than the render arm's pair —
/// a compute pipeline with a refused stage publishes nothing at all, and
/// nothing is what leaves the dispatch's slots `Unknown`.
pub(crate) fn compute(kernel: &ShaderReflection) -> Result<PublishedUsage, UnpublishableStage> {
    stage_usage(kernel).map(PublishedUsage::compute)
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::reflect::{ResourceBinding, ShaderStage};

    fn reflection(bindings: Vec<ResourceBinding>) -> ShaderReflection {
        ShaderReflection {
            reflection_version: metal2vulkan::reflect::REFLECTION_VERSION,
            stage: ShaderStage::Fragment,
            entry_point: None,
            bindings,
            argument_buffer_fields: vec![],
            vertex_attributes: vec![],
            varyings: vec![],
            render_targets: vec![],
            depth_members: vec![],
            depth_qualifier: None,
            stencil_members: vec![],
            local_size: None,
            max_work_group_size: None,
            vertex_builtins: None,
            tessellation: None,
            imageblock_layouts: vec![],
            implicit_imageblock_attachments: vec![],
            fragment_imageblock: None,
            datalayout: None,
            descriptor_layout: Default::default(),
            kernel_dispatch: None,
            runtime_sampler_specializations: vec![],
            runtime_storage_image_specializations: vec![],
            function_constants: vec![],
        }
    }

    fn binding(
        kind: ResourceKind,
        metal_index: u32,
        access: Option<ResourceAccess>,
    ) -> ResourceBinding {
        ResourceBinding {
            kind,
            metal_index,
            descriptor: None,
            param_index: None,
            stage_input_location: None,
            address_space: None,
            declared_size: None,
            extent: None,
            footprint: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: None,
            access,
            static_sampler: None,
        }
    }

    /// The three answers a reflection can give, and the fourth a reader must
    /// not confuse with them.
    ///
    /// A constant buffer narrows to a read, a sampled texture to a read, a
    /// storage image to a read-write, and a device buffer whose direction the
    /// facade does not carry stays `Unknown` — *declared* `Unknown`, which is
    /// not the same as absent: an absent slot is unreferenced and contributes
    /// no participation at all, which is the answer that would delete a hazard
    /// edge if it were reached by accident.
    #[test]
    fn a_reflection_narrows_what_it_states_and_declares_what_it_does_not() {
        let r = reflection(vec![
            binding(ResourceKind::Buffer, 0, Some(ResourceAccess::ReadOnly)),
            binding(ResourceKind::Buffer, 1, None),
            binding(ResourceKind::Texture, 0, Some(ResourceAccess::Sampled)),
            binding(ResourceKind::StorageImage, 1, Some(ResourceAccess::Storage)),
            binding(ResourceKind::Sampler, 0, None),
        ]);
        let usage = stage_usage(&r).expect("every kind here has a row");
        assert_eq!(usage.buffer(0), Some(AccessMode::Read));
        assert_eq!(usage.buffer(1), Some(AccessMode::Unknown));
        assert_eq!(
            usage.buffer(2),
            None,
            "a slot no binding names is unreferenced, which is the whole win"
        );
        assert_eq!(usage.texture(0), Some(AccessMode::Read));
        assert_eq!(usage.texture(1), Some(AccessMode::ReadWrite));
    }

    /// A synthetic index refuses the stage rather than being placed.
    ///
    /// Its `metal_index` is not the guest's, so writing it into a table the
    /// guest's slot numbers index would answer for whatever the guest bound
    /// there — and the answer would be *narrower*, which is the direction that
    /// loses edges.
    #[test]
    fn a_synthesized_binding_refuses_the_stage_instead_of_taking_a_guests_slot() {
        let r = reflection(vec![binding(
            ResourceKind::EmbeddedArgBufferTexture,
            0,
            Some(ResourceAccess::Sampled),
        )]);
        assert_eq!(stage_usage(&r), Err(UnpublishableStage::SyntheticIndex));
    }

    /// The vertex fetch's buffers are read, whether or not the shader names
    /// them.
    ///
    /// A `[[stage_in]]` vertex shader names none of them — the vertex
    /// descriptor does, and the translator lowers it to input attributes — so a
    /// table built from the reflection alone calls every vertex buffer
    /// unreferenced and stops ordering writes to it before the draw.
    #[test]
    fn the_vertex_fetchs_buffers_are_read_even_where_the_shader_is_silent() {
        let r = reflection(vec![binding(
            ResourceKind::Buffer,
            0,
            Some(ResourceAccess::ReadOnly),
        )]);
        let plan = VertexBindPlan::for_test(&[0, 3]);
        let usage = vertex_usage(&r, &plan).expect("a buffer and two attribute slots");
        assert_eq!(
            usage.buffer(0),
            Some(AccessMode::Read),
            "named by both, and the two agree"
        );
        assert_eq!(
            usage.buffer(3),
            Some(AccessMode::Read),
            "named only by the attribute list, and the fetch reads it"
        );
        assert_eq!(usage.buffer(1), None, "named by neither");
    }

    /// A kernel publishes its own stage and neither render one.
    ///
    /// The distinction the `PublishedUsage` shape exists for: asking a compute
    /// pipeline for its vertex stage must answer "not published" rather than
    /// "published, references nothing", because the second would delete every
    /// participation a mis-staged reader asked about.
    #[test]
    fn a_kernel_publishes_the_compute_stage_and_leaves_the_render_stages_unpublished() {
        use reims_vgpu_protocol::render::ShaderStage;

        let r = reflection(vec![
            binding(ResourceKind::Buffer, 0, Some(ResourceAccess::ReadOnly)),
            binding(
                ResourceKind::StorageImage,
                2,
                Some(ResourceAccess::WriteOnly),
            ),
            binding(ResourceKind::ThreadgroupBuffer, 4, None),
        ]);
        let published = compute(&r).expect("every kind here has a row");
        let kernel = published
            .stage(None)
            .expect("the compute stage is the one this publishes");
        assert_eq!(kernel.buffer(0), Some(AccessMode::Read));
        assert_eq!(
            kernel.buffer(4),
            Some(AccessMode::Unknown),
            "threadgroup memory occupies a buffer index and binds no guest              resource, so it is declared rather than left to read as unreferenced"
        );
        assert_eq!(kernel.texture(2), Some(AccessMode::Write));
        assert_eq!(kernel.texture(0), None, "no binding names it");
        assert!(
            published.stage(Some(ShaderStage::Vertex)).is_none()
                && published.stage(Some(ShaderStage::Fragment)).is_none(),
            "a dispatch has no render stages, and 'unpublished' is the true              answer rather than an empty table"
        );
    }

    /// A kernel this mapping cannot enumerate whole publishes nothing.
    ///
    /// One stage means there is no partial answer to keep: the render arm can
    /// publish a vertex table while the fragment one refuses, because each is
    /// exactly true on its own. A refused kernel leaves the dispatch's slots
    /// `Unknown`, which is what a table stopping short would silently not do.
    #[test]
    fn a_kernel_with_an_unenumerable_binding_publishes_nothing_at_all() {
        let r = reflection(vec![
            binding(ResourceKind::Buffer, 0, Some(ResourceAccess::ReadOnly)),
            binding(
                ResourceKind::EmbeddedArgBufferTexture,
                1,
                Some(ResourceAccess::Sampled),
            ),
        ]);
        assert_eq!(compute(&r), Err(UnpublishableStage::SyntheticIndex));
    }

    /// The compute fixed-function fetch refuses the stage, and is told apart
    /// from every other synthetic index.
    ///
    /// It is the vertex fetch's problem with no vertex descriptor to solve it:
    /// the guest binds the stage-input buffer at a slot its *pipeline*
    /// descriptor names, while the reflection's index for the kind is the
    /// lowest one the kernel's own buffers leave free. Publishing would call
    /// the guest's slot unreferenced and delete a read edge, so this refuses —
    /// and it carries its own reason because it is the one of these with a
    /// known way to close it.
    #[test]
    fn a_kernel_stage_in_refuses_the_stage_under_its_own_reason() {
        let r = reflection(vec![
            binding(ResourceKind::Buffer, 1, Some(ResourceAccess::ReadOnly)),
            binding(ResourceKind::KernelStageInput, 0, None),
        ]);
        assert_eq!(compute(&r), Err(UnpublishableStage::KernelStageInput));
        assert_eq!(
            UnpublishableStage::KernelStageInput.slug(),
            "binding_usage_kernel_stage_in"
        );
    }

    /// The translator's two placeholders are skipped, not refused.
    ///
    /// Each sits at the first binding in its band that no Metal resource
    /// claims, so its index is by construction one the guest did not bind.
    /// Leaving them out of both tables says nothing about a slot the guest can
    /// spell, and refusing for them would cost every other slot of the stage
    /// its answer for nothing.
    #[test]
    fn the_translators_placeholders_cost_the_stage_nothing() {
        let r = reflection(vec![
            binding(ResourceKind::Buffer, 0, Some(ResourceAccess::ReadOnly)),
            binding(ResourceKind::SynthesizedNullTexture, 7, None),
            binding(ResourceKind::SynthesizedReadSampler, 7, None),
        ]);
        let published = compute(&r).expect("a placeholder is not a refusal");
        let kernel = published.stage(None).expect("the compute stage");
        assert_eq!(kernel.buffer(0), Some(AccessMode::Read));
        assert_eq!(
            kernel.texture(7),
            None,
            "the placeholder binds no guest texture, and its band index is one \
             no Metal texture claims"
        );
    }
}
