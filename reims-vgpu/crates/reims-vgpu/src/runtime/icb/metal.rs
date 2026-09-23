//! The Metal rail's half of the indirect command buffer: the host
//! `MTLIndirectCommandBuffer` its parent's decode describes.
//!
//! [`super`] owns the guest contract — the serializer-object descriptor, the command
//! layout, and the encode/decode of one command slot's bytes — and that half is
//! rail-neutral by construction: it is the guest's own serialization and it
//! reads the same whichever rail executes it. Everything here is the other
//! half, and every item names an `MTL*` object: materializing an ICB, the
//! per-`(task, ref)` host cache, filling a render or compute slot into it, the
//! vertex descriptor a filled render slot needs, and the compute PSO that must
//! be built with `supportIndirectCommandBuffers`.
//!
//! Split out rather than left as seventeen `cfg`-ed items interleaved through
//! the decode. Interleaved, the boundary was invisible: reading [`super`] gave
//! no way to tell which of two neighbouring functions the Vulkan arm compiles,
//! and the answer was an attribute several screens up. The sibling
//! [`crate::runtime::draw::metal`] made the same split for the same reason.
//!
//! Nothing here is reached except through [`crate::backend::Backend`]'s Metal
//! arm or through `compute_session::metal`, so the module carries the gate once
//! and no item inside it repeats one.

use super::*;

/// Materialize a host Metal ICB from a decoded create descriptor (uncached).
pub fn materialize_metal_icb(
    desc: &IndirectCommandBufferDescriptor,
) -> Result<::metal::IndirectCommandBuffer, IcbStatus> {
    use crate::backend::metal::runtime::system_device;
    use ::metal::{IndirectCommandBufferDescriptor, MTLResourceOptions};

    if desc.max_command_count == 0 {
        return Err(IcbStatus::Args("icb_materialize_zero_command_count"));
    }
    let Some(device) = system_device() else {
        return Err(IcbStatus::NoMetal("icb_materialize_no_metal"));
    };
    let mtl_desc = IndirectCommandBufferDescriptor::new();
    // Pass wire commandTypes bits through as-is (SDK layout). Do not use
    // metal-0.33's MTLIndirectCommandType bitflags: ConcurrentDispatch is
    // mis-shifted and mesh bits (1<<7 / 1<<8) are omitted — from_bits_truncate
    // drops unknown bits and yields an empty ICB.
    crate::backend::metal::raw_metal::icb_descriptor_set_command_types(
        mtl_desc.as_ref(),
        desc.command_types as u64,
    );
    mtl_desc.set_inherit_buffers(desc.inherit_buffers());
    mtl_desc.set_inherit_pipeline_state(desc.inherit_pipeline_state());
    mtl_desc.set_max_vertex_buffer_bind_count(desc.max_vertex_buffer_bind_count as u64);
    mtl_desc.set_max_fragment_buffer_bind_count(desc.max_fragment_buffer_bind_count as u64);
    mtl_desc.set_max_kernel_buffer_bind_count(desc.max_kernel_buffer_bind_count as u64);
    // Prefer create-body count; fall back to layout-implied TG slot count. The
    // create-body count widens to meet the layout's: the body declares it in a
    // byte, the layout implies it from two 32-bit offsets, and the wider of the
    // two is what Metal is told.
    let max_tg = u32::from(desc.max_kernel_threadgroup_memory_bind_count)
        .max(icb_layout_kernel_tg_slot_count(&desc.layout));
    if max_tg > 0 {
        crate::backend::metal::raw_metal::set_max_kernel_threadgroup_memory_bind_count(
            mtl_desc.as_ref(),
            u64::from(max_tg),
        );
    }
    // Mesh / object bind counts from create body (macOS 14+).
    crate::backend::metal::raw_metal::set_max_mesh_buffer_bind_count(
        mtl_desc.as_ref(),
        desc.max_mesh_buffer_bind_count as u64,
    );
    crate::backend::metal::raw_metal::set_max_object_buffer_bind_count(
        mtl_desc.as_ref(),
        desc.max_object_buffer_bind_count as u64,
    );
    if desc.max_object_threadgroup_memory_bind_count > 0 {
        crate::backend::metal::raw_metal::set_max_object_threadgroup_memory_bind_count(
            mtl_desc.as_ref(),
            desc.max_object_threadgroup_memory_bind_count as u64,
        );
    }

    let options = MTLResourceOptions::from_bits_truncate(desc.options as u64);
    let Some(icb) = crate::backend::metal::raw_metal::new_indirect_command_buffer(
        device,
        &mtl_desc,
        desc.max_command_count as u64,
        options,
    ) else {
        return Err(IcbStatus::MetalFailed("icb_materialize_allocation_failed"));
    };
    Ok(icb)
}

struct HostIcbEntry {
    desc: IndirectCommandBufferDescriptor,
    icb: ::metal::IndirectCommandBuffer,
    /// Keep compute PSOs alive while command slots reference them.
    retained_psos: Vec<::metal::ComputePipelineState>,
    /// Keep render PSOs alive while command slots reference them.
    retained_psos_render: Vec<::metal::RenderPipelineState>,
    retained_buffers: Vec<::metal::Buffer>,
    /// GVA writeback descriptors for buffers bound into filled commands.
    writebacks: Vec<IcbWriteback>,
    /// True once at least one host fill or guest-memory fill has landed.
    has_fills: bool,
}

struct IcbWriteback {
    bind: crate::runtime::compute_exec::ComputeBufferBind,
    gva: u64,
    /// Host staging length (GPU result copied here after execute, then to GVA).
    len: usize,
    /// The staging walk's page set, carried from the [`StagedBuffer`] this slot
    /// was recorded from. A cached ICB replays long after the stage, so the
    /// writeback has to be bounded by where the buffer resolved *then*; a walk
    /// taken at replay time answers where the GVA points now, which is the
    /// question that lets a recycled page take the write.
    pages: std::collections::HashSet<u64>,
    mtl: ::metal::Buffer,
}

fn icb_cache() -> &'static parking_lot::Mutex<HashMap<(u32, u32), HostIcbEntry>> {
    static CACHE: OnceLock<parking_lot::Mutex<HashMap<(u32, u32), HostIcbEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| parking_lot::Mutex::new(HashMap::new()))
}

/// Drop every cached host ICB.
///
/// The Metal half of [`super::clear_icb_cache`], which owns the neutral
/// registry beside it: one entry point clears both, because a registry entry
/// outliving its host object would name a descriptor no
/// `MTLIndirectCommandBuffer` was built from.
pub(crate) fn clear_host_icb_cache() {
    icb_cache().lock().clear();
}

/// Resolve guest ICB ref → host Metal ICB, reusing the per-(task,ref) cache.
pub fn resolve_metal_icb<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
) -> Result<
    (
        IndirectCommandBufferDescriptor,
        ::metal::IndirectCommandBuffer,
    ),
    IcbStatus,
> {
    // The registry owns the descriptor and decides when a create body has
    // changed enough to invalidate what was recorded against it, so the host
    // object is materialized from the same answer the portable decode reads.
    let desc = resolve_icb_record(state, host, task_id, icb_ref)?;
    let mut cache = icb_cache().lock();
    if let Some(entry) = cache.get(&(task_id, icb_ref)) {
        // Descriptor must still match the create body we materialize from.
        if entry.desc.max_command_count == desc.max_command_count
            && entry.desc.command_types == desc.command_types
        {
            return Ok((entry.desc.clone(), entry.icb.clone()));
        }
    }
    let icb = materialize_metal_icb(&desc)?;
    cache.insert(
        (task_id, icb_ref),
        HostIcbEntry {
            desc: desc.clone(),
            icb: icb.clone(),
            retained_psos: Vec::new(),
            retained_psos_render: Vec::new(),
            retained_buffers: Vec::new(),
            writebacks: Vec::new(),
            has_fills: false,
        },
    );
    Ok((desc, icb))
}

/// Fill a host `MTLIndirectCommandBuffer` from the guest's command memory.
///
/// The decode is [`decode_icb_command_range`]; this is only the Metal half that
/// applies each decoded slot.
pub fn fill_icb_from_command_memory<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    range_location: u64,
    range_length: u64,
) -> Result<(), IcbStatus> {
    for fill in
        decode_icb_command_range(state, host, task_id, icb_ref, range_location, range_length)?
    {
        match fill {
            IcbCommandFill::Compute(f) => fill_compute_command(state, host, task_id, icb_ref, &f)?,
            IcbCommandFill::Render(f) => fill_render_command(state, host, task_id, icb_ref, &f)?,
        }
    }
    Ok(())
}

/// An attribute of the guest's serializer-object vertex-input block that this device could
/// not encode, which refuses the pipeline that declared it.
///
/// Carries what the [`DroppedVertexAttribute`] line reports, so the caller's
/// refusal and the log line name the same attribute and the same word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VertexAttributeUnencodable {
    pub slug: &'static str,
    pub location: u32,
    pub value: u32,
}

/// Build an `MTLVertexDescriptor` from the serializer-object pipeline vertex-input block.
///
/// `Ok(None)` ⇒ the pipeline declares no vertex input at all (SSBO-only, or
/// every entry undeclared); `Err` ⇒ an attribute the guest *did* declare could
/// not be encoded, and the pipeline must be refused rather than built.
pub(crate) fn metal_vertex_descriptor_from_attrs(
    attrs: &[crate::runtime::decode::resource::VertexAttribute],
) -> Result<Option<::metal::VertexDescriptor>, VertexAttributeUnencodable> {
    metal_vertex_descriptor_from_attrs_for_draw(attrs, false)
}

/// Build `MTLVertexDescriptor` from serializer-object vertex attributes.
///
/// When `for_patches` is true and a layout lacks an explicit step function,
/// use `PerPatchControlPoint` (SDK value 4) so post-tessellation vertex
/// functions receive control-point attributes correctly.
///
/// # One unencodable attribute refuses the whole pipeline
///
/// This used to skip the attribute, encode the rest, and hand back `Some(vd)` as
/// long as one survived — so the PSO was built with a `[[stage_in]]` struct
/// missing a field and the shader read whatever occupied it. Wrong geometry,
/// not an error, and nothing downstream could tell.
///
/// **The Vulkan arm already answers this correctly** and is what settles it:
/// `DrawPreparationDecline::VertexAttributeFormat` and
/// `..::VertexStepFunctionUnsupported` refuse the draw on exactly these two
/// words. Two arms consuming one wire form had two different answers, and the
/// one that skipped was the one with no way to say so.
///
/// A `format` or `stride` of zero is *not* this case. `MTLVertexFormatInvalid`
/// is 0, so a zero there is the guest declaring no attribute at that index —
/// the same shape as an unattached colour slot — and skipping it is what the
/// wire says to do. It is counted rather than assumed, because the count is
/// what would say if the reading were ever wrong.
pub(crate) fn metal_vertex_descriptor_from_attrs_for_draw(
    attrs: &[crate::runtime::decode::resource::VertexAttribute],
    for_patches: bool,
) -> Result<Option<::metal::VertexDescriptor>, VertexAttributeUnencodable> {
    use crate::backend::metal::mtl_enum;
    use ::metal::{MTLVertexStepFunction, VertexDescriptor};

    if attrs.is_empty() {
        return Ok(None);
    }
    let vd = VertexDescriptor::new().to_owned();
    let mut any = false;
    for a in attrs {
        if a.format == 0 || a.stride == 0 {
            crate::runtime::drain::note_store_route("icb_vertex_attr_undeclared");
            continue;
        }
        // Both words come straight off the guest's serializer-object descriptor and had no
        // check at all — they were reinterpreted as `MTLVertexFormat` and
        // `MTLVertexStepFunction` directly.
        let Some(format) = mtl_enum::vertex_format(a.format) else {
            let slug = "icb_vertex_attr_format_unsupported";
            note_dropped_vertex_attribute(slug, a.location, a.format);
            return Err(VertexAttributeUnencodable {
                slug,
                location: a.location,
                value: a.format,
            });
        };
        let step_ordinal = a.step_function_ordinal(if for_patches {
            MTLVertexStepFunction::PerPatchControlPoint as u32
        } else {
            MTLVertexStepFunction::PerVertex as u32
        });
        let Some(step) = mtl_enum::vertex_step_function(step_ordinal) else {
            let slug = "icb_vertex_attr_step_function_unsupported";
            note_dropped_vertex_attribute(slug, a.location, step_ordinal);
            return Err(VertexAttributeUnencodable {
                slug,
                location: a.location,
                value: step_ordinal,
            });
        };
        any = true;
        if let Some(attr) = vd.attributes().object_at(a.location as u64) {
            attr.set_format(format);
            attr.set_offset(a.offset as u64);
            attr.set_buffer_index(a.buffer_index as u64);
        }
        if let Some(layout) = vd.layouts().object_at(a.buffer_index as u64) {
            layout.set_stride(a.stride as u64);
            layout.set_step_function(step);
            layout.set_step_rate(a.step_rate() as u64);
        }
    }
    Ok(if any { Some(vd) } else { None })
}

/// A vertex attribute this device could not encode, named by which of its two
/// enum words the guest set to something Metal does not declare.
///
/// The line, beside the [`VertexAttributeUnencodable`] the caller refuses on.
/// Both exist because they answer to different readers: the refusal stops the
/// pipeline and the line says which attribute and which word stopped it, once
/// per pair, on a path a cache miss would otherwise repeat indefinitely.
struct DroppedVertexAttribute {
    slug: &'static str,
}

impl crate::observe::Decline for DroppedVertexAttribute {
    fn slug(&self) -> &'static str {
        self.slug
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

fn note_dropped_vertex_attribute(slug: &'static str, location: u32, value: u32) {
    use crate::observe::Decline as _;
    let decline = DroppedVertexAttribute { slug };
    crate::runtime::drain::note_store_route(decline.slug());
    crate::observe::Emit::decline("icb_vertex_attr", &decline)
        .field("location", location)
        .field("value", value)
        // One line per (location, value) pair: a pipeline rebuilt on every
        // cache miss would otherwise repeat the same drop indefinitely.
        .fail_once(((location as u64) << 32) | value as u64);
}

/// The five `MTLPrimitiveType` values the ICB wire encodes, by SDK ordinal.
/// `slug` names the caller so a refused value still says which draw form it
/// came from — the Draw and DrawIndexed arms shared this mapping verbatim and
/// differed only in that slug.
fn icb_primitive_type(
    primitive_type: u16,
    slug: &'static str,
) -> Result<::metal::MTLPrimitiveType, IcbStatus> {
    use ::metal::MTLPrimitiveType;
    match primitive_type {
        0 => Ok(MTLPrimitiveType::Point),
        1 => Ok(MTLPrimitiveType::Line),
        2 => Ok(MTLPrimitiveType::LineStrip),
        3 => Ok(MTLPrimitiveType::Triangle),
        4 => Ok(MTLPrimitiveType::TriangleStrip),
        _ => Err(IcbStatus::Args(slug)),
    }
}

/// Fill one **render** command slot on a cached host ICB (Metal IndirectRenderCommand).
///
/// Builds an ICB-capable render PSO from the serializer-object render pipeline's vertex/
/// fragment MTLBs (color0 = BGRA8Unorm, matching product mapping/scanout).
/// When the serializer-object body carries a vertex-input block, attaches an
/// `MTLVertexDescriptor` so `[[stage_in]]` attributes bind correctly.
pub fn fill_render_command<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    fill: &IcbRenderFill,
) -> Result<(), IcbStatus> {
    use crate::backend::metal::runtime::{new_buffer_from_host, system_device};
    use crate::runtime::compute_exec::metal::stage_buffer;
    use crate::runtime::compute_exec::ComputeBufferBind;
    use crate::runtime::decode::resource::{
        decode_function_descriptor, decode_render_pipeline_descriptor, FunctionDescriptor,
        OBJECT_TYPE_FUNCTION, OBJECT_TYPE_SERIALIZER_OBJECT,
    };
    use crate::runtime::gva_mem;
    use ::metal::{
        MTLIndexType, MTLPixelFormat, MeshRenderPipelineDescriptor, RenderPipelineDescriptor,
    };

    if icb_ref == 0 {
        return Err(IcbStatus::Args("icb_frc_ref_zero"));
    }
    let Some(device) = system_device() else {
        return Err(IcbStatus::NoMetal("icb_frc_no_metal"));
    };
    // Need create flags before staging: pipeline is required on the ICB
    // command unless inheritPipelineState (parent encoder supplies PSO).
    let (icb_desc, _) = resolve_metal_icb(state, host, task_id, icb_ref)?;
    // Host-fill path may already set offset; wire-decode path sets wire_va.
    let mut fill_resolved = fill.clone();
    resolve_render_fill_offsets(state, host, task_id, &mut fill_resolved)?;
    let fill = &fill_resolved;

    let is_patches = matches!(
        fill.draw,
        IcbRenderDraw::Patches { .. } | IcbRenderDraw::IndexedPatches { .. }
    );
    let is_mesh = matches!(
        fill.draw,
        IcbRenderDraw::MeshThreads(_) | IcbRenderDraw::MeshThreadgroups(_)
    );

    // Pipeline is required on the ICB command unless inheritPipelineState.
    // Mirrors fill_compute_command: when inherit, parent encoder supplies PSO
    // at execute (draw::apply_icb_encoder_inheritance).
    let pso = if !icb_desc.inherit_pipeline_state() {
        if fill.pipeline_ref == 0 {
            return Err(IcbStatus::Args("icb_frc_pipeline_ref_zero"));
        }
        let (_entry, desc_bytes) = objects::resolve_descriptor(
            state,
            host,
            task_id,
            fill.pipeline_ref,
            &[OBJECT_TYPE_SERIALIZER_OBJECT],
        )
        .map_err(|rung| {
            let slug = crate::observe::ladder_slugs!("icb_frc_pipeline")(rung);
            match rung {
                objects::LadderRung::NoListEntry
                | objects::LadderRung::DescRead { .. }
                | objects::LadderRung::NoTaskSpace => IcbStatus::Missing(slug),
                objects::LadderRung::WrongType { .. } => IcbStatus::BadDescriptor(slug),
            }
        })?;
        let rp = decode_render_pipeline_descriptor(&desc_bytes).map_err(|_| {
            IcbStatus::BadDescriptor(crate::observe::ladder_slug!(
                "icb_frc_pipeline",
                desc_decode
            ))
        })?;
        let load_fn = |func_ref: u32| -> Result<Vec<u8>, IcbStatus> {
            let (_entry, d) = objects::resolve_descriptor(
                state,
                host,
                task_id,
                func_ref,
                &[OBJECT_TYPE_FUNCTION],
            )
            .map_err(|rung| {
                let slug = crate::observe::ladder_slugs!("icb_frc_function")(rung);
                match rung {
                    objects::LadderRung::NoListEntry
                    | objects::LadderRung::DescRead { .. }
                    | objects::LadderRung::NoTaskSpace => IcbStatus::Missing(slug),
                    objects::LadderRung::WrongType { .. } => IcbStatus::BadDescriptor(slug),
                }
            })?;
            let f: FunctionDescriptor = decode_function_descriptor(&d).map_err(|_| {
                IcbStatus::BadDescriptor(crate::observe::ladder_slug!(
                    "icb_frc_function",
                    desc_decode
                ))
            })?;
            if f.blob_gva == 0 || f.blob_size < 4 {
                return Err(IcbStatus::Args("icb_frc_function_blob_empty"));
            }
            // Guest blob_size is authoritative — no product 1 MiB MTLB ceiling.
            let len = crate::runtime::draw::host_alloc_len(f.blob_size as u64)
                .ok_or(IcbStatus::Args("icb_frc_function_blob_too_large"))?;
            let mut mtlb = vec![0u8; len];
            gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                f.blob_gva,
                &mut mtlb,
                state.page_shift,
            )
            .map_err(|_| IcbStatus::MetalFailed("icb_frc_function_blob_read"))?;
            Ok(mtlb)
        };

        if rp.fragment_func_ref == 0 {
            return Err(IcbStatus::Missing("icb_frc_no_fragment_function"));
        }
        if is_mesh {
            // Mesh stage: mesh SPI `mesh_func_ref` (tag 0x02 under shape 0x14)
            // or classic `vertex_func_ref` (mesh-only / dual-export metallib).
            if rp.mesh_func_ref == 0 && rp.vertex_func_ref == 0 {
                return Err(IcbStatus::Missing("icb_frc_no_mesh_or_vertex_function"));
            }
        } else if rp.vertex_func_ref == 0 {
            return Err(IcbStatus::Missing("icb_frc_no_vertex_function"));
        }

        let frag = load_fn(rp.fragment_func_ref)?;
        let flib = device
            .new_library_with_data(&frag)
            .map_err(|_| IcbStatus::MetalFailed("icb_frc_fragment_library_load"))?;
        let fnames = flib.function_names();
        if fnames.len() != 1 {
            return Err(IcbStatus::Args("icb_frc_fragment_function_count"));
        }
        let ff = flib
            .get_function(&fnames[0], None)
            .map_err(|_| IcbStatus::MetalFailed("icb_frc_fragment_function_get"))?;

        // Mesh draws need MTLMeshRenderPipelineDescriptor + mesh descriptor factory.
        // Prefer mesh SPI serializer-object shape (tag 0x14; 0x01 object / 0x02 mesh / 0x03 frag);
        // else dual-export or mesh-only metallib in classic `vertex_func_ref`.
        let built = if is_mesh {
            use crate::backend::metal::raw_metal::{
                function_type, MTL_FUNCTION_TYPE_MESH, MTL_FUNCTION_TYPE_OBJECT,
            };
            use ::metal::Library;

            let pick_typed = |lib: &Library,
                              want: u64,
                              allow_single: bool|
             -> Result<Option<::metal::Function>, IcbStatus> {
                let names = lib.function_names();
                if names.is_empty() {
                    return Err(IcbStatus::Args("icb_frc_mesh_library_empty"));
                }
                let mut typed = None;
                for name in names.iter() {
                    let f = lib
                        .get_function(name, None)
                        .map_err(|_| IcbStatus::MetalFailed("icb_frc_mesh_typed_function_get"))?;
                    if function_type(f.as_ref()) == want {
                        typed = Some(f);
                        break;
                    }
                }
                if typed.is_none() && allow_single && names.len() == 1 {
                    typed =
                        Some(lib.get_function(&names[0], None).map_err(|_| {
                            IcbStatus::MetalFailed("icb_frc_mesh_single_function_get")
                        })?);
                }
                Ok(typed)
            };

            let mut mesh_fn = None;
            let mut object_fn = None;
            // Keep libraries alive for the Function refs they own.
            let mut mesh_lib_keep = None;
            let mut object_lib_keep = None;
            let mut dual_lib_keep = None;

            if rp.mesh_func_ref != 0 {
                let mtlb = load_fn(rp.mesh_func_ref)?;
                let lib = device
                    .new_library_with_data(&mtlb)
                    .map_err(|_| IcbStatus::MetalFailed("icb_frc_mesh_library_load"))?;
                mesh_fn = pick_typed(&lib, MTL_FUNCTION_TYPE_MESH, true)?;
                mesh_lib_keep = Some(lib);
            }
            if rp.object_func_ref != 0 {
                let otlb = load_fn(rp.object_func_ref)?;
                let lib = device
                    .new_library_with_data(&otlb)
                    .map_err(|_| IcbStatus::MetalFailed("icb_frc_object_library_load"))?;
                object_fn = pick_typed(&lib, MTL_FUNCTION_TYPE_OBJECT, true)?;
                object_lib_keep = Some(lib);
            }
            // Dual-export / mesh-only fallback when mesh tag absent, or object tag
            // absent and dual-export can supply the object stage.
            if (mesh_fn.is_none() || object_fn.is_none()) && rp.vertex_func_ref != 0 {
                let vtlb = load_fn(rp.vertex_func_ref)?;
                let lib = device
                    .new_library_with_data(&vtlb)
                    .map_err(|_| IcbStatus::MetalFailed("icb_frc_dual_library_load"))?;
                let names = lib.function_names();
                if names.is_empty() {
                    return Err(IcbStatus::Args("icb_frc_dual_library_empty"));
                }
                for name in names.iter() {
                    let f = lib
                        .get_function(name, None)
                        .map_err(|_| IcbStatus::MetalFailed("icb_frc_dual_function_get"))?;
                    match function_type(f.as_ref()) {
                        MTL_FUNCTION_TYPE_MESH if mesh_fn.is_none() => mesh_fn = Some(f),
                        MTL_FUNCTION_TYPE_OBJECT if object_fn.is_none() => object_fn = Some(f),
                        _ if mesh_fn.is_none() && names.len() == 1 => mesh_fn = Some(f),
                        _ => {}
                    }
                }
                dual_lib_keep = Some(lib);
            }

            let Some(mesh_f) = mesh_fn else {
                return Err(IcbStatus::Args("icb_frc_no_mesh_function_resolved"));
            };
            let mdesc = MeshRenderPipelineDescriptor::new();
            mdesc.set_mesh_function(Some(mesh_f.as_ref()));
            if let Some(ref of) = object_fn {
                mdesc.set_object_function(Some(of.as_ref()));
            }
            mdesc.set_fragment_function(Some(&ff));
            crate::backend::metal::raw_metal::mesh_pipeline_set_support_indirect_command_buffers(
                mdesc.as_ref(),
                true,
            );
            if let Some(ca) = mdesc.color_attachments().object_at(0) {
                ca.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            }
            // Keep mesh_f / object_fn / libraries alive until PSO is built.
            let pso = device
                .new_mesh_render_pipeline_state(&mdesc)
                .map_err(|_| IcbStatus::MetalFailed("icb_frc_mesh_pipeline_state"))?;
            drop(object_fn);
            drop(mesh_f);
            drop(mesh_lib_keep);
            drop(object_lib_keep);
            drop(dual_lib_keep);
            pso
        } else {
            let vert = load_fn(rp.vertex_func_ref)?;
            let vlib = device
                .new_library_with_data(&vert)
                .map_err(|_| IcbStatus::MetalFailed("icb_frc_vertex_library_load"))?;
            let vnames = vlib.function_names();
            if vnames.len() != 1 {
                return Err(IcbStatus::Args("icb_frc_vertex_function_count"));
            }
            let vf = vlib
                .get_function(&vnames[0], None)
                .map_err(|_| IcbStatus::MetalFailed("icb_frc_vertex_function_get"))?;
            let pdesc = RenderPipelineDescriptor::new();
            pdesc.set_vertex_function(Some(&vf));
            pdesc.set_fragment_function(Some(&ff));
            pdesc.set_support_indirect_command_buffers(true);
            if let Some(ca) = pdesc.color_attachments().object_at(0) {
                ca.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            }
            // Tessellation PSO fields required for drawPatches / drawIndexedPatches
            // (metal-0.33 leaves these as TODOs — raw msg_send).
            if is_patches {
                let cp_index_ty = match fill.draw {
                    IcbRenderDraw::IndexedPatches { .. } => {
                        // UInt16 control-point indices (product fill stages buffer bytes).
                        crate::backend::metal::raw_metal::MTL_TESSELLATION_CONTROL_POINT_INDEX_UINT16
                    }
                    _ => {
                        crate::backend::metal::raw_metal::MTL_TESSELLATION_CONTROL_POINT_INDEX_NONE
                    }
                };
                crate::backend::metal::raw_metal::configure_tessellation_pipeline(
                    pdesc.as_ref(),
                    16,
                    cp_index_ty,
                );
            }
            // Stage-in / control-point: serializer-object vertex-input block → MTLVertexDescriptor.
            // Patch draws force PerPatchControlPoint when the layout does not already
            // carry a step function (host tessellation oracle fixture).
            // Three answers, and the two that are not `Ok(Some)` used to be one
            // `if let` that ignored both. A pipeline declaring attributes and
            // getting no descriptor is a PSO with no `[[stage_in]]` at all,
            // which is not the same as a pipeline that declared none — the
            // sibling call in `draw::metal_icb` separates them and this one did
            // not, so the same wire form had two answers one file apart.
            match metal_vertex_descriptor_from_attrs_for_draw(&rp.vertex_attributes, is_patches) {
                Ok(Some(vd)) => pdesc.set_vertex_descriptor(Some(vd.as_ref())),
                Ok(None) if rp.vertex_attributes.is_empty() => {}
                Ok(None) => {
                    return Err(IcbStatus::BadDescriptor(
                        "icb_frc_vertex_descriptor_missing",
                    ))
                }
                Err(refusal) => return Err(IcbStatus::BadDescriptor(refusal.slug)),
            }
            device
                .new_render_pipeline_state(&pdesc)
                .map_err(|_| IcbStatus::MetalFailed("icb_frc_render_pipeline_state"))?
        };
        Some(built)
    } else {
        None
    };

    // (index, stage, has_vertex_stride, stride, buffer)
    let mut staged: Vec<(u32, IcbRenderBindStage, bool, u64, ::metal::Buffer)> = Vec::new();
    for b in &fill.buffers {
        let stage = b.effective_stage();
        let bind = ComputeBufferBind {
            index: b.index,
            buffer_ref: b.buffer_ref,
            offset: b.offset,
            attribute_stride: b.attribute_stride,
            has_attribute_stride: b.has_attribute_stride,
        };
        let s = stage_buffer(state, host, task_id, &bind)
            .map_err(|_| IcbStatus::Missing("icb_frc_bind_stage_buffer"))?;
        let mtl = new_buffer_from_host(device, s.bytes.as_ptr(), s.bytes.len())
            .ok_or(IcbStatus::MetalFailed("icb_frc_bind_host_buffer"))?;
        staged.push((
            b.index,
            stage,
            b.has_attribute_stride && stage == IcbRenderBindStage::Vertex,
            b.attribute_stride,
            mtl,
        ));
    }

    // Stage index / patch / tessellation factor buffers by object-list ref.
    let stage_buffer = |buffer_ref: u32, offset: u64| -> Result<::metal::Buffer, IcbStatus> {
        if buffer_ref == 0 {
            return Err(IcbStatus::Args("icb_frc_buffer_ref_zero"));
        }
        let bind = ComputeBufferBind {
            index: 0,
            buffer_ref,
            offset,
            attribute_stride: 0,
            has_attribute_stride: false,
        };
        let s = stage_buffer(state, host, task_id, &bind)
            .map_err(|_| IcbStatus::Missing("icb_frc_buffer_stage_buffer"))?;
        new_buffer_from_host(device, s.bytes.as_ptr(), s.bytes.len())
            .ok_or(IcbStatus::MetalFailed("icb_frc_buffer_host_buffer"))
    };

    let index_mtl = match fill.draw {
        IcbRenderDraw::Indexed {
            index_buffer_ref,
            index_buffer_offset,
            index_type,
            index_count,
            ..
        } => {
            let elem = match index_type {
                0 => 2usize, // MTLIndexTypeUInt16
                1 => 4usize, // MTLIndexTypeUInt32
                _ => return Err(IcbStatus::Args("icb_frc_index_type_unknown")),
            };
            let need = (index_count as usize)
                .checked_mul(elem)
                .ok_or(IcbStatus::Args("icb_frc_index_span_overflow"))?;
            if need == 0 {
                return Err(IcbStatus::Args("icb_frc_index_span_zero"));
            }
            let mtl = stage_buffer(index_buffer_ref, index_buffer_offset)?;
            // Product stages the index window at offset 0 in the retained buffer.
            let _ = need;
            Some(mtl)
        }
        IcbRenderDraw::Primitives { .. }
        | IcbRenderDraw::Patches { .. }
        | IcbRenderDraw::IndexedPatches { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => None,
    };

    // Optional patch-index buffer (nullable in Metal API).
    let patch_index_mtl = match fill.draw {
        IcbRenderDraw::Patches {
            patch_index_buffer_ref,
            patch_index_buffer_offset,
            ..
        }
        | IcbRenderDraw::IndexedPatches {
            patch_index_buffer_ref,
            patch_index_buffer_offset,
            ..
        } if patch_index_buffer_ref != 0 => Some(stage_buffer(
            patch_index_buffer_ref,
            patch_index_buffer_offset,
        )?),
        _ => None,
    };

    let control_point_index_mtl = match fill.draw {
        IcbRenderDraw::IndexedPatches {
            control_point_index_buffer_ref,
            control_point_index_buffer_offset,
            ..
        } => Some(stage_buffer(
            control_point_index_buffer_ref,
            control_point_index_buffer_offset,
        )?),
        _ => None,
    };

    // Tessellation factor buffer is required by Metal for drawPatches variants.
    let tess_factor_mtl = match fill.draw {
        IcbRenderDraw::Patches {
            tessellation_factor,
            ..
        }
        | IcbRenderDraw::IndexedPatches {
            tessellation_factor,
            ..
        } => {
            if tessellation_factor.buffer_ref == 0 {
                return Err(IcbStatus::Args("icb_frc_tess_factor_ref_zero"));
            }
            Some(stage_buffer(
                tessellation_factor.buffer_ref,
                tessellation_factor.offset,
            )?)
        }
        _ => None,
    };

    let mut cache = icb_cache().lock();
    let entry = cache
        .get_mut(&(task_id, icb_ref))
        .ok_or(IcbStatus::Missing("icb_frc_not_cached"))?;
    if fill.command_index as u64 >= entry.icb.size() {
        return Err(IcbStatus::Args("icb_frc_command_index_past_capacity"));
    }
    let cmd = entry
        .icb
        .indirect_render_command_at_index(fill.command_index as u64);
    if let Some(ref pso) = pso {
        cmd.set_render_pipeline_state(pso);
    }
    // When inheritBuffers, vertex/fragment buffers come from the parent encoder
    // at execute (see draw::encode_icb_execute_and_writeback).
    if !entry.desc.inherit_buffers() {
        // Every index is checked before any is bound, so a refusal leaves the
        // command slot as it was rather than half filled.
        for (idx, stage, _, _, _) in &staged {
            refuse_render_bind_past_declared_max(*stage, *idx, &entry.desc)?;
        }
        for (idx, stage, has_stride, stride, mtl) in &staged {
            match stage {
                IcbRenderBindStage::Fragment => {
                    cmd.set_fragment_buffer(*idx as u64, Some(mtl.as_ref()), 0);
                }
                IcbRenderBindStage::Mesh => {
                    crate::backend::metal::raw_metal::icb_set_mesh_buffer(
                        cmd,
                        Some(mtl.as_ref()),
                        0,
                        *idx as u64,
                    );
                }
                IcbRenderBindStage::Object => {
                    crate::backend::metal::raw_metal::icb_set_object_buffer(
                        cmd,
                        Some(mtl.as_ref()),
                        0,
                        *idx as u64,
                    );
                }
                IcbRenderBindStage::Vertex if *has_stride => {
                    crate::backend::metal::raw_metal::icb_set_vertex_buffer_attribute_stride(
                        cmd,
                        Some(mtl.as_ref()),
                        0,
                        *stride,
                        *idx as u64,
                    );
                }
                IcbRenderBindStage::Vertex => {
                    cmd.set_vertex_buffer(*idx as u64, Some(mtl.as_ref()), 0);
                }
            }
        }
    }
    // Object-stage threadgroup memory (mesh pipelines with objectFunction).
    for tg in &fill.object_threadgroup_memory {
        // Metal requires length multiple of 16 when non-zero (same as compute TG).
        if tg.length != 0 && tg.length % 16 != 0 {
            return Err(IcbStatus::Args("icb_frc_object_tg_length_alignment"));
        }
        // The index is a Metal argument-table slot and Metal answers an
        // over-range one by throwing, which aborts the process rather than
        // failing this fill. Same table and same reason as the direct compute
        // encoder's bind; see `REIMS_VGPU_METAL_MAX_THREADGROUP_MEMORY`.
        if !crate::backend::metal::util::valid_threadgroup_memory_index(tg.index) {
            return Err(IcbStatus::Args("icb_frc_object_tg_index_over_table"));
        }
        crate::backend::metal::raw_metal::icb_set_object_threadgroup_memory_length(
            cmd,
            tg.length,
            tg.index as u64,
        );
    }
    match fill.draw {
        IcbRenderDraw::Primitives {
            primitive_type,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => {
            let prim = icb_primitive_type(primitive_type, "icb_frc_draw_primitive_type")?;
            cmd.draw_primitives(
                prim,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            );
        }
        IcbRenderDraw::Indexed {
            primitive_type,
            index_type,
            index_count,
            index_buffer_offset: _,
            instance_count,
            base_vertex,
            base_instance,
            ..
        } => {
            let prim = icb_primitive_type(primitive_type, "icb_frc_indexed_primitive_type")?;
            let ity = match index_type {
                0 => MTLIndexType::UInt16,
                1 => MTLIndexType::UInt32,
                _ => return Err(IcbStatus::Args("icb_frc_indexed_index_type")),
            };
            let idx_buf = index_mtl
                .as_ref()
                .ok_or(IcbStatus::Missing("icb_frc_indexed_no_index_buffer"))?;
            // SDK: baseVertex is NSInteger (signed). Wire stores a u64 bit pattern
            // of the signed value (ld64 as i64). metal-0.33 types the ICB method
            // as NSUInteger — use raw msg_send with NSInteger.
            // Fail-closed only when the value does not fit NSInteger (platform width).
            let base_vertex_ns = base_vertex as ::metal::NSInteger;
            if base_vertex_ns as i64 != base_vertex {
                return Err(IcbStatus::Args("icb_frc_base_vertex_range"));
            }
            crate::backend::metal::raw_metal::icb_draw_indexed_primitives(
                cmd,
                prim,
                index_count,
                ity,
                idx_buf.as_ref(),
                0, // staged window starts at offset 0
                instance_count,
                base_vertex_ns,
                base_instance,
            );
        }
        IcbRenderDraw::Patches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            instance_count,
            base_instance,
            tessellation_factor,
            ..
        } => {
            if patch_count == 0 || number_of_patch_control_points == 0 {
                return Err(IcbStatus::Args("icb_frc_patches_zero_count"));
            }
            let tess = tess_factor_mtl
                .as_ref()
                .ok_or(IcbStatus::Missing("icb_frc_patches_no_tess_buffer"))?;
            // patchIndexBuffer is nullable in the SDK; product uses raw msg_send.
            crate::backend::metal::raw_metal::icb_draw_patches(
                cmd,
                number_of_patch_control_points as u64,
                patch_start,
                patch_count,
                patch_index_mtl.as_ref().map(|b| b.as_ref()),
                0,
                instance_count,
                base_instance,
                tess.as_ref(),
                0,
                tessellation_factor.instance_stride,
            );
        }
        IcbRenderDraw::IndexedPatches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            instance_count,
            base_instance,
            tessellation_factor,
            ..
        } => {
            if patch_count == 0 || number_of_patch_control_points == 0 {
                return Err(IcbStatus::Args("icb_frc_indexed_patches_zero_count"));
            }
            let tess = tess_factor_mtl
                .as_ref()
                .ok_or(IcbStatus::Missing("icb_frc_indexed_patches_no_tess_buffer"))?;
            let cp = control_point_index_mtl.as_ref().ok_or(IcbStatus::Missing(
                "icb_frc_indexed_patches_no_control_points",
            ))?;
            // patchIndexBuffer is nullable in the SDK.
            crate::backend::metal::raw_metal::icb_draw_indexed_patches(
                cmd,
                number_of_patch_control_points as u64,
                patch_start,
                patch_count,
                patch_index_mtl.as_ref().map(|b| b.as_ref()),
                0,
                cp.as_ref(),
                0,
                instance_count,
                base_instance,
                tess.as_ref(),
                0,
                tessellation_factor.instance_stride,
            );
        }
        IcbRenderDraw::MeshThreads(mesh) | IcbRenderDraw::MeshThreadgroups(mesh) => {
            use crate::backend::metal::raw_metal;
            let threads = matches!(fill.draw, IcbRenderDraw::MeshThreads(_));
            // All three extents are checked per component, not by their first
            // one: Metal validates an `MTLSize` in every dimension, so a zero in
            // `grid[1]` is as unencodable as one in `grid[0]` and used to reach
            // the selector. See `protocol::dispatch::mesh_draw_dims`, which also
            // owns the one substitution allowed here — an absent object
            // threadgroup read as 1.
            let Some(dims) =
                crate::protocol::dispatch::mesh_draw_dims(mesh.grid, mesh.object_tg, mesh.mesh_tg)
            else {
                return Err(IcbStatus::Args(if threads {
                    "icb_frc_mesh_threads_zero_dims"
                } else {
                    "icb_frc_mesh_threadgroups_zero_dims"
                }));
            };
            if dims.object_tg_defaulted {
                // Correct when the pipeline has no object stage, and wrong when
                // it has one — which this site cannot tell apart either, so the
                // reliance is reported rather than assumed. A reading here beside
                // a mesh pipeline that declares an object function is the bug.
                crate::observe::fail(format!(
                    "icb_mesh_object_tg_defaulted threads={threads} \
                     object_tg={:?} (read as 1; correct only with no object stage)",
                    mesh.object_tg
                ));
            }
            let size = |d: [u32; 3]| raw_metal::mtl_size(d[0] as u64, d[1] as u64, d[2] as u64);
            let grid = size(dims.grid);
            let obj_tg = size(dims.object_tg);
            let mesh_tg = size(dims.mesh_tg);
            if threads {
                raw_metal::icb_draw_mesh_threads(cmd, grid, obj_tg, mesh_tg);
            } else {
                raw_metal::icb_draw_mesh_threadgroups(cmd, grid, obj_tg, mesh_tg);
            }
        }
    }
    if let Some(pso) = pso {
        entry.retained_psos_render.push(pso);
    }
    for (_, _, _, _, mtl) in staged {
        entry.retained_buffers.push(mtl);
    }
    if let Some(mtl) = index_mtl {
        entry.retained_buffers.push(mtl);
    }
    if let Some(mtl) = patch_index_mtl {
        entry.retained_buffers.push(mtl);
    }
    if let Some(mtl) = control_point_index_mtl {
        entry.retained_buffers.push(mtl);
    }
    if let Some(mtl) = tess_factor_mtl {
        entry.retained_buffers.push(mtl);
    }
    entry.has_fills = true;
    Ok(())
}

/// Clone writeback slots for a cached ICB into a session nested job (after execute).
pub(crate) fn export_icb_writeback_job(
    task_id: u32,
    icb_ref: u32,
) -> Option<crate::runtime::compute_exec::metal::NestedDispatchJob> {
    use crate::runtime::compute_exec::metal::nested_job_from_icb_buffers;
    use crate::runtime::compute_exec::StagedBuffer;

    let cache = icb_cache().lock();
    let entry = cache.get(&(task_id, icb_ref))?;
    if entry.writebacks.is_empty() {
        return None;
    }
    let mut staged = Vec::with_capacity(entry.writebacks.len());
    let mut mtl = Vec::with_capacity(entry.writebacks.len());
    for w in &entry.writebacks {
        staged.push(StagedBuffer {
            bind: w.bind.clone(),
            gva: w.gva,
            bytes: vec![0u8; w.len],
            pages: w.pages.clone(),
        });
        mtl.push(w.mtl.clone());
    }
    Some(nested_job_from_icb_buffers(staged, mtl))
}

/// Build a compute PSO with `supportIndirectCommandBuffers` (required for ICB
/// fills and for parent-encoder `inheritPipelineState`).
pub(crate) fn new_icb_compute_pso(
    device: &::metal::Device,
    mtlb: &[u8],
) -> Result<::metal::ComputePipelineState, IcbStatus> {
    use ::metal::ComputePipelineDescriptor;

    // Load sole function from MTLB (same contract as product compute path).
    let library = device
        .new_library_with_data(mtlb)
        .map_err(|_| IcbStatus::MetalFailed("icb_pso_library_load"))?;
    let names = library.function_names();
    if names.len() != 1 {
        return Err(IcbStatus::Args("icb_pso_function_count"));
    }
    let function = library
        .get_function(&names[0], None)
        .map_err(|_| IcbStatus::MetalFailed("icb_pso_function_get"))?;
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&function));
    desc.set_support_indirect_command_buffers(true);
    device
        .new_compute_pipeline_state(&desc)
        .map_err(|_| IcbStatus::MetalFailed("icb_pso_pipeline_state"))
}

/// Fill one compute command slot on a cached host ICB from guest object-list state.
///
/// Mirrors Metal: `indirectComputeCommandAtIndex` → set PSO / kernel buffers /
/// concurrent dispatch. Stages buffer contents into shared Metal buffers
/// and records GVA writebacks for post-execute flush.
///
/// When the ICB was created with `inheritPipelineState` / `inheritBuffers`, those
/// resources are **not** recorded into the slot — the parent compute encoder
/// supplies them at `executeCommandsInBuffer` (see
/// [`crate::backend::compute_session::ComputeSession::encode_icb`]).
pub fn fill_compute_command<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    fill: &IcbComputeFill,
) -> Result<(), IcbStatus> {
    use crate::backend::metal::raw_metal::mtl_size;
    use crate::backend::metal::runtime::{new_buffer_from_host, system_device};
    use crate::runtime::compute_exec::metal::stage_buffer;
    use crate::runtime::compute_exec::{load_compute_pipeline, ComputeBufferBind};
    use crate::runtime::mtlb::{load_mtlb, AirLoadRail};

    if icb_ref == 0 {
        return Err(IcbStatus::Args("icb_fcc_ref_zero"));
    }
    let Some(device) = system_device() else {
        return Err(IcbStatus::NoMetal("icb_fcc_no_metal"));
    };

    // Ensure the host ICB exists in the cache; need create flags before staging.
    let (desc, _) = resolve_metal_icb(state, host, task_id, icb_ref)?;
    let mut fill_resolved = fill.clone();
    resolve_compute_fill_offsets(state, host, task_id, &mut fill_resolved)?;
    let fill = &fill_resolved;

    // Pipeline is required on the ICB command unless inheritPipelineState.
    let pso = if !desc.inherit_pipeline_state() {
        if fill.pipeline_ref == 0 {
            return Err(IcbStatus::Args("icb_fcc_pipeline_ref_zero"));
        }
        let pipeline = load_compute_pipeline(state, host, task_id, fill.pipeline_ref)
            .ok_or(IcbStatus::Missing("icb_fcc_pipeline_load"))?;
        let mtlb = load_mtlb(
            state,
            host,
            task_id,
            pipeline.kernel_func_ref,
            AirLoadRail::Compute,
        )
        .ok_or(IcbStatus::Missing("icb_fcc_mtlb_load"))?;
        Some(new_icb_compute_pso(device, &mtlb)?)
    } else {
        None
    };

    // Kernel buffers: stage only when not inheritBuffers (parent encoder owns them).
    let mut staged_binds: Vec<(
        u32,
        bool,
        u64,
        ::metal::Buffer,
        crate::runtime::compute_exec::StagedBuffer,
    )> = Vec::new();
    if !desc.inherit_buffers() {
        for b in &fill.buffers {
            if b.buffer_ref == 0 {
                return Err(IcbStatus::Args("icb_fcc_bind_ref_zero"));
            }
            let bind = ComputeBufferBind {
                index: b.index,
                buffer_ref: b.buffer_ref,
                offset: b.offset,
                attribute_stride: b.attribute_stride,
                has_attribute_stride: b.has_attribute_stride,
            };
            // The slug is dropped by this remap, not lost: `stage_buffer`
            // fail-logs the check that refused before returning, so the line is
            // already on the sink under `compute_stage_buf`. `IcbStatus` gets
            // its own vocabulary when it is registered.
            let staged = stage_buffer(state, host, task_id, &bind).map_err(|e| match e {
                crate::runtime::compute_exec::ComputeStatus::MissingBuffer(_) => {
                    IcbStatus::Missing("icb_fcc_bind_stage_missing")
                }
                crate::runtime::compute_exec::ComputeStatus::GuestIo(_) => {
                    IcbStatus::MetalFailed("icb_fcc_bind_stage_guest_io")
                }
                _ => IcbStatus::Args("icb_fcc_bind_stage_other"),
            })?;
            let mtl = new_buffer_from_host(device, staged.bytes.as_ptr(), staged.bytes.len())
                .ok_or(IcbStatus::MetalFailed("icb_fcc_bind_host_buffer"))?;
            staged_binds.push((
                b.index,
                b.has_attribute_stride,
                b.attribute_stride,
                mtl,
                staged,
            ));
        }
    }

    let mut cache = icb_cache().lock();
    let entry = cache
        .get_mut(&(task_id, icb_ref))
        .ok_or(IcbStatus::Missing("icb_fcc_not_cached"))?;
    if fill.command_index as u64 >= entry.icb.size() {
        return Err(IcbStatus::Args("icb_fcc_command_index_past_capacity"));
    }
    // maxKernelBufferBindCount: reject binds past the create descriptor.
    if !entry.desc.inherit_buffers() {
        for (idx, _, _, _, _) in &staged_binds {
            if *idx as u64 >= entry.desc.max_kernel_buffer_bind_count as u64 {
                return Err(IcbStatus::Args("icb_fcc_bind_index_past_max"));
            }
        }
    }

    let cmd = entry
        .icb
        .indirect_compute_command_at_index(fill.command_index as u64);
    if let Some(ref pso) = pso {
        cmd.set_compute_pipeline_state(pso);
    }
    // When inheritBuffers, kernel buffers come from the parent compute encoder.
    if !entry.desc.inherit_buffers() {
        for (idx, has_stride, stride, mtl, _) in &staged_binds {
            if *has_stride {
                crate::backend::metal::raw_metal::icb_set_kernel_buffer_attribute_stride(
                    cmd,
                    Some(mtl.as_ref()),
                    0,
                    *stride,
                    *idx as u64,
                );
            } else {
                cmd.set_kernel_buffer(*idx as u64, Some(mtl.as_ref()), 0);
            }
        }
    }
    for tg in &fill.threadgroup_memory {
        // Metal requires length multiple of 16 when non-zero; zero clears.
        if tg.length != 0 && tg.length % 16 != 0 {
            return Err(IcbStatus::Args("icb_fcc_tg_length_alignment"));
        }
        cmd.set_threadgroup_memory_length(tg.index as u64, tg.length);
    }
    if fill.barrier {
        cmd.set_barrier();
    } else {
        cmd.clear_barrier();
    }
    match fill.dispatch {
        IcbFillDispatch::ConcurrentThreadgroups {
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
        } => {
            if grid_x == 0 || grid_y == 0 || grid_z == 0 || tg_x == 0 || tg_y == 0 || tg_z == 0 {
                return Err(IcbStatus::Args("icb_fcc_threadgroups_zero_dims"));
            }
            cmd.concurrent_dispatch_threadgroups(
                mtl_size(grid_x as u64, grid_y as u64, grid_z as u64),
                mtl_size(tg_x as u64, tg_y as u64, tg_z as u64),
            );
        }
        IcbFillDispatch::ConcurrentThreads {
            threads_x,
            threads_y,
            threads_z,
            tg_x,
            tg_y,
            tg_z,
        } => {
            if threads_x == 0
                || threads_y == 0
                || threads_z == 0
                || tg_x == 0
                || tg_y == 0
                || tg_z == 0
            {
                return Err(IcbStatus::Args("icb_fcc_threads_zero_dims"));
            }
            cmd.concurrent_dispatch_threads(
                mtl_size(threads_x as u64, threads_y as u64, threads_z as u64),
                mtl_size(tg_x as u64, tg_y as u64, tg_z as u64),
            );
        }
    }

    if let Some(pso) = pso {
        entry.retained_psos.push(pso);
    }
    // Writebacks only for buffers recorded into the ICB (not inheritBuffers).
    for (_, _, _, mtl, staged) in staged_binds {
        entry.writebacks.push(IcbWriteback {
            bind: staged.bind.clone(),
            gva: staged.gva,
            len: staged.bytes.len(),
            pages: staged.pages.clone(),
            mtl: mtl.clone(),
        });
        entry.retained_buffers.push(mtl);
    }
    entry.has_fills = true;
    Ok(())
}
