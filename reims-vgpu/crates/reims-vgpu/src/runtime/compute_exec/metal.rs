//! The Metal rail's compute dispatch: staged binds into `MTLComputeCommandEncoder`.
//!
//! The sibling of [`super::vulkan`]. [`super`] resolves the guest's dispatch —
//! decodes the command, stages buffers and textures out of guest memory, and
//! decides what must be written back — and hands the result here; this module
//! is the half that names Metal ABI records, retains Metal objects for the life
//! of a command buffer, and drives the encoder.
//!
//! It also owns the deferred writeback for *nested* dispatches, which the Vulkan
//! rail has no form of: a dispatch encoded onto a parent session's encoder
//! cannot read its own output back until that session commits, so
//! [`NestedDispatchJob`] carries the staging until [`flush_nested_jobs`].

use super::*;

/// This rail's half of a staged compute texture. See [`RailStage`].
///
/// [`RailStage`]: crate::runtime::compute_exec::RailStage
#[derive(Debug, Default)]
pub(crate) struct MetalStage {
    /// The guest ref this was staged from. Carried so a refusal downstream can
    /// name the object the guest bound and not only the slot it bound it to.
    /// Read by this rail's format refusal; the Vulkan rail reaches its images by
    /// another route and never asks.
    pub(crate) texture_ref: u32,
}

impl RailStage for MetalStage {
    /// This rail keeps no residency mirror, so neither residency fact survives
    /// staging — see `super::vulkan::resident_serve` for the rail that does.
    fn stage(
        texture_ref: u32,
        _residency: Option<ComputeStorageResidencyCandidate>,
        _serve: Option<ResidentServe>,
    ) -> Self {
        Self { texture_ref }
    }
}

impl StagedTexture<MetalStage> {
    /// The storage-image selector for this texture's guest pixel format,
    /// or a named refusal.
    ///
    /// Sample-only formats such as `RGB9E5Float` have no selector by design, so
    /// this is a real class rather than an internal error — a guest binding one
    /// into a compute slot loses that bind, and the line has to say which
    /// object at which slot in which format.
    ///
    /// Three sites asked this one question and each carried its own answer:
    /// `reason=metal_selector_missing` twice and `reason=no_backend_selector`
    /// once, under two event names, returning three different refusal slugs,
    /// with one line carrying `ref`, another `storage` and the third neither.
    /// A grep for any of the three names found a third of the occurrences.
    pub(crate) fn storage_selector_or_refuse(
        &self,
        task_id: u32,
        pipeline_ref: u32,
    ) -> Result<pixel_format::StorageImageSelector, ComputeStatus> {
        self.storage_selector.ok_or_else(|| {
            crate::observe::fail(format!(
                "compute_texture_format fail reason=no_backend_selector task={task_id} \
                 pipe={pipeline_ref} bind={} ref={} fmt={:#x} storage={}",
                self.binding, self.rail.texture_ref, self.pixel_format, self.is_storage as u8
            ));
            ComputeStatus::Unsupported("compute_no_backend_selector")
        })
    }
}

/// Conservative whole-allocation staging used by the Metal-direct callers,
/// which do not translate the shader through the reflection-producing path.
pub(crate) fn stage_buffer<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    bind: &ComputeBufferBind,
) -> Result<StagedBuffer, ComputeStatus> {
    stage_buffer_with_extent(state, host, task_id, bind, None)
}

use crate::backend::metal::abi::{ReimsVgpuComputeSampledImage, ReimsVgpuStorageImage};

/// Split staged compute textures into the two ABI image lists Metal binds.
///
/// A storage-capable bind becomes a `ReimsVgpuStorageImage` the kernel writes
/// through; everything else becomes a sampled image. Both rails that reach the
/// direct-Metal encoder — the ICB session's inherited binds and the standalone
/// dispatch — carried a copy of this, byte-identical apart from the refusal
/// above.
#[allow(clippy::type_complexity)]
pub(crate) fn split_staged_textures(
    staged: &mut [StagedTexture<MetalStage>],
    task_id: u32,
    pipeline_ref: u32,
) -> Result<
    (
        Vec<ReimsVgpuStorageImage>,
        Vec<ReimsVgpuComputeSampledImage>,
    ),
    ComputeStatus,
> {
    let mut storage: Vec<ReimsVgpuStorageImage> = Vec::new();
    let mut sampled: Vec<ReimsVgpuComputeSampledImage> = Vec::new();
    for t in staged {
        let selector = t.storage_selector_or_refuse(task_id, pipeline_ref)?;
        if t.is_storage {
            storage.push(ReimsVgpuStorageImage {
                binding: t.binding,
                format: selector,
                width: t.width,
                height: t.height,
                data: t.bytes.as_mut_ptr(),
                len: t.bytes.len(),
            });
        } else {
            if t.mip_levels > 1 {
                // `ReimsVgpuComputeSampledImage` carries one level's texels, so
                // this rail would bind the base and answer every
                // `read(coord, lod)` above it with nothing. Refuse by name
                // rather than serve a pyramid flattened to its base.
                crate::observe::fail(format!(
                    "compute_stage_tex metal_fail reason=sampled_mip_levels task={task_id}                      pipe={pipeline_ref} bind={} levels={} {}x{}",
                    t.binding, t.mip_levels, t.width, t.height
                ));
                return Err(ComputeStatus::Unsupported("metal_sampled_mip_levels"));
            }
            sampled.push(ReimsVgpuComputeSampledImage::unswizzled(
                t.binding,
                selector,
                t.width,
                t.height,
                t.bytes.as_ptr(),
                t.bytes.len(),
            ));
        }
    }
    Ok((storage, sampled))
}

/// One nested dispatch's deferred writeback (GPU → host staging → GVA after session commit).
pub(crate) struct NestedDispatchJob {
    staged_bufs: Vec<StagedBuffer>,
    /// Storage textures only (sampled need no writeback).
    storage_tex: Vec<StagedTexture<MetalStage>>,
    mtl_buffers: Vec<::metal::Buffer>,
    mtl_storage: Vec<::metal::Texture>,
}

/// Build a deferred writeback job for ICB-filled kernel buffers (no storage textures).
pub(crate) fn nested_job_from_icb_buffers(
    staged_bufs: Vec<StagedBuffer>,
    mtl_buffers: Vec<::metal::Buffer>,
) -> NestedDispatchJob {
    nested_job_from_icb_resources(staged_bufs, mtl_buffers, Vec::new(), Vec::new())
}

/// Staged compute buffers as the C ABI records the Metal encoder reads.
///
/// The pointers borrow `staged`, so the returned vector must not outlive it.
/// `backing_*` stay null: a staged buffer owns its bytes, and only the
/// indirect-argument path fills a backing allocation in afterwards.
#[cfg(all(target_os = "macos", feature = "backend-metal"))]
fn abi_buffers(staged: &mut [StagedBuffer]) -> Vec<crate::backend::metal::abi::ReimsVgpuBuffer> {
    use crate::backend::metal::abi::ReimsVgpuBuffer;
    staged
        .iter_mut()
        .map(|s| ReimsVgpuBuffer {
            binding: s.bind.index,
            data: s.bytes.as_mut_ptr(),
            len: s.bytes.len(),
            attribute_stride: s.bind.attribute_stride,
            has_attribute_stride: u32::from(s.bind.has_attribute_stride),
            reserved0: 0,
            backing_data: std::ptr::null_mut(),
            backing_len: 0,
            backing_offset: 0,
        })
        .collect()
}

/// Deferred writeback for parent-encoder ICB inheritance (buffers + storage textures).
pub(crate) fn nested_job_from_icb_resources(
    staged_bufs: Vec<StagedBuffer>,
    mtl_buffers: Vec<::metal::Buffer>,
    storage_tex: Vec<StagedTexture<MetalStage>>,
    mtl_storage: Vec<::metal::Texture>,
) -> NestedDispatchJob {
    NestedDispatchJob {
        staged_bufs,
        storage_tex,
        mtl_buffers,
        mtl_storage,
    }
}

pub(crate) fn flush_nested_jobs<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    jobs: &mut [NestedDispatchJob],
) -> ComputeStatus {
    use crate::backend::metal::abi::ReimsVgpuStorageImage;
    use crate::backend::metal::compute::compute_writeback_from_mtl;

    let mut err_buf = [0i8; 256];
    for job in jobs.iter_mut() {
        let mut reims_vgpu_bufs = abi_buffers(&mut job.staged_bufs);
        let mut storage: Vec<ReimsVgpuStorageImage> = job
            .storage_tex
            .iter_mut()
            .map(|t| ReimsVgpuStorageImage {
                binding: t.binding,
                format: t
                    .storage_selector
                    .expect("storage texture staged with a storage selector"),
                width: t.width,
                height: t.height,
                data: t.bytes.as_mut_ptr(),
                len: t.bytes.len(),
            })
            .collect();
        let st = compute_writeback_from_mtl(
            &mut reims_vgpu_bufs,
            &job.mtl_buffers,
            &mut storage,
            &job.mtl_storage,
            (err_buf.as_mut_ptr(), err_buf.len()),
        );
        if !st.is_ok() {
            return ComputeStatus::MetalFailed("compute_nested_writeback_metal");
        }
        for s in &job.staged_bufs {
            if let Err(e) = writeback_buffer(state, host, task_id, None, "nested_flush", s) {
                return e;
            }
        }
        for t in &job.storage_tex {
            if let Err(e) = writeback_texture(state, host, task_id, t) {
                return e;
            }
        }
    }
    ComputeStatus::Ok
}

fn stage_input_to_apv(
    si: &ComputeStageInputDescriptor,
) -> crate::backend::metal::abi::ReimsVgpuComputeStageInputDescriptor {
    use crate::backend::metal::abi::{
        ReimsVgpuComputeStageInputAttribute, ReimsVgpuComputeStageInputDescriptor,
        ReimsVgpuComputeStageInputLayout, REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES,
        REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS,
    };
    let mut out = ReimsVgpuComputeStageInputDescriptor {
        word0: si.word0,
        header0: si.header0,
        header1: si.header1,
        attribute_count: si.attributes.len() as u32,
        layout_count: si.layouts.len() as u32,
        index_type: si.index_type,
        index_buffer_index: si.index_buffer_index,
        attributes: [ReimsVgpuComputeStageInputAttribute {
            raw_bits: 0,
            location: 0,
            format: 0,
            offset: 0,
            buffer_index: 0,
            reserved0: 0,
        }; REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES],
        layouts: [ReimsVgpuComputeStageInputLayout {
            raw_bits: 0,
            buffer_index: 0,
            step_function: 0,
            step_rate: 0,
            stride: 0,
        }; REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS],
    };
    for (i, a) in si
        .attributes
        .iter()
        .enumerate()
        .take(REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES)
    {
        out.attributes[i] = ReimsVgpuComputeStageInputAttribute {
            raw_bits: a.raw_bits,
            location: a.location,
            format: a.format,
            offset: a.offset,
            buffer_index: a.buffer_index,
            reserved0: 0,
        };
    }
    for (i, l) in si
        .layouts
        .iter()
        .enumerate()
        .take(REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS)
    {
        out.layouts[i] = ReimsVgpuComputeStageInputLayout {
            raw_bits: l.raw_bits,
            buffer_index: l.buffer_index,
            step_function: l.step_function,
            step_rate: l.step_rate,
            stride: l.stride,
        };
    }
    out
}

pub(crate) fn execute_dispatch_metal<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    dispatch: &DispatchRecord,
    session: Option<&mut crate::runtime::compute_session::metal::MetalSession>,
) -> ComputeStatus {
    use crate::backend::metal::abi::texture_binds_as_storage;
    use crate::backend::metal::abi::{
        ReimsVgpuComputeImageblockDimensions, ReimsVgpuComputeStageInRegion,
        ReimsVgpuComputeStageInRegionIndirectArguments, ReimsVgpuComputeTextureUsage,
        ReimsVgpuSampler, ReimsVgpuThreadgroupMemory, REIMS_VGPU_BINDING_SAMPLER_BASE,
        REIMS_VGPU_BINDING_TEXTURE_BASE,
    };
    use crate::backend::metal::compute::{
        compute_core, compute_encode_on_encoder, reflect_compute_textures_mtlb,
    };
    if acc.pipeline_ref == 0 {
        return ComputeStatus::MissingPipeline("compute_mtl_pipeline_ref_zero");
    }
    let Some(pipeline) = load_compute_pipeline(state, host, task_id, acc.pipeline_ref) else {
        return ComputeStatus::MissingPipeline("compute_mtl_pipeline_load");
    };
    let Some(mtlb) = load_mtlb(
        state,
        host,
        task_id,
        pipeline.kernel_func_ref,
        AirLoadRail::Compute,
    ) else {
        return ComputeStatus::MissingMtlb("compute_mtl_mtlb_load");
    };

    let DispatchDims {
        grid,
        threadgroup: tg,
        dispatch_threads,
    } = match resolve_dispatch_dims_reported(state, host, task_id, dispatch, acc) {
        Ok(v) => v,
        Err(e) => return e,
    };

    // No narrowing here, and none possible: the accumulator holds a
    // `DispatchType`, which `reims_vgpu_protocol` refused to build from a word
    // outside the enumeration. The ordinal is spelled once, by the type, on its
    // way across the ABI boundary.
    let dispatch_type = acc.dispatch_type.word();

    // Stage-input descriptor from pipeline (optional).
    let reims_vgpu_stage_input = pipeline.stage_input.as_ref().map(stage_input_to_apv);

    // Direct / indirect stage-in region.
    let direct_region = acc
        .stage_in_region
        .as_ref()
        .map(|r| ReimsVgpuComputeStageInRegion {
            origin_x: r.origin_x,
            origin_y: r.origin_y,
            origin_z: r.origin_z,
            size_x: r.size_x,
            size_y: r.size_y,
            size_z: r.size_z,
        });
    let mut indirect_region_args: Option<ReimsVgpuComputeStageInRegionIndirectArguments> = None;
    if let Some(ind) = &acc.stage_in_region_indirect {
        let raw = match read_buffer_window(
            state,
            host,
            task_id,
            ind.buffer_ref,
            ind.buffer_offset,
            STAGE_IN_INDIRECT_ARGS_LEN,
        ) {
            Ok(b) => b,
            Err(e) => return e,
        };
        indirect_region_args = Some(ReimsVgpuComputeStageInRegionIndirectArguments {
            origin_x: ld32(&raw[0..]),
            origin_y: ld32(&raw[4..]),
            origin_z: ld32(&raw[8..]),
            size_x: ld32(&raw[12..]),
            size_y: ld32(&raw[16..]),
            size_z: ld32(&raw[20..]),
        });
    }
    let imageblock = acc
        .imageblock
        .as_ref()
        .map(|d| ReimsVgpuComputeImageblockDimensions {
            width: d.width,
            height: d.height,
        });
    let tg_mem: Vec<ReimsVgpuThreadgroupMemory> = acc
        .threadgroup_memory
        .iter()
        .map(|t| ReimsVgpuThreadgroupMemory {
            index: t.index,
            length: t.length,
        })
        .collect();

    let mut staged_bufs: Vec<StagedBuffer> = Vec::new();
    for b in &acc.buffers {
        match stage_buffer_with_extent(state, host, task_id, b, None) {
            Ok(s) => staged_bufs.push(s),
            Err(e) => return e,
        }
    }

    // Texture reflection: access decides storage vs sampled materialization.
    // The reflection owns its own list — no caller-side capacity, so a kernel
    // declaring more bindings than some local buffer happened to hold is not a
    // refused dispatch.
    let mut err_buf = [0i8; 256];
    let usages: Vec<ReimsVgpuComputeTextureUsage> = if acc.textures.is_empty() {
        Vec::new()
    } else {
        match reflect_compute_textures_mtlb(&mtlb, (err_buf.as_mut_ptr(), err_buf.len())) {
            Ok(u) => u,
            Err(st) => return ComputeStatus::RailRefused(st),
        }
    };

    let mut staged_tex: Vec<StagedTexture<MetalStage>> = Vec::new();
    for t in &acc.textures {
        let binding = REIMS_VGPU_BINDING_TEXTURE_BASE + t.index;
        let is_storage = texture_binds_as_storage(&usages, binding);
        let stage_call_started = std::time::Instant::now();
        match stage_texture_raw(state, host, task_id, t.texture_ref, binding, is_storage) {
            Ok(s) => {
                // Measure-only: localize per-texture stage cost (the
                // transition-window guest stall).
                let us = stage_call_started.elapsed().as_micros() as u64;
                if us > 1500 {
                    crate::observe::off(format!(
                        "compute_stage_slow pipe={} ref={} bind={binding} storage={} {}x{} fmt={:#x} us={us}",
                        acc.pipeline_ref,
                        t.texture_ref,
                        is_storage as u8,
                        s.width,
                        s.height,
                        s.pixel_format
                    ));
                }
                staged_tex.push(s)
            }
            Err(e) => return e,
        }
    }

    // Samplers.
    let mut reims_vgpu_samplers: Vec<ReimsVgpuSampler> = Vec::new();
    for s in &acc.samplers {
        let sampler = match objects::resolve_sampler_state(state, host, task_id, s.sampler_ref) {
            Ok(sampler) => sampler,
            Err(objects::SamplerResolveError::Rung(rung)) => {
                return ComputeStatus::MissingSampler(crate::observe::ladder_slugs!(
                    "compute_mtl_sampler"
                )(rung))
            }
            Err(objects::SamplerResolveError::Decode { .. }) => {
                return ComputeStatus::MissingSampler(crate::observe::ladder_slug!(
                    "compute_mtl_sampler",
                    desc_decode
                ))
            }
        };
        reims_vgpu_samplers.push(crate::runtime::draw::metal::sampler_record(
            REIMS_VGPU_BINDING_SAMPLER_BASE + s.index,
            &sampler.descriptor,
            s.has_lod_clamp.then_some((s.lod_min_bits, s.lod_max_bits)),
            false,
        ));
    }

    let mut reims_vgpu_bufs = abi_buffers(&mut staged_bufs);

    // Keep raw pointers valid: build storage/sampled from staged_tex after mut split.
    let (mut storage, sampled) =
        match split_staged_textures(&mut staged_tex, task_id, acc.pipeline_ref) {
            Ok(split) => split,
            Err(e) => return e,
        };

    // Nested: encode onto open session encoder; writeback after segment commit.
    if let Some(sess) = session {
        let retain = match compute_encode_on_encoder(
            &sess.device,
            &sess.encoder,
            &mtlb,
            &mut reims_vgpu_bufs,
            &mut storage,
            &sampled,
            &reims_vgpu_samplers,
            &tg_mem,
            direct_region.as_ref(),
            indirect_region_args.as_ref(),
            imageblock.as_ref(),
            reims_vgpu_stage_input.as_ref(),
            dispatch_threads,
            grid,
            tg,
            (err_buf.as_mut_ptr(), err_buf.len()),
        ) {
            Ok(r) => r,
            Err(st) => return ComputeStatus::RailRefused(st),
        };
        // Split storage textures out of staged_tex for deferred writeback alignment.
        let storage_tex: Vec<StagedTexture<MetalStage>> =
            staged_tex.into_iter().filter(|t| t.is_storage).collect();
        if storage_tex.len() != retain.images.len() {
            return ComputeStatus::MetalFailed("compute_mtl_retain_image_count");
        }
        sess.retained.extend(retain.buffers.iter().cloned());
        sess.retained.extend(retain.indirect.iter().cloned());
        sess.nested_jobs.push(NestedDispatchJob {
            staged_bufs,
            storage_tex,
            mtl_buffers: retain.buffers,
            mtl_storage: retain.images,
        });
        return ComputeStatus::Ok;
    }

    let st = compute_core(
        &mtlb,
        &mut reims_vgpu_bufs,
        &mut storage,
        &sampled,
        &reims_vgpu_samplers,
        &tg_mem,
        direct_region.as_ref(),
        indirect_region_args.as_ref(),
        imageblock.as_ref(),
        reims_vgpu_stage_input.as_ref(),
        dispatch_threads,
        dispatch_type,
        grid,
        tg,
        (err_buf.as_mut_ptr(), err_buf.len()),
    );
    if !st.is_ok() {
        return ComputeStatus::RailRefused(st);
    }

    for s in &staged_bufs {
        if let Err(e) = writeback_buffer(
            state,
            host,
            task_id,
            Some(acc.pipeline_ref),
            "metal_dispatch",
            s,
        ) {
            return e;
        }
    }
    for t in &staged_tex {
        if let Err(e) = writeback_texture(state, host, task_id, t) {
            return e;
        }
    }
    ComputeStatus::Ok
}
