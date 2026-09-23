//! Product-path compute bind/dispatch for `SEGMENT_TYPE_COMPUTE`.
//!
//! Executable surface:
//! - `0xd0` set compute pipeline (serializer-object → kernel function MTLB + optional stage-input)
//! - `0xcb` / `0xd9` set buffers (+ optional attribute stride for dynamic stage-input layouts)
//! - `0xcf` / `0xda` set buffer offset (+ optional attribute stride)
//! - `0xce` set textures (normal-texture GVA + mapper-ref-texture; sample vs storage via reflection)
//! - `0xcc` / `0xcd` set samplers (+ optional LOD clamp)
//! - `0xd1` direct stage-in region / `0xd2` indirect stage-in region (guest buffer args)
//! - `0xd3` threadgroup memory length
//! - `0xd8` imageblock dimensions
//! - `0xc8`/`0xca` direct dispatch; `0xc9`/`0xe6` indirect (guest args → direct encode)
//! - `0xdb` dispatch type (serial/concurrent)
//!
//! Fences: stream walk (`fence_exec`). Control-flow (`0xdc`–`0xe2`) encodes
//! host Metal SPI on a multi-record [`crate::runtime::compute_session`] (same
//! encoder for the segment). ICB (`0xe4`/`0xe5`) materializes type-7 `0x36` and
//! executes filled host command slots (CPU fill via [`crate::runtime::icb`];
//! stream fill opcodes remain unknown). Nested dispatches on an open session
//! encode onto that encoder (inside SPI); writeback runs after session commit.
//! Barriers and compressed-texture flush are ordered no-ops.
//!
//! One-shot encode uses [`crate::backend::metal::compute::compute_core`]; nested
//! encode uses `compute_encode_on_encoder`. Buffer and storage-image writeback
//! is GVA / mapper-ref-texture staged.

// The backend the process executes on, reached only through the trait.
use crate::backend::Backend as _;
use crate::model::DeviceState;
use crate::protocol::endian::ld32;
use crate::protocol::pixel_format;
// This device's own decoder, kept for the rows the closure ledger has not
// settled: the fence pair, the seven control-flow records and the two indirect
// command executions. The seventeen it has settled arrive as a
// `reims_vgpu_protocol::decode::compute::ComputeRecord`.
use crate::runtime::decode::compute_spi::Command as ComputeCommand;
use crate::runtime::decode::resource::{
    decode_heap_texture, decode_serializer_object_descriptor, decode_texture_descriptor,
    texture_view_opcode, ComputeStageInputDescriptor, Descriptor as ResourceDescriptor,
    HEAP_TEXTURE_OPCODE, HEAP_TEXTURE_WIDE_OPCODE, OBJECT_TYPE_BUFFER,
    OBJECT_TYPE_SERIALIZER_OBJECT, OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS,
    OBJECT_TYPE_TEXTURE_VIEW, TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE,
    TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE,
};
use crate::runtime::draw::{host_alloc_len, StoreTargetPages};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapping_write;
use crate::runtime::mtlb::{load_mtlb, AirLoadRail};
use crate::runtime::objects;
use reims_vgpu_protocol::compute::DispatchType;
use reims_vgpu_protocol::decode::compute::{ComputeRecord, DispatchRecord, Extent as RecordExtent};

/// Cap on Metal compute buffer slots (matches backend `REIMS_VGPU_METAL_MAX_BUFFERS`).
pub const MAX_COMPUTE_BUFFER_SLOTS: u32 = 31;
/// Cap on compute texture stream indices (Metal bind = `TEXTURE_BINDING_BASE +
/// index`). Metal's compute texture argument table, and Apple's serializer's:
/// this rail refused indices 31..127 only because the descriptor binding band
/// was that narrow, which `spirv_bind::widen_sampled_bands` fixed.
pub const MAX_COMPUTE_TEXTURE_SLOTS: u32 = 128;
/// Cap on compute sampler stream indices (Metal bind = `SAMPLER_BINDING_BASE +
/// index`). Metal's sampler argument table, which is genuinely 16.
pub const MAX_COMPUTE_SAMPLER_SLOTS: u32 = 16;

// The two caps above are what keeps a stream index inside its own descriptor
// band: this rail binds a texture at `TEXTURE_BINDING_BASE + index` and a
// sampler at `SAMPLER_BINDING_BASE + index`, so a cap that let an index reach
// the next base would make a texture resolve against a sampler's reflection
// entry — and `reflected_compute_texture` would answer `Absent` for it, which
// this rail treats as "the shader does not use this binding" and skips. A
// silent drop, from two constants that never name each other.
//
// `backend::metal::constants` states the same relation for the Metal argument
// tables, in the same form and for the same reason; this side had the caps and
// the bands in two modules with nothing between them.
const _: () = assert!(
    crate::runtime::spirv_bind::TEXTURE_BINDING_BASE + MAX_COMPUTE_TEXTURE_SLOTS
        <= crate::runtime::spirv_bind::SAMPLER_BINDING_BASE
);
const _: () = assert!(
    crate::runtime::spirv_bind::SAMPLER_BINDING_BASE + MAX_COMPUTE_SAMPLER_SLOTS
        <= crate::runtime::spirv_bind::COLOR_INPUT_BINDING_BASE
);

// The three caps above hold the same three measured numbers as
// `reims_vgpu_wire::ops::bind_limit`, and until this gate nothing compared them.
// `bind_limit`'s own module doc says the truncation "is a property of the
// stage's argument table, not of an encoder" and names
// `compute_set_textures_over_bind_limit`, `compute_set_buffers_over_bind_limit`
// and `compute_set_samplers_over_bind_limit` as the captures it was read from —
// so these are compute-rail measurements, not render ones borrowed.
//
// Only one direction is a bug. A cap **below** Apple's table is guest work this
// device refuses: `ComputeBindOverflow` reports it, but a dispatch still runs
// missing that bind, and the render rail already carries the identical gate
// (`exec::apply_binds`' three `const` assertions) for the identical fact. A cap
// **above** it is headroom, which costs nothing and is why this is `<=` rather
// than the render rail's `==` — the other direction, a slot this device accepts
// but cannot name in the descriptor band, is what the two assertions directly
// above already refuse.
//
// A drift here would otherwise surface only as dropped compute binds on a live
// guest, with correct-looking output everywhere the kernel happened not to read
// the missing slot.
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::BUFFER <= MAX_COMPUTE_BUFFER_SLOTS);
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::TEXTURE <= MAX_COMPUTE_TEXTURE_SLOTS);
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::SAMPLER <= MAX_COMPUTE_SAMPLER_SLOTS);
/// `MTLDispatchThreadgroupsIndirectArguments` = three `uint32_t` (12 bytes).
pub const INDIRECT_THREADGROUPS_ARGS_LEN: usize = 12;
/// `MTLDispatchThreadsIndirectArguments` = six `uint32_t` (24 bytes).
pub const INDIRECT_THREADS_ARGS_LEN: usize = 24;
/// `MTLStageInRegionIndirectArguments` = six `uint32_t` (24 bytes).
pub const STAGE_IN_INDIRECT_ARGS_LEN: usize = 24;

/// A compute resource bind dropped because its slot index exceeds the
/// argument-table cap.
///
/// The guest bound a real resource (`ref != 0`, or a non-empty threadgroup
/// allocation) at a slot this device cannot represent, so the dispatch runs
/// *missing that bind* — wrong compute output with no other symptom.
///
/// The cap comparison is exclusive (`index >= MAX_*`) to match the backend,
/// which sizes its argument-table arrays to exactly these counts
/// (`[false; REIMS_VGPU_METAL_MAX_BUFFERS]`) and guards
/// `idx >= REIMS_VGPU_METAL_MAX_*` before indexing — so slot `MAX` is out of
/// range and a bind there is a genuine drop, not a boundary the accum should
/// have accepted.
///
/// # It is a `Decline` rather than a `format!`, and that is the point
///
/// This was a hand-rolled line: `observe::fail(format!(…))` behind a private
/// `Mutex<HashSet<(table, index)>>`. Both halves were a second spelling of
/// something the crate already owns — `Emit::fail_once` latches on
/// `(slug, discriminant)` in one process-global set, which is the same dedup
/// with the same shape.
///
/// Keeping a private one had a cost beyond the duplication. The four slugs
/// below lived inside a format string, where nobody looking for this crate's
/// decline vocabulary would find them: a future decline spelling
/// `sampler_index_overflow` would have shared this path's latch and silenced
/// one of the two for the life of the boot, and nothing would have failed. They
/// are `slug()` bodies now for that reason.
///
/// The rendered line is unchanged but for the trailing parenthetical, which the
/// `k=v` shape has no room for and which this doc now carries instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComputeBindOverflow {
    Buffer { index: u32, arg: u32, cap: u32 },
    Texture { index: u32, arg: u32, cap: u32 },
    Sampler { index: u32, arg: u32, cap: u32 },
}

impl ComputeBindOverflow {
    fn parts(&self) -> (u32, u32, u32) {
        match *self {
            Self::Buffer { index, arg, cap }
            | Self::Texture { index, arg, cap }
            | Self::Sampler { index, arg, cap } => (index, arg, cap),
        }
    }

    /// Emit on the fail channel, once per `(table, slot)` this boot.
    ///
    /// Runs on the drain worker (off the QEMU main core). The latch is what
    /// keeps a repeating dispatch from flooding; a healthy guest — one binding
    /// within the Metal argument-table caps — never reaches here at all.
    fn emit(self) {
        let (index, ..) = self.parts();
        crate::observe::Emit::decline("compute_bind_overflow", &self).fail_once(u64::from(index));
    }
}

impl crate::observe::Decline for ComputeBindOverflow {
    fn slug(&self) -> &'static str {
        match self {
            Self::Buffer { .. } => "buffer_index_overflow",
            Self::Texture { .. } => "texture_index_overflow",
            Self::Sampler { .. } => "sampler_index_overflow",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let (index, arg, cap) = self.parts();
        vec![
            ("index", index.to_string()),
            ("arg", arg.to_string()),
            ("cap", cap.to_string()),
        ]
    }
}

/// One buffer bind entry, whichever record carried it.
///
/// Five records bind into this encoder's argument tables and only two shapes of
/// buffer entry reach them: `setBuffers:offsets:withRange:` writes a ref and an
/// offset, and `setBuffers:offsets:attributeStrides:withRange:` writes a stride
/// beside them. What turns an entry into a slot is the same either way, so it
/// is stated once and the entry types answer for their own layout.
///
/// **A stride is `Option`, and that is the point.** The device decoder used to
/// hand the accumulator a flat entry with an `attribute_stride` and a
/// `has_attribute_stride` beside it, so "the record carried no stride field"
/// and "the record carried a stride of zero" were two spellings of the same
/// bytes and only the flag told them apart. Here the record with no stride
/// field cannot produce a `Some`.
pub trait BufferBindEntry {
    fn buffer_ref(&self) -> u32;
    fn offset(&self) -> u64;
    fn stride(&self) -> Option<u64>;
}

/// One texture or sampler bind entry, whichever record carried it.
///
/// The plain form is a bare ref; `setSamplers:lodMinClamps:lodMaxClamps:` adds
/// a clamp pair. The clamps are carried as bit patterns because that is what
/// this device binds with — the wire's `f32` becomes bits once, here, rather
/// than at each of the two producers.
pub trait ObjectBindEntry {
    fn object_ref(&self) -> u32;
    fn lod_clamp(&self) -> Option<(u32, u32)> {
        None
    }
}

impl BufferBindEntry for reims_vgpu_wire::ops::render::BufferBind {
    fn buffer_ref(&self) -> u32 {
        self.buffer_ref.get()
    }
    fn offset(&self) -> u64 {
        self.offset.get()
    }
    fn stride(&self) -> Option<u64> {
        None
    }
}

impl BufferBindEntry for reims_vgpu_wire::ops::render::BufferStrideBind {
    fn buffer_ref(&self) -> u32 {
        self.buffer_ref.get()
    }
    fn offset(&self) -> u64 {
        self.offset.get()
    }
    fn stride(&self) -> Option<u64> {
        Some(self.attribute_stride.get())
    }
}

impl ObjectBindEntry for reims_vgpu_wire::ops::render::RefBind {
    fn object_ref(&self) -> u32 {
        self.object_ref.get()
    }
}

impl ObjectBindEntry for reims_vgpu_wire::ops::render::SamplerLodBind {
    fn object_ref(&self) -> u32 {
        self.sampler_ref.get()
    }
    fn lod_clamp(&self) -> Option<(u32, u32)> {
        Some((
            self.lod_min_clamp.get().to_bits(),
            self.lod_max_clamp.get().to_bits(),
        ))
    }
}

#[derive(Clone, Debug, Default)]
pub struct ComputeBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    pub offset: u64,
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeTextureBind {
    /// Stream texture index (`0xce first + i`); Metal bind = 32 + index.
    pub index: u32,
    pub texture_ref: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeSamplerBind {
    /// Stream sampler index; Metal bind = 64 + index.
    pub index: u32,
    pub sampler_ref: u32,
    pub lod_min_bits: u32,
    pub lod_max_bits: u32,
    pub has_lod_clamp: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ThreadgroupMemoryBind {
    pub index: u32,
    pub length: u64,
}

#[derive(Clone, Debug, Default)]
pub struct StageInRegion {
    pub origin_x: u64,
    pub origin_y: u64,
    pub origin_z: u64,
    pub size_x: u64,
    pub size_y: u64,
    pub size_z: u64,
}

#[derive(Clone, Debug, Default)]
pub struct StageInRegionIndirect {
    pub buffer_ref: u32,
    pub buffer_offset: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ImageblockDimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeAccum {
    pub pipeline_ref: u32,
    pub buffers: Vec<ComputeBufferBind>,
    pub textures: Vec<ComputeTextureBind>,
    pub samplers: Vec<ComputeSamplerBind>,
    pub threadgroup_memory: Vec<ThreadgroupMemoryBind>,
    /// Last direct `0xd1` stage-in region (cleared by `0xd2`).
    pub stage_in_region: Option<StageInRegion>,
    /// Last `0xd2` indirect stage-in (clears direct region).
    pub stage_in_region_indirect: Option<StageInRegionIndirect>,
    /// Last `0xd8` imageblock dimensions.
    pub imageblock: Option<ImageblockDimensions>,
    /// The pass's dispatch type, as `writeDescriptor` last stated it.
    ///
    /// A total type rather than the ordinal: `reims_vgpu_protocol` refuses a
    /// word outside `MTLDispatchType` at the lift, so the accumulator cannot
    /// hold a value no encoder can be opened with. It used to hold the raw
    /// `u32`, and the narrowing sat at the far end of the rail inside
    /// `execute_dispatch_metal` as an `if == CONCURRENT` — which silently read
    /// every unrecognised value as `Serial` on the one arm that looked, and on
    /// the Vulkan arm was stored and read by nobody.
    pub dispatch_type: DispatchType,
    /// A bind this accumulator could not hold, and so did not record.
    ///
    /// The three bind walks skip an index past their argument table — there is
    /// no slot to put it in — and that used to be the whole of it: the walk
    /// `continue`d and the dispatch went ahead with the guest's binding simply
    /// absent, which is a wrong result rather than a refused one. Nothing
    /// downstream refuses on a missing binding, because a kernel that does not
    /// sample the slot is indistinguishable from one whose bind landed.
    ///
    /// Recording it here is what lets [`resolve_dispatch_dims_reported`] — the
    /// one gate both dispatch executors pass through — refuse instead. Sticky
    /// for the accumulator's life on purpose: the binding stays unrepresentable
    /// until the guest clears that slot, and every dispatch in between would
    /// run without it.
    pub(crate) refused_bind: Option<ComputeBindOverflow>,
}

impl ComputeAccum {
    pub fn set_pipeline(&mut self, pipeline_ref: u32) {
        if pipeline_ref != 0 {
            self.pipeline_ref = pipeline_ref;
        }
    }

    /// Retire a recorded refusal the guest has just cleared.
    ///
    /// A nil bind at the slot that overflowed says the guest no longer wants
    /// anything there, so what this accumulator holds is once again what the
    /// guest asked for and the dispatch is representable again. Without this
    /// the sticky refusal would outlive the condition that caused it and refuse
    /// every later dispatch in the encoder over a slot nobody is binding — a
    /// remembered refusal gone stale, which is a class this tree already has a
    /// scan for.
    ///
    /// Matched on the index alone. The three tables are disjoint slot spaces so
    /// a clear could in principle name another class's slot, but only one
    /// refusal is ever held and it carries the class it came from, so the pair
    /// cannot be misread.
    fn clear_refusal_at(&mut self, index: u32) {
        if self.refused_bind.is_some_and(|r| r.parts().0 == index) {
            self.refused_bind = None;
        }
    }

    pub fn bind_buffers<E: BufferBindEntry>(&mut self, first: u32, entries: &[E]) {
        for (i, e) in entries.iter().enumerate() {
            let index = first.saturating_add(i as u32);
            let entry_ref = e.buffer_ref();
            if entry_ref == 0 {
                // A nil entry clears the slot. Retaining the previous bind
                // instead is not a stale read but a write: the retained buffer
                // is staged again on the next dispatch, and reflection calling
                // it writable sends the dispatch's output back into a guest
                // resource the guest explicitly unbound. Same rule the render
                // rail states on `ExecResult::buffer_unbinds` and applies in
                // `exec::apply_binds`, over the same wire form.
                self.buffers.retain(|b| b.index != index);
                self.clear_refusal_at(index);
                crate::runtime::drain::note_store_route("compute_unbind_buffer");
                continue;
            }
            if index >= MAX_COMPUTE_BUFFER_SLOTS {
                let over = ComputeBindOverflow::Buffer {
                    index,
                    arg: entry_ref,
                    cap: MAX_COMPUTE_BUFFER_SLOTS,
                };
                over.emit();
                self.refused_bind.get_or_insert(over);
                continue;
            }
            let stride = e.stride();
            let bind = ComputeBufferBind {
                index,
                buffer_ref: entry_ref,
                offset: e.offset(),
                attribute_stride: stride.unwrap_or_default(),
                has_attribute_stride: stride.is_some(),
            };
            if let Some(slot) = self.buffers.iter_mut().find(|b| b.index == index) {
                *slot = bind;
            } else {
                self.buffers.push(bind);
            }
        }
    }

    pub fn set_buffer_offset(&mut self, index: u32, offset: u64, attribute_stride: Option<u64>) {
        if let Some(slot) = self.buffers.iter_mut().find(|b| b.index == index) {
            slot.offset = offset;
            if let Some(s) = attribute_stride {
                slot.attribute_stride = s;
                slot.has_attribute_stride = true;
            }
        }
    }

    pub fn bind_textures<E: ObjectBindEntry>(&mut self, first: u32, entries: &[E]) {
        for (i, e) in entries.iter().enumerate() {
            let index = first.saturating_add(i as u32);
            let entry_ref = e.object_ref();
            if entry_ref == 0 {
                // Clears the slot; see `bind_buffers`. A retained texture is
                // the sharper case of the two, because `writeback_texture`
                // lands the dispatch's result in the guest surface behind it.
                self.textures.retain(|t| t.index != index);
                self.clear_refusal_at(index);
                crate::runtime::drain::note_store_route("compute_unbind_texture");
                continue;
            }
            if index >= MAX_COMPUTE_TEXTURE_SLOTS {
                let over = ComputeBindOverflow::Texture {
                    index,
                    arg: entry_ref,
                    cap: MAX_COMPUTE_TEXTURE_SLOTS,
                };
                over.emit();
                self.refused_bind.get_or_insert(over);
                continue;
            }
            let bind = ComputeTextureBind {
                index,
                texture_ref: entry_ref,
            };
            if let Some(slot) = self.textures.iter_mut().find(|t| t.index == index) {
                *slot = bind;
            } else {
                self.textures.push(bind);
            }
        }
    }

    pub fn bind_samplers<E: ObjectBindEntry>(&mut self, first: u32, entries: &[E]) {
        for (i, e) in entries.iter().enumerate() {
            let index = first.saturating_add(i as u32);
            let entry_ref = e.object_ref();
            if entry_ref == 0 {
                // Clears the slot; see `bind_buffers`.
                self.samplers.retain(|s| s.index != index);
                self.clear_refusal_at(index);
                crate::runtime::drain::note_store_route("compute_unbind_sampler");
                continue;
            }
            if index >= MAX_COMPUTE_SAMPLER_SLOTS {
                let over = ComputeBindOverflow::Sampler {
                    index,
                    arg: entry_ref,
                    cap: MAX_COMPUTE_SAMPLER_SLOTS,
                };
                over.emit();
                self.refused_bind.get_or_insert(over);
                continue;
            }
            let clamp = e.lod_clamp();
            let bind = ComputeSamplerBind {
                index,
                sampler_ref: entry_ref,
                lod_min_bits: clamp.map_or(0, |(min, _)| min),
                lod_max_bits: clamp.map_or(0, |(_, max)| max),
                has_lod_clamp: clamp.is_some(),
            };
            if let Some(slot) = self.samplers.iter_mut().find(|s| s.index == index) {
                *slot = bind;
            } else {
                self.samplers.push(bind);
            }
        }
    }

    /// Record a `setThreadgroupMemoryLength:atIndex:` for the next dispatch.
    ///
    /// **No bound here, on purpose.** The three bind setters above each refuse a
    /// slot past a cap because the protocol states one — the guest's serializer
    /// truncates a plural bind at exactly those counts, so a record naming a
    /// higher slot cannot have come from a well-formed guest. This record is
    /// singular, carries a full `u32`, and the guest applies no bound to it, so
    /// there is no protocol cap to compare against.
    ///
    /// What does bound it is the *host's* argument table, and only one backend
    /// has one: `backend::metal::compute::bind_threadgroup_memory` refuses at
    /// `REIMS_VGPU_METAL_MAX_THREADGROUP_MEMORY` and names the check. The Vulkan
    /// rail consumes none of these binds — SPIR-V declares workgroup shared
    /// memory statically — so a cap applied here would have taken slots away
    /// from an arm that has no table to run out of. That is the mistake
    /// [`crate::runtime::draw::MAX_SAMPLER_BIND_SLOTS`]' doc names, and a cap of
    /// 16 sat here making it until the host table size was known.
    pub fn set_threadgroup_memory(&mut self, index: u32, length: u64) {
        let bind = ThreadgroupMemoryBind { index, length };
        if let Some(slot) = self
            .threadgroup_memory
            .iter_mut()
            .find(|t| t.index == index)
        {
            *slot = bind;
        } else {
            self.threadgroup_memory.push(bind);
        }
    }

    pub fn set_stage_in_region(&mut self, region: StageInRegion) {
        self.stage_in_region_indirect = None;
        self.stage_in_region = Some(region);
    }

    pub fn set_stage_in_region_indirect(&mut self, buffer_ref: u32, buffer_offset: u64) {
        if buffer_ref == 0 {
            return;
        }
        self.stage_in_region = None;
        self.stage_in_region_indirect = Some(StageInRegionIndirect {
            buffer_ref,
            buffer_offset,
        });
    }

    pub fn set_imageblock(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.imageblock = Some(ImageblockDimensions { width, height });
    }
}

/// The compute rail's refusal vocabulary.
///
/// Every refusing variant carries the **registered slug of the check that
/// refused**, not just its class. Before that payload existed, nine of these
/// variants were payload-free and 129 construction sites collapsed into them —
/// `MetalFailed` alone spoke for 38 checks, `MissingTexture` for 25 — so a live
/// `compute_dispatches_fail` counter told you a dispatch died and nothing else.
/// The slug is what makes the class greppable; the class is what decides the
/// caller's recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeStatus {
    Ok,
    /// A rail refused with structure: the class of the failure, the registered
    /// slug of the check, and the facts that check was looking at. Neutral and
    /// ungated — see [`crate::backend::refusal::RailRefusal`] — because a
    /// variant that named one rail gave this enum two shapes across a feature
    /// boundary and left the other rail no way to refuse with structure.
    RailRefused(crate::backend::refusal::RailRefusal),
    MissingPipeline(&'static str),
    MissingMtlb(&'static str),
    MissingBuffer(&'static str),
    MissingTexture(&'static str),
    MissingSampler(&'static str),
    BadGrid(&'static str),
    GuestIo(&'static str),
    MetalFailed(&'static str),
    NoMetal(&'static str),
    Unsupported(&'static str),
}

impl crate::observe::Refusal for ComputeStatus {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            // The only non-refusal. Keeping it in the same enum is what makes
            // `Emit::refusal` unable to log a success by accident.
            Self::Ok => None,
            Self::RailRefused(refusal) => refusal.refusal(),
            Self::MissingPipeline(slug)
            | Self::MissingMtlb(slug)
            | Self::MissingBuffer(slug)
            | Self::MissingTexture(slug)
            | Self::MissingSampler(slug)
            | Self::BadGrid(slug)
            | Self::GuestIo(slug)
            | Self::MetalFailed(slug)
            | Self::NoMetal(slug)
            | Self::Unsupported(slug) => Some(slug),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        // The class next to the reason: `MissingTexture` vs `MetalFailed` is
        // what the caller acted on, and a reader correlating a log line with a
        // recovery path needs both.
        if let Self::RailRefused(refusal) = self {
            let mut fields = crate::observe::Refusal::fields(refusal);
            fields.push(("recovery", "metal_failed".to_string()));
            return fields;
        }
        vec![("class", self.class().to_string())]
    }
}

impl ComputeStatus {
    /// The variant name, for the `class=` field and for call sites that render
    /// their own line.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            // The two names the boot logs have carried since these were
            // `MetalBackend`. They are the *refusal's* class, not the rail's,
            // and stay spelled this way so a longitudinal grep still finds
            // them.
            Self::RailRefused(refusal) => {
                if refusal.is_args() {
                    "metal_args"
                } else {
                    "metal_execute"
                }
            }
            Self::MissingPipeline(_) => "missing_pipeline",
            Self::MissingMtlb(_) => "missing_mtlb",
            Self::MissingBuffer(_) => "missing_buffer",
            Self::MissingTexture(_) => "missing_texture",
            Self::MissingSampler(_) => "missing_sampler",
            Self::BadGrid(_) => "bad_grid",
            Self::GuestIo(_) => "guest_io",
            Self::MetalFailed(_) => "metal_failed",
            Self::NoMetal(_) => "no_metal",
            Self::Unsupported(_) => "unsupported",
        }
    }

    /// The registered slug this status carries, or `"ok"` when it is not a
    /// refusal. For sites that render a `reason=` into a longer line of their
    /// own rather than building one with [`crate::observe::Emit`].
    pub fn reason(&self) -> &'static str {
        use crate::observe::Refusal as _;
        self.refusal().unwrap_or("ok")
    }
}

/// A malformed translated kernel module before descriptor reflection/execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeSpirvDecline {
    HeaderTooShort { len: usize, minimum: usize },
    LengthMisaligned { len: usize, alignment: usize },
}

impl crate::observe::Decline for ComputeSpirvDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::HeaderTooShort { .. } => "compute_spirv_header_too_short",
            Self::LengthMisaligned { .. } => "compute_spirv_length_misaligned",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::HeaderTooShort { len, minimum } => {
                vec![("len", len.to_string()), ("minimum", minimum.to_string())]
            }
            Self::LengthMisaligned { len, alignment } => vec![
                ("len", len.to_string()),
                ("alignment", alignment.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(ComputeSpirvDecline);

impl std::error::Error for ComputeSpirvDecline {}

/// A reflected kernel resource whose Vulkan ABI this runtime cannot yet
/// populate. Kept separate from malformed SPIR-V: the translation is valid,
/// but executing it without decoding the owner argument buffer would bind the
/// wrong resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeReflectionDecline {
    ReflectedResourceUnsupported {
        pipeline_ref: u32,
        index: u32,
        binding: Option<u32>,
        kind: &'static str,
    },
    ReflectedInterfaceUnsupported {
        pipeline_ref: u32,
        feature: &'static str,
        count: usize,
    },
    /// The reflected exact-thread dispatch names a push-constant offset whose
    /// payload would not fit the range the translator publishes. Refused rather
    /// than clamped: a truncated range is a shader reading bytes no one wrote.
    DispatchPushRangeUnavailable { pipeline_ref: u32 },
    /// The translator refused to plan this launch's regions, so the dispatch
    /// does not reach the device. Its own text rides the emitter's `detail`
    /// field rather than the reason, which stays a stable slug.
    DispatchPlanRefused { pipeline_ref: u32 },
}

impl crate::observe::Decline for ComputeReflectionDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ReflectedResourceUnsupported { .. } => "compute_reflection_resource_unsupported",
            Self::ReflectedInterfaceUnsupported { .. } => {
                "compute_reflection_interface_unsupported"
            }
            Self::DispatchPushRangeUnavailable { .. } => {
                "compute_reflection_dispatch_push_range_unavailable"
            }
            Self::DispatchPlanRefused { .. } => "compute_reflection_dispatch_plan_refused",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ReflectedResourceUnsupported {
                pipeline_ref,
                index,
                binding,
                kind,
            } => vec![
                ("pipeline_ref", pipeline_ref.to_string()),
                ("index", index.to_string()),
                (
                    "binding",
                    binding.map_or_else(|| "none".to_string(), |value| value.to_string()),
                ),
                ("kind", (*kind).to_string()),
            ],
            Self::ReflectedInterfaceUnsupported {
                pipeline_ref,
                feature,
                count,
            } => vec![
                ("pipeline_ref", pipeline_ref.to_string()),
                ("feature", (*feature).to_string()),
                ("count", count.to_string()),
            ],
            Self::DispatchPushRangeUnavailable { pipeline_ref } => {
                vec![("pipeline_ref", pipeline_ref.to_string())]
            }
            Self::DispatchPlanRefused { pipeline_ref } => {
                vec![("pipeline_ref", pipeline_ref.to_string())]
            }
        }
    }
}

crate::observe::decline_display!(ComputeReflectionDecline);

impl std::error::Error for ComputeReflectionDecline {}

/// Apply one decoded compute command to accum, or run a dispatch / sequencing op.
///
/// `seg` carries the whole segment's mutable state: the accum this record
/// updates, the multi-record encoder a dispatch encodes onto when one is open,
/// and the latched sequencing failure (ICB / control encode error) that refuses
/// later dispatches.
pub fn apply_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    record: &ComputeRecord<'_>,
    seg: &mut crate::runtime::compute_session::ComputeSegment,
) -> Option<ComputeStatus> {
    let started = std::time::Instant::now();
    let out = apply_record_inner(state, host, task_id, record, seg);
    crate::runtime::drain::note_drain_phase(crate::runtime::drain::DrainPhase::Compute, started);
    out
}

/// Apply one sequencing record the closure ledger has not settled.
///
/// The control-flow SPI and the two indirect-command executions open and drive
/// the segment's multi-record encoder, and this device decodes them itself
/// because the ledger has no contract for them to lift against. They are kept
/// out of [`apply_record`] rather than folded into it: a settled record and an
/// unsettled one are different claims, and a single entry point taking both
/// would be the place a reader stops being able to tell which is which.
pub fn apply_sequencing_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &ComputeCommand,
    seg: &mut crate::runtime::compute_session::ComputeSegment,
) -> ComputeStatus {
    let started = std::time::Instant::now();
    let out = crate::runtime::compute_session::apply_sequencing(state, host, task_id, cmd, seg);
    crate::runtime::drain::note_drain_phase(crate::runtime::drain::DrainPhase::Compute, started);
    out
}

fn apply_record_inner<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    record: &ComputeRecord<'_>,
    seg: &mut crate::runtime::compute_session::ComputeSegment,
) -> Option<ComputeStatus> {
    match record {
        ComputeRecord::SetPipeline(r) => {
            seg.acc.set_pipeline(r.pipeline_ref);
            None
        }
        // The two plural buffer binds reach one setter because the accumulator
        // owns what an entry becomes; the stride's presence rides in the entry
        // type rather than in a flag beside it.
        ComputeRecord::BindBuffers(r) => {
            seg.acc.bind_buffers(r.first, r.entries);
            None
        }
        ComputeRecord::BindBuffersWithStride(r) => {
            seg.acc.bind_buffers(r.first, r.entries);
            None
        }
        // One arm for both rebind forms: `stride` is `None` on the record that
        // carries no stride field, which is exactly the argument the setter
        // takes.
        ComputeRecord::RebindBufferOffset(r) => {
            seg.acc.set_buffer_offset(r.index, r.offset, r.stride);
            None
        }
        ComputeRecord::BindTextures(r) => {
            seg.acc.bind_textures(r.first, r.entries);
            None
        }
        ComputeRecord::BindSamplers(r) => {
            seg.acc.bind_samplers(r.first, r.entries);
            None
        }
        ComputeRecord::BindSamplersWithLod(r) => {
            seg.acc.bind_samplers(r.first, r.entries);
            None
        }
        ComputeRecord::WriteDescriptor(r) => {
            seg.acc.dispatch_type = r.dispatch_type;
            None
        }
        ComputeRecord::SetStageInRegion(r) => {
            seg.acc.set_stage_in_region(StageInRegion {
                origin_x: r.origin.x,
                origin_y: r.origin.y,
                origin_z: r.origin.z,
                size_x: r.size.width,
                size_y: r.size.height,
                size_z: r.size.depth,
            });
            None
        }
        ComputeRecord::SetStageInRegionIndirect(r) => {
            seg.acc
                .set_stage_in_region_indirect(r.source.buffer_ref, r.source.offset);
            None
        }
        ComputeRecord::SetThreadgroupMemory(r) => {
            seg.acc.set_threadgroup_memory(r.index, r.length);
            None
        }
        ComputeRecord::SetImageblockSize(r) => {
            seg.acc.set_imageblock(r.width, r.height);
            None
        }
        ComputeRecord::Dispatch(dispatch) => {
            if seg.block.is_some() {
                return Some(ComputeStatus::Unsupported("dispatch_in_sequencing_block"));
            }
            // Open multi-record session (control-flow SPI): encode on that encoder.
            if let Some(sess) = seg.session.as_mut() {
                return Some(
                    crate::backend::selected()
                        .execute_dispatch_nested(state, host, task_id, &seg.acc, dispatch, sess),
                );
            }
            Some(
                crate::backend::selected()
                    .execute_dispatch(state, host, task_id, &seg.acc, dispatch),
            )
        }
    }
}

pub(crate) struct LoadedComputePipeline {
    pub kernel_func_ref: u32,
    /// Product-ready stage-input. `None` means the descriptor declared none —
    /// and only that. A descriptor whose entries exceeded the decoder's caps
    /// refuses the pipeline (`stage_input_over_cap`) rather than landing here as
    /// `None`, because the two are different guest programs.
    pub stage_input: Option<ComputeStageInputDescriptor>,
}

/// What a serializer-object's stage-input block means for the pipeline carrying it.
///
/// Three outcomes, and the whole point of naming them is that two of them are
/// not the same: [`Self::Absent`] is a kernel that declares no per-thread input,
/// and [`Self::OverCap`] is one that declares more than this decoder kept. They
/// used to collapse into one `None`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum StageInputVerdict {
    /// No block, or a block naming neither an attribute nor a layout.
    Absent,
    /// Carry it to the backend.
    Use,
    /// The decoder dropped entries. Refuse the pipeline.
    OverCap,
}

/// Classify a decoded stage-input block. Free function so the distinction above
/// is testable without a device, a host or a resolvable descriptor.
pub(crate) fn classify_stage_input(si: Option<&ComputeStageInputDescriptor>) -> StageInputVerdict {
    let Some(si) = si else {
        return StageInputVerdict::Absent;
    };
    if si.dropped_attributes != 0 || si.dropped_layouts != 0 {
        return StageInputVerdict::OverCap;
    }
    if si.attributes.is_empty() && si.layouts.is_empty() {
        return StageInputVerdict::Absent;
    }
    StageInputVerdict::Use
}

pub(crate) fn load_compute_pipeline<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Option<LoadedComputePipeline> {
    // ref==0 is "no pipeline bound" (legitimate) — silent. Other None = a bound
    // pipeline that failed to materialize → caller's coarse MissingPipeline; log
    // the reason (audit).
    if pipeline_ref == 0 {
        return None;
    }
    let report = crate::observe::RungReport::new("compute_load_pipeline", "pipe_ref");
    let miss = |reason: &str, detail: String| -> Option<LoadedComputePipeline> {
        report.reason(task_id, pipeline_ref, reason, &detail);
        None
    };
    let (_entry, desc) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        pipeline_ref,
        &[OBJECT_TYPE_SERIALIZER_OBJECT],
    ) {
        Ok(found) => found,
        Err(rung) => {
            report.rung(task_id, pipeline_ref, rung);
            return None;
        }
    };
    let Ok(decoded) = decode_serializer_object_descriptor(&desc) else {
        return miss(
            crate::observe::ladder_slug!("", desc_decode),
            format!("desc_len={}", desc.len()),
        );
    };
    match decoded {
        ResourceDescriptor::ComputePipeline(cp) if cp.kernel_func_ref != 0 => {
            // The slot holds a compute pipeline, so the model can hold its name
            // — and its destroy arrives after the guest has cleared the slot,
            // which makes this the only moment. Inside the arm that has decided
            // what the descriptor *is*: a serializer object of some other
            // subtype is not this kind and must not be named as one.
            crate::runtime::objects::note_named_at_construction(
                state,
                host,
                task_id,
                pipeline_ref,
                "compute_pipeline_model_named",
                "compute_pipeline_model_unnamed",
            );
            // A descriptor that named more entries than the decoder kept refuses
            // the whole pipeline. Dropping only the stage-input is not "failing
            // closed": `stage_input: None` is what a kernel declaring no
            // per-thread input looks like, so the two become indistinguishable
            // and the dispatch runs with its stage_in fetch silently absent. On
            // the Vulkan arm it is worse than wrong output — `compute_linux`
            // refuses any pipeline carrying a stage-input, and a dropped one
            // walked straight past that refusal.
            let stage_input = match classify_stage_input(cp.stage_input.as_ref()) {
                StageInputVerdict::Absent => None,
                StageInputVerdict::Use => cp.stage_input,
                StageInputVerdict::OverCap => {
                    let si = cp.stage_input.as_ref().expect("OverCap implies a block");
                    return miss(
                        "stage_input_over_cap",
                        format!(
                            "attrs={} dropped_attrs={} layouts={} dropped_layouts={}",
                            si.attributes.len(),
                            si.dropped_attributes,
                            si.layouts.len(),
                            si.dropped_layouts
                        ),
                    );
                }
            };
            Some(LoadedComputePipeline {
                kernel_func_ref: cp.kernel_func_ref,
                stage_input,
            })
        }
        ResourceDescriptor::ComputePipeline(_) => miss("kernel_func_zero", String::new()),
        _ => miss("not_compute_pipeline", String::new()),
    }
}

/// Read `len` bytes from a buffer at `offset` (product + session helpers).
pub(crate) fn read_buffer_window<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>, ComputeStatus> {
    // `ref == 0` is the crate-wide unbound sentinel, not object-list index 0 —
    // every sibling loader guards it and `objects::resolve_descriptor`'s doc says
    // so. Kept as its own refusal rather than folded into the rungs: "the guest
    // bound no buffer" and "the guest named a buffer that is not there" are
    // different statements, and only the second is a resolution failure.
    if buffer_ref == 0 {
        return Err(ComputeStatus::MissingBuffer("compute_buf_win_ref_unbound"));
    }
    // Every other refusal gets its own name too. This used to call a local
    // `Option`-returning helper and label all four `compute_buf_win_no_backing` —
    // the *last* of the four, and so wrong about a ref that names nothing, a ref
    // holding some other object, and a descriptor that would not read or decode.
    let (base, size) =
        objects::resolve_buffer_span(state, host, task_id, buffer_ref).map_err(|refusal| {
            ComputeStatus::MissingBuffer(match refusal {
                objects::BufferSpanRefusal::Rung(rung) => {
                    crate::observe::ladder_slugs!("compute_buf_win")(rung)
                }
                objects::BufferSpanRefusal::Decode => {
                    crate::observe::ladder_slug!("compute_buf_win", desc_decode)
                }
                objects::BufferSpanRefusal::NoBacking => "compute_buf_win_no_backing",
            })
        })?;
    if offset
        .checked_add(len as u64)
        .map(|e| e > size)
        .unwrap_or(true)
    {
        return Err(ComputeStatus::MissingBuffer("compute_buf_win_oob"));
    }
    let gva = base
        .checked_add(offset)
        .ok_or(ComputeStatus::MissingBuffer("compute_buf_win_gva_overflow"))?;
    let mut bytes = vec![0u8; len];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut bytes,
        state.page_shift,
    )
    .map_err(|_| ComputeStatus::GuestIo("compute_buf_win_read"))?;
    Ok(bytes)
}

pub(crate) struct StagedBuffer {
    pub bind: ComputeBufferBind,
    pub gva: u64,
    pub bytes: Vec<u8>,
    /// Guest pages this buffer resolved to when it was staged — before the
    /// dispatch, and before a nested session accumulated however many more
    /// jobs before flushing. `writeback_buffer` runs at the far end of that
    /// gap, so a walk taken there answers where the address points now rather
    /// than whether it is still this buffer's memory. Empty when the
    /// stage-time walk resolved nothing, which leaves the write unbounded as
    /// it was; the writer's own walk then fails closed on its own terms.
    pub pages: std::collections::HashSet<u64>,
}

pub(crate) fn stage_buffer_with_extent<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    bind: &ComputeBufferBind,
    extent_cap: Option<u64>,
) -> Result<StagedBuffer, ComputeStatus> {
    // Eight distinct checks answer with `MissingBuffer`; the status carries
    // which one, so the caller's line and this one name the same slug.
    let miss = |st: ComputeStatus, detail: String| -> Result<StagedBuffer, ComputeStatus> {
        crate::observe::fail(format!(
            "compute_stage_buf fail reason={} ref={} off={:#x} {detail}",
            st.reason(),
            bind.buffer_ref,
            bind.offset
        ));
        Err(st)
    };
    let (_entry, desc_bytes) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        bind.buffer_ref,
        &[OBJECT_TYPE_BUFFER],
    ) {
        Ok(found) => found,
        Err(rung) => {
            return miss(
                ComputeStatus::MissingBuffer(crate::observe::ladder_slugs!("compute_stage_buf")(
                    rung,
                )),
                match rung {
                    objects::LadderRung::WrongType { got } => format!("ot={got}"),
                    objects::LadderRung::NoListEntry
                    | objects::LadderRung::DescRead { .. }
                    | objects::LadderRung::NoTaskSpace => String::new(),
                },
            )
        }
    };
    let Ok(desc) = crate::runtime::decode::resource::decode_buffer_descriptor(&desc_bytes) else {
        return miss(
            ComputeStatus::MissingBuffer(crate::observe::ladder_slug!(
                "compute_stage_buf",
                desc_decode
            )),
            format!("desc_len={}", desc_bytes.len()),
        );
    };
    // Device page_shift (x86=12): handle<<shift is the guest VA. Using the arm
    // default (14) mis-places buffers → walker Unmapped (live compute GuestIo).
    let Some((base_gva, size)) = desc.backing_gva_size(state.page_shift) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_no_backing"),
            format!("handle={:#x}", desc.handle),
        );
    };
    if bind.offset >= size {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_off_oob"),
            format!("size={size:#x}"),
        );
    }
    let full = size - bind.offset;
    let avail = extent_cap.map_or(full, |cap| full.min(cap));
    let Some(want) = host_alloc_len(avail).filter(|&n| n > 0) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_want_bad"),
            format!("size={size:#x} avail={avail:#x}"),
        );
    };
    let Some(gva) = base_gva.checked_add(bind.offset) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_gva_overflow"),
            format!("base={base_gva:#x} size={size:#x}"),
        );
    };
    let mut bytes = vec![0u8; want];
    if let Err(e) = gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut bytes,
        state.page_shift,
    ) {
        // Full walk diagnosis on one line — max learn from a single product boot.
        let walk = gva_mem::diagnose_gva_walk(host, &state.tasks, task_id, gva, state.page_shift);
        // Also probe object base (no offset) in case only the offset page fails.
        let base_walk = if gva != base_gva {
            gva_mem::diagnose_gva_walk(host, &state.tasks, task_id, base_gva, state.page_shift)
        } else {
            String::new()
        };
        crate::observe::fail(format!(
            "compute_stage_buf_gva task={task_id} ref={} base={base_gva:#x} off={:#x} gva={gva:#x} want={want} size={size:#x} page_shift={} err={e:?} | {walk}{}",
            bind.buffer_ref,
            bind.offset,
            state.page_shift,
            if base_walk.is_empty() {
                String::new()
            } else {
                format!(" | base_walk {base_walk}")
            }
        ));
        return Err(ComputeStatus::GuestIo("compute_stage_buf_gva_read"));
    }
    // Count only a cap that actually staged. A failed walk saved no traffic and
    // must not make the rail look effective merely because reflection answered.
    if avail < full {
        crate::runtime::drain::note_store_route("compute_buffer_extent_narrowed");
        crate::runtime::drain::note_store_route_n(
            "compute_buffer_extent_saved_bytes",
            full - avail,
        );
    }
    let pages = staged_span_pages(state, host, task_id, gva, bytes.len() as u64);
    Ok(StagedBuffer {
        bind: bind.clone(),
        gva,
        bytes,
        pages,
    })
}

enum TextureWriteback {
    None,
    Linear {
        texture_ref: u32,
        gva: u64,
        pixel_format: u16,
        row_stride: u64,
        width: u32,
        height: u32,
        bpp: u32,
        /// Guest pages this window resolved to when the texture was staged,
        /// i.e. **before** the dispatch that produces the bytes.
        ///
        /// `writeback_texture` runs after the GPU has finished, and the guest
        /// runs on its own vCPUs across that gap; a walk taken then answers
        /// where the address points *now*, which is a different question from
        /// whether it is still this texture's memory. Empty when the stage-time
        /// walk resolved nothing, which leaves the write unbounded exactly as
        /// it was — the writer's own walk fails closed on its own terms.
        ///
        /// Ordered as well as a membership set, because the GPU-direct arm reads
        /// index `i` as page `i` of the window. See [`staged_window_pages`].
        pages: StoreTargetPages,
    },
    MapperRefTexture {
        mapping_id: u32,
        /// The window this bind was staged against — a byte offset into the
        /// mapping, the surface's row pitch, and one past the last byte the
        /// window may touch.
        ///
        /// Resolved once, at stage time, through the plane the bind actually
        /// names: `ref_texture_sample_window` when the wire carried a ref-texture view's
        /// plane index, `mapper_ref_texture_sample_window` otherwise. Both the read that
        /// seeds the image and the write that lands it use exactly these three
        /// numbers, so the two cannot name different bytes of one surface.
        surface_offset: u64,
        surface_bpr: u32,
        span_end: u64,
        width: u32,
        height: u32,
        /// The guest pixel format the window above was resolved against, and
        /// the only texel measurement this record carries.
        ///
        /// It is the bind's own staged format rather than the mapping's current
        /// declaration, because every byte offset above is arithmetic over it:
        /// judging the same window under a declaration that has since changed
        /// would be judging a different window from the one staged. Bytes per
        /// texel is derived from it at each consumer rather than carried
        /// alongside, so the two cannot disagree.
        format: u16,
    },
}

/// Guest pages a linear storage window resolves to at stage time.
///
/// Taken before the dispatch so the record names the memory the *command* was
/// issued against, not whatever the address points at once the GPU is done. An
/// empty record means the walk resolved nothing and the writeback stays
/// unbounded, which is what it was before this existed.
///
/// # Why this keeps the walk's order and not just its membership
///
/// The walk visits every page of the span in guest-virtual order and reports an
/// unresolved one as `None` rather than skipping it. Collecting straight into a
/// `HashSet` — which this did — throws both of those away, and neither can be
/// recovered afterwards: sorting the set yields ascending *physical* order,
/// which is not the window's order once the guest's mapping is scattered, and
/// nothing in a set says whether a page went missing.
///
/// The row-by-row host writer only ever asked "is this page one of mine?", so
/// the loss did not show. A GPU-direct copy asks the other question — it reads
/// index `i` as page `i` of the window — and a short or reordered vector lands
/// the frame at the wrong guest addresses with nothing noticing, because the
/// copy converts nothing and checks nothing. [`StoreTargetPages`] is the render
/// rail's existing answer to exactly this and carries both forms from the one
/// walk, so this rail takes that type rather than growing a third spelling.
fn staged_window_pages<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    height: u32,
) -> StoreTargetPages {
    let Some(span) = row_stride.checked_mul(height as u64) else {
        return StoreTargetPages::empty();
    };
    if gva == 0 || span == 0 {
        return StoreTargetPages::empty();
    }
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        gva,
        span,
        state.page_shift,
    );
    StoreTargetPages::from_ordered(&ordered, span)
}

/// [`staged_window_pages`] for a flat span — the buffer rail's shape.
fn staged_span_pages<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> std::collections::HashSet<u64> {
    let mut pages = std::collections::HashSet::new();
    if gva == 0 || span == 0 {
        return pages;
    }
    pages.extend(gva_mem::task_gva_page_gpa_set(
        host,
        &state.tasks,
        task_id,
        gva,
        span,
        state.page_shift,
    ));
    pages
}

/// What one rail additionally carries about a staged compute texture.
///
/// # Why a type parameter and not more fields
///
/// [`StagedTexture`] used to hold both rails' private per-binding state inline,
/// each field behind a `#[cfg]`: five for Vulkan (the descriptor array slot and
/// count, the storage-residency candidate, what a resident could already serve,
/// and the multisample target a `texture2d_ms` read binds), one for Metal (the
/// guest ref its format refusal names). Twelve `cfg` lines repeated across four
/// construction sites, and on the both-rails build — where every one of those
/// gates is *on* — a single struct carrying both rails' privates with nothing
/// but convention keeping either out of the other's.
///
/// A type parameter says the same thing and makes it true: `StagedTexture<MetalStage>`
/// has no Vulkan field to read, on any build, because the field is not in the
/// type. It also collapses the construction sites, which now name one `rail`
/// instead of repeating the gates.
///
/// The rail is handed the neutral facts and decides what to keep. Both facts
/// are already neutral — the residency key is a model type and
/// [`ResidentServe`] is what [`crate::backend::Backend::resident_serve`]
/// answers — so this is the rail narrowing a neutral answer, not the neutral
/// layer computing a rail's input.
pub(crate) trait RailStage: Sized {
    /// This rail's half of one staged binding, from the neutral facts of it.
    ///
    /// `texture_ref` is the guest object reference this binding was staged
    /// from. `residency` is the storage-mirror window the staging corresponds
    /// to, for a storage binding and `None` for anything else, and `serve` is
    /// what the running rail said it could already serve for that window —
    /// which is also why `bytes` may be a zero placeholder.
    ///
    /// Three arguments rather than one struct of them because a rail that keeps
    /// no residency mirror ignores the last two, and an ignored *argument* is
    /// ignored while an unread *field* is dead weight the compiler is right to
    /// report on the build where no rail reads it.
    fn stage(
        texture_ref: u32,
        residency: Option<ComputeStorageResidencyCandidate>,
        serve: Option<ResidentServe>,
    ) -> Self;
}

/// The storage-mirror window a staged binding corresponds to.
///
/// Both fields are neutral, which is why the staging rails below can produce it
/// without naming a rail; only a rail that keeps a residency mirror stores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ComputeStorageResidencyCandidate {
    pub(crate) key: crate::model::ComputeStorageResidencyKey,
    pub(crate) seed_generation: u32,
}

pub(crate) struct StagedTexture<R: RailStage> {
    pub binding: u32,
    /// Raw Metal pixel format from the exact texture/view descriptor.
    pub pixel_format: u16,
    /// Product storage-selector ABI when this Metal format is storage-capable.
    /// Sample-only formats such as RGB9E5Float intentionally have no selector.
    /// The contract's storage-image selector for this texture's format, or
    /// `None` for a format that is not a storage image.
    ///
    /// Carried as the enum rather than as its `u32` ordinal. It used to be
    /// narrowed to `u32` the moment `pixel_format::storage_selector` produced
    /// it, at three staging sites, which pushed the coverage question past every
    /// compiler that could have answered it: both backends then matched raw
    /// integers, and the Metal one had silently been missing a member.
    pub storage_selector: Option<pixel_format::StorageImageSelector>,
    pub width: u32,
    pub height: u32,
    /// How many mip levels `bytes` carries, base first, packed tightly by
    /// [`reims_vgpu_protocol::extent::tight_pyramid_spans`].
    ///
    /// `1` on every rail but the normal-texture linear one, and `1` there too for a
    /// storage binding or a view that already names a level: a compute write
    /// names one level and a levelled view exposes one. Where it is greater,
    /// `width`/`height` remain level 0's extent and every other level's is
    /// `mip_extent(width, n)` — the pyramid is a derivation of this geometry
    /// and not a second one, so no level's extent is stored twice.
    pub mip_levels: u32,
    pub bytes: Vec<u8>,
    pub is_storage: bool,
    writeback: TextureWriteback,
    /// The running rail's own half of this binding. See [`RailStage`].
    pub rail: R,
}

/// What an engine-resident copy of a window can serve one staged binding.
///
/// `Seed` means a storage binding's output is already GPU-resident at this
/// generation, so the guest read that would seed it is unnecessary. `Sample`
/// names the resident key a sampled binding reads directly instead.
///
/// Which variant a binding can receive is fixed by `is_storage`, not chosen:
/// [`Backend::resident_serve`]'s two arms are the two variants. That is why the
/// consumers split the same way — the storage rail reads only the seed and the
/// sampled rail only the source.
///
/// Neutral, because the question is neutral: every staging site asks the
/// running rail what it already holds, and a rail that holds nothing answers
/// `None` from the trait's default. The type is not a Vulkan detail leaking
/// outward — it is the shape of the answer, and only the answer's *content*
/// differs by rail.
///
/// [`Backend::resident_serve`]: crate::backend::Backend::resident_serve
#[cfg_attr(
    not(feature = "backend-vulkan"),
    allow(
        dead_code,
        reason = "no rail this build compiled constructs one; every rail still reads the type"
    )
)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum ResidentServe {
    Seed(u32),
    Sample(crate::model::ComputeStorageResidencyKey, u32),
}

impl ResidentServe {
    /// The generation a seeded resident is held at, or `None` for a sampled
    /// one — whose generation belongs to its key rather than to the guest read
    /// this binding skipped.
    pub(crate) fn seed_generation(self) -> Option<u32> {
        match self {
            Self::Seed(generation) => Some(generation),
            Self::Sample(..) => None,
        }
    }

    /// The resident a sampled binding reads directly, or `None` for a seeded
    /// one.
    pub(crate) fn sample_source(self) -> Option<(crate::model::ComputeStorageResidencyKey, u32)> {
        match self {
            Self::Sample(key, generation) => Some((key, generation)),
            Self::Seed(_) => None,
        }
    }
}

/// Stage an opcode-9 buffer-backed texture: tight raw texels read out of the
/// buffer the guest named, at its declared offset and row pitch.
///
/// The contract is `newTextureWithBuffer:descriptor:offset:bytesPerRow:` — the
/// texels *are* the buffer's bytes, reinterpreted through the embedded texture
/// descriptor. [`crate::runtime::draw`] executes the same record for the draw
/// rail; this is its compute twin and reads the same fields the same way,
/// because one wire form with two disagreeing readers is the defect shape this
/// repository keeps finding. This arm used to refuse the form outright.
///
/// Unlike the draw twin this does **not** convert to RGBA8. [`StagedTexture`]
/// carries `pixel_format` beside `bytes`, so the native texels survive; the
/// draw arm narrows because its consumer takes RGBA8, and reports the loss as
/// `buftex_narrowed`. Here there is no loss to report.
///
/// De-pitching is the whole of the work: the guest's rows are `bytes_per_row`
/// apart and only the leading tight row is texels. The rest is padding the
/// guest may have written anything into, and folding it into the image is
/// exactly the failure the conformance battery fills padding with a distinct
/// pattern to catch.
fn stage_buffer_texture<R: RailStage, M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    binding: u32,
    is_storage: bool,
    bt: &crate::runtime::decode::resource::BufferTextureDescriptor,
) -> Result<StagedTexture<R>, ComputeStatus> {
    let (width, height) = (bt.desc.width, bt.desc.height);
    if width == 0 || height == 0 {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=zero_geom ref={texture_ref} buf={} {width}x{height}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported("compute_buftex_zero_geom"));
    }
    // A storage binding would have to write *back* through the buffer, which is
    // a destination contract this arm has no evidence for: no case in the
    // battery binds a buffer-backed texture writable, and inventing a writeback
    // here would widen the repair past what it can show. Refused under its own
    // name so the two questions stay separable in the log.
    if is_storage {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=storage_destination ref={texture_ref} buf={} {width}x{height}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported(
            "compute_buffer_texture_storage_unsupported",
        ));
    }
    let format = if bt.desc.pixel_format != 0 {
        bt.desc.pixel_format
    } else {
        crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM
    };
    let Some(tight) = pixel_format::tight_row_bytes(width, format) else {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=unknown_fmt ref={texture_ref} buf={} fmt={format:#x} {width}x{height}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported("compute_buftex_fmt"));
    };
    // A declared `bytesPerRow` of 0 means tight rows — the API default a
    // single-row or unpadded texture serializes as. Same reading as the draw
    // twin; the two arms must not differ on it.
    let bpr = if bt.bytes_per_row == 0 {
        u64::from(tight)
    } else {
        bt.bytes_per_row
    };
    if bpr < u64::from(tight) {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=bpr_short ref={texture_ref} buf={} bpr={bpr} tight={tight} {width}x{height} fmt={format:#x}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported("compute_buftex_bpr_short"));
    }
    // The span the guest's rows actually occupy. Every row is `bpr` apart, but
    // the last one only needs its texels: demanding a full trailing pitch would
    // refuse a texture whose final row sits at the very end of the allocation.
    let Some(span) = bpr
        .checked_mul(u64::from(height) - 1)
        .and_then(|s| s.checked_add(u64::from(tight)))
        .and_then(|s| usize::try_from(s).ok())
    else {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=span_overflow ref={texture_ref} buf={} bpr={bpr} {width}x{height}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported("compute_buftex_span"));
    };
    // A buffer-backed texture is two contract references over one allocation —
    // the texture-view texture the guest binds and the buffer that owns the
    // storage — and a debt may be armed under either. The draw twin pays for
    // both; so does this.
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    let raw = read_buffer_window(state, host, task_id, bt.buffer_ref, bt.offset, span)?;

    let tight = tight as usize;
    let bpr = bpr as usize;
    let mut bytes = vec![0u8; tight * height as usize];
    for y in 0..height as usize {
        let src = y * bpr;
        bytes[y * tight..(y + 1) * tight].copy_from_slice(&raw[src..src + tight]);
    }
    crate::observe::off(format!(
        "compute_stage_tex buftex_ok ref={texture_ref} buf={} fmt={format:#x} {width}x{height} off={} bpr={bpr} tight={tight}",
        bt.buffer_ref, bt.offset
    ));
    Ok(StagedTexture {
        binding,
        pixel_format: format,
        storage_selector: pixel_format::storage_selector(format),
        // A buffer-backed texture view is one level of one buffer.
        mip_levels: 1,
        width,
        height,
        bytes,
        is_storage,
        writeback: TextureWriteback::None,
        rail: R::stage(texture_ref, None, None),
    })
}

/// Load tight raw texels for a compute texture binding (normal-texture, ref-texture→surface, or
/// mapper-ref-texture).
///
/// Ref-texture (`RefTextureHandle`) is the live CI wallpaper path (`compute_stage_tex … ot=5`). RE
/// (ref-texture wire + `runtime::draw` sample path): surfaceID@0 is a backing object id (= mapping
/// mid). Product draw samples call [`objects::ensure_surface_for_present`] on that id and stage
/// from the **mapping registry**, never re-resolving the surface id through the compute task's
/// object list (that list uses a separate texture-ref namespace — live ensure=1 then
/// MissingTexture/GuestIo class when `resolve_mapper_ref_texture(task, sid)` hit a different
/// mapper-ref-texture slot).
pub(crate) fn stage_texture_raw<R: RailStage, M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    binding: u32,
    is_storage: bool,
) -> Result<StagedTexture<R>, ComputeStatus> {
    // Ref-texture RefTextureHandle → surface_id (live CI binds ot5).
    let mut stage_ref = texture_ref;
    let mut from_ref_texture = false;
    let mut from_backing_direct = false;
    let mut ref_texture_record: Option<objects::RefTextureView> = None;
    let mut view_level = 0;
    let mut view_pixel_format = None;
    let mut heap_texture = None;
    let mut buffer_texture: Option<crate::runtime::decode::resource::BufferTextureDescriptor> =
        None;
    // A linear texture object (normal-texture) must resolve through its own
    // descriptor, never through the mapping registry: its numeric ref shares
    // the id space with backing record mids, so the `mappings.contains(ref)`
    // fallback below would wrongly grab a same-numbered surface (live class:
    // `ref=N ot=2` dragged into the mapper-ref-texture path and failing silently against
    // the biplanar wallpaper mid). Same collision the ref-texture path documents.
    // Resolve the object-list entry once: `ref_is_linear` and the ref_texture/backing
    // classification below both read it for the same ref, and the guest object
    // list is immutable for the life of the dispatch (the device never writes
    // those pages). `ListObjectEntry` is `Copy`, so one guest-DMA read+decode
    // serves both instead of two.
    let ref_entry = objects::lookup_list_entry(state, host, task_id, texture_ref);
    if let Some(entry) = ref_entry {
        if entry.object_type == OBJECT_TYPE_TEXTURE_VIEW {
            let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) else {
                crate::observe::fail(format!(
                    "compute_stage_tex view_fail reason=no_desc ref={texture_ref} desc_len={}",
                    entry.descriptor_length
                ));
                return Err(ComputeStatus::MissingTexture(crate::observe::ladder_slug!(
                    "compute_stage_tex_view",
                    desc_read
                )));
            };
            let opcode = texture_view_opcode(&desc).unwrap_or(0);
            // Both opcodes are the same record: the wide one is what the guest's
            // serializer emits with `TextureDescriptor2` on. The length each
            // implies is `decode_heap_texture`'s to check — this site used to
            // check it too, against the narrow constant alone, which would have
            // rejected every wide record before its decoder saw the opcode.
            if opcode == HEAP_TEXTURE_OPCODE || opcode == HEAP_TEXTURE_WIDE_OPCODE {
                let record = match decode_heap_texture(&desc) {
                    Ok(record) => record,
                    Err(error) => {
                        crate::observe::Emit::decline("compute_stage_tex_heap", &error)
                            .field("ref", texture_ref)
                            .field("len", desc.len())
                            .fail();
                        return Err(ComputeStatus::MissingTexture(
                            "compute_stage_tex_heap_bad_record",
                        ));
                    }
                };
                let (heap_ref, use_offset, offset) =
                    (record.heap_ref, record.use_offset, record.offset);
                if heap_ref == 0 {
                    crate::observe::fail(format!(
                        "compute_stage_tex heap_fail reason=zero_heap ref={texture_ref}"
                    ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_heap_zero_ref",
                    ));
                }
                let body = if record.wide {
                    crate::runtime::heap_query::decode_wide_serialized_texture_descriptor(
                        record.descriptor,
                    )
                } else {
                    crate::runtime::heap_query::decode_serialized_texture_descriptor(
                        record.descriptor,
                    )
                };
                let descriptor = match body {
                    Ok(descriptor) => descriptor,
                    Err(error) => {
                        crate::observe::Emit::decline("compute_stage_tex_heap", &error)
                            .field("ref", texture_ref)
                            .field("heap", heap_ref)
                            .field("use_offset", use_offset)
                            .field("offset", format!("{offset:#x}"))
                            .fail();
                        return Err(ComputeStatus::MissingTexture(crate::observe::ladder_slug!(
                            "compute_stage_tex_heap",
                            desc_decode
                        )));
                    }
                };
                // What the reference names, measured. See
                // `objects::note_heap_reference` for why the placement is
                // executed and the reference still reported.
                let _ = objects::note_heap_reference(state, host, task_id, heap_ref);
                heap_texture = Some((heap_ref, use_offset, offset, descriptor));
            }
            if heap_texture.is_some() {
                // Heap textures are complete resource objects, not texture
                // views. Their backing is a host GPU residency identity.
            } else if opcode == TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE
                || opcode == TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE
            {
                // The same wire form `runtime::draw` decodes and executes, read
                // through the same decoder. This arm used to refuse it.
                let bt =
                    match crate::runtime::decode::resource::decode_buffer_texture_descriptor(&desc)
                    {
                        Ok(bt) => bt,
                        Err(error) => {
                            crate::observe::Emit::decline("compute_stage_tex_buftex", &error)
                                .field("ref", texture_ref)
                                .field("opcode", format!("{opcode:#x}"))
                                .field("len", desc.len())
                                .fail();
                            return Err(ComputeStatus::MissingTexture(
                                "compute_stage_tex_buftex_desc",
                            ));
                        }
                    };
                buffer_texture = Some(bt);
            } else {
                let view = match crate::runtime::draw::resolve_texture_view_reasoned(
                    state,
                    host,
                    task_id,
                    texture_ref,
                ) {
                    Ok(view) => view,
                    Err(reason) => {
                        crate::observe::Emit::decline("compute_stage_tex_view_resolve", &reason)
                            .field("ref", texture_ref)
                            .field("opcode", format!("{opcode:#x}"))
                            .fail_once(texture_ref as u64);
                        return Err(ComputeStatus::MissingTexture(
                            "compute_stage_tex_view_resolve",
                        ));
                    }
                };
                if view
                    .swizzle
                    .as_ref()
                    .is_some_and(|plan| !pixel_format::swizzle_is_identity(plan))
                {
                    crate::observe::fail(format!(
                        "compute_stage_tex view_fail reason=swizzle_unsupported ref={texture_ref} base={} opcode={opcode} storage={}",
                        view.base_texture_ref, is_storage as u8
                    ));
                    return Err(ComputeStatus::Unsupported(
                        "compute_view_swizzle_unsupported",
                    ));
                }
                stage_ref = view.base_texture_ref;
                view_level = view.level;
                view_pixel_format = view.pixel_format;
            }
        }
    }
    if let Some(bt) = buffer_texture {
        return stage_buffer_texture(state, host, task_id, texture_ref, binding, is_storage, &bt);
    }
    if let Some((heap_ref, use_offset, offset, descriptor)) = heap_texture {
        if descriptor.texture_type != 2
            || descriptor.depth != 1
            || descriptor.mipmap_level_count != 1
            || descriptor.sample_count != 1
            || descriptor.array_length != 1
        {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=shape ref={texture_ref} heap={heap_ref} type={} dims={}x{}x{} mips={} samples={} array={} use_offset={} offset={offset:#x}",
                descriptor.texture_type,
                descriptor.width,
                descriptor.height,
                descriptor.depth,
                descriptor.mipmap_level_count,
                descriptor.sample_count,
                descriptor.array_length,
                use_offset as u8
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_shape"));
        }
        let (width, height, format) =
            (descriptor.width, descriptor.height, descriptor.pixel_format);
        let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=fmt_bytes ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height}"
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_fmt_bytes"));
        };
        let storage_selector = pixel_format::storage_selector(format);
        if is_storage && storage_selector.is_none() {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=fmt_storage ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height}"
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_fmt_storage"));
        }
        let Some(need) = (width as usize)
            .checked_mul(height as usize)
            .and_then(|texels| texels.checked_mul(bpp as usize))
        else {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=host_len ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} bpp={bpp}"
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_host_len"));
        };
        let key = crate::model::ComputeStorageResidencyKey::heap(
            task_id,
            texture_ref,
            width,
            height,
            format,
        );
        let serve = match state.compute_storage_residency.get(&key).copied() {
            None => None,
            Some(generation) => match crate::backend::selected()
                .resident_serve(key, generation, is_storage, format)
            {
                // A heap texture has no guest window to re-read: once the mirror
                // claims a resident, the engine's copy is the only content, so a
                // resident the engine can no longer serve is a loss, not a
                // fallback. The window-backed rails below fall through to the
                // guest read here instead; this is the arm that must not.
                None => {
                    crate::observe::fail(format!(
                            "compute_stage_tex heap_fail reason=resident_lost ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} gen={generation} use_offset={} offset={offset:#x}",
                            use_offset as u8
                        ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_heap_resident_lost",
                    ));
                }
                serve => serve,
            },
        };
        let seed_generation = serve.and_then(ResidentServe::seed_generation).unwrap_or(0);
        crate::observe::off(format!(
            "compute_stage_tex heap_ok ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} storage={} seed_gen={seed_generation} resident_sample={} use_offset={} offset={offset:#x}",
            is_storage as u8,
            serve.and_then(ResidentServe::sample_source).is_some() as u8,
            use_offset as u8
        ));
        return Ok(StagedTexture {
            binding,
            pixel_format: format,
            storage_selector,
            // The heap arm refuses a descriptor declaring more than one level
            // above, so a heap texture reaching here is single-level.
            mip_levels: 1,
            width,
            height,
            bytes: vec![0; need],
            is_storage,
            writeback: TextureWriteback::None,
            rail: R::stage(
                texture_ref,
                is_storage.then_some(ComputeStorageResidencyCandidate {
                    key,
                    seed_generation,
                }),
                serve,
            ),
        });
    }
    let stage_entry = objects::lookup_list_entry(state, host, task_id, stage_ref);
    let ref_is_linear = stage_entry
        .map(|e| {
            e.object_type == OBJECT_TYPE_TEXTURE
                || e.object_type == OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS
        })
        .unwrap_or(false);
    if let Some(entry) = stage_entry {
        if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
            if let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) {
                if let Ok(t5) = reims_vgpu_wire::device_desc::ref_texture_header(&desc) {
                    let sid = t5.surface_id.get();
                    if sid != 0 {
                        stage_ref = sid;
                        from_ref_texture = true;
                        ref_texture_record = objects::decode_ref_texture_view(&desc);
                        let ok = objects::ensure_surface_for_present(state, host, sid);
                        // Per-bind ref-texture descriptor RE census (args@+8 holds the
                        // serialized plane texture; product stage uses mapping geom
                        // only today). This is measurement, not a failure — it fired
                        // ~600×/boot on the always-on sink (same descriptor re-dumped
                        // per bind, no dedup), drowning genuine failures. Verbose-gated;
                        // build the head-hex only when REIMS_VGPU_DRAW_LOG is on. A genuine
                        // ensure failure surfaces downstream as `MissingTexture` (the
                        // mapping lookup below misses), so no always-on line is lost.
                        crate::observe::when_verbose(|| {
                            // The owner task the view names. `note_ref_texture_owner_task`
                            // is the always-on check on its value; this echo carries
                            // it beside the descriptor it came out of.
                            let owner_task = t5.owner_task.get();
                            let args_n = desc.len().saturating_sub(objects::TYPE5_ARGS);
                            let mut args_hex = String::new();
                            if args_n > 0 {
                                let n = args_n.min(48);
                                args_hex.reserve(n * 2);
                                for b in &desc[objects::TYPE5_ARGS..objects::TYPE5_ARGS + n] {
                                    use std::fmt::Write as _;
                                    let _ = write!(args_hex, "{b:02x}");
                                }
                                if args_n > n {
                                    args_hex.push('…');
                                }
                            }
                            crate::observe::line(format!(
                                "compute_stage_tex ref_texture ref={texture_ref} sid={sid} ensure={} owner_task={owner_task} desc_len={} args_n={args_n} args_hex={args_hex}",
                                ok as u8,
                                desc.len(),
                            ));
                        });
                    }
                }
            }
        } else if entry.object_type == objects::OBJECT_TYPE_BACKING {
            // Direct backing record bind (same id space as present mids).
            from_backing_direct = true;
            let _ = objects::ensure_surface_for_present(state, host, stage_ref);
        }
    }

    // Ref-texture / direct backing: surface id **is** the mapping mid. Never call
    // resolve_mapper_ref_texture(task, sid) — task object-list indices collide with texture refs.
    let mapping_id_opt = if from_ref_texture || from_backing_direct {
        if stage_ref != 0 && state.mappings.contains_key(&stage_ref) {
            Some(stage_ref)
        } else {
            None
        }
    } else if ref_is_linear {
        // Linear texture: never fall back to the mapping registry (id-space
        // collision with backing record mids). Force the normal-texture path.
        None
    } else {
        objects::resolve_mapper_ref_texture(state, host, task_id, stage_ref).or_else(|| {
            if stage_ref != 0 && state.mappings.contains_key(&stage_ref) {
                Some(stage_ref)
            } else {
                None
            }
        })
    };
    if mapping_id_opt.is_none() && from_ref_texture {
        crate::observe::fail(format!(
            "compute_stage_tex ref_texture_no_map ref={texture_ref} sid={stage_ref}"
        ));
        return Err(ComputeStatus::MissingTexture(
            "compute_stage_tex_ref_texture_no_map",
        ));
    }
    if let Some(mapping_id) = mapping_id_opt {
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        // Geom/format: a ref-texture record is the exact Metal texture view over
        // the IOSurface bytes. It is authoritative even for a stageable
        // single-plane mapping: the live BGRA8 desktop target is exposed as a
        // row-byte-equivalent, quarter-width RGBA32Uint view. Backing direct
        // refs use base mapping geometry. Mapper-ref-texture refs may prefer the
        // IOSurface descriptor on this task's object list.
        if view_level != 0 {
            crate::observe::fail(format!(
                "compute_stage_tex view_fail reason=mapper_ref_texture_mip ref={texture_ref} base={stage_ref} level={view_level} mapping={mapping_id}"
            ));
            return Err(ComputeStatus::Unsupported(
                "compute_view_mapper_ref_texture_mip",
            ));
        }
        let (width, height, format) = if from_ref_texture || from_backing_direct {
            let m = state
                .mappings
                .get(&mapping_id)
                .ok_or(ComputeStatus::MissingTexture(
                    "compute_stage_tex_mapping_gone",
                ))?;
            let multiplanar = objects::mapping_is_multiplanar(m);
            let mapping_stageable =
                m.has_geom && m.width != 0 && m.height != 0 && m.format != 0 && !multiplanar;
            if let Some(rec) = ref_texture_record {
                // `mapper_ref_texture_sample_window` below matches actual plane records by
                // geometry+bpe and otherwise verifies a packed row-compatible
                // view over the same bytes. Per-bind measurement (view vs base
                // geom), not a failure — verbose-gated to keep the always-on sink
                // for genuine failures.
                crate::observe::line(format!(
                    "compute_stage_tex ref_texture_view mapping={mapping_id} view={}x{} fmt={:#x} base={}x{} fmt={:#x} multiplanar={}",
                    rec.width,
                    rec.height,
                    rec.pixel_format,
                    m.width,
                    m.height,
                    m.format,
                    multiplanar as u8
                ));
                (rec.width, rec.height, rec.pixel_format)
            } else if !mapping_stageable {
                if !m.has_geom || m.width == 0 || m.height == 0 {
                    crate::observe::fail(format!(
                        "compute_stage_tex mapper_ref_texture_fail reason=no_geom mapping={mapping_id} pages={} has_geom={}",
                        m.page_entries.len(),
                        m.has_geom as u8
                    ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_mapper_ref_texture_no_geom",
                    ));
                } else if multiplanar {
                    // Multi-plane IOSurface without a plane record: fail closed,
                    // do not invent BGRA sample of the whole surface.
                    crate::observe::fail(format!(
                        "compute_stage_tex mapper_ref_texture_fail reason=multiplane mapping={mapping_id} {}x{} fmt={:#x} pages={} (no ref-texture plane record)",
                        m.width,
                        m.height,
                        m.format,
                        m.page_entries.len()
                    ));
                    return Err(ComputeStatus::Unsupported("stage_tex_multiplane_no_plane"));
                } else {
                    // Single-plane unknown format: fail closed (no BGRA invent).
                    crate::observe::fail(format!(
                        "compute_stage_tex mapper_ref_texture_fail reason=fmt_unknown mapping={mapping_id} {}x{} pages={}",
                        m.width,
                        m.height,
                        m.page_entries.len()
                    ));
                    return Err(ComputeStatus::Unsupported("stage_tex_fmt_unknown"));
                }
            } else {
                (m.width, m.height, m.format)
            }
        } else {
            // Three ways the surface's own IOSurface descriptor can fail to
            // answer — no list entry, no descriptor bytes, or bytes that do not
            // decode as an IOSurfaceTexture — and all three fall back to the
            // mapping's latched geometry. Kept sequential rather than chained so
            // the `&mut state` the lookups need does not overlap the `&state` the
            // fallback reads.
            let mut from_descriptor = None;
            if let Some(entry) = objects::lookup_list_entry(state, host, task_id, stage_ref) {
                if let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) {
                    if let Ok(ResourceDescriptor::IOSurfaceTexture {
                        width,
                        height,
                        pixel_format,
                        ..
                    }) = crate::runtime::decode::resource::decode_iosurface_texture_descriptor(
                        &desc_bytes,
                    ) {
                        from_descriptor = Some((width, height, or_bgra8(pixel_format)));
                    }
                }
            }
            match from_descriptor {
                Some(geom) => geom,
                None => mapping_geom_format(state, mapping_id)?,
            }
        };
        if width == 0 || height == 0 {
            return Err(ComputeStatus::MissingTexture("compute_stage_tex_zero_geom"));
        }
        // sRGB color-renderable surfaces stage as unorm storage (same bpp).
        let view_format = match crate::runtime::draw::effective_view_sample_format_reasoned(
            format,
            view_pixel_format,
        ) {
            Ok(view_format) => view_format,
            Err(refusal) => {
                // `term=` is what says whether this is a gap in this crate's
                // format table or the guest asking for something Metal forbids,
                // and `role=` says which rail would have had to take it — the
                // two questions the next reader has, and the two the old
                // `format_incompatible` could not answer. The bind dies here,
                // before the storage check, so without `role=` the log cannot
                // say whether the missing rail is a sampled layout or a storage
                // selector.
                crate::observe::fail(format!(
                    "compute_stage_tex view_fail reason=format_incompatible term={refusal} \
                     role={} ref={texture_ref} base={stage_ref} base_fmt={format:#x} \
                     view_fmt={view_pixel_format:?} {width}x{height} mapping={mapping_id}",
                    if is_storage { "storage" } else { "sampled" }
                ));
                return Err(ComputeStatus::Unsupported("compute_view_format"));
            }
        };
        let stage_fmt = match view_format {
            pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB => pixel_format::MTL_FORMAT_BGRA8_UNORM,
            pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB => pixel_format::MTL_FORMAT_RGBA8_UNORM,
            other => other,
        };
        let bpp = match pixel_format::bytes_per_pixel(stage_fmt) {
            Some(v) => v,
            None => {
                crate::observe::fail(format!(
                    "compute_stage_tex mapper_ref_texture_fail reason=fmt_bytes mapping={mapping_id} {width}x{height} fmt={format:#x}"
                ));
                return Err(ComputeStatus::Unsupported("stage_tex_fmt_bytes"));
            }
        };
        let storage_selector = pixel_format::storage_selector(stage_fmt);
        if is_storage && storage_selector.is_none() {
            crate::observe::fail(format!(
                "compute_stage_tex mapper_ref_texture_fail reason=fmt_storage mapping={mapping_id} {width}x{height} fmt={format:#x}"
            ));
            return Err(ComputeStatus::Unsupported("stage_tex_fmt_storage"));
        }
        let m = state
            .mappings
            .get(&mapping_id)
            .ok_or(ComputeStatus::MissingTexture(
                "compute_stage_tex_mapping_gone",
            ))?;
        let map_generation = m.map_generation;
        // Read off `m` here because the borrow does not reach the staging
        // below; what this generation *means* for the staging is decided there,
        // once `serve` is known.
        let content_generation = m.content_generation;
        let pages_n = m.page_entries.len();
        // Wire backing `length` (page-aligned getResidentSize), stashed as device_desc.alloc_size.
        // Independent of plane w/h and of MapMemory2 IOAccelMemory length — measure-only.
        let wire_len = crate::protocol::iosurface_pages::decode_device_surface(&m.device_desc)
            .map(|s| s.alloc_size as u64)
            .unwrap_or(0);
        // A ref-texture record names its IOSurface plane on the wire (record `+0x20`,
        // the `newTextureWithDescriptor:iosurface:plane:` argument), so the
        // plane is decided, not inferred. Mapper-ref-texture carries no such field and must
        // still match a plane record by geometry — which is ambiguous whenever
        // two planes share dims and bytes-per-element (v0a8 Y and alpha), and
        // declines rather than picking one. The draw path already binds ref-texture
        // views by index; this is the same resolution on the staging path.
        let window = match ref_texture_record {
            Some(rec) => mapping_write::ref_texture_sample_window(
                m,
                rec.plane_index,
                width,
                height,
                stage_fmt,
            ),
            None => mapping_write::mapper_ref_texture_sample_window(m, width, height, stage_fmt),
        };
        let (surface_offset, surface_bpr, span_end) = match window {
            Some(w) => w,
            None => {
                // What the descriptor said, so a refusal names which of its
                // fields the texture could not be placed against. `reach` is the
                // byte count this geometry needs; a descriptor whose alloc is
                // smaller is a different failure from one whose plane records
                // matched nothing.
                let ds = crate::protocol::iosurface_pages::decode_device_surface(&m.device_desc);
                let (dw, dh, dbpr, dalloc) = ds
                    .as_ref()
                    .map(|s| (s.width, s.height, s.bytes_per_row, s.alloc_size))
                    .unwrap_or((0, 0, 0, 0));
                let reach = crate::protocol::iosurface_pages::packed_span_estimate(
                    stage_fmt, width, height,
                )
                .unwrap_or(0);
                crate::observe::fail(format!(
                    "compute_stage_tex mapper_ref_texture_fail reason=window mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} pages={pages_n} wire_len={wire_len} desc={dw}x{dh} bpr={dbpr} alloc={dalloc} reach={reach}"
                ));
                return Err(ComputeStatus::MissingTexture(
                    "compute_stage_tex_mapper_ref_texture_window",
                ));
            }
        };
        let tight = (width as u64)
            .checked_mul(bpp as u64)
            .ok_or(ComputeStatus::Unsupported("stage_tex_tight_bpr_overflow"))?
            as u32;
        if from_ref_texture && ref_texture_record.is_some() {
            // Per-bind ref-texture sample-window measurement, not a failure — verbose-gated
            // (was a per-bind always-on line). Genuine window failures above emit
            // `mapper_ref_texture_fail reason=window` always-on.
            crate::observe::line(format!(
                "compute_stage_tex ref_texture_view_window mapping={mapping_id} view={width}x{height} fmt={stage_fmt:#x} bpp={bpp} tight={tight} surface_off={surface_offset} surface_bpr={surface_bpr} span_end={span_end}"
            ));
        }
        let need_u64 = (tight as u64)
            .checked_mul(height as u64)
            .ok_or(ComputeStatus::Unsupported("stage_tex_need_overflow"))?;
        let Some(need) = host_alloc_len(need_u64) else {
            crate::observe::fail(format!(
                "compute_stage_tex mapper_ref_texture_fail reason=host_len mapping={mapping_id} need={need_u64}"
            ));
            return Err(ComputeStatus::Unsupported("stage_tex_host_len"));
        };
        let page_bytes = (pages_n as u64).saturating_mul(1u64 << state.page_shift);
        if page_bytes < span_end {
            crate::observe::fail(format!(
                "compute_stage_tex mapper_ref_texture_fail reason=span mapping={mapping_id} {width}x{height} pages={pages_n} page_bytes={page_bytes} span_end={span_end} bpr={surface_bpr} wire_len={wire_len}"
            ));
            return Err(ComputeStatus::GuestIo(
                "compute_stage_tex_mapper_ref_texture_span",
            ));
        }
        let residency_key = crate::model::ComputeStorageResidencyKey {
            mapping_id,
            map_generation,
            surface_offset,
            surface_bpr,
            span_end,
            width,
            height,
            pixel_format: stage_fmt,
            texture_ref: 0,
        };
        // Chained-dispatch restage skip: when guest pages still hold exactly
        // our own last writeback for THIS WINDOW (mirror entry survives only
        // while no intersecting guest write lands — `DeviceState::
        // invalidate_storage_residency_window`, called from mapping_write and
        // mapper, drops every mirror entry whose byte window overlaps the
        // write and keeps the disjoint siblings) AND the engine still holds the
        // resident image at the mirror's generation, reading ~15 MB from guest
        // pages reproduces what the GPU already has. The mapping-level content
        // generation may have advanced via disjoint sibling windows
        // (ping-pong canvases), so the gate pairs mirror↔engine directly.
        // The zero placeholder is never seeded — the engine fails visibly
        // with `vk_compute_exec_resident_seed_generation_lost` if the resident
        // vanishes by acquire time.
        // Copy-on-sample is the same gate: a sampled input of a window whose
        // current content the engine already holds GPU-resident (a prior
        // dispatch's storage output — live class: the dispatch samples the very
        // window it storage-writes) never needs the guest read either.
        let serve = state
            .compute_storage_residency
            .get(&residency_key)
            .copied()
            .and_then(|mirror_generation| {
                crate::backend::selected().resident_serve(
                    residency_key,
                    mirror_generation,
                    is_storage,
                    stage_fmt,
                )
            });
        // The generation this staging is at. Unlike the heap and linear rails,
        // this one's fallback is the mapping's own content generation rather
        // than zero — a seed overrides it and anything else leaves it alone.
        //
        // Derived from `serve` rather than assigned into a `mut`, so the value
        // the census reports and the value the candidate carries cannot get out
        // of step, and so a rail that serves nothing needs no gate here: it
        // answers `None` and this is `content_generation`, which is what that
        // arm always used.
        let seed_generation = serve
            .and_then(ResidentServe::seed_generation)
            .unwrap_or(content_generation);
        if serve.and_then(ResidentServe::seed_generation).is_some() {
            crate::observe::off(format!(
                "compute_stage_resident_skip mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} gen={seed_generation} bytes={need}"
            ));
        } else if let Some((_, generation)) = serve.and_then(ResidentServe::sample_source) {
            crate::observe::off(format!(
                "compute_stage_resident_sample mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} gen={generation} bytes={need}"
            ));
        }
        let mut bytes = vec![0u8; need];
        if serve.is_none()
            && !mapping_write::read_rect_raw_at(
                state,
                host,
                mapping_id,
                mapping_write::SurfaceWindow {
                    base_off: surface_offset,
                    bpr: surface_bpr,
                    span_end,
                    bpp,
                },
                mapping_write::Rect {
                    origin_x: 0,
                    origin_y: 0,
                    width,
                    height,
                },
                &mut bytes,
                tight,
            )
        {
            crate::observe::fail(format!(
                "compute_stage_tex mapper_ref_texture_fail reason=read mapping={mapping_id} {width}x{height} off={surface_offset} bpr={surface_bpr} span_end={span_end} pages={pages_n}"
            ));
            return Err(ComputeStatus::GuestIo(
                "compute_stage_tex_mapper_ref_texture_read",
            ));
        }
        let writeback = if is_storage {
            TextureWriteback::MapperRefTexture {
                mapping_id,
                surface_offset,
                surface_bpr,
                span_end,
                width,
                height,
                format: stage_fmt,
            }
        } else {
            TextureWriteback::None
        };
        if from_ref_texture {
            // Per-bind ref-texture stage SUCCESS census — not a failure; verbose-gated
            // (was always-on, ~300/boot). Genuine ref-texture stage failures above emit
            // `mapper_ref_texture_fail reason=<slug>` always-on.
            crate::observe::line(format!(
                "compute_stage_tex ref_texture_ok ref={texture_ref} sid={mapping_id} {width}x{height} fmt={stage_fmt:#x} pages={pages_n}"
            ));
        }
        return Ok(StagedTexture {
            binding,
            pixel_format: stage_fmt,
            storage_selector,
            // Metal forbids a mipmapped IOSurface texture.
            mip_levels: 1,
            width,
            height,
            bytes,
            is_storage,
            writeback,
            rail: R::stage(
                texture_ref,
                is_storage.then_some(ComputeStorageResidencyCandidate {
                    key: residency_key,
                    seed_generation,
                }),
                serve,
            ),
        });
    }

    // normal-texture linear. Fail-visible: name which gate rejected (live class:
    // silent ot=2 MissingTexture, journal 2026-07-14 compute census).
    // The reason travels *in* the status now, so this line and the caller's
    // both name the registered slug rather than a local shorthand only this
    // closure understood.
    let linear_fail = |st: ComputeStatus, detail: String| {
        crate::observe::fail(format!(
            "compute_stage_tex linear_fail reason={} ref={texture_ref} {detail}",
            st.reason()
        ));
        Err(st)
    };
    let (_entry, desc_bytes) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        stage_ref,
        &[OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS],
    ) {
        Ok(found) => found,
        Err(rung) => {
            return linear_fail(
                ComputeStatus::MissingTexture(crate::observe::ladder_slugs!("compute_linear_tex")(
                    rung,
                )),
                match rung {
                    objects::LadderRung::WrongType { got } => format!("ot={got}"),
                    objects::LadderRung::NoListEntry
                    | objects::LadderRung::DescRead { .. }
                    | objects::LadderRung::NoTaskSpace => String::new(),
                },
            );
        }
    };
    let Ok(tex) = decode_texture_descriptor(&desc_bytes) else {
        return linear_fail(
            ComputeStatus::MissingTexture(crate::observe::ladder_slug!(
                "compute_linear_tex",
                desc_decode
            )),
            format!("len={}", desc_bytes.len()),
        );
    };
    if tex.declared_pixel_format().is_none() {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_no_fmt"),
            String::new(),
        );
    }
    let Some(stage_format) =
        crate::runtime::draw::effective_view_sample_format(tex.pixel_format, view_pixel_format)
    else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_view_format"),
            format!(
                "base={stage_ref} base_fmt={:#x} view_fmt={view_pixel_format:?}",
                tex.pixel_format
            ),
        );
    };
    let Some(bpp) = pixel_format::bytes_per_pixel(stage_format) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_fmt_bytes"),
            format!("fmt={stage_format:#x}"),
        );
    };
    let storage_selector = pixel_format::storage_selector(stage_format);
    if is_storage && storage_selector.is_none() {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_fmt_storage"),
            format!("fmt={stage_format:#x}"),
        );
    }
    let Some((gva, layout)) = tex.level_gva(view_level, state.page_shift) else {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_no_level"),
            format!(
                "base={stage_ref} level={view_level} handle={:#x} alloc={} levels={} data_off={} page_shift={}",
                tex.handle,
                tex.allocation_size,
                tex.levels.len(),
                tex.data_offset,
                state.page_shift
            ),
        );
    };
    let w = layout.width;
    let h = layout.height;
    if w == 0 || h == 0 || layout.row_stride == 0 {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_zero_geom"),
            format!("{w}x{h} stride={}", layout.row_stride),
        );
    }
    let Some(tight) = (w as u64).checked_mul(bpp as u64).map(|v| v as usize) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_tight_overflow"),
            format!("{w}x{h} bpp={bpp}"),
        );
    };
    if layout.row_stride < tight as u64 {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_stride_lt_tight"),
            format!("stride={} tight={tight} {w}x{h}", layout.row_stride),
        );
    }
    let Some(need) = tight.checked_mul(h as usize) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_need_overflow"),
            format!("{w}x{h} bpp={bpp}"),
        );
    };
    // Which levels of the guest's declared chain this binding serves.
    //
    // A storage write names one level and a levelled view already exposes one,
    // so both stay at the base. Everything else stages the declared pyramid,
    // because `read(coord, lod)` and `sample(_, _, level(lod))` name a level of
    // it and an image built with only the base answers the first with nothing
    // and the second with level 0.
    let mut level_sources = vec![LinearLevelSource {
        gva,
        row_stride: layout.row_stride,
    }];
    if !is_storage && view_level == 0 {
        level_sources.extend(linear_extra_levels(
            &tex,
            state.page_shift,
            w,
            h,
            bpp,
            texture_ref,
        ));
    }
    let Some(pyramid) = reims_vgpu_protocol::extent::tight_pyramid_spans(
        w,
        h,
        level_sources.len() as u32,
        bpp as usize,
    ) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_pyramid_layout"),
            format!("{w}x{h} bpp={bpp} levels={}", level_sources.len()),
        );
    };
    let Some(pyramid_need) = pyramid
        .last()
        .and_then(|last| last.offset.checked_add(last.len))
    else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_pyramid_span"),
            format!("{w}x{h} bpp={bpp} levels={}", level_sources.len()),
        );
    };
    // Level 0 of the packed pyramid and the single image this window already
    // sized are two independent derivations of one length — `tight * h` here,
    // `mip_extent(w, 0) * mip_extent(h, 0) * bpp` there. If they ever
    // disagreed, the upload would be apportioned to levels by a layout the
    // reader below does not share, and level 1 would hold level 0's tail.
    if pyramid.first().map(|base| base.len) != Some(need) {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_pyramid_base"),
            format!(
                "base={:?} need={need} {w}x{h} bpp={bpp}",
                pyramid.first().map(|base| base.len)
            ),
        );
    }
    // The identity every cache question below asks about. One derivation so
    // the resident probe, the flush-and-serve, the plain serve and the
    // per-level read cannot drift from each other — and so a level's cache key
    // is that level's own rows and extent rather than the base's.
    let level_window = |source: &LinearLevelSource,
                        span: &reims_vgpu_protocol::extent::MipLevelSpan| {
        crate::runtime::surface_cache::LinearWindow {
            task_id,
            texture_ref: stage_ref,
            gva: source.gva,
            pixel_format: stage_format,
            width: span.width,
            height: span.height,
            row_stride: source.row_stride,
        }
    };
    // Only the base can be resident: a resident is one window at one level.
    let window = level_window(&level_sources[0], &pyramid[0]);
    // Linear-window residency identity — mirrors the host_linear_textures
    // entry exactly. Absent when the stride overflows the key field (no live
    // class; such a window simply stays on the bytes path).
    let span = layout.row_stride.saturating_mul(h as u64);
    let linear_key = (layout.row_stride <= u32::MAX as u64).then(|| {
        crate::model::ComputeStorageResidencyKey::linear(
            task_id,
            stage_ref,
            gva,
            layout.row_stride as u32,
            span,
            w,
            h,
            stage_format,
        )
    });
    let mut bytes = vec![0u8; pyramid_need];
    // Resident-authoritative window (deferred linear writeback): consume the
    // rail's resident without bytes when possible; otherwise flush it into the
    // entry first — falling through to the raw guest read would silently serve
    // the pre-chain seed pages.
    let resident = match (
        linear_key,
        crate::runtime::surface_cache::linear_texture_resident_gen(state, &window),
    ) {
        (Some(key), Some(resident_gen)) => Some((
            key,
            resident_gen,
            crate::backend::selected().resident_serve(key, resident_gen, is_storage, stage_format),
        )),
        _ => None,
    };
    // A resident is one window at one level, so it can only answer for the
    // base. Serving a pyramid from it would leave every level above the base
    // unwritten — which is exactly the defect the pyramid repairs — so a
    // multi-level binding reads its own bytes and the rail refuses the pair
    // outright as `vk_compute_exec_resident_sample_is_not_a_pyramid`.
    let serve = if level_sources.len() > 1 {
        None
    } else {
        resident.and_then(|(_, _, serve)| serve)
    };
    if let Some(generation) = serve.and_then(ResidentServe::seed_generation) {
        crate::observe::off(format!(
            "compute_stage_linear_resident_seed task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={generation}",
            tex.pixel_format
        ));
    } else if let Some((_, generation)) = serve.and_then(ResidentServe::sample_source) {
        crate::observe::off(format!(
            "compute_stage_linear_resident_sample task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={generation}",
            stage_format
        ));
    }
    if serve.is_some() {
        // The rail's resident serves this window; no cache/guest read.
    } else {
        // Level 0 is the window built above; every level after it is the same
        // read against that level's own rows, so the cache is consulted per
        // level and one level's bytes can never answer for another's.
        for (span, source) in pyramid.iter().zip(level_sources.iter()) {
            let level_tight = span.len / span.height.max(1) as usize;
            read_linear_level(
                state,
                host,
                task_id,
                texture_ref,
                &level_window(source, span),
                source.gva,
                source.row_stride,
                level_tight,
                span.height,
                &mut bytes[span.offset..span.offset + span.len],
            )?;
        }
    }
    let writeback = if is_storage {
        TextureWriteback::Linear {
            texture_ref: stage_ref,
            gva,
            pixel_format: stage_format,
            row_stride: layout.row_stride,
            width: w,
            height: h,
            bpp,
            pages: staged_window_pages(state, host, task_id, gva, layout.row_stride, h),
        }
    } else {
        TextureWriteback::None
    };
    // Deferred-writeback candidacy: a linear storage output of a format the
    // BGRA mirror ignores keeps the engine resident authoritative — the
    // readback, cache store, and next chained upload all disappear (the
    // fade-window blur pyramid class). If the GVA is
    // mapped at writeback time (the sync path would have written guest
    // pages), the deferred-writeback arm records a flush obligation with a
    // defer-time page index so aliased raw-GVA readers land it first.
    let mut residency = None;
    if is_storage {
        if let Some(key) = linear_key {
            if !crate::runtime::surface_cache::linear_mirrorable(stage_format) {
                let seed = serve
                    .and_then(ResidentServe::seed_generation)
                    .unwrap_or_else(|| {
                        state
                            .host_linear_textures
                            .get(&(task_id, stage_ref))
                            .map(|e| e.host_gen)
                            .unwrap_or(0)
                    });
                residency = Some(ComputeStorageResidencyCandidate {
                    key,
                    seed_generation: seed,
                });
            }
        }
    }
    Ok(StagedTexture {
        binding,
        pixel_format: stage_format,
        storage_selector,
        // The levels actually placed, which is the declared count when the
        // descriptor places all of them and a reported-short prefix when it
        // does not.
        mip_levels: level_sources.len() as u32,
        width: w,
        height: h,
        bytes,
        is_storage,
        writeback,
        rail: R::stage(texture_ref, residency, serve),
    })
}

/// Fill `dst` with one linear texture level's tight rows, from the surface
/// cache when it still holds this exact window and from guest pages otherwise.
///
/// Its own function because a mip chain reads the same thing once per level
/// against a different `gva`, `row_stride` and extent, and a loop that inlined
/// this would have been the place where level `n` was read with level 0's
/// stride.
#[allow(
    clippy::too_many_arguments,
    reason = "a level is its own window, gva, stride, extent and destination, and \
              collapsing them into a struct here would hide which of them a caller varies"
)]
fn read_linear_level<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    window: &crate::runtime::surface_cache::LinearWindow,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    dst: &mut [u8],
) -> Result<(), ComputeStatus> {
    if let Some(cached) = crate::runtime::surface_cache::get_linear_texture(state, window) {
        if cached.len() == dst.len() {
            dst.copy_from_slice(cached);
            crate::observe::off(format!(
                "compute_stage_tex linear_cache task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={}x{height} row_stride={row_stride}",
                window.pixel_format, window.width
            ));
            return Ok(());
        }
        // A cache entry keyed to this window whose length is not this window's
        // is a key that stopped identifying its contents. Read the guest pages
        // rather than serve it, and say so — silently trusting it is how one
        // level's texels would reach another's.
        crate::observe::fail(format!(
            "compute_stage_tex linear_cache_len task={task_id} ref={texture_ref} gva={gva:#x} cached={} want={}",
            cached.len(),
            dst.len()
        ));
    }
    // The bulk/row reads below walk raw task GVAs; a Store's
    // guest-page write is submitted and not waited on.
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    crate::runtime::render_writeback::settle_guest_writes(
        crate::runtime::render_writeback::SettleSite::ComputeStageTexture,
    );
    if read_linear_texture_bulk(state, host, task_id, gva, row_stride, tight, height, dst) {
        // One cached-view walk for the whole span (render-path bulk analog).
        return Ok(());
    }
    let mut row = vec![0u8; tight];
    for y in 0..height {
        let row_gva = gva
            .checked_add(
                (y as u64)
                    .checked_mul(row_stride)
                    .ok_or(ComputeStatus::GuestIo(
                        "compute_stage_tex_linear_row_offset",
                    ))?,
            )
            .ok_or(ComputeStatus::GuestIo("compute_stage_tex_linear_row_gva"))?;
        if let Err(e) = gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            row_gva,
            &mut row,
            state.page_shift,
        ) {
            // First failing row only — full walk status for one-boot diagnosis.
            if y == 0 {
                let walk = gva_mem::diagnose_gva_walk(
                    host,
                    &state.tasks,
                    task_id,
                    row_gva,
                    state.page_shift,
                );
                crate::observe::fail(format!(
                    "compute_stage_tex_gva task={task_id} ref={texture_ref} gva={row_gva:#x} y=0 page_shift={} err={e:?} | {walk}",
                    state.page_shift
                ));
            }
            return Err(ComputeStatus::GuestIo("compute_stage_tex_linear_row_read"));
        }
        let off = (y as usize) * tight;
        dst[off..off + tight].copy_from_slice(&row);
    }
    Ok(())
}

/// One level of a linear texture as this rail stages it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LinearLevelSource {
    gva: u64,
    row_stride: u64,
}

/// Levels 1.. of a normal texture's declared mip chain, as far as the
/// descriptor actually places them.
///
/// The prefix, not the set: a level that will not resolve makes every level
/// above it unreachable too, because the packed pyramid the host image is built
/// from has no way to express a hole. Truncation is dropped guest work, so it is
/// reported by name rather than left to read as a texture that simply has fewer
/// levels.
///
/// Extents are checked against [`reims_vgpu_protocol::extent::mip_extent`] because
/// the packed layout is derived from the base geometry alone; a level whose
/// declared extent disagrees would be read at one size and copied at another.
fn linear_extra_levels(
    tex: &crate::runtime::decode::resource::TextureDescriptor,
    page_shift: u32,
    base_width: u32,
    base_height: u32,
    bpp: u32,
    texture_ref: u32,
) -> Vec<LinearLevelSource> {
    let declared = tex.mipmap_level_count.max(1);
    let mut out = Vec::new();
    for level in 1..declared {
        let want_w = reims_vgpu_protocol::extent::mip_extent(base_width, level);
        let want_h = reims_vgpu_protocol::extent::mip_extent(base_height, level);
        let refuse = |reason: &str, detail: String| {
            crate::observe::fail(format!(
                "compute_stage_tex mip_truncated reason={reason} ref={texture_ref} level={level}                  staged={} declared={declared} want={want_w}x{want_h} {detail}",
                level
            ));
        };
        let Some((level_gva, layout)) = tex.level_gva(level, page_shift) else {
            refuse("no_level", String::new());
            break;
        };
        if layout.width != want_w || layout.height != want_h {
            refuse("extent", format!("got={}x{}", layout.width, layout.height));
            break;
        }
        if layout.row_stride < u64::from(want_w).saturating_mul(u64::from(bpp)) {
            refuse("stride_lt_tight", format!("stride={}", layout.row_stride));
            break;
        }
        out.push(LinearLevelSource {
            gva: level_gva,
            row_stride: layout.row_stride,
        });
    }
    out
}

/// Read a strided linear texture span through one cached GVA view (a single
/// page-table walk for the whole texture), de-striding rows into `bytes`
/// (tight rows). Returns `false` when the span cannot be packed — the caller
/// falls back to the per-row walk. Live transition cost of the per-row walk
/// was ~8–23 ms of `stage_us` per Core Image dispatch.
#[allow(
    clippy::too_many_arguments,
    reason = "the bulk path keeps the decoded texture window and row layout explicit"
)]
fn read_linear_texture_bulk<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    bytes: &mut [u8],
) -> bool {
    if height == 0 || tight == 0 || bytes.len() < (height as usize).saturating_mul(tight) {
        return false;
    }
    if row_stride == tight as u64 {
        return crate::runtime::gva_view::read_span(state, host, task_id, gva, bytes);
    }
    let Some(span_len) = (height as u64 - 1)
        .checked_mul(row_stride)
        .and_then(|v| v.checked_add(tight as u64))
    else {
        return false;
    };
    let Some((ptr, avail)) =
        crate::runtime::gva_view::host_ptr_for_span(state, host, task_id, gva, span_len)
    else {
        return false;
    };
    if (avail as u64) < span_len {
        return false;
    }
    for y in 0..height as usize {
        let src = (y as u64).saturating_mul(row_stride) as usize;
        let dst = y * tight;
        // SAFETY: host_ptr_for_span guarantees `span_len` readable bytes at
        // `ptr`; `src + tight <= span_len` for every row by construction.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.add(src),
                bytes[dst..dst + tight].as_mut_ptr(),
                tight,
            );
        }
    }
    true
}

/// Write tight rows of a linear storage texture through one fresh-walked
/// span mapping. Stride padding bytes are left untouched —
/// consumers address rows by `row_stride`, so padding is dead space and
/// writing it is never observable. Returns `false` when the span cannot be
/// packed or the write is outside the task's recorded map spans — the caller
/// falls back to the per-row walk (which fails visibly per contract).
#[allow(
    clippy::too_many_arguments,
    reason = "the bulk path keeps the decoded texture window and row layout explicit"
)]
fn write_linear_texture_bulk<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    bytes: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> bool {
    if height == 0 || tight == 0 || bytes.len() < (height as usize).saturating_mul(tight) {
        return false;
    }
    let Some(span_len) = (height as u64 - 1)
        .checked_mul(row_stride)
        .and_then(|v| v.checked_add(tight as u64))
    else {
        return false;
    };
    // Fresh PT walk at write time — never a cached view (stale-view class) —
    // carrying `allowed` so a deferred window's bytes cannot reach a page
    // outside the set it was armed on, however the guest re-points the range
    // between the flush decision and this walk.
    let Some(span_map) = crate::runtime::gva_view::map_fresh_span_within(
        state, host, task_id, gva, span_len, allowed,
    ) else {
        return false;
    };
    let ptr = span_map.ptr;
    for y in 0..height as usize {
        let src = y * tight;
        let dst = (y as u64).saturating_mul(row_stride) as usize;
        // SAFETY: map_fresh_span guarantees `span_len` writable bytes at
        // `ptr`; `dst + tight <= span_len` for every row by construction.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes[src..src + tight].as_ptr(), ptr.add(dst), tight);
        }
    }
    crate::runtime::gva_view::unmap_fresh_span(host, span_map);
    true
}

fn writeback_texture<R: RailStage, M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &StagedTexture<R>,
) -> Result<(), ComputeStatus> {
    // Which destination namespace a compute storage output lands in, and — on
    // the linear arm — whether its guest rows are dense. Both are properties of
    // the guest's own window rather than of this device, and neither is
    // otherwise reported: `rectwr_*` and `linear*` are shared with several
    // other rails, so they cannot be read as this rail's split. Observed only;
    // nothing branches on these.
    match &tex.writeback {
        TextureWriteback::None => crate::runtime::drain::note_store_route("compute_wb_none"),
        TextureWriteback::Linear {
            width,
            bpp,
            row_stride,
            ..
        } => {
            crate::runtime::drain::note_store_route("compute_wb_linear");
            // A dense window is one `VkBufferCopy` run per guest run; a padded
            // one needs a rectangle copy per run per row fragment, which is the
            // difference between a handful of regions and a few hundred.
            crate::runtime::drain::note_store_route(
                if u64::from(*width) * u64::from(*bpp) == *row_stride {
                    "compute_wb_linear_dense"
                } else {
                    "compute_wb_linear_padded"
                },
            );
        }
        TextureWriteback::MapperRefTexture {
            width,
            format,
            surface_bpr,
            ..
        } => {
            crate::runtime::drain::note_store_route("compute_wb_mapper_ref_texture");
            let tight = pixel_format::bytes_per_pixel(*format).map(|bpp| width.saturating_mul(bpp));
            crate::runtime::drain::note_store_route(if tight == Some(*surface_bpr) {
                "compute_wb_mapper_ref_texture_dense"
            } else {
                "compute_wb_mapper_ref_texture_padded"
            });
        }
    }

    match &tex.writeback {
        TextureWriteback::None => Ok(()),
        TextureWriteback::Linear {
            texture_ref,
            gva,
            pixel_format,
            row_stride,
            width,
            height,
            bpp,
            pages,
        } => {
            let tight = (*width as usize) * (*bpp as usize);
            let required = tight.saturating_mul(*height as usize);
            if tight > *row_stride as usize || tex.bytes.len() < required {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=linear_layout bind={} gva={gva:#x} dims={}x{} bpp={} row_stride={} tight={} bytes={} required={required}",
                    tex.binding,
                    width,
                    height,
                    bpp,
                    row_stride,
                    tight,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo("compute_wb_tex_linear_layout"));
            }
            let window = crate::runtime::surface_cache::LinearWindow {
                task_id,
                texture_ref: *texture_ref,
                gva: *gva,
                pixel_format: *pixel_format,
                width: *width,
                height: *height,
                row_stride: *row_stride,
            };
            if !crate::runtime::surface_cache::store_linear_texture(state, &window, &tex.bytes) {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=linear_cache_store task={task_id} ref={texture_ref} bind={} gva={gva:#x} fmt={pixel_format:#x} dims={}x{} bpp={} row_stride={} bytes={}",
                    tex.binding,
                    width,
                    height,
                    bpp,
                    row_stride,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo("compute_wb_tex_linear_cache_store"));
            }
            crate::runtime::surface_cache::mirror_linear_color_cache(
                state, host, &window, &tex.bytes,
            );
            // Kept although the span is no longer needed here: the overflow is
            // a real refusal with a name, and `write_linear_guest_within` would only
            // return a bare `false` for it.
            let Some(_span) = row_stride.checked_mul(*height as u64) else {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=linear_span_overflow task={task_id} ref={texture_ref} bind={} gva={gva:#x} dims={}x{} row_stride={row_stride}",
                    tex.binding, width, height
                ));
                return Err(ComputeStatus::GuestIo(
                    "compute_wb_tex_linear_span_overflow",
                ));
            };
            // This used to return early on `reason=linear_unmapped` whenever the
            // range fell outside the task's notified spans, which on a live boot
            // discarded six glyph-atlas writebacks a boot (79x52, 90x20, 8x8 …)
            // whose pages were mapped the whole time — only the notification had
            // not arrived. The graceful degradation it provided is real and is
            // kept; what changed is that it is now keyed on the condition itself
            // rather than on a proxy that also catches healthy writes.
            match write_linear_guest_within(
                state,
                host,
                task_id,
                *gva,
                *row_stride,
                tight,
                *height,
                &tex.bytes,
                &format!("bind={}", tex.binding),
                (!pages.membership().is_empty()).then_some(pages.membership()),
            ) {
                LinearWrite::Written => {
                    // The mirror above cached these bytes as unevictable
                    // because the write had not happened yet. It has, so the
                    // guest can re-derive them and the byte cap may reclaim the
                    // entry. The `Unmapped` arm below deliberately does not:
                    // its own comment is that the host cache keeps the
                    // authoritative bytes.
                    crate::runtime::surface_cache::note_gva_landed(state, *gva);
                    Ok(())
                }
                // Nothing resolves under this task, so there is nowhere to put
                // the result. The host cache keeps the authoritative bytes and
                // sampling still serves them, so failing the whole dispatch
                // would cost more than it protects.
                LinearWrite::Unmapped => {
                    crate::observe::fail(format!(
                        "compute_writeback_tex cache_only reason=linear_unmapped task={task_id} ref={texture_ref} bind={} gva={gva:#x} fmt={pixel_format:#x} dims={}x{} bpp={} row_stride={row_stride}",
                        tex.binding, width, height, bpp
                    ));
                    Ok(())
                }
                LinearWrite::Failed => {
                    Err(ComputeStatus::GuestIo("compute_wb_tex_linear_guest_write"))
                }
            }
        }
        TextureWriteback::MapperRefTexture {
            mapping_id,
            surface_offset,
            surface_bpr,
            span_end,
            width,
            height,
            format,
        } => {
            // Derived here rather than carried beside the format, so the record
            // holds one answer about its texel size. Staging refused this bind
            // outright if the format had no byte width, so a `None` here is a
            // format that changed identity between stage and landing rather
            // than an unsupported one — refuse it by name instead of writing
            // rows at a width nothing declared.
            let Some(bpp) = pixel_format::bytes_per_pixel(*format) else {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=mapper_ref_texture_format_unsized task={task_id} bind={} mid={mapping_id} fmt={format:#x}",
                    tex.binding
                ));
                return Err(ComputeStatus::GuestIo(
                    "compute_wb_tex_mapper_ref_texture_format",
                ));
            };
            let tight = width.saturating_mul(bpp);
            if !mapping_write::write_full_rect_raw_at(
                state,
                host,
                *mapping_id,
                *surface_offset,
                *surface_bpr,
                *span_end,
                *width,
                *height,
                bpp,
                &tex.bytes,
                tight,
            ) {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=mapper_ref_texture_mapping_write task={task_id} bind={} mid={} surface_offset={surface_offset:#x} surface_bpr={} span_end={span_end:#x} dims={}x{} bpp={} bytes={} tight={tight}",
                    tex.binding,
                    mapping_id,
                    surface_bpr,
                    width,
                    height,
                    bpp,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo(
                    "compute_wb_tex_mapper_ref_texture_write",
                ));
            }
            Ok(())
        }
    }
}

/// What a linear guest writeback did, for callers that must tell "there is
/// nowhere to put this" apart from "putting it there went wrong".
///
/// A bare `bool` collapsed those, and the collapse was load-bearing: the only
/// caller able to degrade gracefully was doing so off a *different* condition
/// (the range being outside the task's notified spans) that also caught healthy
/// writes. `-> bool` crossing a module boundary is exactly where that regrows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearWrite {
    /// Every row landed in guest memory.
    Written,
    /// The task's page tables resolve nothing at this GVA, so no write was
    /// possible. Callers keep the host cache and carry on.
    Unmapped,
    /// A write was attempted and did not complete — bad layout, an arithmetic
    /// overflow, or a per-row refusal. Already fail-logged with its own reason.
    Failed,
}

/// Write tight-row `bytes` into a strided linear guest window through fresh
/// task page-table walks (bulk view when packable, per-row fallback), bounded
/// to the guest pages the caller was authorised to write. Fail lines carry
/// `ctx` for the call site.
///
/// There is no unbounded sibling. Both doors onto this rail — the deferred
/// flush and the post-dispatch writeback — hand content produced earlier to a
/// walk taken later, so both need the bound; a wrapper passing `None` would
/// only be a way to reach the rail without one.
///
/// The linear compute rail defers exactly as the GVA render rail does, and it
/// re-walks at flush time for the same reason, so it has the same hazard and
/// takes the same answer: the armed page set travels into the walk that
/// resolves the destination, and both the bulk view and the per-row fallback
/// carry it. Leaving the bound on one of the two would make it depend on how
/// the guest happened to lay the pages out.
#[allow(
    clippy::too_many_arguments,
    reason = "the linear writer mirrors the window's guest geometry"
)]
pub(crate) fn write_linear_guest_within<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    bytes: &[u8],
    ctx: &str,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> LinearWrite {
    if write_linear_texture_bulk(
        state, host, task_id, gva, row_stride, tight, height, bytes, allowed,
    ) {
        return LinearWrite::Written;
    }
    // The bulk path declines for several reasons and the per-row fallback below
    // covers all but one of them. The exception is "nothing is mapped here",
    // which no amount of retrying per row can fix, so it is answered once here
    // rather than discovered `height` times.
    if !crate::runtime::gva_mem::any_task_gva_page_resolves(
        host,
        &state.tasks,
        task_id,
        gva,
        1,
        state.page_shift,
    ) {
        return LinearWrite::Unmapped;
    }
    let mut row = vec![0u8; row_stride as usize];
    for y in 0..height {
        let src_off = (y as usize) * tight;
        row[..tight].copy_from_slice(&bytes[src_off..src_off + tight]);
        // Pad rest of row with zeros already present.
        let Some(row_offset) = (y as u64).checked_mul(row_stride) else {
            crate::observe::fail(format!(
                "compute_writeback_tex fail reason=linear_row_offset_overflow {ctx} gva={gva:#x} y={y} row_stride={row_stride}"
            ));
            return LinearWrite::Failed;
        };
        let Some(row_gva) = gva.checked_add(row_offset) else {
            crate::observe::fail(format!(
                "compute_writeback_tex fail reason=linear_gva_overflow {ctx} gva={gva:#x} y={y} row_offset={row_offset:#x}"
            ));
            return LinearWrite::Failed;
        };
        if let Err(e) = gva_mem::write_task_gva_product_within(
            state,
            host,
            task_id,
            row_gva,
            &row[..row_stride as usize],
            allowed,
        ) {
            crate::observe::fail(format!(
                "compute_writeback_tex fail reason=linear_gva_write task={task_id} {ctx} gva={row_gva:#x} y={y} row_stride={row_stride} height={height} err={e:?}"
            ));
            return LinearWrite::Failed;
        }
    }
    LinearWrite::Written
}

fn writeback_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    pipe_ref: Option<u32>,
    context: &str,
    staged: &StagedBuffer,
) -> Result<(), ComputeStatus> {
    if let Err(e) = gva_mem::write_task_gva_product_within(
        state,
        host,
        task_id,
        staged.gva,
        &staged.bytes,
        (!staged.pages.is_empty()).then_some(&staged.pages),
    ) {
        crate::observe::fail(format!(
            "compute_writeback_buf fail reason=task_gva_write task={task_id} pipe={} context={context} idx={} ref={} gva={:#x} len={} off={:#x} err={e:?}",
            pipe_ref
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".into()),
            staged.bind.index,
            staged.bind.buffer_ref,
            staged.gva,
            staged.bytes.len(),
            staged.bind.offset
        ));
        return Err(ComputeStatus::GuestIo("compute_wb_buf_task_gva_write"));
    }
    Ok(())
}

/// An absent IOSurface pixel format means BGRA8: a mapper-ref-texture surface the guest
/// mapped without a format word is scanout-ordered by the display contract, and
/// this is the one place that default is written down.
fn or_bgra8(pixel_format: u16) -> u16 {
    if pixel_format != 0 {
        pixel_format
    } else {
        pixel_format::MTL_FORMAT_BGRA8_UNORM
    }
}

/// Latched geometry and pixel format of a mapper-ref-texture mapping, for a surface whose
/// own IOSurface descriptor could not be read.
///
/// Three separate descriptor failures share this fallback, and spelling it out at
/// each of them made one block of nineteen lines appear three times in a row.
fn mapping_geom_format(
    state: &DeviceState,
    mapping_id: u32,
) -> Result<(u32, u32, u16), ComputeStatus> {
    let m = state
        .mappings
        .get(&mapping_id)
        .ok_or(ComputeStatus::MissingTexture(
            "compute_stage_tex_mapping_gone",
        ))?;
    if !m.has_geom || m.width == 0 || m.height == 0 {
        return Err(ComputeStatus::MissingTexture(
            "compute_stage_tex_mapping_no_geom",
        ));
    }
    Ok((m.width, m.height, or_bgra8(m.format)))
}

fn u32_dim(v: u64) -> Result<u32, ComputeStatus> {
    if v == 0 || v > u32::MAX as u64 {
        Err(ComputeStatus::BadGrid("compute_grid_dim_range"))
    } else {
        Ok(v as u32)
    }
}

/// The dispatch extents, narrowed from the wire's `u64` by [`u32_dim`].
///
/// The type is [`reims_vgpu_protocol::extent::Extent3`], which both this decoder
/// and the Metal backend it dispatches through now name. It used to be private
/// here, which protected construction and stopped at the backend call — see its
/// doc for why that was the wrong half of the journey to protect.
use reims_vgpu_protocol::extent::Extent3;

// The two rails. Named rather than re-exported flat: each owns a dispatch
// executor with the same neutral signature, and this module reaches whichever
// one the process runs on only through `Backend`.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub mod metal;
#[cfg(feature = "backend-vulkan")]
pub mod stall_watchdog;
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

// The two constructors are free functions here rather than an inherent `impl`
// on `Extent3`, because both refuse with `ComputeStatus` and one reads a decoded
// `Size3` — device vocabulary the contract crate cannot name, and Rust's orphan
// rule says so. The extent type stays shared; only the narrowing that produces
// it from *this* device's wire belongs to this decoder.

/// An [`Extent3`] from a lifted record extent, refusing each component out of
/// range.
fn extent_from_wire(s: RecordExtent) -> Result<Extent3, ComputeStatus> {
    Ok(Extent3 {
        x: u32_dim(s.width)?,
        y: u32_dim(s.height)?,
        z: u32_dim(s.depth)?,
    })
}

/// An [`Extent3`] from three consecutive LE `u32`s of an indirect-arguments
/// buffer at `at`. One stride expression rather than six offset literals: the
/// literals were `0, 4, 8` and `12, 16, 20` written out, where a transposition
/// is invisible.
fn extent_from_indirect(raw: &[u8], at: usize) -> Result<Extent3, ComputeStatus> {
    Ok(Extent3 {
        x: u32_dim(u64::from(ld32(&raw[at..])))?,
        y: u32_dim(u64::from(ld32(&raw[at + 4..])))?,
        z: u32_dim(u64::from(ld32(&raw[at + 8..])))?,
    })
}

/// Grid and threadgroup extents for one dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchDims {
    grid: Extent3,
    threadgroup: Extent3,
    /// The guest asked for `dispatchThreads` — an exact thread count — rather
    /// than whole threadgroups.
    dispatch_threads: bool,
}

/// [`resolve_dispatch_dims`], with the refusal named on the always-on log.
///
/// Both dispatch executors want the same thing on failure: the decline, the
/// command kind, the wire grid and threadgroup, and how many textures were
/// bound. Naming it once is what keeps the Metal and Vulkan arms from
/// drifting into two spellings of the same refusal.
fn resolve_dispatch_dims_reported<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    dispatch: &DispatchRecord,
    acc: &ComputeAccum,
) -> Result<DispatchDims, ComputeStatus> {
    // A bind the accumulator could not hold refuses the dispatch here, before
    // either executor reads the state. It is checked at this gate rather than
    // at the bind because the bind walk has no dispatch to refuse — and a
    // dispatch that runs with the guest's binding simply absent is a wrong
    // result the guest is never told about, which is the one thing this device
    // is not allowed to do. The slot is past Metal's own argument table, so a
    // firing is a record Apple's serializer cannot emit; refusing costs a
    // healthy zero and buys the guarantee.
    if let Some(over) = acc.refused_bind {
        let (index, arg, cap) = over.parts();
        crate::observe::Emit::decline("compute_dispatch", &over)
            .field("kind", format!("{:?}", dispatch_kind(dispatch)))
            .field("refused_index", index)
            .field("refused_arg", arg)
            .field("table", cap)
            .fail_once(u64::from(index));
        return Err(ComputeStatus::Unsupported(
            "compute_dispatch_bind_past_table",
        ));
    }
    resolve_dispatch_dims(state, host, task_id, dispatch).inspect_err(|e| {
        // The extents come from whichever the record actually carries. An
        // indirect dispatch states no grid, and the flat command reported one
        // anyway — three zeroes that read as a grid the guest asked for.
        let (grid, threadgroup) = match dispatch {
            DispatchRecord::Threadgroups(r) => (Some(r.groups), Some(r.threads_per_group)),
            DispatchRecord::Threads(r) => (Some(r.threads), Some(r.threads_per_group)),
            DispatchRecord::ThreadgroupsIndirect(r) => (None, Some(r.threads_per_group)),
            DispatchRecord::ThreadsIndirect(_) => (None, None),
        };
        let extent = |e: Option<RecordExtent>| {
            e.map_or_else(
                || "from-buffer".to_string(),
                |e| format!("[{},{},{}]", e.width, e.height, e.depth),
            )
        };
        crate::observe::line(format!(
            "compute_resolve_dims fail {e:?} kind={:?} grid={} tg={} ntex={}",
            dispatch_kind(dispatch),
            extent(grid),
            extent(threadgroup),
            acc.textures.len()
        ));
    })
}

/// Which of the four dispatches this is, for a line that reports one.
fn dispatch_kind(dispatch: &DispatchRecord) -> reims_vgpu_protocol::compute::ComputeKind {
    ComputeRecord::Dispatch(*dispatch).kind()
}

/// Resolve grid/threadgroup dims for direct or indirect dispatches.
///
/// Total over the four dispatches: there is no "not a dispatch" arm left to
/// refuse, because a record that is not one cannot be passed. `DispatchRecord`
/// carries exactly the fields its own form has — the fully indirect dispatch
/// has no threadgroup extent at all, where the flat command carried a zeroed
/// one — so no arm reads a field the record did not state.
fn resolve_dispatch_dims<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    dispatch: &DispatchRecord,
) -> Result<DispatchDims, ComputeStatus> {
    match dispatch {
        // Every dimension comes from the wire. `u32_dim` refuses `0` and
        // anything past `u32::MAX` with `BadGrid("compute_grid_dim_range")`, so
        // a malformed grid is a named refusal rather than a substitution.
        DispatchRecord::Threadgroups(r) => Ok(DispatchDims {
            grid: extent_from_wire(r.groups)?,
            threadgroup: extent_from_wire(r.threads_per_group)?,
            dispatch_threads: false,
        }),
        DispatchRecord::Threads(r) => Ok(DispatchDims {
            grid: extent_from_wire(r.threads)?,
            threadgroup: extent_from_wire(r.threads_per_group)?,
            dispatch_threads: true,
        }),
        DispatchRecord::ThreadgroupsIndirect(r) => {
            let raw = read_buffer_window(
                state,
                host,
                task_id,
                r.source.buffer_ref,
                r.source.offset,
                INDIRECT_THREADGROUPS_ARGS_LEN,
            )?;
            Ok(DispatchDims {
                grid: extent_from_indirect(&raw, 0)?,
                threadgroup: extent_from_wire(r.threads_per_group)?,
                dispatch_threads: false,
            })
        }
        DispatchRecord::ThreadsIndirect(r) => {
            let raw = read_buffer_window(
                state,
                host,
                task_id,
                r.source.buffer_ref,
                r.source.offset,
                INDIRECT_THREADS_ARGS_LEN,
            )?;
            // MTLDispatchThreadsIndirectArguments: threadsPerGrid[3], threadsPerThreadgroup[3].
            Ok(DispatchDims {
                grid: extent_from_indirect(&raw, 0)?,
                threadgroup: extent_from_indirect(&raw, 12)?,
                dispatch_threads: true,
            })
        }
    }
}

#[cfg(test)]
mod tests;
