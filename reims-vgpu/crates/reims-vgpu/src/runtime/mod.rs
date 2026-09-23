//! Behavior: turn guest work into HostActions / backend jobs.
//!
//! Drain FIFOs, parse wire (using [`crate::contract`]), resolve memory, plan
//! ops, update [`crate::model`] state. No GPU API calls here.

/// The split of [`chain_phase`]'s largest column, `binds_us`.
pub mod bind_phase;
/// Product-path blit fill/copy execution against guest GVA.
pub mod blit_exec;
/// Draw-time buffer binds, resolved once per reference and held until the
/// guest moves the addresses under them.
///
/// Ungated: a bind window is a `GuestRun` over this device's own import of
/// guest RAM and names nothing a rail owns. Which rail fills the registry is a
/// fact about that rail, and it is answered by the registry being empty.
pub mod bound_buffers;
/// Guest-declared write generations for task-local GVA resources.
pub mod buffer_write_gen;
/// Always-on proxies and censuses, one per measured bug class.
pub mod census;
/// Where a draw chain's wall clock goes on the runtime side of the engine
/// boundary, which is 82% of it.
pub mod chain_phase;
/// The byte runs in which a newly rendered row differs from the guest's.
pub mod changed_runs;
/// Product-path compute bind/dispatch (pipeline + buffers + direct dispatch).
// See the note on `backend::metal`: `Status` is a 264-byte `Copy` payload
// carried on failure paths, and boxing it would cost the refusal vocabulary
// that makes each one greppable.
#[allow(clippy::result_large_err, clippy::large_enum_variant)]
pub mod compute_exec;
/// Multi-record compute encoder session (control-flow SPI + ICB execute).
// See the note on `backend::metal`: `Status` is a 264-byte `Copy` payload
// carried on failure paths, and boxing it would cost the refusal vocabulary
// that makes each one greppable.
#[allow(clippy::result_large_err, clippy::large_enum_variant)]
pub mod compute_session;
pub mod decode;
pub mod drain;
/// The always-on log sink every decline and census writes to
/// (`/tmp/reims-vgpu-fail.log`); `line()` is the `REIMS_VGPU_DRAW_LOG=1`-gated tier.
/// CmdExecIndirect2 stream walk + mapper-ref-texture resolve.
pub mod exec;
/// Product-path event + encoder fence sync (event/blit/compute/render domains).
pub mod fence_exec;
/// Is the hypervisor's guest-write generation a sound cache key for the
/// zero-copy sampled gathers? Measurement, not policy.
pub mod gather_witness;
/// Guest-physical control-plane writes via HostOps map_pages.
pub mod gpa_map;
/// The last few pieces of work handed to the GPU, so a host GPU hang can name
/// what it was running instead of only that a fence stopped signalling.
pub mod gpu_hang_trail;
/// The bound on every GPU reference to guest RAM — one import per RAMBlock,
/// and the only type that can name a byte inside one.
///
/// Re-exported from `reims_vgpu_memory` under the path every caller already
/// writes (`crate::runtime::guest_ram::…`) so moving the module out did not
/// move a call site. The crate boundary is what makes the bound structural:
/// nothing there can see a resource table, a Vulkan handle or a QEMU
/// structure, so there is no second way to answer "how big is this" and no
/// place for a raw pointer and an offset to escape together.
pub use reims_vgpu_memory as guest_ram;
/// This process's imports of guest RAM, and the one place a guest physical
/// address becomes a bindable reference.
pub mod guest_ram_map;
/// Scattered guest windows → image-copy rectangles. Pure arithmetic, ungated so
/// both backends and every test arm reach it.
/// Task GVA → guest RAM reads.
pub mod gva_mem;
/// Which GVA render targets a Store has stamped, and what the two write
/// witnesses said at the time. Neutral: it keys on the guest span a rail's
/// resident stands for — see `Backend::gva_witness_key` — and not on the rail's
/// own name for that resident.
pub mod gva_store_witness;
/// Task-GVA HostOps views (MapMemory2 / UnmapMemory lifecycle).
pub mod gva_view;
/// CmdHeapTextureSizeAndAlign wire decode + host requirement query.
pub mod heap_query;
pub mod host;
/// Which guest pages this device has written, and when — the half of the
/// guest-write witness the hypervisor's dirty bitmap cannot supply.
pub mod host_writes;
/// Serializer-object ICB (0x36) materialization, host command fills, execute writeback.
pub mod icb;

/// Metal draw encode + writeback when MTLBs resolve.
// See the note on `backend::metal`: `Status` is a 264-byte `Copy` payload
// carried on failure paths, and boxing it would cost the refusal vocabulary
// that makes each one greppable.
#[allow(clippy::result_large_err, clippy::large_enum_variant)]
pub mod draw;
/// From a drained packet to the packet the semantic model admits.
pub mod ingress;
pub mod input;
/// Process-global metal2vulkan SPIR-V cache (AIR content hash → SPIR-V).
pub mod m2v_cache;
/// IOSurface mapper capture + page-table resolve.
pub mod map_audit;
pub mod mapper;
/// Write host BGRA into guest mapping pages (render writeback).
pub mod mapping_write;
/// generateMipmaps for multi-mip normal-texture linear textures.
pub mod mipmap;
pub mod mmio;
/// MTLB container → wrapped-AIR carve for metal2vulkan.
pub mod mtlb;
pub mod node_guard;
/// Object-list lookup and mapper-ref-texture registration.
pub mod objects;
/// The bytes an admitted packet is executed from, held while the model decides
/// when it runs.
pub mod parked;
pub mod plan;
/// Whether a range's page-table entries are in the state the guest's own next
/// edit of them requires — the direction that is ordered is the map.
pub mod range_coverage;
pub mod released_pages;
/// Transfer a host-resident render frame into guest pages when synchronization
/// or a guest-memory reader makes the bytes observable.
pub mod render_pass;
pub mod render_writeback;
/// A rail's own name for a resident render target, opaque to the layers that
/// carry it. See the module doc for the ledger this exists for.
pub mod resident_target;
/// The guest's per-resource validity quad, from both of its producers.
pub mod resource_validity;
/// The split of [`chain_phase`]'s largest *undivided* column, `sampled_us`.
pub mod sampled_phase;
/// Guest surface → host BGRA8 for the QEMU console.
pub mod scanout;
/// SPIR-V set-0 binding relocation for metal2vulkan + internal Vulkan engine (Linux).
pub mod spirv_bind;
mod spirv_layout;
/// How wide a translated vertex shader's stage-in reads are, per `Location`.
pub mod spirv_vertex_input;
/// Host surface cache (Linux/Vulkan discrete-GPU present, kb §8.5).
pub mod surface_cache;
/// Whether a host-side copy of a mapper-ref-texture surface's pixels is still
/// that surface's content, per the hypervisor's witness.
pub mod surface_currency;
/// The wire task word a command payload carries → a live task slot.
pub mod task_slot;
/// Texture / mapper-ref-texture geometry registration.
pub mod texture;
/// Track host-authoritative surface and GVA frames, and transfer them on demand.
pub mod writeback_debt;

/// The unit-test host double, gated with its definition. An ungated re-export
/// would keep it reachable and so keep it in the staticlib.
#[cfg(test)]
pub(crate) use host::FakeHost;
pub(crate) use host::{HostAction, HostOps};
