//! Turning a declared barrier into stage and access masks, and refusing the
//! stages this host has no equivalent for.
//!
//! # Why the translation is here and the barrier is not
//!
//! Every barrier opcode on the guest's wire is a proven no-op *for a particular
//! executor's submission granularity* — one host submission per pass boundary,
//! one per dispatch. [`reims_vgpu_core::sync`] therefore keeps the barrier as an
//! operation with its declared scope and stages intact, because the replacement
//! exists in order to batch differently and a merged batch is precisely where
//! the no-op stops being one.
//!
//! Which masks that becomes, and whether this host can express the stages the
//! guest named, is a host question. So it is here, and nothing above this crate
//! decides it.
//!
//! # The plan is `synchronization2`, and the legacy form is a mapping of it
//!
//! Vulkan 1.2 is the baseline and `VK_KHR_synchronization2` is an extension
//! there, so a host may need the older `vkCmdPipelineBarrier` masks. The rule
//! the plan sets is that a legacy emitter may exist only if it consumes the
//! *same* plan — so [`BarrierPlan`] is the one answer and
//! [`BarrierPlan::legacy`] maps it, rather than translating the guest's request
//! a second time. Two translations would be two readings of one request, and the
//! host on the older path would be the one nobody tested.
//!
//! The mapping is not a truncation. `synchronization2` split the old
//! `SHADER_READ` into sampled and storage reads and gave them bits above the
//! 32-bit space, so the older form is genuinely coarser and the map says by how
//! much in one table. [`BarrierPlan::unmapped_bits`] reports what the table does
//! not cover.
//!
//! The table's obligation is to every value that reaches [`BarrierPlan::legacy`],
//! not to the values [`translate`] happens to produce. A plan is four flag words
//! and anything in this crate may build one — [`crate::layout`] does, for the
//! image transitions it plans, and its vocabulary is not this module's: it names
//! whole-stage sets like `ALL_TRANSFER` and `ALL_GRAPHICS` and the transfer
//! accesses, because a layout transition is not a guest barrier. Scoping the
//! table to "what `translate` emits" would leave a legacy host recording a
//! staging upload with an empty access mask — an ordering that is not weaker
//! than the guest asked for, it is absent, and the upload races the sample that
//! reads it.
//!
//! So the map is total: a bit with no row widens to `ALL_COMMANDS` or to
//! `MEMORY_READ | MEMORY_WRITE` rather than vanishing. That is the same choice
//! [`BarrierTarget::Resources`] makes below — too strong costs throughput and
//! too weak is a race — and it makes "the older path silently ordered less"
//! unrepresentable rather than checked for. The rows still exist, and are still
//! the answer: widening is the floor under a missing one, not a substitute for
//! it.
//!
//! # A stage this host cannot express is refused by name
//!
//! Metal's tile stage has no core Vulkan equivalent, and its object and mesh
//! stages need `VK_EXT_mesh_shader`. Folding either into the fragment stage
//! would execute a barrier the guest did not ask for and silently drop the
//! ordering it did — so they are [`Decline`]s, and the mesh case is gated on the
//! capability being present rather than on a device or driver name.
//!
//! Undeclared bits are refused for the same reason
//! [`reims_vgpu_core::sync`]'s vocabulary keeps them representable: masking a
//! bit down to its declared neighbours reports a guest request the device is
//! sure it understood.
//!
//! # A stage mask and an access mask are one answer, not two
//!
//! Vulkan performs each access at particular stages, and a barrier whose
//! access mask names an access its stage mask cannot perform is invalid use
//! (VUID-VkMemoryBarrier2-srcAccessMask-03900 and its `dst` twin). The
//! consequence that matters is not the validation message: a colour write
//! happens at `COLOR_ATTACHMENT_OUTPUT`, so a barrier sourced at
//! `FRAGMENT_SHADER` does not wait for it, and `textureBarrier` — which is
//! *defined* as the colour write becoming readable — would order nothing it
//! was asked to.
//!
//! The guest cannot name those stages, because Metal does not have them:
//! `MTLRenderStageFragment` covers fragment shading together with the depth
//! and stencil tests and the colour writes that follow it. So the fragment
//! stage translates to four Vulkan stages, and the vertex stage to one.
//!
//! The access is then intersected with what those stages can perform. That is
//! not the masking this module refuses elsewhere: an access a stage cannot
//! perform did not happen in that stage, so the intersection removes nothing
//! the guest asked for. A guest ordering render-target memory against the
//! vertex stage gets an execution dependency and no memory one, which is the
//! whole of what it asked for.
//!
//! `ACCESS_STAGES` is the table, and it has to cover every access this
//! module emits for the same reason [`BarrierPlan::unmapped_bits`] exists: a
//! row that is missing does not fail, it silently drops the access.
//!
//! # What this does not do
//!
//! A barrier over a listed set of resources becomes one image or buffer memory
//! barrier per resource, and this cannot name a resource — the list is a window
//! into a transaction's own arena. So the masks come back and the caller applies
//! them to the resources it resolved. The masks are the part that is a host
//! policy decision; which resources they cover is not.

use ash::vk;
use reims_vgpu_core::sync::RenderStages;
use reims_vgpu_core::sync::{BarrierOp, BarrierScope, BarrierTarget};

/// What this host can express, as capabilities rather than names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageSupport {
    /// `VK_EXT_mesh_shader` is present and enabled, so the object and mesh
    /// stages have equivalents.
    pub mesh_shader: bool,
}

/// Why a barrier could not be translated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// The scope named bits the API does not declare.
    UndeclaredScopeBits { bits: u32 },
    /// The stage mask named bits the API does not declare.
    UndeclaredStageBits { bits: u32 },
    /// The guest named the tile stage. Metal's tile shading has no core Vulkan
    /// equivalent, and the imageblock memory it orders is not a thing this rail
    /// has to order.
    TileStage,
    /// The guest named the object or mesh stage and this host has no mesh
    /// shading. A capability answer, not a device one.
    MeshStagesUnavailable { stages: u32 },
}

impl Decline {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::UndeclaredScopeBits { .. } => "vk_barrier_undeclared_scope_bits",
            Self::UndeclaredStageBits { .. } => "vk_barrier_undeclared_stage_bits",
            Self::TileStage => "vk_barrier_tile_stage",
            Self::MeshStagesUnavailable { .. } => "vk_barrier_mesh_stages_unavailable",
        }
    }
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::UndeclaredScopeBits { bits } | Self::UndeclaredStageBits { bits } => {
                write!(f, "{} bits={bits:#x}", self.slug())
            }
            Self::TileStage => write!(f, "{}", self.slug()),
            Self::MeshStagesUnavailable { stages } => {
                write!(f, "{} stages={stages:#x}", self.slug())
            }
        }
    }
}

/// The one answer: what has to finish, and what may then proceed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BarrierPlan {
    pub src_stages: vk::PipelineStageFlags2,
    pub dst_stages: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
}

/// The same plan in the pre-`synchronization2` flag types.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LegacyBarrier {
    pub src_stages: vk::PipelineStageFlags,
    pub dst_stages: vk::PipelineStageFlags,
    pub src_access: vk::AccessFlags,
    pub dst_access: vk::AccessFlags,
}

impl BarrierPlan {
    /// Whether this plan orders anything at all.
    ///
    /// A guest may legally send a barrier over an empty scope. That orders
    /// nothing, and saying so here keeps "the guest asked for nothing" apart
    /// from "the device decided it owed nothing" — the two answers a barrier
    /// census has to distinguish.
    #[must_use]
    pub fn orders_anything(self) -> bool {
        !self.src_stages.is_empty() || !self.dst_stages.is_empty()
    }

    /// Map onto the pre-`synchronization2` flags.
    ///
    /// Coarser by construction: the older vocabulary has one `SHADER_READ`
    /// where the newer has a sampled read and a storage read. The map is the one
    /// place that coarsening happens, and [`BarrierPlan::unmapped_bits`] is what
    /// says it lost nothing rather than a comment claiming it.
    #[must_use]
    pub fn legacy(self) -> LegacyBarrier {
        LegacyBarrier {
            src_stages: map_stages(self.src_stages),
            dst_stages: map_stages(self.dst_stages),
            src_access: map_access(self.src_access),
            dst_access: map_access(self.dst_access),
        }
    }

    /// The bits with no entry in the legacy map, or empty.
    ///
    /// Non-empty means the map has no row for a value that reached a plan, so
    /// [`Self::legacy`] answered it by widening to `ALL_COMMANDS` or to
    /// `MEMORY_READ | MEMORY_WRITE` — correct, and coarser than the row would
    /// have been. This is what says a row is missing; it is not what makes the
    /// older path safe, because widening already did that.
    #[must_use]
    pub fn unmapped_bits(self) -> (u64, u64) {
        let stages = (self.src_stages | self.dst_stages).as_raw() & !mapped_stage_bits();
        let access = (self.src_access | self.dst_access).as_raw() & !mapped_access_bits();
        (stages, access)
    }
}

/// Every `synchronization2` stage that can reach a plan, and its older
/// equivalent.
///
/// Wider than [`translate`]'s own output, because [`crate::layout`] builds
/// plans too and a layout transition names stage *sets* — a transfer is not a
/// stage this module would emit for a guest barrier, and it is exactly what the
/// image upload path orders against.
const STAGE_MAP: &[(vk::PipelineStageFlags2, vk::PipelineStageFlags)] = &[
    (
        vk::PipelineStageFlags2::ALL_TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
    ),
    (
        vk::PipelineStageFlags2::ALL_GRAPHICS,
        vk::PipelineStageFlags::ALL_GRAPHICS,
    ),
    (
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    ),
    (
        vk::PipelineStageFlags2::VERTEX_SHADER,
        vk::PipelineStageFlags::VERTEX_SHADER,
    ),
    (
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    ),
    (
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
    ),
    (
        vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
    ),
    (
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
    ),
    (
        vk::PipelineStageFlags2::TASK_SHADER_EXT,
        vk::PipelineStageFlags::TASK_SHADER_EXT,
    ),
    (
        vk::PipelineStageFlags2::MESH_SHADER_EXT,
        vk::PipelineStageFlags::MESH_SHADER_EXT,
    ),
];

/// Every `synchronization2` access that can reach a plan, and its older
/// equivalent.
///
/// Two rows collapse onto `SHADER_READ`: that is the coarsening the older
/// vocabulary forces, and it is stated here rather than discovered. The transfer
/// rows are here for the reason [`STAGE_MAP`]'s are.
const ACCESS_MAP: &[(vk::AccessFlags2, vk::AccessFlags)] = &[
    (
        vk::AccessFlags2::TRANSFER_READ,
        vk::AccessFlags::TRANSFER_READ,
    ),
    (
        vk::AccessFlags2::TRANSFER_WRITE,
        vk::AccessFlags::TRANSFER_WRITE,
    ),
    (
        vk::AccessFlags2::UNIFORM_READ,
        vk::AccessFlags::UNIFORM_READ,
    ),
    (
        vk::AccessFlags2::SHADER_SAMPLED_READ,
        vk::AccessFlags::SHADER_READ,
    ),
    (
        vk::AccessFlags2::SHADER_STORAGE_READ,
        vk::AccessFlags::SHADER_READ,
    ),
    (
        vk::AccessFlags2::SHADER_STORAGE_WRITE,
        vk::AccessFlags::SHADER_WRITE,
    ),
    (
        vk::AccessFlags2::INPUT_ATTACHMENT_READ,
        vk::AccessFlags::INPUT_ATTACHMENT_READ,
    ),
    (
        vk::AccessFlags2::COLOR_ATTACHMENT_READ,
        vk::AccessFlags::COLOR_ATTACHMENT_READ,
    ),
    (
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
    ),
    (
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
    ),
    (
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
    ),
    (vk::AccessFlags2::MEMORY_READ, vk::AccessFlags::MEMORY_READ),
    (
        vk::AccessFlags2::MEMORY_WRITE,
        vk::AccessFlags::MEMORY_WRITE,
    ),
];

/// Every shader stage this module can emit. The carrier of every access a
/// shader performs, which is most of the table below.
const SHADER_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
        | vk::PipelineStageFlags2::TASK_SHADER_EXT.as_raw()
        | vk::PipelineStageFlags2::MESH_SHADER_EXT.as_raw(),
);

/// Every stage this module can emit, which is what carries `MEMORY_READ` and
/// `MEMORY_WRITE` — the two accesses every stage performs.
const ALL_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    SHADER_STAGES.as_raw()
        | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw(),
);

/// Which stages perform each access this module emits — Vulkan's "supported
/// access types" relation, restricted to the accesses that can reach a plan.
///
/// This is the table that makes a stage mask and an access mask one answer.
/// Every access below must also be produced by [`scope_access`] or by one of
/// the two literal targets in [`translate`], and every access those produce
/// must appear here: a row that is missing silently drops the access when the
/// intersection is taken, which is the failure mode this table exists to make
/// impossible. [`access_without_carrier`] is the check.
const ACCESS_STAGES: &[(vk::AccessFlags2, vk::PipelineStageFlags2)] = &[
    (vk::AccessFlags2::UNIFORM_READ, SHADER_STAGES),
    (vk::AccessFlags2::SHADER_SAMPLED_READ, SHADER_STAGES),
    (vk::AccessFlags2::SHADER_STORAGE_READ, SHADER_STAGES),
    (vk::AccessFlags2::SHADER_STORAGE_WRITE, SHADER_STAGES),
    (
        vk::AccessFlags2::INPUT_ATTACHMENT_READ,
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
    ),
    (
        vk::AccessFlags2::COLOR_ATTACHMENT_READ,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    ),
    (
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    ),
    (
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
        vk::PipelineStageFlags2::from_raw(
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
        ),
    ),
    (
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        vk::PipelineStageFlags2::from_raw(
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
        ),
    ),
    (vk::AccessFlags2::MEMORY_READ, ALL_STAGES),
    (vk::AccessFlags2::MEMORY_WRITE, ALL_STAGES),
];

/// The part of `access` that `stages` can actually perform.
///
/// Removes nothing the guest asked for: an access a stage cannot perform did
/// not happen in that stage. See the module doc.
fn access_within(access: vk::AccessFlags2, stages: vk::PipelineStageFlags2) -> vk::AccessFlags2 {
    ACCESS_STAGES
        .iter()
        .filter(|(bit, carriers)| access.contains(*bit) && stages.intersects(*carriers))
        .fold(vk::AccessFlags2::empty(), |acc, (bit, _)| acc | *bit)
}

/// The bits of `access` with no row in `ACCESS_STAGES`, or zero.
///
/// Non-zero would mean an access reaches a plan that `access_within` then
/// discards whatever the stages are — dropped ordering, silently.
#[must_use]
pub fn access_without_carrier(access: vk::AccessFlags2) -> u64 {
    let covered = ACCESS_STAGES
        .iter()
        .fold(0, |acc, (bit, _)| acc | bit.as_raw());
    access.as_raw() & !covered
}

fn mapped_stage_bits() -> u64 {
    STAGE_MAP.iter().fold(0, |acc, (new, _)| acc | new.as_raw())
}

fn mapped_access_bits() -> u64 {
    ACCESS_MAP
        .iter()
        .fold(0, |acc, (new, _)| acc | new.as_raw())
}

/// Total: a stage bit with no row widens to `ALL_COMMANDS`.
///
/// The alternative is that the bit contributes nothing, which is not a coarser
/// answer but a missing one — see the module doc.
fn map_stages(stages: vk::PipelineStageFlags2) -> vk::PipelineStageFlags {
    let mapped = STAGE_MAP
        .iter()
        .filter(|(new, _)| stages.contains(*new))
        .fold(vk::PipelineStageFlags::empty(), |acc, (_, old)| acc | *old);
    if stages.as_raw() & !mapped_stage_bits() == 0 {
        mapped
    } else {
        mapped | vk::PipelineStageFlags::ALL_COMMANDS
    }
}

/// Total: an access bit with no row widens to `MEMORY_READ | MEMORY_WRITE`.
///
/// As [`map_stages`].
fn map_access(access: vk::AccessFlags2) -> vk::AccessFlags {
    let mapped = ACCESS_MAP
        .iter()
        .filter(|(new, _)| access.contains(*new))
        .fold(vk::AccessFlags::empty(), |acc, (_, old)| acc | *old);
    if access.as_raw() & !mapped_access_bits() == 0 {
        mapped
    } else {
        mapped | vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
    }
}

/// Translate one declared barrier.
///
/// # Errors
///
/// If the record named an undeclared bit, or a stage this host cannot express.
pub fn translate(op: &BarrierOp, support: StageSupport) -> Result<BarrierPlan, Decline> {
    let src_stages = stages(op.after_stages, support)?;
    let dst_stages = stages(op.before_stages, support)?;
    let (src_access, dst_access) = match op.target {
        // The record named a scope of memory. Both halves get the scope's
        // access, because the barrier is an ordering of that memory against
        // itself: the guest wrote it through one stage set and reads it through
        // the other.
        BarrierTarget::Scope(scope) => {
            let bits = scope.undeclared_bits();
            if bits != 0 {
                return Err(Decline::UndeclaredScopeBits { bits });
            }
            let access = scope_access(scope);
            (access, access)
        }
        // The record named resources and no usage, so the direction is not
        // established. Conservative both ways: an ordering that is too strong
        // costs throughput, and one that is too weak is a race.
        BarrierTarget::Resources(_) => (
            vk::AccessFlags2::MEMORY_WRITE,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        ),
        // `textureBarrier` makes previously written fragments readable by later
        // ones in the same pass. That is exactly colour-attachment write
        // becoming shader read; the attachments are the pass's and not the
        // record's, which is why the caller supplies them.
        BarrierTarget::Texture => (
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::INPUT_ATTACHMENT_READ,
        ),
    };
    // A barrier over nothing orders nothing, and the plan says so by being
    // empty rather than by carrying stages with no access.
    if !op.orders_anything() {
        return Ok(BarrierPlan::default());
    }
    // One answer, not two: each half keeps only the access its own stages
    // perform. See the module doc for why this loses nothing.
    Ok(BarrierPlan {
        src_stages,
        dst_stages,
        src_access: access_within(src_access, src_stages),
        dst_access: access_within(dst_access, dst_stages),
    })
}

/// The access mask one declared scope names.
fn scope_access(scope: BarrierScope) -> vk::AccessFlags2 {
    let mut access = vk::AccessFlags2::empty();
    if scope.0 & BarrierScope::BUFFERS != 0 {
        access |= vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::SHADER_STORAGE_WRITE
            | vk::AccessFlags2::UNIFORM_READ;
    }
    if scope.0 & BarrierScope::TEXTURES != 0 {
        access |= vk::AccessFlags2::SHADER_SAMPLED_READ
            | vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::SHADER_STORAGE_WRITE;
    }
    if scope.0 & BarrierScope::RENDER_TARGETS != 0 {
        access |= vk::AccessFlags2::COLOR_ATTACHMENT_READ
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    access
}

/// The pipeline stages one declared stage mask names.
///
/// `None` is the compute rail: its barrier selectors carry no stage argument at
/// all, so the stage is the one its records execute in rather than an empty
/// mask. An empty mask would be a barrier that orders nothing, which is not what
/// a compute barrier is.
fn stages(
    declared: Option<RenderStages>,
    support: StageSupport,
) -> Result<vk::PipelineStageFlags2, Decline> {
    let Some(declared) = declared else {
        return Ok(vk::PipelineStageFlags2::COMPUTE_SHADER);
    };
    let bits = declared.undeclared_bits();
    if bits != 0 {
        return Err(Decline::UndeclaredStageBits { bits });
    }
    if declared.0 & RenderStages::TILE != 0 {
        return Err(Decline::TileStage);
    }
    let mesh = declared.0 & (RenderStages::OBJECT | RenderStages::MESH);
    if mesh != 0 && !support.mesh_shader {
        return Err(Decline::MeshStagesUnavailable { stages: mesh });
    }
    let mut out = vk::PipelineStageFlags2::empty();
    if declared.0 & RenderStages::VERTEX != 0 {
        out |= vk::PipelineStageFlags2::VERTEX_SHADER;
    }
    if declared.0 & RenderStages::FRAGMENT != 0 {
        // Four, because Metal has one. `MTLRenderStageFragment` covers
        // fragment shading and the per-fragment fixed function that follows
        // it, and the guest has no other spelling for the stages that perform
        // the depth, stencil and colour accesses. Emitting only the shader
        // stage would leave every attachment access in the plan uncarried, and
        // `textureBarrier` unable to name its own source.
        out |= vk::PipelineStageFlags2::FRAGMENT_SHADER
            | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
    }
    if declared.0 & RenderStages::OBJECT != 0 {
        out |= vk::PipelineStageFlags2::TASK_SHADER_EXT;
    }
    if declared.0 & RenderStages::MESH != 0 {
        out |= vk::PipelineStageFlags2::MESH_SHADER_EXT;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Use;
    use reims_vgpu_core::sync::ResourceSpan;

    fn scope(bits: u32, after: Option<u32>, before: Option<u32>) -> BarrierOp {
        BarrierOp {
            target: BarrierTarget::Scope(BarrierScope(bits)),
            after_stages: after.map(RenderStages),
            before_stages: before.map(RenderStages),
        }
    }

    const RENDER: StageSupport = StageSupport { mesh_shader: false };
    const MESH: StageSupport = StageSupport { mesh_shader: true };

    /// A compute barrier's selectors carry no stages, and the compute stage is
    /// what its records execute in — not an empty mask, which would order
    /// nothing.
    #[test]
    fn a_stageless_barrier_is_the_compute_stage_and_not_an_empty_mask() {
        let plan = translate(
            &scope(BarrierScope::BUFFERS | BarrierScope::TEXTURES, None, None),
            RENDER,
        )
        .expect("declared bits only");
        assert_eq!(plan.src_stages, vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert_eq!(plan.dst_stages, vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert!(plan.orders_anything());
        assert!(plan
            .src_access
            .contains(vk::AccessFlags2::SHADER_STORAGE_WRITE));
        assert!(plan
            .dst_access
            .contains(vk::AccessFlags2::SHADER_SAMPLED_READ));
    }

    /// `MTLRenderStageFragment` is four Vulkan stages, and it has to be: the
    /// colour write it names happens at `COLOR_ATTACHMENT_OUTPUT`, so a plan
    /// sourced only at `FRAGMENT_SHADER` would not wait for it.
    const FRAGMENT_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
        vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
            | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw(),
    );

    #[test]
    fn a_render_barrier_carries_the_stages_the_record_declared() {
        let plan = translate(
            &scope(
                BarrierScope::RENDER_TARGETS,
                Some(RenderStages::FRAGMENT),
                Some(RenderStages::VERTEX | RenderStages::FRAGMENT),
            ),
            RENDER,
        )
        .expect("declared bits only");
        assert_eq!(plan.src_stages, FRAGMENT_STAGES);
        assert_eq!(
            plan.dst_stages,
            vk::PipelineStageFlags2::VERTEX_SHADER | FRAGMENT_STAGES
        );
        assert!(plan
            .src_access
            .contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE));
    }

    /// The rule the whole table serves: every access in a half is one the
    /// stages of that same half perform. Asserted over every combination of
    /// scope and declared stages, not over the handful of cases below.
    #[test]
    fn no_half_of_any_plan_names_an_access_its_own_stages_cannot_perform() {
        let declared = [
            RenderStages::VERTEX,
            RenderStages::FRAGMENT,
            RenderStages::OBJECT,
            RenderStages::MESH,
        ];
        let mut seen_attachment = false;
        let mut seen_shader = false;
        for scope_bits in
            0..=(BarrierScope::BUFFERS | BarrierScope::TEXTURES | BarrierScope::RENDER_TARGETS)
        {
            for after in 0..(1u32 << declared.len()) {
                for before in 0..(1u32 << declared.len()) {
                    let mask = |set: u32| {
                        declared
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| set & (1 << i) != 0)
                            .fold(0, |acc, (_, bit)| acc | bit)
                    };
                    let op = scope(scope_bits, Some(mask(after)), Some(mask(before)));
                    let Ok(plan) = translate(&op, MESH) else {
                        continue;
                    };
                    for (access, stages) in [
                        (plan.src_access, plan.src_stages),
                        (plan.dst_access, plan.dst_stages),
                    ] {
                        assert_eq!(
                            access_within(access, stages),
                            access,
                            "{op:?} names {access:?} at {stages:?}"
                        );
                        seen_attachment |=
                            access.contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
                        seen_shader |= access.contains(vk::AccessFlags2::SHADER_SAMPLED_READ);
                    }
                }
            }
        }
        // Not vacuous: the sweep reached both an attachment access and a
        // shader one rather than only empty plans.
        assert!(seen_attachment && seen_shader);
    }

    /// A guest may order render-target memory against the vertex stage. No
    /// attachment access happens there, so what it asked for is an execution
    /// dependency and no memory one — the plan says exactly that instead of
    /// naming an access the stage cannot perform.
    #[test]
    fn a_scope_no_stage_in_the_mask_touches_becomes_an_execution_dependency() {
        let plan = translate(
            &scope(
                BarrierScope::RENDER_TARGETS,
                Some(RenderStages::VERTEX),
                Some(RenderStages::VERTEX),
            ),
            RENDER,
        )
        .expect("declared bits only");
        assert_eq!(plan.src_stages, vk::PipelineStageFlags2::VERTEX_SHADER);
        assert_eq!(plan.src_access, vk::AccessFlags2::empty());
        assert_eq!(plan.dst_access, vk::AccessFlags2::empty());
        assert!(
            plan.orders_anything(),
            "an execution dependency is still an ordering"
        );
    }

    /// A row missing from the table is not a compile error and not a refusal:
    /// the intersection would drop the access whatever the stages were. So
    /// every access this module can produce is checked against the table.
    #[test]
    fn every_access_this_module_emits_has_a_carrier() {
        let mut every = vk::AccessFlags2::empty();
        for bits in
            0..=(BarrierScope::BUFFERS | BarrierScope::TEXTURES | BarrierScope::RENDER_TARGETS)
        {
            every |= scope_access(BarrierScope(bits));
        }
        // The two targets that carry a literal access rather than a scope.
        every |= vk::AccessFlags2::MEMORY_READ
            | vk::AccessFlags2::MEMORY_WRITE
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::SHADER_SAMPLED_READ
            | vk::AccessFlags2::INPUT_ATTACHMENT_READ;
        assert_eq!(access_without_carrier(every), 0);
        // And the check is not vacuous: an access nothing here emits has no
        // carrier and would be reported.
        assert_ne!(access_without_carrier(vk::AccessFlags2::HOST_WRITE), 0);
    }

    /// A barrier over nothing is legal to send and orders nothing. It must not
    /// come back as stages with no access, which a caller would emit.
    #[test]
    fn an_empty_scope_orders_nothing() {
        let plan = translate(&scope(0, Some(RenderStages::FRAGMENT), None), RENDER)
            .expect("zero is a declared value");
        assert_eq!(plan, BarrierPlan::default());
        assert!(!plan.orders_anything());
    }

    #[test]
    fn an_empty_resource_list_orders_nothing() {
        let op = BarrierOp {
            target: BarrierTarget::Resources(ResourceSpan { start: 0, len: 0 }),
            after_stages: Some(RenderStages(RenderStages::FRAGMENT)),
            before_stages: Some(RenderStages(RenderStages::VERTEX)),
        };
        assert_eq!(translate(&op, RENDER), Ok(BarrierPlan::default()));
    }

    /// The record names resources and no usage, so the direction is not
    /// established. Too strong costs throughput; too weak is a race.
    #[test]
    fn a_resource_list_with_no_usage_is_ordered_conservatively() {
        let op = BarrierOp {
            target: BarrierTarget::Resources(ResourceSpan { start: 4, len: 3 }),
            after_stages: None,
            before_stages: None,
        };
        let plan = translate(&op, RENDER).expect("no stage bits to be wrong");
        assert_eq!(plan.src_access, vk::AccessFlags2::MEMORY_WRITE);
        assert_eq!(
            plan.dst_access,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
        );
    }

    /// Masking a bit down to its declared neighbours reports a guest request
    /// the device is sure it understood.
    #[test]
    fn an_undeclared_bit_is_refused_and_not_masked() {
        assert_eq!(
            translate(&scope(BarrierScope::BUFFERS | 0x80, None, None), RENDER),
            Err(Decline::UndeclaredScopeBits { bits: 0x80 })
        );
        assert_eq!(
            translate(&scope(BarrierScope::BUFFERS, Some(0x40), None), RENDER),
            Err(Decline::UndeclaredStageBits { bits: 0x40 })
        );
    }

    /// Folding the tile stage into the fragment stage would execute a barrier
    /// the guest did not ask for and drop the ordering it did.
    #[test]
    fn the_tile_stage_is_refused_by_name() {
        assert_eq!(
            translate(
                &scope(
                    BarrierScope::TEXTURES,
                    Some(RenderStages::FRAGMENT | RenderStages::TILE),
                    Some(RenderStages::FRAGMENT)
                ),
                MESH
            ),
            Err(Decline::TileStage),
            "and mesh support does not make a tile stage expressible"
        );
    }

    /// Gated on the capability, not on a device.
    #[test]
    fn the_mesh_stages_follow_the_capability() {
        let op = scope(
            BarrierScope::TEXTURES,
            Some(RenderStages::OBJECT | RenderStages::MESH),
            Some(RenderStages::FRAGMENT),
        );
        assert_eq!(
            translate(&op, RENDER),
            Err(Decline::MeshStagesUnavailable {
                stages: RenderStages::OBJECT | RenderStages::MESH
            })
        );
        let plan = translate(&op, MESH).expect("the capability is present");
        assert_eq!(
            plan.src_stages,
            vk::PipelineStageFlags2::TASK_SHADER_EXT | vk::PipelineStageFlags2::MESH_SHADER_EXT
        );
    }

    /// Every barrier this module can translate, over the whole closed input
    /// space, and the two claims the older path rests on.
    ///
    /// The input space really is closed and small: three declared scope bits,
    /// five declared stage bits on each of two masks plus the "carried none"
    /// case, three target shapes, and two hosts. So "everything this module
    /// emits" can be swept rather than sampled --- and the sampled version
    /// below it was six operations, which is not a covering of 3140.
    ///
    /// Two laws, and the second is not implied by the first:
    ///
    /// - no bit reaches a plan without an entry in the legacy map, which is
    ///   what [`BarrierPlan::unmapped_bits`] says;
    /// - a plan that orders memory still orders memory after the map. A row
    ///   whose older equivalent were empty would be *mapped* and still drop
    ///   the access, and a legacy host would be handed a barrier that orders
    ///   nothing while the `synchronization2` host ordered a write against a
    ///   read.
    #[test]
    fn every_barrier_this_module_can_translate_survives_the_older_vocabulary() {
        const DECLARED_STAGES: u32 = RenderStages::VERTEX
            | RenderStages::FRAGMENT
            | RenderStages::TILE
            | RenderStages::OBJECT
            | RenderStages::MESH;

        let mut translated = 0_u32;
        for support in [RENDER, MESH] {
            for after in (0..=DECLARED_STAGES).map(Some).chain([None]) {
                for before in (0..=DECLARED_STAGES).map(Some).chain([None]) {
                    let targets = [
                        BarrierTarget::Texture,
                        BarrierTarget::Resources(ResourceSpan { start: 0, len: 1 }),
                    ]
                    .into_iter()
                    .chain(
                        (0..=(BarrierScope::BUFFERS
                            | BarrierScope::TEXTURES
                            | BarrierScope::RENDER_TARGETS))
                            .map(|bits| BarrierTarget::Scope(BarrierScope(bits))),
                    );
                    for target in targets {
                        let op = BarrierOp {
                            target,
                            after_stages: after.map(RenderStages),
                            before_stages: before.map(RenderStages),
                        };
                        // A decline is this module refusing a stage it cannot
                        // express, which the tests above cover; there is no
                        // plan to carry to the older path.
                        let Ok(plan) = translate(&op, support) else {
                            continue;
                        };
                        translated += 1;
                        assert_eq!(
                            plan.unmapped_bits(),
                            (0, 0),
                            "a bit with no older equivalent reached the plan for {op:?}"
                        );
                        let legacy = plan.legacy();
                        assert_eq!(
                            plan.src_access.is_empty(),
                            legacy.src_access.is_empty(),
                            "the map emptied a source access for {op:?}"
                        );
                        assert_eq!(
                            plan.dst_access.is_empty(),
                            legacy.dst_access.is_empty(),
                            "the map emptied a destination access for {op:?}"
                        );
                        assert_eq!(
                            plan.src_stages.is_empty(),
                            legacy.src_stages.is_empty(),
                            "the map emptied a source stage for {op:?}"
                        );
                        assert_eq!(
                            plan.dst_stages.is_empty(),
                            legacy.dst_stages.is_empty(),
                            "the map emptied a destination stage for {op:?}"
                        );
                    }
                }
            }
        }
        // Stated so a change that empties an arm --- a decline that starts
        // firing for every input, say --- is a failure and not a sweep that
        // silently checks nothing.
        assert_eq!(translated, 3140, "the swept space changed shape");
    }

    /// The legacy path consumes the same plan. A map that dropped a bit would
    /// ask the host on the older path for less ordering than the guest wrote.
    #[test]
    fn the_legacy_map_covers_everything_this_module_emits() {
        let cases = [
            scope(BarrierScope::BUFFERS, None, None),
            scope(BarrierScope::TEXTURES, None, None),
            scope(
                BarrierScope::RENDER_TARGETS,
                Some(RenderStages::FRAGMENT),
                Some(RenderStages::VERTEX),
            ),
            scope(
                BarrierScope::BUFFERS | BarrierScope::TEXTURES | BarrierScope::RENDER_TARGETS,
                Some(RenderStages::VERTEX | RenderStages::FRAGMENT),
                Some(RenderStages::FRAGMENT),
            ),
            BarrierOp {
                target: BarrierTarget::Texture,
                after_stages: Some(RenderStages(RenderStages::FRAGMENT)),
                before_stages: Some(RenderStages(RenderStages::FRAGMENT)),
            },
            BarrierOp {
                target: BarrierTarget::Resources(ResourceSpan { start: 0, len: 1 }),
                after_stages: None,
                before_stages: None,
            },
            scope(
                BarrierScope::TEXTURES,
                Some(RenderStages::OBJECT | RenderStages::MESH),
                Some(RenderStages::FRAGMENT),
            ),
        ];
        for op in cases {
            let plan = translate(&op, MESH).expect("declared bits only");
            assert_eq!(
                plan.unmapped_bits(),
                (0, 0),
                "a bit with no older equivalent reached the plan for {op:?}"
            );
        }
    }

    /// [`BarrierPlan::legacy`]'s production caller is [`crate::record`], which
    /// hands it a [`crate::layout::Transition`] --- and a transition's four
    /// words come from [`Use::stages`], [`Use::access`] and `NONE`. So the
    /// sweep that says the map covers has to be over *that* vocabulary as much
    /// as over [`translate`]'s, and it is the one that reaches a transfer.
    #[test]
    fn every_layout_transition_survives_the_older_vocabulary() {
        const USES: &[Use] = &[
            Use::ColorAttachment,
            Use::DepthStencilAttachment,
            Use::DepthStencilRead,
            Use::SampledRead,
            Use::Storage,
            Use::TransferSrc,
            Use::TransferDst,
            Use::Present,
        ];
        for from in USES {
            for to in USES {
                let plan = BarrierPlan {
                    src_stages: from.stages(),
                    dst_stages: to.stages(),
                    src_access: from.access(),
                    dst_access: to.access(),
                };
                assert_eq!(
                    plan.unmapped_bits(),
                    (0, 0),
                    "no row for the transition {from:?} -> {to:?}"
                );
                let legacy = plan.legacy();
                assert_eq!(
                    plan.src_access.is_empty(),
                    legacy.src_access.is_empty(),
                    "{from:?} -> {to:?} lost its source access on the older path"
                );
                assert_eq!(
                    plan.dst_access.is_empty(),
                    legacy.dst_access.is_empty(),
                    "{from:?} -> {to:?} lost its destination access on the older path"
                );
                assert_eq!(
                    plan.src_stages.is_empty(),
                    legacy.src_stages.is_empty(),
                    "{from:?} -> {to:?} lost its source stages on the older path"
                );
                assert_eq!(
                    plan.dst_stages.is_empty(),
                    legacy.dst_stages.is_empty(),
                    "{from:?} -> {to:?} lost its destination stages on the older path"
                );
            }
        }
    }

    /// The concrete one: a staging upload becoming a sampled read. With no
    /// rows for it, both halves came back empty --- and [`crate::record`] reads
    /// an empty stage mask as `TOP_OF_PIPE` to `BOTTOM_OF_PIPE`, so a legacy
    /// host recorded a layout change with no memory dependency at all and the
    /// sample read whatever was in the image.
    #[test]
    fn a_staging_upload_still_orders_the_sample_that_reads_it() {
        let legacy = BarrierPlan {
            src_stages: Use::TransferDst.stages(),
            dst_stages: Use::SampledRead.stages(),
            src_access: Use::TransferDst.access(),
            dst_access: Use::SampledRead.access(),
        }
        .legacy();
        assert_eq!(legacy.src_stages, vk::PipelineStageFlags::TRANSFER);
        assert_eq!(legacy.src_access, vk::AccessFlags::TRANSFER_WRITE);
        assert_eq!(
            legacy.dst_stages,
            vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::COMPUTE_SHADER
        );
        assert_eq!(legacy.dst_access, vk::AccessFlags::SHADER_READ);
    }

    /// A bit with no row is coarser on the older path, never absent --- and
    /// only its own half is coarsened.
    #[test]
    fn a_bit_with_no_row_widens_rather_than_vanishing() {
        let plan = BarrierPlan {
            src_stages: vk::PipelineStageFlags2::HOST,
            dst_stages: vk::PipelineStageFlags2::VERTEX_SHADER,
            src_access: vk::AccessFlags2::HOST_WRITE,
            dst_access: vk::AccessFlags2::UNIFORM_READ,
        };
        assert_ne!(
            plan.unmapped_bits(),
            (0, 0),
            "and the check still reports the missing rows"
        );
        let legacy = plan.legacy();
        assert!(legacy
            .src_stages
            .contains(vk::PipelineStageFlags::ALL_COMMANDS));
        assert!(legacy.src_access.contains(vk::AccessFlags::MEMORY_WRITE));
        assert!(legacy.src_access.contains(vk::AccessFlags::MEMORY_READ));
        assert_eq!(
            legacy.dst_stages,
            vk::PipelineStageFlags::VERTEX_SHADER,
            "the half whose bits all have rows is not widened"
        );
        assert_eq!(legacy.dst_access, vk::AccessFlags::UNIFORM_READ);
    }

    /// The coarsening the older vocabulary forces, stated rather than
    /// discovered: two distinct reads become one.
    #[test]
    fn the_two_shader_reads_become_one_in_the_older_vocabulary() {
        let plan = translate(&scope(BarrierScope::TEXTURES, None, None), RENDER)
            .expect("declared bits only");
        assert!(plan
            .src_access
            .contains(vk::AccessFlags2::SHADER_SAMPLED_READ));
        assert!(plan
            .src_access
            .contains(vk::AccessFlags2::SHADER_STORAGE_READ));
        let legacy = plan.legacy();
        assert!(legacy.src_access.contains(vk::AccessFlags::SHADER_READ));
        assert!(legacy.src_access.contains(vk::AccessFlags::SHADER_WRITE));
        assert_eq!(
            legacy.src_stages,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            "and the stage maps one-to-one"
        );
    }

    /// A mesh-stage barrier maps onto the extension's own older stage bits, so
    /// a host with mesh shading and no `synchronization2` is still served.
    #[test]
    fn the_mesh_stages_have_an_older_equivalent() {
        let plan = translate(
            &scope(
                BarrierScope::TEXTURES,
                Some(RenderStages::MESH),
                Some(RenderStages::FRAGMENT),
            ),
            MESH,
        )
        .expect("the capability is present");
        assert_eq!(plan.unmapped_bits(), (0, 0));
        assert_eq!(
            plan.legacy().src_stages,
            vk::PipelineStageFlags::MESH_SHADER_EXT
        );
    }

    #[test]
    fn a_texture_barrier_orders_the_passes_own_attachments() {
        let plan = translate(
            &BarrierOp {
                target: BarrierTarget::Texture,
                after_stages: Some(RenderStages(RenderStages::FRAGMENT)),
                before_stages: Some(RenderStages(RenderStages::FRAGMENT)),
            },
            RENDER,
        )
        .expect("declared bits only");
        assert_eq!(plan.src_access, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
        assert!(plan
            .dst_access
            .contains(vk::AccessFlags2::SHADER_SAMPLED_READ));
        assert!(
            plan.orders_anything(),
            "it names nothing and orders its pass"
        );
        // And the write it waits for is carried by the stage that performs
        // it. Sourced at `FRAGMENT_SHADER` alone, this barrier would let the
        // read run before the write it exists to order.
        assert!(plan
            .src_stages
            .contains(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT));
    }

    #[test]
    fn every_decline_has_its_own_slug() {
        let all = [
            Decline::UndeclaredScopeBits { bits: 1 },
            Decline::UndeclaredStageBits { bits: 1 },
            Decline::TileStage,
            Decline::MeshStagesUnavailable { stages: 1 },
        ];
        let mut slugs: Vec<&str> = all.iter().map(|d| d.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count);
        assert!(Decline::TileStage.to_string().contains("tile_stage"));
    }
}
