//! Compute segment sequencing: control-flow encode + ICB fail-closed.
//!
//! ## Control-flow (`0xdc`–`0xe2`)
//!
//! Host Metal exposes SPI (`encodeStartIf` / `While` / `DoWhile` family) on the
//! real AGX compute encoder (runtime-probed; not in public headers). Product
//! path opens a multi-record [`ComputeSession`] encoder and records those SPI
//! calls with condition buffers staged from guest GVA.
//!
//! Nested dispatches under an open session encode onto the **same** encoder via
//! [`crate::backend::metal::compute::compute_encode_on_encoder`] so they sit
//! inside the SPI region. The session commits once at segment end; GVA
//! writeback is deferred until then.
//!
//! ## ICB (`0xe4` / `0xe5`)
//!
//! Serializer-object tag `0x36` materializes a host `MTLIndirectCommandBuffer` (cached per
//! task/ref). Command fills use the host Metal fill API in
//! [`crate::runtime::icb`] — the stream carries no fill opcodes. Execute
//! applies **parent-encoder inheritance** from stream [`ComputeAccum`] (Metal:
//! buffers when `inheritBuffers`, pipeline when `inheritPipelineState`; textures/
//! samplers are never recordable into classic `MTLIndirectComputeCommand` and
//! always come from the encoder when present), then encodes
//! `executeCommandsInBuffer` SPI. Buffer writebacks from ICB fills and from
//! inherited encoder binds flush after session commit. Failures latch
//! [`SequencingBlock::IndirectCommandBuffer`].

// The backend the process executes on, reached only through the trait, and the
// session handle it hands back. The handle's shape is neutral; what is behind
// it belongs to `backend::compute_session` and is never named here.
use crate::backend::compute_session::ComputeSession;
use crate::backend::Backend as _;
use crate::model::DeviceState;
use crate::runtime::compute_exec::{ComputeAccum, ComputeStatus};
use crate::runtime::decode::compute_spi::{Command as ComputeCommand, Kind};
use crate::runtime::host::{HostMemory, HostOps};
use reims_vgpu_protocol::compute::DispatchType;

/// Latched reason that blocks later dispatches in the same compute segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequencingBlock {
    ControlFlow,
    IndirectCommandBuffer,
}

// The Metal rail's encoder. Named, not re-exported: this module owns the
// segment's sequencing and nothing here should be able to name an `MTLDevice`.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) mod metal;

/// The mutable state of one `SEGMENT_TYPE_COMPUTE` segment.
///
/// These three share a single lifetime: they come into existence when the
/// segment opens, every record in the segment reads and mutates them together,
/// and the session commits when the segment ends. Passing them as one value
/// keeps that lifetime visible at each call site.
#[derive(Default)]
pub struct ComputeSegment {
    /// Pipeline / bind state accumulated across the segment's records.
    pub acc: ComputeAccum,
    /// Multi-record encoder, opened on demand by the first control-flow or ICB
    /// record and committed at segment end.
    pub session: Option<ComputeSession>,
    /// Latched sequencing failure; once set it refuses later dispatches.
    pub block: Option<SequencingBlock>,
}

pub fn ensure_session(
    session: &mut Option<ComputeSession>,
    dispatch_type: DispatchType,
) -> Result<&mut ComputeSession, ComputeStatus> {
    if session.is_none() {
        *session = Some(crate::backend::selected().open_compute_session(dispatch_type)?);
    }
    Ok(session.as_mut().unwrap())
}

pub fn apply_sequencing<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &ComputeCommand,
    seg: &mut ComputeSegment,
) -> ComputeStatus {
    if seg.block.is_some() {
        return ComputeStatus::Unsupported("sequencing_block_active");
    }
    match cmd.kind {
        Kind::ControlStartDoWhile
        | Kind::ControlEndDoWhile
        | Kind::ControlStartWhile
        | Kind::ControlEndWhile
        | Kind::ControlStartIf
        | Kind::ControlStartElse
        | Kind::ControlEndIf => {
            let sess = match ensure_session(&mut seg.session, seg.acc.dispatch_type) {
                Ok(s) => s,
                Err(e) => {
                    seg.block = Some(SequencingBlock::ControlFlow);
                    return e;
                }
            };
            let st = sess.encode_control(state, host, task_id, cmd);
            if !matches!(st, ComputeStatus::Ok) {
                seg.block = Some(SequencingBlock::ControlFlow);
            }
            st
        }
        Kind::ExecuteCommandsInBuffer | Kind::ExecuteCommandsInBufferIndirect => {
            let sess = match ensure_session(&mut seg.session, seg.acc.dispatch_type) {
                Ok(s) => s,
                Err(e) => {
                    seg.block = Some(SequencingBlock::IndirectCommandBuffer);
                    return e;
                }
            };
            let st = sess.encode_icb(state, host, task_id, cmd, &seg.acc);
            // Latch only on failure so successful materialize+execute does not
            // block later dispatches in the segment.
            if !matches!(st, ComputeStatus::Ok) {
                seg.block = Some(SequencingBlock::IndirectCommandBuffer);
            }
            st
        }
        _ => ComputeStatus::Unsupported("sequencing_unknown_kind"),
    }
}

/// Finish an open session at compute-segment end (no-op if none).
pub fn finish_session<M: HostMemory + HostOps>(
    session: &mut Option<ComputeSession>,
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
) -> Option<ComputeStatus> {
    session.take().map(|s| s.finish(host, state, task_id))
}

#[cfg(test)]
mod tests {

    use super::*;
    // The Metal rail by name, not `backend::selected()`: these tests open a
    // Metal compute session and assert what it does with it, and a binary
    // carrying both rails may be *running* the other one.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    use crate::backend::metal::MetalBackend;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    use crate::protocol::endian::{st32, st64};
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, RESOURCE_PAGE_SHIFT,
    };
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    use crate::runtime::gva_mem;
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    use crate::runtime::gva_mem::write_task_gva_arm64e;

    use crate::runtime::host::FakeHost;

    /// A `dispatchThreadgroups:threadsPerThreadgroup:` record, as the lift
    /// makes it.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    fn threadgroups(
        groups: [u64; 3],
        threads_per_group: [u64; 3],
    ) -> reims_vgpu_protocol::decode::compute::DispatchRecord {
        use reims_vgpu_protocol::decode::compute::{DispatchRecord, Extent, Threadgroups};
        let extent = |[width, height, depth]: [u64; 3]| Extent {
            width,
            height,
            depth,
        };
        DispatchRecord::Threadgroups(Threadgroups {
            groups: extent(groups),
            threads_per_group: extent(threads_per_group),
        })
    }

    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    #[test]
    fn metal_reflection_status_survives_the_session_handoff() {
        use crate::observe::{Emit, Refusal as _};

        let status = crate::backend::metal::error::Status::execute(
            "metal_compute_reflection_pso_create_failed",
        );
        let carried = ComputeStatus::RailRefused(status);
        assert_eq!(
            carried.refusal(),
            Some("metal_compute_reflection_pso_create_failed")
        );
        assert_eq!(
            Emit::refusal("compute_session", &carried)
                .expect("the session must preserve the backend refusal")
                .render(),
            "compute_session reason=metal_compute_reflection_pso_create_failed \
             class=execute recovery=metal_failed"
        );
    }

    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    #[test]
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    fn control_if_else_spi_session_commits() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(1, 0, 32));

        // Condition buffer: u32 == 5 at offset 0.
        let cond = 5u32.to_le_bytes();
        let buf_gva = 5u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &cond);
        let mut bdesc = vec![0u8; 16];
        st64(&mut bdesc[0..], 4);
        st32(&mut bdesc[8..], 5);
        let bdesc_gva = 0x180u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], bdesc_gva, &bdesc);
        {
            let off = list_object_entry_offset(7, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }

        let mut session = MetalBackend
            .open_compute_session(DispatchType::Serial)
            .expect("metal session");
        let start = ComputeCommand {
            kind: Kind::ControlStartIf,
            condition_buffer_ref: 7,
            condition_buffer_offset: 0,
            condition_comparison: 2, // Equal
            condition_reference_value: 5,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &start),
            ComputeStatus::Ok
        );
        assert_eq!(session.control_depth(), 1);

        let els = ComputeCommand {
            kind: Kind::ControlStartElse,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &els),
            ComputeStatus::Ok
        );

        let end = ComputeCommand {
            kind: Kind::ControlEndIf,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &end),
            ComputeStatus::Ok
        );
        assert_eq!(session.control_depth(), 0);
        assert_eq!(session.finish(&mut host, &mut state, 1), ComputeStatus::Ok);
    }

    #[test]
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    fn nested_dispatch_under_if_writeback() {
        use crate::runtime::compute_exec::ComputeBufferBind;
        use crate::runtime::decode::resource::{
            OBJECT_TYPE_FUNCTION, OBJECT_TYPE_SERIALIZER_OBJECT, PIPELINE_TAG_KERNEL_FUNC,
            SERIALIZER_OBJECT_COMPUTE_PIPELINE, SERIALIZER_OBJECT_FIRST_TLVS,
        };
        use std::path::PathBuf;

        let mtlb_paths =
            [PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/compute_mul3add1.mtlb")];
        let mtlb = mtlb_paths
            .iter()
            .find_map(|p| std::fs::read(p).ok())
            .expect("compute_mul3add1.mtlb fixture");

        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(1, 0, 32));

        // Condition == 1 at buffer ref 8.
        let cond = 1u32.to_le_bytes();
        let cond_gva = 4u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], cond_gva, &cond);
        let mut cdesc = vec![0u8; 16];
        st64(&mut cdesc[0..], 4);
        st32(&mut cdesc[8..], 4);
        let cdesc_gva = 0x100u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], cdesc_gva, &cdesc);
        {
            let off = list_object_entry_offset(8, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&cdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }

        // Kernel function + pipeline + data buffer (same shape as mul3add1 unit).
        let blob_gva = 5u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], blob_gva, &mtlb);
        let mut fdesc = vec![0u8; 32];
        st64(&mut fdesc[0..], blob_gva);
        st32(&mut fdesc[8..], mtlb.len() as u32);
        let fdesc_gva = 0x140u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], fdesc_gva, &fdesc);
        {
            let off = list_object_entry_offset(5, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_FUNCTION as u32) | (32u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&fdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }
        let mut pdesc = vec![0u8; 32];
        st32(&mut pdesc[0..], SERIALIZER_OBJECT_COMPUTE_PIPELINE);
        st32(&mut pdesc[4..], 32);
        pdesc[SERIALIZER_OBJECT_FIRST_TLVS] = 1;
        pdesc[SERIALIZER_OBJECT_FIRST_TLVS + 1] = PIPELINE_TAG_KERNEL_FUNC;
        pdesc[SERIALIZER_OBJECT_FIRST_TLVS + 2] = 4;
        st32(&mut pdesc[SERIALIZER_OBJECT_FIRST_TLVS + 3..], 5);
        let pdesc_gva = 0x180u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], pdesc_gva, &pdesc);
        {
            let off = list_object_entry_offset(6, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_SERIALIZER_OBJECT as u32) | (32u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&pdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }
        let data = [1u32, 2, 3, 4];
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf_gva = 6u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);
        let mut bdesc = vec![0u8; 16];
        st64(&mut bdesc[0..], 16);
        st32(&mut bdesc[8..], 6);
        let bdesc_gva = 0x1c0u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], bdesc_gva, &bdesc);
        {
            let off = list_object_entry_offset(7, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }

        // Phase A: nested dispatch alone on a session (no control SPI).
        {
            let mut session = MetalBackend
                .open_compute_session(DispatchType::Serial)
                .expect("session");
            let mut acc = ComputeAccum::default();
            acc.set_pipeline(6);
            acc.buffers.push(ComputeBufferBind {
                index: 0,
                buffer_ref: 7,
                offset: 0,
                attribute_stride: 0,
                has_attribute_stride: false,
            });
            let dcmd = threadgroups([1, 1, 1], [4, 1, 1]);
            assert_eq!(
                MetalBackend.execute_dispatch_nested(
                    &mut state,
                    &mut host,
                    1,
                    &acc,
                    &dcmd,
                    &mut session
                ),
                ComputeStatus::Ok
            );
            assert_eq!(session.deferred_writeback_count(), 1);
            assert_eq!(session.finish(&mut host, &mut state, 1), ComputeStatus::Ok);
            let mut back = [0u8; 16];
            assert!(gva_mem::read_task_gva(
                &host,
                &state.tasks[1],
                buf_gva,
                &mut back,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
            let out: Vec<u32> = back
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(out, vec![4, 7, 10, 13], "session-only nested writeback");
        }

        // Reset data for phase B (if-wrapped).
        let data = [1u32, 2, 3, 4];
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);

        // Phase B: if wraps nested dispatch. Concurrent encoder is the intended
        // SPI host for encodeStartIf. Wire comparison is the Reims VGPU encoder's enum
        // (not MTLCompareFunction): Equal=0 for buffer==reference (probed).
        let mut session = MetalBackend
            .open_compute_session(DispatchType::Concurrent)
            .expect("session");
        let start = ComputeCommand {
            kind: Kind::ControlStartIf,
            condition_buffer_ref: 8,
            condition_comparison: 0, // SPI Equal (buffer == reference)
            condition_reference_value: 1,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &start),
            ComputeStatus::Ok
        );
        let mut acc = ComputeAccum::default();
        acc.set_pipeline(6);
        acc.buffers.push(ComputeBufferBind {
            index: 0,
            buffer_ref: 7,
            offset: 0,
            attribute_stride: 0,
            has_attribute_stride: false,
        });
        let dcmd = threadgroups([1, 1, 1], [4, 1, 1]);
        assert_eq!(
            MetalBackend.execute_dispatch_nested(
                &mut state,
                &mut host,
                1,
                &acc,
                &dcmd,
                &mut session
            ),
            ComputeStatus::Ok
        );
        let end = ComputeCommand {
            kind: Kind::ControlEndIf,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &end),
            ComputeStatus::Ok
        );
        assert_eq!(session.finish(&mut host, &mut state, 1), ComputeStatus::Ok);
        let mut back = [0u8; 16];
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            buf_gva,
            &mut back,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        let out: Vec<u32> = back
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(out, vec![4, 7, 10, 13], "if-wrapped nested writeback");
    }

    #[test]
    fn icb_latches_sequencing_block() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut seg = ComputeSegment::default();
        let cmd = ComputeCommand {
            kind: Kind::ExecuteCommandsInBuffer,
            indirect_command_buffer_ref: 1,
            ..ComputeCommand::default()
        };
        let st = apply_sequencing(&mut state, &mut host, 1, &cmd, &mut seg);
        // Missing list entry → MissingBuffer; latches sequencing block.
        // Non-Apple metal stubs may short-circuit to NoMetal (Linux product).
        assert!(
            matches!(
                st,
                ComputeStatus::MissingBuffer(_)
                    | ComputeStatus::Unsupported(_)
                    | ComputeStatus::NoMetal(_)
            ),
            "unexpected {st:?}"
        );
        assert_eq!(seg.block, Some(SequencingBlock::IndirectCommandBuffer));
        if let Some(s) = seg.session.take() {
            let _ = s.finish(&mut host, &mut state, 1);
        }
    }
}
