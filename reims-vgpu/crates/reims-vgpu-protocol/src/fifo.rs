//! Child FIFO record layouts, and the two child commands decoded here.
//!
//! This module holds the parts of the FIFO contract the live drain path reads:
//! the resource-list / invalidate / synchronize record layout, the
//! display-descriptor timing entries, and the `EXEC_INDIRECT2` header offsets.
//!
//! The **opcodes** those records belong to are not here. `reims_vgpu::model::regs`
//! holds one table for the whole device — root and child together — and this
//! module used to restate five of its child entries, four of them byte-identical
//! declarations that nothing imported while the drain's dispatch matched on
//! `regs`. A second table of the same numbers cannot be kept honest by anything
//! in the toolchain: a correction to an opcode the RE got wrong reaches whichever
//! copy its author was reading, and the other one keeps compiling.
//!
//! Packet framing itself — reading a ring, walking headers and stamps, writing
//! back head — is the device model's: it runs against live guest memory and
//! reports each failure as a fault. Nothing here touches memory it was not
//! handed.

use crate::endian::{ld16, ld32, ld64, st16, st32, st64};
use alloc::string::{String, ToString as _};
use alloc::vec;
use alloc::vec::Vec;

// --- child record layout, as the PVG command table numbers them ---

/// CmdInvalidateResources / CmdSynchronizeResources shared header.
pub const CHILD_RESOURCE_LIST_TASK_ID: u32 = 0x00;
pub const CHILD_RESOURCE_LIST_COUNT: u32 = 0x04;
pub const CHILD_RESOURCE_LIST_HEADER_LEN: u32 = 8;
/// Per-object record on Invalidate: `{object_id u32}` + 4 validity-op bytes.
pub const CHILD_INVALIDATE_RECORD_LEN: u32 = 8;
/// Per-object record on Synchronize: `{object_id u32}` only (no validity ops).
pub const CHILD_SYNCHRONIZE_RECORD_LEN: u32 = 4;
/// `CmdDefineTask2`: the identity, extent and page-table root of one task.
///
/// The first word is **not** a task id. It is `(task_id << 1) | is_kernel_task`,
/// and the guest's kernel-task and user-task registrations differ only in that
/// low bit — the kernel task's own id is `0`, so a raw word of `0x1` is the
/// kernel task and not user task 1. Reading the word as an id indexes every
/// slot at twice its number.
pub const DEFINE_TASK_RAW_ID: usize = 0x00;
/// The task's address-space length. Eight bytes: the next field is at `0x0c`,
/// and a reader taking the low half truncates the length of any task spanning
/// 4 GiB or more.
pub const DEFINE_TASK_LENGTH: usize = 0x04;
/// The page-table directory's page frame.
pub const DEFINE_TASK_DIRECTORY_PFN: usize = DEFINE_TASK_LENGTH + 8;
/// Bytes the command carries, derived from its last field.
pub const DEFINE_TASK_LEN: usize = DEFINE_TASK_DIRECTORY_PFN + 4;
/// Bits the raw word is shifted by to recover the task id.
pub const DEFINE_TASK_ID_SHIFT: u32 = 1;

/// The identity a `DefineTask2` registers: a slot, and the class that owns it.
///
/// Total in the raw word — both halves are decoded, so nothing about the word
/// is left unaccounted for and neither half can be read as the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefineTaskId {
    pub task_id: u32,
    /// Whether the guest registered this as its kernel task.
    pub kernel: bool,
}

impl DefineTaskId {
    /// Split a raw first word into its two halves.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self {
            task_id: raw >> DEFINE_TASK_ID_SHIFT,
            kernel: raw & 1 != 0,
        }
    }

    /// The word a guest sends for this identity.
    ///
    /// The inverse of [`Self::from_raw`], so a test can build a request the way
    /// the guest builds it rather than shifting by hand and getting the
    /// polarity of the low bit from the same place the decoder does.
    #[must_use]
    pub const fn to_raw(self) -> u32 {
        (self.task_id << DEFINE_TASK_ID_SHIFT) | (self.kernel as u32)
    }
}

/// A task definition, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefineTaskCommand {
    pub id: DefineTaskId,
    pub length: u64,
    pub directory_pfn: u32,
}

/// Decode a task definition.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the command's four fields.
pub fn decode_define_task(payload: &[u8]) -> Result<DefineTaskCommand, ShortPayload> {
    if payload.len() < DEFINE_TASK_LEN {
        return Err(ShortPayload {
            plen: payload.len(),
            need: DEFINE_TASK_LEN,
        });
    }
    Ok(DefineTaskCommand {
        id: DefineTaskId::from_raw(ld32(&payload[DEFINE_TASK_RAW_ID..])),
        length: ld64(&payload[DEFINE_TASK_LENGTH..]),
        directory_pfn: ld32(&payload[DEFINE_TASK_DIRECTORY_PFN..]),
    })
}

/// `CmdDeleteTask`: the task slot, and nothing else.
///
/// A plain id, **not** the doubled word [`DEFINE_TASK_RAW_ID`] carries — the
/// same asymmetry the other resource-list commands have.
pub const DELETE_TASK_ID: usize = 0x00;
pub const DELETE_TASK_LEN: usize = 4;

/// Decode a task deletion.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the id. Typed rather than
/// defaulted to zero: task `0` is the kernel task, so a payload too short to
/// read that used to name the one slot whose teardown costs the most.
pub fn decode_delete_task(payload: &[u8]) -> Result<u32, ShortPayload> {
    if payload.len() < DELETE_TASK_LEN {
        return Err(ShortPayload {
            plen: payload.len(),
            need: DELETE_TASK_LEN,
        });
    }
    Ok(ld32(&payload[DELETE_TASK_ID..]))
}

/// `CmdSetObjectList`: where a task's object list lives and how long it is.
pub const SET_OBJECT_LIST_TASK_ID: usize = 0x00;
pub const SET_OBJECT_LIST_PFN: usize = 0x04;
pub const SET_OBJECT_LIST_COUNT: usize = SET_OBJECT_LIST_PFN + 4;
/// Bytes the command carries, derived from its last field. Only a payload
/// *below* this is refused — a longer one is accepted and its tail ignored,
/// which is what a wider field (a 64-bit page address, a byte length beside a
/// count) would look like from here.
pub const SET_OBJECT_LIST_LEN: usize = SET_OBJECT_LIST_COUNT + 4;

/// An object-list bind, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetObjectListCommand {
    pub task_id: u32,
    /// The page frame the list starts at.
    pub pfn: u32,
    /// How many entries it holds.
    pub count: u32,
}

/// Decode an object-list bind.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the command's three words.
pub fn decode_set_object_list(payload: &[u8]) -> Result<SetObjectListCommand, ShortPayload> {
    if payload.len() < SET_OBJECT_LIST_LEN {
        return Err(ShortPayload {
            plen: payload.len(),
            need: SET_OBJECT_LIST_LEN,
        });
    }
    Ok(SetObjectListCommand {
        task_id: ld32(&payload[SET_OBJECT_LIST_TASK_ID..]),
        pfn: ld32(&payload[SET_OBJECT_LIST_PFN..]),
        count: ld32(&payload[SET_OBJECT_LIST_COUNT..]),
    })
}

/// `CmdDeleteIOSurfaceBacking2` (`0x36`): an object and a task, **in that
/// order**.
///
/// The same two `u32`s [`TASK_OBJECT_TASK_ID`]'s record carries and the reverse
/// arrangement of them. That is the whole reason this is a separate decode: the
/// two records are indistinguishable by length, both fields are plain `u32`s,
/// and reading one at the other's offsets resolves the object as a task and the
/// task as an object without any check failing.
///
/// The object here is a **mapping** id — the surface whose host backing is
/// being retired — and not an object-list ref. The two number spaces overlap.
pub const DELETE_BACKING_OBJECT_ID: usize = 0x00;
pub const DELETE_BACKING_TASK_ID: usize = DELETE_BACKING_OBJECT_ID + 4;
pub const DELETE_BACKING_LEN: usize = DELETE_BACKING_TASK_ID + 4;

/// A backing retirement, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeleteBackingCommand {
    /// The mapping whose backing is retired.
    pub object_id: u32,
    pub task_id: u32,
}

/// Decode a backing retirement.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the command's two words.
pub fn decode_delete_backing(payload: &[u8]) -> Result<DeleteBackingCommand, ShortPayload> {
    if payload.len() < DELETE_BACKING_LEN {
        return Err(ShortPayload {
            plen: payload.len(),
            need: DELETE_BACKING_LEN,
        });
    }
    Ok(DeleteBackingCommand {
        object_id: ld32(&payload[DELETE_BACKING_OBJECT_ID..]),
        task_id: ld32(&payload[DELETE_BACKING_TASK_ID..]),
    })
}

/// `CmdDeleteObject` (`0x28`): a task, then one self-describing destroy record.
///
/// The guest writes the task id into four bytes of command space and copies the
/// record in after it, so the command declares no length of its own — the
/// record's header carries one, and it is the record's, not the command's.
pub const DELETE_OBJECT_TASK_ID: usize = 0x00;
/// Where the record starts. Every offset inside it is relative to this.
pub const DELETE_OBJECT_RECORD: usize = DELETE_OBJECT_TASK_ID + 4;
/// The record's own byte length: its header's second word, and therefore four
/// bytes into the record rather than into the command.
pub const DELETE_OBJECT_RECORD_LEN: usize = DELETE_OBJECT_RECORD + 4;
/// The floor — the task word plus the record's own header. A payload below it
/// cannot even say how long the record claims to be.
pub const DELETE_OBJECT_LEN: usize = DELETE_OBJECT_RECORD_LEN + 4;

/// A destroy command, framed.
///
/// The record is handed back as bytes: what is in it is the destroy family's
/// business, and this layer's job is that the slice really is inside the
/// payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeleteObjectCommand<'a> {
    pub task_id: u32,
    pub record: &'a [u8],
}

/// Why a destroy command's framing does not hold.
///
/// Two cases and not one, because they are different guest defects: a payload
/// too short to hold a record header means the command was truncated, while a
/// record claiming more than the payload holds means the record's own length
/// word is wrong. Only the second says anything about the record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeleteObjectError {
    ShortHeader(ShortPayload),
    RecordTruncated(ShortPayload),
}

impl DeleteObjectError {
    #[must_use]
    pub const fn short(self) -> ShortPayload {
        match self {
            Self::ShortHeader(short) | Self::RecordTruncated(short) => short,
        }
    }

    /// The reason stem the refusal reports under, without the `_short` suffix
    /// the failure channel appends.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ShortHeader(_) => "delete_object",
            Self::RecordTruncated(_) => "delete_object_record",
        }
    }
}

/// Frame a destroy command: the task it acts in, and the record it carries.
///
/// # Errors
///
/// [`DeleteObjectError`] when the payload cannot hold a record header, or when
/// the record's declared length runs past the payload. Both bounds are checked
/// here so a corrupt packet is reported as corrupt rather than as a command
/// this device merely has not implemented — those are different problems and
/// only one of them is closed by writing a handler.
pub fn decode_delete_object(payload: &[u8]) -> Result<DeleteObjectCommand<'_>, DeleteObjectError> {
    if payload.len() < DELETE_OBJECT_LEN {
        return Err(DeleteObjectError::ShortHeader(ShortPayload {
            plen: payload.len(),
            need: DELETE_OBJECT_LEN,
        }));
    }
    let record_len = ld32(&payload[DELETE_OBJECT_RECORD_LEN..]) as usize;
    // The record starts one word in, so it is `record_len + DELETE_OBJECT_RECORD`
    // that has to fit. Saturating, because the guest's length is an arbitrary
    // `u32` and the sum is what overflows.
    let need = record_len.saturating_add(DELETE_OBJECT_RECORD);
    if payload.len() < need {
        return Err(DeleteObjectError::RecordTruncated(ShortPayload {
            plen: payload.len(),
            need,
        }));
    }
    Ok(DeleteObjectCommand {
        task_id: ld32(&payload[DELETE_OBJECT_TASK_ID..]),
        record: &payload[DELETE_OBJECT_RECORD..],
    })
}

/// `CmdMapMemory2` (`0x39`) and `CmdUnmapMemory` (`0x22`): the interval of one
/// task's GPU virtual address space the guest has just changed.
///
/// **The command names an address range and no object.** The guest has already
/// applied the interval to its own page table and this packet is the notice; a
/// reader that took the second word for an object ref would resolve whatever
/// object happened to share the low half of a virtual address.
///
/// The two opcodes carry the identical record — they differ only in which
/// direction the guest moved — so the offsets are stated once and both decode
/// through [`decode_map_memory`].
pub const MAP_MEMORY_TASK_ID: usize = 0x00;
/// The base of the interval. Eight bytes, and *not* four: a task's virtual
/// address space is 64-bit and the high half is not padding.
pub const MAP_MEMORY_GVA: usize = 0x04;
/// The interval's length in bytes, also eight.
pub const MAP_MEMORY_LENGTH: usize = MAP_MEMORY_GVA + 8;
/// Bytes the record occupies, derived from its last field rather than written
/// as a literal beside the offsets it has to agree with.
pub const MAP_MEMORY_LEN: usize = MAP_MEMORY_LENGTH + 8;

/// A map-or-unmap notice, decoded.
///
/// Carries no direction: which of the two opcodes arrived is the caller's, and
/// putting it here would mean this decoder held a table of opcodes — see this
/// module's rule against a second copy of the opcode space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapMemoryCommand {
    pub task_id: u32,
    /// The base of the interval, in the task's GPU virtual address space.
    pub gva: u64,
    pub length: u64,
}

/// Decode a map-or-unmap notice.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the record's three fields.
pub fn decode_map_memory(payload: &[u8]) -> Result<MapMemoryCommand, ShortPayload> {
    if payload.len() < MAP_MEMORY_LEN {
        return Err(ShortPayload {
            plen: payload.len(),
            need: MAP_MEMORY_LEN,
        });
    }
    Ok(MapMemoryCommand {
        task_id: ld32(&payload[MAP_MEMORY_TASK_ID..]),
        gva: ld64(&payload[MAP_MEMORY_GVA..]),
        length: ld64(&payload[MAP_MEMORY_LENGTH..]),
    })
}

/// `CmdDeleteResource` (`0x25`) and `CmdReplacePhysical` (`0x3c`): a task and
/// one of its objects.
///
/// Two commands, one record, and the offsets are stated once because the two
/// arms that read them were two spellings of the same pair.
///
/// **`CmdDeleteIOSurfaceBacking2` (`0x36`) is not one of them.** Its payload is
/// the same two words in the *opposite* order — `{object_id, task_id}` — so a
/// reader that reached for this record would resolve the object as a task and
/// the task as an object. It has its own decode, and this note is why it is not
/// folded in here.
pub const TASK_OBJECT_TASK_ID: usize = 0x00;
pub const TASK_OBJECT_OBJECT_ID: usize = TASK_OBJECT_TASK_ID + 4;
pub const TASK_OBJECT_LEN: usize = TASK_OBJECT_OBJECT_ID + 4;
/// The two device-info request forms, and where each keeps its words.
///
/// The newer form carries a parse ceiling the older one does not, and it sits
/// *before* the two words they share — so the two forms' counts and reply
/// frames are at different offsets, and reading one form's request at the
/// other's offsets takes the ceiling for a count and the count for a page
/// frame. That is not a near miss: it would write the reply to whatever page
/// the pair count happens to name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceInfoForm {
    /// `[count][reply_pfn]`. No parse ceiling.
    WithoutKeyLimit,
    /// `[key_table_len][count][reply_pfn]`.
    WithKeyLimit,
}

impl DeviceInfoForm {
    /// Byte offset of the parse ceiling, for the form that carries one.
    ///
    /// The newer form prepends it, which is the entire difference between the
    /// two shapes — so every other offset here is stated relative to that.
    #[must_use]
    pub const fn key_table_len_offset(self) -> Option<usize> {
        match self {
            Self::WithoutKeyLimit => None,
            Self::WithKeyLimit => Some(0x00),
        }
    }

    /// Byte offset of the pair capacity.
    #[must_use]
    pub const fn pair_capacity_offset(self) -> usize {
        match self {
            Self::WithoutKeyLimit => 0x00,
            Self::WithKeyLimit => 0x04,
        }
    }

    /// Byte offset of the reply page frame.
    #[must_use]
    pub const fn reply_pfn_offset(self) -> usize {
        self.pair_capacity_offset() + 4
    }

    /// Bytes the request must carry.
    #[must_use]
    pub const fn request_len(self) -> usize {
        self.reply_pfn_offset() + 4
    }
}

/// A device-info request, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfoRequest {
    pub form: DeviceInfoForm,
    /// How many eight-byte pairs the guest's reply buffer holds — its
    /// allocation size in bytes, shifted right by three. One page, so 512 on a
    /// 4 KiB guest.
    ///
    /// The guest re-reads this word from its own staging copy to bound the
    /// walk, so it is never the host's to widen. The walk ends at whichever
    /// comes first: this many pairs, or a key of zero.
    pub pair_capacity: u32,
    /// One past the highest key the guest's walker parses, when the form
    /// carries one.
    ///
    /// **A table length, not a highest key**, so the reply may name every key
    /// strictly *below* this word. The guest writes a literal: 18 against a
    /// walker whose jump table runs the terminator through case 17, and the
    /// sibling compute-info request writes 5 against a table of case 0 through
    /// case 4. Read as a maximum, that 5 invents a key 5 with no arm, no field
    /// and no meaning — the guest's own walker sends it to the same skip arm as
    /// key 900. The reference host writes the reply under `keyLimit > K`, which
    /// is this polarity exactly.
    ///
    /// A **separate** bound from [`Self::pair_capacity`]: this bounds *which*
    /// keys the reply may name, that one bounds *how many* pairs fit. A reply
    /// is correct only when it respects both.
    ///
    /// `None` for [`DeviceInfoForm::WithoutKeyLimit`], and that is a claim
    /// rather than an absence. Nothing this device has driven issues that
    /// older opcode and no disassembly of its builder has been read, so its
    /// reply is bounded by the count alone. A ceiling read where there is none
    /// would sit at that form's count offset, taking the count for a ceiling
    /// and the reply frame for a count.
    pub key_table_len: Option<u32>,
    /// The page frame the reply is written to.
    pub reply_pfn: u32,
}

impl DeviceInfoRequest {
    /// The two bounds a reply to this request must respect.
    ///
    /// A form carrying no parse ceiling is bounded by its pair count alone, and
    /// that reading belongs here rather than at each caller: it was
    /// `key_table_len.unwrap_or(u32::MAX)` at one arm and a bare `u32::MAX` at
    /// the other, which are two spellings of one decision about a field only
    /// one of the two forms has.
    #[must_use]
    pub const fn reply_bounds(self) -> crate::info_reply::ReplyBounds {
        crate::info_reply::ReplyBounds {
            key_table_len: match self.key_table_len {
                Some(len) => len,
                None => u32::MAX,
            },
            count: self.pair_capacity,
        }
    }
}

/// Decode a device-info request.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the form's words.
pub fn decode_device_info(
    form: DeviceInfoForm,
    payload: &[u8],
) -> Result<DeviceInfoRequest, ShortPayload> {
    let need = form.request_len();
    if payload.len() < need {
        return Err(ShortPayload {
            plen: payload.len(),
            need,
        });
    }
    Ok(DeviceInfoRequest {
        form,
        key_table_len: form.key_table_len_offset().map(|at| ld32(&payload[at..])),
        pair_capacity: ld32(&payload[form.pair_capacity_offset()..]),
        reply_pfn: ld32(&payload[form.reply_pfn_offset()..]),
    })
}

/// A compute-info request, decoded.
///
/// The device-info query's sibling, and *not* one of its forms: it names a task
/// and a pipeline before the two bounds they share, and its reply destination
/// is a full guest address rather than a page frame. Folding it into
/// [`DeviceInfoForm`] would put a 64-bit address where a 32-bit frame number
/// goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeInfoRequest {
    /// The task word as the guest sent it. Resolving it is the caller's — the
    /// wire carries a raw word and which task it names is device state.
    pub raw_task: u32,
    /// The pipeline whose limits are being asked about.
    pub pipeline_ref: u32,
    /// One past the highest key the guest's walker parses. The same reading as
    /// [`DeviceInfoRequest::key_table_len`], including the polarity: the guest
    /// writes 5 against a table of case 0 through case 4.
    pub key_table_len: u32,
    /// How many eight-byte pairs the reply buffer holds.
    pub pair_capacity: u32,
    /// Where the reply goes: a full guest address, not a page frame.
    pub reply_gva: u64,
}

impl ComputeInfoRequest {
    /// The two bounds a reply to this request must respect. Both are always
    /// present here — this form has no ceiling-less variant.
    #[must_use]
    pub const fn reply_bounds(self) -> crate::info_reply::ReplyBounds {
        crate::info_reply::ReplyBounds {
            key_table_len: self.key_table_len,
            count: self.pair_capacity,
        }
    }
}

/// Bytes a compute-info request carries.
pub const COMPUTE_INFO_REQUEST_LEN: usize = 24;

/// Decode a compute-info request.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the request's six words.
pub fn decode_compute_info(payload: &[u8]) -> Result<ComputeInfoRequest, ShortPayload> {
    if payload.len() < COMPUTE_INFO_REQUEST_LEN {
        return Err(ShortPayload {
            plen: payload.len(),
            need: COMPUTE_INFO_REQUEST_LEN,
        });
    }
    Ok(ComputeInfoRequest {
        raw_task: ld32(payload),
        pipeline_ref: ld32(&payload[4..]),
        key_table_len: ld32(&payload[8..]),
        pair_capacity: ld32(&payload[12..]),
        // Bounded by the length check above, so the two halves are there.
        reply_gva: u64::from(ld32(&payload[16..])) | (u64::from(ld32(&payload[20..])) << 32),
    })
}

// --- CmdHeapTextureSizeAndAlign request and reply framing ---

/// Byte offset of the task word. Raw, as every request layout here carries it:
/// which task it names is device state and not the wire's.
pub const HEAP_TEXTURE_TASK_ID: usize = 0x00;
/// Byte offset of the reply destination — a full guest address, not a page
/// frame.
pub const HEAP_TEXTURE_REPLY_GVA: usize = HEAP_TEXTURE_TASK_ID + 4;
/// Byte offset of how many bytes the guest has set aside at that address.
pub const HEAP_TEXTURE_REPLY_LENGTH: usize = HEAP_TEXTURE_REPLY_GVA + 8;
/// Byte offset of the embedded record's declared length.
pub const HEAP_TEXTURE_SERIALIZER_LENGTH: usize = HEAP_TEXTURE_REPLY_LENGTH + 8;
/// Bytes the request header occupies, and therefore where the embedded
/// serializer record starts.
pub const HEAP_TEXTURE_REQUEST_HEADER_LEN: usize = HEAP_TEXTURE_SERIALIZER_LENGTH + 4;

/// Byte offset of the size the host requires, within the reply.
pub const HEAP_TEXTURE_REPLY_SIZE: usize = 0x00;
/// Byte offset of the alignment the host requires.
pub const HEAP_TEXTURE_REPLY_ALIGN: usize = HEAP_TEXTURE_REPLY_SIZE + 8;
/// Bytes the reply occupies: an `MTLSizeAndAlign`, two little-endian `u64`s.
///
/// This is the floor the request's own `reply_len` is checked against, and it
/// is the whole of the answer — the reply has no variable part, which is why
/// [`crate::info_reply`]'s bounds do not apply to it.
pub const HEAP_TEXTURE_REPLY_LEN: usize = HEAP_TEXTURE_REPLY_ALIGN + 8;

/// The record the request embeds: `heapTextureSizeAndAlignWithDescriptor:`.
///
/// Its tag is that selector's opcode and its length is that record's length,
/// both taken from the crate that derived them rather than written again here.
pub const HEAP_TEXTURE_SERIALIZED_TAG: u32 =
    reims_vgpu_wire::ops::texture::OPCODE_HEAP_TEXTURE_SIZE_AND_ALIGN;
pub const HEAP_TEXTURE_SERIALIZED_LEN: usize =
    reims_vgpu_wire::ops::texture::HEAP_TEXTURE_SIZE_AND_ALIGN_TOTAL_LEN as usize;

/// The host's answer: an `MTLSizeAndAlign`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeAndAlign {
    pub size: u64,
    pub align: u64,
}

impl SizeAndAlign {
    /// The bytes the guest reads back, at the offsets above.
    #[must_use]
    pub fn encode(self) -> [u8; HEAP_TEXTURE_REPLY_LEN] {
        let mut out = [0u8; HEAP_TEXTURE_REPLY_LEN];
        st64(&mut out[HEAP_TEXTURE_REPLY_SIZE..], self.size);
        st64(&mut out[HEAP_TEXTURE_REPLY_ALIGN..], self.align);
        out
    }
}

/// A heap-texture query request, framed.
///
/// The descriptor is borrowed rather than decoded: which of the two
/// `PGSerializedTextureDescriptor` bodies a record carries is a property of the
/// record's opcode, and this request's opcode names the narrow one — but
/// turning those bytes into fields is the reader's, not the framing's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapTextureRequest<'a> {
    /// The task word as the guest sent it.
    pub raw_task: u32,
    /// Where the reply goes.
    pub reply_gva: u64,
    /// How many bytes the guest set aside there. At least
    /// [`HEAP_TEXTURE_REPLY_LEN`], which [`decode_heap_texture_query`] checks.
    pub reply_len: u64,
    /// The embedded record's payload: a `PGSerializedTextureDescriptor` body.
    pub descriptor: &'a [u8],
}

/// Why a heap-texture query request is not one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapTextureRefusal {
    /// Too short to hold the request header.
    Short(ShortPayload),
    /// There is nowhere to write the reply: a null address, or a window that
    /// cannot hold an `MTLSizeAndAlign`.
    ///
    /// Refused rather than clamped. A short window means the guest and this
    /// device disagree about what the answer is, and writing the part that fits
    /// hands it a size with no alignment.
    ReplyDestination { gva: u64, len: u64 },
    /// The declared serializer length is not the embedded record's length, or
    /// is not the rest of the payload.
    ///
    /// One refusal for both because they are one claim: the request declares
    /// how long its record is, and that number has to agree with the record
    /// this opcode carries *and* with the bytes that arrived. A request where
    /// either fails is one whose two halves were written by different ideas of
    /// how long it is.
    SerializerLength { declared: u32, plen: usize },
    /// The embedded record is not the selector this request asks with.
    SerializerTag { found: u32 },
}

impl HeapTextureRefusal {
    /// The slug this refusal reports under.
    ///
    /// Inherent, because a caller that may not depend on `observe` still has to
    /// name the refusal it forwards — the same rule [`ShortPayload::SLUG`] is
    /// stated by.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Short(_) => "heap_query_short_payload",
            Self::ReplyDestination { .. } => "heap_query_bad_reply_length",
            Self::SerializerLength { .. } => "heap_query_bad_serializer_length",
            Self::SerializerTag { .. } => "heap_query_unknown_serializer_tag",
        }
    }
}

/// Frame a `CmdHeapTextureSizeAndAlign` request.
///
/// Returns the request's three words and a borrow of the embedded descriptor
/// body. The record's own `{opcode, length}` head is read through
/// [`mod@reims_vgpu_wire::op`] rather than at offsets restated here, so the
/// framing this device checks is the framing that crate derived.
///
/// # Errors
///
/// [`HeapTextureRefusal`]: too short for the header, a reply destination that
/// cannot hold the answer, a declared record length that disagrees with either
/// the opcode or the bytes, or a record of another selector.
pub fn decode_heap_texture_query(
    payload: &[u8],
) -> Result<HeapTextureRequest<'_>, HeapTextureRefusal> {
    if payload.len() < HEAP_TEXTURE_REQUEST_HEADER_LEN {
        return Err(HeapTextureRefusal::Short(ShortPayload {
            plen: payload.len(),
            need: HEAP_TEXTURE_REQUEST_HEADER_LEN,
        }));
    }
    let reply_gva = ld64(&payload[HEAP_TEXTURE_REPLY_GVA..]);
    let reply_len = ld64(&payload[HEAP_TEXTURE_REPLY_LENGTH..]);
    if reply_gva == 0 || reply_len < HEAP_TEXTURE_REPLY_LEN as u64 {
        return Err(HeapTextureRefusal::ReplyDestination {
            gva: reply_gva,
            len: reply_len,
        });
    }
    let declared = ld32(&payload[HEAP_TEXTURE_SERIALIZER_LENGTH..]);
    if declared as usize != HEAP_TEXTURE_SERIALIZED_LEN
        || payload.len() != HEAP_TEXTURE_REQUEST_HEADER_LEN + declared as usize
    {
        return Err(HeapTextureRefusal::SerializerLength {
            declared,
            plen: payload.len(),
        });
    }
    // The record's head is the serializer's own, so it is read through the
    // crate that derived it. `op` refuses a declared length that is under the
    // head or over the bytes present; the check above has already pinned it to
    // this opcode's one length, so the only thing left to judge is the tag.
    let record =
        reims_vgpu_wire::op(&payload[HEAP_TEXTURE_REQUEST_HEADER_LEN..], 0).map_err(|_| {
            HeapTextureRefusal::SerializerLength {
                declared,
                plen: payload.len(),
            }
        })?;
    if record.opcode() != HEAP_TEXTURE_SERIALIZED_TAG {
        return Err(HeapTextureRefusal::SerializerTag {
            found: record.opcode(),
        });
    }
    Ok(HeapTextureRequest {
        raw_task: ld32(&payload[HEAP_TEXTURE_TASK_ID..]),
        reply_gva,
        reply_len,
        descriptor: record.payload,
    })
}

/// `CmdDefineFifo` / `CmdFreeFifo`: the channel id, and nothing else.
///
/// One word, and it is the whole payload either command needs. Named rather
/// than written as a literal `4` at each site that bounds it, because the two
/// arms that read it and the two that length-check it were four separate
/// spellings of one number.
pub const CHANNEL_LIFETIME_CHANNEL_ID: u32 = 0x00;
pub const CHANNEL_LIFETIME_LEN: u32 = 4;
/// Hardcoded pageon second dword from `pageBacking` (LE bytes `01 00 00 01`).
///
/// Not a free-form bitfield. PVG host `invalidateResources:` treats the four
/// bytes after `ref` as:
/// `clear_host_valid | set_host_valid | clear_guest_valid | set_guest_valid`.
/// Pageon = clear hostValid + set guestValid (host cache stale; guest pages live).
pub const CHILD_INVALIDATE_PAGEON_FLAGS: u32 = 0x0100_0001;
/// A fixed-size FIFO payload shorter than the command that emitted it.
///
/// One type for every fixed-shape command here, because the fact is always the
/// same one: the guest sent fewer bytes than the command's own layout needs, so
/// there is nothing to read. The *command* is the caller's field to add — it
/// has the opcode and the channel — which is what keeps this from needing a
/// variant per command and keeps a new fixed-size decoder from needing a new
/// refusal.
///
/// One slug, shared across the commands that use it, on the same footing as
/// [`ResourceListDecodeError`]'s two: the caller's `op=` field is what tells
/// them apart in the log, and without it neither type could name the command
/// anyway — a decoder does not know its own opcode.
///
/// It exists because the alternative was in production: a caller checked the
/// floor with its own literal, reported that, and then called a decoder that
/// checked the same floor again and returned a bare `Option` whose `None` arm
/// dropped the packet in silence. Two checks of one fact, one of them
/// unreachable and the other unreported if it ever were.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShortPayload {
    pub plen: usize,
    pub need: usize,
}

impl ShortPayload {
    /// The slug this refusal reports under.
    ///
    /// A constant as well as a [`reims_vgpu_observe::Decline`] method, for the
    /// same reason [`ResourceListDecodeError::slug`] is inherent: a layer that
    /// may not depend on `observe` still has to name the refusal it forwards.
    pub const SLUG: &'static str = "fifo_payload_short";
}

impl reims_vgpu_observe::Decline for ShortPayload {
    fn slug(&self) -> &'static str {
        Self::SLUG
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("plen", self.plen.to_string()),
            ("need", self.need.to_string()),
        ]
    }
}

/// Why a `CmdInvalidateResources` / `CmdSynchronizeResources` payload did not
/// decode.
///
/// These two decoders returned a bare `Option`, and the drain's `None` arm could
/// only say `decode_fail` with the first two words of the payload. Three
/// different conditions reached it and a reader could not tell them apart —
/// which matters here more than in most decoders, because of what each refusal
/// costs. A dropped Invalidate leaves this device serving host-cached pixels for
/// a resource the guest has just CPU-written; a dropped Synchronize lets the
/// guest CPU-read pages whose deferred writeback never landed, which is the
/// class the sibling handler still names "boot-25 black-wallpaper".
///
/// One of those three conditions was **not a malformed payload at all**. A
/// `CHILD_RESOURCE_LIST_MAX_COUNT` of 256 sat above the length check and refused
/// any list longer than that, on the stated basis that "guest pageBacking
/// hardcodes count=1". It bounded nothing: the `Vec::with_capacity(count)` it
/// appeared to protect runs *after* `payload.len() < need`, so the payload's own
/// length is what caps the allocation, and `checked_mul` already covers the
/// arithmetic. Its only effect was to refuse a well-formed 257-entry list the
/// guest is entitled to send. It is gone, and this enum is what remains: the two
/// conditions that are genuinely the payload disagreeing with itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceListDecodeError {
    /// Shorter than the `{task_id, count}` header, so the count cannot be read.
    ShortHeader { plen: usize },
    /// The declared count needs more bytes than the payload carries. `need` is
    /// header + `count` × record length, so `need` against `plen` says by how
    /// much, and the record length distinguishes the two commands (8 vs 4).
    Truncated {
        count: u32,
        plen: usize,
        /// `u64`, not `usize`. This is a *guest-declared* size — header plus
        /// `count` records, where `count` is a word the guest wrote — so on a
        /// 32-bit host it can exceed what a `usize` holds, and a truncated one
        /// reads as a payload that is long enough. Keeping it `u64` means the
        /// comparison against `plen` is the only thing that has to be right,
        /// rather than a saturation nothing on this build's target can reach.
        need: u64,
    },
}

impl ResourceListDecodeError {
    /// The slug this refusal reports under.
    ///
    /// Inherent as well as a [`reims_vgpu_observe::Decline`] method, because a
    /// layer that may not depend on `observe` still has to be able to name the
    /// refusal it is forwarding — and naming it by writing the string again is
    /// how two spellings of one check come to exist.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ShortHeader { .. } => "resource_list_short_header",
            Self::Truncated { .. } => "resource_list_truncated",
        }
    }
}

impl reims_vgpu_observe::Decline for ResourceListDecodeError {
    fn slug(&self) -> &'static str {
        Self::slug(*self)
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match *self {
            Self::ShortHeader { plen } => vec![("plen", plen.to_string())],
            Self::Truncated { count, plen, need } => vec![
                ("count", count.to_string()),
                ("plen", plen.to_string()),
                ("need", need.to_string()),
            ],
        }
    }
}

pub const DISPLAY_DESC_TIMING_BASE: u64 = 0x210;
pub const DISPLAY_DESC_TIMING_STRIDE: u32 = 0x10;
pub const DISPLAY_TIMING_WIDTH: u32 = 0x00;
pub const DISPLAY_TIMING_HEIGHT: u32 = 0x02;
pub const DISPLAY_TIMING_REFRESH: u32 = 0x04;
pub const DISPLAY_TIMING_TAIL0: u32 = 0x08;
pub const DISPLAY_TIMING_TAIL1: u32 = 0x0c;
pub const DISPLAY_TIMING_REFRESH_FRAC_BITS: u32 = 16;

pub const CHILD_EXEC_INDIRECT_TASK_ID: u32 = 0x00;
pub const CHILD_EXEC_INDIRECT_RESOURCE_COUNT: u32 = 0x04;
pub const CHILD_EXEC_INDIRECT_CMDBUF_COUNT: u32 = 0x08;
pub const CHILD_EXEC_INDIRECT_HEADER_LEN: u32 = 12;
pub const CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN: u32 = 24;
pub const CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN: u32 = 16;
pub const CHILD_EXEC_INDIRECT_CMDBUF_GVA: u32 = 0x00;
pub const CHILD_EXEC_INDIRECT_CMDBUF_LENGTH: u32 = 0x08;

/// Per-resource descriptor offsets inside the EXEC_INDIRECT2 resource table.
///
/// The queue writes one 24-byte record per live list entry: `{object_id u32}`
/// followed by the same four validity-op bytes a `CmdInvalidateResources`
/// record carries, then 16 trailing bytes it zeroes.
pub const CHILD_EXEC_RESOURCE_OBJECT_ID: u32 = 0x00;
pub const CHILD_EXEC_RESOURCE_VALIDITY_OPS: u32 = 0x04;
pub const CHILD_EXEC_RESOURCE_TAIL: u32 = 0x08;
pub const CHILD_EXEC_RESOURCE_TAIL_LEN: u32 = 16;

// --- display-descriptor timing entries ---

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DisplayTimingEntry {
    pub width: u16,
    pub height: u16,
    pub refresh_1616: u32,
    pub tail0: u32,
    pub tail1: u32,
}

fn bounded_entry_offset_from_base(
    index: u32,
    base: u64,
    byte_capacity: u64,
    entry_len: u32,
) -> Option<u64> {
    if entry_len == 0 {
        return None;
    }
    let entry_offset = index as u64 * entry_len as u64;
    let offset = base.checked_add(entry_offset)?;
    if offset > byte_capacity || byte_capacity - offset < entry_len as u64 {
        None
    } else {
        Some(offset)
    }
}

pub fn display_refresh_hz_1616(refresh_hz: u32) -> Option<u32> {
    if refresh_hz > (u32::MAX >> DISPLAY_TIMING_REFRESH_FRAC_BITS) {
        return None;
    }
    Some(refresh_hz << DISPLAY_TIMING_REFRESH_FRAC_BITS)
}

pub fn encode_display_timing_entry(entry: &DisplayTimingEntry, dst: &mut [u8]) -> bool {
    if dst.len() < DISPLAY_DESC_TIMING_STRIDE as usize {
        return false;
    }
    st16(&mut dst[DISPLAY_TIMING_WIDTH as usize..], entry.width);
    st16(&mut dst[DISPLAY_TIMING_HEIGHT as usize..], entry.height);
    st32(
        &mut dst[DISPLAY_TIMING_REFRESH as usize..],
        entry.refresh_1616,
    );
    st32(&mut dst[DISPLAY_TIMING_TAIL0 as usize..], entry.tail0);
    st32(&mut dst[DISPLAY_TIMING_TAIL1 as usize..], entry.tail1);
    true
}

pub fn display_timing_entry_offset(index: u32, byte_capacity: u64) -> Option<u64> {
    bounded_entry_offset_from_base(
        index,
        DISPLAY_DESC_TIMING_BASE,
        byte_capacity,
        DISPLAY_DESC_TIMING_STRIDE,
    )
}

/// Validity ops packed after object_id in a CmdInvalidateResources record.
///
/// Wire layout (PVG host + guest pageon hardcode): four **u8** fields, not a bit mask.
/// Non-zero means apply that op to the resource's hostValid/guestValid state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InvalidateValidityOps {
    pub clear_host_valid: u8,
    pub set_host_valid: u8,
    pub clear_guest_valid: u8,
    pub set_guest_valid: u8,
}

impl InvalidateValidityOps {
    /// Decode LE dword as four validity-op bytes (`0x01000001` → clr host + set guest).
    pub fn from_le_dword(flags: u32) -> Self {
        let b = flags.to_le_bytes();
        Self {
            clear_host_valid: b[0],
            set_host_valid: b[1],
            clear_guest_valid: b[2],
            set_guest_valid: b[3],
        }
    }

    /// The four bytes back as the guest's dword.
    ///
    /// The inverse of [`Self::from_le_dword`], and not test-only: a refusal
    /// about a quad has to be able to say which quad arrived, and the records
    /// this comes off do not all keep the raw word beside it.
    #[must_use]
    pub const fn to_le_dword(self) -> u32 {
        u32::from_le_bytes([
            self.clear_host_valid,
            self.set_host_valid,
            self.clear_guest_valid,
            self.set_guest_valid,
        ])
    }

    /// Pageon hardcode: clr hostValid + set guestValid.
    pub const PAGEON: Self = Self {
        clear_host_valid: 1,
        set_host_valid: 0,
        clear_guest_valid: 0,
        set_guest_valid: 1,
    };

    /// Whether this quad is the guest-write transition: the guest's pages
    /// became the authoritative content and the host's copy stopped being one.
    ///
    /// **The only quad with an established meaning.** `pageBacking` writes
    /// [`Self::PAGEON`] and writes nothing else, so that pairing — and only
    /// that pairing — is a transition this device knows how to make. The four
    /// fields are independent on the wire and a build that set a different
    /// combination would be asking for something no capture has shown; a
    /// reader that applied the guest-write transition to it anyway would be
    /// moving content authority the guest did not move, which is a stale draw
    /// or a lost CPU write rather than a wrong log line.
    ///
    /// So this is a question with a `false` answer rather than a classifier
    /// with a default. What a caller does with `false` is the caller's, and in
    /// this project it is a typed refusal.
    #[must_use]
    pub const fn is_guest_write(self) -> bool {
        self.clear_host_valid != 0
            && self.set_guest_valid != 0
            && self.set_host_valid == 0
            && self.clear_guest_valid == 0
    }
}

/// One CmdInvalidateResources object record (RE: `pageBacking` second `getCommandBytes(8)`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InvalidateResourceRecord {
    pub object_id: u32,
    /// LE dword form of the four validity-op bytes (see [`InvalidateValidityOps`]).
    pub flags: u32,
    pub ops: InvalidateValidityOps,
}

/// FIFO CmdInvalidateResources (0x34) payload (RE pageBacking + live plen=16).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InvalidateResourcesCommand {
    pub task_id: u32,
    pub count: u32,
    pub records: Vec<InvalidateResourceRecord>,
}

/// A task and one of its objects: the whole payload of `CmdDeleteResource`
/// (`0x25`) and of `CmdReplacePhysical` (`0x3c`).
///
/// See [`TASK_OBJECT_TASK_ID`] for the third command that carries the same two
/// words in the other order and must not reach this decode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaskObjectCommand {
    pub task_id: u32,
    pub object_id: u32,
}

/// FIFO CmdSynchronizeResources (0x35) payload (RE synchronizeForUnwire + live plen=12).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SynchronizeResourcesCommand {
    pub task_id: u32,
    pub count: u32,
    pub object_ids: Vec<u32>,
}

/// Decode CmdInvalidateResources: header `{task_id, count}` + `count × {object_id, flags}`.
///
/// Guest `pageBacking` always writes count=1 and flags=`0x1000001` on the observed
/// guest driver; the decoder accepts any count the payload is long enough to
/// carry, which is what makes it forward-compatible with a serializer that
/// batches. The payload's length is the only bound — see
/// [`ResourceListDecodeError`] for the cap that used to sit above it and what
/// that cap was actually doing.
pub fn decode_invalidate_resources(
    payload: &[u8],
) -> Result<InvalidateResourcesCommand, ResourceListDecodeError> {
    let plen = payload.len();
    if plen < CHILD_RESOURCE_LIST_HEADER_LEN as usize {
        return Err(ResourceListDecodeError::ShortHeader { plen });
    }
    let task_id = ld32(&payload[CHILD_RESOURCE_LIST_TASK_ID as usize..]);
    let count = ld32(&payload[CHILD_RESOURCE_LIST_COUNT as usize..]);
    let need = resource_list_need(count, CHILD_INVALIDATE_RECORD_LEN);
    if (plen as u64) < need {
        return Err(ResourceListDecodeError::Truncated { count, plen, need });
    }
    // Bounded by `need <= plen`, which was just checked: a declared count the
    // payload cannot carry never reaches here, so this reserves at most one
    // record per record's worth of bytes the guest actually sent.
    let mut records = Vec::with_capacity(count as usize);
    let mut off = CHILD_RESOURCE_LIST_HEADER_LEN as usize;
    for _ in 0..count {
        let object_id = ld32(&payload[off..]);
        let flags = ld32(&payload[off + 4..]);
        records.push(InvalidateResourceRecord {
            object_id,
            flags,
            ops: InvalidateValidityOps::from_le_dword(flags),
        });
        off += CHILD_INVALIDATE_RECORD_LEN as usize;
    }
    Ok(InvalidateResourcesCommand {
        task_id,
        count,
        records,
    })
}

/// Bytes a resource-list payload must carry for the count it declares.
///
/// One spelling for both commands, because the only thing that differs between
/// them is `record_len`, and a second copy of this arithmetic is how the two
/// would come to disagree about what "long enough" means.
const fn resource_list_need(count: u32, record_len: u32) -> u64 {
    // Exact, and it cannot wrap: `count` is a `u32` and `record_len` is 4 or 8,
    // so the product is at most 2^35 and the sum fits `u64` with room to spare.
    // Nothing narrows it afterwards — see `ResourceListDecodeError::Truncated`
    // for why the answer is not a `usize`.
    count as u64 * record_len as u64 + CHILD_RESOURCE_LIST_HEADER_LEN as u64
}

/// One entry of the per-resource table an `EXEC_INDIRECT2` payload carries
/// between its 12-byte header and its command-buffer descriptors.
///
/// The guest builds this in `writeInvalidates`, one record per live entry of the
/// submission's `AppleParavirtSegmentResourceList`. The first eight bytes are
/// byte-identical in layout to a `CmdInvalidateResources` record, so [`ops`]
/// comes off the same [`InvalidateValidityOps`] decoder — the record *lengths*
/// differ (8 vs 24), the quad does not.
///
/// `clear_host_valid` is sourced from `AppleParavirtResource::shouldInvalidateHost()`,
/// which is a `lock btr` test-and-clear of the resource's dirty bit plus a sticky
/// flag it also clears. `writeInvalidates` is its only caller, so the guest's
/// statement that it CPU-wrote a resource is delivered here exactly once and is
/// never resent.
///
/// [`ops`]: Self::ops
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecResourceDesc {
    pub object_id: u32,
    pub ops: InvalidateValidityOps,
    /// Bytes `+0x08..0x18`. Zeroed by the Ventura 13.7.8 x86 guest driver; kept
    /// raw rather than dropped so a build that populates them is visible instead
    /// of silently discarded.
    pub tail: [u8; CHILD_EXEC_RESOURCE_TAIL_LEN as usize],
}

impl ExecResourceDesc {
    /// How many of the 16 trailing bytes this record actually sets.
    pub fn tail_nonzero_bytes(&self) -> u32 {
        self.tail.iter().filter(|b| **b != 0).count() as u32
    }
}

/// The `EXEC_INDIRECT2` header: whose task the submission is, and how many
/// records follow it in each of the two tables.
///
/// A type rather than three loose `ld32`s at every reader, because the three
/// are read together and a reader that took the counts without the task would
/// have a table of object refs and no namespace to resolve them in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecIndirectHeader {
    pub task_id: u32,
    pub resource_count: u32,
    pub cmdbuf_count: u32,
}

/// Decode the `EXEC_INDIRECT2` header.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the three words. The counts
/// are the guest's and are *not* checked against the payload here — that is
/// each table decoder's own bound, because the two tables have different record
/// lengths and a reader of one has no business being refused by the other's
/// length.
pub fn decode_exec_header(payload: &[u8]) -> Result<ExecIndirectHeader, ShortPayload> {
    let need = CHILD_EXEC_INDIRECT_HEADER_LEN as usize;
    if payload.len() < need {
        return Err(ShortPayload {
            plen: payload.len(),
            need,
        });
    }
    Ok(ExecIndirectHeader {
        task_id: ld32(&payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..]),
        resource_count: ld32(&payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..]),
        cmdbuf_count: ld32(&payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..]),
    })
}

/// Decode the `EXEC_INDIRECT2` resource table: `resource_count` records of
/// [`CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN`] bytes, starting right after the
/// header.
///
/// The same refusal the other list decoders here give, so a malformed or
/// truncated submission is a caller-visible failure with a reason rather than a
/// partial table. The bound is checked before the allocation, so a hostile
/// `resource_count` cannot reserve memory the payload does not back.
///
/// # Errors
///
/// [`ResourceListDecodeError::ShortHeader`] where the header is not there, and
/// [`ResourceListDecodeError::Truncated`] where the declared table is not.
pub fn decode_exec_resource_table(
    payload: &[u8],
) -> Result<Vec<ExecResourceDesc>, ResourceListDecodeError> {
    let plen = payload.len();
    let count = decode_exec_header(payload)
        .map_err(|_| ResourceListDecodeError::ShortHeader { plen })?
        .resource_count;
    // Saturating rather than checked: a count that overflows the multiply is a
    // count no payload can back, which is the same answer `need > plen` gives —
    // and it is the answer with the guest's own numbers in it.
    let need = (CHILD_EXEC_INDIRECT_HEADER_LEN as u64).saturating_add(
        (count as u64).saturating_mul(CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as u64),
    );
    if need > plen as u64 {
        return Err(ResourceListDecodeError::Truncated { count, plen, need });
    }
    let mut descs = Vec::with_capacity(count as usize);
    let mut off = CHILD_EXEC_INDIRECT_HEADER_LEN as usize;
    for _ in 0..count {
        let object_id = ld32(&payload[off + CHILD_EXEC_RESOURCE_OBJECT_ID as usize..]);
        let flags = ld32(&payload[off + CHILD_EXEC_RESOURCE_VALIDITY_OPS as usize..]);
        let tail_off = off + CHILD_EXEC_RESOURCE_TAIL as usize;
        let mut tail = [0u8; CHILD_EXEC_RESOURCE_TAIL_LEN as usize];
        tail.copy_from_slice(&payload[tail_off..tail_off + CHILD_EXEC_RESOURCE_TAIL_LEN as usize]);
        descs.push(ExecResourceDesc {
            object_id,
            ops: InvalidateValidityOps::from_le_dword(flags),
            tail,
        });
        off += CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
    }
    Ok(descs)
}

/// `CmdDisplaySetSharedStatePage`'s payload: `{u32 pipe index, u32 page PFN}`.
///
/// The page itself — what the device reads out of it and writes back into it —
/// is the display layer's. This is only where the two words are.
pub const CHILD_SHARED_STATE_INDEX: usize = 0x00;
pub const CHILD_SHARED_STATE_PFN: usize = 0x04;
pub const CHILD_SHARED_STATE_LEN: usize = 8;

/// Which display pipe, and the page frame its shared state lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedStatePage {
    pub index: u32,
    pub pfn: u32,
}

/// Decode `CmdDisplaySetSharedStatePage` (`0x01`, on either channel).
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the two words.
pub fn decode_shared_state(payload: &[u8]) -> Result<SharedStatePage, ShortPayload> {
    if payload.len() < CHILD_SHARED_STATE_LEN {
        return Err(ShortPayload {
            plen: payload.len(),
            need: CHILD_SHARED_STATE_LEN,
        });
    }
    Ok(SharedStatePage {
        index: ld32(&payload[CHILD_SHARED_STATE_INDEX..]),
        pfn: ld32(&payload[CHILD_SHARED_STATE_PFN..]),
    })
}

// --- the two cursor commands' records --------------------------------------

/// `CmdSetCursorGlyph`: where the sprite is, and how to read it.
///
/// Word zero is not read. The device has never needed it and what it carries is
/// not established, so it is left unnamed rather than given a speculative name
/// that a later reader would take for a decoded field. The same is true of the
/// word at `0x28` that [`CURSOR_GLYPH_LEN`] reaches past `hot_y` to include.
pub const CURSOR_GLYPH_TASK_ID: usize = 0x04;
pub const CURSOR_GLYPH_VIRTUAL_OFFSET: usize = 0x08;
pub const CURSOR_GLYPH_MAPPED_LENGTH: usize = 0x10;
pub const CURSOR_GLYPH_STRIDE: usize = 0x18;
pub const CURSOR_GLYPH_WIDTH: usize = 0x20;
pub const CURSOR_GLYPH_HEIGHT: usize = 0x22;
pub const CURSOR_GLYPH_HOT_X: usize = 0x24;
pub const CURSOR_GLYPH_HOT_Y: usize = 0x26;
pub const CURSOR_GLYPH_LEN: usize = 0x2c;

/// One `CmdSetCursorGlyph` record, as the wire spells it.
///
/// `stride` is the field's own width and not the device's. The wire carries
/// eight bytes; whether a value that does not fit a pixel pitch is usable is
/// the consumer's question, and narrowing here would answer it by wrapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CursorGlyphRecord {
    /// The task whose address space `virtual_offset` is in.
    pub task_id: u32,
    pub virtual_offset: u64,
    pub mapped_length: u64,
    pub stride: u64,
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
}

/// Decode `CmdSetCursorGlyph` (`0x04`).
///
/// Layout only. Every bound that makes a glyph *usable* — the sprite the
/// consumer will allocate, whether the hotspot is inside the sprite, whether
/// the mapped length covers the rows — is the consumer's, because the consumer
/// is the only thing that can refuse.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the record.
pub fn decode_cursor_glyph(payload: &[u8]) -> Result<CursorGlyphRecord, ShortPayload> {
    if payload.len() < CURSOR_GLYPH_LEN {
        return Err(ShortPayload {
            plen: payload.len(),
            need: CURSOR_GLYPH_LEN,
        });
    }
    Ok(CursorGlyphRecord {
        task_id: ld32(&payload[CURSOR_GLYPH_TASK_ID..]),
        virtual_offset: ld64(&payload[CURSOR_GLYPH_VIRTUAL_OFFSET..]),
        mapped_length: ld64(&payload[CURSOR_GLYPH_MAPPED_LENGTH..]),
        stride: ld64(&payload[CURSOR_GLYPH_STRIDE..]),
        width: ld16(&payload[CURSOR_GLYPH_WIDTH..]),
        height: ld16(&payload[CURSOR_GLYPH_HEIGHT..]),
        hot_x: ld16(&payload[CURSOR_GLYPH_HOT_X..]),
        hot_y: ld16(&payload[CURSOR_GLYPH_HOT_Y..]),
    })
}

/// `CmdSetCursorShow`: the visibility flag at word one. Word zero is not read,
/// for [`CURSOR_GLYPH_TASK_ID`]'s reason.
pub const CURSOR_SHOW_FLAG: usize = 0x04;
pub const CURSOR_SHOW_LEN: usize = 8;

/// Decode `CmdSetCursorShow` (`0x05`): whether the cursor is drawn.
///
/// Any non-zero value shows it. The wire carries a word and the command is a
/// boolean, so treating one as the other is the whole of the contract.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the two words.
pub fn decode_cursor_show(payload: &[u8]) -> Result<bool, ShortPayload> {
    if payload.len() < CURSOR_SHOW_LEN {
        return Err(ShortPayload {
            plen: payload.len(),
            need: CURSOR_SHOW_LEN,
        });
    }
    Ok(ld32(&payload[CURSOR_SHOW_FLAG..]) != 0)
}

/// Decode `CmdDefineFifo` / `CmdFreeFifo`: the channel id at word zero.
///
/// The two commands share a payload because they are the two ends of one
/// lifetime, so they share a decoder. Which of them a packet is stays the
/// caller's — a decoder does not know its own opcode.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the one word.
pub fn decode_channel_lifetime(payload: &[u8]) -> Result<u32, ShortPayload> {
    let need = CHANNEL_LIFETIME_LEN as usize;
    if payload.len() < need {
        return Err(ShortPayload {
            plen: payload.len(),
            need,
        });
    }
    Ok(ld32(&payload[CHANNEL_LIFETIME_CHANNEL_ID as usize..]))
}

/// Decode a `{task_id, object_id}` payload: `CmdDeleteResource` (`0x25`) or
/// `CmdReplacePhysical` (`0x3c`), eight bytes.
///
/// `CmdReplacePhysical` is emitted once per attached resource at the tail of a
/// re-commit into the GPU page table — after the range was released, its pages
/// were wired to different host frames, and the new PFNs were written back at
/// the *same* GPU-VA. It therefore carries no address of its own: the GVA is
/// unchanged, and only the translation behind it moved.
///
/// `task_id` is a plain slot id, as the other resource-list commands carry it,
/// and not the doubled `DefineTask2` word.
///
/// # Errors
///
/// [`ShortPayload`] when the payload cannot hold the command's eight bytes.
/// Typed rather than a bare `None`, so callers report it instead of checking
/// the same floor themselves and dropping the `None` in silence.
pub fn decode_task_object(payload: &[u8]) -> Result<TaskObjectCommand, ShortPayload> {
    let need = TASK_OBJECT_LEN;
    if payload.len() < need {
        return Err(ShortPayload {
            plen: payload.len(),
            need,
        });
    }
    Ok(TaskObjectCommand {
        task_id: ld32(&payload[TASK_OBJECT_TASK_ID..]),
        object_id: ld32(&payload[TASK_OBJECT_OBJECT_ID..]),
    })
}

/// Decode CmdSynchronizeResources: header `{task_id, count}` + `count × {object_id}`.
///
/// Guest `synchronizeForUnwire` uses `getCommandBytes(4)` for the object cell (no flags).
pub fn decode_synchronize_resources(
    payload: &[u8],
) -> Result<SynchronizeResourcesCommand, ResourceListDecodeError> {
    let plen = payload.len();
    if plen < CHILD_RESOURCE_LIST_HEADER_LEN as usize {
        return Err(ResourceListDecodeError::ShortHeader { plen });
    }
    let task_id = ld32(&payload[CHILD_RESOURCE_LIST_TASK_ID as usize..]);
    let count = ld32(&payload[CHILD_RESOURCE_LIST_COUNT as usize..]);
    let need = resource_list_need(count, CHILD_SYNCHRONIZE_RECORD_LEN);
    if (plen as u64) < need {
        return Err(ResourceListDecodeError::Truncated { count, plen, need });
    }
    // Bounded by the length check above; see `decode_invalidate_resources`.
    let mut object_ids = Vec::with_capacity(count as usize);
    let mut off = CHILD_RESOURCE_LIST_HEADER_LEN as usize;
    for _ in 0..count {
        object_ids.push(ld32(&payload[off..]));
        off += CHILD_SYNCHRONIZE_RECORD_LEN as usize;
    }
    Ok(SynchronizeResourcesCommand {
        task_id,
        count,
        object_ids,
    })
}

#[cfg(test)]
mod cursor_tests {
    use super::*;

    /// The offsets are the record, so a test that rebuilt them from the same
    /// constants would agree with any layout at all. This spells the bytes.
    #[test]
    fn a_glyph_record_decodes_to_the_bytes_the_wire_carried() {
        let mut payload = vec![0u8; CURSOR_GLYPH_LEN];
        payload[0x00..0x04].copy_from_slice(&0xdead_beef_u32.to_le_bytes());
        payload[0x04..0x08].copy_from_slice(&7_u32.to_le_bytes());
        payload[0x08..0x10].copy_from_slice(&0x1_0000_u64.to_le_bytes());
        payload[0x10..0x18].copy_from_slice(&0x4000_u64.to_le_bytes());
        payload[0x18..0x20].copy_from_slice(&256_u64.to_le_bytes());
        payload[0x20..0x22].copy_from_slice(&64_u16.to_le_bytes());
        payload[0x22..0x24].copy_from_slice(&32_u16.to_le_bytes());
        payload[0x24..0x26].copy_from_slice(&3_u16.to_le_bytes());
        payload[0x26..0x28].copy_from_slice(&5_u16.to_le_bytes());
        assert_eq!(
            decode_cursor_glyph(&payload),
            Ok(CursorGlyphRecord {
                task_id: 7,
                virtual_offset: 0x1_0000,
                mapped_length: 0x4000,
                stride: 256,
                width: 64,
                height: 32,
                hot_x: 3,
                hot_y: 5,
            }),
            "word zero is not a field, and reading it as one would shift every \
             other offset by four"
        );
    }

    /// The field is eight bytes on the wire. Narrowing it here would wrap a
    /// stride that does not fit into a small one that passes every bound the
    /// consumer checks, and the sprite would be read from the wrong rows.
    #[test]
    fn a_stride_that_does_not_fit_a_word_is_carried_whole() {
        let mut payload = vec![0u8; CURSOR_GLYPH_LEN];
        payload[0x18..0x20].copy_from_slice(&0x1_0000_0100_u64.to_le_bytes());
        assert_eq!(
            decode_cursor_glyph(&payload).expect("long enough").stride,
            0x1_0000_0100
        );
    }

    #[test]
    fn a_glyph_payload_one_byte_short_is_refused() {
        assert_eq!(
            decode_cursor_glyph(&vec![0u8; CURSOR_GLYPH_LEN - 1]),
            Err(ShortPayload {
                plen: CURSOR_GLYPH_LEN - 1,
                need: CURSOR_GLYPH_LEN,
            })
        );
    }

    #[test]
    fn any_non_zero_word_shows_the_cursor() {
        let mut payload = vec![0u8; CURSOR_SHOW_LEN];
        assert_eq!(decode_cursor_show(&payload), Ok(false));
        payload[CURSOR_SHOW_FLAG] = 2;
        assert_eq!(
            decode_cursor_show(&payload),
            Ok(true),
            "the command is a boolean and the wire carries a word"
        );
        // And the flag is word one: a value in word zero shows nothing.
        let mut only_word_zero = vec![0u8; CURSOR_SHOW_LEN];
        only_word_zero[0] = 0xff;
        assert_eq!(decode_cursor_show(&only_word_zero), Ok(false));
    }

    #[test]
    fn a_show_payload_one_byte_short_is_refused() {
        assert_eq!(
            decode_cursor_show(&vec![0u8; CURSOR_SHOW_LEN - 1]),
            Err(ShortPayload {
                plen: CURSOR_SHOW_LEN - 1,
                need: CURSOR_SHOW_LEN,
            })
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endian::st32;

    /// The header's three words, and a payload one byte short of them refuses
    /// rather than reading a count out of whatever follows.
    #[test]
    fn an_exec_header_is_three_words_and_a_short_one_refuses() {
        let mut payload = vec![0u8; CHILD_EXEC_INDIRECT_HEADER_LEN as usize];
        st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 7);
        st32(
            &mut payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..],
            2,
        );
        st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 5);
        assert_eq!(
            decode_exec_header(&payload),
            Ok(ExecIndirectHeader {
                task_id: 7,
                resource_count: 2,
                cmdbuf_count: 5,
            })
        );
        assert_eq!(
            decode_exec_header(&payload[..payload.len() - 1]),
            Err(ShortPayload {
                plen: CHILD_EXEC_INDIRECT_HEADER_LEN as usize - 1,
                need: CHILD_EXEC_INDIRECT_HEADER_LEN as usize,
            })
        );
    }

    /// The four bytes after an EXEC resource record's object ref are the same
    /// validity quad a standalone invalidate carries, read at this table's own
    /// 24-byte stride.
    #[test]
    fn an_exec_resource_record_carries_the_same_validity_quad() {
        let payload = exec_payload_with_table(&[(11, 0x0100_0001, [0u8; 16])]);
        let table = decode_exec_resource_table(&payload).expect("one record, and it is there");
        assert_eq!(table.len(), 1);
        assert_eq!(table[0].object_id, 11);
        assert!(table[0].ops.is_guest_write());
        assert_eq!(table[0].ops.to_le_dword(), 0x0100_0001);
        assert_eq!(table[0].tail_nonzero_bytes(), 0);
    }

    /// The guest-write transition is one exact quad, not "some bits set".
    ///
    /// The four fields are independent bytes, so a reader that tested any one
    /// of them would accept three quads that ask for something else — and each
    /// of those moves content authority in a direction no capture has shown.
    #[test]
    fn only_the_pageon_quad_is_the_guest_write_transition() {
        assert!(InvalidateValidityOps::PAGEON.is_guest_write());
        assert_eq!(InvalidateValidityOps::PAGEON.to_le_dword(), 0x0100_0001);
        assert!(InvalidateValidityOps::from_le_dword(0x0100_0001).is_guest_write());

        // Asks for nothing.
        assert!(!InvalidateValidityOps::default().is_guest_write());
        // Each half alone is not the pair.
        assert!(!InvalidateValidityOps::from_le_dword(0x0000_0001).is_guest_write());
        assert!(!InvalidateValidityOps::from_le_dword(0x0100_0000).is_guest_write());
        // The opposite direction, and the pair with an extra op beside it.
        assert!(!InvalidateValidityOps::from_le_dword(0x0001_0100).is_guest_write());
        assert!(!InvalidateValidityOps::from_le_dword(0x0100_0101).is_guest_write());
    }

    /// The raw first word is an id and a class bit, and both halves survive a
    /// round trip.
    ///
    /// The kernel task's own id is `0`, so `0x1` is the kernel task and not
    /// user task 1 — the one pair a reader that ignores the low bit gets
    /// exactly backwards.
    #[test]
    fn a_define_task_word_is_an_id_and_a_class_bit() {
        assert_eq!(
            DefineTaskId::from_raw(0x1),
            DefineTaskId {
                task_id: 0,
                kernel: true
            },
            "the kernel task's id is zero"
        );
        assert_eq!(
            DefineTaskId::from_raw(0x2),
            DefineTaskId {
                task_id: 1,
                kernel: false
            },
            "user task 1 sends two"
        );
        for task_id in [0u32, 1, 3, 0x7fff_ffff] {
            for kernel in [false, true] {
                let id = DefineTaskId { task_id, kernel };
                assert_eq!(DefineTaskId::from_raw(id.to_raw()), id);
            }
        }
    }

    /// The definition's four fields, including the length that is eight bytes
    /// wide and sat between two `u32`s.
    #[test]
    fn a_task_definition_carries_a_sixty_four_bit_extent() {
        let mut bytes = vec![0u8; DEFINE_TASK_LEN];
        st32(
            &mut bytes[DEFINE_TASK_RAW_ID..],
            DefineTaskId {
                task_id: 5,
                kernel: false,
            }
            .to_raw(),
        );
        bytes[DEFINE_TASK_LENGTH..DEFINE_TASK_LENGTH + 8]
            .copy_from_slice(&0x0000_0004_0000_0000u64.to_le_bytes());
        st32(&mut bytes[DEFINE_TASK_DIRECTORY_PFN..], 0x2a);
        assert_eq!(
            decode_define_task(&bytes),
            Ok(DefineTaskCommand {
                id: DefineTaskId {
                    task_id: 5,
                    kernel: false
                },
                length: 0x0000_0004_0000_0000,
                directory_pfn: 0x2a,
            }),
            "a task spanning 16 GiB keeps its high half"
        );
        assert_eq!(
            decode_define_task(&bytes[..DEFINE_TASK_LEN - 1]),
            Err(ShortPayload {
                plen: DEFINE_TASK_LEN - 1,
                need: DEFINE_TASK_LEN,
            })
        );
    }

    /// A delete names a task, and a payload too short to hold one is refused
    /// rather than read as task `0` — which is the kernel task.
    #[test]
    fn a_short_delete_task_does_not_name_the_kernel_task() {
        assert_eq!(decode_delete_task(&7u32.to_le_bytes()), Ok(7));
        assert_eq!(
            decode_delete_task(&[0u8; DELETE_TASK_LEN - 1]),
            Err(ShortPayload {
                plen: DELETE_TASK_LEN - 1,
                need: DELETE_TASK_LEN,
            })
        );
    }

    /// Three words, and a longer payload is accepted with its tail ignored.
    #[test]
    fn an_object_list_bind_is_three_words_and_tolerates_a_longer_payload() {
        let mut bytes = vec![0u8; SET_OBJECT_LIST_LEN + 8];
        st32(&mut bytes[SET_OBJECT_LIST_TASK_ID..], 2);
        st32(&mut bytes[SET_OBJECT_LIST_PFN..], 0x310);
        st32(&mut bytes[SET_OBJECT_LIST_COUNT..], 64);
        let expected = SetObjectListCommand {
            task_id: 2,
            pfn: 0x310,
            count: 64,
        };
        assert_eq!(decode_set_object_list(&bytes), Ok(expected));
        assert_eq!(
            decode_set_object_list(&bytes[..SET_OBJECT_LIST_LEN]),
            Ok(expected)
        );
        assert_eq!(
            decode_set_object_list(&bytes[..SET_OBJECT_LIST_LEN - 1]),
            Err(ShortPayload {
                plen: SET_OBJECT_LIST_LEN - 1,
                need: SET_OBJECT_LIST_LEN,
            })
        );
    }

    /// The notice's three fields, and the two that are eight bytes wide.
    ///
    /// A reader that took the address for a `u32` would decode a base with its
    /// high half discarded and then read the length from the middle of the
    /// address — so a mapping high in the task's space would be reported as a
    /// mapping low in it, of a nonsense length. Both halves are driven here.
    #[test]
    fn a_map_notice_is_a_task_and_a_sixty_four_bit_interval() {
        let mut bytes = vec![0u8; MAP_MEMORY_LEN];
        st32(&mut bytes[MAP_MEMORY_TASK_ID..], 3);
        bytes[MAP_MEMORY_GVA..MAP_MEMORY_GVA + 8]
            .copy_from_slice(&0x0000_7f12_3456_1000u64.to_le_bytes());
        bytes[MAP_MEMORY_LENGTH..MAP_MEMORY_LENGTH + 8]
            .copy_from_slice(&0x01c3_e000u64.to_le_bytes());
        assert_eq!(
            decode_map_memory(&bytes),
            Ok(MapMemoryCommand {
                task_id: 3,
                gva: 0x0000_7f12_3456_1000,
                length: 0x01c3_e000,
            })
        );
    }

    /// One byte short of the record is a refusal that names the floor, not a
    /// notice with a zero-filled tail — a length read short is an interval this
    /// device would retire the wrong pages for.
    #[test]
    fn a_map_notice_one_byte_short_is_refused_with_its_floor() {
        let bytes = vec![0u8; MAP_MEMORY_LEN];
        assert!(decode_map_memory(&bytes).is_ok());
        assert_eq!(
            decode_map_memory(&bytes[..MAP_MEMORY_LEN - 1]),
            Err(ShortPayload {
                plen: MAP_MEMORY_LEN - 1,
                need: MAP_MEMORY_LEN,
            })
        );
    }

    /// RE pageBacking: plen=16 = header + one 8-byte record; LE `01 00 00 01`
    /// = clear_host_valid + set_guest_valid (PVG validity-op bytes).
    #[test]
    fn decode_invalidate_pageon_shape() {
        let mut p = [0u8; 16];
        st32(&mut p[0..], 0); // task
        st32(&mut p[4..], 1); // count
        st32(&mut p[8..], 0x2a); // object_id
        st32(&mut p[12..], CHILD_INVALIDATE_PAGEON_FLAGS);
        let c = decode_invalidate_resources(&p).expect("decode");
        assert_eq!(c.task_id, 0);
        assert_eq!(c.count, 1);
        assert_eq!(c.records.len(), 1);
        assert_eq!(c.records[0].object_id, 0x2a);
        assert_eq!(c.records[0].flags, CHILD_INVALIDATE_PAGEON_FLAGS);
        assert_eq!(c.records[0].ops, InvalidateValidityOps::PAGEON);
        assert_eq!(c.records[0].ops.clear_host_valid, 1);
        assert_eq!(c.records[0].ops.set_host_valid, 0);
        assert_eq!(c.records[0].ops.clear_guest_valid, 0);
        assert_eq!(c.records[0].ops.set_guest_valid, 1);
        // LE memory: not bit0|bit24 as independent product bits.
        assert_eq!(CHILD_INVALIDATE_PAGEON_FLAGS.to_le_bytes(), [1, 0, 0, 1]);
    }

    #[test]
    fn invalidate_validity_ops_roundtrip() {
        let ops = InvalidateValidityOps {
            clear_host_valid: 1,
            set_host_valid: 0,
            clear_guest_valid: 1,
            set_guest_valid: 0,
        };
        assert_eq!(InvalidateValidityOps::from_le_dword(ops.to_le_dword()), ops);
    }

    /// RE synchronizeForUnwire: plen=12 = header + one u32 object_id.
    #[test]
    fn decode_synchronize_unwire_shape() {
        let mut p = [0u8; 12];
        st32(&mut p[0..], 4); // task
        st32(&mut p[4..], 1);
        st32(&mut p[8..], 99);
        let c = decode_synchronize_resources(&p).expect("decode");
        assert_eq!(c.task_id, 4);
        assert_eq!(c.count, 1);
        assert_eq!(c.object_ids, vec![99]);
    }

    #[test]
    fn decode_invalidate_multi_object_when_payload_long() {
        let mut p = [0u8; 8 + 16];
        st32(&mut p[0..], 1);
        st32(&mut p[4..], 2);
        st32(&mut p[8..], 10);
        st32(&mut p[12..], 0x1000001);
        st32(&mut p[16..], 11);
        st32(&mut p[20..], 0x1000001);
        let c = decode_invalidate_resources(&p).expect("decode");
        assert_eq!(c.count, 2);
        assert_eq!(c.records[0].object_id, 10);
        assert_eq!(c.records[1].object_id, 11);
    }

    /// The three descriptor offsets must tile the stride `exec` uses to skip
    /// the table. If they ever disagree, the decoded records and the cmdbuf
    /// section would be read from different places in the same payload.
    #[test]
    fn exec_resource_desc_offsets_tile_the_stride() {
        assert_eq!(CHILD_EXEC_RESOURCE_OBJECT_ID, 0);
        assert_eq!(CHILD_EXEC_RESOURCE_VALIDITY_OPS, 4);
        assert_eq!(
            CHILD_EXEC_RESOURCE_TAIL + CHILD_EXEC_RESOURCE_TAIL_LEN,
            CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN
        );
    }

    /// Build an EXEC_INDIRECT2 payload with `descs` resource records and no
    /// command buffers.
    fn exec_payload_with_table(descs: &[(u32, u32, [u8; 16])]) -> Vec<u8> {
        let mut p = vec![
            0u8;
            CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                + descs.len() * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize
        ];
        st32(
            &mut p[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..],
            descs.len() as u32,
        );
        for (i, (id, flags, tail)) in descs.iter().enumerate() {
            let off = CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                + i * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
            st32(&mut p[off + CHILD_EXEC_RESOURCE_OBJECT_ID as usize..], *id);
            st32(
                &mut p[off + CHILD_EXEC_RESOURCE_VALIDITY_OPS as usize..],
                *flags,
            );
            let t = off + CHILD_EXEC_RESOURCE_TAIL as usize;
            p[t..t + CHILD_EXEC_RESOURCE_TAIL_LEN as usize].copy_from_slice(tail);
        }
        p
    }

    /// RE writeInvalidates: `{object_id}` + the same validity quad an
    /// invalidate record carries, at stride 24 with 16 trailing bytes.
    #[test]
    fn decode_exec_resource_table_reads_id_quad_and_tail() {
        let mut tail = [0u8; 16];
        tail[0] = 0xaa;
        tail[15] = 0x01;
        let p =
            exec_payload_with_table(&[(0x2a, 0x0000_0001, [0u8; 16]), (0x2b, 0x0000_0100, tail)]);
        let descs = decode_exec_resource_table(&p).expect("decode");
        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].object_id, 0x2a);
        assert_eq!(descs[0].ops.clear_host_valid, 1);
        assert_eq!(descs[0].ops.set_host_valid, 0);
        assert_eq!(descs[0].tail_nonzero_bytes(), 0);
        assert_eq!(descs[1].object_id, 0x2b);
        assert_eq!(descs[1].ops.clear_host_valid, 0);
        assert_eq!(descs[1].ops.set_host_valid, 1);
        assert_eq!(descs[1].tail_nonzero_bytes(), 2);
    }

    /// The quad decoder is shared with `CmdInvalidateResources`; only the record
    /// length differs. A second decoder for the same four bytes would be a
    /// second place for the field order to drift.
    #[test]
    fn exec_table_and_invalidate_record_decode_the_same_quad() {
        let p = exec_payload_with_table(&[(7, CHILD_INVALIDATE_PAGEON_FLAGS, [0u8; 16])]);
        let descs = decode_exec_resource_table(&p).expect("decode");
        assert_eq!(descs[0].ops, InvalidateValidityOps::PAGEON);
    }

    #[test]
    fn decode_exec_resource_table_empty_when_count_zero() {
        let p = exec_payload_with_table(&[]);
        assert_eq!(decode_exec_resource_table(&p).expect("decode").len(), 0);
    }

    /// `resource_count` is guest-controlled. A count the payload cannot back
    /// must refuse, not read past the buffer and not reserve for records that
    /// are not there.
    #[test]
    fn decode_exec_resource_table_rejects_count_the_payload_cannot_back() {
        let mut p = exec_payload_with_table(&[(1, 0, [0u8; 16])]);
        st32(&mut p[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..], 2);
        assert!(decode_exec_resource_table(&p).is_err());
        st32(
            &mut p[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..],
            u32::MAX,
        );
        assert!(decode_exec_resource_table(&p).is_err());
    }

    #[test]
    fn decode_exec_resource_table_rejects_short_header() {
        assert!(decode_exec_resource_table(&[0u8; 4]).is_err());
    }

    #[test]
    fn decode_invalidate_rejects_short_for_count() {
        let mut p = [0u8; 12]; // header claims count=2 but only 4 payload bytes
        st32(&mut p[0..], 0);
        st32(&mut p[4..], 2);
        st32(&mut p[8..], 1);
        assert_eq!(
            decode_invalidate_resources(&p),
            Err(ResourceListDecodeError::Truncated {
                count: 2,
                plen: 12,
                need: 24,
            }),
            "a count the payload cannot carry must name itself, not share a \
             verdict with a short header"
        );
        assert_eq!(
            decode_invalidate_resources(&[0u8; 4]),
            Err(ResourceListDecodeError::ShortHeader { plen: 4 })
        );
    }

    /// A list longer than the retired `CHILD_RESOURCE_LIST_MAX_COUNT` decodes.
    ///
    /// That cap was 256, it sat above the length check, and it bounded nothing
    /// the length check did not already bound — so its only effect was to refuse
    /// a well-formed list. Refusing an Invalidate leaves this device serving
    /// host-cached pixels for a resource the guest has just CPU-written, so the
    /// cost of that refusal was not a lost log line.
    ///
    /// 257 rather than a round number: one past the retired bound, so a
    /// reinstated cap of any plausible size fails here.
    #[test]
    fn a_list_longer_than_the_retired_cap_decodes_when_the_payload_carries_it() {
        const COUNT: u32 = 257;
        let mut p = vec![
            0u8;
            CHILD_RESOURCE_LIST_HEADER_LEN as usize
                + COUNT as usize * CHILD_INVALIDATE_RECORD_LEN as usize
        ];
        st32(&mut p[0..], 3);
        st32(&mut p[4..], COUNT);
        for i in 0..COUNT as usize {
            let at =
                CHILD_RESOURCE_LIST_HEADER_LEN as usize + i * CHILD_INVALIDATE_RECORD_LEN as usize;
            st32(&mut p[at..], 0x100 + i as u32);
            st32(&mut p[at + 4..], CHILD_INVALIDATE_PAGEON_FLAGS);
        }
        let c = decode_invalidate_resources(&p).expect("the payload carries every record");
        assert_eq!(c.task_id, 3);
        assert_eq!(c.records.len(), COUNT as usize);
        assert_eq!(c.records[256].object_id, 0x100 + 256);

        // Same for the sibling command, whose records are 4 bytes rather than 8
        // — the one thing that differs between the two decoders.
        let mut s = vec![
            0u8;
            CHILD_RESOURCE_LIST_HEADER_LEN as usize
                + COUNT as usize * CHILD_SYNCHRONIZE_RECORD_LEN as usize
        ];
        st32(&mut s[4..], COUNT);
        let c = decode_synchronize_resources(&s).expect("the payload carries every record");
        assert_eq!(c.object_ids.len(), COUNT as usize);
    }

    /// The length a count needs cannot wrap into a small number.
    ///
    /// `u32::MAX` records of 8 bytes is 32 GiB, which no payload carries — the
    /// point is that it stays impossible. Computed in `u64` and saturated into
    /// `usize`, so a 32-bit host cannot truncate `need` into something a short
    /// payload satisfies, which would turn a bound into an out-of-range read.
    #[test]
    fn a_count_that_cannot_fit_reports_truncated_rather_than_wrapping() {
        let mut p = [0u8; 16];
        st32(&mut p[4..], u32::MAX);
        match decode_invalidate_resources(&p) {
            Err(ResourceListDecodeError::Truncated { count, plen, need }) => {
                assert_eq!((count, plen), (u32::MAX, 16));
                assert!(need > plen as u64, "need={need} must exceed the payload");
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    /// The two `{u32, u32}` records disagree about which word is which, and
    /// the same eight bytes decode to swapped fields.
    ///
    /// Nothing about the bytes can tell them apart — same length, both fields
    /// plain `u32`s — so this is what the separation buys: a reader that picked
    /// the wrong decode retires the backing of whatever mapping shares the
    /// task's number.
    #[test]
    fn the_backing_retirement_reverses_the_task_object_pair() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x1111_1111u32.to_le_bytes());
        bytes.extend_from_slice(&0x2222_2222u32.to_le_bytes());
        let pair = decode_task_object(&bytes).expect("eight bytes");
        let retire = decode_delete_backing(&bytes).expect("eight bytes");
        assert_eq!(
            (pair.task_id, pair.object_id),
            (retire.object_id, retire.task_id),
            "one record's task is the other's object"
        );
        assert_eq!(
            DELETE_BACKING_LEN, TASK_OBJECT_LEN,
            "and nothing else tells them apart"
        );
        assert_eq!(
            decode_delete_backing(&bytes[..DELETE_BACKING_LEN - 1]),
            Err(ShortPayload {
                plen: DELETE_BACKING_LEN - 1,
                need: DELETE_BACKING_LEN,
            })
        );
    }

    /// A destroy command's two bounds, and the fact that they are two.
    ///
    /// A payload too short for a record header and a record whose length runs
    /// past the payload are different guest defects, and only the second says
    /// anything about the record.
    #[test]
    fn a_destroy_command_is_framed_by_the_records_own_length() {
        // Task 1, a twelve-byte destroy record: `{opcode, len}` then the ref.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0x1234u32.to_le_bytes());
        bytes.extend_from_slice(&12u32.to_le_bytes());
        bytes.extend_from_slice(&32u32.to_le_bytes());
        let framed = decode_delete_object(&bytes).expect("the record fits");
        assert_eq!(framed.task_id, 1);
        assert_eq!(
            framed.record.len(),
            12,
            "the record is the payload past the task word"
        );

        // A record claiming one byte more than the payload holds.
        let mut over = bytes.clone();
        over[DELETE_OBJECT_RECORD_LEN..DELETE_OBJECT_RECORD_LEN + 4]
            .copy_from_slice(&13u32.to_le_bytes());
        let err = decode_delete_object(&over).expect_err("the record does not fit");
        assert_eq!(err.slug(), "delete_object_record");
        assert_eq!(err.short().need, 13 + DELETE_OBJECT_RECORD);

        // A length that would overflow the sum is refused, not wrapped.
        let mut huge = bytes.clone();
        huge[DELETE_OBJECT_RECORD_LEN..DELETE_OBJECT_RECORD_LEN + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode_delete_object(&huge),
            Err(DeleteObjectError::RecordTruncated(_))
        ));

        // And too short to hold a header at all.
        let err =
            decode_delete_object(&bytes[..DELETE_OBJECT_LEN - 1]).expect_err("no record header");
        assert_eq!(err.slug(), "delete_object");
        assert_eq!(
            err.short(),
            ShortPayload {
                plen: DELETE_OBJECT_LEN - 1,
                need: DELETE_OBJECT_LEN,
            }
        );
    }

    /// The task comes first, and the two commands carrying this record agree.
    ///
    /// Its two words are adjacent `u32`s, so either reading looks plausible: a
    /// decode that swapped them would re-point the wrong resource on
    /// `CmdReplacePhysical` and delete the wrong one on `CmdDeleteResource`.
    /// The third command with these two words carries them in the *opposite*
    /// order and deliberately does not reach here — see
    /// [`TASK_OBJECT_TASK_ID`].
    #[test]
    fn a_task_object_record_reads_its_task_before_its_object() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0x1122_3344u32.to_le_bytes());
        payload.extend_from_slice(&0x5566_7788u32.to_le_bytes());
        assert_eq!(
            decode_task_object(&payload),
            Ok(TaskObjectCommand {
                task_id: 0x1122_3344,
                object_id: 0x5566_7788,
            })
        );
        // One byte short is not a replace. Refused rather than clamped: acting
        // on object id zero re-points whatever holds slot zero.
        for plen in 0..TASK_OBJECT_LEN {
            assert_eq!(
                decode_task_object(&payload[..plen]),
                Err(ShortPayload {
                    plen,
                    need: TASK_OBJECT_LEN
                }),
                "{plen}"
            );
        }
        // And a longer payload decodes the same two words rather than refusing;
        // the length is the guest's and a trailer this device does not name is
        // not a malformed packet.
        payload.extend_from_slice(&[0xff; 16]);
        assert_eq!(
            decode_task_object(&payload).map(|c| c.object_id),
            Ok(0x5566_7788)
        );
    }

    /// A synchronize payload too short to hold its header is `ShortHeader` and
    /// not `Truncated`, because the count could not be read at all — and the
    /// two refusals are what a reader uses to tell "the guest sent a stub" from
    /// "the guest declared more than it sent".
    #[test]
    fn a_synchronize_too_short_for_its_header_cannot_report_a_count() {
        for plen in 0..CHILD_RESOURCE_LIST_HEADER_LEN as usize {
            let payload = alloc::vec![0u8; plen];
            assert_eq!(
                decode_synchronize_resources(&payload),
                Err(ResourceListDecodeError::ShortHeader { plen })
            );
            assert_eq!(
                decode_invalidate_resources(&payload),
                Err(ResourceListDecodeError::ShortHeader { plen })
            );
        }
    }

    /// Every length in a FIFO payload is the guest's, including the counts these
    /// decoders read *out of* the payload and then index with.
    ///
    /// The bounds are written down — `resource_list_need` saturates, the exec
    /// table checks before it reserves — and the cases someone thought of are
    /// tested above. This is the case nobody thought of: arbitrary bytes at
    /// arbitrary lengths, including the lengths right at each decoder's
    /// threshold, where an off-by-one lives.
    ///
    /// The claim is not that these payloads decode. It is that every one of them
    /// either refuses or produces a value that accounts for the bytes it claims:
    /// a list whose record count matches its declared count, and a declared
    /// count the payload could actually carry.
    #[test]
    fn no_payload_of_any_shape_panics_or_claims_bytes_it_was_not_given() {
        // xorshift64*, so the sweep is reproducible and carries no dependency.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        // Per decoder, not one aggregate. Three of the four decoders below
        // shared no counter at all, so the floors at the end were met by
        // `decode_invalidate_resources` alone and a sibling that refused every
        // one of the 4096 payloads would still have read as covered.
        let (mut decoded, mut refused) = (0usize, 0usize);
        let (mut sync_decoded, mut sync_refused) = (0usize, 0usize);
        let (mut table_decoded, mut table_refused) = (0usize, 0usize);
        for _ in 0..4096 {
            // Lengths clustered on the thresholds, where an off-by-one lives:
            // the two header lengths, the three record strides, and the exec
            // header, plus a few bytes either side of each.
            let base =
                [0usize, 4, 8, 12, 16, 24, 32, 36][usize::try_from(next() % 8).expect("small")];
            let jitter = usize::try_from(next() % 5).expect("small");
            let len = base + jitter;
            let mut payload = Vec::with_capacity(len);
            for _ in 0..len {
                payload.push(u8::try_from(next() & 0xff).expect("masked"));
            }
            // Uniformly random bytes make every declared count astronomical, so
            // every payload refuses and the sweep proves only that refusing does
            // not panic. Half of them get a count a payload of this size could
            // plausibly carry, which is where the indexing actually happens.
            if next() % 2 == 0 && len >= 8 {
                let count = u32::try_from(next() % 5).expect("small");
                payload[4..8].copy_from_slice(&count.to_le_bytes());
            }

            match decode_invalidate_resources(&payload) {
                Ok(cmd) => {
                    decoded += 1;
                    assert_eq!(cmd.records.len() as u64, u64::from(cmd.count));
                    assert!(
                        resource_list_need(cmd.count, CHILD_INVALIDATE_RECORD_LEN) <= len as u64
                    );
                }
                Err(_) => refused += 1,
            }
            match decode_synchronize_resources(&payload) {
                Ok(cmd) => {
                    sync_decoded += 1;
                    assert_eq!(cmd.object_ids.len() as u64, u64::from(cmd.count));
                    assert!(
                        resource_list_need(cmd.count, CHILD_SYNCHRONIZE_RECORD_LEN) <= len as u64
                    );
                }
                Err(_) => sync_refused += 1,
            }
            match decode_exec_resource_table(&payload) {
                Ok(descs) => {
                    table_decoded += 1;
                    let count = ld32(&payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..]);
                    assert_eq!(descs.len() as u64, u64::from(count));
                    assert!(
                        CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                            + descs.len() * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize
                            <= len
                    );
                }
                Err(_) => table_refused += 1,
            }
            assert_eq!(decode_task_object(&payload).is_ok(), len >= TASK_OBJECT_LEN);
            // No length in this sweep can carry a well-formed record, so what
            // this pins is the refusal path: every rejection is a value and
            // none of the four checks indexes past what it was handed.
            if let Ok(request) = decode_heap_texture_query(&payload) {
                assert_eq!(
                    len,
                    HEAP_TEXTURE_REQUEST_HEADER_LEN + HEAP_TEXTURE_SERIALIZED_LEN
                );
                assert_eq!(
                    request.descriptor.len(),
                    HEAP_TEXTURE_SERIALIZED_LEN - reims_vgpu_wire::op::OP_HEADER_LEN,
                );
            }
        }
        // A sweep where everything refuses proves only that refusing does not
        // panic, and one where nothing does proves only the happy path.
        for (name, count) in [
            ("invalidate decoded", decoded),
            ("invalidate refused", refused),
            ("synchronize decoded", sync_decoded),
            ("synchronize refused", sync_refused),
            ("resource table decoded", table_decoded),
            ("resource table refused", table_refused),
        ] {
            assert!(count > 100, "only {count} of 4096 payloads: {name}");
        }
    }

    /// The two forms do not share offsets, and reading one at the other's is
    /// not a near miss: the ceiling becomes the count and the count becomes the
    /// page frame the reply is written to.
    #[test]
    fn the_two_device_info_forms_read_different_words() {
        let mut bytes = Vec::new();
        for word in [18u32, 512, 0xF00D, 0xDEAD] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        assert_eq!(
            decode_device_info(DeviceInfoForm::WithKeyLimit, &bytes),
            Ok(DeviceInfoRequest {
                form: DeviceInfoForm::WithKeyLimit,
                key_table_len: Some(18),
                pair_capacity: 512,
                reply_pfn: 0xF00D,
            })
        );
        assert_eq!(
            decode_device_info(DeviceInfoForm::WithoutKeyLimit, &bytes),
            Ok(DeviceInfoRequest {
                form: DeviceInfoForm::WithoutKeyLimit,
                key_table_len: None,
                pair_capacity: 18,
                reply_pfn: 512,
            }),
            "the older form has no ceiling, so its count is word zero and its \
             reply frame is word one"
        );
    }

    /// The older form's ceiling is absent, not zero. Zero would be a walker
    /// that parses no keys at all, which is a different statement from a form
    /// that carries no ceiling.
    #[test]
    fn the_older_form_reports_no_key_limit_rather_than_a_zero_one() {
        let bytes = alloc::vec![0xffu8; 16];
        assert_eq!(
            decode_device_info(DeviceInfoForm::WithoutKeyLimit, &bytes)
                .expect("long enough")
                .key_table_len,
            None
        );
        assert_eq!(
            decode_device_info(DeviceInfoForm::WithKeyLimit, &bytes)
                .expect("long enough")
                .key_table_len,
            Some(0xffff_ffff)
        );
    }

    /// Each form refuses at its own floor, one byte under.
    #[test]
    fn a_device_info_request_under_its_forms_length_is_refused() {
        for form in [
            DeviceInfoForm::WithoutKeyLimit,
            DeviceInfoForm::WithKeyLimit,
        ] {
            let need = form.request_len();
            let bytes = alloc::vec![0u8; need];
            assert!(decode_device_info(form, &bytes).is_ok());
            assert_eq!(
                decode_device_info(form, &bytes[..need - 1]),
                Err(ShortPayload {
                    plen: need - 1,
                    need
                })
            );
        }
        assert_eq!(DeviceInfoForm::WithoutKeyLimit.request_len(), 8);
        assert_eq!(DeviceInfoForm::WithKeyLimit.request_len(), 12);
        // Every word a form names is inside the request it declares, and no
        // two of them share a slot.
        for form in [
            DeviceInfoForm::WithoutKeyLimit,
            DeviceInfoForm::WithKeyLimit,
        ] {
            assert!(form.reply_pfn_offset() + 4 <= form.request_len());
            assert_ne!(form.pair_capacity_offset(), form.reply_pfn_offset());
            if let Some(at) = form.key_table_len_offset() {
                assert_ne!(at, form.pair_capacity_offset());
                assert_ne!(at, form.reply_pfn_offset());
            }
        }
    }

    /// Every word of a compute-info request comes out of its own slot, and the
    /// reply destination is a full 64-bit address rather than the low half of
    /// one — the guest puts a `u64` there and a decoder that read four bytes
    /// would answer into the bottom 4 GiB of wherever it pointed.
    #[test]
    fn a_compute_info_request_reads_six_words_and_a_full_address() {
        let mut bytes = Vec::new();
        for word in [7u32, 9, 5, 512] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        assert_eq!(
            decode_compute_info(&bytes),
            Ok(ComputeInfoRequest {
                raw_task: 7,
                pipeline_ref: 9,
                key_table_len: 5,
                pair_capacity: 512,
                reply_gva: 0x1234_5678_9abc_def0,
            })
        );
    }

    /// One byte under is refused, at exactly the length the request declares.
    #[test]
    fn a_compute_info_request_under_its_length_is_refused() {
        let bytes = alloc::vec![0u8; COMPUTE_INFO_REQUEST_LEN];
        assert!(decode_compute_info(&bytes).is_ok());
        for plen in 0..COMPUTE_INFO_REQUEST_LEN {
            assert_eq!(
                decode_compute_info(&bytes[..plen]),
                Err(ShortPayload {
                    plen,
                    need: COMPUTE_INFO_REQUEST_LEN
                }),
                "{plen}"
            );
        }
    }

    /// A live `CmdHeapTextureSizeAndAlign` request, framed.
    ///
    /// The words are the capture's: task 1, a reply at `0x162200` sixteen bytes
    /// long, then a forty-byte `heapTextureSizeAndAlignWithDescriptor:` record.
    fn heap_texture_request() -> Vec<u8> {
        let words = [
            0x1u32, 0x162200, 0x0, 0x10, 0x0, 0x28, 0x16, 0x28, 0x7d0342, 0xb4, 0x87, 0x1, 0x10001,
            0x200001, 0x0, 0x0,
        ];
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    /// The header's four fields, and the record it hands back.
    ///
    /// The descriptor is the record's *payload*, not the record: a caller that
    /// received the head as well would read the opcode as a texture type.
    #[test]
    fn a_heap_texture_request_frames_its_words_and_its_record() {
        let bytes = heap_texture_request();
        let request = decode_heap_texture_query(&bytes).expect("framed");
        assert_eq!(request.raw_task, 1);
        assert_eq!(request.reply_gva, 0x162200);
        assert_eq!(request.reply_len, HEAP_TEXTURE_REPLY_LEN as u64);
        assert_eq!(
            request.descriptor.len(),
            HEAP_TEXTURE_SERIALIZED_LEN - reims_vgpu_wire::op::OP_HEADER_LEN,
        );
        assert_eq!(
            request.descriptor,
            &bytes[HEAP_TEXTURE_REQUEST_HEADER_LEN + reims_vgpu_wire::op::OP_HEADER_LEN..],
        );
    }

    /// Every offset is where the one before it ends, and the reply is two
    /// `u64`s.
    #[test]
    fn the_request_header_and_the_reply_are_each_their_fields() {
        assert_eq!(HEAP_TEXTURE_REQUEST_HEADER_LEN, 24);
        assert_eq!(HEAP_TEXTURE_REPLY_LEN, 16);
        assert_eq!(HEAP_TEXTURE_SERIALIZED_LEN, 40);
        let reply = SizeAndAlign {
            size: 0x78000,
            align: 0x80,
        }
        .encode();
        assert_eq!(ld64(&reply[HEAP_TEXTURE_REPLY_SIZE..]), 0x78000);
        assert_eq!(ld64(&reply[HEAP_TEXTURE_REPLY_ALIGN..]), 0x80);
    }

    /// A reply window that cannot hold both words is refused, not clamped.
    ///
    /// Writing the size and dropping the alignment would leave the guest
    /// placing a heap texture at whatever alignment its buffer already held.
    #[test]
    fn a_reply_window_under_an_mtl_size_and_align_is_refused() {
        for len in 0..HEAP_TEXTURE_REPLY_LEN as u64 {
            let mut bytes = heap_texture_request();
            st64(&mut bytes[HEAP_TEXTURE_REPLY_LENGTH..], len);
            assert_eq!(
                decode_heap_texture_query(&bytes),
                Err(HeapTextureRefusal::ReplyDestination { gva: 0x162200, len }),
                "{len}"
            );
        }
    }

    /// A null destination is refused however long the window claims to be.
    #[test]
    fn a_reply_address_of_zero_is_nowhere_to_write() {
        let mut bytes = heap_texture_request();
        st64(&mut bytes[HEAP_TEXTURE_REPLY_GVA..], 0);
        st64(&mut bytes[HEAP_TEXTURE_REPLY_LENGTH..], 4096);
        assert_eq!(
            decode_heap_texture_query(&bytes),
            Err(HeapTextureRefusal::ReplyDestination { gva: 0, len: 4096 }),
        );
    }

    /// The declared record length has to be this opcode's length *and* the rest
    /// of the payload — one refusal, because it is one claim.
    #[test]
    fn a_declared_record_length_must_be_the_opcode_s_and_the_payload_s() {
        let bytes = heap_texture_request();
        for declared in [
            HEAP_TEXTURE_SERIALIZED_LEN as u32 - 1,
            HEAP_TEXTURE_SERIALIZED_LEN as u32 + 1,
        ] {
            let mut short = bytes.clone();
            st32(&mut short[HEAP_TEXTURE_SERIALIZER_LENGTH..], declared);
            assert_eq!(
                decode_heap_texture_query(&short),
                Err(HeapTextureRefusal::SerializerLength {
                    declared,
                    plen: bytes.len(),
                }),
                "{declared}"
            );
        }
        // The declared length is right and the bytes are not — in both
        // directions. Short is also what the record view refuses, since a
        // record cannot declare more than it was handed; **long is not**, and
        // it is the half that matters: a request with a tail decodes to a
        // record whose trailing bytes belong to whatever follows it.
        for plen in [bytes.len() - 4, bytes.len() + 4] {
            let mut resized = bytes.clone();
            resized.resize(plen, 0);
            assert_eq!(
                decode_heap_texture_query(&resized),
                Err(HeapTextureRefusal::SerializerLength {
                    declared: HEAP_TEXTURE_SERIALIZED_LEN as u32,
                    plen,
                }),
                "{plen}"
            );
        }
    }

    /// A record of another selector is a routing mistake, and is named as one.
    #[test]
    fn a_record_of_another_selector_is_refused_by_tag() {
        let mut bytes = heap_texture_request();
        st32(&mut bytes[HEAP_TEXTURE_REQUEST_HEADER_LEN..], 0x99);
        assert_eq!(
            decode_heap_texture_query(&bytes),
            Err(HeapTextureRefusal::SerializerTag { found: 0x99 }),
        );
    }

    /// One byte under the header is refused, at exactly the length it declares.
    #[test]
    fn a_heap_texture_request_under_its_header_is_refused() {
        let bytes = heap_texture_request();
        for plen in 0..HEAP_TEXTURE_REQUEST_HEADER_LEN {
            assert_eq!(
                decode_heap_texture_query(&bytes[..plen]),
                Err(HeapTextureRefusal::Short(ShortPayload {
                    plen,
                    need: HEAP_TEXTURE_REQUEST_HEADER_LEN,
                })),
                "{plen}"
            );
        }
    }
}
