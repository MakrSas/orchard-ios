//! The refusal-closure ledger: one recorded outcome per decodable operation.
//!
//! # What it is for
//!
//! Before the execution architecture can be replaced, the replacement's
//! vocabulary has to be exhaustive, and "exhaustive" has to be a measurement
//! rather than a claim. The claim is easy to make and impossible to check: every
//! rail in this project can point at an arm for the records it has seen, and
//! none of them could say what happens to the records it has not.
//!
//! So this is the measurement. Every operation the wire layer can decode has
//! exactly one row here, and every row carries exactly one of four outcomes:
//!
//! * [`Closure::Implemented`] — the device performs the semantic operation.
//! * [`Closure::ProvenNoOp`] — the operation is a no-op **on a named capability
//!   cell**, and the row names the cell. A no-op that is only true because guest
//!   pages are the single copy of resource content stops being true on a
//!   topology where they are not, and the cell is what makes that legible.
//! * [`Closure::Refused`] — the operation is unsupported and the device says so
//!   by name, on the always-on failure channel, every time.
//! * [`Closure::Unresolved`] — the outcome is not established. This blocks the
//!   cutover of that operation and nothing else.
//!
//! Two things are deliberately unspellable. "The current workload does not
//! issue it" is not an outcome, because a workload is not a contract. "The old
//! backend drops it too" is not an outcome, because the old backend is what is
//! being replaced. Anything that would have been recorded as either is
//! [`Closure::Unresolved`], which is honest and which counts.
//!
//! # Where the row set comes from
//!
//! [`reims_vgpu_wire::manifest::MANIFEST`] enumerates the serializer's selectors
//! from the runtime rather than from anyone's list, so the row set is not
//! curated either: its test module requires one row for every `(class, opcode)` the
//! manifest records and one for every selector it covers without a fixed opcode.
//! When the manifest grows a selector, this ledger stops compiling its tests
//! until someone records what the device does about it.
//!
//! The converse does not hold, and [`OFF_MANIFEST`] is why. `class_copyMethodList`
//! reports the methods a class declares itself and does not walk superclasses,
//! and the encoders share one. A selector declared only on that base class is
//! callable on every encoder while being invisible to the manifest, so the
//! ledger carries rows the manifest has no counterpart for — and names them,
//! rather than letting "absent from the manifest" quietly mean "does not exist".
//!
//! # What a reader should do with it
//!
//! [`counts`] is the cutover gate. The replacement's ingress may switch for an
//! operation class only when no operation in it is [`Closure::Unresolved`];
//! [`blocking`] is that list.

/// The serializer ordering domain an operation belongs to.
///
/// One rail is one submission-ordering domain on the wire, which is why this is
/// not simply "the class the selector is declared on": the root rail's object
/// lifecycle records arrive outside any encoder segment, and the four encoder
/// rails each frame their own records. Two rails may number the same opcode
/// differently — `0x18` is a render fence update and nothing at all on the blit
/// rail, whose fence pair is `0x13c`/`0x13d` — so an opcode alone is never a
/// key here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rail {
    /// Object creation and destruction, outside any encoder segment.
    Root,
    Render,
    Compute,
    Blit,
    Info,
    /// The event/synchronisation encoder. It has no manifest class — see
    /// [`OFF_MANIFEST`] — so [`Rail::class`] has no name to return for it.
    Event,
}

impl Rail {
    /// The serializer class this rail's operations are declared on, spelled as
    /// [`reims_vgpu_wire::manifest::Entry::class`] spells it.
    pub const fn class(self) -> &'static str {
        match self {
            Self::Root => "PGSerializer",
            Self::Render => "PGSerializerRenderCommandEncoder",
            Self::Compute => "PGSerializerComputeCommandEncoder",
            Self::Blit => "PGSerializerBlitCommandEncoder",
            Self::Info => "PGSerializerInfoCommandEncoder",
            // No capture has driven this encoder, so the manifest has no class
            // for it and `from_class` can never answer with it. The empty
            // string is not a name it might collide with.
            Self::Event => "",
        }
    }

    /// Every rail, so a caller enumerating the ledger by domain cannot miss one.
    pub const ALL: &'static [Rail] = &[
        Rail::Root,
        Rail::Render,
        Rail::Compute,
        Rail::Blit,
        Rail::Info,
        Rail::Event,
    ];

    /// The rail a manifest class belongs to.
    pub fn from_class(class: &str) -> Option<Self> {
        if class.is_empty() {
            return None;
        }
        Self::ALL.iter().copied().find(|r| r.class() == class)
    }
}

/// What the device does about one decodable operation.
///
/// Every variant carries prose because a bare discriminant is the failure this
/// ledger exists to prevent: `ProvenNoOp` with nothing beside it is
/// indistinguishable from a guess, and the guess is the common case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Closure {
    /// The device performs the semantic operation.
    Implemented {
        /// What performs it, and at what granularity.
        evidence: &'static str,
    },
    /// The operation is a no-op, and the capability cell that makes it one.
    ProvenNoOp {
        /// The host/topology property the proof rests on. When the cell stops
        /// holding, so does the row.
        cell: &'static str,
        /// Why the operation asks for nothing on that cell.
        evidence: &'static str,
    },
    /// The operation is unsupported and the device refuses it by name.
    Refused {
        /// The census route or decline slug the refusal lands on.
        route: &'static str,
        /// What the contract says, and why refusing is the exact behavior.
        evidence: &'static str,
    },
    /// The outcome is not established. Blocks this operation's cutover.
    Unresolved {
        /// What has to be learned, stated as the thing that is not known.
        question: &'static str,
    },
}

impl Closure {
    /// Whether this outcome blocks the replacement's cutover for its operation.
    pub const fn blocks_cutover(&self) -> bool {
        matches!(self, Self::Unresolved { .. })
    }

    /// A stable one-word name, for a census line or a report.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Implemented { .. } => "implemented",
            Self::ProvenNoOp { .. } => "proven_noop",
            Self::Refused { .. } => "refused",
            Self::Unresolved { .. } => "unresolved",
        }
    }
}

/// One decodable operation and its recorded outcome.
#[derive(Clone, Copy, Debug)]
pub struct Op {
    pub rail: Rail,
    /// The record's opcode, or `None` when the record carries no fixed one —
    /// either because the opcode field is guest data (the `withCommand:`
    /// selectors write their argument into it) or because the record has no
    /// opcode field at all (segment framing).
    pub opcode: Option<u32>,
    /// The serializer selector(s) that emit this record, `; `-joined when more
    /// than one does.
    pub selector: &'static str,
    pub closure: Closure,
}

/// Operations the ledger carries that the wire manifest has no row for, and
/// why each one is real anyway.
///
/// The manifest enumerates selectors from the Objective-C runtime, which is
/// what makes the row set a measurement rather than a curated list — so an
/// exception has to be named one at a time or "absent from the manifest" stops
/// meaning "does not exist". Two things put an operation here:
///
/// * **Inherited from the encoder base class.** `class_copyMethodList` reports
///   the methods a class declares itself and does not walk superclasses, and
///   the encoders share one. The residency pair is the family that taught this:
///   the `stages:`-qualified overrides are declared on the render encoder and
///   visible, while the unqualified forms are inherited and invisible — so
///   reading "the compute encoder does not declare residency" as "a compute
///   encoder cannot receive a residency call" is an inference the hole invites,
///   and this project already drew it once.
/// * **No capture has driven the encoder.** The manifest names what Apple's
///   serializer was *observed* to emit, so an encoder class no oracle case has
///   exercised has no rows at all — which is a hole the size of a whole rail,
///   not of a selector.
pub const OFF_MANIFEST: &[(Rail, u32, &str)] = &[
    (
        Rail::Render,
        0x0086,
        "useHeaps:count: is declared on the shared encoder base class",
    ),
    (
        Rail::Render,
        0x0087,
        "useResources:count:usage: is declared on the shared encoder base class",
    ),
    (
        Rail::Compute,
        0x0086,
        "useHeaps:count: is declared on the shared encoder base class",
    ),
    (
        Rail::Compute,
        0x0087,
        "useResources:count:usage: is declared on the shared encoder base class",
    ),
    (
        Rail::Event,
        0x0190,
        "the event encoder has no manifest class: no oracle case has driven it",
    ),
    (
        Rail::Event,
        0x0191,
        "the event encoder has no manifest class: no oracle case has driven it",
    ),
    (
        Rail::Event,
        0x0192,
        "the event encoder has no manifest class: no oracle case has driven it",
    ),
];
/// How many operations sit at each outcome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub implemented: usize,
    pub proven_noop: usize,
    pub refused: usize,
    pub unresolved: usize,
}

impl Counts {
    pub const fn total(&self) -> usize {
        self.implemented + self.proven_noop + self.refused + self.unresolved
    }
}

/// Tally the whole ledger, or one rail's part of it.
pub fn counts(rail: Option<Rail>) -> Counts {
    let mut c = Counts::default();
    for op in LEDGER.iter().filter(|o| rail.is_none_or(|r| o.rail == r)) {
        match op.closure {
            Closure::Implemented { .. } => c.implemented += 1,
            Closure::ProvenNoOp { .. } => c.proven_noop += 1,
            Closure::Refused { .. } => c.refused += 1,
            Closure::Unresolved { .. } => c.unresolved += 1,
        }
    }
    c
}

/// Every operation whose outcome is not established.
pub fn blocking() -> impl Iterator<Item = &'static Op> {
    LEDGER.iter().filter(|o| o.closure.blocks_cutover())
}

/// The row for one operation, if the ledger has one.
pub fn find(rail: Rail, opcode: u32) -> Option<&'static Op> {
    LEDGER
        .iter()
        .find(|o| o.rail == rail && o.opcode == Some(opcode))
}

/// The ledger. One row per decodable operation; see the module docs for what
/// the four outcomes mean and what may not be spelled as one.
pub const LEDGER: &[Op] = &[
    Op {
        rail: Rail::Root,
        opcode: Some(0x0001),
        selector: "newTextureWithDescriptor:allocator:",
        closure: Closure::Implemented {
            evidence: "object-list texture creation; the wide form at 0x34 is the same descriptor at 64-bit extents",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0003),
        selector: "newSamplerStateWithDescriptor:allocator:",
        closure: Closure::Implemented {
            evidence: "sampler descriptor decoded and retained per task; retired by 0x3eb",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0004),
        selector: "newDepthStencilStateWithDescriptor:allocator:",
        closure: Closure::Implemented {
            evidence: "depth/stencil descriptor decoded and retained per task; applied per draw; retired by 0x3ea",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0007),
        selector: "newTextureViewWithPixelFormat:baseTexture:allocator:",
        closure: Closure::Implemented {
            evidence: "texture view over a base texture; the view's format and type are carried to the backend",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0008),
        selector: "newTextureViewWithPixelFormat:textureType:levels:slices:baseTexture:allocator:",
        closure: Closure::Implemented {
            evidence: "level/slice-ranged texture view; the range is carried as a subresource window",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0009),
        selector: "newTextureWithBuffer:descriptor:offset:bytesPerRow:allocator:",
        closure: Closure::Implemented {
            evidence: "buffer-backed texture; the buffer's rows and offset are the texture's backing",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x000c),
        selector: "newIOSurfaceTextureWithDescriptor:plane:allocator:",
        closure: Closure::Implemented {
            evidence: "IOSurface-backed texture, including the biplanar split, through the surface-page contract",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x000d),
        selector: "newFenceWithAllocator:",
        closure: Closure::Unresolved {
            question: "fence-object creation carries no host lifetime here: fence update/wait are executed against the ref alone, so a create the device never sees cannot refuse an unknown ref",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0015),
        selector: "newTextureWithDescriptor:heap:offset:useOffset:allocator:",
        closure: Closure::Implemented {
            evidence: "heap-backed texture at an offset the guest supplies",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0016),
        selector: "heapTextureSizeAndAlignWithDescriptor:allocator:",
        closure: Closure::Implemented {
            evidence: "heap texture size/align query, answered from the same descriptor arithmetic the allocation uses",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x001b),
        selector: "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:baseTexture:allocator:",
        closure: Closure::Implemented {
            evidence: "swizzled texture view; the component swizzle is carried to the backend",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0032),
        selector: "newRasterizationRateMapWithDescriptor:allocator:; resetRasterizationRateMapWithDescriptor:existingID:allocator:",
        closure: Closure::Unresolved {
            question: "rasterization rate maps are not represented; a created map has no host object and later references cannot be resolved or refused by name",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0034),
        selector: "newTextureWithDescriptor:allocator:",
        closure: Closure::Implemented {
            evidence: "wide form of 0x01, same descriptor at 64-bit extents",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0036),
        selector: "newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:options:allocator:",
        closure: Closure::Implemented {
            evidence: "indirect command buffer creation with its command layout",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0037),
        selector: "newTextureWithBuffer:descriptor:offset:bytesPerRow:allocator:",
        closure: Closure::Implemented {
            evidence: "wide form of 0x09",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0038),
        selector: "newTextureWithDescriptor:heap:offset:useOffset:allocator:",
        closure: Closure::Implemented {
            evidence: "wide form of 0x15",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x0039),
        selector: "newIOSurfaceTextureWithDescriptor:plane:allocator:",
        closure: Closure::Implemented {
            evidence: "wide form of 0x0c",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03e8),
        selector: "deleteBufferRef:allocator:",
        closure: Closure::Unresolved {
            question: "buffer destroy has no per-kind registry to retire; the ref is in the serializer's space, not the object list's, so the record is counted and named rather than applied",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03e9),
        selector: "deleteTextureRef:allocator:",
        closure: Closure::Unresolved {
            question: "texture destroy has no per-kind registry to retire (see 0x3e8)",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03ea),
        selector: "deleteDepthStencilStateRef:allocator:",
        closure: Closure::Implemented {
            evidence: "retires the task-local depth/stencil-state registry entry",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03eb),
        selector: "deleteSamplerStateRef:allocator:",
        closure: Closure::Implemented {
            evidence: "retires the task-local sampler-state registry entry",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03ed),
        selector: "deleteFunctionRef:allocator:",
        closure: Closure::ProvenNoOp {
            cell: "this device holds functions by the content of their shader bytes and not by \
                   guest ref, so there is no per-ref registry a retirement could name",
            evidence: "measured rather than assumed, and the measurement is the type pair \
                       rather than an integer resolving. A driven macos-15 boot sent 5 of these; \
                       every one named no entry at all in the guest's own object list \
                       (delete_object_ref_no_list_entry, with type_agrees and type_differs both \
                       silent), so the ref is in the serializer's per-kind space and nothing \
                       this device keys by it exists. Retiring nothing is the operation, not a \
                       gap in it",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03ee),
        selector: "deleteComputePipelineStateRef:allocator:",
        closure: Closure::ProvenNoOp {
            cell: "compute pipeline states are keyed by the function and constants they were \
                   built from, not by guest ref (see 0x3ed)",
            evidence: "the same boot's 3 compute-pipeline destroys, measured the same way and \
                       with the same answer: no object-list entry at either ref",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03ef),
        selector: "deleteRenderPipelineStateRef:allocator:",
        closure: Closure::Implemented {
            evidence: "retires the task-local render-pipeline-state registry entry",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03f1),
        selector: "deleteFenceRef:allocator:",
        closure: Closure::Implemented {
            evidence: "retires the fence generations this device holds under that ref. The ref \
                       space was the open question and it was measured: a driven macos-15 boot's \
                       two fence deletes both named a ref this device already held generations \
                       for (delete_fence_ref_live=2, _absent=0), so the serializer's fence space \
                       and the space a command stream's fence records use are the same numbers. \
                       Not retiring them was a defect rather than a gap --- a wait is satisfied \
                       when the stored generation is at or past its target, so a generation \
                       outliving its fence let the next fence to be handed that ref pass its \
                       first wait with nothing behind it",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03f4),
        selector: "deleteHeapRef:allocator:",
        closure: Closure::Unresolved {
            question: "heap destroy has no per-kind registry to retire (see 0x3e8)",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03f6),
        selector: "deleteRasterizationRateMapRef:allocator:",
        closure: Closure::Unresolved {
            question: "rate-map destroy has no per-kind registry to retire (see 0x32)",
        },
    },
    Op {
        rail: Rail::Root,
        opcode: Some(0x03f7),
        selector: "deleteIndirectCommandBufferRef:allocator:",
        closure: Closure::Unresolved {
            question: "ICB destroy has no per-kind registry to retire (see 0x3e8)",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0000),
        selector: "drawPrimitives:vertexStart:vertexCount:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0001),
        selector: "drawPrimitives:vertexStart:vertexCount:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0002),
        selector: "drawPrimitives:vertexStart:vertexCount:instanceCount:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0003),
        selector: "drawPrimitives:vertexStart:vertexCount:instanceCount:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0004),
        selector: "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0005),
        selector: "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0006),
        selector: "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0007),
        selector: "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0008),
        selector: "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0009),
        selector: "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x000a),
        selector: "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x000b),
        selector: "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:",
        closure: Closure::Implemented {
            evidence: "direct draw; counts, base vertex/instance and index window are lifted and encoded",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x000c),
        selector: "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:; drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:",
        closure: Closure::Unresolved {
            question: "tessellated draw: patch counts are bounds-checked but no field is lifted and no tessellator exists, so the geometry is dropped with a count",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x000d),
        selector: "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:",
        closure: Closure::Unresolved {
            question: "tessellated draw: patch counts are bounds-checked but no field is lifted and no tessellator exists, so the geometry is dropped with a count",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x000f),
        selector: "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:",
        closure: Closure::Unresolved {
            question: "tessellated draw: patch counts are bounds-checked but no field is lifted and no tessellator exists, so the geometry is dropped with a count",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0010),
        selector: "drawPrimitives:indirectBuffer:indirectBufferOffset:",
        closure: Closure::Implemented {
            evidence: "indirect draw; the argument buffer is resolved and the draw encoded from it",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0011),
        selector: "drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:",
        closure: Closure::Implemented {
            evidence: "indexed indirect draw; as 0x10 with the index buffer and type",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0012),
        selector: "drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:",
        closure: Closure::Unresolved {
            question: "indirect tessellated draw: as the direct forms, and the patch counts are additionally in GPU-written memory",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0013),
        selector: "drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:",
        closure: Closure::Unresolved {
            question: "indirect tessellated draw: as the direct forms, and the patch counts are additionally in GPU-written memory",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0014),
        selector: "executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
        closure: Closure::Implemented {
            evidence: "executeCommandsInBuffer, indirect form; accumulated on the pass and executed at pass end",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0015),
        selector: "executeCommandsInBuffer:withRange:",
        closure: Closure::Implemented {
            evidence: "executeCommandsInBuffer over an explicit range; as 0x14",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0016),
        selector: "memoryBarrierWithResources:count:afterStages:beforeStages:",
        closure: Closure::ProvenNoOp {
            cell: "one host submission per pass boundary",
            evidence: "a resource-scoped barrier inside a pass is implied by the submission boundary this rail already places at pass granularity",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0017),
        selector: "memoryBarrierWithScope:afterStages:beforeStages:",
        closure: Closure::ProvenNoOp {
            cell: "one host submission per pass boundary",
            evidence: "as 0x16, for the scope-qualified form",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0018),
        selector: "updateFence:afterStages:",
        closure: Closure::Implemented {
            evidence: "render-encoder fence update, executed on the fence domain the render rail owns",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0019),
        selector: "waitForFence:beforeStages:",
        closure: Closure::Implemented {
            evidence: "render-encoder fence wait, executed on the fence domain the render rail owns",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x001a),
        selector: "writeDescriptor",
        closure: Closure::Implemented {
            evidence: "render-pass descriptor: attachments, load/store actions, clear values and extents",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x001b),
        selector: "useHeap:stages:; useHeaps:count:stages:",
        closure: Closure::Unresolved {
            question: "useHeap: stages are lifted and the record is priced as a heap declaration, but execution treats residency as a no-op. The selector carries no usage argument, so the read/write split that decides whether the no-op is sound cannot be read from it at all",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x001e),
        selector: "writeDescriptor",
        closure: Closure::Refused {
            route: "render_pass_raster_sample_count_dropped",
            evidence: "the pass default raster sample count is honoured at 1, which is the API default and what this rail renders at; any other value is refused by name rather than rendered at the wrong rate",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0020),
        selector: "writeDescriptor",
        closure: Closure::Unresolved {
            question: "programmable sample positions move fragments within a pixel; dropped with a count and no refusal, so the loss is unpriced",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0021),
        selector: "writeDescriptor",
        closure: Closure::Unresolved {
            question: "a pass rasterization rate map has no host object (see root 0x32); dropped with a count",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0022),
        selector: "writeDescriptor",
        closure: Closure::Unresolved {
            question: "imageblock sample length is tile-shader pass geometry with no executor; dropped with a count",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0023),
        selector: "writeDescriptor",
        closure: Closure::Unresolved {
            question: "tile threadgroup memory length is tile-shader pass geometry with no executor; dropped with a count",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0024),
        selector: "writeDescriptor",
        closure: Closure::Unresolved {
            question: "tile size is tile-shader pass geometry with no executor; dropped with a count",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0065),
        selector: "setBlendColorRed:green:blue:alpha:",
        closure: Closure::Implemented {
            evidence: "blend colour, carried to the pipeline's blend state",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0066),
        selector: "setColorStoreAction:atIndex:",
        closure: Closure::Implemented {
            evidence: "colour store-action override, applied to the declared pass slot",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0067),
        selector: "setColorStoreActionOptions:atIndex:",
        closure: Closure::Refused {
            route: "render_store_action_options_dropped",
            evidence: "store-action options carry only MTLStoreActionOptionCustomSamplePositions, which asks a multisample resolve to use the pass's programmable sample positions; this rail sets none, so the none value is honoured because it is the API default and asks for nothing, and every other value — including the undeclared bits a mask would fold onto the flag — is refused by name rather than counted",        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0068),
        selector: "setDepthStencilState:",
        closure: Closure::Implemented {
            evidence: "depth/stencil state bind, resolved from the task-local registry",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0069),
        selector: "setDepthStoreAction:",
        closure: Closure::Implemented {
            evidence: "depth store-action override",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x006a),
        selector: "setDepthStoreActionOptions:",
        closure: Closure::Refused {
            route: "render_store_action_options_dropped",
            evidence: "depth store-action options; as 0x67",        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x006b),
        selector: "setCullMode:",
        closure: Closure::Implemented {
            evidence: "cull mode, carried as an ordinal and translated by the running rail",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x006c),
        selector: "setDepthBias:slopeScale:clamp:",
        closure: Closure::Implemented {
            evidence: "depth bias, slope scale and clamp",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x006d),
        selector: "setDepthClipMode:",
        closure: Closure::Implemented {
            evidence: "depth clip mode, carried as an ordinal and translated by the running rail",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x006e),
        selector: "setFragmentBuffer:offset:atIndex:; setFragmentBuffers:offsets:withRange:; setFragmentBytes:length:atIndex:",
        closure: Closure::Implemented {
            evidence: "fragment buffer binds; the encoder's binding table takes the delta",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x006f),
        selector: "setFragmentBufferOffset:atIndex:",
        closure: Closure::Implemented {
            evidence: "fragment buffer offset, against the already-bound buffer",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0070),
        selector: "setFragmentSamplerState:atIndex:; setFragmentSamplerStates:withRange:; setFragmentTexture:atTextureIndex:samplerState:atSamplerIndex:",
        closure: Closure::Implemented {
            evidence: "fragment sampler binds",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0071),
        selector: "setFragmentSamplerState:lodMinClamp:lodMaxClamp:atIndex:; setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
        closure: Closure::Implemented {
            evidence: "fragment sampler binds with an LOD clamp",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0072),
        selector: "setFragmentTexture:atIndex:; setFragmentTexture:atTextureIndex:samplerState:atSamplerIndex:; setFragmentTextures:withRange:",
        closure: Closure::Implemented {
            evidence: "fragment texture binds",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0073),
        selector: "setFrontFacingWinding:",
        closure: Closure::Implemented {
            evidence: "winding order",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0074),
        selector: "setRenderPipelineState:",
        closure: Closure::Implemented {
            evidence: "render pipeline state bind, resolved from the task-local registry",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0075),
        selector: "setScissorRect:",
        closure: Closure::Implemented {
            evidence: "single scissor rectangle",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0076),
        selector: "setScissorRects:count:",
        closure: Closure::Implemented {
            evidence: "scissor rectangle array",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0077),
        selector: "setStencilFrontReferenceValue:backReferenceValue:; setStencilReferenceValue:",
        closure: Closure::Implemented {
            evidence: "stencil reference values",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0078),
        selector: "setStencilStoreAction:",
        closure: Closure::Implemented {
            evidence: "stencil store-action override",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0079),
        selector: "setStencilStoreActionOptions:",
        closure: Closure::Refused {
            route: "render_store_action_options_dropped",
            evidence: "stencil store-action options; as 0x67",        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x007a),
        selector: "setTessellationFactorBuffer:offset:instanceStride:",
        closure: Closure::Unresolved {
            question: "tessellation factor buffer is the state half of a tessellated draw; dropped with a count that must track the draw's own",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x007b),
        selector: "setTessellationFactorScale:",
        closure: Closure::Unresolved {
            question: "tessellation factor scale; dropped with a count whenever it is not the 1.0 default",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x007c),
        selector: "setTriangleFillMode:",
        closure: Closure::Implemented {
            evidence: "triangle fill mode, carried as an ordinal and translated by the running rail",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x007d),
        selector: "setVertexBuffer:offset:atIndex:; setVertexBuffers:offsets:withRange:; setVertexBytes:length:atIndex:",
        closure: Closure::Implemented {
            evidence: "vertex buffer binds",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x007e),
        selector: "setVertexBufferOffset:atIndex:",
        closure: Closure::Implemented {
            evidence: "vertex buffer offset",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x007f),
        selector: "setVertexSamplerState:atIndex:; setVertexSamplerStates:withRange:",
        closure: Closure::Implemented {
            evidence: "vertex sampler binds",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0080),
        selector: "setVertexSamplerState:lodMinClamp:lodMaxClamp:atIndex:; setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
        closure: Closure::Implemented {
            evidence: "vertex sampler binds with an LOD clamp",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0081),
        selector: "setVertexTexture:atIndex:; setVertexTextures:withRange:",
        closure: Closure::Implemented {
            evidence: "vertex texture binds",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0082),
        selector: "setViewport:",
        closure: Closure::Implemented {
            evidence: "single viewport",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0083),
        selector: "setViewports:count:",
        closure: Closure::Implemented {
            evidence: "viewport array",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0084),
        selector: "setVisibilityResultMode:offset:",
        closure: Closure::Implemented {
            evidence: "visibility result mode arms the draw's occlusion query at the named offset",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0085),
        selector: "textureBarrier",
        closure: Closure::ProvenNoOp {
            cell: "one host submission per pass boundary",
            evidence: "texture barrier; as 0x16",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0088),
        selector: "setLineWidth:",
        closure: Closure::Implemented {
            evidence: "line width, latched per encoder and set per draw by the running rail; the Vulkan rail sets it wherever the draw rasterizes lines and refuses a width its host cannot serve, and the Metal rail names the loss its encoder has no setter for",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0089),
        selector: "useResource:usage:stages:; useResources:count:usage:stages:",
        closure: Closure::Unresolved {
            question: "useResource: usage and stages are lifted and classified, and a read declaration is the case a per-draw binder owes nothing on. What is not established is what the record obliges: whether an indirectly-referenced resource the guest never declares is legal to touch, and whether a write declaration orders the GPU's writes against a later read this device would not otherwise order. Both are terms of the residency contract; neither can be read off which declarations a guest happens to make",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0099),
        selector: "setVertexAmplificationMode:value:",
        closure: Closure::Unresolved {
            question: "vertex amplification mode; dropped with a count whenever it asks for more than one view",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x009a),
        selector: "setVertexAmplificationCount:viewMappings:",
        closure: Closure::Unresolved {
            question: "vertex amplification count; as 0x99, and the record's view mappings are a second loss counted apart from it — they offset the viewport and render-target array indices a view rasterises into, so a count of one whose mapping is not the identity is a draw aimed at a slice this rail does not aim it at",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x009b),
        selector: "dispatchThreadsPerTile:",
        closure: Closure::Unresolved {
            question: "tile dispatch: work the guest asked for that this rail has no tile pipeline to run",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x009c),
        selector: "setThreadgroupMemoryLength:offset:atIndex:",
        closure: Closure::Unresolved {
            question: "tile threadgroup (imageblock) memory; no tile argument table exists to bind into",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x009d),
        selector: "setTileBuffer:offset:atIndex:; setTileBuffers:offsets:withRange:; setTileBytes:length:atIndex:",
        closure: Closure::Unresolved {
            question: "tile buffer bind; as 0x9c",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x009e),
        selector: "setTileBufferOffset:atIndex:",
        closure: Closure::Unresolved {
            question: "tile buffer offset; as 0x9c",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x009f),
        selector: "setTileSamplerState:atIndex:; setTileSamplerStates:withRange:",
        closure: Closure::Unresolved {
            question: "tile sampler bind; as 0x9c",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x00a0),
        selector: "setTileSamplerState:lodMinClamp:lodMaxClamp:atIndex:; setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
        closure: Closure::Unresolved {
            question: "tile sampler bind with LOD clamp; as 0x9c",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x00a1),
        selector: "setTileTexture:atIndex:; setTileTextures:withRange:",
        closure: Closure::Unresolved {
            question: "tile texture bind; as 0x9c",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x00a2),
        selector: "dispatchThreadsPerTile:inRegion:",
        closure: Closure::Unresolved {
            question: "region-bounded tile dispatch; as 0x9b",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x00a3),
        selector: "dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:",
        closure: Closure::Unresolved {
            question: "region-bounded tile dispatch with a render-target index; as 0x9b",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x00a4),
        selector: "getTileDimensions:",
        closure: Closure::Unresolved {
            question: "getTileDimensions is a query whose answer the guest reads back from its own buffer; leaving it unwritten hands the guest whatever the buffer last held, which is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x00a5),
        selector: "setVertexBuffer:offset:attributeStride:atIndex:; setVertexBuffers:offsets:attributeStrides:withRange:; setVertexBytes:length:attributeStride:atIndex:",
        closure: Closure::Implemented {
            evidence: "vertex buffer binds carrying an attribute stride",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x00a6),
        selector: "setVertexBufferOffset:attributeStride:atIndex:",
        closure: Closure::Implemented {
            evidence: "vertex buffer offset carrying an attribute stride",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00c8),
        selector: "dispatchThreadgroups:threadsPerThreadgroup:",
        closure: Closure::Implemented {
            evidence: "dispatch by threadgroup count",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00c9),
        selector: "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:",
        closure: Closure::Implemented {
            evidence: "indirect dispatch by threadgroup count",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00ca),
        selector: "dispatchThreads:threadsPerThreadgroup:",
        closure: Closure::Implemented {
            evidence: "dispatch by thread count",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00cb),
        selector: "setBuffer:offset:atIndex:; setBuffers:offsets:withRange:; setBytes:length:atIndex:",
        closure: Closure::Implemented {
            evidence: "buffer binds on the compute argument table",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00cc),
        selector: "setSamplerState:atIndex:; setSamplerStates:withRange:",
        closure: Closure::Implemented {
            evidence: "sampler binds",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00cd),
        selector: "setSamplerState:lodMinClamp:lodMaxClamp:atIndex:; setSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
        closure: Closure::Implemented {
            evidence: "sampler binds with an LOD clamp",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00ce),
        selector: "setTexture:atIndex:; setTextures:withRange:",
        closure: Closure::Implemented {
            evidence: "texture binds",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00cf),
        selector: "setBufferOffset:atIndex:",
        closure: Closure::Implemented {
            evidence: "buffer offset against an already-bound buffer",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d0),
        selector: "setComputePipelineState:",
        closure: Closure::Implemented {
            evidence: "compute pipeline state bind",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d1),
        selector: "setStageInRegion:",
        closure: Closure::Implemented {
            evidence: "stage-in region",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d2),
        selector: "setStageInRegionWithIndirectBuffer:indirectBufferOffset:",
        closure: Closure::Implemented {
            evidence: "stage-in region read from an indirect buffer",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d3),
        selector: "setThreadgroupMemoryLength:atIndex:",
        closure: Closure::Implemented {
            evidence: "threadgroup memory length per index",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d4),
        selector: "updateFence:",
        closure: Closure::Unresolved {
            question: "compute-encoder fence update is ordering the guest stated explicitly and nothing executes it; the render rail's fence domain does not serve this encoder",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d5),
        selector: "waitForFence:",
        closure: Closure::Unresolved {
            question: "compute-encoder fence wait; as 0xd4",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d6),
        selector: "memoryBarrierWithResources:count:",
        closure: Closure::ProvenNoOp {
            cell: "one host submission per dispatch",
            evidence: "consecutive dispatches are separated by a queue submission, which is stronger than the resource barrier the record asks for",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d7),
        selector: "dispatchThreadgroups:threadsPerThreadgroup:; dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:; dispatchThreads:threadsPerThreadgroup:; dispatchThreadsWithIndirectBuffer:indirectBufferOffset:; executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:; executeCommandsInBuffer:withRange:; maybeEmitSerialBarrier; memoryBarrierWithScope:",
        closure: Closure::ProvenNoOp {
            cell: "one host submission per dispatch",
            evidence: "as 0xd6, for the scope-qualified form",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d8),
        selector: "setImageblockWidth:height:",
        closure: Closure::Implemented {
            evidence: "imageblock dimensions",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00d9),
        selector: "setBuffer:offset:attributeStride:atIndex:; setBuffers:offsets:attributeStrides:withRange:; setBytes:length:attributeStride:atIndex:",
        closure: Closure::Implemented {
            evidence: "buffer binds carrying an attribute stride",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00da),
        selector: "setBufferOffset:attributeStride:atIndex:",
        closure: Closure::Implemented {
            evidence: "buffer offset carrying an attribute stride",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00db),
        selector: "writeDescriptor",
        closure: Closure::Implemented {
            evidence: "pass dispatch type; serial and concurrent are both named and out-of-contract values refuse",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00dc),
        selector: "encodeStartDoWhile",
        closure: Closure::Unresolved {
            question: "do/while sequencing block: the control-flow SPI opens a multi-record session whose predication this rail does not evaluate",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00dd),
        selector: "encodeEndDoWhile:offset:comparison:referenceValue:",
        closure: Closure::Unresolved {
            question: "end of a do/while block; as 0xdc",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00de),
        selector: "encodeStartWhile:offset:comparison:referenceValue:",
        closure: Closure::Unresolved {
            question: "while block; as 0xdc",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00df),
        selector: "encodeEndWhile",
        closure: Closure::Unresolved {
            question: "end of a while block; as 0xdc",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00e0),
        selector: "encodeStartIf:offset:comparison:referenceValue:",
        closure: Closure::Unresolved {
            question: "if block; as 0xdc",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00e1),
        selector: "encodeStartElse",
        closure: Closure::Unresolved {
            question: "else block; as 0xdc",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00e2),
        selector: "encodeEndIf",
        closure: Closure::Unresolved {
            question: "end of an if block; as 0xdc",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00e3),
        selector: "insertCompressedTextureReinterpretationFlush",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are written directly; no host compressed representation exists",
            evidence: "a compressed-texture flush has no host metadata to make visible",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00e4),
        selector: "executeCommandsInBuffer:withRange:",
        closure: Closure::Unresolved {
            question: "executeCommandsInBuffer over a range, on the compute rail: routed to the sequencing owner rather than executed as ICB work",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00e5),
        selector: "executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
        closure: Closure::Unresolved {
            question: "executeCommandsInBuffer, indirect form, on the compute rail; as 0xe4",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x00e6),
        selector: "dispatchThreadsWithIndirectBuffer:indirectBufferOffset:",
        closure: Closure::Implemented {
            evidence: "indirect dispatch by thread count",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x012c),
        selector: "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:; copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:",
        closure: Closure::Implemented {
            evidence: "buffer to texture copy, row and image pitches honoured",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x012d),
        selector: "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
        closure: Closure::Implemented {
            evidence: "buffer to buffer copy",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x012e),
        selector: "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:; copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:",
        closure: Closure::Implemented {
            evidence: "texture to buffer copy",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x012f),
        selector: "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
        closure: Closure::Implemented {
            evidence: "texture region copy",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0130),
        selector: "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:",
        closure: Closure::Implemented {
            evidence: "texture region copy with blit options",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0131),
        selector: "copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:",
        closure: Closure::Unresolved {
            question: "ICB copy changes what a later executeCommandsInBuffer runs; dropping it leaves the destination holding stale commands, which executes rather than merely losing work",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0132),
        selector: "fillBuffer:range:value:",
        closure: Closure::Implemented {
            evidence: "buffer fill with a byte value",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0133),
        selector: "generateMipmapsForTexture:",
        closure: Closure::Implemented {
            evidence: "mipmap generation",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0134),
        selector: "optimizeContentsForCPUAccess:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are written directly; there is no host-private texture layout",
            evidence: "optimizeContentsForCPUAccess has no host representation to re-lay-out",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0135),
        selector: "optimizeContentsForGPUAccess:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are written directly; there is no host-private texture layout",
            evidence: "optimizeContentsForGPUAccess; as 0x134",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0136),
        selector: "optimizeContentsForCPUAccess:slice:level:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are written directly; there is no host-private texture layout",
            evidence: "slice/level form of 0x134",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0137),
        selector: "optimizeContentsForGPUAccess:slice:level:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are written directly; there is no host-private texture layout",
            evidence: "slice/level form of 0x135",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0138),
        selector: "optimizeIndirectCommandBuffer:withRange:",
        closure: Closure::ProvenNoOp {
            cell: "no host ICB is materialised on this rail",
            evidence: "optimizeIndirectCommandBuffer is a reuse hint; skipping it costs speed and not semantics",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0139),
        selector: "resetCommandsInBuffer:withRange:",
        closure: Closure::Unresolved {
            question: "an ICB reset the device drops leaves commands live that the guest retired; as 0x131 this executes stale work rather than losing work",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x013a),
        selector: "synchronizeResource:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are the single copy of resource content",
            evidence: "synchronizeResource has no host-side copy to make CPU-visible",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x013b),
        selector: "synchronizeTexture:slice:level:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are the single copy of resource content",
            evidence: "slice/level form of 0x13a",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x013c),
        selector: "updateFence:",
        closure: Closure::Implemented {
            evidence: "blit-encoder fence update, executed on the fence domain the blit rail owns",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x013d),
        selector: "waitForFence:",
        closure: Closure::Implemented {
            evidence: "blit-encoder fence wait, executed on the fence domain the blit rail owns",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x013e),
        selector: "copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:; copyFromTexture:toTexture:",
        closure: Closure::Implemented {
            evidence: "slice/level ranged texture to texture copy",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x013f),
        selector: "fillBuffer:range:pattern4:",
        closure: Closure::Implemented {
            evidence: "buffer fill with a four-byte pattern",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0140),
        selector: "fillTexture:level:slice:region:bytes:length:",
        closure: Closure::Unresolved {
            question: "fillTexture from bytes is a write the guest expects to land; dropping it leaves the region holding what it held before and the guest reads back content it believes it wrote",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0141),
        selector: "fillTexture:level:slice:region:color:; fillTexture:level:slice:region:color:pixelFormat:",
        closure: Closure::Unresolved {
            question: "fillTexture from a colour additionally needs the clear colour converted into the destination's pixel format; as 0x140",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0142),
        selector: "invalidateCompressedTexture:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are written directly; no lossless-compression metadata exists here",
            evidence: "invalidateCompressedTexture has no host compressed representation to mark stale",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: Some(0x0143),
        selector: "invalidateCompressedTexture:slice:level:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are written directly; no lossless-compression metadata exists here",
            evidence: "slice/level form of 0x142",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01c2),
        selector: "computePipelineStateInfo:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01c3),
        selector: "heapTextureDescriptorSizeAndAlign:sizeAndAlign:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01c4),
        selector: "getRasterizationRateMapInfo:layerCount:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01c5),
        selector: "copyRasterizationRateParameterBuffer:buffer:bufferOffset:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01c6),
        selector: "mapScreenToPhysicalCoordinates:forScreenCoordinate:forLayer:toPhysicalCoordinate:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01c7),
        selector: "mapPhysicalToScreenCoordinates:forPhysicalCoordinate:forLayer:toScreenCoordinate:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01c9),
        selector: "renderPipelineStateInfo:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01ca),
        selector: "renderPipelineStateImageBlockMemoryLength:imageblockDimensions:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01cb),
        selector: "computePipelineStateImageBlockMemoryLength:imageblockDimensions:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01cd),
        selector: "bufferHostResourceInfo:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01ce),
        selector: "textureHostResourceInfo:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01cf),
        selector: "heapHostResourceInfo:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01d0),
        selector: "samplerStateHostResourceInfo:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01d1),
        selector: "icbHostResourceInfo:info:",
        closure: Closure::Unresolved {
            question: "the ICB host-resource query is the one info record this rail decodes, and it always declines: the answer is not computed. Its reply pair is named on the failure channel, which is where the answer would go, but the query is still unanswered",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01d2),
        selector: "renderPipelineHostResourceInfo:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01d3),
        selector: "computePipelineHostResourceInfo:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01d4),
        selector: "depthStencilHostResourceInfo:info:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: Some(0x01d5),
        selector: "heapTextureDescriptorSizeAndAlign:sizeAndAlign:",
        closure: Closure::Unresolved {
            question: "info-encoder query: the guest hands over a reply buffer and reads it back regardless of whether anything was written, so an unanswered query is a wrong answer rather than a dropped command",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0086),
        selector: "useHeaps:count:",
        closure: Closure::Unresolved {
            question: "unqualified useHeaps inherited from the serializer base class. It carries neither usage nor stages, so it is priced as a heap declaration and nothing about the access can be read from it",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: Some(0x0087),
        selector: "useResources:count:usage:",
        closure: Closure::Unresolved {
            question: "unqualified useResources inherited from the serializer base class. Usage is lifted and classified as on 0x89, widened to 32 bits by this form; there are no stages",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x0086),
        selector: "useHeaps:count:",
        closure: Closure::Unresolved {
            question: "unqualified useHeaps inherited from the serializer base class. It carries neither usage nor stages, so it is priced as a heap declaration and nothing about the access can be read from it",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: Some(0x0087),
        selector: "useResources:count:usage:",
        closure: Closure::Unresolved {
            question: "unqualified useResources inherited from the serializer base class. Usage is lifted and classified as on 0x89, widened to 32 bits by this form; there are no stages",
        },
    },
    Op {
        rail: Rail::Render,
        opcode: None,
        selector: "beginSegment:protectionOptions:",
        closure: Closure::ProvenNoOp {
            cell: "segment framing, not a record",
            evidence: "beginSegment writes the segment header the stream walker already frames every record with; it carries no opcode field and no operation of its own",
        },
    },
    Op {
        rail: Rail::Compute,
        opcode: None,
        selector: "beginSegment:protectionOptions:",
        closure: Closure::ProvenNoOp {
            cell: "segment framing, not a record",
            evidence: "beginSegment writes the segment header the stream walker already frames every record with; it carries no opcode field and no operation of its own",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: None,
        selector: "beginSegment:protectionOptions:",
        closure: Closure::ProvenNoOp {
            cell: "segment framing, not a record",
            evidence: "beginSegment writes the segment header the stream walker already frames every record with; it carries no opcode field and no operation of its own",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: None,
        selector: "optimize:slice:level:withCommand:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are written directly; there is no host-private texture layout",
            evidence: "emits 0x136 or 0x137 through its command: argument and closes with them",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: None,
        selector: "optimize:withCommand:",
        closure: Closure::ProvenNoOp {
            cell: "guest pages are written directly; there is no host-private texture layout",
            evidence: "the command: argument is written into the record's own opcode field, so this selector emits 0x134 or 0x135 and closes with them",
        },
    },
    Op {
        rail: Rail::Blit,
        opcode: None,
        selector: "optimizeReset:withRange:withCommand:",
        closure: Closure::Unresolved {
            question: "emits 0x138 or 0x139 through its command: argument; the reset half is unresolved, so this selector is unresolved with it",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: None,
        selector: "beginSegment:protectionOptions:",
        closure: Closure::ProvenNoOp {
            cell: "segment framing, not a record",
            evidence: "beginSegment writes the segment header the stream walker already frames every record with; it carries no opcode field and no operation of its own",
        },
    },
    Op {
        rail: Rail::Info,
        opcode: None,
        selector: "mapCoordinateInternal:fromCoordinate:forLayer:toCoordinate:command:",
        closure: Closure::Unresolved {
            question: "emits 0x1c6 or 0x1c7 through its command: argument; both are unanswered info queries",
        },
    },
    Op {
        rail: Rail::Event,
        opcode: Some(0x0190),
        selector: "waitForEvent:value:",
        closure: Closure::Implemented {
            evidence: "an event wait against the task's event generation; unmet leaves the packet pending rather than dropping it",
        },
    },
    Op {
        rail: Rail::Event,
        opcode: Some(0x0191),
        selector: "signalEvent:value:",
        closure: Closure::Implemented {
            evidence: "advances the task's event generation; a signal that does not advance it is a no-op by the API's own monotonic rule rather than by this device's choice",
        },
    },
    Op {
        rail: Rail::Event,
        opcode: Some(0x0192),
        selector: "waitForEvent:value:timeout:",
        closure: Closure::Refused {
            route: "event_wait_timeout_unsupported",
            evidence: "a bounded wait needs a clock this device does not run against the guest's, so it is refused by name every time rather than executed as the unbounded wait it is not",
        },
    },
];

/// The gates that make the ledger a measurement rather than a list.
///
/// Every one of these is about *closure* — that the row set is exactly the
/// decodable operation set — rather than about whether any particular outcome
/// is the right one. No test can check the latter; that is what the evidence
/// prose is for, and why no variant can be written without it.
#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_wire::manifest::{Coverage, MANIFEST};
    use std::collections::{BTreeMap, BTreeSet};

    /// Manifest selectors whose record carries a fixed opcode, keyed by
    /// `(rail, opcode)`. This is the set the ledger must cover.
    fn manifest_opcodes() -> BTreeMap<(Rail, u32), BTreeSet<&'static str>> {
        let mut out: BTreeMap<(Rail, u32), BTreeSet<&'static str>> = BTreeMap::new();
        for e in MANIFEST {
            let Some(rail) = Rail::from_class(e.class) else {
                panic!("manifest class {} has no rail", e.class);
            };
            for &op in e.opcodes {
                out.entry((rail, op)).or_default().insert(e.selector);
            }
        }
        out
    }

    /// Manifest selectors covered by a view whose record has no fixed opcode.
    fn manifest_opcodeless() -> BTreeSet<(Rail, &'static str)> {
        MANIFEST
            .iter()
            .filter(|e| matches!(e.coverage, Coverage::CoveredNoFixedOpcode { .. }))
            .map(|e| (Rail::from_class(e.class).expect("rail"), e.selector))
            .collect()
    }

    fn ledger_opcodes() -> BTreeSet<(Rail, u32)> {
        LEDGER
            .iter()
            .filter_map(|o| o.opcode.map(|c| (o.rail, c)))
            .collect()
    }

    /// A selector the serializer emits and the ledger has not judged is exactly
    /// the silence this crate exists to remove.
    #[test]
    fn every_manifest_opcode_has_a_row() {
        let have = ledger_opcodes();
        let missing: Vec<_> = manifest_opcodes()
            .keys()
            .copied()
            .filter(|k| !have.contains(k))
            .collect();
        assert!(
            missing.is_empty(),
            "wire manifest records these operations and the closure ledger does not judge them: {missing:#x?}"
        );
    }

    /// The other direction, with one named exception: an inherited base-class
    /// operation is real and invisible to the manifest, so it may be in the
    /// ledger alone — but only by name, so the exception cannot grow silently.
    #[test]
    fn every_row_is_a_manifest_operation_or_a_named_base_class_one() {
        let manifest = manifest_opcodes();
        let base: BTreeSet<_> = OFF_MANIFEST.iter().map(|(r, op, _)| (*r, *op)).collect();
        let stray: Vec<_> = ledger_opcodes()
            .into_iter()
            .filter(|k| !manifest.contains_key(k) && !base.contains(k))
            .collect();
        assert!(
            stray.is_empty(),
            "ledger rows for operations the wire manifest does not record and OFF_MANIFEST does not name: {stray:#x?}"
        );
    }

    #[test]
    fn every_named_off_manifest_operation_has_a_row() {
        let have = ledger_opcodes();
        let missing: Vec<_> = OFF_MANIFEST
            .iter()
            .map(|(r, op, _)| (*r, *op))
            .filter(|k| !have.contains(k))
            .collect();
        assert!(
            missing.is_empty(),
            "OFF_MANIFEST without a row: {missing:#x?}"
        );
    }

    #[test]
    fn every_opcodeless_manifest_selector_has_a_row() {
        let have: BTreeSet<_> = LEDGER
            .iter()
            .filter(|o| o.opcode.is_none())
            .map(|o| (o.rail, o.selector))
            .collect();
        let missing: Vec<_> = manifest_opcodeless()
            .into_iter()
            .filter(|k| !have.contains(k))
            .collect();
        assert!(
            missing.is_empty(),
            "manifest selectors covered without a fixed opcode and unjudged here: {missing:?}"
        );
    }

    /// Two rows for one operation is two answers to one question, and the
    /// reader has no way to tell which one the device obeys.
    #[test]
    fn no_operation_is_judged_twice() {
        let mut seen = BTreeSet::new();
        for o in LEDGER {
            let key = (o.rail, o.opcode, o.selector);
            assert!(seen.insert(key), "duplicate ledger row: {key:?}");
        }
        let mut by_opcode = BTreeSet::new();
        for o in LEDGER.iter().filter(|o| o.opcode.is_some()) {
            let key = (o.rail, o.opcode.unwrap());
            assert!(by_opcode.insert(key), "two rows for {key:#x?}");
        }
    }

    /// An outcome with no reasoning beside it is a guess wearing a variant
    /// name. Nothing here can be recorded without saying why.
    #[test]
    fn every_row_states_its_reasoning() {
        for o in LEDGER {
            let strings: &[(&str, &str)] = &match o.closure {
                Closure::Implemented { evidence } => [("evidence", evidence), ("", "-")],
                Closure::ProvenNoOp { cell, evidence } => [("cell", cell), ("evidence", evidence)],
                Closure::Refused { route, evidence } => [("route", route), ("evidence", evidence)],
                Closure::Unresolved { question } => [("question", question), ("", "-")],
            };
            for (field, value) in strings.iter().filter(|(f, _)| !f.is_empty()) {
                assert!(
                    value.len() > 8,
                    "{:?} {:#x?} has an empty or token `{field}`",
                    o.rail,
                    o.opcode
                );
            }
            assert!(
                !o.selector.is_empty(),
                "{:?} {:#x?} names no selector",
                o.rail,
                o.opcode
            );
        }
    }

    /// **Neither unspellable outcome may be spelled, in any row.**
    ///
    /// The module doc names two: "the current workload does not issue it" and
    /// "the old backend drops it too". Both are easy to write into a row's
    /// reasoning without noticing, because both feel like evidence — and both
    /// would let the cutover gate pass on a claim about behaviour rather than
    /// about the contract. A ledger closed on either is a ledger that says
    /// nothing about the records a guest has not made yet, which is the whole
    /// reason it exists.
    ///
    /// An `Unresolved` row's `question` is checked for the same words, because
    /// a question that names a workload as its closing condition is that
    /// outcome written down in advance — which is how `0x0089` had it.
    #[test]
    fn no_row_rests_on_a_workload_or_on_the_backend_being_replaced() {
        const UNSPELLABLE: &[&str] = &[
            "workload",
            "driven boot",
            "boot shows",
            "guest issues",
            "old backend",
            "legacy backend",
            "previous backend",
        ];
        for o in LEDGER {
            let reasoning = match o.closure {
                Closure::Implemented { evidence } | Closure::Refused { evidence, .. } => {
                    [evidence, ""]
                }
                Closure::ProvenNoOp { cell, evidence } => [cell, evidence],
                Closure::Unresolved { question } => [question, ""],
            };
            for text in reasoning {
                for bad in UNSPELLABLE {
                    assert!(
                        !text.contains(bad),
                        "{:?} {:#x?} rests on `{bad}`, which is not an outcome: {text}",
                        o.rail,
                        o.opcode
                    );
                }
            }
        }
    }

    /// `counts` must see every row exactly once, or the cutover gate is
    /// reporting on a subset of the ledger.
    #[test]
    fn counts_cover_the_whole_ledger() {
        assert_eq!(counts(None).total(), LEDGER.len());
        let per_rail: usize = Rail::ALL.iter().map(|&r| counts(Some(r)).total()).sum();
        assert_eq!(per_rail, LEDGER.len());
        assert_eq!(counts(None).unresolved, blocking().count());
    }

    #[test]
    fn find_answers_from_the_ledger() {
        for o in LEDGER.iter().filter(|o| o.opcode.is_some()) {
            let got = find(o.rail, o.opcode.unwrap()).expect("row is findable");
            assert_eq!(got.selector, o.selector);
        }
    }

    /// The Seam 0 exit reading, printed rather than asserted: this number is
    /// what the remaining seams reduce, and pinning it would turn every honest
    /// re-classification into a test edit.
    #[test]
    fn report_the_blocking_set() {
        let c = counts(None);
        println!(
            "closure ledger: {} operations — {} implemented, {} proven no-op, {} refused, {} unresolved",
            c.total(),
            c.implemented,
            c.proven_noop,
            c.refused,
            c.unresolved
        );
        for &rail in Rail::ALL {
            let c = counts(Some(rail));
            println!("  {rail:?}: {} of {} unresolved", c.unresolved, c.total());
        }
        for o in blocking() {
            let opcode = match o.opcode {
                Some(c) => format!("{c:#06x}"),
                None => "  --  ".to_string(),
            };
            println!(
                "  BLOCKING {:8} {opcode} {}",
                format!("{:?}", o.rail),
                o.selector
            );
        }
    }
}
