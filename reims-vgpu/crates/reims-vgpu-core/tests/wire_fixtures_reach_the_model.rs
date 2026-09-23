//! Run Apple's own records through the model's decode-and-resolve path.
//!
//! `reims-vgpu-wire` captures what the serializer emits and pins its layouts;
//! `reims-vgpu-protocol` lifts those layouts into records and this crate turns
//! them into operations. This is the only test that puts the whole path against
//! bytes Apple actually produced.
//!
//! # The claim
//!
//! **A record Apple's serializer produced is never declined for a reason about
//! its shape.** The model may decline an operation whose ledger row is open, or
//! settled as a refusal, or absent — those are contract answers, they are
//! reported below, and they are not failures. What it may not do is call a
//! well-formed record short, over-long or over-counted, because that is the
//! guest's work lost to a layout this project has wrong.
//!
//! The distinction is the whole design. `reims-vgpu`'s equivalent test found
//! four real divergences that way, and each of them looked exactly like the
//! contract answers on either side of it until the two were separated.
//!
//! # The second claim, which a single decode cannot make
//!
//! A capture arena is zero-filled exactly where a guest's command ring is not,
//! so a field read *wider* than the serializer writes agrees with the fixture
//! by accident. The oracle measures a per-bit `written_mask` — every case
//! captured twice under complementary fills — and
//! [`the_model_reads_no_bit_apples_serializer_never_wrote`] repaints every
//! unwritten bit to all-zero and then to all-one and requires the same resolved
//! operation both times. A model whose answer moves read a byte the serializer
//! never wrote, which in a guest is stale ring rather than data.
//!
//! Both tests share one call into the model, and here there is no second copy
//! to write: `resolve::operation` has a single entry and the ledger inside it
//! picks the decoder.
//!
//! Fixtures are not committed. With none present both tests are `ignored`,
//! decided at build time by `build.rs`.

#![cfg(wire_fixtures)]

use reims_vgpu_core::access::{BackingId, ByteRange};
use reims_vgpu_core::exec::{ExecArenas, ResolvedOperation};
use reims_vgpu_core::identity::{ChannelId, ObjectListRef, ResourceId, SlotGeneration, TaskId};
use reims_vgpu_core::lifecycle::{Lifecycle, LifecycleOp, Storage};
use reims_vgpu_core::resolve::{operation, RefResolver, ResolveRefusal};
use reims_vgpu_protocol::closure::{Rail, LEDGER};
use reims_vgpu_protocol::decode::{op, DecodeRefusal};
use reims_vgpu_testkit::{fixtures, unhex};

/// A resolver that answers every ref.
///
/// Resolution is not what this test is about: a fixture's refs are the oracle
/// stub's, no object table exists here, and an unresolvable ref would report a
/// missing object where the question is about bytes. Answering everything means
/// every refusal that does reach the assertions came from the record.
struct Everything;

impl RefResolver for Everything {
    fn resource(&self, object_ref: u32) -> Option<ResourceId> {
        Some(ResourceId {
            slot: ObjectListRef(object_ref),
            generation: SlotGeneration(1),
        })
    }
}

/// What the model made of one record.
enum Verdict {
    /// It became an operation.
    Resolved(ResolvedOperation),
    /// The ledger declined it: no row, an open row, or a settled refusal. A
    /// contract answer, reported and not failed.
    Declined(&'static str),
    /// The model refused a record the serializer produced, for a reason about
    /// the record's shape. That is a layout this project has wrong.
    WrongShape(&'static str),
}

/// Read one record the way production would.
fn read(rail: Rail, bytes: &[u8]) -> Verdict {
    let Ok(view) = op(bytes, 0) else {
        // The framing itself did not parse, which is a shape claim about a
        // buffer the serializer wrote.
        return Verdict::WrongShape("op_framing_refused");
    };
    let mut arenas = ExecArenas::default();
    match operation(rail, &view, &Everything, &mut arenas) {
        Ok(resolved) => Verdict::Resolved(resolved),
        Err(ResolveRefusal::Decode(
            refusal @ (DecodeRefusal::UnknownOpcode { .. }
            | DecodeRefusal::Unjudged { .. }
            | DecodeRefusal::RefusedByContract { .. }),
        )) => Verdict::Declined(refusal.reason()),
        Err(ResolveRefusal::Decode(refusal)) => Verdict::WrongShape(refusal.reason()),
        Err(refusal) => Verdict::WrongShape(refusal.reason()),
    }
}

/// Whether a case's selector wrote a segment header rather than a record.
///
/// A segment header has **no opcode**: it is the eight bytes
/// `-beginSegment:protectionOptions:` writes, and it reaches the model through
/// `ExecBuilder::begin_segment` rather than through `resolve::operation`.
/// Routing it by selector is what keeps a boundary from being read as whatever
/// record its first four bytes happen to spell — which for a segment header is
/// its own length, and for a zero-length one is opcode zero.
fn is_boundary(selector: &str) -> bool {
    selector.starts_with("beginSegment")
}

/// One captured record: what it was called, what emitted it, and its bytes
/// beside the per-bit mask of which of them the serializer actually wrote.
struct Case {
    name: String,
    selector: String,
    rail: Rail,
    bytes: Vec<u8>,
    written_mask: Vec<u8>,
}

/// Every fixture that is a record rather than a boundary.
///
/// Cases whose class names no rail are counted rather than silently dropped:
/// `PGSerializerCommandEncoder` is the shared base class, and a record captured
/// on it was emitted by whichever encoder inherited the selector — the capture
/// does not say which, so there is no rail to dispatch on.
fn cases() -> Vec<Case> {
    let json = fixtures();
    let mut skipped_base_class = 0usize;
    let mut out = Vec::new();
    for case in json["cases"].as_array().expect("cases is an array") {
        let name = case["name"].as_str().expect("name").to_string();
        let class = case["class"].as_str().expect("class");
        let selector = case["selector"].as_str().expect("selector").to_string();
        if is_boundary(&selector) {
            continue;
        }
        let Some(rail) = Rail::from_class(class) else {
            skipped_base_class += 1;
            continue;
        };
        out.push(Case {
            name,
            selector,
            rail,
            bytes: unhex(case["buffer"].as_str().expect("buffer")),
            written_mask: unhex(case["written_mask"].as_str().expect("written_mask")),
        });
    }
    assert!(
        !out.is_empty(),
        "fixtures were present and carried no cases"
    );
    println!(
        "cases: {} records, {skipped_base_class} on the shared base class",
        out.len()
    );
    out
}

/// Every boundary record Apple wrote parses, and names a segment the model has.
///
/// Boundaries are checked here rather than dropped from the sweep. One carries
/// no opcode and cannot go through `resolve::operation`, but "no opcode" is
/// exactly why it must not be silently skipped: a header read as a record would
/// be read as whatever its length field spells, and for a segment that has not
/// ended yet that is opcode zero.
///
/// **A protected begin-segment writes three records, and only two of them are
/// headers.** The type-5 envelope header, then its eight-byte payload carrying
/// the options word, then the encoder's own header. The payload is eight bytes
/// like a header, and read as one it would call itself a render segment — the
/// options word `0x44` lands where a header's length goes and zero lands where
/// its type does.
///
/// What tells them apart is the measured written mask, not a field: a header
/// allocates eight bytes and writes **seven**, and the envelope's payload
/// writes all eight. That is the same measurement the wire crate's own poison
/// sweep produces, and it is a property of what the serializer did rather than
/// of how a fixture was named.
#[test]
fn every_boundary_record_apple_wrote_names_a_segment_the_model_has() {
    use reims_vgpu_protocol::segment::{segment_role, SegmentRole};
    let json = fixtures();
    let mut headers = 0usize;
    let mut envelopes = 0usize;
    for case in json["cases"].as_array().expect("cases is an array") {
        let selector = case["selector"].as_str().expect("selector");
        if !is_boundary(selector) {
            continue;
        }
        let name = case["name"].as_str().expect("name");
        let class = case["class"].as_str().expect("class");
        let rail = Rail::from_class(class).unwrap_or_else(|| panic!("{name}: {class} has no rail"));
        let bytes = unhex(case["buffer"].as_str().expect("buffer"));
        let mask = unhex(case["written_mask"].as_str().expect("written_mask"));
        let expect = &case["expect"];

        if mask.iter().all(|&byte| byte == 0xff) {
            // The envelope's payload: one word, and every byte of it written.
            let envelope = reims_vgpu_wire::ops::segment::protection_options_envelope(&bytes)
                .unwrap_or_else(|e| panic!("{name}: the envelope payload did not parse: {e:?}"));
            assert_eq!(
                envelope.protection_options.get(),
                expect["protection_options"].as_u64().expect("options"),
                "{name}: the envelope carried a different options word"
            );
            envelopes += 1;
            continue;
        }

        let header = reims_vgpu_wire::ops::segment::segment_header(&bytes)
            .unwrap_or_else(|e| panic!("{name}: a segment header did not parse: {e:?}"));
        if let Some(expected_type) = expect["segment_type"].as_u64() {
            assert_eq!(
                u64::from(header.segment_type),
                expected_type,
                "{name}: the header's type is not the one the capture asked for"
            );
        }
        let role = segment_role(header.segment_type).unwrap_or_else(|| {
            panic!("{name}: segment type {} names nothing", header.segment_type)
        });
        // A capture on an encoder class must produce that encoder's segment,
        // not merely *a* segment: reading the type off the wrong byte would
        // still land on a valid role. The envelope header is the exception —
        // it is the protection envelope and belongs to no encoder.
        if let SegmentRole::Encoder(kind) = role {
            assert_eq!(
                kind.rail(),
                rail,
                "{name}: segment kind disagrees with the capture class"
            );
        }
        headers += 1;
    }
    println!("boundaries: {headers} headers, {envelopes} envelope payloads");
    assert!(headers > 0, "no segment header reached this test");
    assert!(envelopes > 0, "no protection envelope reached this test");
}

/// The root rail is object lifecycle, not a command stream: creations and
/// deletions reach the model through a different transaction with no encoder,
/// so a stream decoder declining them is correct rather than a gap.
fn is_stream_rail(rail: Rail) -> bool {
    !matches!(rail, Rail::Root)
}

/// No record Apple produced is declined for its shape.
#[test]
fn no_record_apples_serializer_produced_is_refused_for_its_shape() {
    let mut resolved = 0usize;
    let mut declined: std::collections::BTreeMap<&str, Vec<String>> =
        std::collections::BTreeMap::new();
    let mut wrong = Vec::new();

    for case in cases() {
        let Case {
            name,
            selector,
            rail,
            bytes,
            ..
        } = case;
        if !is_stream_rail(rail) {
            continue;
        }
        match read(rail, &bytes) {
            Verdict::Resolved(_) => resolved += 1,
            Verdict::Declined(reason) => {
                if reason == "decode_opcode_unknown" {
                    // The ledger is keyed by opcode, and six of its rows carry
                    // none: the blit `withCommand:` selectors write their
                    // command argument *into* the opcode field, and the info
                    // rail's coordinate mapping is a request rather than a
                    // record. A lookup by opcode cannot reach those rows, so
                    // "no row" here must mean "an opcode-less row", and this is
                    // what says so. Anything else is an opcode Apple emits that
                    // the ledger has never heard of.
                    assert!(
                        LEDGER.iter().any(|row| row.rail == rail
                            && row.opcode.is_none()
                            && row.selector.split("; ").any(|s| s == selector)),
                        "{name}: {selector} has no ledger row at all"
                    );
                }
                declined.entry(reason).or_default().push(name.clone());
            }
            Verdict::WrongShape(reason) => wrong.push(format!("{name} ({rail:?}): {reason}")),
        }
    }

    println!("resolved: {resolved}");
    for (reason, names) in &declined {
        println!("declined: {reason} x{}", names.len());
    }
    if let Some(names) = declined.get("decode_opcode_unknown") {
        println!("reached only by selector: {}", names.join(", "));
    }
    assert!(
        wrong.is_empty(),
        "records the serializer produced were refused for their shape:\n  {}",
        wrong.join("\n  ")
    );
    assert!(resolved > 0, "no fixture reached the model at all");
}

/// The model's answer does not move when bytes the serializer never wrote do.
#[test]
fn the_model_reads_no_bit_apples_serializer_never_wrote() {
    let mut checked = 0usize;
    let mut moved = Vec::new();

    for case in cases() {
        let Case {
            name,
            rail,
            bytes,
            written_mask: mask,
            ..
        } = case;
        if !is_stream_rail(rail) {
            continue;
        }
        assert_eq!(
            mask.len(),
            bytes.len(),
            "{name}: the written mask and the buffer are different lengths"
        );
        // Every bit the mask says was not written, painted both ways. The two
        // fills are complements, so nothing can agree by accident.
        let mut zeroed = bytes.clone();
        let mut oned = bytes.clone();
        for (index, &written) in mask.iter().enumerate() {
            zeroed[index] &= written;
            oned[index] |= !written;
        }
        if zeroed == oned {
            // Every bit was written; there is nothing to repaint.
            continue;
        }
        checked += 1;
        let a = signature(read(rail, &zeroed));
        let b = signature(read(rail, &oned));
        if a != b {
            moved.push(format!("{name} ({rail:?}): {a} vs {b}"));
        }
    }

    println!("records with unwritten bits: {checked}");
    assert!(
        moved.is_empty(),
        "the model read a bit the serializer never wrote:\n  {}",
        moved.join("\n  ")
    );
}

/// A verdict rendered for comparison against the same record's other fill.
///
/// Nothing parses this. It only has to be faithful and stable, and the resolved
/// operation's own `Debug` is both — it prints every field the model kept, so a
/// field that moved with an unwritten bit shows up as a difference here.
fn signature(verdict: Verdict) -> String {
    match verdict {
        Verdict::Resolved(op) => format!("resolved {op:?}"),
        Verdict::Declined(reason) => format!("declined {reason}"),
        Verdict::WrongShape(reason) => format!("wrong-shape {reason}"),
    }
}

/// Every record Apple produced can be placed in the encoder that emitted it.
///
/// The tests above stop at [`operation`]: they prove a record decodes and
/// resolves. This one carries it one link further and puts it through
/// [`ExecBuilder`], which is the only way a resolved operation ever reaches a
/// transaction — `begin_segment`, `record`, `end_segment`, `finish`.
///
/// That link asks a question resolution cannot. [`ExecBuilder::record`] routes
/// on the operation's own [`OperationClass`]: five classes name exactly one
/// rail and are refused inside any other segment, and the rest are admitted by
/// whichever encoder is open only if that encoder carries the class at all. So
/// an operation whose class disagrees with the encoder class the capture came
/// from is `RailMismatch` here — and it is *nothing* in the tests above,
/// because `resolve::operation` is handed the rail rather than deriving it.
///
/// The rail is the fixture's, from the serializer class that emitted the
/// record. That is the whole point: the model is being asked to admit each
/// record where Apple actually wrote it, not where the model would have
/// preferred it.
///
/// A record the ledger declines is skipped rather than failed, for the same
/// reason it is above — a contract answer is not a defect. What must not
/// happen is a resolved operation the model then refuses to place.
#[test]
fn every_record_apple_produced_is_one_the_model_can_place_where_it_was_written() {
    use reims_vgpu_core::exec::ExecBuilder;
    use reims_vgpu_protocol::segment::{SegmentKind, SegmentLifetime};

    let mut placed = 0usize;
    let mut accesses = 0usize;
    let mut with_accesses = 0usize;
    let mut misplaced = Vec::new();
    let mut per_rail: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();

    for case in cases() {
        let Case {
            name, rail, bytes, ..
        } = case;
        if !is_stream_rail(rail) {
            continue;
        }
        let Ok(view) = op(&bytes, 0) else { continue };
        // Every stream rail has a segment kind — `of_rail` answers `None` only
        // for the root, which `is_stream_rail` already excluded.
        let kind = SegmentKind::of_rail(rail).expect("a stream rail names a segment");

        let mut builder = ExecBuilder::new();
        // The builder's own arenas, because a pass descriptor filed during
        // resolution is the one `ResolvedOperation::participations` reads back.
        let resolver = Recording::new();
        let Ok(op) = operation(rail, &view, &resolver, builder.arenas_mut()) else {
            continue;
        };
        // A registry holding every name this record resolved, so the accesses
        // it declares are placed by the owner that owns names, heaps and
        // content rather than by a stub.
        let mut model = registry_holding(&resolver.seen());
        let mut place = || -> Result<(), String> {
            builder
                .begin_segment(kind.wire_type(), SegmentLifetime::SELF_CONTAINED)
                .map_err(|r| format!("begin: {}", r.reason()))?;
            builder
                .record(op, &mut model.task_access(TASK, DOMAIN))
                .map_err(|r| format!("record: {}", r.reason()))?;
            builder
                .end_segment()
                .map_err(|r| format!("end: {}", r.reason()))?;
            Ok(())
        };
        match place() {
            Ok(()) => {
                let tx = builder
                    .finish()
                    .expect("an encoder that began and ended leaves nothing open");
                assert_eq!(
                    tx.record_count(),
                    1,
                    "{name}: the record was accepted and then not carried"
                );
                // Every access the transaction carries came from this
                // record's own participation, so the two cannot disagree —
                // which they could for as long as the accesses were declared
                // beside the record instead of derived from it.
                for access in &tx.accesses {
                    assert_eq!(
                        access.domain, DOMAIN,
                        "{name}: an access landed in another submission domain"
                    );
                }
                accesses += tx.accesses.len();
                if !tx.accesses.is_empty() {
                    with_accesses += 1;
                }
                placed += 1;
                *per_rail.entry(format!("{rail:?}")).or_default() += 1;
            }
            Err(why) => misplaced.push(format!("{name} ({rail:?}, {kind:?}): {why}")),
        }
    }

    for (rail, count) in &per_rail {
        println!("placed: {rail} x{count}");
    }
    println!("transactions carrying accesses: {with_accesses}, accesses: {accesses}");
    assert!(
        misplaced.is_empty(),
        "records the serializer produced could not be placed in the encoder that wrote them:\n  {}",
        misplaced.join("\n  ")
    );
    assert!(placed > 0, "no fixture reached a transaction at all");
    assert!(
        with_accesses > 0,
        "no record's participation became an access; the derivation reached nothing"
    );
}

/// The task and submission domain every fixture transaction is admitted into.
const TASK: TaskId = TaskId(1);
const DOMAIN: ChannelId = ChannelId(0);

/// A lifecycle owner with a dedicated resource in every slot `seen` names.
///
/// Dedicated rather than heap-placed, because a fixture's refs are the oracle
/// stub's and nothing in a capture says which of them share storage. The extent
/// is generous for the same reason: the stub's offsets are its own, and a
/// window that left the resource would report this test's setup as a model
/// refusal.
fn registry_holding(seen: &std::collections::BTreeSet<u32>) -> Lifecycle {
    let mut model = Lifecycle::new();
    // The effects are the caller's obligation and there are none here: a task
    // definition owes no transfer and frees no storage.
    let _ = model
        .apply(&LifecycleOp::DefineTask {
            directory: reims_vgpu_core::identity::DirectoryFrame(0x1000),
            task: TASK,
            kernel: false,
        })
        .expect("a fresh task");
    for slot in seen {
        let _ = model
            .apply(&LifecycleOp::CreateResource {
                task: TASK,
                slot: ObjectListRef(*slot),
                storage: Storage::Dedicated {
                    backing: BackingId(u64::from(*slot)),
                    extent: ByteRange {
                        offset: 0,
                        length: 1 << 40,
                    },
                },
            })
            .expect("a free slot");
    }
    model
}

/// [`Everything`], with a note of every ref it answered.
struct Recording(std::cell::RefCell<std::collections::BTreeSet<u32>>);

impl Recording {
    fn new() -> Self {
        Self(std::cell::RefCell::new(std::collections::BTreeSet::new()))
    }

    fn seen(&self) -> std::collections::BTreeSet<u32> {
        self.0.borrow().clone()
    }
}

impl RefResolver for Recording {
    fn resource(&self, object_ref: u32) -> Option<ResourceId> {
        self.0.borrow_mut().insert(object_ref);
        Some(ResourceId {
            slot: ObjectListRef(object_ref),
            generation: SlotGeneration(1),
        })
    }
}

/// A draw or a dispatch declares the memory Apple's own bind records named,
/// and only that.
///
/// **The one claim in this file that spans two records.** Everything above puts
/// one captured record through the model and asks what it made of it. A bind
/// makes nothing of itself — it writes a slot and touches no memory — and a
/// draw's own fields name at most an index buffer. What a draw reads is the
/// join of the two, held by the encoder, and it is a join no per-record test
/// can see.
///
/// `drawPrimitives:vertexStart:vertexCount:` and
/// `dispatchThreadgroups:threadsPerThreadgroup:` name no memory at all, so
/// every access their transactions carry came out of the binding table. The entry
/// stride is what this puts at risk: a buffer bind's entries are twelve bytes
/// each and not sixteen, and a model that read them at the wrong stride would
/// bind whatever the next entry's low word spells — which is a resource, so it
/// resolves, and only a fixture that named a *known* ref can tell the two
/// apart.
///
/// Samplers are the same shape and the opposite answer. A sampler bind's
/// entries are a serializer ref exactly as a texture bind's are, and the *only*
/// thing that says one contributes memory and the other does not is which
/// opcode carried it. So the sampler fixtures run through the same join and are
/// required to leave the draw declaring nothing.
///
/// Both encoders, because they are two vocabularies. The render binds carry
/// their stage in the opcode and the compute ones have no stage at all, so the
/// tables they land in are chosen by two separate matches — and a compute bind
/// that reached a render table, or the other way round, is a footprint that
/// silently loses everything the encoder bound.
#[test]
fn a_draw_declares_the_memory_apples_own_bind_records_named() {
    use reims_vgpu_core::exec::ExecBuilder;
    use reims_vgpu_protocol::segment::{SegmentKind, SegmentLifetime};
    use reims_vgpu_wire::ops::render::{
        is_buffer_bind, OPCODE_DRAW, OPCODE_SET_FRAGMENT_SAMPLER, OPCODE_SET_FRAGMENT_SAMPLER_LOD,
        OPCODE_SET_FRAGMENT_TEXTURE, OPCODE_SET_VERTEX_BUFFER_STRIDE, OPCODE_SET_VERTEX_SAMPLER,
        OPCODE_SET_VERTEX_SAMPLER_LOD, OPCODE_SET_VERTEX_TEXTURE,
    };

    /// Whether a bind of this opcode puts memory in a draw's footprint.
    ///
    /// Buffers and textures do; samplers do not. Keyed on the opcode because
    /// that is the only thing that distinguishes them — the entries are the
    /// same layout.
    fn binds_memory(rail: Rail, opcode: u32) -> Option<bool> {
        use reims_vgpu_wire::ops::compute;
        match rail {
            Rail::Render => {
                // The strided vertex bind is a buffer bind that
                // `is_buffer_bind` does not answer for: it is a different
                // opcode with a different entry layout, and it is the only
                // record whose stage is fixed by the API rather than by the
                // opcode pair.
                if is_buffer_bind(opcode)
                    || matches!(
                        opcode,
                        OPCODE_SET_VERTEX_TEXTURE
                            | OPCODE_SET_FRAGMENT_TEXTURE
                            | OPCODE_SET_VERTEX_BUFFER_STRIDE
                    )
                {
                    return Some(true);
                }
                matches!(
                    opcode,
                    OPCODE_SET_VERTEX_SAMPLER
                        | OPCODE_SET_FRAGMENT_SAMPLER
                        | OPCODE_SET_VERTEX_SAMPLER_LOD
                        | OPCODE_SET_FRAGMENT_SAMPLER_LOD
                )
                .then_some(false)
            }
            Rail::Compute => {
                if matches!(
                    opcode,
                    compute::OPCODE_SET_BUFFER
                        | compute::OPCODE_SET_BUFFER_STRIDE
                        | compute::OPCODE_SET_TEXTURE
                ) {
                    return Some(true);
                }
                matches!(
                    opcode,
                    compute::OPCODE_SET_SAMPLER | compute::OPCODE_SET_SAMPLER_LOD
                )
                .then_some(false)
            }
            _ => None,
        }
    }

    /// The record that reads the tables on each rail, and names nothing itself.
    fn trigger_opcode(rail: Rail) -> u32 {
        match rail {
            Rail::Compute => reims_vgpu_wire::ops::compute::OPCODE_DISPATCH_THREADGROUPS,
            _ => OPCODE_DRAW,
        }
    }

    let all = cases();
    let triggers: Vec<&Case> = [Rail::Render, Rail::Compute]
        .into_iter()
        .map(|rail| {
            all.iter()
                .find(|c| {
                    c.rail == rail
                        && op(&c.bytes, 0).is_ok_and(|v| v.opcode() == trigger_opcode(rail))
                })
                .unwrap_or_else(|| panic!("the capture has a direct {rail:?} trigger"))
        })
        .collect();

    let mut checked = 0usize;
    let mut with_memory = 0usize;
    let mut without_memory = 0usize;
    for case in &all {
        let rail = case.rail;
        let Ok(view) = op(&case.bytes, 0) else {
            continue;
        };
        let Some(binds_memory) = binds_memory(rail, view.opcode()) else {
            continue;
        };
        let trigger = triggers
            .iter()
            .find(|t| t.rail == rail)
            .expect("both rails have one");
        let kind = SegmentKind::of_rail(rail).expect("a stream rail names a segment");

        let mut builder = ExecBuilder::new();
        let resolver = Recording::new();
        let Ok(bind) = operation(rail, &view, &resolver, builder.arenas_mut()) else {
            continue;
        };
        let named: std::collections::BTreeSet<u32> =
            resolver.seen().into_iter().filter(|r| *r != 0).collect();
        assert!(
            !named.is_empty(),
            "{}: a bind that resolved no ref",
            case.name
        );
        let trigger_view = op(&trigger.bytes, 0).expect("checked above");
        let drawn = operation(rail, &trigger_view, &resolver, builder.arenas_mut())
            .expect("a draw or dispatch the ledger judged");

        let mut model = registry_holding(&resolver.seen());
        builder
            .begin_segment(kind.wire_type(), SegmentLifetime::SELF_CONTAINED)
            .expect("a segment of the rail's own kind");
        builder
            .record(bind, &mut model.task_access(TASK, DOMAIN))
            .expect("a bind Apple wrote");
        builder
            .record(drawn, &mut model.task_access(TASK, DOMAIN))
            .expect("a draw or dispatch Apple wrote");
        builder.end_segment().expect("the segment ends");
        let tx = builder.finish().expect("nothing left open");

        // `registry_holding` gives slot `n` backing `n`, so the backing a
        // declared access names is the ref the bind resolved.
        let declared: std::collections::BTreeSet<u32> = tx
            .accesses
            .iter()
            .filter_map(|a| match a.key {
                reims_vgpu_core::access::AccessKey::Range(r, _)
                | reims_vgpu_core::access::AccessKey::Subresource(r, _)
                | reims_vgpu_core::access::AccessKey::Whole(r) => u32::try_from(r.backing.0).ok(),
                reims_vgpu_core::access::AccessKey::Heap(_)
                | reims_vgpu_core::access::AccessKey::DomainOnly => None,
            })
            .collect();
        let expected = if binds_memory {
            named.clone()
        } else {
            std::collections::BTreeSet::new()
        };
        assert_eq!(
            declared, expected,
            "{}: the draw declared {declared:?} and the bind named {named:?}",
            case.name
        );
        checked += 1;
        if binds_memory {
            with_memory += 1;
        } else {
            without_memory += 1;
        }
    }
    println!(
        "bind-then-draw fixtures checked: {checked} \
         ({with_memory} naming memory, {without_memory} not)"
    );
    assert!(
        with_memory > 0 && without_memory > 0,
        "the join saw {with_memory} binds that name memory and {without_memory} \
         that do not; both answers have to be reached or one of them is untested"
    );
}

/// A rebind moves an offset and keeps the slot bound, in Apple's own records.
///
/// `setVertexBufferOffset:atIndex:` and its four siblings name **no buffer**:
/// the slot keeps whatever it holds and only the offset — and, for the strided
/// forms, the stride — changes. That makes them the one binding record whose
/// correct handling is invisible to every test above. A model that treated one
/// as a bind would read a buffer ref out of a word that holds an offset and
/// rebind the slot to it; one that treated it as an unbind would leave the next
/// draw declaring nothing. Both are a footprint that disagrees with the
/// encoder, and both resolve without any refusal.
///
/// So: a real bind, then a real rebind, then a real draw, and the draw must
/// still declare exactly what the bind named.
#[test]
fn a_rebind_apple_wrote_keeps_the_slot_the_bind_before_it_filled() {
    use reims_vgpu_core::exec::ExecBuilder;
    use reims_vgpu_protocol::segment::{SegmentKind, SegmentLifetime};
    use reims_vgpu_wire::ops::compute;
    use reims_vgpu_wire::ops::render::{
        is_buffer_bind, OPCODE_DRAW, OPCODE_SET_FRAGMENT_BUFFER_OFFSET,
        OPCODE_SET_VERTEX_BUFFER_OFFSET, OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
    };

    /// Whether this opcode moves an already-bound buffer's offset.
    fn is_rebind(rail: Rail, opcode: u32) -> bool {
        match rail {
            Rail::Render => matches!(
                opcode,
                OPCODE_SET_VERTEX_BUFFER_OFFSET
                    | OPCODE_SET_FRAGMENT_BUFFER_OFFSET
                    | OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE
            ),
            Rail::Compute => matches!(
                opcode,
                compute::OPCODE_SET_BUFFER_OFFSET | compute::OPCODE_SET_BUFFER_OFFSET_STRIDE
            ),
            _ => false,
        }
    }

    /// A buffer bind on the rail, whose slot a rebind can move.
    fn is_bind(rail: Rail, opcode: u32) -> bool {
        match rail {
            Rail::Render => is_buffer_bind(opcode),
            Rail::Compute => opcode == compute::OPCODE_SET_BUFFER,
            _ => false,
        }
    }

    let all = cases();
    let mut checked = 0usize;
    for rail in [Rail::Render, Rail::Compute] {
        let kind = SegmentKind::of_rail(rail).expect("a stream rail names a segment");
        let trigger_opcode = match rail {
            Rail::Compute => compute::OPCODE_DISPATCH_THREADGROUPS,
            _ => OPCODE_DRAW,
        };
        let trigger = all
            .iter()
            .find(|c| c.rail == rail && op(&c.bytes, 0).is_ok_and(|v| v.opcode() == trigger_opcode))
            .expect("a direct draw or dispatch");
        let bind = all
            .iter()
            .find(|c| c.rail == rail && op(&c.bytes, 0).is_ok_and(|v| is_bind(rail, v.opcode())))
            .expect("a buffer bind");

        for case in &all {
            if case.rail != rail {
                continue;
            }
            let Ok(view) = op(&case.bytes, 0) else {
                continue;
            };
            if !is_rebind(rail, view.opcode()) {
                continue;
            }

            let mut builder = ExecBuilder::new();
            let resolver = Recording::new();
            let bind_view = op(&bind.bytes, 0).expect("checked above");
            let Ok(bound) = operation(rail, &bind_view, &resolver, builder.arenas_mut()) else {
                continue;
            };
            let named: std::collections::BTreeSet<u32> =
                resolver.seen().into_iter().filter(|r| *r != 0).collect();
            assert!(!named.is_empty(), "{}: the bind named nothing", bind.name);
            let Ok(rebind) = operation(rail, &view, &resolver, builder.arenas_mut()) else {
                continue;
            };
            // The claim that makes the rest of this test mean something: a
            // rebind resolves no ref of its own, so anything the draw declares
            // after it came from the bind.
            assert_eq!(
                resolver
                    .seen()
                    .into_iter()
                    .filter(|r| *r != 0)
                    .collect::<std::collections::BTreeSet<u32>>(),
                named,
                "{}: a rebind resolved a ref, so it named a buffer",
                case.name
            );
            let trigger_view = op(&trigger.bytes, 0).expect("checked above");
            let drawn = operation(rail, &trigger_view, &resolver, builder.arenas_mut())
                .expect("a draw or dispatch the ledger judged");

            let mut model = registry_holding(&resolver.seen());
            builder
                .begin_segment(kind.wire_type(), SegmentLifetime::SELF_CONTAINED)
                .expect("a segment of the rail's own kind");
            for record in [bound, rebind, drawn] {
                builder
                    .record(record, &mut model.task_access(TASK, DOMAIN))
                    .expect("a record Apple wrote");
            }
            builder.end_segment().expect("the segment ends");
            let tx = builder.finish().expect("nothing left open");

            let declared: std::collections::BTreeSet<u32> = tx
                .accesses
                .iter()
                .filter_map(|a| match a.key {
                    reims_vgpu_core::access::AccessKey::Range(r, _)
                    | reims_vgpu_core::access::AccessKey::Subresource(r, _)
                    | reims_vgpu_core::access::AccessKey::Whole(r) => {
                        u32::try_from(r.backing.0).ok()
                    }
                    reims_vgpu_core::access::AccessKey::Heap(_)
                    | reims_vgpu_core::access::AccessKey::DomainOnly => None,
                })
                .collect();
            assert_eq!(
                declared, named,
                "{}: after the rebind the draw declared {declared:?} and the bind named {named:?}",
                case.name
            );
            checked += 1;
        }
    }
    println!("rebind fixtures checked: {checked}");
    assert!(
        checked > 0,
        "no rebind fixture reached the join; the test proved nothing"
    );
}

/// A record cannot participate in a resource it never named.
///
/// The link after placement. [`ResolvedOperation::participations`] is what
/// turns a resolved stream into the accesses a transaction is built from, and
/// nothing before this asks Apple's own records for them: the per-class methods
/// it aggregates were each reachable only from their own unit tests.
///
/// The invariant checked here is the one an aggregation can get wrong without
/// any test noticing. Every participation names a `ResourceId`, and the only
/// place a `ResourceId` can come from is [`RefResolver::resource`] — so a
/// resource in the answer that the resolver was never asked for is one an arm
/// fabricated, whether from a stale field, a mis-transcribed variant or a copy
/// of the wrong operand. The resolver records what it was asked, and the
/// answer must be a subset of that.
///
/// The counts are printed rather than asserted per class, because which
/// fixtures exist is the oracle's business. What *is* asserted is that some
/// record named memory at all — an aggregation that returned nothing for
/// everything would satisfy the subset rule perfectly.
#[test]
fn a_record_never_participates_in_a_resource_it_did_not_name() {
    let mut fabricated = Vec::new();
    let mut naming = 0usize;
    let mut participations_total = 0usize;
    let mut per_class: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

    for case in cases() {
        let Case {
            name, rail, bytes, ..
        } = case;
        if !is_stream_rail(rail) {
            continue;
        }
        let Ok(view) = op(&bytes, 0) else { continue };
        let resolver = Recording::new();
        let mut arenas = ExecArenas::default();
        let Ok(resolved) = operation(rail, &view, &resolver, &mut arenas) else {
            continue;
        };
        let asked = resolver.seen();

        let mut parts = Vec::new();
        resolved.participations(&arenas, &mut parts);
        if !parts.is_empty() {
            naming += 1;
            participations_total += parts.len();
            *per_class
                .entry(format!("{:?}", resolved.class()))
                .or_default() += parts.len();
        }
        for part in &parts {
            if !asked.contains(&part.resource.slot.0) {
                fabricated.push(format!(
                    "{name} ({rail:?}): participation names ref {} and the record asked for {asked:?}",
                    part.resource.slot.0
                ));
            }
        }
    }

    for (class, count) in &per_class {
        println!("participations: {class} x{count}");
    }
    println!("records naming memory: {naming}, participations: {participations_total}");
    assert!(
        fabricated.is_empty(),
        "a participation named a resource its record never resolved:\n  {}",
        fabricated.join("\n  ")
    );
    assert!(
        naming > 0,
        "no fixture named any memory; the aggregation reached nothing"
    );
}

/// Apple's own segment header frames Apple's own records, and the model walks
/// the result.
///
/// The test above puts each record through `ExecBuilder` by calling
/// `begin_segment`, `record` and `end_segment` by hand. Hand-driving those three
/// is exactly what production cannot do: it is handed *bytes*, and something has
/// to find the segments in them. [`walk::exec`] is that something, and this is
/// the only test that gives it bytes the serializer produced.
///
/// # What is synthesized, and why only that
///
/// One field. A segment header's `length` is written by `-endEncoding`, after
/// the records, so a capture taken at `-beginSegment:` necessarily reads zero
/// there — every header fixture does. Filling it in is supplying the one thing
/// the capture cannot contain; the other seven bytes are Apple's, including the
/// eighth the serializer never writes, which is left exactly as captured
/// precisely so that a walk which read it as a field would answer differently.
///
/// The records are Apple's, unmodified, in the encoder class that emitted them.
///
/// # The claim
///
/// Every record of a rail that resolves on its own is still there after a walk
/// of the whole segment. That is not implied by the per-record test: a walk
/// carries state across records — the cursor's position, the arenas a resolver
/// files into, the accumulating access list — and a record that resolves alone
/// can still be lost to a walk that misplaces the next segment boundary or
/// hands the resolver an arena it then discards.
///
/// # The classes whose only captured header asks to continue something
#[test]
fn apples_own_segment_header_frames_apples_own_records() {
    use reims_vgpu_core::exec::ExecBuilder;
    use reims_vgpu_core::walk;
    use reims_vgpu_protocol::segment::SegmentKind;
    use reims_vgpu_wire::ops::segment::SEGMENT_HEADER_LEN;

    let headers = captured_headers();
    let mut walked = 0usize;
    let mut refused_spanning = 0usize;
    let mut records_placed = 0usize;
    let mut accesses = 0usize;

    for &kind in SegmentKind::ALL {
        let Some(header) = headers.get(&kind) else {
            continue;
        };
        // Every record of this rail that resolves standing alone. Ones the
        // ledger declines are left out: this test is about the walk, and a
        // stream carrying an unjudged opcode refuses as a whole by design.
        let body: Vec<Vec<u8>> = cases()
            .into_iter()
            .filter(|case| case.rail == kind.rail())
            .filter(|case| matches!(read(case.rail, &case.bytes), Verdict::Resolved(_)))
            .map(|case| case.bytes)
            .collect();
        if body.is_empty() {
            continue;
        }

        let continues = reims_vgpu_wire::ops::segment::segment_header(header)
            .expect("a captured header parses")
            .begin_flag
            != 0;
        let mut bytes = header.clone();
        let length = SEGMENT_HEADER_LEN + body.iter().map(Vec::len).sum::<usize>();
        // The one field `-endEncoding` fills in, and the only byte this test
        // writes.
        bytes[..4].copy_from_slice(&(length as u32).to_le_bytes());
        for record in &body {
            bytes.extend_from_slice(record);
        }

        // One resolver for the whole walk, so the registry below holds every
        // name the segment's records between them ask for.
        let resolver = Recording::new();
        for record in &body {
            let Ok(view) = op(record, 0) else { continue };
            let _ = operation(
                kind.rail(),
                &view,
                &resolver,
                &mut reims_vgpu_core::exec::ExecArenas::default(),
            );
        }
        let mut model = registry_holding(&resolver.seen());

        let outcome = walk::exec(
            &bytes,
            &resolver,
            &mut model.task_access(TASK, DOMAIN),
            ExecBuilder::new(),
        );
        if continues {
            assert_eq!(
                outcome.err().map(|r| r.reason()),
                Some("stream_continuation_without_encoder"),
                "{}: a header claiming to continue an encoder found one to continue",
                kind.name()
            );
            refused_spanning += 1;
            continue;
        }
        let tx = outcome.unwrap_or_else(|refusal| {
            panic!(
                "{}: a segment of {} records the model resolves did not walk: {} at {:?}",
                kind.name(),
                body.len(),
                refusal.reason(),
                refusal.site()
            )
        });

        assert_eq!(
            tx.streams.len(),
            1,
            "{}: one segment produced more than one encoder",
            kind.name()
        );
        assert_eq!(
            tx.record_count(),
            body.len(),
            "{}: the walk lost records the per-record path keeps",
            kind.name()
        );
        for access in &tx.accesses {
            assert_eq!(
                access.domain,
                DOMAIN,
                "{}: an access landed in another submission domain",
                kind.name()
            );
        }
        println!(
            "walked: {} x{} records, {} accesses",
            kind.name(),
            tx.record_count(),
            tx.accesses.len()
        );
        records_placed += tx.record_count();
        accesses += tx.accesses.len();
        walked += 1;
    }

    println!(
        "segments walked: {walked}, records: {records_placed}, accesses: {accesses}, \
         headers refused for continuation: {refused_spanning}"
    );
    assert!(walked > 0, "no captured segment header reached the walk");
    assert!(
        refused_spanning > 0,
        "no captured header exercised the unoffered-continuation refusal"
    );
    assert!(
        accesses > 0,
        "the walk derived no access at all; the chain from bytes to a hazard edge is broken"
    );
}

/// The segment header each encoder class wrote, as captured.
///
/// A header is told from an envelope payload by the measured written mask, the
/// same way [`every_boundary_record_apple_wrote_names_a_segment_the_model_has`]
/// tells them apart: a header allocates eight bytes and writes seven, and the
/// payload writes all eight. Headers whose continuation bit is set are left out
/// — see the caller.
fn captured_headers(
) -> std::collections::BTreeMap<reims_vgpu_protocol::segment::SegmentKind, Vec<u8>> {
    use reims_vgpu_protocol::segment::{segment_role, SegmentRole};
    let json = fixtures();
    let mut out = std::collections::BTreeMap::new();
    for case in json["cases"].as_array().expect("cases is an array") {
        let selector = case["selector"].as_str().expect("selector");
        if !is_boundary(selector) {
            continue;
        }
        let mask = unhex(case["written_mask"].as_str().expect("written_mask"));
        if mask.iter().all(|&byte| byte == 0xff) {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer"));
        let header = reims_vgpu_wire::ops::segment::segment_header(&bytes)
            .expect("a captured header parses");
        if let Some(SegmentRole::Encoder(kind)) = segment_role(header.segment_type) {
            // A header that does not ask to continue anything is preferred, and
            // one that does is kept only where the class captured nothing else
            // — see the caller, which walks the first and requires a refusal
            // from the second.
            match out.entry(kind) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(bytes);
                }
                std::collections::btree_map::Entry::Occupied(mut slot) => {
                    let held = reims_vgpu_wire::ops::segment::segment_header(slot.get())
                        .expect("a captured header parses");
                    if held.begin_flag != 0 && header.begin_flag == 0 {
                        slot.insert(bytes);
                    }
                }
            }
        }
    }
    out
}
