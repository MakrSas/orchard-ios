//! The architecture plan's "heap allocations per steady-state draw: 0", turned
//! into a number this suite asserts.
//!
//! # Why a structural zero needs an instrument
//!
//! A structural zero that nothing measures is a claim. The way it stops being
//! true is not a visible regression either: a helper that returns a `Vec` on a
//! per-access path costs one trip into the allocator per access and shows up as
//! a percent or two of drain duty spread evenly across a profile, which is
//! exactly the shape that survives review. Nobody bisects to it, because no
//! single line got slower.
//!
//! So the counts are asserted. A path that has to allocate says how many times
//! it does and why, and a change that adds a trip fails here rather than in a
//! profile six weeks later.
//!
//! # This is an integration test because the library forbids `unsafe`
//!
//! `reims-vgpu-core` carries `#![forbid(unsafe_code)]`, which is a claim about
//! the semantic model worth more than the convenience of measuring from
//! inside it. A `GlobalAlloc` implementation is unavoidably `unsafe`, and an
//! integration test is its own crate — so the instrument lives out here, the
//! model keeps its forbid, and the measurement is taken through the public API
//! a caller would use anyway.
//!
//! The instrument itself is [`reims_vgpu_testkit::allocations`], shared with
//! the rail's own suite: its counter is per thread and off unless a
//! measurement asks, so tests still run in parallel.

use reims_vgpu_core::access::{
    AccessIntent, AccessKey, AccessMode, BackingId, ByteRange, ResourceKey,
};
use reims_vgpu_core::depend::DependencyGraph;
use reims_vgpu_core::identity::{ChannelId, IngressOrdinal};
use reims_vgpu_testkit::allocations::{measure, Counting};

#[global_allocator]
static ALLOCATOR: Counting = Counting::new();

/// The instrument first. A measurement that cannot see a known allocation is
/// worth nothing, and one that counts allocations the body did not make would
/// make every assertion below unfalsifiable.
#[test]
fn the_counter_sees_a_trip_and_sees_only_the_body() {
    let (v, one) = measure(|| {
        let mut v: Vec<u64> = Vec::with_capacity(4);
        v.push(1);
        v
    });
    assert_eq!(v.len(), 1);
    assert_eq!(one, 1, "one reservation, no growth");

    let (_, none) = measure(|| 1 + 1);
    assert_eq!(none, 0, "arithmetic does not allocate");

    let outside: Vec<u64> = (0..8).collect();
    let (_, still_none) = measure(|| 1 + 1);
    assert_eq!(still_none, 0, "work outside a measurement is not counted");
    assert_eq!(outside.len(), 8);

    let (_, grew) = measure(|| {
        let mut v: Vec<u8> = Vec::new();
        for n in 0..64u8 {
            v.push(n);
        }
        v
    });
    assert!(grew > 1, "growth re-enters the allocator: {grew}");
}

fn range(backing: u64, offset: u64, length: u64) -> AccessKey {
    AccessKey::Range(
        ResourceKey {
            backing: BackingId(backing),
            heap: None,
        },
        ByteRange { offset, length },
    )
}

fn intent(key: AccessKey, mode: AccessMode) -> AccessIntent {
    AccessIntent {
        domain: ChannelId(1),
        key,
        mode,
        api_stages: 0,
        input_content_version: None,
        output_content_version: None,
    }
}

/// One draw's worth of accesses admitted into a graph that has already seen
/// the same shape many times over.
///
/// # What "steady state" means here
///
/// A guest that has been drawing for a while re-touches the resources it
/// already touched: the same vertex buffer, the same uniform block, the same
/// sampled texture, frame after frame. The graph's indexes therefore already
/// have a bucket for each backing, already grown to the size that shape needs,
/// and admitting one more of the same draws no new capacity from anywhere.
/// That is the state the plan's zero is about — not the first admission
/// against a cold graph, which legitimately builds the buckets. A drain also
/// compacts, so the warm-up does: without it the indexes grow with the whole
/// history rather than with what is live, and "steady state" would be
/// measuring a graph that is still growing.
///
/// # The claim is that the count does not scale, not that it is zero
///
/// Two trips are structural and named in the code: the wait list `admit` hands
/// back, which its signature owns, and the per-ordinal index bucket a new
/// transaction needs. Neither is per access. So rather than pin an exact
/// figure — which would be a fact about `Vec`'s growth policy as much as about
/// this crate — the test admits two shapes, one four times the size of the
/// other, and asserts the count barely moves. A helper that allocated per
/// access or per comparison would fail that immediately, which is the
/// regression this exists to catch.
fn warm_graph_admission(accesses: &[AccessIntent]) -> usize {
    let mut graph = DependencyGraph::new();
    for n in 0..512u64 {
        let _ = graph.admit(IngressOrdinal(n), accesses);
        if n >= 4 {
            graph.retire(IngressOrdinal(n - 4));
        }
        if n % 32 == 0 {
            graph.compact();
        }
    }
    let (waits, allocations) = measure(|| graph.admit(IngressOrdinal(512), accesses));
    assert!(!waits.is_empty(), "a warm graph has something to wait for");
    allocations
}

fn buffer_accesses(count: u64) -> Vec<AccessIntent> {
    (0..count)
        .map(|n| {
            intent(
                range(n, 0, 256),
                if n % 3 == 0 {
                    AccessMode::Write
                } else {
                    AccessMode::Read
                },
            )
        })
        .collect()
}

#[test]
fn admitting_a_warm_draw_does_not_allocate_per_access() {
    let small = warm_graph_admission(&buffer_accesses(8));
    let large = warm_graph_admission(&buffer_accesses(32));
    assert!(
        small <= 4,
        "{small} trips for eight warm accesses; the structural ones are the \
         returned wait list and the per-ordinal index bucket"
    );
    assert!(
        large <= small + 2,
        "{large} trips for thirty-two accesses against {small} for eight: \
         the cost is scaling with the accesses"
    );
}

/// The same claim for the wider shape: a draw touching a heap-placed resource
/// is reachable through two indexes, so its candidate list is the one most
/// likely to be rebuilt per access.
#[test]
fn a_heap_placed_draw_does_not_allocate_per_candidate_list() {
    use reims_vgpu_core::access::HeapId;

    let heap = HeapId {
        id: 9,
        membership_generation: 1,
    };
    let placed = |count: u64| -> Vec<AccessIntent> {
        (0..count)
            .map(|n| {
                intent(
                    AccessKey::Range(
                        ResourceKey {
                            backing: BackingId(100 + n),
                            heap: Some(heap),
                        },
                        ByteRange {
                            offset: n * 1024,
                            length: 512,
                        },
                    ),
                    if n % 4 == 0 {
                        AccessMode::Write
                    } else {
                        AccessMode::Read
                    },
                )
            })
            .collect()
    };

    let small = warm_graph_admission(&placed(6));
    let large = warm_graph_admission(&placed(24));
    assert!(small <= 4, "{small} trips for six heap-placed accesses");
    assert!(
        large <= small + 2,
        "{large} trips for twenty-four heap-placed accesses against {small} for six"
    );
}

/// The read a warm frame takes: a replica that already holds the bytes.
///
/// This is the overwhelmingly common shape once a frame's resources have been
/// resident for a frame or two, and it is the shape the per-byte freshness
/// representation exists to answer cheaply. Answering it by computing the owed
/// set and finding it empty built a `RangeSet` per read; asking whether the
/// bytes are covered asks the same question and builds nothing.
#[test]
fn a_read_a_replica_already_holds_allocates_nothing() {
    use reims_vgpu_core::access::{ByteRange as Bytes, ContentVersion};
    use reims_vgpu_core::content::{ContentLedger, Replica};

    let backing = BackingId(1);
    let whole = Bytes {
        offset: 0,
        length: 1 << 20,
    };
    let mut ledger = ContentLedger::new();
    ledger.declare(backing, whole, Replica::GuestPages);
    // A frame's worth of scattered device-side production, so the freshness
    // set has real members rather than one.
    for n in 0..64u64 {
        ledger.write(
            backing,
            Bytes {
                offset: n * 4096,
                length: 2048,
            },
            Replica::DeviceOwned,
        );
    }

    let read = Bytes {
        offset: 8192,
        length: 512,
    };
    let (answer, allocations) =
        measure(|| ledger.transfer_for_read(backing, read, Replica::DeviceOwned));
    assert!(answer.is_none(), "the device wrote these bytes itself");
    assert_eq!(
        allocations, 0,
        "a read of bytes the replica already holds builds nothing"
    );

    // And the version query beside it, which a planner asks for the same read.
    let (version, none) = measure(|| ledger.version_of(backing, read));
    assert!(version.is_some());
    assert_eq!(
        none, 0,
        "asking which version covers a range builds nothing"
    );
    assert_ne!(ledger.newest_version(backing), Some(ContentVersion(0)));
}

/// The read that does owe a transfer still says what it owes, and the cost of
/// saying so does not scale with how fragmented the backing is.
#[test]
fn a_read_that_owes_a_transfer_pays_for_the_answer_and_not_for_the_search() {
    use reims_vgpu_core::access::ByteRange as Bytes;
    use reims_vgpu_core::content::{ContentLedger, Replica};

    let backing = BackingId(2);
    let build = |pieces: u64| {
        let mut ledger = ContentLedger::new();
        ledger.declare(
            backing,
            Bytes {
                offset: 0,
                length: 1 << 22,
            },
            Replica::GuestPages,
        );
        for n in 0..pieces {
            ledger.write(
                backing,
                Bytes {
                    offset: n * 4096,
                    length: 2048,
                },
                Replica::DeviceOwned,
            );
        }
        ledger
    };

    let read = Bytes {
        offset: 0,
        length: 1 << 16,
    };
    let mut few = build(16);
    let mut many = build(256);
    let (owed_few, cost_few) =
        measure(|| few.transfer_for_read(backing, read, Replica::GuestPages));
    let (owed_many, cost_many) =
        measure(|| many.transfer_for_read(backing, read, Replica::GuestPages));
    assert!(owed_few.is_some() && owed_many.is_some());
    assert!(
        cost_many <= cost_few + 4,
        "{cost_many} trips over 256 pieces against {cost_few} over 16: the \
         search is allocating per member"
    );
}

/// Appending records to one transaction's builder does not allocate per
/// record.
///
/// # What this can and cannot claim
///
/// A transaction owns its records: [`ExecWork`] hands `Vec`s to whoever takes
/// the transaction, so a builder starting empty must grow them, and that
/// growth is `O(log n)` trips for `n` records. What must never happen is a
/// trip *per record* — a scratch buffer rebuilt per operation, a participation
/// list allocated and dropped, a per-record `Vec` inside a resolved operation.
/// `ExecBuilder::record` already takes its participation scratch out and puts
/// it back for exactly that reason, and this is what checks that it still
/// does.
///
/// So the assertion is the growth law: quadrupling the record count may add a
/// couple of doublings per vector and nothing more.
#[test]
fn appending_records_to_a_transaction_does_not_allocate_per_record() {
    use reims_vgpu_core::access::{AccessRefusal, Participation, ResourceKey};
    use reims_vgpu_core::blit::BlitOp;
    use reims_vgpu_core::exec::{ExecBuilder, ResolvedOperation};
    use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
    use reims_vgpu_core::stream::{SegmentKind, SegmentLifetime};

    fn id(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn everything(p: &Participation) -> Result<AccessIntent, AccessRefusal> {
        Ok(p.resolve(
            ChannelId(1),
            ResourceKey {
                backing: BackingId(u64::from(p.resource.slot.0)),
                heap: None,
            },
            None,
            None,
        ))
    }

    fn run(records: u32) -> usize {
        let mut builder = ExecBuilder::new();
        builder
            .begin_encoder(
                SegmentKind::Blit,
                SegmentLifetime {
                    continues_previous: false,
                    continues_into_next: false,
                },
            )
            .expect("an encoder may open");
        let ((), allocations) = measure(|| {
            for n in 0..records {
                let op = ResolvedOperation::Blit(BlitOp::BufferToBuffer {
                    source: id(n % 8),
                    source_offset: 0,
                    dest: id(8 + n % 8),
                    dest_offset: 0,
                    size: 256,
                });
                builder
                    .record(op, &mut everything)
                    .expect("a blit in a blit encoder");
            }
        });
        allocations
    }

    // The slope, measured twice a decade apart rather than once. "Fewer than
    // sixty-four trips for sixty-four records" reads as "not one per record"
    // and is not: the fixed cost is three, so that bound had sixty trips of
    // headroom and a cost of one trip per four records would have satisfied
    // it.
    let (four, sixteen) = (run(4), run(16));
    let (small, large) = (run(64), run(256));
    assert!(
        sixteen - four <= 6,
        "four records cost {four} and sixteen cost {sixteen}: four times the \
         records for {} more trips",
        sixteen - four
    );
    assert!(
        large - small <= 8,
        "64 records cost {small} and 256 cost {large}: four times the records \
         for {} more trips",
        large - small
    );
    assert!(
        run(1) <= 8,
        "{} trips to append one blit to a fresh builder",
        run(1)
    );
}

/// A frame through the present stream with the previous one still in flight.
///
/// A device that does not stall between frames acquires the next image while
/// the last one is still queued, completes it, and then queues the new one —
/// FIFO allows exactly that much overlap and no more, since a frame may only
/// be queued at the head of the order. The overlap is the point: the in-flight
/// list is non-empty at every acquire, and an implementation that collects the
/// used image indexes into a `Vec` per acquire looks free when nothing is in
/// flight (an empty iterator does not allocate) and costs a trip per frame the
/// moment the pipeline fills.
///
/// Nothing on this path is proportional to anything. The in-flight list is at
/// most the returned image count long and its entries are plain values, so a
/// trip into the allocator here is a per-frame heap cost with nothing to buy
/// it.
///
/// `queue` returns the frames a superseding order dropped, which is a `Vec` its
/// signature owns; under FIFO it is always empty, and an empty `Vec` does not
/// allocate. FIFO is also the steady state of every pathway that ships.
#[test]
fn a_frame_through_an_overlapped_present_stream_allocates_nothing() {
    use reims_vgpu_core::present::{Order, PresentStream, Ticket};

    let mut stream = PresentStream::new(Order::Fifo);
    stream.configure(3, 3);

    fn cycle(stream: &mut PresentStream, queued: &mut Option<Ticket>) {
        let ticket = stream.acquire().expect("a free image");
        stream.ready(&ticket).expect("drawn");
        if let Some(previous) = queued.take() {
            stream.complete(&previous).expect("shown");
        }
        assert!(
            stream.queue(&ticket).expect("at the head").is_empty(),
            "nothing is superseded under FIFO"
        );
        *queued = Some(ticket);
    }

    let mut queued = None;
    for _ in 0..16 {
        cycle(&mut stream, &mut queued);
    }
    assert_eq!(
        stream.in_flight(),
        1,
        "a frame is still queued when the next one is acquired"
    );

    let ((), allocations) = measure(|| {
        for _ in 0..16 {
            cycle(&mut stream, &mut queued);
        }
    });
    assert_eq!(
        allocations, 0,
        "sixteen overlapped frames over three images"
    );
}

/// The whole ingress walk, from EXEC bytes to a finished transaction.
///
/// `appending_records_to_a_transaction_does_not_allocate_per_record` measures
/// the builder alone: it hands `ExecBuilder::record` an already-resolved
/// operation, so it never touches the step before it. What that step does is
/// turn a wire record into a `ResolvedOperation`, and the records that carry a
/// counted list — a bind of eight vertex buffers is the ordinary one — have
/// somewhere to put it. That somewhere is [`ExecArenas`], and the arenas exist
/// for exactly this reason: a resolver that built a `Vec` per counted record
/// would cost a trip into the allocator on every bind in every frame, and the
/// builder-only measurement cannot see it.
///
/// So the walk is driven whole. The assertion is the growth law rather than a
/// zero: a fresh builder per EXEC starts with empty vectors and its first
/// records have to grow them, which is a per-*EXEC* cost the plan's per-draw
/// zero does not forbid. Quadrupling the records may add a couple of doublings
/// per vector and nothing more; anything per record shows up as a factor of
/// four.
#[test]
fn walking_an_exec_does_not_allocate_per_record() {
    use reims_vgpu_core::access::{AccessRefusal, Participation, ResourceKey};
    use reims_vgpu_core::exec::ExecBuilder;
    use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
    use reims_vgpu_core::resolve::RefResolver;
    use reims_vgpu_core::walk;
    use reims_vgpu_protocol::segment::{SegmentKind, SegmentLifetime};
    use reims_vgpu_wire::ops::segment::SEGMENT_HEADER_LEN;

    /// Every ref names a live resource, so the walk reaches the builder rather
    /// than refusing on the first name.
    struct Everything;

    impl RefResolver for Everything {
        fn resource(&self, object_ref: u32) -> Option<ResourceId> {
            Some(ResourceId {
                slot: ObjectListRef(object_ref),
                generation: SlotGeneration(1),
            })
        }
    }

    fn everything(p: &Participation) -> Result<AccessIntent, AccessRefusal> {
        Ok(p.resolve(
            ChannelId(1),
            ResourceKey {
                backing: BackingId(u64::from(p.resource.slot.0)),
                heap: None,
            },
            None,
            None,
        ))
    }

    fn framed(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let length = (reims_vgpu_protocol::decode::OP_HEADER_LEN + payload.len()) as u32;
        out.extend_from_slice(&opcode.to_le_bytes());
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// `setVertexBuffers:offsets:withRange:` over eight slots — the counted
    /// list the arenas exist for. The entry stride is twelve: a ref and a
    /// `u64` offset.
    fn bind_eight() -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&8u32.to_le_bytes());
        for buffer in 1..=8u32 {
            payload.extend_from_slice(&buffer.to_le_bytes());
            payload.extend_from_slice(&0u64.to_le_bytes());
        }
        framed(
            reims_vgpu_wire::ops::render::OPCODE_SET_VERTEX_BUFFER,
            &payload,
        )
    }

    /// `drawPrimitives:vertexStart:vertexCount:`, which names no memory of its
    /// own, so what it declares came from the slots the bind above filled.
    fn draw() -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&3u32.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&3u16.to_le_bytes());
        framed(reims_vgpu_wire::ops::render::OPCODE_DRAW, &payload)
    }

    fn stream(pairs: u32) -> Vec<u8> {
        let mut records = Vec::new();
        for _ in 0..pairs {
            records.push(bind_eight());
            records.push(draw());
        }
        let body: usize = records.iter().map(Vec::len).sum();
        let mut out = Vec::new();
        out.extend_from_slice(&((SEGMENT_HEADER_LEN + body) as u32).to_le_bytes());
        out.push(SegmentKind::Render.wire_type());
        let lifetime = SegmentLifetime::SELF_CONTAINED;
        out.push(u8::from(lifetime.continues_previous));
        out.push(u8::from(lifetime.continues_into_next));
        // The byte the serializer never writes.
        out.push(0xaa);
        for r in records {
            out.extend_from_slice(&r);
        }
        out
    }

    let cost = |pairs: u32| -> usize {
        // The bytes are built outside the measurement: what is under test is
        // the walk over them, not the fixture that wrote them.
        let bytes = stream(pairs);
        let (work, allocations) =
            measure(|| walk::exec(&bytes, &Everything, &mut everything, ExecBuilder::new()));
        let work = work.expect("every ref resolves and the encoder closes");
        assert_eq!(
            work.record_count(),
            2 * pairs as usize,
            "every record reached the transaction, so the walk really ran"
        );
        allocations
    };

    // The marginal cost, measured twice a decade apart. Anything per record
    // shows up here as a factor of four; amortised vector growth shows up as
    // a couple of doublings.
    let (one, eight) = (cost(1), cost(8));
    let (small, large) = (cost(32), cost(128));
    assert!(
        eight - one <= 8,
        "one pair costs {one} and eight cost {eight}: eight times the records \
         for {} more trips",
        eight - one
    );
    assert!(
        large - small <= 8,
        "32 pairs cost {small} and 128 cost {large}: four times the records \
         for {} more trips",
        large - small
    );

    // And the fixed cost, named rather than left implicit --- it is what the
    // two bounds above are differences of, and a bound written only as
    // "fewer than one per record" would have been satisfied by it alone.
    //
    // Every vector the builder and the arenas carry starts empty, so the
    // first record that reaches each one allocates it. That is why walking a
    // single pair already costs a dozen-odd trips and walking a hundred and
    // twenty-eight costs barely more.
    //
    // The window moved down when `SlotTable` stopped storing a slot per index
    // the guest reached: a table is one vector rather than two, and marking a
    // draw's declarations is a write over the bound slots rather than a
    // `clear` and a `resize` that reallocates.
    //
    // It is per EXEC rather than per draw, so it is outside the plan's zero.
    // What would remove it is a recycled builder: `walk::exec` consumes one
    // and `ExecWork` does not hand its storage back, so every EXEC starts
    // cold. That door is deliberately not opened here --- there is no
    // production caller to size a pool against, and a pool with no caller is
    // a guess. Pinning the number is what makes adding one a visible
    // improvement and losing the arenas a visible regression.
    assert!(
        (12..=24).contains(&one),
        "{one} trips to walk one bind and one draw through a fresh builder"
    );
}
