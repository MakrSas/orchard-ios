use reims_vgpu_wire::device_desc::BackingBuilder;

use super::*;
use crate::model::{DeviceId, StorageIncarnation, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::protocol::endian::{ld32, st16, st32, st64};
use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
use crate::protocol::iosurface_pages::DEVICE_DESC_PLANE_COUNT;
use crate::runtime::decode::resource::SERIALIZER_OBJECT_SAMPLER;
use crate::runtime::host::FakeHost;

#[test]
fn mapper_ref_texture_fail_latch_dedups_per_task_ref_and_rearms_on_clear() {
    // Flood guard for the per-draw-per-ref resolve path: a genuinely-broken
    // mapper-ref-texture ref logs each reason once, isolates per (task,ref), and
    // re-arms on resolve. Unique ids so this never races real refs across
    // the process-global latch.
    let (t, r, r2) = (0xAB01u32, 0xCD01u32, 0xCD02u32);
    clear_mapper_ref_texture_fail(t, r);
    clear_mapper_ref_texture_fail(t, r2);
    let seen = |task: u32, rf: u32, reason: &'static str| {
        mapper_ref_texture_fail_latch()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&(task, rf, reason))
    };
    note_mapper_ref_texture_fail(t, r, "mapper_ref_texture_register", "x".into());
    assert!(seen(t, r, "mapper_ref_texture_register"));
    // Distinct reason on the same ref tracked independently.
    note_mapper_ref_texture_fail(t, r, "mapper_ref_texture_desc_read", "x".into());
    assert!(seen(t, r, "mapper_ref_texture_desc_read"));
    // A different ref is untouched.
    assert!(!seen(t, r2, "mapper_ref_texture_register"));
    note_mapper_ref_texture_fail(t, r2, "mapper_ref_texture_register", "x".into());
    // Clearing r re-arms only r, leaves r2.
    clear_mapper_ref_texture_fail(t, r);
    assert!(!seen(t, r, "mapper_ref_texture_register"));
    assert!(!seen(t, r, "mapper_ref_texture_desc_read"));
    assert!(seen(t, r2, "mapper_ref_texture_register"));
    clear_mapper_ref_texture_fail(t, r2);
}

fn setup_task_with_list(host: &mut FakeHost, state: &mut DeviceState) {
    // Same 1-level map as gva_mem test: GVA page 0 → data pfn 4.
    let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
    let root_gpa = 3u64 << PAGE_SHIFT_ARM64E;
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    host.map_range(data_gpa, 0x200, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);

    state.define_task(1, 0x1000, 2);
    // list base GVA 0 (pfn field 0 allowed)
    assert!(state.set_object_list(1, 0, 8));
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 11u32 | (0x20u32 << 8));
    entry[4..12].copy_from_slice(&0x40u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 12, &entry);
    let mut desc = [0u8; 0x20];
    st32(&mut desc[0..], 9);
    st16(&mut desc[0x16..], 0x50);
    st32(&mut desc[0x18..], 64);
    st32(&mut desc[0x1c..], 32);
    let _ = host.write_gpa(data_gpa + 0x40, &desc);
}

#[test]
fn resolve_mapper_ref_texture_from_list() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    // Sanity: list entry readable
    let e = lookup_list_entry(&state, &host, 1, 1).expect("list entry");
    assert_eq!(e.object_type, 11);
    assert_eq!(e.descriptor_gva, 0x40);
    let mid = resolve_mapper_ref_texture(&mut state, &host, 1, 1).expect("mapper_ref_texture");
    assert_eq!(mid, 9);
    let m = state.mappings.get(&9).unwrap();
    assert!(m.has_geom);
    assert_eq!((m.width, m.height, m.format), (64, 32, 0x50));
}

/// Registering a mapper-ref-texture is construction, not bind-time repair.
///
/// Once the task owns the texture object, later binds retrieve that object and
/// must not replay its serialized descriptor over mutable mapping state. A new
/// descriptor can take effect only after the resource lifetime ends.
#[test]
fn a_retained_mapper_ref_texture_runs_construction_side_effects_once() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let resource = resolve_resource(&state, &host, 1, 1).expect("construction");

    assert_eq!(
        resolve_mapper_ref_texture_resource(&mut state, 1, 1, &resource),
        Some(9)
    );
    {
        let mapping = state.mappings.get_mut(&9).expect("registered mapping");
        mapping.width = 17;
        mapping.height = 19;
        mapping.format = 0x71;
    }

    assert_eq!(
        resolve_mapper_ref_texture_resource(&mut state, 1, 1, &resource),
        Some(9)
    );
    let mapping = &state.mappings[&9];
    assert_eq!(
        (mapping.width, mapping.height, mapping.format),
        (17, 19, 0x71),
        "a warm bind must not replay immutable construction input"
    );
}

/// Physical replacement is the event that re-arms backing resolution for a
/// retained texture. A warm bind accepts the already-latched page plan; once
/// invalidation clears that plan, the same bind must enter the resolver again
/// rather than treating object retention as proof that old pages remain live.
#[test]
fn a_texture_bind_reuses_backing_until_physical_invalidation() {
    let host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(9));
    {
        let mapping = state.mappings.get_mut(&9).expect("surface mapping");
        mapping.mapped = true;
        mapping.has_geom = true;
        mapping.width = 64;
        mapping.height = 32;
        mapping.page_entries = vec![0x1234_5001];
    }

    assert!(ensure_surface_for_texture_bind(&mut state, &host, 9));
    assert!(state.invalidate_mapping_pages(9));
    assert!(
        !ensure_surface_for_texture_bind(&mut state, &host, 9),
        "the invalidated page plan must be rebuilt before the texture binds"
    );
}

/// A list entry and descriptor are construction input for a resource, not
/// mutable bind-time state.
///
/// Moving the task's object list changes where future resources are
/// constructed from; it does not retarget an object that is already live. An
/// explicit delete ends that lifetime, and reusing the reference constructs a
/// new object from the then-current descriptor.
#[test]
fn resources_keep_construction_input_until_explicit_delete() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;

    let first = resolve_resource(&state, &host, 1, 1).expect("first construction");
    assert_eq!(ld32(&first.descriptor), 9);

    // Rewrite the descriptor and move the list somewhere unreadable. Neither
    // operation changes the already-registered object.
    let _ = host.write_gpa(data_gpa + 0x40, &10u32.to_le_bytes());
    assert!(state.set_object_list(1, 0xdead, 8));
    let retained = resolve_resource(&state, &host, 1, 1).expect("registered object");
    assert!(Arc::ptr_eq(&first, &retained));
    assert_eq!(ld32(&retained.descriptor), 9);
    assert!(matches!(
        retained.decoded(),
        Ok(
            crate::runtime::decode::resource::Descriptor::IOSurfaceTexture {
                mapping_id: 9,
                width: 64,
                height: 32,
                ..
            }
        )
    ));
    assert_eq!(
        resolve_mapper_ref_texture_resource(&mut state, 1, 1, &retained),
        Some(9),
        "the retained typed object resolves after its construction bytes become unreadable"
    );

    // Delete and reuse is the lifecycle edge that permits the same reference
    // to name a newly-constructed resource.
    assert!(state.delete_object(1, 1));
    assert!(state.set_object_list(1, 0, 8));
    let replacement = resolve_resource(&state, &host, 1, 1).expect("replacement construction");
    assert!(!Arc::ptr_eq(&first, &replacement));
    assert_eq!(ld32(&replacement.descriptor), 10);
    assert_eq!(
        resolve_mapper_ref_texture_resource(&mut state, 1, 1, &replacement),
        Some(10),
        "the replacement lifetime runs its own construction side effects"
    );
}

#[test]
fn task_lifetime_retires_all_of_its_resource_objects() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let resource = resolve_resource(&state, &host, 1, 1).expect("construction");
    assert!(state.constructed_object(1, 1).is_some());

    assert!(state.delete_task(1));
    assert!(state.constructed_object(1, 1).is_none());
    assert_eq!(
        ld32(&resource.descriptor),
        9,
        "an outstanding host owner remains valid"
    );
}

#[test]
fn the_resource_registry_accepts_exactly_the_resource_constructor_types() {
    let accepted: Vec<u8> = (0..=u8::MAX)
        .filter(|&object_type| object_type_is_resource(object_type))
        .collect();
    assert_eq!(accepted, [1, 2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15]);
}

/// Serializer state has its own lifetime. A `DeleteResource`-scoped registry
/// must not retain its descriptor and hide a later update.
#[test]
fn non_resource_descriptors_are_read_again() {
    use crate::runtime::decode::resource::OBJECT_TYPE_SERIALIZER_OBJECT;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let mut entry = [0u8; 12];
    st32(
        &mut entry,
        u32::from(OBJECT_TYPE_SERIALIZER_OBJECT) | (4u32 << 8),
    );
    entry[4..].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 24, &entry);
    let _ = host.write_gpa(data_gpa + 0x80, &1u32.to_le_bytes());

    let (_, first) = resolve_descriptor(&state, &host, 1, 2, &[OBJECT_TYPE_SERIALIZER_OBJECT])
        .expect("first serializer descriptor");
    assert_eq!(ld32(&first), 1);
    assert!(state.constructed_object(1, 2).is_none());

    let _ = host.write_gpa(data_gpa + 0x80, &2u32.to_le_bytes());
    let (_, second) = resolve_descriptor(&state, &host, 1, 2, &[OBJECT_TYPE_SERIALIZER_OBJECT])
        .expect("updated serializer descriptor");
    assert_eq!(ld32(&second), 2);
    assert!(state.constructed_object(1, 2).is_none());
}

fn put_sampler_object(host: &mut FakeHost, ref_: u32, descriptor_gva: u64, lod_min: f32) {
    use crate::runtime::decode::resource::OBJECT_TYPE_SERIALIZER_OBJECT;
    use reims_vgpu_wire::ops::sampler::NEW_SAMPLER_TOTAL_LEN;

    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry,
        u32::from(OBJECT_TYPE_SERIALIZER_OBJECT) | (NEW_SAMPLER_TOTAL_LEN << 8),
    );
    st64(&mut entry[4..], descriptor_gva);
    let _ = host.write_gpa(
        data_gpa + u64::from(ref_) * OBJECT_LIST_ENTRY_LEN as u64,
        &entry,
    );

    let mut descriptor = vec![0u8; NEW_SAMPLER_TOTAL_LEN as usize];
    st32(&mut descriptor, SERIALIZER_OBJECT_SAMPLER);
    st32(&mut descriptor[4..], NEW_SAMPLER_TOTAL_LEN);
    st32(&mut descriptor[8..], ref_);
    st32(&mut descriptor[12..], 0x8400_0000);
    st32(&mut descriptor[20..], lod_min.to_bits());
    let _ = host.write_gpa(data_gpa + descriptor_gva, &descriptor);
}

#[test]
fn sampler_construction_is_retained_until_its_own_explicit_delete() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    put_sampler_object(&mut host, 2, 0x80, 1.25);

    let first = resolve_sampler_state(&state, &host, 1, 2).expect("first sampler");
    assert_eq!(first.descriptor.lod_min_clamp, 1.25);

    // Neither mutable descriptor bytes nor a moved object-list pointer mutate
    // an already-constructed sampler object.
    put_sampler_object(&mut host, 2, 0x80, 7.5);
    assert!(state.set_object_list(1, 0xdead, 8));
    let retained = resolve_sampler_state(&state, &host, 1, 2).expect("retained sampler");
    assert!(Arc::ptr_eq(&first, &retained));
    assert_eq!(retained.descriptor.lod_min_clamp, 1.25);

    // The sampler API's delete edge, not resource deletion, permits ref reuse.
    assert!(state.task_sampler_states.delete(1, 2));
    assert!(state.set_object_list(1, 0, 8));
    let replacement = resolve_sampler_state(&state, &host, 1, 2).expect("replacement sampler");
    assert!(!Arc::ptr_eq(&first, &replacement));
    assert_eq!(replacement.descriptor.lod_min_clamp, 7.5);
}

/// Constructing a sampler names its slot, because the destroy that ends its
/// life arrives when nothing else can.
///
/// A sampler is built out of the guest's object list once and answered from
/// `task_sampler_states` forever after, and the guest clears the list slot
/// before it sends `CmdDeleteObject`. So the destroy resolves against neither a
/// cached construction nor a live entry — a driven macos-15 boot found the
/// semantic model holding a name for **zero of 1984** of them. A retirement the
/// model cannot name is one it cannot be told about.
///
/// The name is taken at construction because that is the only moment the list
/// entry and the model's namespace are both available.
#[test]
fn constructing_a_sampler_names_its_slot_so_its_destroy_can_still_be_resolved() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    put_sampler_object(&mut host, 2, 0x80, 1.25);

    assert_eq!(
        state.object_name(1, 2),
        None,
        "nothing has asked what this slot names yet"
    );
    let _ = resolve_sampler_state(&state, &host, 1, 2).expect("first sampler");
    let named = state
        .object_name(1, 2)
        .expect("construction names the slot it constructed from");

    // The guest clears the slot, which is what it does before the destroy. The
    // name outlives the entry, which is the whole point.
    let cleared = [0u8; OBJECT_LIST_ENTRY_LEN];
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let _ = host.write_gpa(data_gpa + 2 * OBJECT_LIST_ENTRY_LEN as u64, &cleared);
    assert_eq!(
        lookup_list_entry(&state, &host, 1, 2),
        None,
        "the fixture must really have cleared the entry, or this proves nothing"
    );
    assert_eq!(
        super::name_resource(&state, &host, 1, 2),
        Some(named),
        "the production ref resolver still answers, so the destroy has a \
         resource to name"
    );
}

/// Naming at construction is a name that outlives the guest's list entry, and
/// it is taken for every object type rather than for resources alone.
///
/// The kinds this door is called for are the ones a driven boot measured
/// arriving at `CmdDeleteObject` with no name the model could hold — depth
/// stencil 4, function 5, compute pipeline 3, against about 2160 destroys — and
/// each is named at its own construction site because there is no shared one.
/// What every site depends on is this: the name is taken while the entry is
/// live, and it answers afterwards, because the guest clears the slot before it
/// sends the destroy.
///
/// The fixture uses a sampler's object-list entry because the entry is the
/// whole input — `name_resource` names a sampler, a function and a pipeline
/// state by their slots exactly as it names a texture, and a door that behaved
/// differently per type would be the bug this asserts against.
#[test]
fn naming_at_construction_outlives_the_guests_list_entry() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    put_sampler_object(&mut host, 3, 0x100, 0.5);

    assert_eq!(state.object_name(1, 3), None);
    super::note_named_at_construction(&state, &host, 1, 3, "named", "unnamed");
    let named = state
        .object_name(1, 3)
        .expect("the door names the slot it was called for");

    // What the guest does before every destroy this population is about.
    let cleared = [0u8; OBJECT_LIST_ENTRY_LEN];
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let _ = host.write_gpa(data_gpa + 3 * OBJECT_LIST_ENTRY_LEN as u64, &cleared);
    assert_eq!(
        lookup_list_entry(&state, &host, 1, 3),
        None,
        "the fixture must really have cleared the entry, or this proves nothing"
    );
    assert_eq!(
        super::name_resource(&state, &host, 1, 3),
        Some(named),
        "so the destroy has a resource to name and does not have to be refused"
    );

    // A slot with no entry at all cannot be named even at construction time,
    // and the door says so rather than inventing one — that is what the
    // `unnamed` route counts.
    super::note_named_at_construction(&state, &host, 1, 5, "named", "unnamed");
    assert_eq!(state.object_name(1, 5), None);
}

#[test]
fn failed_sampler_construction_is_not_retained_and_can_retry() {
    use crate::runtime::decode::resource::OBJECT_TYPE_SERIALIZER_OBJECT;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let mut short_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut short_entry,
        u32::from(OBJECT_TYPE_SERIALIZER_OBJECT) | (4 << 8),
    );
    st64(&mut short_entry[4..], 0x80);
    let _ = host.write_gpa(data_gpa + 24, &short_entry);
    let _ = host.write_gpa(data_gpa + 0x80, &SERIALIZER_OBJECT_SAMPLER.to_le_bytes());

    assert!(matches!(
        resolve_sampler_state(&state, &host, 1, 2),
        Err(SamplerResolveError::Decode { .. })
    ));
    assert!(state.task_sampler_states.get(1, 2).is_none());

    put_sampler_object(&mut host, 2, 0x80, 3.0);
    let sampler = resolve_sampler_state(&state, &host, 1, 2).expect("published retry");
    assert_eq!(sampler.descriptor.lod_min_clamp, 3.0);
}

#[test]
fn task_teardown_retires_sampler_objects_without_touching_outstanding_owners() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    put_sampler_object(&mut host, 2, 0x80, 2.0);
    let sampler = resolve_sampler_state(&state, &host, 1, 2).expect("sampler");

    assert!(state.delete_task(1));
    assert!(state.task_sampler_states.get(1, 2).is_none());
    assert_eq!(sampler.descriptor.lod_min_clamp, 2.0);
}

/// The backing decoder refuses a descriptor it cannot decode as declared, and
/// says which check refused.
///
/// All three of these bounds are correct — IOSurface caps `getPlaneCount` at
/// eight, a plane record the blob does not reach cannot be decoded, and the
/// device descriptor's `allocSize` really is 32 bits.
///
/// Naming them was the first half. The second is that the first two must not
/// publish a *partial* surface, which is what they did: truncating to the cap
/// handed every later reader a surface that simply has eight planes, and
/// defaulting an unreachable record handed them a 0x0 plane at pitch 0 in a slot
/// the guest declared. Neither reads as a decode failure downstream — both are
/// well-formed surfaces — so the loss appears as a layer that samples blank,
/// which is what content that is genuinely empty also looks like. `None` reaches
/// the caller's `backing_fail reason=desc_decode`, which names the surface
/// id.
///
/// Fails without the fix: both decodes return `Some`.
#[test]
fn the_backing_decoder_refuses_what_it_cannot_decode_and_reports_why() {
    // Twelve planes against IOSurface's own ceiling of eight.
    let over_cap = BackingBuilder::new(0x1000, 0x100, 0x4247_5241, 12).with_len(0x24); // 'BGRA'
                                                                                       // A legal
                                                                                       // plane count
                                                                                       // whose
                                                                                       // records the
                                                                                       // blob does
                                                                                       // not reach:
                                                                                       // `with_len`
                                                                                       // stops after
                                                                                       // plane 0, so
                                                                                       // planes 1..=3
                                                                                       // are declared
                                                                                       // and
                                                                                       // unreachable.
    let short_records = BackingBuilder::new(0x1000, 0x100, 0x4247_5241, 4).with_len(0x24);

    reset_backing_decode_drops();
    let cap = crate::observe::FailCapture::start();
    assert!(
        decode_backing(over_cap.bytes()).is_none(),
        "a plane count past IOSurface's own ceiling is a malformed descriptor, \
         and there is no correct prefix of it to publish"
    );
    let over = cap
        .lines()
        .into_iter()
        .find(|l| l.contains("reason=plane_count_over_cap"))
        .expect("an over-cap plane count must be reported");
    assert!(
        over.contains("declared=12") && over.contains("cap=8"),
        "the line must name what the guest asked for and what the device holds: {over}"
    );

    // Same reason twice is one line — the latch is what keeps a per-surface
    // stream from flooding the always-on channel. The *refusal* still applies
    // every time; only the line is deduped.
    let cap2 = crate::observe::FailCapture::start();
    assert!(
        decode_backing(over_cap.bytes()).is_none(),
        "the latch must not turn the second refusal into an acceptance"
    );
    assert!(
        cap2.lines()
            .iter()
            .all(|l| !l.contains("reason=plane_count_over_cap")),
        "a repeat must not spend a second line: {:?}",
        cap2.lines()
    );

    // A declared plane whose record the blob does not reach.
    reset_backing_decode_drops();
    let cap3 = crate::observe::FailCapture::start();
    assert!(
        decode_backing(short_records.bytes()).is_none(),
        "a declared plane the blob does not reach must refuse the surface, \
         not publish a 0x0 plane in its slot"
    );
    let short = cap3
        .lines()
        .into_iter()
        .find(|l| l.contains("reason=plane_record_short"))
        .expect("an unreachable plane record must be reported");
    assert!(
        short.contains("plane=1"),
        "the line names the first plane that could not be reached: {short}"
    );

    // A surface larger than the 32-bit `allocSize` field can express.
    reset_backing_decode_drops();
    let big = BackingBuilder::new((u32::MAX as u64) + 1, 0x100, 0x4247_5241, 1)
        .plane(0, 0, 64, 32, 256, 0);
    let surf = decode_backing(big.bytes()).expect("backing decodes");
    let cap4 = crate::observe::FailCapture::start();
    let _ = synthesize_device_desc_from_backing(&surf);
    let sat = cap4
        .lines()
        .into_iter()
        .find(|l| l.contains("reason=alloc_size_over_u32"))
        .expect("a length the 32-bit allocSize cannot hold must be reported");
    assert!(sat.contains("length=4294967296"), "{sat}");
}

#[test]
fn decode_backing_plane0() {
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 0x100); // backing pfn
    st32(&mut desc[0xc..], 0x4247_5241); // 'BGRA'
    desc[0x10] = 1;
    st32(&mut desc[0x14..], 0); // plane offset
    st32(&mut desc[0x18..], 64);
    st32(&mut desc[0x1c..], 32);
    st32(&mut desc[0x20..], 256); // bpr
    let s = decode_backing(&desc).expect("backing");
    assert_eq!(s.length, 0x1000);
    assert_eq!(s.backing_pfn, 0x100);
    assert_eq!((s.width, s.height, s.bytes_per_row), (64, 32, 256));
    assert_eq!(s.plane_count, 1);
    assert_eq!(s.planes[0].offset, 0);
    assert!(!backing_is_multiplanar(&s));
    assert_eq!(
        iosurface_pixel_format_to_mtl(s.pixel_format),
        crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM
    );
}

/// An `'l10r'` render surface reaches a colour attachment, end to end.
///
/// The bug class: a guest game creates its render surface as a single-plane
/// `kCVPixelFormatType_ARGB2101010LEPacked` IOSurface, this table answered 0 for
/// it, and 0 out of this table is `draw::render_target`'s
/// `rt_backing_base_format` refusal — so every draw of every frame failed and the
/// window was black. Two tables had to answer for that to stop, and a test on
/// either one alone would have passed while the window stayed black, so this
/// walks the whole chain the resolve walks.
///
/// The geometry is the surface measured on the boot that found it: width 1280 at
/// a 5120-byte row is four bytes a texel, which is what the packed word is, and
/// it is the reading that says this FourCC is not a multi-plane media format
/// wearing a colour name.
#[test]
fn an_l10r_surface_resolves_as_a_packed_ten_bit_colour_attachment() {
    use crate::protocol::pixel_format as pf;

    const FOURCC_L10R: u32 = 0x6c31_3072;
    let built =
        BackingBuilder::new(0x384000, 0x100, FOURCC_L10R, 1).plane(0, 0, 1280, 720, 5120, 0);
    let surf = decode_backing(built.bytes()).expect("backing decodes");
    assert!(
        !backing_is_multiplanar(&surf),
        "a packed single-plane colour surface must not read as biplanar"
    );

    // Step one: the FourCC names a Metal format at all. A zero here is the
    // refusal the resolve turns into a dropped colour attachment.
    let mtl = iosurface_pixel_format_to_mtl(surf.pixel_format);
    assert_eq!(
        mtl,
        pf::MTL_FORMAT_BGR10A2_UNORM,
        "'l10r' must name BGR10A2Unorm"
    );

    // Step two: that format is one this device will render into, which is what
    // `translate::pixel::color_attachment` derives its answer from. Both steps
    // were missing and either one alone leaves the frame black.
    assert_eq!(
        pf::render_target_bpp(mtl),
        Some(4),
        "a named format this device will not render into is still a black window"
    );
    // Step three: the frame can be landed in the guest's own pages by a copy
    // that converts nothing, which is the only lossless route for a format whose
    // channels do not sit on byte boundaries.
    assert_eq!(
        pf::store_texel_order(mtl),
        Some(pf::TexelLayout::Bgr10a2Unorm)
    );
    // The row stride the surface declared is the tight stride for this format at
    // this width, so the reading above is the surface's own and not an
    // assumption about what a packed word costs.
    assert_eq!(
        pf::tight_row_bytes(surf.width, mtl),
        Some(surf.bytes_per_row)
    );
}

#[test]
fn fourcc_420f_not_bgra_and_multiplanar() {
    assert_eq!(iosurface_pixel_format_to_mtl(IOSURFACE_FOURCC_420F), 0);
    assert_eq!(iosurface_pixel_format_to_mtl(IOSURFACE_FOURCC_420V), 0);
    assert!(iosurface_fourcc_is_biplanar(IOSURFACE_FOURCC_420F));
    // Unknown FourCC must not invent BGRA.
    assert_eq!(iosurface_pixel_format_to_mtl(0xdead_beef), 0);
}

/// A small value is not an MTLPixelFormat ordinal in disguise.
///
/// The converter used to return `pixel_format as u16` for anything at or
/// below 0x200, deciding which encoding the field was in from how big the
/// number was. Every caller passes a backing `pixelFormat` (+0x0c), which is
/// an IOSurface OSType and therefore never below `'    '` (0x20202020), so
/// a small value arriving here is a bad read — and passing it through
/// published a format the guest never named. Fail closed instead, which is
/// what this function already does for every FourCC it does not know.
#[test]
fn a_small_value_is_not_read_as_an_mtl_ordinal() {
    // 0x50 is MTLPixelFormatBGRA8Unorm. As a backing OSType it is nonsense,
    // and the old magnitude test would have handed it back as a format.
    assert_eq!(iosurface_pixel_format_to_mtl(0x50), 0);
    assert_eq!(iosurface_pixel_format_to_mtl(0x200), 0);
    // Known FourCCs are unaffected — this is the boundary the old test sat
    // on, not a narrowing of what the converter accepts.
    assert_eq!(
        iosurface_pixel_format_to_mtl(0x4247_5241),
        crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM
    );
}

#[test]
fn decode_backing_biplanar_420f_planes() {
    // Wire: plane0 Y 1024×1024 bpr=1024 bpe=1; plane1 UV 512×512 bpr=1024 bpe=2.
    // Live boot: fmt='420f' len=0x180000 plane0 bpr=1024.
    let mut desc = vec![0u8; 0x14 + 2 * 0x10];
    st64(&mut desc[0..], 0x180000);
    st32(&mut desc[8..], 0x200);
    st32(&mut desc[0xc..], IOSURFACE_FOURCC_420F);
    desc[0x10] = 2;
    // plane0
    st32(&mut desc[0x14..], 0); // offset
    st32(&mut desc[0x18..], 1024);
    st32(&mut desc[0x1c..], 1024);
    st32(&mut desc[0x20..], 1024 | (1 << 24)); // bpr | bpe<<24
                                               // plane1
    st32(&mut desc[0x24..], 1024 * 1024); // offset after Y
    st32(&mut desc[0x28..], 512);
    st32(&mut desc[0x2c..], 512);
    st32(&mut desc[0x30..], 1024 | (2 << 24));
    let s = decode_backing(&desc).expect("backing 420f");
    assert!(backing_is_multiplanar(&s));
    assert_eq!(s.plane_count, 2);
    assert_eq!(
        (
            s.planes[0].width,
            s.planes[0].height,
            s.planes[0].bytes_per_row
        ),
        (1024, 1024, 1024)
    );
    assert_eq!(s.planes[0].bytes_per_element, 1);
    assert_eq!(
        (
            s.planes[1].width,
            s.planes[1].height,
            s.planes[1].bytes_per_element
        ),
        (512, 512, 2)
    );
    let dev = synthesize_device_desc_from_backing(&s);
    assert_eq!(dev[DEVICE_DESC_PLANE_COUNT], 2);
    use crate::protocol::iosurface_pages::{
        decode_device_surface, mapping_span_bound, sample_window_from_device_desc,
        DEVICE_DESC_PIXEL_FORMAT,
    };
    assert_eq!(
        ld32(&dev[DEVICE_DESC_PIXEL_FORMAT..]),
        IOSURFACE_FOURCC_420F
    );
    let surf = decode_device_surface(&dev).expect("device");
    assert_eq!(surf.plane_count, 2);
    assert_eq!(surf.alloc_size, 0x180000);
    // Mapper-ref-texture Y plane: R8 1024×1024 matches plane0 (contract geometry key).
    let y = sample_window_from_device_desc(
        Some(&dev),
        None,
        crate::protocol::pixel_format::MTL_FORMAT_R8_UNORM,
        1024,
        1024,
    )
    .expect("Y window");
    assert_eq!(y.0, 0); // offset
    assert_eq!(y.1, 1024); // bpr
                           // UV plane: RG8 half res.
    let uv = sample_window_from_device_desc(
        Some(&dev),
        None,
        crate::protocol::pixel_format::MTL_FORMAT_RG8_UNORM,
        512,
        512,
    )
    .expect("UV window");
    assert_eq!(uv.0, 1024 * 1024);
    assert_eq!(uv.1, 1024);
    // A full 1024² BGRA matches no plane record, so it binds nothing, and
    // its page-sizing estimate still rejects on the wire allocation.
    assert!(sample_window_from_device_desc(
        Some(&dev),
        None,
        crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        1024,
        1024,
    )
    .is_none());
    assert!(mapping_span_bound(
        Some(&dev),
        crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        1024,
        1024,
    )
    .is_none());
}

/// A failed page-table walk is not an address. The device used to answer it
/// with the backing *virtual* address used as a physical one whenever that
/// number happened to be RAM, which put a fabricated PFN into
/// `page_entries` — the list every later reader and writer resolves
/// through. Here the walk cannot resolve the backing GVA and the identity
/// candidate *is* mapped RAM, so the old path would have accepted it.
#[test]
fn resolve_backing_refuses_to_substitute_the_gva_when_the_walk_fails() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    // The identity candidate is backed RAM: `read_gpa` succeeds on it, which
    // is the whole of what the old gate checked.
    host.map_range(0x20u64 << PAGE_SHIFT_X86, 0x2000, 0x5a);
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x200, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    // root[0] carries the object list and descriptors. root[0x20] — the
    // backing GVA page — is left unmapped, so the backing walk fails.
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    state.define_task(1, 0x1000, 2);
    assert!(state.set_object_list(1, 0, 8));
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 0x20); // backing GVA page — unmapped in this task
    st32(&mut desc[0xc..], 0x50);
    desc[0x10] = 1;
    st32(&mut desc[0x18..], 16);
    st32(&mut desc[0x1c..], 16);
    st32(&mut desc[0x20..], 64);
    let _ = host.write_gpa(data_gpa + 0x80, &desc);

    assert!(
        !resolve_backing(&mut state, &host, 3),
        "an untranslatable backing must not resolve"
    );
    // The refusal happens before any mutation, so no fabricated entry is
    // left behind for a later writer to aim at.
    let fabricated = state
        .mappings
        .get(&3)
        .map(|m| m.mapped || !m.page_entries.is_empty())
        .unwrap_or(false);
    assert!(!fabricated, "refusal must not cache a fabricated backing");
}

/// `resolve_backing_ex` probes task 0 first and returns on the first
/// task whose backing applies. The identity guess made task 0 succeed for
/// surfaces it could not translate, so the search stopped there and the
/// owning task was never tried — the surface was then backed by an address
/// derived from a virtual one. Refusing is what lets the loop continue.
///
/// Both tasks list the surface, as task 0 (the kernel/global list) and the
/// owner do in production; only the owner can translate the backing.
#[test]
fn the_task_search_reaches_the_owner_when_task_zero_cannot_translate() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    let dir0_gpa = 2u64 << PAGE_SHIFT_X86;
    let root0_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    let dir1_gpa = 7u64 << PAGE_SHIFT_X86;
    let root1_gpa = 8u64 << PAGE_SHIFT_X86;
    let real_page = 9u64 << PAGE_SHIFT_X86;
    for (gpa, len) in [
        (dir0_gpa, 0x20),
        (root0_gpa, 0x1000),
        (data_gpa, 0x200),
        (dir1_gpa, 0x20),
        (root1_gpa, 0x1000),
        (real_page, 0x1000),
    ] {
        host.map_range(gpa, len, 0);
    }
    // The identity candidate for the backing GVA is RAM, so the old path
    // would have taken it on task 0 rather than moving on.
    host.map_range(0x20u64 << PAGE_SHIFT_X86, 0x1000, 0x5a);

    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir0_gpa, &d);
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 8);
    let _ = host.write_gpa(dir1_gpa, &d);
    // Both roots reach the object list at GVA 0; only task 1's maps the
    // backing GVA page 0x20, and it maps it to `real_page`.
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root0_gpa, &d[..4]);
    let _ = host.write_gpa(root1_gpa, &d[..4]);
    st32(&mut d[..4], 9);
    let _ = host.write_gpa(root1_gpa + 0x20 * 4, &d[..4]);

    state.define_task(0, 0x1000, 2);
    assert!(state.set_object_list(0, 0, 8));
    state.define_task(1, 0x1000, 7);
    assert!(state.set_object_list(1, 0, 8));

    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 0x20);
    st32(&mut desc[0xc..], 0x50);
    desc[0x10] = 1;
    st32(&mut desc[0x18..], 16);
    st32(&mut desc[0x1c..], 16);
    st32(&mut desc[0x20..], 64);
    let _ = host.write_gpa(data_gpa + 0x80, &desc);

    assert!(
        resolve_backing(&mut state, &host, 3),
        "the owning task can translate the backing, so the resolve must succeed"
    );
    let m = state.mappings.get(&3).unwrap();
    assert_eq!(m.page_entries.len(), 1);
    assert_eq!(
        entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
        Some(real_page),
        "the backing must come from the task that could translate it, \
         not from task 0's untranslatable GVA"
    );
}

/// The search stops on the first task that can back a surface, so whether
/// that choice was ever a choice is the thing to count. Nothing on the wire
/// can verify a candidate — the object-list entry carries no identity and
/// the backing descriptor is fully decoded — so the claimant count is the
/// only available reading of the search's exposure, and it has to
/// distinguish "one task lists this id" from "two do".
#[test]
fn a_surface_id_claimed_by_two_tasks_is_counted_as_two() {
    // Two tasks, each with its own directory and root, both listing eight
    // object slots at GVA 0. Task 0's list page holds a backing record at
    // slot 3; task 1's holds a ref-texture there until the second half of the
    // test rewrites it.
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    let dir0_gpa = 2u64 << PAGE_SHIFT_X86;
    let root0_gpa = 3u64 << PAGE_SHIFT_X86;
    let list0_gpa = 4u64 << PAGE_SHIFT_X86;
    let dir1_gpa = 7u64 << PAGE_SHIFT_X86;
    let root1_gpa = 8u64 << PAGE_SHIFT_X86;
    let list1_gpa = 9u64 << PAGE_SHIFT_X86;
    for (gpa, len) in [
        (dir0_gpa, 0x20),
        (root0_gpa, 0x1000),
        (list0_gpa, 0x200),
        (dir1_gpa, 0x20),
        (root1_gpa, 0x1000),
        (list1_gpa, 0x200),
    ] {
        host.map_range(gpa, len, 0);
    }

    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir0_gpa, &d);
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 8);
    let _ = host.write_gpa(dir1_gpa, &d);
    // Each task's GVA page 0 reaches its own list page.
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root0_gpa, &d[..4]);
    st32(&mut d[..4], 9);
    let _ = host.write_gpa(root1_gpa, &d[..4]);

    state.define_task(0, 0x1000, 2);
    assert!(state.set_object_list(0, 0, 8));
    state.define_task(1, 0x1000, 7);
    assert!(state.set_object_list(1, 0, 8));

    // Slot 3 of task 0 is the surface. Both entries carry a descriptor GVA
    // and length, which is what `lookup_list_entry` requires before the type
    // is even looked at.
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], OBJECT_TYPE_BACKING as u32 | (0x30u32 << 8));
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(list0_gpa + 3 * 12, &entry);

    // Task 1 lists a *different object type* at the same slot, so it is not
    // a claimant even though the slot is populated.
    let mut other = [0u8; 12];
    st32(
        &mut other[0..],
        OBJECT_TYPE_REF_TEXTURE as u32 | (0x30u32 << 8),
    );
    other[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(list1_gpa + 3 * 12, &other);

    assert_eq!(
        backing_claimant_tasks(&state, &host, 3),
        vec![0],
        "a populated slot of another object type is not a claim on this id"
    );

    // Now task 1 lists a backing record at the same slot. The id spaces are
    // per task, so this is a second, unrelated surface wearing the same id —
    // and the search would have to break the tie by probe order alone.
    let _ = host.write_gpa(list1_gpa + 3 * 12, &entry);
    assert_eq!(
        backing_claimant_tasks(&state, &host, 3),
        vec![0, 1],
        "both tasks list a backing record at slot 3, so both are claimants"
    );

    // An inactive task cannot be the one the search stops on, so it is not
    // counted either.
    state.tasks[1].active = false;
    assert_eq!(
        backing_claimant_tasks(&state, &host, 3),
        vec![0],
        "an inactive task is not a claimant"
    );
}

/// Force-resolve must rebuild the cached page table when the task PT
/// translation of the backing GVA moved (same surface id, same geometry,
/// new physical pages — the early-boot FB vs WindowServer reallocation).
#[test]
fn resolve_backing_force_rebuilds_when_task_translation_moves() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    let old_page = 5u64 << PAGE_SHIFT_X86;
    let new_page = 6u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x200, 0);
    host.map_range(old_page, 0x1000, 0x11);
    host.map_range(new_page, 0x1000, 0x22);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    // root[0] = data page (object list + descriptors), root[1] = old backing.
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    st32(&mut d[..4], 5);
    let _ = host.write_gpa(root_gpa + 4, &d[..4]);
    state.define_task(1, 0x1000, 2);
    assert!(state.set_object_list(1, 0, 8));
    // Backing entry at surface_id=3, descriptor at GVA 0x80.
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 1); // backing_pfn = GVA page 1
    st32(&mut desc[0xc..], 0x50);
    desc[0x10] = 1;
    st32(&mut desc[0x18..], 16);
    st32(&mut desc[0x1c..], 16);
    st32(&mut desc[0x20..], 64);
    let _ = host.write_gpa(data_gpa + 0x80, &desc);

    assert!(resolve_backing(&mut state, &host, 3));
    {
        let m = state.mappings.get(&3).unwrap();
        assert_eq!(m.page_entries.len(), 1);
        assert_eq!(
            entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
            Some(old_page)
        );
        assert_eq!(m.map_generation, 1);
    }
    // Guest remaps GVA page 1 onto a new physical page (same id/geometry).
    st32(&mut d[..4], 6);
    let _ = host.write_gpa(root_gpa + 4, &d[..4]);
    assert!(resolve_backing_force(&mut state, &host, 3));
    {
        let m = state.mappings.get(&3).unwrap();
        assert_eq!(
            entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
            Some(new_page),
            "force-resolve must follow the moved translation"
        );
        assert_eq!(m.map_generation, 2, "page move bumps map_generation");
    }
    // Unchanged translation: force keeps the table without a rebuild.
    assert!(resolve_backing_force(&mut state, &host, 3));
    let m = state.mappings.get(&3).unwrap();
    assert_eq!(m.map_generation, 2);
    assert_eq!(
        entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
        Some(new_page)
    );
}

/// A genuine backing failure (a surface whose descriptor decoded fine but
/// whose page-backing construction fails) must be fail-visible with a
/// `reason=` slug, deduped per `(surface_id, reason)`, and re-armed when the
/// surface next backs cleanly — never a silent `return false` that paints
/// stale/black with no log. Locks the backing blind-spot closure.
#[test]
fn apply_backing_fail_latches_reason_and_rearms() {
    let host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    // A surface_id other backing tests do not touch (they use 3).
    let sid = 11u32;
    clear_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
    assert!(!backing_fail_latch()
        .lock()
        .unwrap()
        .contains_key(&(sid, "task_inactive")));
    // Small valid length (page_count = 1) so the alloc-guard passes, then an
    // undefined/inactive task_id hits the `task_inactive` site — the drain
    // race where a decoded surface's owning task died before backing landed.
    let surf = BackingRecord {
        length: 0x1000,
        backing_pfn: 0x20,
        pixel_format: 0,
        plane_count: 1,
        planes: [BackingPlane::default(); TYPE4_PLANE_CAP],
        width: 16,
        height: 16,
        bytes_per_row: 64,
    };
    assert!(!apply_backing(&mut state, &host, 5, sid, &surf));
    assert!(
        !backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "task_inactive")),
        "one task's probe is not a backing failure: the search has other \
         tasks to try, and reporting here is what put `reason=translate` \
         lines under surfaces that then backed cleanly"
    );
    // The search running out of tasks is what turns the probe's reason into
    // a reported failure.
    flush_backing_fail(sid);
    assert!(
        backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "task_inactive")),
        "an exhausted search must report the first probe's reason slug"
    );
    // A clean backing on the same surface re-arms the latch.
    clear_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
    assert!(
        !backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "task_inactive")),
        "clear_backing_fail must re-arm so a later failure logs again"
    );
}

/// A refusal that the next attach resolves is reported as the recovery it is,
/// and only when the backing that landed is the one the refusal named.
///
/// `backing_fail reason=translate` reads as lost guest work and usually is
/// not: `st=zero-pfn` means the guest had not finished mapping when the
/// per-present path walked it, and the refusal exists so the device asks again
/// rather than substituting a guess. Every one of the six on a driven boot
/// recovered, within 1-21 ms, and nothing in the log said so — see
/// [`super::clear_backing_fail`].
///
/// The match is on the **backing address**, never on `surface_id`: ids recycle
/// within a boot and across geometries, so a clean attach on a recycled id must
/// re-arm the latch (it does, as the test above locks) without claiming that the
/// earlier, different surface recovered.
/// A refusal leaves the latch three ways, and all three are now countable.
///
/// Recovered, superseded, or still there. The third is the only one that can be
/// lost guest work, and before `backing_superseded` and
/// `backing_outstanding` it was indistinguishable from the second: a
/// driven boot with five `backing_fail` lines and four
/// `backing_recovered` lines gave a reader no way to tell a refusal the
/// guest walked away from apart from one that never came back, short of
/// hand-matching backing GVAs across the log. That is how this gap was found.
///
/// The identity is what makes the census readable, so it is what this asserts:
/// every refusal that leaves the latch is accounted for by exactly one of the
/// three, and the residue is the census line's `n`.
#[test]
fn every_backing_refusal_leaves_the_latch_recovered_superseded_or_counted() {
    use crate::runtime::drain::store_route_count;

    // Ids no other test in this module uses; the latch is process-global.
    let sid = 0x4d2u32;
    let gva = 0x4222000u64;
    clear_backing_fail(sid, gva);

    // Superseded: refused at `gva`, backed somewhere else. Not a recovery, and
    // it used to be the silent one.
    let before = store_route_count("backing_superseded");
    let recovered_before = store_route_count("backing_recovered");
    defer_backing_fail(sid, "translate", Some(gva), "backing_fail probe".into());
    flush_backing_fail(sid);
    clear_backing_fail(sid, gva + 0x1000);
    assert_eq!(
        store_route_count("backing_superseded"),
        before + 1,
        "a refusal dropped because the surface backed elsewhere must be counted"
    );
    assert_eq!(
        store_route_count("backing_recovered"),
        recovered_before,
        "and must not be claimed as a recovery"
    );

    // Recovered: refused at `gva`, backed at `gva`. Counts on the other route
    // and leaves the superseded count alone — the two must not double-count one
    // refusal, which is what would make the identity stop holding.
    let before = store_route_count("backing_superseded");
    let recovered_before = store_route_count("backing_recovered");
    defer_backing_fail(sid, "translate", Some(gva), "backing_fail probe".into());
    flush_backing_fail(sid);
    clear_backing_fail(sid, gva);
    assert_eq!(
        store_route_count("backing_recovered"),
        recovered_before + 1,
        "an attach on the backing the refusal named is a recovery"
    );
    assert_eq!(
        store_route_count("backing_superseded"),
        before,
        "and is not also a supersede"
    );
}

/// The census names an outstanding refusal, and says nothing when there is none.
///
/// `oldest_ms` is the field that distinguishes a retry caught mid-flight from a
/// surface this device never backed, so a line without it would report the same
/// thing in both cases — which is the state this replaced.
#[test]
fn the_outstanding_census_names_the_oldest_refusal_and_is_otherwise_silent() {
    let sid = 0x4d3u32;
    let gva = 0x4333000u64;
    clear_backing_fail(sid, gva);

    // Other tests in this module share the latch, so assert about *this* sid
    // rather than about emptiness — a bare `is_none()` would be order-dependent.
    let mine = |line: &Option<String>| {
        line.as_deref()
            .is_some_and(|l| l.contains(&format!("sid={sid}")))
    };
    assert!(
        !mine(&backing_outstanding_census()),
        "nothing is latched for this surface yet"
    );

    defer_backing_fail(sid, "translate", Some(gva), "backing_fail probe".into());
    flush_backing_fail(sid);
    let line = backing_outstanding_census().expect("a latched refusal must be censused");
    assert!(
        line.starts_with("backing_outstanding n=") && line.contains("oldest_ms="),
        "the line must carry both the count and the age: {line}"
    );
    assert!(
        line.contains(&format!("gva={gva:#x}")) || !mine(&Some(line.clone())),
        "when this surface is the oldest, the line must name its backing: {line}"
    );

    // Retiring it removes it from the census, by either route.
    clear_backing_fail(sid, gva);
    assert!(
        !mine(&backing_outstanding_census()),
        "a recovered refusal is no longer outstanding"
    );
}

/// A retried refusal and an abandoned one must not read the same.
///
/// The distinction the census exists to make, and the one it could not make for
/// two sessions. A surface the device asks for every frame and is refused every
/// frame is losing guest work; a surface it asked for once and never again is
/// one the guest stopped presenting. Both sit in the latch as `n=1`.
///
/// The trap this pins: `note_backing_fail` refreshes its timestamp on a repeat, so
/// `oldest_ms` alone reads **backwards** — a live retry holds it near zero and
/// an abandoned refusal lets it grow with the clock. `attempts` is what makes
/// the line state which of the two it is without anyone re-deriving that.
#[test]
fn a_retried_backing_refusal_counts_its_attempts_and_an_abandoned_one_does_not() {
    let sid = 0x4d3u32;
    let gva = 0x4188000u64;
    clear_backing_fail(sid, gva);
    let is_mine = |line: &Option<String>| {
        line.as_ref()
            .is_some_and(|l| l.contains(&format!("sid={sid} ")))
    };

    // Asked once and refused.
    defer_backing_fail(sid, "translate", Some(gva), "first refusal".into());
    flush_backing_fail(sid);
    let line = backing_outstanding_census().expect("a latched refusal is censused");
    if is_mine(&Some(line.clone())) {
        assert!(
            line.contains("attempts=1"),
            "one refusal is one attempt: {line}"
        );
        assert!(
            line.contains("since_last_ms=") && line.contains("oldest_ms="),
            "both ages travel, or `attempts` cannot be placed in time: {line}"
        );
    }

    // Asked again and refused again, four more times. The fail channel stays
    // quiet — this is the per-present path and one line a frame would flood it
    // — so the count is the only thing saying the device is still trying.
    for _ in 0..4 {
        defer_backing_fail(sid, "translate", Some(gva), "retry refusal".into());
        flush_backing_fail(sid);
    }
    let line = backing_outstanding_census().expect("still latched");
    if is_mine(&Some(line.clone())) {
        assert!(
            line.contains("attempts=5"),
            "four retries after the first must be counted, not deduped away: {line}"
        );
    }

    clear_backing_fail(sid, gva);
    assert!(
        !is_mine(&backing_outstanding_census()),
        "the latch re-arms once the surface backs"
    );
}

#[test]
fn a_backing_refusal_the_next_attach_resolves_is_reported_as_recovered() {
    fn log_mark() -> usize {
        crate::observe::redirect_logs_for_tests();
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len()
    }
    fn log_since(mark: usize) -> String {
        let body = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
        body[mark.min(body.len())..].to_string()
    }

    // A surface id no other test in this module uses.
    let sid = 0x4d1u32;
    let gva = 0x4112000u64;
    clear_backing_fail(sid, gva);

    // A reported refusal naming this backing...
    let mark = log_mark();
    defer_backing_fail(sid, "translate", Some(gva), "backing_fail probe".into());
    flush_backing_fail(sid);
    assert!(
        log_since(mark).contains("backing_fail probe"),
        "the exhausted search must report the probe's reason"
    );

    // ...that a later attach on a *different* backing must not claim.
    let mark = log_mark();
    clear_backing_fail(sid, gva + 0x1000);
    assert!(
        !log_since(mark).contains("backing_recovered"),
        "a recycled surface id is not evidence that the earlier backing landed"
    );
    assert!(
        !backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "translate")),
        "the latch must still re-arm, or a later genuine failure goes unlogged"
    );

    // The same refusal, then an attach on the backing it named, is a recovery.
    let mark = log_mark();
    defer_backing_fail(sid, "translate", Some(gva), "backing_fail probe".into());
    flush_backing_fail(sid);
    clear_backing_fail(sid, gva);
    let log = log_since(mark);
    assert!(
        log.contains("backing_recovered")
            && log.contains(&format!("sid={sid}"))
            && log.contains("reason=translate")
            && log.contains(&format!("gva={gva:#x}")),
        "a refusal whose backing then landed must say so:\n{log}"
    );
}

/// A refused walk must say **which** of the walk's checks refused.
///
/// The walk distinguishes fifteen refusals and this rail reported one word,
/// `translate`, for all of them — so "the guest has not filled in this leaf
/// PTE yet" and "this device could not read the task root at all" produced
/// identical log lines while wanting opposite responses. Both halves are
/// locked here: the walk names its failing check, and the detail line
/// carries that name verbatim.
///
/// The fixture maps GVA page 0 and nothing else, so the *same* task walks
/// clean for one address and refuses for the next. Asserting the clean case
/// too is what keeps this from passing vacuously: a fixture in which every
/// walk fails would satisfy the refusal assertions on its own.
#[test]
fn a_refused_backing_walk_names_the_check_that_refused() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let task = state.tasks.get(1).expect("fixture defines task 1");

    // Control: the address the fixture does map walks all the way down.
    let mapped = gva_mem::diagnose_task_slot(&host, task, 1, 0, PAGE_SHIFT_ARM64E);
    assert!(
        mapped.contains("st=ok"),
        "fixture must be able to translate, got {mapped:?}"
    );

    // The case the rig produces: a backing whose leaf entry the guest has
    // not written. Page 1 shares the fixture's root and has no PTE.
    let gva = 1u64 << PAGE_SHIFT_ARM64E;
    let walk = gva_mem::diagnose_task_slot(&host, task, 1, gva, PAGE_SHIFT_ARM64E);
    assert!(
        walk.contains("st=zero-pfn"),
        "an unwritten leaf must be reported as zero-pfn, got {walk:?}"
    );
    assert!(
        walk.contains("lvl=") && walk.contains("idx="),
        "the refusal must name where in the walk it stopped, got {walk:?}"
    );

    let line = backing_translate_fail_detail(202, 1, 0, 640, gva, &walk);
    assert!(line.contains("reason=translate"), "{line}");
    assert!(line.contains("sid=202"), "{line}");
    assert!(line.contains("page=0/640"), "{line}");
    assert!(
        line.contains(&format!("walk=[{walk}]")),
        "the refusal must carry the walk diagnosis verbatim, got {line}"
    );
}

/// A refused object-list entry read names the three inputs its address came
/// from, not just the address.
///
/// `gva_mem`'s own refusal can only print the gva, because it is generic over
/// every caller. Here the gva is derived — `(list_pfn << page_shift) +
/// ref * entry_len` — and a driven x86 boot emits ten of these all reading
/// `gva=0x11b0`, which is `pfn = 1, ref = 36` and is equally consistent with
/// the guest not having mapped its list yet and with this device resolving a
/// ref against the wrong task. The address alone cannot separate those; the
/// inputs can, which is why they have to be on the line.
///
/// Asserts the fields rather than the prose, so rewording the parenthetical
/// does not fail it.
#[test]
fn a_refused_object_list_entry_names_the_geometry_behind_its_address() {
    let task = crate::model::TaskEntry {
        active: true,
        length: 0x1000,
        directory_pfn: 2,
        object_list_pfn: 1,
        object_list_count: 64,
    };
    let entry_gva = (1u64 << PAGE_SHIFT_X86) + 36 * OBJECT_LIST_ENTRY_LEN as u64;
    let line = list_entry_unreadable_detail(3, 36, &task, entry_gva);

    assert!(line.contains("task=3"), "{line}");
    assert!(line.contains("ref=36"), "{line}");
    assert!(line.contains("gva=0x11b0"), "{line}");
    assert!(
        line.contains("list_pfn=1"),
        "the pfn the address was built from must be on the line: {line}"
    );
    assert!(
        line.contains("list_count=64"),
        "the count that admitted this ref must be on the line: {line}"
    );
    assert!(
        line.contains("entry_len=12"),
        "the stride the offset was scaled by must be on the line: {line}"
    );
}

/// A task the guest has defined but never given an object list to must
/// resolve **nothing** — not another task's list.
///
/// This reproduces, at unit scale, what the rail was measured doing on every
/// boot. `TaskEntry::define` used to invent `object_list_pfn = 1` and
/// `count = 0x100000`, so a task with no `SetObjectList` still computed an
/// entry address of `0x1000 + off`. Nothing is mapped there for that task,
/// the walk failed `gva_zero_pfn`, and `read_task_gva_by_id` then walked
/// task `5 >> 1 == 2`'s page table at the same address — where task 2's
/// object list genuinely lives — and decoded task 2's entry as task 5's.
///
/// Task 2's own lookup is asserted first so the fixture is known to be real:
/// a test where the donor list is unreadable would pass for the wrong reason.
#[test]
fn a_task_with_no_object_list_resolves_nothing_not_its_neighbours_list() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    // PTE for GVA page 1 (0x1000) → pfn 4, so task 2's list is readable.
    let mut pte = [0u8; 4];
    st32(&mut pte, 4);
    let _ = host.write_gpa(root_gpa + 4, &pte);

    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_BACKING as u32) | (0x40u32 << 8),
    );
    entry[4..12].copy_from_slice(&0xdead_0000u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa, &entry);

    // Task 2 owns a real list at pfn 1. Task 5 has a directory that maps
    // nothing, and `5 >> 1 == 2`.
    state.define_task(2, 0x1000, 2);
    assert!(state.set_object_list(2, 1, 4));
    state.define_task(5, 0x1000, 9);

    let donor = lookup_list_entry(&state, &host, 2, 0);
    assert!(
        donor.is_some(),
        "fixture is not real: task 2's own list must be readable"
    );

    // The behavioural claim first, so a regression fails on the corruption
    // itself rather than on the field that causes it.
    assert_eq!(
        lookup_list_entry(&state, &host, 5, 0),
        None,
        "task 5 has no object list, so it must resolve nothing — returning \
         Some here is task 2's entry answering for task 5"
    );
    assert_eq!(
        state.tasks[5].object_list_pfn, 0,
        "a defined task has no list until SetObjectList says so"
    );
    assert_eq!(state.tasks[5].object_list_count, 0);
}

/// The probe and the named lookup must give the **same answer**. Only whether a
/// miss is reportable differs.
///
/// This is the half a regression would break. `probe_list_entry` exists because
/// `backing_probe_order` walks every live task asking who owns a surface, so it
/// misses on every task before the owner — 18 `gva_read_refused` lines per
/// driven boot, all of them the search working. Quietening that is only correct
/// while it still *answers* identically; a probe that skipped the liveness test,
/// or read a different address, would pass a "no line was emitted" check and
/// still be wrong.
///
/// Same fixture as the test above: task 2 owns a readable list at pfn 1, task 5
/// has a directory that maps nothing, and `5 >> 1 == 2` — so a probe that fell
/// through to a neighbour's page table would answer `Some` for task 5.
#[test]
fn the_probe_and_the_named_lookup_answer_identically() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    let mut pte = [0u8; 4];
    st32(&mut pte, 4);
    let _ = host.write_gpa(root_gpa + 4, &pte);
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_BACKING as u32) | (0x40u32 << 8),
    );
    entry[4..12].copy_from_slice(&0xdead_0000u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa, &entry);

    state.define_task(2, 0x1000, 2);
    assert!(state.set_object_list(2, 1, 4));
    state.define_task(5, 0x1000, 9);

    for (task, ref_, what) in [
        (2u32, 0u32, "the owner's own entry"),
        (5, 0, "a task whose list does not translate"),
        (2, 3, "a slot inside the list the guest never filled"),
        (77, 0, "a task nothing defined"),
    ] {
        assert_eq!(
            probe_list_entry(&state, &host, task, ref_),
            lookup_list_entry(&state, &host, task, ref_),
            "probe and named lookup disagree on {what} (task {task}, ref {ref_})"
        );
    }

    // And the fixture is real in both directions, so the loop above cannot be
    // passing by answering `None` to everything.
    assert!(
        probe_list_entry(&state, &host, 2, 0).is_some(),
        "the owner must be found through the probe — that is what the search is for"
    );
    assert_eq!(probe_list_entry(&state, &host, 5, 0), None);
}

fn setup_backing_candidate(
    host: &mut FakeHost,
    state: &mut DeviceState,
    surface_id: u32,
    desc_gva: u64,
    desc_len: u32,
) -> u64 {
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    state.define_task(1, 0x1000, 2);
    assert!(state.set_object_list(1, 0, surface_id + 1));

    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_BACKING as u32) | (desc_len << 8),
    );
    entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    let entry_gpa = data_gpa + surface_id as u64 * OBJECT_LIST_ENTRY_LEN as u64;
    let _ = host.write_gpa(entry_gpa, &entry);
    data_gpa
}

/// Once task-scan lookup finds an actual backing candidate, descriptor read
/// failure is no longer speculative: the surface has an owner but cannot get
/// backing. It must be fail-visible with a stable reason slug.
#[test]
fn resolve_backing_candidate_logs_descriptor_read_failure() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let sid = 17u32;
    clear_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
    let _ = setup_backing_candidate(&mut host, &mut state, sid, 0x3000, 0x30);

    assert!(!resolve_backing(&mut state, &host, sid));
    assert!(
        backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "desc_read")),
        "surface-type candidate with unreadable descriptor must name desc_read"
    );
    clear_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
}

/// A readable but invalid backing descriptor used to fall through to the
/// resolver tail with no site reason. Keep it fail-visible without logging
/// absent/non-surface speculative probes.
#[test]
fn resolve_backing_candidate_logs_descriptor_decode_failure() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let sid = 18u32;
    clear_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
    let data_gpa = setup_backing_candidate(&mut host, &mut state, sid, 0x80, 0x30);
    let bad_desc = vec![0u8; 0x30];
    let _ = host.write_gpa(data_gpa + 0x80, &bad_desc);

    assert!(!resolve_backing(&mut state, &host, sid));
    assert!(
        backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "desc_decode")),
        "surface-type candidate with invalid descriptor must name desc_decode"
    );
    clear_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
}

/// Live wire bytes (boot 093019 `compute_stage_tex ref_texture … args_hex`):
/// R8 1024×1024 = Y plane view of a biplanar 1024×1024 surface.
#[test]
fn decode_ref_texture_view_live_r8_y_plane() {
    let mut desc = vec![0u8; 8];
    st32(&mut desc[TYPE5_SURFACE_ID..], 8);
    // args blob: kind 0x2f, len 0x30, own_ref 0x15, record R8 1024×1024 d=1.
    let args = [
        0x2fu8, 0, 0, 0, 0x30, 0, 0, 0, 0x15, 0, 0, 0, // kind, blob_len, own_ref
        0x42, 0x01, 0x0a, 0x00, // tag, unk, fmt=R8
        0x00, 0x04, 0x00, 0x00, // width 1024
        0x00, 0x04, 0x00, 0x00, // height 1024
        0x01, 0x00, 0x00, 0x00, // depth 1
        0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x00, // trailer (unconsumed)
    ];
    desc.extend_from_slice(&args);
    let rec = decode_ref_texture_view(&desc).expect("live R8 record decodes");
    assert_eq!(rec.pixel_format, 0x0a);
    assert_eq!((rec.width, rec.height, rec.depth), (1024, 1024, 1));
    // Short record (no +0x20 field) defaults to plane 0.
    assert_eq!(rec.plane_index, 0);
}

/// Live 56-byte wire blob from the BLIT copy-source path (x86 Ventura
/// 13.7.8, 2026-07-19 `blit t5_view_decode sid=34`): a full-color
/// texture view (BGRA8_sRGB 1024×768 window backing) carries the sibling
/// record tag `0x62`, not the biplanar `0x42`. Same field layout — must
/// decode, or the blit path drops the copy.
#[test]
fn decode_ref_texture_view_live_0x62_color_window_view() {
    // Exact leading 40 bytes observed, zero-padded to the 56-byte desc_len.
    let head: [u8; 40] = [
        0x22, 0x00, 0x00, 0x00, // surface_id = 34
        0x00, 0x00, 0x00, 0x00, // field
        0x2f, 0x00, 0x00, 0x00, // kind 0x2f
        0x30, 0x00, 0x00, 0x00, // blob_len 0x30
        0x0b, 0x00, 0x00, 0x00, // own_ref 0x0b
        0x62, 0x00, 0x51, 0x00, // tag=0x62, unk, fmt=0x51 BGRA8_sRGB
        0x00, 0x04, 0x00, 0x00, // width 1024
        0x00, 0x03, 0x00, 0x00, // height 768
        0x01, 0x00, 0x00, 0x00, // depth 1
        0x01, 0x00, 0x01, 0x00, // trailer
    ];
    let mut desc = head.to_vec();
    desc.resize(56, 0); // plane field (+0x20 in record) reads 0
    let rec = decode_ref_texture_view(&desc).expect("0x62 color view must decode");
    assert_eq!(rec.pixel_format, 0x51);
    assert_eq!((rec.width, rec.height, rec.depth), (1024, 768, 1));
    assert_eq!(rec.plane_index, 0);
}

/// Live 56-byte wire blob (boot 20260717-063043, v0a8 hero): the record
/// carries the `newTextureWithDescriptor:iosurface:plane:` plane at
/// `+0x20` — Y views carry 0, the RG8 chroma view 1, the same-geometry
/// alpha view 2. Geometry cannot separate Y from alpha; this field does.
#[test]
fn decode_ref_texture_view_live_v0a8_alpha_plane_index() {
    let mut desc = vec![0u8; 8];
    st32(&mut desc[TYPE5_SURFACE_ID..], 0x6d);
    let args = [
        0x2fu8, 0, 0, 0, 0x30, 0, 0, 0, 0x82, 0x01, 0, 0, // kind, blob_len, own_ref
        0x42, 0x01, 0x0a, 0x00, // tag, unk, fmt=R8
        0xb2, 0x03, 0x00, 0x00, // width 946
        0x5e, 0x01, 0x00, 0x00, // height 350
        0x01, 0x00, 0x00, 0x00, // depth 1
        0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x00, // trailer
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
        0x02, 0x00, 0x00, 0x00, // IOSurface plane index = 2 (alpha)
    ];
    desc.extend_from_slice(&args);
    let rec = decode_ref_texture_view(&desc).expect("live v0a8 alpha record decodes");
    assert_eq!(rec.pixel_format, 0x0a);
    assert_eq!((rec.width, rec.height, rec.depth), (946, 350, 1));
    assert_eq!(rec.plane_index, 2);
}

/// The owner-task census must read the dword the guest wrote, and must be
/// able to tell 0 from anything else.
///
/// A census whose extraction is wrong reports 0 forever whatever the wire
/// says, and 0 is the answer this device already assumes — so the failing
/// case would be indistinguishable from the healthy one, which is the whole
/// point of having it. Pinning the offset against a descriptor whose *other*
/// leading dword is non-zero is what makes an off-by-four visible.
#[test]
fn the_ref_texture_owner_task_is_read_from_its_own_dword() {
    let mut desc = [0u8; TYPE5_MIN_LEN];
    st32(&mut desc[TYPE5_SURFACE_ID..], 0xabcd);
    assert_eq!(
        ld32(&desc[TYPE5_OWNER_TASK..]),
        0,
        "the surface id must not be read as the owner task"
    );
    st32(&mut desc[TYPE5_OWNER_TASK..], 7);
    assert_eq!(ld32(&desc[TYPE5_OWNER_TASK..]), 7);
    assert_eq!(
        ld32(&desc[TYPE5_SURFACE_ID..]),
        0xabcd,
        "writing the owner task must not disturb the surface id"
    );
    // Both fields sit inside the minimum descriptor — the array above is
    // exactly `TYPE5_MIN_LEN` and indexing it proves that — so the census can
    // never be silently skipped on a well-formed record.
    assert_eq!(TYPE5_OWNER_TASK, TYPE5_SURFACE_ID + 4);
}

#[test]
fn decode_ref_texture_view_fail_closed() {
    // Short descriptor (no record).
    let mut short = vec![0u8; 8];
    st32(&mut short[TYPE5_SURFACE_ID..], 8);
    assert!(decode_ref_texture_view(&short).is_none());
    // Wrong record tag.
    let mut bad_tag = vec![0u8; 8];
    st32(&mut bad_tag[TYPE5_SURFACE_ID..], 8);
    bad_tag.extend_from_slice(&[0u8; 12]);
    bad_tag.extend_from_slice(&[
        0x41, 0x01, 0x0a, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x01, 0, 0, 0,
    ]);
    assert!(decode_ref_texture_view(&bad_tag).is_none());
    // Non-2D (depth != 1) fails closed.
    let mut vol = vec![0u8; 8];
    st32(&mut vol[TYPE5_SURFACE_ID..], 8);
    vol.extend_from_slice(&[0u8; 12]);
    vol.extend_from_slice(&[
        0x42, 0x07, 0x50, 0x00, 0x40, 0, 0, 0, 0x40, 0, 0, 0, 0x40, 0, 0, 0,
    ]);
    assert!(decode_ref_texture_view(&vol).is_none());
    // Zero width fails closed.
    let mut zw = vec![0u8; 8];
    st32(&mut zw[TYPE5_SURFACE_ID..], 8);
    zw.extend_from_slice(&[0u8; 12]);
    zw.extend_from_slice(&[
        0x42, 0x01, 0x0a, 0x00, 0, 0, 0, 0, 0x00, 0x04, 0, 0, 0x01, 0, 0, 0,
    ]);
    assert!(decode_ref_texture_view(&zw).is_none());
}

/// The probe's notion of "undecoded" must be exactly the bytes
/// `decode_backing` skips, and it must distinguish two surfaces on
/// those bytes alone.
///
/// This is the measurement that blocks the largest deletion in the present
/// path: nothing decoded at surface-create time separates a desktop
/// swapchain buffer from a same-geometry offscreen tile, so membership is
/// reconstructed by half a dozen downstream mechanisms. If the guest is
/// telling us in the undecoded span, the probe has to be able to see it.
/// The two arms of the backing freshness test must accept exactly the same
/// backings, because only one of them rebuilds when it says no.
///
/// The force arm returns through `win_backing_search` **without** calling
/// `apply_backing`, so `set_mapping_geom` and
/// `synthesize_device_desc_from_backing` are both skipped. It used to compare
/// width alone while the non-force arm compared width and height, and
/// `ensure_surface_for_present` calls the force arm precisely to catch a
/// wire geometry change — so a height change that stayed inside the same
/// page count left the mapping describing the previous incarnation, on the
/// path whose job was to notice.
///
/// Neither arm compared format, and a surface id recycled at identical
/// dimensions with a different pixel format keeps the old bytes-per-pixel
/// for every read window built over it.
#[test]
fn a_latched_backing_is_stale_when_any_of_geometry_or_format_moved() {
    use crate::protocol::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA8_UNORM};
    let surf = |w: u32, h: u32, fourcc: u32| BackingRecord {
        length: 0x1000,
        backing_pfn: 1,
        pixel_format: fourcc,
        plane_count: 1,
        planes: Default::default(),
        width: w,
        height: h,
        bytes_per_row: w * 4,
    };
    // 'BGRA' and 'RGBA' are distinct single-plane FourCCs at one bpp, so a
    // swap between them is invisible to a dimensions-only test.
    const BGRA: u32 = 0x4247_5241;
    const RGBA: u32 = 0x5247_4241;
    assert_eq!(
        latched_mapping_format(&surf(8, 4, BGRA)),
        MTL_FORMAT_BGRA8_UNORM
    );
    assert_eq!(
        latched_mapping_format(&surf(8, 4, RGBA)),
        MTL_FORMAT_RGBA8_UNORM
    );

    let m = MappingEntry {
        width: 8,
        height: 4,
        format: MTL_FORMAT_BGRA8_UNORM,
        ..Default::default()
    };
    assert!(backing_matches_latched_geom(&m, &surf(8, 4, BGRA)));
    assert!(
        !backing_matches_latched_geom(&m, &surf(8, 5, BGRA)),
        "a height change must be stale on both arms"
    );
    assert!(!backing_matches_latched_geom(&m, &surf(9, 4, BGRA)));
    assert!(
        !backing_matches_latched_geom(&m, &surf(8, 4, RGBA)),
        "same dimensions, different format: every read window's bpp comes from it"
    );
}

/// A multi-plane backing must compare equal to itself.
///
/// The latch stores `0` for it — the decoder's refusal to name a single
/// colour format — while the raw FourCC conversion may well return a real
/// format. A freshness test that compared the raw conversion would find
/// `0 != BGRA8` on every present and rebuild the backing forever, which is
/// the failure a shared `latched_mapping_format` exists to make impossible.
#[test]
fn a_multiplane_backing_compares_equal_to_the_zero_it_latched() {
    let mut surf = BackingRecord {
        length: 0x1000,
        backing_pfn: 1,
        pixel_format: 0x4247_5241, // 'BGRA' — a format the converter knows
        plane_count: 2,
        planes: Default::default(),
        width: 8,
        height: 4,
        bytes_per_row: 32,
    };
    assert_ne!(
        iosurface_pixel_format_to_mtl(surf.pixel_format),
        0,
        "the fixture only means something if the raw conversion resolves"
    );
    assert_eq!(latched_mapping_format(&surf), 0, "multi-plane latches 0");

    let m = MappingEntry {
        width: 8,
        height: 4,
        format: 0,
        ..Default::default()
    };
    assert!(backing_matches_latched_geom(&m, &surf));
    // Dropping to one plane makes it a single-plane BGRA8 surface, which is
    // a real change of what the mapping describes.
    surf.plane_count = 1;
    assert!(!backing_matches_latched_geom(&m, &surf));
}

/// A single-plane surface must publish plane 0's offset, because both its
/// consumers fold it in and one of them is the other pathway.
///
/// `decode_backing_plane` reads four fields; the surface-level convenience
/// copies on `BackingRecord` take three, and the synthesizer's single-plane
/// arm used to publish only those three. A surface whose pixels start past
/// the base of its allocation was then read and written at 0 — the
/// multi-plane arm has always published each plane's offset, and
/// `sample_window_from_device_surface` treats `base_offset` exactly as
/// `sample_window_from_device_plane` treats a plane's.
#[test]
fn a_single_plane_backing_publishes_the_offset_its_pixels_start_at() {
    use crate::protocol::iosurface_pages::{decode_device_surface, sample_window_from_device_desc};
    use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    const BASE: u32 = 0x800;
    let (w, h, bpr) = (8u32, 4u32, 32u32);
    let mut surf = BackingRecord {
        length: 0x4000,
        backing_pfn: 1,
        pixel_format: 0x4247_5241, // 'BGRA'
        plane_count: 1,
        planes: Default::default(),
        width: w,
        height: h,
        bytes_per_row: bpr,
    };
    surf.planes[0] = BackingPlane {
        offset: BASE,
        width: w,
        height: h,
        bytes_per_row: bpr,
        bytes_per_element: 4,
    };
    assert!(
        !backing_is_multiplanar(&surf),
        "the single-plane arm is the one under test"
    );

    let desc = synthesize_device_desc_from_backing(&surf);
    let decoded = decode_device_surface(&desc).expect("device descriptor");
    assert_eq!(
        decoded.plane_count, 0,
        "single-plane publishes no plane records"
    );
    assert_eq!(decoded.base_offset, BASE);

    // The consumer, not just the field: the sample window must start at the
    // offset and its span must end past it, or publishing it bought nothing.
    let (off, got_bpr, end) =
        sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_BGRA8_UNORM, w, h)
            .expect("surface-level window");
    assert_eq!(off, BASE as u64);
    assert_eq!(got_bpr, bpr);
    assert_eq!(
        end,
        BASE as u64 + (h as u64 - 1) * bpr as u64 + (w as u64 * 4)
    );

    // Zero stays zero — the ordinary case must not gain an offset.
    surf.planes[0].offset = 0;
    let zero = synthesize_device_desc_from_backing(&surf);
    assert_eq!(decode_device_surface(&zero).expect("desc").base_offset, 0);
}

/// The device descriptor's format word must survive both of the encodings
/// it is written in.
///
/// The x86 synthesizer writes an MTL ordinal for a known single-plane
/// surface and the raw OSType otherwise; the arm64 mapper reads the guest's
/// own descriptor, where media surfaces carry a FourCC. Narrowing with
/// `as u16` is correct for one of those and destroys the other — `'BGRA'`
/// becomes `0x5241`, which no format table accepts, so the mapping ends up
/// with a format that refuses every sample window and every render target.
#[test]
fn the_device_descriptor_format_word_survives_both_of_its_encodings() {
    use crate::protocol::pixel_format::{
        bytes_per_pixel, MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA16_FLOAT,
    };

    const BGRA_FOURCC: u32 = 0x4247_5241;

    // The failure the narrowing produced, stated as the thing not to return.
    assert!(
        bytes_per_pixel((BGRA_FOURCC & 0xffff) as u16).is_none(),
        "the truncation's output is not a format, which is why it was a bug"
    );
    assert_eq!(
        device_desc_format_to_mtl(BGRA_FOURCC),
        MTL_FORMAT_BGRA8_UNORM
    );

    // An ordinal fits in the descriptor's own 16-bit format fields and is
    // passed through as itself — including one above the old 0x200
    // magnitude boundary, which is why the test is width and not size.
    assert_eq!(
        device_desc_format_to_mtl(MTL_FORMAT_BGRA8_UNORM as u32),
        MTL_FORMAT_BGRA8_UNORM
    );
    assert_eq!(
        device_desc_format_to_mtl(MTL_FORMAT_RGBA16_FLOAT as u32),
        MTL_FORMAT_RGBA16_FLOAT
    );
    // MTLPixelFormatBGRA10_XR is 552, above the 0x200 boundary an earlier
    // magnitude test used and which `iosurface_pixel_format_to_mtl` records
    // as having been wrong for exactly this format. It still fits in 16
    // bits, so the width test carries it where a size test did not.
    assert_eq!(device_desc_format_to_mtl(552), 552);

    // Fail closed, not BGRA8: a multi-plane OSType and an unknown one.
    assert_eq!(device_desc_format_to_mtl(IOSURFACE_FOURCC_420F), 0);
    assert_eq!(device_desc_format_to_mtl(0x5A5A_5A5A), 0);
    assert_eq!(device_desc_format_to_mtl(0), 0);
}

/// The backing probe order must visit task 0 first, the hint next, and every
/// **live** task exactly once.
///
/// It is the thing that makes the search terminate on the first probe for
/// every surface this device has ever resolved, so its shape is the whole
/// cost of the search. Two properties are load-bearing and neither is
/// obvious from the iterator chain: no task may be probed **twice** (a
/// duplicate is a wasted guest read on the hot present path), and no live task
/// may be **missed** (a missed one is a surface that cannot be found at all).
///
/// The tail used to be `1..MAX_TASKS`, and this test asserted a length of
/// exactly 256. It now walks the live ids, so the assertions are about the
/// task set rather than about a constant — which is the point of the change,
/// and is why a length assertion against a number would silently stop meaning
/// anything.
#[test]
fn the_backing_probe_order_visits_task_zero_first_and_every_live_task_once() {
    use std::collections::HashSet;

    let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
    // Deliberately sparse, and one id far past the retired 256 ceiling: the
    // probe must reach a task the old fixed range could not even name.
    let live = [0u32, 1, 7, 300, 70_000];
    for id in live {
        state.define_task(id, 0x1000, 2);
    }

    for hint in [0u32, 1, 7, 70_000] {
        let order = backing_probe_order(&state.tasks, hint);
        assert_eq!(order[0], 0, "task 0 leads for hint {hint}");
        if hint != 0 {
            assert_eq!(order[1], hint, "the hint is probed second");
        }
        let seen: HashSet<u32> = order.iter().copied().collect();
        assert_eq!(
            seen.len(),
            order.len(),
            "no task probed twice for hint {hint}: {order:?}"
        );
        assert!(
            live.iter().all(|t| seen.contains(t)),
            "every live task probed for hint {hint}: {order:?}"
        );
    }

    // A dead task is not probed. It never could be — the probe's own liveness
    // test refused it — so yielding it only ever cost a guest read's worth of
    // work per present.
    let order = backing_probe_order(&state.tasks, 0);
    assert!(
        !order.contains(&9),
        "an id nothing defined must not be probed: {order:?}"
    );

    // A hint naming no live task adds a probe that the liveness test at the
    // probe then refuses, and must not lose a live one or duplicate task 0.
    let order = backing_probe_order(&state.tasks, u32::MAX);
    let seen: HashSet<u32> = order.iter().copied().collect();
    assert_eq!(seen.len(), order.len(), "{order:?}");
    assert!(live.iter().all(|t| seen.contains(t)), "{order:?}");
}

#[test]
fn undecoded_backing_span_is_exactly_what_the_decoder_skips() {
    // One plane: the decoder consumes 0x14..0x24, so the tail starts there.
    let built = BackingBuilder::new(0x800000, 0x1234, 0x4247_5241, 1) // 'BGRA'
        .plane(0, 0, 1920, 1080, 1920 * 4, 0)
        .with_len(0x40);
    let a = built.bytes().to_vec();

    // Every decoded field can change without moving the undecoded span.
    let b = BackingBuilder::new(0x900000, 0x9999, 0x4c31_3062, 1)
        .plane(0, 0, 1280, 720, 1280 * 4, 0)
        .with_len(0x40);
    assert_eq!(
        undecoded_backing_bytes(&a),
        undecoded_backing_bytes(b.bytes()),
        "changing only decoded fields must not look like a new shape"
    );

    // The span covers the three bytes after plane_count and the whole tail
    // past the plane records the decoder consumed.
    for probe in [0x11usize, 0x13, 0x24, 0x3f] {
        let mut c = a.clone();
        c[probe] ^= 0xff;
        assert_ne!(
            undecoded_backing_bytes(&a),
            undecoded_backing_bytes(&c),
            "byte {probe:#x} is undecoded and must be visible to the probe"
        );
    }

    // Bytes the decoder DOES read must not be in the span, or ordinary
    // surface-to-surface variation would look like a new shape forever.
    // `plane_count` (+0x10) is excluded on purpose: it is decoded AND it
    // moves the span's own boundary, which the two-plane case below pins.
    for probe in [0x00usize, 0x08, 0x0c, 0x14, 0x23] {
        let mut c = a.clone();
        c[probe] ^= 0xff;
        assert_eq!(
            undecoded_backing_bytes(&a),
            undecoded_backing_bytes(&c),
            "byte {probe:#x} is decoded and must stay out of the span"
        );
    }

    // A second plane moves the boundary: 0x24..0x34 becomes decoded.
    let two = BackingBuilder::new(0x800000, 0x1234, 0x4247_5241, 2)
        .plane(0, 0, 1920, 1080, 1920 * 4, 0)
        .with_len(0x40);
    assert_eq!(
        undecoded_backing_bytes(two.bytes()).len(),
        undecoded_backing_bytes(&a).len() - TYPE4_PLANE_STRIDE,
        "the span shrinks by exactly one plane record"
    );

    // A record too short to decode reports nothing rather than a partial
    // span that would compare unequal against every real one.
    assert!(undecoded_backing_bytes(&a[..TYPE4_MIN_LEN - 1]).is_empty());
}

/// The shared ladder asks its three questions in the only order they can be
/// asked, and names which one refused.
///
/// The order is the point, not an implementation detail: a type tag cannot be
/// checked before the entry is found, and a descriptor cannot be read before the
/// entry says where it is. Twenty rails wrote this out by hand and any of them
/// could have reordered or dropped a rung without the compiler noticing — which
/// is what [`LadderRung`] being a value rather than three separate `else`
/// branches removes.
#[test]
fn the_shared_ladder_names_the_rung_that_refused() {
    use crate::runtime::decode::resource::{OBJECT_TYPE_BUFFER, OBJECT_TYPE_MAPPER_REF_TEXTURE};

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);

    // Ref 1 is a mapper-ref-texture entry whose descriptor is mapped: all three rungs pass.
    let (entry, bytes) = resolve_descriptor(&state, &host, 1, 1, &[OBJECT_TYPE_MAPPER_REF_TEXTURE])
        .expect("all rungs pass");
    assert_eq!(entry.object_type, OBJECT_TYPE_MAPPER_REF_TEXTURE);
    assert!(!bytes.is_empty(), "the descriptor bytes come back with it");

    // Same ref, asked for as a buffer: the tag it found travels with the
    // refusal, so a rail no longer re-formats `ot=` from an entry it has
    // already dropped.
    assert_eq!(
        resolve_descriptor(&state, &host, 1, 1, &[OBJECT_TYPE_BUFFER]),
        Err(LadderRung::WrongType {
            got: OBJECT_TYPE_MAPPER_REF_TEXTURE
        })
    );

    // A ref past the end of the list. Asked for *no* acceptable type at all, so
    // a resolver that checked the tag first would have to answer `WrongType`;
    // answering `NoListEntry` is what proves the lookup runs first.
    assert_eq!(
        resolve_descriptor(&state, &host, 1, 9999, &[]),
        Err(LadderRung::NoListEntry)
    );

    // An entry whose descriptor GVA is not mapped: found, right type, unreadable
    // — the rung that separates "the guest never registered this" from "the
    // guest registered it and its descriptor is not resident right now".
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_MAPPER_REF_TEXTURE) | (0x20u32 << 8),
    );
    entry[4..12].copy_from_slice(&0xdead_0000u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 24, &entry);
    // The declared length travels with the rung: the entry above says 0x20
    // bytes, and by the time a rail reports this the entry is gone.
    assert_eq!(
        resolve_descriptor(&state, &host, 1, 2, &[OBJECT_TYPE_MAPPER_REF_TEXTURE]),
        Err(LadderRung::DescRead { declared_len: 0x20 })
    );
}

/// A re-point over a ref-keyed host copy drops that copy, so the next resolve
/// reads the pages the guest just rewired instead of bytes read from the old
/// ones.
///
/// `ReplacePhysical` says by its own contract that the PFNs under this object
/// have changed. The mapping rail discharges that through
/// `invalidate_mapping_pages`, but `host_texture_surfaces` and
/// `host_linear_textures` are keyed by object-list ref and carry no page list,
/// so nothing in them can notice — and this device holds a copy under exactly
/// those keys for resources that own no mapping. Measured on a driven x86/PCI boot under
/// `web-content-probe`: 7 texture and 1 linear against 32 that held nothing, so
/// the guest was being served stale content on an ordinary browsing workload.
///
/// Fails without the fix: both entries survive the packet.
#[test]
fn a_repoint_drops_the_ref_keyed_host_copies_of_the_object() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let (task, object) = (7u32, 4242u32);

    state.host_texture_surfaces.insert(
        (task, object),
        crate::model::HostSurface {
            width: 4,
            height: 4,
            bgra: std::sync::Arc::new(vec![0xAB; 4 * 4 * 4]),
            host_gen: 1,
            producer_object_type: 0,
            last_touch: 0,
            backing: None,
            guest_holds_bytes: false,
            source_gva: 0,
        },
    );
    state.host_texture_surfaces.insert(
        (task + 1, object),
        crate::model::HostSurface {
            width: 4,
            height: 4,
            bgra: std::sync::Arc::new(vec![0xEF; 4 * 4 * 4]),
            host_gen: 2,
            producer_object_type: 0,
            last_touch: 0,
            backing: None,
            guest_holds_bytes: false,
            source_gva: 0,
        },
    );
    state.host_linear_textures.insert(
        (task, object),
        crate::model::HostLinearTexture {
            gva: 0x1000,
            pixel_format: 0,
            width: 4,
            height: 4,
            row_stride: 16,
            bytes: vec![0xCD; 64],
            host_gen: 1,
            resident_gen: 0,
        },
    );
    // No mapping owns the id, which is the route this covers: three quarters of
    // the re-points on a driven boot take it.
    assert!(!state.mappings.contains_key(&object));

    super::replace_physical(&mut state, &mut host, task, object);

    assert!(
        !state.host_texture_surfaces.contains_key(&(task, object)),
        "the ref-keyed texture copy was read from pages the guest has re-pointed"
    );
    assert!(
        state
            .host_texture_surfaces
            .contains_key(&(task + 1, object)),
        "a re-point must not evict another task's same-numbered texture copy"
    );
    assert!(
        !state.host_linear_textures.contains_key(&(task, object)),
        "the ref-keyed linear copy was read from pages the guest has re-pointed"
    );
}

/// ReplacePhysical's object id is local to the task carried beside it. A
/// mapping id is a different namespace even when the integers happen to be
/// equal.
///
/// This is the compositor failure class: task 0 owns backing record 1 while
/// task 1 owns mapper-ref-texture resource 1, which resolves to mapping 9. Re-pointing the
/// latter must retire mapping 9 and leave task 0's surface intact. The old
/// global-id-first route did the opposite.
#[test]
fn a_repoint_resolves_the_resource_in_its_task_before_touching_a_mapping() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_with_list(&mut host, &mut state);

    assert_eq!(resolve_mapper_ref_texture(&mut state, &host, 1, 1), Some(9));

    assert!(state.map_surface(1));
    {
        let surface = state.mappings.get_mut(&1).expect("surface mapping");
        surface.mapped = true;
        surface.page_entries = vec![0x1234_5001];
        surface.backing_walk = Some(crate::model::BackingWalk {
            task_id: 0,
            backing_pfn: 0x20,
            map_generation: surface.map_generation,
        });
    }
    {
        let resource = state
            .mappings
            .get_mut(&9)
            .expect("mapper-ref-texture mapping");
        resource.mapped = true;
        resource.page_entries = vec![0x6789_a001];
    }

    super::replace_physical(&mut state, &mut host, 1, 1);

    assert_eq!(
        state.mappings[&1].page_entries,
        vec![0x1234_5001],
        "a same-number resource in another task does not own this surface"
    );
    assert!(
        state.mappings[&9].page_entries.is_empty(),
        "the task-local mapper-ref-texture association names the mapping to invalidate"
    );
}

/// A direct backing resource is routed by the task provenance latched with its
/// page walk, so tightening the namespace must not suppress genuine surface
/// re-points.
#[test]
fn a_repoint_retires_a_backing_mapping_owned_by_the_packet_task() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    assert!(state.map_surface(7));
    let prior_generation = {
        let surface = state.mappings.get_mut(&7).expect("surface mapping");
        surface.mapped = true;
        surface.page_entries = vec![0x1234_5001];
        surface.backing_walk = Some(crate::model::BackingWalk {
            task_id: 3,
            backing_pfn: 0x20,
            map_generation: surface.map_generation,
        });
        surface.map_generation
    };

    super::replace_physical(&mut state, &mut host, 3, 7);

    assert!(state.mappings[&7].page_entries.is_empty());
    assert_ne!(state.mappings[&7].map_generation, prior_generation);
    assert_ne!(
        state.mappings[&7]
            .backing_walk
            .expect("the old walk remains only as provenance")
            .map_generation,
        state.mappings[&7].map_generation,
        "the generation bump makes the retired walk unusable as currency"
    );
}

/// Storage named by an address rather than by a mapping gets an incarnation,
/// and it advances on exactly the events that make the pages behind that
/// *window* different pages.
///
/// The counter exists so a canonical backing identity can be a window *and* a
/// number. A window alone repeats across a physical replacement — same
/// guest-virtual address, different host frames — and equal ids would let a
/// claim on the old frames be satisfied by the new ones, handing storage back
/// under a live reader. Each assertion below is one way that could happen.
///
/// It is keyed on the window and not on the reference, and the assertion that
/// two references over one window move together is the one a driven boot made
/// load-bearing: it found exactly that, on the compositor's scanout buffer.
#[test]
fn address_named_storage_counts_its_incarnations() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    // Two references over one window, and a third over another, so the test can
    // say both that the count is shared where the storage is and separate where
    // it is not.
    let (task, object, twin, neighbour) = (1u32, 2u32, 3u32, 4u32);
    let write_object = |host: &mut FakeHost, ref_: u32, desc_gva: u64, handle: u64| {
        let mut desc = [0u8; 16];
        desc[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
        desc[8..16].copy_from_slice(&handle.to_le_bytes());
        let _ = host.write_gpa(data_gpa + desc_gva, &desc);
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
        );
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        let _ = host.write_gpa(data_gpa + u64::from(ref_) * 12, &entry);
    };
    let (handle, other_handle) = (0x20u64, 0x30u64);
    let window = handle << PAGE_SHIFT_ARM64E;
    let other_window = other_handle << PAGE_SHIFT_ARM64E;
    write_object(&mut host, object, 0x100, handle);
    write_object(&mut host, twin, 0x120, handle);
    write_object(&mut host, neighbour, 0x140, other_handle);
    for ref_ in [object, twin, neighbour] {
        assert!(state.insert_object(task, ref_));
    }

    // The first incarnation is a value, not an absence: nothing has re-pointed
    // this window, which is a fact about it and not missing information.
    let first = state.storage_incarnation(task, window);
    assert_eq!(
        first,
        StorageIncarnation::default(),
        "nothing has happened yet"
    );
    // Every incarnation this window has ever carried. Distinctness is the whole
    // property, and a pairwise assertion would only catch neighbours.
    let mut seen = vec![first];

    // A re-point that reaches no mapping and no host copy still advances it.
    // The packet is an announcement about the guest's memory; what this device
    // happened to be caching does not decide whether the pages moved.
    super::replace_physical(&mut state, &mut host, task, object);
    let after_repoint = state.storage_incarnation(task, window);
    assert_ne!(after_repoint, first, "the announcement was not recorded");
    seen.push(after_repoint);
    assert_eq!(
        state.storage_incarnation(task, other_window),
        first,
        "a re-point of one window moved another window's storage"
    );

    // The other name over the same window moves with it. This is the assertion
    // the reference-keyed version could not make, and the one a boot proved is
    // needed: a claim held under `twin` must stop naming the frames `object`'s
    // packet has just replaced.
    //
    // The twin's window is resolved the way production resolves it, from its
    // own object-list record, rather than reusing this test's `window` -- the
    // claim is that the two references arrive at one key, and asserting it with
    // the key already in hand would assert nothing.
    let twin_entry = lookup_list_entry(&state, &host, task, twin).expect("the twin is listed");
    let twin_descriptor =
        super::read_descriptor(&state, &host, task, &twin_entry).expect("its descriptor reads");
    let (twin_window, _) = super::backing_window(state.page_shift, &twin_entry, &twin_descriptor)
        .expect("a buffer names a window");
    assert_eq!(
        twin_window, window,
        "the two references are over one allocation, which is the case the \
         reading found on a live compositor"
    );
    assert_eq!(
        state.storage_incarnation(task, twin_window),
        after_repoint,
        "the second name over this window kept the incarnation the re-point \
         retired, so a claim under it would still be satisfied by the old frames"
    );

    // Releasing a name is deliberately *not* one of the events. Other storage
    // at this reference is a different window and is distinct already; the same
    // storage back is the same backing, and a bump would deny it.
    assert!(state.delete_object(task, object));
    assert_eq!(
        state.storage_incarnation(task, window),
        after_repoint,
        "releasing a name is not a statement about the storage behind a window"
    );

    // A task teardown ends every window the task held at once, and a task id
    // comes back.
    assert!(state.delete_task(task));
    let after_teardown = state.storage_incarnation(task, window);
    assert!(
        !seen.contains(&after_teardown),
        "the task teardown left this window on an incarnation it already had"
    );
    seen.push(after_teardown);
    assert_ne!(
        state.storage_incarnation(task, other_window),
        first,
        "a window the guest published and this device never re-pointed has no \
         per-window entry, so only the task epoch can carry it across a teardown"
    );

    // Taking the id again moves nothing further, and does not have to: the
    // teardown is the event that separated the two tasks' storage.
    state.define_task(task, 0x1000, 0x40);
    assert_eq!(
        state.storage_incarnation(task, window),
        after_teardown,
        "the new task's window left the incarnation its teardown established"
    );

    // And a redefinition of a live task is the same event: it drops the task's
    // objects and may root a different physical page under the same addresses.
    let before_redefine = state.storage_incarnation(task, window);
    state.define_task(task, 0x1000, 0x41);
    assert_ne!(state.storage_incarnation(task, window), before_redefine);
    assert!(
        !seen.contains(&state.storage_incarnation(task, window)),
        "an incarnation this window already had came back"
    );

    // A different task's identically-addressed window is its own count, which
    // is what makes the key task-local rather than a global address.
    assert_eq!(state.storage_incarnation(task + 1, window), first);
}

/// A re-point that reaches nothing changes nothing, and does not invent a
/// removal for a neighbouring ref.
///
/// The counterpart to the test above and the reason the repair is keyed on
/// `(task, ref)` rather than on the ref alone for the linear map: 32 of 40
/// re-points on the measured boot held no state at all, and one that started
/// evicting its neighbours would turn a benign majority into a new loss.
#[test]
fn a_repoint_of_an_object_this_device_holds_nothing_for_touches_no_neighbour() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let (task, object, neighbour) = (7u32, 4242u32, 4243u32);

    state.host_linear_textures.insert(
        (task, neighbour),
        crate::model::HostLinearTexture {
            gva: 0x1000,
            pixel_format: 0,
            width: 4,
            height: 4,
            row_stride: 16,
            bytes: vec![0xCD; 64],
            host_gen: 1,
            resident_gen: 0,
        },
    );
    // The same ref under a different task must also survive.
    state.host_linear_textures.insert(
        (task + 1, object),
        crate::model::HostLinearTexture {
            gva: 0x2000,
            pixel_format: 0,
            width: 4,
            height: 4,
            row_stride: 16,
            bytes: vec![0xEF; 64],
            host_gen: 1,
            resident_gen: 0,
        },
    );

    super::replace_physical(&mut state, &mut host, task, object);

    assert!(
        state.host_linear_textures.contains_key(&(task, neighbour)),
        "a different ref in the same task is a different object"
    );
    assert!(
        state.host_linear_textures.contains_key(&(task + 1, object)),
        "the same ref in a different task is a different object"
    );
}

/// Each way an object-list lookup comes back empty gets its own route.
///
/// The whole reason [`super::ListMiss`] exists is that eight causes shared one
/// `reason=no_list_entry`, so a boot losing draws could not say whether this
/// device had cleared a task's list under the guest or the guest had not
/// published the object yet. Two variants sharing a route string — the obvious
/// copy-paste when a ninth is added — rebuilds exactly that, and rebuilds it
/// invisibly, because a merged population still reads as a clean count.
#[test]
fn every_object_list_miss_names_a_different_check() {
    let routes: Vec<&'static str> = super::ListMiss::ALL.iter().map(|m| m.route()).collect();
    let mut unique = routes.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        routes.len(),
        "two object-list misses share a route, so their counts add up as one: {routes:?}"
    );
    assert!(
        routes.iter().all(|r| r.starts_with("list_miss_")),
        "the family shares a prefix so a boot can rank it in one grep: {routes:?}"
    );
}

/// The claimant banding must separate a real ownership signal from the confound
/// that nearly buried it.
///
/// Every task registers its object list at the same `pfn = 1`, so on a busy
/// guest "some other task has something at slot 3" is close to a tautology. The
/// first version of this instrument was a yes/no and answered yes to every miss
/// on macos-26, which reads as a finding and is not one. The band against the
/// live task count is what makes the difference visible, so each boundary is
/// pinned here:
///
/// - nobody has it — the guest has not published it, and the fix is to wait;
/// - exactly one other task has it — a real ownership signal, the object is in
///   a list this device did not look in;
/// - all of the others have it — the slot index is just populated everywhere and
///   this search cannot tell ownership from coincidence.
///
/// The asking task is excluded from the count, so "all" must compare against
/// `live - 1`. Comparing against `live` would make "all" unreachable and silently
/// demote every genuine all-claim to "many".
#[test]
fn a_claimant_count_is_banded_against_the_tasks_that_could_have_claimed() {
    use super::slot_empty_claim_route as band;

    assert_eq!(band(0, 8), "list_miss_slot_empty_claimed_nowhere");
    assert_eq!(band(1, 8), "list_miss_slot_empty_claimed_by_one");
    assert_eq!(band(4, 8), "list_miss_slot_empty_claimed_by_many");
    assert_eq!(
        band(7, 8),
        "list_miss_slot_empty_claimed_by_all",
        "seven others out of eight live tasks is every task that could have claimed"
    );

    // Two tasks total: the one asking and one other. That other claiming is
    // both "one" and "all", and "one" is the reading that matters — it is the
    // ownership signal, while "all" only ever means the search is uninformative.
    assert_eq!(band(1, 2), "list_miss_slot_empty_claimed_by_one");

    // A single live task has nobody else to claim, and must not be reported as
    // a unanimous claim over an empty population.
    assert_eq!(band(0, 1), "list_miss_slot_empty_claimed_nowhere");
    assert_eq!(band(0, 0), "list_miss_slot_empty_claimed_nowhere");
}

/// A zero reusable slot has no resolvable tenant, even when an earlier tenant
/// was observed successfully.
#[test]
fn a_freed_or_between_tenants_slot_never_answers_from_an_earlier_generation() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let slot = data_gpa + 12; // ref 1, at `ref * 12`

    let first = lookup_list_entry(&state, &host, 1, 1).expect("the published ref resolves");
    assert_eq!((first.object_type, first.descriptor_gva), (11, 0x40));

    // Deletion clears the index. The packet has no generation with which to
    // prove that the first tenant, rather than a later one, is the answer.
    let _ = host.write_gpa(slot, &[0u8; 12]);
    assert_eq!(
        lookup_list_entry(&state, &host, 1, 1),
        None,
        "a zero reusable slot cannot name its earlier tenant"
    );

    // A later allocation reuses the same index for another object.
    let mut reused = [0u8; 12];
    st32(&mut reused[0..], 2u32 | (0x30u32 << 8));
    reused[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(slot, &reused);
    let now = lookup_list_entry(&state, &host, 1, 1).expect("the reused slot resolves");
    assert_eq!(
        (now.object_type, now.descriptor_gva),
        (2, 0x80),
        "a reused index must resolve to its new object, never the retired one"
    );

    // Clearing the reused tenant must not resurrect either generation.
    let _ = host.write_gpa(slot, &[0u8; 12]);
    assert_eq!(
        lookup_list_entry(&state, &host, 1, 1),
        None,
        "a second empty tenancy gap cannot resurrect the first or second object"
    );
}

/// A guest-VA window is claimed by one reference, and the claim table says who
/// got there first.
///
/// This is the state behind [`super::note_backing_window_alias`], and what it
/// answers decides whether the incarnation `BackingId` mixes in may stay on the
/// reference. Each assertion is one way the claim could stop meaning "these two
/// names are over one piece of storage".
#[test]
fn one_guest_window_is_claimed_by_one_reference() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let (task, other_task, first, second) = (3u32, 4u32, 9u32, 10u32);
    let window = 0x4_0000u64;
    state.define_task(task, 0x1000, 0x40);
    state.define_task(other_task, 0x2000, 0x40);

    assert_eq!(
        state.claim_backing_window(task, first, window),
        None,
        "nothing held this window, so the first reference takes it"
    );
    assert_eq!(
        state.claim_backing_window(task, first, window),
        None,
        "the same reference re-resolving its own window is not two names"
    );
    assert_eq!(
        state.claim_backing_window(task, second, window),
        Some(first),
        "a second reference over one window is the sighting the reading is for"
    );
    assert_eq!(
        state.claim_backing_window(task, second, window),
        Some(first),
        "the holder does not change hands, so a repeat reports the same pair \
         rather than the two references taking turns"
    );

    // The window is task-local. Two tasks name their own address spaces, and a
    // number they happen to share is not one piece of storage -- the same
    // reason an object ref in task B is never tried as task A's mapping id.
    assert_eq!(
        state.claim_backing_window(other_task, second, window),
        None,
        "one number in two address spaces is two windows"
    );

    // A task teardown ends every name in it, so the claims go with them. An
    // entry outliving its namespace would answer for a window nothing names.
    state.delete_task(task);
    assert_eq!(
        state.claim_backing_window(task, second, window),
        None,
        "the teardown released the window, so the next claimant is the first"
    );
}

/// Two textures at different offsets in one allocation are one backing.
///
/// The descriptor states two addresses and they are not interchangeable:
/// `handle << page_shift` is where the allocation begins and `+ data_offset`
/// is where the texels begin. A backing identity taken from the texel base
/// would give one allocation as many identities as it has tenants — and the
/// hazard edge between two tenants that write over each other would then never
/// be drawn, which is the false-distinctness direction. So the window is the
/// allocation's, and the offset belongs to the extent.
///
/// This also protects the alias reading itself: keyed on the texel base, two
/// textures sharing an allocation would never collide and the reading would be
/// silent for exactly the case it exists to find.
#[test]
fn two_textures_in_one_allocation_have_one_window() {
    use crate::runtime::decode::resource::{
        LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_TYPE_TEXTURE, TEXTURE_DESC_DATA_OFFSET,
        TEXTURE_DESC_GEOMETRY_LEN,
    };

    let handle = 0x20u32;
    let texture = |data_offset: u32| {
        let mut desc = vec![0u8; TEXTURE_DESC_GEOMETRY_LEN];
        desc[LINEAR_DESC_SIZE..LINEAR_DESC_SIZE + 8].copy_from_slice(&0x8000u64.to_le_bytes());
        st32(&mut desc[LINEAR_DESC_HANDLE..], handle);
        st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], data_offset);
        desc
    };
    let entry = ListObjectEntry {
        object_type: OBJECT_TYPE_TEXTURE,
        descriptor_length: TEXTURE_DESC_GEOMETRY_LEN as u32,
        descriptor_gva: 0x1000,
    };

    let allocation = u64::from(handle) << PAGE_SHIFT_X86;
    assert_eq!(
        super::backing_window(PAGE_SHIFT_X86, &entry, &texture(0)),
        Some((allocation, 0x8000)),
        "a texture at offset zero begins where its allocation does"
    );
    assert_eq!(
        super::backing_window(PAGE_SHIFT_X86, &entry, &texture(0x2000)),
        Some((allocation, 0x8000)),
        "a texture placed further into the same allocation is the same backing, \
         and reading its texel base here would say it is a different one"
    );
}

/// Only the two object types that name storage by an address in their own task
/// produce a window.
///
/// A view, a function, a serializer object and a mapper-ref texture each name
/// storage through something else or name none at all, and a window invented
/// for one of them would be a window over another object's bytes -- which is
/// the false-equality direction, the one that hands memory back under a live
/// reader.
#[test]
fn only_address_named_object_types_have_a_window() {
    use crate::runtime::decode::resource::{
        OBJECT_TYPE_BUFFER, OBJECT_TYPE_FUNCTION, OBJECT_TYPE_MAPPER_REF_TEXTURE,
        OBJECT_TYPE_SERIALIZER_OBJECT, OBJECT_TYPE_TEXTURE_VIEW,
    };

    // A linear descriptor: size then handle, which is the whole of what the
    // window is built from.
    let mut buffer_desc = [0u8; 16];
    buffer_desc[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
    buffer_desc[8..16].copy_from_slice(&0x20u64.to_le_bytes());

    let entry = |object_type: u8| ListObjectEntry {
        object_type,
        descriptor_length: buffer_desc.len() as u32,
        descriptor_gva: 0x1000,
    };

    assert_eq!(
        super::backing_window(PAGE_SHIFT_X86, &entry(OBJECT_TYPE_BUFFER), &buffer_desc),
        Some((0x20u64 << PAGE_SHIFT_X86, 0x3000)),
        "a buffer names its backing by handle and size"
    );
    for named_elsewhere in [
        OBJECT_TYPE_FUNCTION,
        OBJECT_TYPE_SERIALIZER_OBJECT,
        OBJECT_TYPE_TEXTURE_VIEW,
        OBJECT_TYPE_MAPPER_REF_TEXTURE,
    ] {
        assert_eq!(
            super::backing_window(PAGE_SHIFT_X86, &entry(named_elsewhere), &buffer_desc),
            None,
            "type {named_elsewhere} does not name storage by an address in its task, \
             so it has no window of its own however the bytes happen to read"
        );
    }

    // A handle of zero is not a window at address zero. The descriptor's own
    // arithmetic refuses it, and this is where that refusal is relied on.
    let mut unpublished = buffer_desc;
    unpublished[8..16].copy_from_slice(&0u64.to_le_bytes());
    assert_eq!(
        super::backing_window(PAGE_SHIFT_X86, &entry(OBJECT_TYPE_BUFFER), &unpublished),
        None,
        "a descriptor whose handle the guest has not written names no window"
    );
}

/// One name after another over one window is not two names at once.
///
/// The guest frees an object by writing over its own object-list record, with
/// no packet at all, and then reuses the pages. This device is never told, so
/// its construction cache goes on holding the freed reference — which is why
/// the alias reading asks the *guest's* list whether the holder is still there
/// and not its own cache. Without that, every recycled allocation on a busy
/// guest reads as two names over one backing, and the reading that decides
/// where the incarnation lives would be decided by a confound.
#[test]
fn a_window_whose_holder_the_guest_freed_is_not_an_alias() {
    use crate::runtime::drain::census::store_route_count;
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, first, second) = (1u32, 2u32, 3u32);

    // One linear descriptor, past the eight list entries, named by both
    // references in turn.
    let desc_gva = 0x100u64;
    let handle = 0x20u64;
    let window = handle << PAGE_SHIFT_ARM64E;
    let mut desc = [0u8; 16];
    desc[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
    desc[8..16].copy_from_slice(&handle.to_le_bytes());
    let _ = host.write_gpa(data_gpa + desc_gva, &desc);

    let write_slot = |host: &mut FakeHost, ref_: u32, present: bool| {
        let mut entry = [0u8; 12];
        if present {
            st32(
                &mut entry[0..],
                u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
            );
            entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        }
        let _ = host.write_gpa(data_gpa + u64::from(ref_) * 12, &entry);
    };
    write_slot(&mut host, first, true);

    assert!(
        resolve_resource(&state, &host, task, first).is_ok(),
        "the first reference constructs and claims the window"
    );
    write_slot(&mut host, second, true);
    let collisions = store_route_count("backing_window_collision");
    let reclaims = store_route_count("backing_window_reclaimed");
    assert_eq!(
        super::note_backing_window_alias(&state, &host, task, second, window, 0x3000),
        Some(first),
        "both references are in the guest's list naming one window, which is \
         the sighting this reading exists to make"
    );
    assert_eq!(
        (
            store_route_count("backing_window_collision") - collisions,
            store_route_count("backing_window_reclaimed") - reclaims,
        ),
        (1, 0),
        "the collision is counted and it was not a reclaim, which is what makes \
         a boot's zero sightings mean no aliases rather than no collisions"
    );

    // The guest frees the first object the only way it can: it writes over its
    // own record. No packet arrives, so the construction cache still holds it.
    write_slot(&mut host, first, false);
    assert!(
        state.constructed_object(task, first).is_some(),
        "the cache cannot see a free, which is the whole difficulty"
    );
    assert_eq!(
        super::note_backing_window_alias(&state, &host, task, second, window, 0x3000),
        None,
        "the guest's list no longer names the holder, so this is a reuse"
    );
    assert_eq!(
        (
            store_route_count("backing_window_collision") - collisions,
            store_route_count("backing_window_reclaimed") - reclaims,
        ),
        (2, 1),
        "the reuse is a second collision and the one reclaim, so the two \
         counters separate what the sighting count alone cannot"
    );

    // And the window has changed hands, so a later claimant is compared
    // against the reference that actually holds it.
    assert_eq!(
        state.claim_backing_window(task, second, window),
        None,
        "the live reference took the window when the dead holder lost it"
    );
}

/// A re-point drops the host copies of every reference over the window, not
/// only of the reference the packet names.
///
/// Both host caches are keyed `(task, reference)`, and a driven macos-15 boot
/// found the compositor holding its 1920×1080 scanout allocation under **two**
/// live references at once. Dropping only the named one leaves the other
/// serving a texture built from pages the packet has just said are different
/// pages — a wrong surface with no refusal anywhere, on the one buffer this
/// device re-points.
///
/// Fails without the fix: the peer's copy survives the packet.
#[test]
fn a_repoint_drops_every_reference_over_the_window_it_moved() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, named, peer, elsewhere) = (1u32, 2u32, 3u32, 4u32);

    let write_object = |host: &mut FakeHost, ref_: u32, desc_gva: u64, handle: u64| {
        let mut desc = [0u8; 16];
        desc[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
        desc[8..16].copy_from_slice(&handle.to_le_bytes());
        let _ = host.write_gpa(data_gpa + desc_gva, &desc);
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
        );
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        let _ = host.write_gpa(data_gpa + u64::from(ref_) * 12, &entry);
    };
    // Two references over one allocation, and a third over another so the
    // sweep is shown to be selective rather than a task-wide flush.
    write_object(&mut host, named, 0x100, 0x20);
    write_object(&mut host, peer, 0x120, 0x20);
    write_object(&mut host, elsewhere, 0x140, 0x30);

    let surface = |fill: u8| crate::model::HostSurface {
        width: 4,
        height: 4,
        bgra: std::sync::Arc::new(vec![fill; 4 * 4 * 4]),
        host_gen: 1,
        producer_object_type: 0,
        last_touch: 0,
        backing: None,
        guest_holds_bytes: false,
        source_gva: 0,
    };
    for (ref_, fill) in [(named, 0xAAu8), (peer, 0xBB), (elsewhere, 0xCC)] {
        state
            .host_texture_surfaces
            .insert((task, ref_), surface(fill));
    }

    super::replace_physical(&mut state, &mut host, task, named);

    assert!(
        !state.host_texture_surfaces.contains_key(&(task, named)),
        "the named reference's copy was read from pages the guest has re-pointed"
    );
    assert!(
        !state.host_texture_surfaces.contains_key(&(task, peer)),
        "the second reference over the same allocation kept a copy of the pages \
         the packet says have already changed, and every later bind of it would \
         have been served them"
    );
    assert!(
        state.host_texture_surfaces.contains_key(&(task, elsewhere)),
        "a reference over a different allocation was not re-pointed, and a sweep \
         that dropped it would turn one packet into a task-wide cache flush"
    );
}

/// A guest object reference becomes a canonical backing identity, or a refusal
/// that says which contract term is missing.
///
/// Two references over one allocation get **one** identity — that is what the
/// dependency compiler draws its hazard edges on, and what a name-derived id
/// gets wrong in the direction that drops the edge. A re-point gives the same
/// window a **different** identity, because the pages changed and work already
/// accepted must keep the old ones.
#[test]
fn one_allocation_has_one_identity_and_a_repoint_gives_it_another() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, named, peer, elsewhere) = (1u32, 2u32, 3u32, 4u32);

    let write_object = |host: &mut FakeHost, ref_: u32, desc_gva: u64, handle: u64| {
        let mut desc = [0u8; 16];
        desc[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
        desc[8..16].copy_from_slice(&handle.to_le_bytes());
        let _ = host.write_gpa(data_gpa + desc_gva, &desc);
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
        );
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        let _ = host.write_gpa(data_gpa + u64::from(ref_) * 12, &entry);
    };
    write_object(&mut host, named, 0x100, 0x20);
    write_object(&mut host, peer, 0x120, 0x20);
    write_object(&mut host, elsewhere, 0x140, 0x30);

    let id = |state: &DeviceState, host: &FakeHost, ref_: u32| {
        super::backing_id(state, host, task, ref_).expect("a buffer names an allocation")
    };
    let before = id(&state, &host, named);
    assert_eq!(
        id(&state, &host, peer),
        before,
        "two references over one allocation are one backing, and an identity \
         that told them apart would drop the hazard edge between them"
    );
    assert_ne!(
        id(&state, &host, elsewhere),
        before,
        "a different allocation is different storage"
    );
    assert_eq!(
        id(&state, &host, named),
        before,
        "asking twice is not an event, and an identity that moved when it was \
         read would never compare equal to itself"
    );

    super::replace_physical(&mut state, &mut host, task, named);
    let after = id(&state, &host, named);
    assert_ne!(
        after, before,
        "the packet says these are different pages, and an equal identity would \
         let a claim on the old ones be satisfied by the new"
    );
    assert_eq!(
        id(&state, &host, peer),
        after,
        "the second name over the window moved with it"
    );

    // The mapper-ref texture the fixture lists at reference 1 reaches its
    // storage through a mapping, which has its own counter. The reference is
    // followed to the mapping its descriptor names — 9 — and what comes back is
    // that mapping's refusal, not a number derived from an address it never had.
    assert_eq!(
        super::backing_id(&state, &host, task, 1),
        Err(super::BackingIdRefusal::ThroughMapping {
            mapping_id: 9,
            refusal: super::MappingBackingRefusal::Unlisted,
        }),
        "the refusal names which mapping was asked, so a reading says what is \
         missing rather than that this type is not answered for"
    );
    assert!(state.map_surface(9));
    let through_ref = super::backing_id(&state, &host, task, 1)
        .expect("the mapping the descriptor names has a surface now");
    assert_eq!(
        through_ref,
        super::mapping_backing_id(&state, 9).expect("and it is mintable directly"),
        "the reference has the identity that mapping's storage has — one number, \
         reached the two ways a caller can hold the question"
    );
    assert_ne!(
        through_ref, after,
        "which is still not an address-named window's"
    );
}

/// A guest mapping's surface becomes a canonical backing identity, and it is
/// the mapping's generation that separates its incarnations.
///
/// The other half of the identity from
/// `one_allocation_has_one_identity_and_a_repoint_gives_it_another`, and the
/// half whose incarnation counter is not the window's. A re-point of an object
/// that owns a mapping never advances the window counter — it drops the page
/// list and bumps `map_generation` — so an identity that read the window would
/// sit still across the one packet that says the pages have moved.
///
/// The last assertion is the one that makes the two halves one identity space:
/// a mapping-reached and an address-reached piece of storage must never arrive
/// at the same number, because the dependency compiler compares the number with
/// no idea which route minted it.
#[test]
fn a_mapping_has_one_identity_and_a_replaced_page_list_gives_it_another() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let (mid, other) = (7u32, 8u32);
    // Taken first so the disjointness check at the end has a window identity
    // that predates every mapping mint.
    let first_window = state.backing_identity(1, 0x2_0000);

    assert_eq!(
        super::mapping_backing_id(&state, mid),
        Err(super::MappingBackingRefusal::Unlisted),
        "a mapping id this device holds nothing for names no storage, and a \
         number minted for it would be a number for the id itself"
    );

    // A geometry declaration creates the entry without mapping a surface into
    // it. The slot exists; the storage does not.
    assert!(state.set_mapping_geom(mid, 1920, 1080, 0));
    assert_eq!(
        super::mapping_backing_id(&state, mid),
        Err(super::MappingBackingRefusal::Unmapped),
        "an entry the guest has never mapped a surface into is still a slot \
         number rather than storage"
    );

    assert!(state.map_surface(mid));
    assert!(state.map_surface(other));
    let before = super::mapping_backing_id(&state, mid).expect("a mapped surface is storage");
    assert_eq!(
        super::mapping_backing_id(&state, mid),
        Ok(before),
        "asking twice is not an event, and an identity that moved when it was \
         read would never compare equal to itself"
    );
    let second = super::mapping_backing_id(&state, other).expect("also mapped");
    assert_ne!(second, before, "a different mapping is different storage");

    let e = state.mappings.get_mut(&mid).expect("just mapped");
    DeviceState::bump_map_generation(e);
    let after = super::mapping_backing_id(&state, mid).expect("still mapped");
    assert_ne!(
        after, before,
        "the page list under this mapping has been replaced, and an equal \
         identity would let a claim on the old pages be satisfied by the new"
    );
    assert_eq!(
        super::mapping_backing_id(&state, mid),
        Ok(after),
        "the new incarnation is an identity too, not a value that moves every \
         time it is asked for"
    );

    // Every identity either route has minted, pairwise distinct — the whole
    // reason one is a `u64` both routes may produce. `first_window` was taken
    // before any mapping was asked about, so a mapping table with its own
    // counter would have handed its first mint that very number.
    let mut minted = [
        ("the window asked for before any mapping", first_window),
        ("the mapping's first incarnation", before),
        ("a second mapping", second),
        ("the first mapping's replaced page list", after),
        (
            "a window asked for after the mapping mints",
            state.backing_identity(1, 0x3_0000),
        ),
    ];
    minted.sort_by_key(|&(_, id)| id);
    for pair in minted.windows(2) {
        assert_ne!(
            pair[0].1, pair[1].1,
            "{} and {} are unrelated storage, and storage reached by address and \
             storage reached through a mapping share one identity space — two \
             counters would make them alias",
            pair[0].0, pair[1].0
        );
    }
}

/// A dual-plane texture and a buffer over one allocation are one backing.
///
/// The consequence, at the identity, of the window the decoder now answers for
/// a two-plane object. While `Descriptor::backing_window` said `None` for it
/// this reference had no identity at all — `backing_id` classified it as
/// storage reached through a mapping, which it never was — so nothing could
/// order a read of it against a write of the buffer sharing its bytes.
#[test]
fn a_dual_plane_texture_shares_one_identity_with_a_buffer_over_its_allocation() {
    use crate::runtime::decode::resource::tests::dual_plane_body;
    use crate::runtime::decode::resource::OBJECT_TYPE_DUAL_PLANE_TEXTURE;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, planes, buffer) = (1u32, 2u32, 3u32);
    // `dual_plane_body` writes this handle into the object header both planes
    // are cut from.
    const SHARED_HANDLE: u64 = 0x51;

    let write_entry = |host: &mut FakeHost, ref_: u32, ty: u8, len: usize, desc_gva: u64| {
        let mut entry = [0u8; 12];
        st32(&mut entry[0..], u32::from(ty) | ((len as u32) << 8));
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        let _ = host.write_gpa(data_gpa + u64::from(ref_) * 12, &entry);
    };

    let body = dual_plane_body([1, 1], (1920, 1080), 0x50);
    let _ = host.write_gpa(data_gpa + 0x400, &body);
    write_entry(
        &mut host,
        planes,
        OBJECT_TYPE_DUAL_PLANE_TEXTURE,
        body.len(),
        0x400,
    );

    let mut buffer_desc = [0u8; 16];
    buffer_desc[0..8].copy_from_slice(&0x40_0000u64.to_le_bytes());
    buffer_desc[8..16].copy_from_slice(&SHARED_HANDLE.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 0x500, &buffer_desc);
    write_entry(
        &mut host,
        buffer,
        OBJECT_TYPE_BUFFER,
        buffer_desc.len(),
        0x500,
    );

    let id = |ref_: u32| super::backing_id(&state, &host, task, ref_);
    assert_eq!(
        id(planes),
        id(buffer),
        "two names for one allocation are one backing, whether the guest cut \
         two planes out of it or a flat buffer"
    );
    assert!(
        id(planes).is_ok(),
        "and both of them have an identity: a two-plane texture is address-named \
         like any other normal texture, so refusing it as mapping-named withheld \
         one the device had"
    );
}

/// Every constructed object is counted against the identity, and each refusal
/// is counted under its own name.
///
/// The census has no consumer to fail if it stops working — the identity it
/// measures has none yet either — so this is what says it is still measuring.
/// The denominator is the point: a boot with no refusals and a boot with no
/// constructions read the same without it.
#[test]
fn every_construction_is_counted_against_the_backing_identity() {
    use crate::runtime::drain::store_route_count;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, buffer) = (1u32, 2u32);

    let mut desc = [0u8; 16];
    desc[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
    desc[8..16].copy_from_slice(&0x20u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 0x100, &desc);
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x100u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + u64::from(buffer) * 12, &entry);

    let asked = store_route_count("backing_identity_asked");
    let minted = store_route_count("backing_identity_minted");
    resolve_resource(&state, &host, task, buffer).expect("a buffer constructs");
    assert_eq!(
        store_route_count("backing_identity_asked"),
        asked + 1,
        "the denominator moves for every construction, whatever the answer"
    );
    assert_eq!(
        store_route_count("backing_identity_minted"),
        minted + 1,
        "a buffer names an allocation, so it has an identity"
    );

    // The fixture's reference 1 is a mapper-ref texture over a mapping this
    // state has never mapped a surface into, which is the mapping's refusal
    // rather than this object's.
    let through = store_route_count("backing_id_mapping_names_no_surface");
    resolve_resource(&state, &host, task, 1).expect("a mapper-ref texture constructs");
    assert_eq!(
        store_route_count("backing_id_mapping_names_no_surface"),
        through + 1,
        "and a refusal is counted under the arm it took, not as one number that \
         cannot say which term is missing"
    );
    assert_eq!(
        store_route_count("backing_identity_asked"),
        asked + 2,
        "an object with no identity is still in the denominator"
    );
}

/// A surface-backing object declares the mapping's storage, because its
/// object-list slot *is* the mapping id.
///
/// It used to declare `NoBytes`, on the grounds that the address route names no
/// storage for `OBJECT_TYPE_BACKING` — which is true and is not the whole
/// answer, because the mapping route does. A driven macos-15 boot measured the
/// cost: every one of 36 re-points the semantic model was never told about was
/// a backing object whose slot also named a live mapping surface.
///
/// The type is what makes the shared integer one namespace rather than two.
/// `backing_claimant_tasks` already finds a surface's owner by probing exactly
/// this slot for exactly this type.
#[test]
fn a_surface_backing_object_declares_the_storage_of_the_mapping_at_its_own_slot() {
    use reims_vgpu_core::access::ByteRange;
    use reims_vgpu_core::lifecycle::Storage;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, surface) = (1u32, 3u32);

    // The backing's own record: length and first page frame, no planes.
    let mut desc = [0u8; 0x20];
    desc[0..8].copy_from_slice(&0x8000u64.to_le_bytes());
    st32(&mut desc[8..], 0x1234);
    let _ = host.write_gpa(data_gpa + 0x100, &desc);
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_BACKING) | (0x20u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x100u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + u64::from(surface) * 12, &entry);

    let listed = lookup_list_entry(&state, &host, task, surface).expect("listed");
    let bytes = read_descriptor(&state, &host, task, &listed).expect("descriptor");

    // Before the guest maps a surface into that slot there is storage to name
    // and no identity for it, and the refusal is the mapping's own, carried.
    assert_eq!(
        super::declared_storage(&state, task, surface, &listed, &bytes),
        Err(super::StorageRefusal::Backing(
            super::BackingIdRefusal::ThroughMapping {
                mapping_id: surface,
                refusal: super::MappingBackingRefusal::Unlisted,
            }
        )),
        "not `NoBytes` — the object owns bytes and the mapper has not been told \
         about them yet, which are different facts"
    );

    assert!(state.map_surface(surface));
    assert_eq!(
        super::declared_storage(&state, task, surface, &listed, &bytes),
        Ok(Storage::Dedicated {
            backing: super::mapping_backing_id(&state, surface).expect("mapped"),
            extent: ByteRange {
                offset: 0,
                length: 0x8000,
            },
        }),
        "the identity is the mapping's, so a mapper-ref texture over the same \
         surface aliases it by construction; the extent is the whole backing, \
         because a backing has no window of anything larger"
    );

    // And that is what a re-point of it now hands the model.
    let (_, backing, extent) =
        super::repointed_storage(&state, &host, task, surface).expect("a backing names storage");
    assert_eq!(
        (backing, extent.length),
        (
            super::mapping_backing_id(&state, surface).expect("mapped"),
            0x8000
        ),
    );
}

/// A re-point that the semantic model is not told about says which of six
/// facts stopped it, because only one of them costs anything.
///
/// The caller used to count one `replace_physical_storage_undescribable`, and a
/// driven boot put 34 of 204 re-points in it. A reference that owns no bytes
/// and a reference whose real storage this device has no vocabulary for read
/// identically there, and they are a non-event and a stale-content hazard.
#[test]
fn a_repoint_the_model_is_not_told_about_names_which_fact_stopped_it() {
    use super::RepointStorageRefusal;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, buffer) = (1u32, 2u32);

    let mut desc = [0u8; 16];
    desc[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
    desc[8..16].copy_from_slice(&0x20u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 0x100, &desc);
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x100u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + u64::from(buffer) * 12, &entry);

    let (name, backing, extent) = super::repointed_storage(&state, &host, task, buffer)
        .expect("a buffer names storage the model can be told moved");
    assert_eq!(
        (name, backing),
        (
            super::name_resource(&state, &host, task, buffer).expect("named"),
            super::backing_id(&state, &host, task, buffer).expect("a buffer has one"),
        ),
        "and the operation carries the reference's own name and identity"
    );
    assert_eq!(extent.length, 0x3000);

    // Reference 1 is a mapper-ref texture over mapping 9. Its surface has an
    // identity and its plane within that surface does not, so this device holds
    // real storage it cannot describe: the one refusal that costs.
    assert!(state.map_surface(9));
    let refusal = super::repointed_storage(&state, &host, task, 1)
        .expect_err("the plane of the surface is unrecovered");
    assert_eq!(
        refusal,
        RepointStorageRefusal::Refused(super::StorageRefusal::ExtentUnrecovered {
            object_type: 11
        }),
        "carried whole from the owner that has the term, not re-worded"
    );
    assert_eq!(
        refusal.route(),
        "replace_physical_storage_extent_unrecovered"
    );
    assert!(
        refusal.leaves_authority_stale(),
        "the pages moved and the model was not told, so its content authority \
         for them stays on the pages the guest has already rewired"
    );

    // A reference the guest's list has no entry at stops one step earlier, and
    // that is a re-point with nothing behind it rather than a lost one.
    let unlisted = super::repointed_storage(&state, &host, task, 4000)
        .expect_err("nothing is listed at that reference");
    assert!(
        !unlisted.leaves_authority_stale(),
        "there is no storage for authority to be stale about"
    );
    assert_ne!(
        unlisted.route(),
        refusal.route(),
        "and the two are counted apart, which is the whole reason for the enum"
    );
}

/// The object-list walk's per-object translation: what the semantic model is
/// told each constructed object's storage is.
///
/// Three answers, and the third is the one the model could not represent at all
/// until the namespace learned that a name may own no memory. Most of a guest's
/// list is that third answer.
#[test]
fn a_constructed_object_becomes_the_storage_the_model_is_declared_with() {
    use reims_vgpu_core::access::ByteRange;
    use reims_vgpu_core::lifecycle::Storage;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, buffer) = (1u32, 2u32);

    let mut desc = [0u8; 16];
    desc[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
    desc[8..16].copy_from_slice(&0x20u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 0x100, &desc);
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x100u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + u64::from(buffer) * 12, &entry);

    let list_entry = lookup_list_entry(&state, &host, task, buffer).expect("listed");
    let bytes = read_descriptor(&state, &host, task, &list_entry).expect("descriptor");
    assert_eq!(
        super::declared_storage(&state, task, buffer, &list_entry, &bytes),
        Ok(Storage::Dedicated {
            backing: super::backing_id(&state, &host, task, buffer).expect("a buffer has one"),
            extent: ByteRange {
                offset: 0,
                length: 0x3000,
            },
        }),
        "a buffer is its allocation, and the declaration carries the identity \
         of the storage and the object's own window of it"
    );

    // The fixture's reference 1 is a mapper-ref texture over mapping 9. Its
    // storage is the mapping's and has an identity; which plane of that surface
    // the texture is does not come from this record.
    let t11 = lookup_list_entry(&state, &host, task, 1).expect("listed");
    let t11_bytes = read_descriptor(&state, &host, task, &t11).expect("descriptor");
    assert!(state.map_surface(9));
    assert_eq!(
        super::declared_storage(&state, task, 1, &t11, &t11_bytes),
        Err(super::StorageRefusal::ExtentUnrecovered { object_type: 11 }),
        "until the mapping publishes a device descriptor, nothing says which \
         part of its surface this texture is — and the identity being there is \
         a different fact from the extent being there, which must not read as \
         one"
    );

    // The mapping publishes its surface. The texture is 64x32 at format 0x50
    // (BGRA8, four bytes a pixel), so its plane is 64*4 to a row and the whole
    // of it is this texture's.
    let mut device_desc = vec![0u8; crate::protocol::iosurface_pages::DEVICE_DESC_LEN];
    st64(
        &mut device_desc[crate::protocol::iosurface_pages::DEVICE_DESC_DIMS..],
        (64u64 << 8) | (32u64 << 40),
    );
    st32(
        &mut device_desc[crate::protocol::iosurface_pages::DEVICE_DESC_BPR..],
        64 * 4,
    );
    assert!(state.set_mapping_device_desc(9, &device_desc));
    assert_eq!(
        super::declared_storage(&state, task, 1, &t11, &t11_bytes),
        Ok(Storage::Dedicated {
            backing: super::mapping_backing_id(&state, 9).expect("a mapped surface"),
            extent: ByteRange {
                offset: 0,
                length: 64 * 4 * 32,
            },
        }),
        "the storage is the mapping's and the extent is this texture's plane of \
         it — the surface's own bytes, not an allocation of the texture's own"
    );

    // The case the offset is load-bearing in: a two-plane surface where this
    // texture is the *second* plane. Its bytes start past the surface base, and
    // an extent anchored at zero would claim the first plane's pixels as this
    // texture's content.
    use crate::protocol::iosurface_pages::{
        DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT, DEVICE_PLANE_BPE, DEVICE_PLANE_BPR,
        DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS, DEVICE_PLANE_OFFSET,
    };
    const PLANE1_AT: u64 = 0x1_0000;
    device_desc[DEVICE_DESC_PLANE_COUNT] = 2;
    for (i, (w, h, bpe, at)) in [(128u32, 64u32, 4u16, 0u32), (64, 32, 4, PLANE1_AT as u32)]
        .into_iter()
        .enumerate()
    {
        let base = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
        st32(&mut device_desc[base + DEVICE_PLANE_OFFSET..], at);
        st64(
            &mut device_desc[base + DEVICE_PLANE_DIMS..],
            (u64::from(w) << 8) | (u64::from(h) << 40),
        );
        st32(
            &mut device_desc[base + DEVICE_PLANE_BPR..],
            w * u32::from(bpe),
        );
        st16(&mut device_desc[base + DEVICE_PLANE_BPE..], bpe);
    }
    assert!(state.set_mapping_device_desc(9, &device_desc));
    assert_eq!(
        super::declared_storage(&state, task, 1, &t11, &t11_bytes),
        Ok(Storage::Dedicated {
            backing: super::mapping_backing_id(&state, 9).expect("a mapped surface"),
            extent: ByteRange {
                offset: PLANE1_AT,
                length: 64 * 4 * 32,
            },
        }),
        "the second plane's bytes start past the surface base, and an extent \
         anchored at zero would claim the first plane's pixels"
    );
}

/// The stale-resolution witness says what it examined, not only what it found.
///
/// Its report is `first_sight`-gated, so a boot with no line is a boot where
/// the guest never overwrote a slot **or** a boot where the witness never
/// compared anything, and those are opposite facts about one silence. The
/// denominator is what tells them apart — and it decides the cutover's
/// declaration discipline, because a guest that replaces objects in place needs
/// the device to redeclare and one that does not needs no such path.
#[test]
fn the_stale_resolution_witness_counts_what_it_compared() {
    use crate::runtime::drain::store_route_count;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, buffer) = (1u32, 2u32);

    let write_desc = |host: &mut FakeHost, at: u64, size: u64| {
        let mut desc = [0u8; 16];
        desc[0..8].copy_from_slice(&size.to_le_bytes());
        desc[8..16].copy_from_slice(&0x20u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + at, &desc);
    };
    let write_entry = |host: &mut FakeHost, desc_gva: u64| {
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
        );
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        let _ = host.write_gpa(data_gpa + u64::from(buffer) * 12, &entry);
    };
    write_desc(&mut host, 0x100, 0x3000);
    write_entry(&mut host, 0x100);
    resolve_resource(&state, &host, task, buffer).expect("constructs");

    // A steady retrieval: examined, and in agreement.
    let examined = store_route_count("task_resource_reexamined");
    let rewritten = store_route_count("task_resource_descriptor_rewritten");
    let repointed = store_route_count("task_resource_slot_repointed");
    resolve_resource(&state, &host, task, buffer).expect("retrieves");
    assert_eq!(
        store_route_count("task_resource_reexamined"),
        examined + 1,
        "a cache hit is a comparison, and the agreement rate is only readable \
         against the number of them"
    );
    assert_eq!(
        store_route_count("task_resource_descriptor_rewritten"),
        rewritten,
        "nothing disagreed"
    );

    // The serializer rewrites the descriptor in place: same entry, same
    // address, different object.
    write_desc(&mut host, 0x100, 0x9000);
    resolve_resource(&state, &host, task, buffer).expect("retrieves");
    assert_eq!(
        store_route_count("task_resource_descriptor_rewritten"),
        rewritten + 1,
        "the bytes at the address the entry names changed, which the entry \
         comparison alone would have called agreement"
    );
    assert_eq!(
        store_route_count("task_resource_slot_repointed"),
        repointed,
        "and it is not the other disagreement: the slot still points where it did"
    );

    // The guest points the slot at a different descriptor.
    write_desc(&mut host, 0x180, 0x9000);
    write_entry(&mut host, 0x180);
    resolve_resource(&state, &host, task, buffer).expect("retrieves");
    assert_eq!(
        store_route_count("task_resource_slot_repointed"),
        repointed + 1,
        "the guest replaced the object with no packet, which is the event the \
         model's redeclaration exists for"
    );
}

/// The other half of the reply-destination question, asked of a page frame.
///
/// `CmdGetDeviceInfo` replies into a bare guest page frame, which lies in no
/// task's address space, so the allocation scan above cannot be pointed at it.
/// The frame can still be part of a **mapping's** surface, and that is the
/// collision that matters: the reply write and every access to that surface
/// would come out over two backings and never be ordered against each other.
///
/// Both driven rails read `device_info_reply_after_an_identity = 1`, so the
/// mint-count arm never closes on a real guest and this is the arm that has to.
#[test]
fn a_device_info_reply_frame_is_measured_against_the_mappings_this_device_holds() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};

    fn pte(pfn: u32) -> u32 {
        (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
    }

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    const HELD: u32 = 0x40;
    const FREE: u32 = 0x41;

    // Before anything is minted the closing arm answers, and it answers without
    // looking at a mapping at all — which is why it cannot be the only arm.
    let before = store_route_count("device_info_reply_before_any_identity");
    super::note_device_info_reply_destination(&state, HELD);
    assert_eq!(
        store_route_count("device_info_reply_before_any_identity"),
        before + 1
    );

    // One identity handed out, so the mint count can no longer close anything.
    let _ = state.frame_backing_identity(0xdead);
    assert!(state.backing_identities_minted() > 0);

    state.mappings.insert(
        7,
        crate::model::MappingEntry {
            mapped: true,
            page_entries: vec![pte(0x10), pte(HELD), pte(0x11)],
            ..Default::default()
        },
    );
    // An unmapped entry's page list names pages it no longer holds, so it is
    // not live storage and must not produce a finding.
    state.mappings.insert(
        8,
        crate::model::MappingEntry {
            mapped: false,
            page_entries: vec![pte(FREE)],
            ..Default::default()
        },
    );

    let inside = store_route_count("device_info_reply_frame_in_a_mapping");
    let outside = store_route_count("device_info_reply_frame_in_no_mapping");

    super::note_device_info_reply_destination(&state, HELD);
    assert_eq!(
        store_route_count("device_info_reply_frame_in_a_mapping"),
        inside + 1,
        "the reply page is one of a live mapping's surface pages"
    );

    super::note_device_info_reply_destination(&state, FREE);
    assert_eq!(
        store_route_count("device_info_reply_frame_in_no_mapping"),
        outside + 1,
        "an unmapped mapping's stale page list is not storage anything holds"
    );

    super::note_device_info_reply_destination(&state, 0x99);
    assert_eq!(
        store_route_count("device_info_reply_frame_in_no_mapping"),
        outside + 2,
        "and a page no mapping ever named is the closing answer"
    );
}

/// A query's reply buffer is either part of an allocation this device
/// identifies or it is not, and the instrument says which — with the number it
/// examined.
///
/// The two answers demand different things of the cutover. A reply inside an
/// allocation that got a backing of its own would leave the reply write
/// unordered against every access to that object; one outside every allocation
/// that was resolved to one anyway would be ordered against memory it does not
/// touch. Only a driven guest can say which error is available, and only if the
/// instrument can tell "nothing overlapped" from "nothing was examined".
#[test]
fn a_query_reply_destination_is_measured_against_the_allocations_this_device_holds() {
    use crate::runtime::drain::store_route_count;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, buffer) = (1u32, 2u32);
    // Allocation base 0x20 << 14, one page long.
    const BASE: u64 = 0x20u64 << PAGE_SHIFT_ARM64E;
    const SIZE: u64 = 0x1000;

    let mut desc = [0u8; 16];
    desc[0..8].copy_from_slice(&SIZE.to_le_bytes());
    desc[8..16].copy_from_slice(&0x20u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 0x100, &desc);
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x100u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + u64::from(buffer) * 12, &entry);
    resolve_resource(&state, &host, task, buffer).expect("constructs");

    let scanned = store_route_count("query_reply_scanned");
    let outside = store_route_count("query_reply_outside_every_allocation");
    assert_eq!(
        super::note_query_reply_destination(&state, task, BASE + 0x10, 64),
        Some(buffer),
        "a reply buffer inside the allocation names the reference that holds it"
    );
    assert_eq!(
        store_route_count("query_reply_scanned"),
        scanned + 1,
        "the denominator moves whatever the answer is"
    );

    assert_eq!(
        super::note_query_reply_destination(&state, task, BASE + SIZE, 64),
        None,
        "one byte past the end is outside, because the window is half open"
    );
    assert_eq!(
        store_route_count("query_reply_outside_every_allocation"),
        outside + 1
    );
    assert_eq!(
        super::note_query_reply_destination(&state, task, BASE - 8, 8),
        None,
        "and a buffer that ends exactly where the allocation starts is outside too"
    );
    assert_eq!(
        super::note_query_reply_destination(&state, task, BASE - 8, 16),
        Some(buffer),
        "but one that straddles the boundary is inside: it writes bytes the \
         allocation owns, and an identity of its own would leave those bytes \
         unordered"
    );
    assert_eq!(
        store_route_count("query_reply_scanned"),
        scanned + 4,
        "every ask is counted, including the ones that found nothing"
    );
}

/// A declaration into a task no definition opened is refused, and one into a
/// live task is not.
///
/// The resource-lifecycle group's equivalent of the channel gate G1 had to
/// answer, and it is no longer a census: `Lifecycle::create_resource` refuses
/// `NoSuchTask`, this device holds no namespace of its own to create on demand,
/// and the refusal is the answer the declaration door returns. A driven boot of
/// 3902 declarations read zero of them before the move.
#[test]
fn a_declaration_into_an_undefined_task_is_told_apart_from_one_into_a_live_task() {
    use crate::runtime::drain::store_route_count;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    // `setup_task_with_list` defines task 1 and binds its object list.
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, buffer) = (1u32, 2u32);
    let mut desc = [0u8; 16];
    desc[0..8].copy_from_slice(&0x1000u64.to_le_bytes());
    desc[8..16].copy_from_slice(&0x20u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 0x100, &desc);
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x100u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + u64::from(buffer) * 12, &entry);

    let defined = store_route_count("object_declared_into_a_defined_task");
    let undefined = store_route_count("object_declared_into_an_undefined_task");
    resolve_resource(&state, &host, task, buffer).expect("constructs");
    assert_eq!(
        (
            store_route_count("object_declared_into_a_defined_task"),
            store_route_count("object_declared_into_an_undefined_task"),
        ),
        (defined + 1, undefined),
        "a live task is the case the lifecycle owner also admits"
    );

    // The same declaration in a task nothing defined. The owner refuses it,
    // and the door says so with the rung below `no_list_entry`: the reference is
    // a slot in no namespace at all.
    let undescribable = store_route_count("object_declared_with_undescribable_storage");
    assert_eq!(
        super::declare_object_name(&state, 99, 7, None),
        Err(super::LadderRung::NoTaskSpace),
        "a task nothing defined has no address space for a name to live in"
    );
    assert_eq!(
        store_route_count("object_declared_with_undescribable_storage"),
        undescribable + 1,
        "and storage it could not describe is still counted rather than claimed as \
         `NoBytes` without a word — the refusal is about the task, not the bytes"
    );
    assert_eq!(
        (
            store_route_count("object_declared_into_a_defined_task"),
            store_route_count("object_declared_into_an_undefined_task"),
        ),
        (defined + 1, undefined + 1),
        "and the refusal is on the same counter the census read zero on"
    );
    assert_eq!(
        state.object_name(99, 7),
        None,
        "nothing was named, which is the difference the census was measuring"
    );
}

/// A task redefinition reads whether its root moved from the model, not from a
/// second copy of the previous directory.
///
/// The two routes decide how bad a redefinition is: at the same root the task
/// re-declares the space it already had, and at a different one everything the
/// guest published into the old space reads back as whatever the new pages
/// hold. `DeviceState::define_task` used to answer that from
/// `TaskEntry::directory_pfn` — its own record, beside the owner's — and this
/// asserts it now answers from `Redefinition::root_moved`.
#[test]
fn a_task_redefinition_reads_its_root_move_from_the_model() {
    use crate::runtime::drain::store_route_count;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let same = store_route_count("define_task_redefined_live_same_root");
    let moved = store_route_count("define_task_redefined_live_new_root");

    // A first definition is not a redefinition: there is no previous space for
    // a root to have moved from.
    state.define_task(4, 0x1000, 0x20);
    assert_eq!(
        (
            store_route_count("define_task_redefined_live_same_root"),
            store_route_count("define_task_redefined_live_new_root"),
        ),
        (same, moved),
        "a first definition replaces nothing"
    );

    state.define_task(4, 0x1000, 0x20);
    assert_eq!(
        (
            store_route_count("define_task_redefined_live_same_root"),
            store_route_count("define_task_redefined_live_new_root"),
        ),
        (same + 1, moved),
        "the same root re-declares the space the task already had"
    );

    state.define_task(4, 0x1000, 0x21);
    assert_eq!(
        (
            store_route_count("define_task_redefined_live_same_root"),
            store_route_count("define_task_redefined_live_new_root"),
        ),
        (same + 1, moved + 1),
        "and a different root is a different address space under one id"
    );
}

/// A name is retired by the model, and the teardown the device routes on is the
/// model's answer.
///
/// Two facts in one test because they are one event. `DeviceState::delete_object`
/// asks the owner what the departing name owed, and the owner is also what
/// stops the reference resolving — so a device that kept its own namespace
/// beside it could route on one answer while resolving against the other.
#[test]
fn a_deleted_name_stops_resolving_and_its_teardown_is_the_models() {
    use crate::runtime::drain::store_route_count;
    use reims_vgpu_core::lifecycle::Storage;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.define_task(6, 0x1000, 0x30);

    // An object that owns no memory: the owner answers `NoStorage`, which is
    // its own teardown variant and not a `Now` over a placeholder backing.
    let no_bytes = state
        .declare_object(6, 3, Storage::NoBytes)
        .expect("task 6 is defined")
        .id;
    assert_eq!(state.object_name(6, 3), Some(no_bytes));
    let none = store_route_count("object_teardown_no_storage");
    // The return value is "did a memo or a decoded object go with it", and a
    // name with neither answers `false`. What moved is the name, which is what
    // the two assertions below are about.
    let _ = state.delete_object(6, 3);
    assert_eq!(
        store_route_count("object_teardown_no_storage"),
        none + 1,
        "a name over no bytes owes no teardown, and says so in its own variant"
    );
    assert_eq!(
        state.object_name(6, 3),
        None,
        "and the reference resolves to nothing afterwards"
    );

    // One that owns its own pages: nothing else names them, so the teardown is
    // `Now` and the storage is the caller's to free.
    let dedicated = state
        .declare_object(
            6,
            4,
            Storage::Dedicated {
                backing: reims_vgpu_core::access::BackingId(0x55),
                extent: reims_vgpu_core::access::ByteRange {
                    offset: 0,
                    length: 0x1000,
                },
            },
        )
        .expect("task 6 is defined")
        .id;
    let now = store_route_count("object_teardown_now");
    let _ = state.delete_object(6, 4);
    assert_eq!(
        store_route_count("object_teardown_now"),
        now + 1,
        "its own pages, named by nothing else, are free the moment the name goes"
    );
    assert_eq!(state.object_name(6, 4), None);
    let _ = dedicated;
}

/// A reply destination is identified in the space its own question uses, and
/// the three spaces cannot collide.
///
/// The collision is the whole risk. A reply buffer inside an allocation this
/// device identifies must take *that* allocation's identity — given one of its
/// own, the reply write and every access to the object come out over different
/// backings and the hazard edge between them is never drawn. A page frame, which
/// no task's address space names, must take an identity that can equal no
/// window's and no mapping's; it does, because every route interns on one
/// monotone counter.
#[test]
fn a_reply_destination_is_identified_in_its_own_space_and_collides_with_no_other() {
    use reims_vgpu_core::query::{Destinations as _, QueryKind};
    use reims_vgpu_protocol::fifo::COMPUTE_INFO_REQUEST_LEN;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, buffer) = (1u32, 2u32);
    const BASE: u64 = 0x20u64 << PAGE_SHIFT_ARM64E;
    const SIZE: u64 = 0x1000;

    let mut desc = [0u8; 16];
    desc[0..8].copy_from_slice(&SIZE.to_le_bytes());
    desc[8..16].copy_from_slice(&0x20u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 0x100, &desc);
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x100u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + u64::from(buffer) * 12, &entry);
    let allocation = super::backing_id(&state, &host, task, buffer).expect("the buffer identifies");

    let compute = |gva: u64| {
        // The compute-info request's six words, in the order its decoder reads
        // them: task, pipeline, key-table length, pair capacity, then the reply
        // address as two halves.
        let mut payload = vec![0u8; COMPUTE_INFO_REQUEST_LEN];
        st32(&mut payload[0..], task);
        st32(&mut payload[4..], 3);
        st32(&mut payload[8..], 5);
        st32(&mut payload[12..], 8);
        st32(&mut payload[16..], gva as u32);
        st32(&mut payload[20..], (gva >> 32) as u32);
        payload
    };
    let names = super::TaskNames::new(&state, &host);

    // Inside the allocation: the allocation's identity, at the offset the reply
    // lies at within it.
    let inside = names
        .destination(QueryKind::ComputeInfo, &compute(BASE + 0x40))
        .expect("a reply address resolves");
    assert_eq!(
        inside.backing, allocation,
        "a reply inside an allocation must not be given an identity of its own"
    );
    assert_eq!(inside.bytes.offset, 0x40);

    // Outside every allocation: storage all the same, identified as the window
    // it is, and never equal to the allocation's.
    let outside = names
        .destination(QueryKind::ComputeInfo, &compute(BASE + SIZE + 0x8000))
        .expect("a reply address outside every allocation still names storage");
    assert_ne!(outside.backing, allocation);

    // A page frame, in the third key space. Two asks for one frame are one
    // identity, and it equals neither window identity.
    let device_info = |pfn: u32| {
        use reims_vgpu_protocol::fifo::DeviceInfoForm;
        let form = DeviceInfoForm::WithKeyLimit;
        let mut payload = vec![0u8; 64];
        st32(&mut payload[form.reply_pfn_offset()..], pfn);
        st32(&mut payload[form.pair_capacity_offset()..], 512);
        payload
    };
    let frame = names
        .destination(QueryKind::DeviceInfo, &device_info(0x1a7dc))
        .expect("a page frame is storage");
    assert_eq!(
        frame.backing,
        names
            .destination(QueryKind::DeviceInfo, &device_info(0x1a7dc))
            .expect("asked twice")
            .backing,
        "one frame is one identity"
    );
    assert_ne!(frame.backing, allocation);
    assert_ne!(frame.backing, outside.backing);
    assert_ne!(
        frame.backing,
        names
            .destination(QueryKind::DeviceInfo, &device_info(0x1a7dd))
            .expect("a second frame")
            .backing,
        "two frames are two identities"
    );
}

/// The device-info reply census tells "before any identity was minted" from
/// "after one was", and moves its denominator either way.
///
/// The distinction is the whole instrument. `CmdGetDeviceInfo`'s destination is
/// a guest page frame that can only be given a minted identity, and whether
/// that is sound turns on the identity being equal to nothing else. A census
/// that recorded only the safe case would read identically on a boot where the
/// guest never asked at all, and the term would look closed by silence.
///
/// The population it counts is deliberately the mint counter and not the task
/// or resource tables: a driven boot found one live task and zero resources at
/// reply time, and a census keyed on either would have called that the open
/// case. A task is a namespace and has interned nothing.
#[test]
fn the_device_info_reply_census_separates_a_mint_free_device_from_one_that_has_minted() {
    use crate::runtime::drain::store_route_count;

    let scanned = store_route_count("device_info_reply_scanned");
    let before = store_route_count("device_info_reply_before_any_identity");
    let after = store_route_count("device_info_reply_after_an_identity");

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    super::note_device_info_reply_destination(&state, 0x20);
    assert_eq!(
        (
            store_route_count("device_info_reply_scanned"),
            store_route_count("device_info_reply_before_any_identity"),
            store_route_count("device_info_reply_after_an_identity"),
        ),
        (scanned + 1, before + 1, after),
        "a device that has minted nothing has nothing a fresh identity could \
         equal, and says so"
    );

    // A task on its own interns nothing, so the answer must not move. This is
    // the reading a driven boot falsified for the population census that came
    // before this one.
    setup_task_with_list(&mut host, &mut state);
    super::note_device_info_reply_destination(&state, 0x21);
    assert_eq!(
        (
            store_route_count("device_info_reply_before_any_identity"),
            store_route_count("device_info_reply_after_an_identity"),
        ),
        (before + 2, after),
        "a live task is a namespace, not storage: nothing has been interned and \
         the closing answer still holds"
    );

    // One constructed resource, which is what actually mints.
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, buffer) = (1u32, 2u32);
    let mut desc = [0u8; 16];
    desc[0..8].copy_from_slice(&0x1000u64.to_le_bytes());
    desc[8..16].copy_from_slice(&0x20u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 0x100, &desc);
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
    );
    entry[4..12].copy_from_slice(&0x100u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + u64::from(buffer) * 12, &entry);
    resolve_resource(&state, &host, task, buffer).expect("constructs");
    assert!(
        state.backing_identities_minted() > 0,
        "the fixture must actually mint, or the arm below asserts nothing"
    );

    super::note_device_info_reply_destination(&state, 0x22);
    assert_eq!(
        (
            store_route_count("device_info_reply_scanned"),
            store_route_count("device_info_reply_before_any_identity"),
            store_route_count("device_info_reply_after_an_identity"),
        ),
        (scanned + 3, before + 2, after + 1),
        "with an identity handed out the answer is the open one, and the \
         denominator moved for all three asks"
    );
}

/// The device answers the model's mapping resolver, and it answers with the
/// same identity the mapping route mints.
///
/// Two facts, and the second is the one worth a test: the trait is the model's
/// door into the mapping namespace, and a door that answered with a *different*
/// number from the one the device's own callers get would give the dependency
/// compiler a backing nothing else in the device shares.
#[test]
fn the_device_answers_the_models_mapping_resolver_with_the_identity_it_mints() {
    use reims_vgpu_core::identity::MappingId;
    use reims_vgpu_core::resolve::MappingResolver as _;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert_eq!(
        state.backing(MappingId(9)),
        None,
        "a mapping id this device lists nothing for names no live surface"
    );

    assert!(state.set_mapping_geom(9, 64, 32, 0x50));
    assert_eq!(
        state.backing(MappingId(9)),
        None,
        "and an entry a geometry declaration created, with no surface mapped \
         into it, still names none"
    );

    assert!(state.map_surface(9));
    let minted = super::mapping_backing_id(&state, 9).expect("a mapped surface is storage");
    assert_eq!(
        state.backing(MappingId(9)),
        Some(minted),
        "the model gets the identity the device's own callers get, not a \
         second number for one piece of storage"
    );

    let e = state.mappings.get_mut(&9).expect("mapped");
    DeviceState::bump_map_generation(e);
    assert_ne!(
        state.backing(MappingId(9)),
        Some(minted),
        "and it moves with the page list, because that is what the identity is"
    );
}

/// A guest reference reaches this device's retained bytes only through the name
/// the namespace issued, and a slot the guest reuses is a different name.
///
/// This is the invariant the memo's key exists for. A cache keyed by the slot
/// number outlives the object it was built for — the guest replaces an object by
/// writing over its own object-list record, with no packet — and a stale hit
/// binds the bytes of whatever used to live there, which is a wrong texture
/// rather than a missing one. Keyed by a name whose generation advances on every
/// declaration, the stale entry cannot be spelled.
#[test]
fn a_reference_reaches_its_bytes_only_through_the_name_the_namespace_issued() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let (task, ref_) = (1u32, 1u32);

    assert_eq!(
        state.object_name(task, ref_),
        None,
        "a reference this device has never constructed names nothing"
    );
    assert!(
        state.constructed_object(task, ref_).is_none(),
        "and there is no door to the memo that does not go through a name"
    );

    let built = resolve_resource(&state, &host, task, ref_).expect("construction");
    let first = state
        .object_name(task, ref_)
        .expect("construction names it");
    assert!(
        std::sync::Arc::ptr_eq(&state.constructed_object(task, ref_).expect("memo"), &built),
        "the memo answers under the name the declaration issued"
    );
    assert!(
        std::sync::Arc::ptr_eq(
            &resolve_resource(&state, &host, task, ref_).expect("retrieval"),
            &built
        ),
        "and a second resolution retrieves it rather than constructing again"
    );

    // The guest deletes the object. The name stops resolving, so the retained
    // bytes are unreachable — not because they were removed, but because
    // nothing can name them.
    assert!(state.delete_object(task, ref_));
    assert_eq!(state.object_name(task, ref_), None);
    assert!(state.constructed_object(task, ref_).is_none());
    assert!(
        state.task_resources.get(task, first).is_none(),
        "the memo goes with the name, which is prompt rather than load-bearing"
    );

    // The slot is used again. It is a different name, and the generation is
    // what says so.
    let rebuilt = resolve_resource(&state, &host, task, ref_).expect("reconstruction");
    let second = state.object_name(task, ref_).expect("named again");
    assert_ne!(
        second, first,
        "a reused slot is a new generation, which is the whole reason the memo \
         is not keyed by the slot"
    );
    assert_eq!(second.slot, first.slot, "and the same slot");
    assert!(
        !std::sync::Arc::ptr_eq(&rebuilt, &built),
        "the second occupant is its own object"
    );

    // A task teardown ends every name in it.
    assert!(state.delete_task(task));
    assert_eq!(
        state.object_name(task, ref_),
        None,
        "the address space ended, and every name in it with it"
    );
}

/// The peer question the hot per-reference state asks is answered from the
/// construction cache, and answers nothing when the reference is the only name
/// for its storage.
///
/// `cached_window_peer` decides whether a `(task, reference)` key may go on
/// standing for storage. Getting it wrong in the quiet direction would say a
/// keying is sound when it is not, so each assertion is one way it could go
/// quiet: a reference with no construction, a construction with no window, and
/// a genuine neighbour over a different allocation.
#[test]
fn the_peer_question_answers_only_for_a_shared_allocation() {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let (task, named, peer, elsewhere, unbuilt) = (1u32, 2u32, 3u32, 4u32, 5u32);

    let write_object = |host: &mut FakeHost, ref_: u32, desc_gva: u64, handle: u64| {
        let mut desc = [0u8; 16];
        desc[0..8].copy_from_slice(&0x3000u64.to_le_bytes());
        desc[8..16].copy_from_slice(&handle.to_le_bytes());
        let _ = host.write_gpa(data_gpa + desc_gva, &desc);
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            u32::from(OBJECT_TYPE_BUFFER) | (16u32 << 8),
        );
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        let _ = host.write_gpa(data_gpa + u64::from(ref_) * 12, &entry);
    };
    write_object(&mut host, named, 0x100, 0x20);
    write_object(&mut host, elsewhere, 0x140, 0x30);
    write_object(&mut host, unbuilt, 0x160, 0x20);

    // Only the constructed references count. `unbuilt` is listed over the very
    // same allocation as `named` and is deliberately never resolved, because
    // the caches this question is asked on behalf of hold nothing for a
    // reference the device has not constructed.
    for ref_ in [named, elsewhere] {
        assert!(resolve_resource(&state, &host, task, ref_).is_ok());
    }
    assert_eq!(
        super::cached_window_peer(&state, task, named),
        None,
        "a listed-but-unconstructed reference over the same allocation is not a \
         second name for anything this device is keeping state under"
    );

    // Now a second constructed reference over the same allocation.
    write_object(&mut host, peer, 0x120, 0x20);
    assert!(resolve_resource(&state, &host, task, peer).is_ok());
    assert_eq!(
        super::cached_window_peer(&state, task, named),
        Some(peer),
        "two constructed references over one allocation is the reading that says \
         a per-reference key has stopped standing for storage"
    );
    assert_eq!(
        super::cached_window_peer(&state, task, peer),
        Some(named),
        "the question is symmetric, so neither name is privileged"
    );
    assert_eq!(
        super::cached_window_peer(&state, task, elsewhere),
        None,
        "a reference over its own allocation is the only name for it"
    );

    // The fixture's mapper-ref texture at reference 1 reaches its storage
    // through a mapping and has no window at all, so it is nobody's peer and
    // has none -- rather than matching every other windowless object.
    assert!(resolve_resource(&state, &host, task, 1).is_ok());
    assert_eq!(
        super::cached_window_peer(&state, task, 1),
        None,
        "an object with no window of its own must not pair off with another"
    );
}

/// A heap-placed texture has no window of its own, and its record must be
/// recognised before its bytes are decoded as an allocation.
///
/// A placement arrives under the ordinary texture object type. The texture
/// decoder reads the allocation size and handle at fixed offsets, and on a
/// placement record those offsets hold the record's own opcode, length and heap
/// reference — so it answers with a window that is a number and not an address.
/// A backing identity built on it would be false equality with whatever else
/// landed on the same number, which is the direction that hands storage back
/// under a live reader.
///
/// **The wide form is the one that gets through**, and it is why the check is
/// on the bytes rather than on the length: at 68 bytes it is exactly the
/// texture decoder's minimum, so nothing about its size refuses it. The narrow
/// form is 60 and would be refused by length alone — testing only that would
/// have passed with the guard deleted.
#[test]
fn a_heap_placement_is_refused_an_identity_rather_than_given_a_number() {
    use crate::runtime::decode::resource::{
        descriptor_is_heap_placement, HEAP_TEXTURE_WIDE_LEN, HEAP_TEXTURE_WIDE_OPCODE,
        OBJECT_TYPE_TEXTURE,
    };

    // A wide placement record: its own opcode, length and heap reference where
    // a plain texture descriptor keeps its allocation size and handle.
    let mut placement = vec![0u8; HEAP_TEXTURE_WIDE_LEN];
    st32(&mut placement[0..], HEAP_TEXTURE_WIDE_OPCODE);
    st32(&mut placement[4..], HEAP_TEXTURE_WIDE_LEN as u32);
    st32(&mut placement[8..], 6565);
    assert!(
        descriptor_is_heap_placement(&placement),
        "the record names itself, and that is the only thing that can tell it \
         apart from an allocation of its own"
    );

    let entry = ListObjectEntry {
        object_type: OBJECT_TYPE_TEXTURE,
        descriptor_length: HEAP_TEXTURE_WIDE_LEN as u32,
        descriptor_gva: 0x1000,
    };
    // The bogus answer this refuses, stated so the test is about the guard and
    // not about the decoder happening to fail: without the check the decode
    // succeeds and the heap reference becomes a page handle.
    assert!(
        crate::runtime::decode::resource::decode_descriptor(entry.object_type, &placement)
            .is_ok_and(|decoded| decoded.backing_window(PAGE_SHIFT_X86).is_some()),
        "the wide placement decodes as a texture, which is exactly why the \
         record has to be recognised before the decode is trusted"
    );
    assert_eq!(
        super::backing_window(PAGE_SHIFT_X86, &entry, &placement),
        None,
        "a placement names storage inside a heap, and the bytes at the offsets \
         an allocation would use are its own header"
    );
}

/// The device answers the model's object resolver, per task, and two tasks
/// holding the same reference number get two different answers.
///
/// The trait resolves `object_ref` and nothing else, so the whole question this
/// test exists for is whether the *right* namespace answered. Two tasks reusing
/// one reference number is the ordinary case on this interface — a reference is
/// an index into the task's own object list — and a resolver that let one task's
/// list answer for another's would hand the dependency compiler a `ResourceId`
/// for storage the asking task cannot reach.
#[test]
fn the_device_answers_the_models_object_resolver_from_the_bound_tasks_namespace() {
    use super::TaskRefResolver;
    use reims_vgpu_core::resolve::RefResolver as _;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let (task, ref_) = (1u32, 1u32);

    assert_eq!(
        TaskRefResolver::new(&state, task).resource(ref_),
        None,
        "a slot nothing has been constructed into names nothing"
    );

    resolve_resource(&state, &host, task, ref_).expect("construction");
    let name = state
        .object_name(task, ref_)
        .expect("construction names it");
    assert_eq!(
        TaskRefResolver::new(&state, task).resource(ref_),
        Some(name),
        "the model gets the name the device's own callers get, not a second \
         identity for one object"
    );

    // A task that has never been defined has no namespace, and a reference
    // number it shares with a live task must not reach into that task's list.
    assert_eq!(
        TaskRefResolver::new(&state, task + 1).resource(ref_),
        None,
        "the binding is what decides which namespace answers, and an unbound \
         task's namespace is empty rather than the neighbouring task's"
    );
    assert_eq!(
        TaskRefResolver::new(&state, task + 1).task_id(),
        task + 1,
        "and the binding says which task it is, so a caller can check"
    );

    // The device is also the *source* of namespaces, in the shape the lifecycle
    // joins want — and it must answer the same thing, or a lifetime packet and a
    // command-stream walk would resolve one ref two ways.
    {
        use reims_vgpu_core::identity::TaskId;
        use reims_vgpu_core::resolve::TaskNamespaces;
        let names = super::TaskNames::new(&state, &host);
        assert_eq!(
            TaskNamespaces::resource(&names, TaskId(task), ref_),
            Some(name),
            "both doors into one namespace answer with one name"
        );
        assert_eq!(
            TaskNamespaces::resource(&names, TaskId(task + 1), ref_),
            None,
            "and the source routes by task rather than resolving in whichever \
             namespace it reached first"
        );
    }

    // The name stops resolving when the object does, through the trait as
    // through every other door.
    assert!(state.delete_object(task, ref_));
    assert_eq!(TaskRefResolver::new(&state, task).resource(ref_), None);
    {
        use reims_vgpu_core::identity::TaskId;
        use reims_vgpu_core::resolve::TaskNamespaces;
        // Deleted, and the guest's list still holds the entry — so the source
        // names it again, with a new generation. That is the namespace's rule
        // for a reused slot, and it is why the memo is keyed by a name.
        let names = super::TaskNames::new(&state, &host);
        let after = TaskNamespaces::resource(&names, TaskId(task), ref_);
        assert!(after.is_some());
        assert_ne!(after, Some(name), "a reused slot is a different name");
    }
}

/// The lifetime-ref census counts what it asked as well as what it found, and
/// prices the refusal per packet rather than per ref.
///
/// The denominator is the point of the test. A boot in which every list packet
/// resolved and a boot in which no list packet arrived produce the same zero on
/// `lifetime_ref_unnamed`, and those are opposite facts about the same silence —
/// the first says lazy declaration holds on this rail, the second says nothing
/// at all.
#[test]
fn the_lifetime_ref_census_counts_what_it_asked_and_prices_the_packet() {
    use crate::runtime::drain::store_route_count;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let (task, ref_) = (1u32, 1u32);
    // One past the list's declared count, so the guest's own table cannot name
    // it and no populated namespace could have either.
    const EMPTY_SLOT: u32 = 8;

    let asked = store_route_count("lifetime_ref_asked");
    let named = store_route_count("lifetime_ref_already_named");
    let unnamed = store_route_count("lifetime_ref_unnameable");
    let on_demand = store_route_count("lifetime_ref_named_on_demand");
    let lists = store_route_count("lifetime_ref_list_asked");
    let refusing = store_route_count("lifetime_ref_list_would_refuse");

    // Nothing constructed yet. Ref 1 is a live slot in the guest's list, so the
    // on-demand door names it; the slot above it is empty, and no populated
    // namespace could have answered for it either.
    super::note_lifetime_refs_named(&state, &host, task, &[ref_, EMPTY_SLOT]);
    assert_eq!(store_route_count("lifetime_ref_asked"), asked + 2);
    assert_eq!(
        store_route_count("lifetime_ref_already_named"),
        named,
        "neither had been constructed, so neither was already named"
    );
    assert_eq!(
        store_route_count("lifetime_ref_named_on_demand"),
        on_demand + 1,
        "the guest's own object list is what names it, without constructing it"
    );
    assert_eq!(store_route_count("lifetime_ref_unnameable"), unnamed + 1);
    assert_eq!(store_route_count("lifetime_ref_list_asked"), lists + 1);
    assert_eq!(
        store_route_count("lifetime_ref_list_would_refuse"),
        refusing + 1,
        "one packet, priced once, however many of its refs were unnameable"
    );

    // Naming did not construct: the memo is still empty, and the name is the one
    // a later construction takes rather than a second generation.
    let name = state.object_name(task, ref_).expect("named on demand");
    assert!(
        state.constructed_object(task, ref_).is_none(),
        "naming is the cheap half — no descriptor snapshot, no host object"
    );
    let built = resolve_resource(&state, &host, task, ref_).expect("construction");
    assert_eq!(
        state.object_name(task, ref_),
        Some(name),
        "constructing an already-named reference keeps its name, or the slot \
         would get a second generation and displace an object the guest never \
         deleted"
    );
    assert!(std::sync::Arc::ptr_eq(
        &state.constructed_object(task, ref_).expect("memo"),
        &built
    ));
    assert_eq!(store_route_count("object_declared_over_a_live_name"), 0);

    // A second sighting of a named ref is the cheap arm.
    super::note_lifetime_refs_named(&state, &host, task, &[ref_]);
    assert_eq!(store_route_count("lifetime_ref_already_named"), named + 1);
    assert_eq!(
        store_route_count("lifetime_ref_named_on_demand"),
        on_demand + 1
    );

    // Deleting the object takes the name back — and the guest's list still holds
    // the entry, so the door names it again. That is the namespace's rule, not a
    // leak: a reused slot is a new generation.
    assert!(state.delete_object(task, ref_));
    super::note_lifetime_refs_named(&state, &host, task, &[ref_]);
    assert_eq!(
        store_route_count("lifetime_ref_named_on_demand"),
        on_demand + 2
    );
    assert_ne!(
        state.object_name(task, ref_),
        Some(name),
        "the second occupant of the slot is a different name"
    );
}
